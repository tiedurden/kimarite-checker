"""Stage 2: cluster overlay crops, label each cluster once, move clips into place.

Kimarite is a closed vocabulary rendered with identical pixels every time -- same
font, position, size. So this is image matching against a small recurring set,
NOT open-ended OCR. Clustering 675 crops yields ~15 groups; you name each group
once and the whole dataset is labeled.

Why not OCR: stylized broadcast fonts + near-identical strings (yorikiri vs
yoritaoshi differ by one glyph, different techniques) means OCR silently
corrupts labels. Clustering fails loudly instead -- you SEE the group.

    python label_clusters.py                 # cluster, write review sheets
    python label_clusters.py --apply         # after filling in labels.csv
"""

import argparse
import csv
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np

import kimarite

# Spelling reference for labels.csv, roughly by real-world frequency.
COMMON = [
    "yorikiri", "oshidashi", "hatakikomi", "tsukiotoshi", "uwatenage",
    "yoritaoshi", "okuridashi", "shitatenage", "kotenage", "sukuinage",
    "tsukidashi", "hikiotoshi", "uwatedashinage", "kimedashi", "abisetaoshi",
]

THUMB = 64  # crops downscaled to THUMB-wide grayscale before clustering


def load_thumbs(crops: Path) -> tuple[list[Path], np.ndarray]:
    """Load crops as small grayscale vectors via ffmpeg (no PIL dependency)."""
    paths = sorted(p for p in crops.glob("*.png") if not p.name.startswith("probe_"))
    if not paths:
        sys.exit(f"no crops in {crops}/ -- run segment_bouts.py first")

    vecs, kept = [], []
    for p in paths:
        raw = subprocess.run(
            ["ffmpeg", "-v", "error", "-i", str(p),
             "-vf", f"scale={THUMB}:{THUMB},format=gray",
             "-f", "rawvideo", "-"], capture_output=True).stdout
        if len(raw) < THUMB * THUMB:
            continue
        v = np.frombuffer(raw[: THUMB * THUMB], dtype=np.uint8).astype(np.float32)
        # Per-crop normalization: broadcast brightness drifts between videos, but
        # the glyph SHAPES are what identify the technique.
        v = (v - v.mean()) / (v.std() or 1.0)
        vecs.append(v)
        kept.append(p)
    if not vecs:
        sys.exit("could not decode any crops -- is ffmpeg working?")
    return kept, np.stack(vecs)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops", default="crops", type=Path)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--labels", default="labels.csv", type=Path)
    ap.add_argument("--review", default="review", type=Path)
    ap.add_argument("--clusters", type=int, default=24,
                    help="over-cluster deliberately; merge by giving groups the "
                         "same name in labels.csv")
    ap.add_argument("--apply", action="store_true",
                    help="move clips into data/<kimarite>/ using labels.csv")
    args = ap.parse_args()

    if args.apply:
        return apply_labels(args)

    from sklearn.cluster import AgglomerativeClustering

    paths, X = load_thumbs(args.crops)
    n = min(args.clusters, len(paths))
    print(f"{len(paths)} crops -> {n} clusters")

    # Ward on raw normalized pixels: identical renders of the same string land
    # in the same cluster. Over-clustering is safe (merge via names); under-
    # clustering silently mixes two techniques into one label.
    labels = AgglomerativeClustering(n_clusters=n, linkage="ward").fit_predict(X)

    # Contact sheet per cluster so each group is verifiable at a glance.
    if args.review.exists():
        shutil.rmtree(args.review)
    args.review.mkdir(parents=True)
    counts = Counter(labels)
    for c in sorted(counts):
        members = [p for p, l in zip(paths, labels) if l == c]
        sheet = args.review / f"cluster_{c:02d}_n{len(members)}.png"
        # Up to 12 members stacked vertically -- enough to spot a mixed cluster.
        inputs = []
        for m in members[:12]:
            inputs += ["-i", str(m)]
        k = len(inputs) // 2
        if k == 1:
            shutil.copy(members[0], sheet)
        else:
            subprocess.run(
                ["ffmpeg", "-v", "error", "-y", *inputs,
                 "-filter_complex", f"vstack=inputs={k}", str(sheet)],
                capture_output=True)

    with args.labels.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["cluster", "n", "kimarite", "example"])
        for c in sorted(counts):
            ex = next(p.name for p, l in zip(paths, labels) if l == c)
            w.writerow([c, counts[c], "", ex])

    (args.crops / "_cluster_map.csv").write_text(
        "\n".join(f"{p.name},{l}" for p, l in zip(paths, labels)), encoding="utf-8")

    print(f"""
contact sheets -> {args.review}/     (one image per cluster)
label sheet     -> {args.labels}

Next:
  1. Open each sheet in {args.review}/. All crops in one sheet should show the
     SAME kimarite. If a sheet is mixed, re-run with a higher --clusters.
  2. Fill the `kimarite` column in {args.labels} -- one name per cluster.
     Give two clusters the SAME name to merge them (the same technique often
     splits across clusters due to animation timing).
     Leave a row BLANK to drop that cluster (junk, replays, ambiguous).
  3. python label_clusters.py --apply

Spelling reference: {', '.join(COMMON[:8])} ...""")


def apply_labels(args) -> None:
    if not args.labels.exists():
        sys.exit(f"{args.labels} not found -- run without --apply first")
    cmap_file = args.crops / "_cluster_map.csv"
    if not cmap_file.exists():
        sys.exit(f"{cmap_file} missing -- re-run clustering")

    names, unknown, higi = {}, [], {}
    for row in csv.DictReader(args.labels.open(encoding="utf-8")):
        raw = (row.get("kimarite") or "").strip()
        if not raw:
            continue
        canon = kimarite.normalize(raw)
        if canon is None:
            # Don't silently create a misspelled category -- 'yorikri' would
            # become its own class and quietly split the biggest one in two.
            unknown.append((row["cluster"], raw))
            continue
        if not kimarite.is_technique(canon):
            # higi: the loser lost unaided, so there is no causal motion to
            # learn. `fusen` is worse -- no bout was fought at all.
            higi[row["cluster"]] = canon
            continue
        names[row["cluster"]] = canon

    if unknown:
        print("unrecognized kimarite names -- fix these in labels.csv:",
              file=sys.stderr)
        for c, raw in unknown:
            print(f"  cluster {c}: {raw!r}", file=sys.stderr)
        sys.exit(1)
    if higi:
        print(f"excluding {len(higi)} non-technique cluster(s): "
              f"{', '.join(sorted(set(higi.values())))}")
        print("  (higi -- no technique was applied; nothing to learn)")
    if not names:
        sys.exit(f"no usable kimarite in {args.labels} -- nothing to apply")

    cmap = dict(l.split(",") for l in
                cmap_file.read_text(encoding="utf-8").splitlines() if "," in l)

    src_dir = args.data / "_unlabeled"
    moved, dropped, missing = Counter(), 0, 0
    for png, cluster in cmap.items():
        kimarite = names.get(cluster)
        if not kimarite:
            dropped += 1
            continue
        clip = src_dir / (Path(png).stem + ".mp4")
        if not clip.exists():
            missing += 1
            continue
        dest = args.data / kimarite / clip.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(clip), str(dest))
        moved[kimarite] += 1

    print(f"moved {sum(moved.values())} clips into {args.data}/")
    for k, c in moved.most_common():
        print(f"  {k:<24} {c:>4}")
    if dropped:
        print(f"{dropped} clips left in {src_dir}/ (unlabeled clusters)")
    if missing:
        print(f"{missing} crops had no matching clip", file=sys.stderr)

    # The long tail is real: ~5 techniques dominate, many appear once or twice.
    thin = [k for k, c in moved.items() if c < 15]
    if thin:
        print(f"\nthin classes (<15 clips): {', '.join(thin)}")
        print("train_head.py will fold these into 'other' by default (--min-class).")

    # Warn where the data itself limits what's achievable, so a confusion in the
    # matrix later reads as expected rather than as a bug to chase.
    present = set(moved)
    pairs = [(a, b) for a, b in kimarite.OUTCOME_PAIRS
             if a in present and b in present]
    grips = [(a, b) for a, b in kimarite.GRIP_PAIRS if a in present and b in present]
    if pairs:
        print(f"\noutcome pairs present: "
              f"{', '.join(f'{a}/{b}' for a, b in pairs)}")
        print("  Same technique; differ only by standing vs fallen in the final\n"
              "  ~0.5s. Check segment_bouts.py PAD didn't trim the bout's end.")
    if grips:
        print(f"\ngrip pairs present: {', '.join(f'{a}/{b}' for a, b in grips)}")
        print("  Differ by inside/outside arm position -- often unresolvable at\n"
              "  224x224. Expect persistent confusion; consider merging them.")
    print("\nnext: python extract_features.py")


if __name__ == "__main__":
    main()
