"""Re-cut already-labelled clips with the bout-anchored window.

The OCR labels in hb_labels.csv are correct -- the caption was read properly. Only
the clip WINDOW was wrong (see hb_window.py). So a fix does not need the 20s/bout
OCR pass again: re-locate the bout from the stored caption_t and re-cut, ~2s/bout.

    python hb_recut.py --dry-run     # report what would change, touch nothing
    python hb_recut.py               # re-cut in place, drop unlocatable bouts

Bouts whose window cannot be found are REMOVED from data_hb/ and marked in the CSV
rather than left alone: a stale clip is footage from the wrong part of the video
under a correct label, which is exactly the failure that held the model at chance.
"""

import argparse
import csv
import shutil
import sys
from collections import Counter
from pathlib import Path

from hb_label import mask_filter, probe_dims
from hb_window import cut_clip, find_bout_window


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="hb_labels.csv", type=Path)
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--data", default="data_hb", type=Path)
    ap.add_argument("--cache", default="cache_hb", type=Path,
                    help="embeddings to invalidate for re-cut clips")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--video", help="only this video id")
    args = ap.parse_args()

    if not args.labels.exists():
        sys.exit(f"{args.labels} not found")
    rows = list(csv.DictReader(args.labels.open(encoding="utf-8")))
    if args.video:
        rows = [r for r in rows if args.video in r["video"]]
    if not rows:
        sys.exit("no rows to re-cut")

    dims: dict[str, tuple[int, int]] = {}
    recut, dropped, failed = 0, [], 0
    reasons: Counter = Counter()
    out_rows = []

    for i, r in enumerate(rows, 1):
        vids = list(args.raw.glob(f"*{r['video']}*.mp4"))
        if not vids:
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  SKIP -- not downloaded")
            out_rows.append(r)
            continue
        path = vids[0]
        if r["video"] not in dims:
            dims[r["video"]] = probe_dims(path)
        w, h = dims[r["video"]]

        got = find_bout_window(path, float(r["start"]), float(r["caption_t"]))
        if isinstance(got, str):
            reasons[got] += 1
            dropped.append((r["video"], r["n"], got))
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']} {r['kimarite']:<14} "
                  f"DROP -- {got}")
            if not args.dry_run:
                # Remove the clip AND its embedding: leaving either behind means the
                # next train_head run silently reuses the bad footage.
                for p in (Path(r["clip"]) if r["clip"] else None,
                          args.cache / r["kimarite"] /
                          f"{r['video']}__{int(r['n']):02d}.npy"):
                    if p and p.exists():
                        p.unlink()
            r = {**r, "clip": "", "bout_start": "", "bout_end": ""}
            out_rows.append(r)
            continue

        t0, t1 = got
        old = f" (was {r.get('bout_start') or 'caption-anchored'})"
        print(f"[{i}/{len(rows)}] {r['video']}#{r['n']} {r['kimarite']:<14} "
              f"{t0:.1f}-{t1:.1f} ({t1-t0:.1f}s){old}")
        if args.dry_run:
            recut += 1
            out_rows.append(r)
            continue

        dest = args.data / r["kimarite"] / f"{r['video']}__{int(r['n']):02d}.mp4"
        if cut_clip(path, dest, t0, t1, mask_filter(w, h)):
            recut += 1
            # The cached embedding describes the OLD footage. Deleting it makes
            # extract_features.py re-embed this clip; keeping it would pair new
            # labels with stale vectors and look like it worked.
            npy = args.cache / r["kimarite"] / f"{dest.stem}.npy"
            if npy.exists():
                npy.unlink()
            out_rows.append({**r, "clip": str(dest),
                             "bout_start": round(t0, 2), "bout_end": round(t1, 2)})
        else:
            failed += 1
            print(f"      ffmpeg failed on {dest}")
            out_rows.append(r)

    print(f"\n{'would re-cut' if args.dry_run else 're-cut'} {recut}, "
          f"drop {len(dropped)}, {failed} ffmpeg failures  "
          f"({len(rows)} rows)")
    if reasons:
        print("drop reasons:")
        for reason, n in reasons.most_common():
            print(f"  {n:>4}x {reason}")

    if args.dry_run:
        print("\ndry run -- nothing written. Re-run without --dry-run to apply.")
        return

    # Rewrite the CSV with the bout bounds recorded, so a later run can tell a
    # bout-anchored row from a caption-anchored one.
    fields = list(out_rows[0]) if out_rows else []
    for extra in ("bout_start", "bout_end"):
        if extra not in fields:
            fields.append(extra)
    bak = args.labels.with_suffix(".csv.bak")
    shutil.copy2(args.labels, bak)
    with args.labels.open("w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in out_rows:
            wr.writerow({k: r.get(k, "") for k in fields})
    print(f"\nupdated {args.labels} (backup: {bak})")
    print(f"next: python extract_features.py --data {args.data} --cache {args.cache}")


if __name__ == "__main__":
    main()
