"""Stage 1 (Herbert): OCR the "X WINS BY Y" caption -> label + clip per bout.

Why OCR and not clustering (as label_clusters.py does for NHK): this caption is
prose containing the WINNER'S NAME, e.g.

    KINBOZAN WINS BY OSHIDASHI
    Fujiryoga 0-1 | Kinbozan 1-0

so its pixel width varies per bout. Identical-render clustering cannot group
"KINBOZAN WINS BY OSHIDASHI" with "URA WINS BY OSHIDASHI" -- same technique,
different pixels. OCR + fuzzy-match against the closed 82-name vocabulary is the
right tool, and kimarite.normalize() rejects anything that isn't a real name.

Bout windows come from hb_manifest.py (Herbert hand-writes them in descriptions).
The caption appears a few seconds AFTER the bout ends, during the replay, so we
search the tail of each window and cut the clip from before it.

    python hb_label.py --probe <video-id>    # dump caption crops to check BOX
    python hb_label.py                       # OCR all bouts in hb_bouts.csv
"""

import argparse
import csv
import re
import subprocess
import sys
from collections import Counter
from pathlib import Path

import kimarite

# CALIBRATED on Haru 2026 Day 1 (nnSWd2TFUVo, 854x480) by OCR bisection, NOT by
# eyeballing a tile: the win line occupies y 350-384, x 0-380. Verified to read
# "KINBOZAN WINS BY OSHIDASHI" cleanly. Fractions so it survives a res change.
# Deliberately excludes the score line below it ("Fujiryoga 0-1 | Kinbozan 1-0")
# and the "Next up:" line, both of which bleed garbage into --psm 7 output.
WIN_BOX = (0.00, 0.729, 0.445, 0.800)

# Everything that leaks the answer or the winner, masked out of training clips:
#   win line + score line   bottom-left  (the label itself, plus winner's name)
#   REPLAY badge            bottom-left, just below
#   "Next up: A VS B"       bottom-left, below that
#   shikona boards          top-left     (kanji names of both wrestlers)
#   LIVE badge              top-right
LEAK_BOXES_HB = [
    (0.00, 0.66, 0.55, 0.85),   # win line, score line, REPLAY, "Next up"
    (0.00, 0.00, 0.36, 0.10),   # shikona boards (top-left)
    (0.90, 0.03, 1.00, 0.12),   # LIVE badge (top-right)
]
MASK_MARGIN = 0.02

# The caption shows up during the replay, i.e. AFTER the bout. Search the last
# SEARCH_TAIL seconds of each manifest window at SEARCH_FPS.
SEARCH_TAIL = 45.0
SEARCH_FPS = 1.0   # caption holds ~7s, so ~7 looks at it -> enough for MIN_VOTES

CLIP_LEN = 6.0   # bout footage kept per sample
CLIP_BACK = 4.0  # clip ends this many seconds BEFORE the caption's first frame

# OCR at 480p intermittently drops inter-word spaces ("WINSBYOSHIDASHI") and
# tacks on a garbage glyph or two from the line below. So: spaces optional, and
# take the technique as a greedy run that kimarite.normalize() then has to accept
# -- its fuzzy +/-2-char match absorbs the trailing noise ("OSHIDASHINe" -> oshidashi).
WIN_RE = re.compile(r"([A-Z][A-Za-z]{2,})\s*WIN5?S?\s*BY\s*([A-Za-z]+)", re.IGNORECASE)

# Distinct OCR reads that must agree before a label is accepted. The caption holds
# for ~7s, so at SEARCH_FPS we get several looks at it; requiring agreement kills
# one-off misreads that happen to fuzzy-match a real (wrong) technique name.
MIN_VOTES = 2


def box_px(box, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    return (max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h)),
            int(x0 * w), int(y0 * h))


def probe_dims(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True)
    try:
        w, h = r.stdout.strip().split(",")[0].split("x")[:2]
        return int(w), int(h)
    except ValueError:
        sys.exit(f"ffprobe failed on {path}: {r.stderr.strip()}")


def ocr_strip(path: Path, t: float, w: int, h: int, tmp: Path) -> str:
    """Upscale + binarize the caption strip, then OCR it."""
    cw, ch, cx, cy = box_px(WIN_BOX, w, h)
    png = tmp / "ocr.png"
    # 4x lanczos + grayscale + high contrast: tesseract is far more reliable on
    # large clean glyphs than on 480p broadcast text. lanczos beat nearest here
    # (nearest kept the JPEG mosquito noise as hard edges and OCR read them).
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
         "-vf", f"crop={cw}:{ch}:{cx}:{cy},scale=iw*4:ih*4:flags=lanczos,"
                f"format=gray,eq=contrast=1.8",
         "-frames:v", "1", str(png)],
        capture_output=True)
    if not png.exists():
        return ""
    r = subprocess.run(
        ["tesseract", str(png), "stdout", "--psm", "7", "-l", "eng"],
        capture_output=True, text=True)
    return r.stdout.strip()


def find_caption(path: Path, start: float, end: float, w: int, h: int,
                 tmp: Path) -> tuple[str, str, float, int] | None:
    """Scan the window tail for 'X WINS BY Y'.

    Returns (winner, kimarite, first_seen_t, votes) once MIN_VOTES independent
    frames agree on the technique, else None. Voting matters because a single
    misread can still fuzzy-match a REAL but WRONG technique -- and a silently
    wrong label is worse than a miss, which at least shows up in the miss count.
    """
    t = max(start, end - SEARCH_TAIL)
    step = 1.0 / SEARCH_FPS
    votes: Counter = Counter()
    winners: dict[str, str] = {}
    first_t: dict[str, float] = {}
    while t < end:
        text = ocr_strip(path, t, w, h, tmp)
        if m := WIN_RE.search(text):
            canon = kimarite.normalize(m.group(2))
            if canon:
                votes[canon] += 1
                winners.setdefault(canon, m.group(1).title())
                first_t.setdefault(canon, t)
                if votes[canon] >= MIN_VOTES:
                    return winners[canon], canon, first_t[canon], votes[canon]
        t += step
    return None


def mask_filter(w: int, h: int) -> str:
    parts = []
    for x0, y0, x1, y1 in LEAK_BOXES_HB:
        m = MASK_MARGIN
        bx, by = max(0.0, x0 - m), max(0.0, y0 - m)
        bw, bh = min(1.0, x1 + m) - bx, min(1.0, y1 + m) - by
        parts.append(f"drawbox=x={int(bx*w)}:y={int(by*h)}"
                     f":w={int(bw*w)}:h={int(bh*h)}:color=black:t=fill")
    return ",".join(parts)


def probe(path: Path, out: Path, start: float = 200, span: float = 80) -> None:
    w, h = probe_dims(path)
    cw, ch, cx, cy = box_px(WIN_BOX, w, h)
    out.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start}", "-t", f"{span}",
         "-i", str(path),
         "-vf", f"fps=1,crop={cw}:{ch}:{cx}:{cy},scale=iw*2:ih*2:flags=neighbor,"
                f"tile=1x{int(span)}",
         "-frames:v", "1", str(out / "hb_winbox.png")],
        capture_output=True)
    print(f"{path.name}: {w}x{h}, WIN_BOX -> {cw}x{ch} at ({cx},{cy})")
    print(f"wrote {out/'hb_winbox.png'} -- every row should be the caption strip;\n"
          f"'X WINS BY Y' must be fully inside it. Adjust WIN_BOX if clipped.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", help="video id in raw/ to dump caption crops for")
    ap.add_argument("--bouts", default="hb_bouts.csv", type=Path)
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--data", default="data_hb", type=Path)
    ap.add_argument("--out", default="hb_labels.csv", type=Path)
    ap.add_argument("--limit", type=int, help="stop after N bouts (for testing)")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            sys.exit(f"{tool} not on PATH -- source ./env.sh")

    if args.probe:
        cands = list(args.raw.glob(f"*{args.probe}*.mp4"))
        if not cands:
            sys.exit(f"no video matching {args.probe} in {args.raw}/")
        return probe(cands[0], Path("crops/_probe"))

    if subprocess.run(["which", "tesseract"], capture_output=True).returncode != 0:
        sys.exit("tesseract not on PATH -- winget install UB-Mannheim.TesseractOCR\n"
                 "then reopen the shell (or add its dir in env.sh)")
    if not args.bouts.exists():
        sys.exit(f"{args.bouts} not found -- run hb_manifest.py first")

    rows = list(csv.DictReader(args.bouts.open(encoding="utf-8")))
    if args.limit:
        rows = rows[: args.limit]

    tmp = Path("crops/_tmp")
    tmp.mkdir(parents=True, exist_ok=True)
    dims: dict[str, tuple[int, int]] = {}

    out_rows, hits, misses = [], Counter(), []
    for i, r in enumerate(rows, 1):
        vids = list(args.raw.glob(f"*{r['video']}*.mp4"))
        if not vids:
            misses.append((r["video"], r["n"], "video not downloaded"))
            continue
        path = vids[0]
        if r["video"] not in dims:
            dims[r["video"]] = probe_dims(path)
        w, h = dims[r["video"]]

        start, end = float(r["start"]), float(r["end"])
        found = find_caption(path, start, end, w, h, tmp)
        if not found:
            misses.append((r["video"], r["n"], "no caption OCR'd"))
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  MISS")
            continue
        winner, canon, t_cap, votes = found
        hits[canon] += 1
        print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  {canon:<16} "
              f"({winner}, {votes} votes)")

        # Clip ends before the caption appears; masking covers the rest.
        clip_end = max(start + 1.0, t_cap - CLIP_BACK)
        clip_start = max(start, clip_end - CLIP_LEN)
        dest = args.data / canon / f"{r['video']}__{int(r['n']):02d}.mp4"
        dest.parent.mkdir(parents=True, exist_ok=True)
        ok = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{clip_start:.2f}",
             "-i", str(path), "-t", f"{clip_end-clip_start:.2f}", "-an",
             "-vf", mask_filter(w, h), "-c:v", "libx264", "-preset", "veryfast",
             str(dest)], capture_output=True).returncode == 0
        out_rows.append({**r, "kimarite": canon, "winner": winner, "votes": votes,
                         "caption_t": round(t_cap, 2), "clip": str(dest) if ok else ""})

    with args.out.open("w", newline="", encoding="utf-8") as f:
        if out_rows:
            wr = csv.DictWriter(f, fieldnames=list(out_rows[0]))
            wr.writeheader()
            wr.writerows(out_rows)

    print(f"\n{sum(hits.values())} labeled, {len(misses)} missed -> {args.out}")
    for k, n in hits.most_common(20):
        print(f"  {k:<24} {n:>4}")
    if misses:
        print(f"\nfirst misses: {misses[:5]}")
        rate = len(misses) / max(1, len(rows))
        if rate > 0.2:
            print(f"MISS RATE {rate:.0%} -- re-check WIN_BOX with --probe, or widen\n"
                  f"SEARCH_TAIL (caption may fall outside the searched window tail).")


if __name__ == "__main__":
    main()
