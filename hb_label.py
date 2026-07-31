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
# The caption holds ~7s but is only cleanly legible for part of that (it fades in
# and out, and the frames underneath change). At 1 fps a real caption produced ONE
# clean read plus two misreads, which MIN_VOTES=2 then rejected -- 7 of 21 bouts
# lost that way. At 3 fps the legible stretch yields several agreeing reads.
SEARCH_FPS = 3.0

CLIP_LEN = 6.0   # bout footage kept per sample
CLIP_BACK = 4.0  # clip ends this many seconds BEFORE the caption's first frame

# OCR at 480p intermittently drops inter-word spaces ("WINSBYOSHIDASHI") and
# tacks on a garbage glyph or two from the line below. So: spaces optional, and
# take the technique as a greedy run that kimarite.normalize() then has to accept
# -- its fuzzy +/-2-char match absorbs the trailing noise ("OSHIDASHINe" -> oshidashi).
#
# The technique class allows digits and punctuation because tesseract substitutes
# lookalike glyphs INSIDE the word: "TSUKIOTOSHI" came back as "1SUKIOTOSH!"
# (T->1, I->!). A letters-only class rejected the line outright, which is how 7 of
# 21 bouts were lost. OCR_FIXES undoes the substitution before normalize() sees it.
# The (?:A|4)? after BY absorbs the artefact seen when the space is dropped:
# "WINS BY TSUKIOTOSHI" -> "WINSBYASUKIOTOSHI", where the missing space renders as
# a spurious A/4 glued to the front of the technique.
WIN_RE = re.compile(
    r"([A-Z0-9][A-Za-z0-9]{2,})\s*WIN5?S?\s*[8B]Y\s*(?:[A4](?=[A-Z]{4}))?"
    r"([A-Za-z0-9!|\[\]/\\]+)",
    re.IGNORECASE)

# Glyph confusions observed in real output at 854x480. Applied to the technique
# token only -- never to the winner's name, where a wrong letter is harmless but a
# wrong technique is a corrupt label.
OCR_FIXES = str.maketrans({
    "1": "I", "!": "I", "|": "I", "[": "I", "]": "I", "/": "I", "\\": "I",
    "0": "O", "5": "S", "8": "B", "6": "G", "2": "Z",
})

# Distinct OCR reads that must agree before a label is accepted. Requiring
# agreement is not paranoia: at 1 fps this caption read as "1SUKIOTOSH!"
# (-> tsukiotoshi) once and "TSURIOTOSHI" (-> tsuriotoshi) twice. Those are two
# DIFFERENT real techniques one letter apart (k vs r), so a first-past-the-post
# rule takes whichever crosses the line first and can be confidently wrong.
# So: scan the WHOLE window, then take the plurality winner.
MIN_VOTES = 2
# ...and require the winner to beat the runner-up by this margin, else it's a
# coin flip between two lookalike techniques and we'd rather record a miss.
VOTE_MARGIN = 2


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


def extract_strips(path: Path, start: float, end: float, w: int, h: int,
                   tmp: Path) -> list[Path]:
    """Decode the search window ONCE and write every caption strip as a PNG.

    The original version spawned one ffmpeg per sampled frame: at SEARCH_FPS=3
    over a 45s tail that is 135 process launches per bout, each re-opening the
    container and seeking. Measured 43s/bout, and running 6 of those in parallel
    pinned the machine (12 concurrent transcode/OCR processes, laptop fans at
    full tilt). One sequential decode with fps= does the same sampling in a
    single process.

    Frames land as strip_0001.png, strip_0002.png, ... in timestamp order, so
    index i corresponds to time start + i/SEARCH_FPS.
    """
    cw, ch, cx, cy = box_px(WIN_BOX, w, h)
    for old in tmp.glob("strip_*.png"):
        old.unlink()
    # -ss BEFORE -i so ffmpeg seeks rather than decoding from frame 0.
    # 4x lanczos + grayscale + high contrast: tesseract is far more reliable on
    # large clean glyphs than on 480p broadcast text. lanczos beat nearest here
    # (nearest kept the JPEG mosquito noise as hard edges and OCR read them).
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}",
         "-t", f"{end - start:.2f}", "-i", str(path),
         "-vf", f"fps={SEARCH_FPS},crop={cw}:{ch}:{cx}:{cy},"
                f"scale=iw*4:ih*4:flags=lanczos,format=gray,eq=contrast=1.8",
         str(tmp / "strip_%04d.png")],
        capture_output=True)
    return sorted(tmp.glob("strip_*.png"))


def ocr_png(png: Path) -> str:
    """OCR one already-extracted strip."""
    # Decode explicitly, errors="replace". OCR on broadcast video regularly emits
    # bytes that are not valid in the console's locale encoding (cp1252 on this
    # box), and text=True decodes with that locale -- so a single stray 0x9d from
    # a noisy frame killed the whole run at bout 5 of 21. The garbage frames are
    # exactly the ones we expect to fail the regex anyway; they must not be fatal.
    r = subprocess.run(
        ["tesseract", str(png), "stdout", "--psm", "7", "-l", "eng"],
        capture_output=True)
    return (r.stdout or b"").decode("utf-8", errors="replace").strip()


def normalize_ocr(raw: str) -> str | None:
    """kimarite.normalize(), but tolerant of OCR glyph substitution.

    Tries the raw token first (so a clean read costs nothing), then the
    de-substituted form. "1" is ambiguous -- it stands in for both I and T in this
    font ("1SUKIOTOSH!" is TSUKIOTOSHI, but a medial 1 is usually I) -- so both
    readings are attempted and the first that hits the closed vocabulary wins.
    Anything that matches no real technique still returns None: guessing is worse
    than missing, because a wrong label silently poisons training.
    """
    fixed = raw.translate(OCR_FIXES)
    cands = [raw, fixed, fixed.replace("I", "T", 1)]
    # Tesseract also confuses J/D/O for O in the middle of a word ("TSUKIOJOSHI",
    # "TSUKIOJDSHI" for TSUKIOTOSHI) and prepends a stray letter ("FSUKIOTOSHI",
    # "ASUKIOTOSHI"). normalize()'s +/-2 tolerance covers ONE such error, not two,
    # so try the medial-consonant repair and a leading-char drop as well.
    cands.append(re.sub(r"(?<=[AEIOU])[JD](?=[AEIOU])", "T", fixed))
    cands.append(fixed[1:])
    for cand in cands:
        if hit := kimarite.normalize(cand):
            return hit
    return None


def find_caption(path: Path, start: float, end: float, w: int, h: int,
                 tmp: Path) -> tuple[str, str, float, int] | None:
    """Scan the window tail for 'X WINS BY Y' and return the plurality reading.

    Returns (winner, kimarite, first_seen_t, votes), or None if nothing reached
    MIN_VOTES / VOTE_MARGIN. The whole window is scanned before deciding, rather
    than returning on the first technique to reach MIN_VOTES: two techniques one
    letter apart can both appear among the reads, and whichever got there first
    would win an early-exit race regardless of which is actually on screen.

    A returned miss is a deliberate outcome, not a failure -- it shows up in the
    miss count and can be relabelled by hand. A wrong label cannot be spotted at
    all once it is a directory name.
    """
    t0 = max(start, end - SEARCH_TAIL)
    step = 1.0 / SEARCH_FPS
    votes: Counter = Counter()
    winners: dict[str, str] = {}
    first_t: dict[str, float] = {}
    for i, png in enumerate(extract_strips(path, t0, end, w, h, tmp)):
        text = ocr_png(png)
        if m := WIN_RE.search(text):
            canon = normalize_ocr(m.group(2))
            if canon:
                votes[canon] += 1
                winners.setdefault(canon, m.group(1).title())
                # fps= emits frames evenly from t0, so index maps back to time.
                first_t.setdefault(canon, t0 + i * step)
    if not votes:
        return None
    ranked = votes.most_common()
    top, n = ranked[0]
    runner_up = ranked[1][1] if len(ranked) > 1 else 0
    if n < MIN_VOTES or n - runner_up < VOTE_MARGIN:
        return None
    return winners[top], top, first_t[top], n


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
    # Each worker needs its OWN scratch dir. ocr_strip() always writes ocr.png, so
    # parallel workers sharing one dir would overwrite each other's frame between
    # the ffmpeg write and the tesseract read -- silently labeling bouts with
    # another worker's caption. Shard by video and give each shard a distinct --tmp.
    ap.add_argument("--tmp", default="crops/_tmp", type=Path,
                    help="scratch dir for OCR frames (must be unique per process)")
    ap.add_argument("--resume", action="store_true",
                    help="skip bouts already present in --out (append instead)")
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

    tmp = args.tmp
    tmp.mkdir(parents=True, exist_ok=True)
    dims: dict[str, tuple[int, int]] = {}

    # Resume: a bout whose row is already in --out has been done, so skip the ~40s
    # of OCR. Keyed on video+n, the same key the clip filename uses. This run was
    # killed at 11 clips once (machine load) and every label row was lost, because
    # results were only written at the very end.
    done: set[tuple[str, str]] = set()
    prev_rows: list[dict] = []
    if args.resume and args.out.exists():
        prev_rows = list(csv.DictReader(args.out.open(encoding="utf-8")))
        done = {(p["video"], p["n"]) for p in prev_rows}
        if done:
            print(f"resuming: {len(done)} bout(s) already in {args.out}")

    out_rows, hits, misses, higi = list(prev_rows), Counter(), [], Counter()
    for p in prev_rows:
        hits[p["kimarite"]] += 1

    # Append-and-flush per bout so a kill loses at most the bout in flight.
    fields = list(rows[0]) + ["kimarite", "winner", "votes", "caption_t", "clip"]
    new_file = not (args.resume and args.out.exists() and prev_rows)
    sink = args.out.open("w" if new_file else "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(sink, fieldnames=fields)
    if new_file:
        writer.writeheader()
        writer.writerows(prev_rows)
        sink.flush()

    for i, r in enumerate(rows, 1):
        if (r["video"], r["n"]) in done:
            continue
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

        # Higi (non-techniques) are correct OCR reads that must not become training
        # samples. Some have no bout footage at all -- "fusen" is a forfeit, so the
        # clip is 6s of an empty dohyo -- and the rest (isamiashi, koshikudake,
        # tsukite...) are self-inflicted losses where no technique was applied. Both
        # kinds are noise in a technique classifier's input. Counted, not silently
        # dropped, so the tally still reconciles against the manifest.
        if not kimarite.is_technique(canon):
            higi[canon] += 1
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  {canon:<16} "
                  f"({winner}, {votes} votes)  SKIP -- not a technique")
            continue

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
        row = {**r, "kimarite": canon, "winner": winner, "votes": votes,
               "caption_t": round(t_cap, 2), "clip": str(dest) if ok else ""}
        out_rows.append(row)
        writer.writerow(row)
        sink.flush()   # survive a kill: at most the in-flight bout is lost

    sink.close()

    print(f"\n{sum(hits.values())} labeled, {len(misses)} missed, "
          f"{sum(higi.values())} non-technique -> {args.out}")
    for k, n in hits.most_common(20):
        print(f"  {k:<24} {n:>4}")
    if higi:
        print("\nskipped as non-techniques (higi -- no technique was applied): "
              + ", ".join(f"{k} x{n}" for k, n in higi.most_common()))
    if misses:
        print(f"\nfirst misses: {misses[:5]}")
        rate = len(misses) / max(1, len(rows))
        if rate > 0.2:
            print(f"MISS RATE {rate:.0%} -- re-check WIN_BOX with --probe, or widen\n"
                  f"SEARCH_TAIL (caption may fall outside the searched window tail).")


if __name__ == "__main__":
    main()
