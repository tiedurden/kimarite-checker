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

import numpy as np

import kimarite
from hb_window import cut_clip, find_bout_window, probe_window

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
#   winner name plate       bottom-center, BELOW all of the above
LEAK_BOXES_HB = [
    (0.00, 0.66, 0.55, 0.85),   # win line, score line, REPLAY, "Next up"
    (0.00, 0.00, 0.36, 0.10),   # shikona boards (top-left)
    (0.90, 0.03, 1.00, 0.12),   # LIVE badge (top-right)
    # The winner's shikona on a white plate, e.g. "SHONANNOUMI". It appears AFTER the
    # bout ends but BEFORE the win caption, so bout windows that run to the finish can
    # catch it in their last seconds -- measured inside bout #1 of 28k6Dg0ExjI, whose
    # window ends at 77.7s with the plate up at 76s. That is a label leak as real as
    # the caption itself: the winner's name is not the technique, but it is strongly
    # correlated with it, and a transformer reads it far more easily than it reads
    # wrestling.
    #
    # Full width rather than the measured x 0.385-0.611: the plate is centred and its
    # width tracks the name's length (0.385-0.611 for SHONANNOUMI, 0.422-0.575 for a
    # shorter one), so a tight box calibrated on one name clips a longer one. Nothing
    # below y=0.92 carries bout information -- it is the near edge of the dohyo and the
    # front row of cushions -- so full width costs nothing and removes the guesswork.
    #
    # Measured top edge y=0.933 across 4 bouts; 0.92 gives it margin. It sits below
    # the y<=0.85 box above, which is why it needed its own entry rather than an
    # extension: widening THAT box downward would also have swallowed the lower dohyo
    # through the whole bout.
    (0.00, 0.92, 1.00, 1.00),   # winner name plate (bottom-centre)
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

# Clip bounds come from hb_window.find_bout_window(), which anchors on the video
# (pre-bout banner -> first scene cut after the charge). There is deliberately no
# CLIP_LEN/CLIP_BACK any more: a fixed offset from the caption put 59% of the first
# dataset entirely outside the bout, because the caption fires during the replay
# 13-91s after the charge. See hb_window's module docstring.

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

# --- Skipping frames that cannot hold a caption -----------------------------------
#
# Measured: OCR is 96% of the per-bout cost (24s of 25s), and almost all of that is
# process launches -- 135 tesseract invocations at ~180ms each, on frames of which
# only ~21 contain a caption at all.
#
# Text has a signature the empty strip does not: glyph strokes make many bright/dark
# transitions along each row. Mean absolute horizontal gradient measures it in one
# numpy op on frames ffmpeg is already decoding.
#
# This is NOT a clean separator and must not be treated as one. Over 23 bouts only
# 7 separated cleanly (caption 1.02-18.41 vs other 0.18-16.99, badly overlapping) --
# a bright churning crowd shot has plenty of edge energy too. The same trap as the
# rejected motion gate in hb_window.py: one bout looked perfectly separable.
#
# It works here because separation is not what is needed. The vote logic needs
# MIN_VOTES=2 agreeing reads out of a ~21-frame caption, so the filter only has to
# keep a couple of caption frames, and false positives cost nothing but an OCR call
# that fails the regex. Measured over 34 bouts spanning all 4 videos: at 4.0 the
# WORST bout still keeps 16 caption frames, 0 bouts fall under 2, and OCR drops to
# 31% of frames. 16-vs-2 is headroom, not a knife-edge.
#
# If a new source starts missing captions, RAISE nothing -- set this to 0.0 to
# disable and confirm the filter is the cause before touching the OCR path.
EDGE_MIN = 4.0
EDGE_SCALE = 2          # crop upscale for the gradient pass; 4x costs more, adds nothing


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


def _decode_ocr(out: bytes) -> str:
    """Decode tesseract output tolerantly.

    Decode explicitly, errors="replace". OCR on broadcast video regularly emits
    bytes that are not valid in the console's locale encoding (cp1252 on this box),
    and text=True decodes with that locale -- so a single stray 0x9d from a noisy
    frame killed the whole run at bout 5 of 21. The garbage frames are exactly the
    ones we expect to fail the regex anyway; they must not be fatal.
    """
    return (out or b"").decode("utf-8", errors="replace")


def ocr_png(png: Path) -> str:
    """OCR one already-extracted strip."""
    r = subprocess.run(
        ["tesseract", str(png), "stdout", "--psm", "7", "-l", "eng"],
        capture_output=True)
    return _decode_ocr(r.stdout).strip()


def ocr_batch(pngs: list[Path], tmp: Path) -> list[str]:
    """OCR many strips in ONE tesseract process. Returns one string per input.

    Given a file whose name ends in .txt, tesseract reads it as a LIST of images
    and emits their text separated by form feeds. Measured 24s -> 10s for 135
    strips: the per-launch overhead dominated, not the recognition.

    The output must line up index-for-index with the input, because find_caption
    maps index back to a timestamp for first_seen_t. Tesseract emits exactly one
    page per input in order, so a form-feed split is the mapping -- but it is
    verified against len(pngs) below rather than assumed, and a mismatch falls back
    to per-file OCR. Silently misaligning reads would shift every caption_t and put
    the clip window in the wrong place, which is precisely the class of bug that
    cost the first dataset.
    """
    if not pngs:
        return []
    lst = tmp / "_ocr_list.txt"
    lst.write_text("\n".join(str(p) for p in pngs), encoding="utf-8")
    r = subprocess.run(["tesseract", str(lst), "stdout", "--psm", "7", "-l", "eng"],
                       capture_output=True)
    pages = _decode_ocr(r.stdout).split("\f")
    # Tesseract appends a trailing form feed after the last page on some builds.
    if len(pages) == len(pngs) + 1 and not pages[-1].strip():
        pages.pop()
    if len(pages) != len(pngs):
        print(f"      OCR batch returned {len(pages)} pages for {len(pngs)} strips "
              f"-- falling back to per-file OCR")
        return [ocr_png(p) for p in pngs]
    return [p.strip() for p in pages]


def caption_candidates(path: Path, t0: float, end: float, w: int, h: int
                       ) -> set[int] | None:
    """Frame indices whose caption strip has enough edge energy to hold text.

    Returns None if the gradient pass cannot be run, meaning "OCR everything" --
    degrading to the slow-but-correct path rather than to an empty result.
    """
    if EDGE_MIN <= 0:
        return None
    cw, ch, cx, cy = box_px(WIN_BOX, w, h)
    W, H = cw * EDGE_SCALE, ch * EDGE_SCALE
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.2f}",
         "-t", f"{end - t0:.2f}", "-i", str(path),
         "-vf", f"fps={SEARCH_FPS},crop={cw}:{ch}:{cx}:{cy},"
                f"scale=iw*{EDGE_SCALE}:ih*{EDGE_SCALE}:flags=lanczos,format=gray",
         "-pix_fmt", "gray", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (W * H)
    if n == 0:
        return None
    f = np.frombuffer(raw[: n * W * H], dtype=np.uint8) \
          .reshape(n, H, W).astype(np.float32)
    edge = np.abs(np.diff(f, axis=2)).mean(axis=(1, 2))
    return {int(i) for i in np.flatnonzero(edge > EDGE_MIN)}


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

    strips = extract_strips(path, t0, end, w, h, tmp)
    # Skip strips that cannot hold text before paying for OCR. Both passes sample at
    # SEARCH_FPS from the same t0, so their indices refer to the same frames.
    keep = caption_candidates(path, t0, end, w, h)
    idx = [i for i in range(len(strips))
           if keep is None or i in keep]
    texts = ocr_batch([strips[i] for i in idx], tmp)

    for i, text in zip(idx, texts):
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
    ap.add_argument("--probe-window", metavar="VIDEO_ID",
                    help="dump a contact sheet of DETECTED BOUT WINDOWS for a video "
                         "already present in --out; every row should read "
                         "crouch -> charge -> grapple -> finish")
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

    if args.probe_window:
        # Needs caption_t, which only exists once a bout has been OCR'd -- so this
        # probes an already-labelled video rather than a fresh one. That is the
        # right order anyway: OCR first (cheap to check via --probe), then confirm
        # the window the labels will be paired with.
        cands = list(args.raw.glob(f"*{args.probe_window}*.mp4"))
        if not cands:
            sys.exit(f"no video matching {args.probe_window} in {args.raw}/")
        if not args.out.exists():
            sys.exit(f"{args.out} not found -- label some bouts first, "
                     f"--probe-window needs their caption times")
        labeled = [r for r in csv.DictReader(args.out.open(encoding="utf-8"))
                   if args.probe_window in r["video"] and r.get("caption_t")]
        if not labeled:
            sys.exit(f"no labelled bouts for {args.probe_window} in {args.out}")
        out_dir = Path("crops/_probe")
        out_dir.mkdir(parents=True, exist_ok=True)
        png = out_dir / f"hb_window_{args.probe_window}.png"
        # Spread the sample across the video so an anchor that only works early
        # (different graphics in the makuuchi vs juryo segments) shows up.
        step = max(1, len(labeled) // 6)
        print(probe_window(cands[0], labeled[::step][:6], png))
        print(f"\nwrote {png}\nEvery row should span the bout: crouch, charge, "
              f"grapple, finish, wrestlers\non the dohyo through the middle. Rows of "
              f"crowd shots or close-ups mean the\nanchors need recalibrating for "
              f"this source (see hb_window.py).")
        return

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
    window_misses: list[tuple[str, str, str]] = []
    for p in prev_rows:
        hits[p["kimarite"]] += 1

    # Append-and-flush per bout so a kill loses at most the bout in flight.
    fields = list(rows[0]) + ["kimarite", "winner", "votes", "caption_t", "clip",
                              "bout_start", "bout_end"]
    # Appending to a CSV whose header predates bout_start/bout_end would write the
    # new column order under the old header, silently shifting every value. Rewrite
    # the file instead when the schema has changed.
    stale_header = bool(prev_rows) and set(prev_rows[0]) != set(fields)
    new_file = stale_header or not (args.resume and args.out.exists() and prev_rows)
    if stale_header:
        print(f"{args.out} predates the bout_start/bout_end columns -- rewriting "
              f"its header (old rows keep blank bout bounds)")
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

        # Locate the bout in the video. A bout whose window cannot be found is a
        # MISS, not a clip cut on a guess: the label would be right and the footage
        # wrong, which is undetectable downstream -- it just holds the score at
        # chance. Counted separately from OCR misses so the two failure modes stay
        # distinguishable (bad graphics vs bad manifest window).
        got = find_bout_window(path, start, t_cap)
        if isinstance(got, str):
            window_misses.append((r["video"], r["n"], got))
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  {canon:<16} "
                  f"({winner}, {votes} votes)  NO WINDOW -- {got}")
            continue
        clip_start, clip_end = got

        hits[canon] += 1
        print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  {canon:<16} "
              f"({winner}, {votes} votes)  bout {clip_start:.1f}-{clip_end:.1f} "
              f"({clip_end - clip_start:.1f}s)")

        dest = args.data / canon / f"{r['video']}__{int(r['n']):02d}.mp4"
        ok = cut_clip(path, dest, clip_start, clip_end, mask_filter(w, h))
        row = {**r, "kimarite": canon, "winner": winner, "votes": votes,
               "caption_t": round(t_cap, 2), "clip": str(dest) if ok else "",
               "bout_start": round(clip_start, 2), "bout_end": round(clip_end, 2)}
        out_rows.append(row)
        writer.writerow(row)
        sink.flush()   # survive a kill: at most the in-flight bout is lost

    sink.close()

    print(f"\n{sum(hits.values())} labeled, {len(misses)} OCR missed, "
          f"{len(window_misses)} no bout window, "
          f"{sum(higi.values())} non-technique -> {args.out}")
    for k, n in hits.most_common(20):
        print(f"  {k:<24} {n:>4}")
    if higi:
        print("\nskipped as non-techniques (higi -- no technique was applied): "
              + ", ".join(f"{k} x{n}" for k, n in higi.most_common()))
    if misses:
        print(f"\nfirst OCR misses: {misses[:5]}")
        rate = len(misses) / max(1, len(rows))
        if rate > 0.2:
            print(f"MISS RATE {rate:.0%} -- re-check WIN_BOX with --probe, or widen\n"
                  f"SEARCH_TAIL (caption may fall outside the searched window tail).")
    if window_misses:
        print(f"\nread the caption but could not locate the bout "
              f"({len(window_misses)}):")
        for reason, n in Counter(r for _, _, r in window_misses).most_common():
            print(f"  {n:>4}x {reason}")
        rate = len(window_misses) / max(1, len(rows))
        if rate > 0.2:
            print(f"WINDOW MISS RATE {rate:.0%} -- check the anchors by eye with\n"
                  f"  python hb_label.py --probe-window <video-id>\n"
                  f"The banner anchor is channel-specific, like WIN_BOX.")


if __name__ == "__main__":
    main()
