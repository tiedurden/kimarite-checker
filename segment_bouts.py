"""Stage 1: find kimarite overlays; emit bout clips + label crops.

The overlay is BOTH the label source and a forbidden model input, so it drives
two outputs pointing opposite directions in time:

    overlay onset at T
        --> crops/<vid>__<n>.png   = frame at T+HOLD   (label, via clustering)
        --> data/_unlabeled/<vid>__<n>.mp4 = [T-LEAD-PAD, T-PAD]  (model input)

The clip ENDS before the overlay exists. If the overlay reached a training frame
the model would learn to read the caption instead of watching the wrestling --
high accuracy that collapses to chance on uncaptioned video.

Usage:
    python segment_bouts.py --probe raw/VIDEOID.mp4   # calibrate crop box first
    python segment_bouts.py                           # process all of raw/
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np

# --- overlay geometry: fractions of frame w/h, so resolution-independent -------
# CALIBRATED against NHK GRAND SUMO Juryo highlights (854x480, July 2026 Nagoya).
# The kimarite plate sits bottom-left: romaji name + English gloss underneath,
# e.g. "TSUKIOTOSHI / THRUST DOWN". Measured at ~x 8-275, y 375-465 of 854x480.
KIMARITE_BOX = (0.00, 0.76, 0.34, 0.99)  # x0, y0, x1, y1

BOX = KIMARITE_BOX  # what --probe and onset detection look at

# Other graphics that also carry the answer and must be masked out of clips:
#   ANALYSIS badge  - top-right, persistent during analysis segments
#   caption banner  - top-centre live commentary ("PUSH OUT ATTEMPT BY NISHIKIGI")
#   win/loss plate  - bottom-right, names + records at bout end
# The top-centre banner is the dangerous one: it names the technique DURING the
# bout, inside the clip window, so PAD/masking of the bottom-left alone is not
# enough. See LEAK_BOXES.
LEAK_BOXES = [
    KIMARITE_BOX,
    (0.78, 0.00, 1.00, 0.16),  # ANALYSIS badge (top-right)
    (0.05, 0.00, 0.75, 0.20),  # commentary banner (top-centre)
    (0.63, 0.75, 1.00, 1.00),  # win/loss plate (bottom-right)
]

# --- timing (seconds) ---------------------------------------------------------
HOLD = 1.0   # after onset, when the overlay is fully drawn (animations settle)
# PAD is deliberately small. Seven kimarite pairs (oshidashi/oshitaoshi,
# yorikiri/yoritaoshi, ...) are the SAME technique distinguished only by whether
# the opponent ended up standing or fallen -- evidence that exists solely in the
# final ~0.5s. A large PAD trims exactly that and makes those pairs unlearnable.
# Safe to keep small because MASK_OVERLAY blanks the caption region regardless.
PAD  = 0.2
LEAD = 5.0   # clip length; sumo bouts average ~5s
MIN_GAP = 15.0  # min seconds between accepted onsets (dedupes a held overlay)

# Paint the overlay box black in every extracted clip frame. Belt-and-braces
# against label leakage: if the caption reached a training frame, the model would
# learn to READ the answer rather than watch the wrestling -- scoring ~99% here
# and collapsing to chance on uncaptioned video. Masking makes that impossible,
# which is what lets PAD stay small enough to keep the bout's decisive moment.
MASK_OVERLAY = True
MASK_MARGIN = 0.03  # expand the box slightly; overlays animate in from an edge

SAMPLE_FPS = 4.0  # overlay detection rate; cheap, and onsets last seconds


def probe_dims(path: Path) -> tuple[int, int]:
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(path)],
        capture_output=True, text=True)
    try:
        w, h = r.stdout.strip().split(",")[0].split("x")[:2]
        return int(w), int(h)
    except ValueError:
        sys.exit(f"ffprobe could not read dimensions of {path}: {r.stderr.strip()}")


def crop_filter(w: int, h: int) -> str:
    x0, y0, x1, y1 = BOX
    cw, ch = max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h))
    return f"crop={cw}:{ch}:{int(x0 * w)}:{int(y0 * h)}"


def probe(path: Path, out_dir: Path, n: int = 12) -> None:
    """Dump N evenly-spaced crops of the overlay box so the box can be checked."""
    w, h = probe_dims(path)
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "csv=p=0", str(path)], capture_output=True, text=True
    ).stdout.strip() or 0)
    if dur <= 0:
        sys.exit(f"could not read duration of {path}")

    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"{path.name}: {w}x{h}, {dur:.0f}s -> {n} crops in {out_dir}/")
    for i, t in enumerate(np.linspace(dur * 0.05, dur * 0.95, n)):
        dest = out_dir / f"probe_{i:02d}_t{int(t)}.png"
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
             "-vf", crop_filter(w, h), "-frames:v", "1", str(dest)],
            capture_output=True)
    print("\nInspect them: does the box tightly contain the kimarite text?\n"
          "If not, edit BOX at the top of this file (fractions of w/h) and re-probe.")


def overlay_signal(path: Path, w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """Per-sampled-frame mean abs difference of the overlay box vs. its median.

    The box is static background most of the time, so its median over the whole
    video approximates 'no overlay'. Frames where the overlay is up deviate
    sharply. Robust to broadcast noise; no template or OCR needed.
    """
    x0, y0, x1, y1 = BOX
    cw, ch = max(2, int((x1 - x0) * w)), max(2, int((y1 - y0) * h))
    # Downscale the crop: we need a change signal, not legible text.
    sw, sh = max(8, cw // 8), max(8, ch // 8)

    cmd = ["ffmpeg", "-v", "error", "-i", str(path),
           "-vf", f"fps={SAMPLE_FPS},{crop_filter(w, h)},scale={sw}:{sh},format=gray",
           "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True).stdout
    fsz = sw * sh
    nf = len(raw) // fsz
    if nf == 0:
        return np.array([]), np.array([])

    frames = np.frombuffer(raw[: nf * fsz], dtype=np.uint8).reshape(nf, fsz).astype(np.float32)
    baseline = np.median(frames, axis=0)
    score = np.abs(frames - baseline).mean(axis=1)
    times = np.arange(nf) / SAMPLE_FPS
    return times, score


def find_onsets(times: np.ndarray, score: np.ndarray, thresh_mult: float) -> list[float]:
    """Rising edges of the overlay signal, deduped by MIN_GAP."""
    if len(score) == 0:
        return []
    # Threshold relative to the video's own spread: absolute values vary with
    # broadcast, graphics style, and how busy the background is.
    med = np.median(score)
    mad = np.median(np.abs(score - med)) or 1.0
    thresh = med + thresh_mult * mad

    onsets, above_prev = [], False
    for t, s in zip(times, score):
        above = s > thresh
        if above and not above_prev and (not onsets or t - onsets[-1] >= MIN_GAP):
            onsets.append(float(t))
        above_prev = above
    return onsets


def mask_filter(w: int, h: int) -> str:
    """drawbox filters blacking out EVERY region that can leak the answer.

    Not just the bottom-left kimarite plate: this broadcast also prints live
    commentary top-centre naming the technique mid-bout ("PUSH OUT ATTEMPT BY
    NISHIKIGI"), which lands inside the clip window. Masking only the final
    caption would leave that text in the training frames.
    """
    m = MASK_MARGIN
    parts = []
    for x0, y0, x1, y1 in LEAK_BOXES:
        bx, by = max(0.0, x0 - m), max(0.0, y0 - m)
        bw, bh = min(1.0, x1 + m) - bx, min(1.0, y1 + m) - by
        parts.append(f"drawbox=x={int(bx * w)}:y={int(by * h)}"
                     f":w={int(bw * w)}:h={int(bh * h)}:color=black:t=fill")
    return ",".join(parts)


def cut(path: Path, start: float, dur: float, dest: Path, w: int, h: int) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.2f}", "-i", str(path),
           "-t", f"{dur:.2f}", "-an"]
    if MASK_OVERLAY:
        cmd += ["-vf", mask_filter(w, h)]
    cmd += ["-c:v", "libx264", "-preset", "veryfast", str(dest)]
    r = subprocess.run(cmd, capture_output=True)
    return r.returncode == 0 and dest.exists()


def still(path: Path, t: float, w: int, h: int, dest: Path) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    r = subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.2f}", "-i", str(path),
         "-vf", crop_filter(w, h), "-frames:v", "1", str(dest)],
        capture_output=True)
    return r.returncode == 0 and dest.exists()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=Path, help="dump crops of one video and exit")
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--crops", default="crops", type=Path)
    ap.add_argument("--data", default="data", type=Path)
    ap.add_argument("--thresh-mult", type=float, default=6.0,
                    help="onset sensitivity in MADs; lower finds more")
    ap.add_argument("--expect", type=int, default=15,
                    help="bouts per video; only used to warn on bad detection")
    args = ap.parse_args()

    for tool in ("ffmpeg", "ffprobe"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            sys.exit(f"{tool} not on PATH -- see README.md (Prerequisites)")

    if args.probe:
        probe(args.probe, args.crops / "_probe")
        return

    vids = sorted(args.raw.glob("*.mp4"))
    if not vids:
        sys.exit(f"no .mp4 in {args.raw}/ -- run fetch_playlist.py first")

    # Unlabeled until clustering assigns categories; label_clusters.py moves
    # these into data/<kimarite>/.
    out_clips = args.data / "_unlabeled"
    total, suspect = 0, []

    for i, v in enumerate(vids, 1):
        w, h = probe_dims(v)
        times, score = overlay_signal(v, w, h)
        onsets = find_onsets(times, score, args.thresh_mult)
        print(f"[{i}/{len(vids)}] {v.stem}: {len(onsets)} onsets", end="")

        # Detection quality is the whole ballgame here -- flag anything odd
        # rather than silently producing a corrupt dataset.
        if not (args.expect * 0.6 <= len(onsets) <= args.expect * 1.6):
            suspect.append((v.stem, len(onsets)))
            print("  <-- unexpected count", end="")
        print()

        n = 0
        for onset in onsets:
            start = onset - PAD - LEAD
            if start < 0:
                continue  # overlay too early to have a full clip before it
            n += 1
            base = f"{v.stem}__{n:02d}"
            ok_clip = cut(v, start, LEAD, out_clips / f"{base}.mp4", w, h)
            ok_crop = still(v, onset + HOLD, w, h, args.crops / f"{base}.png")
            if ok_clip and ok_crop:
                total += 1
            else:
                # Never keep a clip without its label crop, or vice versa.
                (out_clips / f"{base}.mp4").unlink(missing_ok=True)
                (args.crops / f"{base}.png").unlink(missing_ok=True)

    print(f"\n{total} bouts -> {out_clips}/ + {args.crops}/")
    if suspect:
        print(f"\n{len(suspect)} video(s) with unexpected onset counts:")
        for stem, c in suspect[:10]:
            print(f"  {stem}: {c} (expected ~{args.expect})")
        print("Tune --thresh-mult, or re-check BOX with --probe.")
    print("\nnext: python label_clusters.py")


if __name__ == "__main__":
    main()
