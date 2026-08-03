"""Locate the BOUT inside a manifest window, so clips contain the wrestling.

WHY THIS EXISTS. hb_label.py used to cut each clip at a fixed offset behind the
win caption (CLIP_BACK = 4s), on the assumption that the caption follows the bout
closely. It does not -- it appears during the REPLAY. Measured caption-minus-
tachiai over 14 bouts ran 13-91s, clustering at 25-40, so the 6s window landed
12-35s AFTER the bout ended. 59% of the first 68-clip dataset held no bout footage
at all: post-bout close-ups of the winner's face, crowd shots and walk-offs.

That was not a small quality problem, it was the whole result. A head trained on
those clips scored at chance (balanced accuracy 0.246 vs 0.250 chance, permutation
p = 0.70) while the SAME embeddings predicted the source video at 0.796 vs 0.250 --
they were healthy and information-rich, encoding venue and lighting, because that
is all most clips contained.

No fixed offset can fix it: bouts run 2-40s and the ceremonial gap varies. So
anchor on the video itself.

START ANCHOR -- the pre-bout name banner. The broadcast holds a saturated
full-width lower-third graphic (both wrestlers' names and records) through the
final crouch and pulls it the instant they charge. The END of the last sustained
banner run before the caption is therefore the tachiai.

END ANCHOR -- the first hard scene cut at least MIN_BOUT seconds after the
tachiai. The camera holds the dohyo for the whole bout and cuts away once it is
over. The cut lands slightly after the finish (the last second or two is usually
the walk-off), which is harmless: we want to SPAN the bout, and VideoMAE samples
16 frames evenly across whatever it is given.

Both anchors depend on this channel's graphics, exactly like WIN_BOX. Verify with
`hb_label.py --probe-window <video-id>` before trusting a long run on a new source.
"""

import re
import subprocess

import numpy as np

# Analysis resolution for the banner signal. Tiny on purpose: this is a
# whole-region colour statistic, not OCR, and 160x90 keeps a 60s window under a
# second of decode.
AW, AH = 160, 90
BANNER_FPS = 4.0

# The banner spans the lower third. Sampling y 0.72-0.98 keeps it clear of the
# wrestlers' bodies (which reach the lower-middle of frame during a bout).
BANNER_Y = (0.72, 0.98)

# The win caption is ALSO a saturated lower-third graphic, so it has to stay out
# of the search or it wins the "last sustained run" every time. 8s of guard: the
# caption fades in over ~1s and the nearest real banner is >13s earlier.
CAPTION_GUARD = 8.0

# The banner is held continuously through the crouch. Requiring a sustained run
# rejects one-frame saturation spikes (a yellow-robed gyoji sweeping past, a
# sponsor board carried across the dohyo).
MIN_BANNER_RUN = 3.0
BANNER_FRAC = 0.6           # of the window's saturation range, for "banner on"

SCENE_THR = 0.35            # ffmpeg scene score for a hard cut
SCENE_SCALE = (320, 180)    # cuts survive downscaling; full-res decode is 4x slower
MIN_BOUT = 3.0              # ignore cuts this soon after the tachiai
FALLBACK_SPAN = 12.0        # no cut found: median bout + finish

# Sanity bounds on caption-minus-tachiai, from the measured distribution. Below
# MIN the "banner" was almost certainly a different graphic (the anchor fired late,
# leaving a 4s span of walk-off); above MAX it belongs to an earlier bout in an
# over-long manifest window. Both are reported as misses -- a wrong window is
# invisible once it is an .mp4, while a miss shows up in the count.
MIN_CAP_GAP = 15.0
MAX_CAP_GAP = 70.0

MIN_SPAN = 3.0
MAX_SPAN = 30.0

# --- Locating the bout directly, rather than trusting an anchor -----------------
#
# The banner anchor alone is not enough. This broadcast shows a wrestler-PROFILE
# graphic (both men's records) in the same lower-third position as the pre-bout name
# banner, so "last sustained saturated lower-third" sometimes locks onto the profile
# card, which is followed by close-up walk-ups rather than by the charge. That put 8
# of 54 re-cut windows entirely in post-bout footage.
#
# Two rejected discriminators, both measured over all 54 windows -- do not re-add
# either without re-measuring:
#   * frame-to-frame MOTION: junk 9.8-35.8 vs good 9.0-35.5, fully overlapping.
#     A close-up of a walking wrestler fills the frame and moves as much as a bout.
#   * total CLAY FRACTION: junk 19-48% vs good 28-57%, overlapping.
#
# What works is the SHAPE of the clay, not its amount: a wide dohyo shot spreads
# pale clay across nearly the full frame WIDTH, while a close-up shows a big blob in
# part of the frame. Measured per-window clay-width: good shots 85-100%, the junk
# ones 43-70%. So instead of anchoring and hoping, score every second of the window
# and take the last sustained run of wide-dohyo frames before the caption -- that run
# IS the bout.
CLAY_COL_FRAC = 0.15    # a column "shows clay" if this fraction of its pixels match
# 0.80, not 0.78: at 0.78 one close-up (08KqsfacUFo#19) scored 0.77 and squeezed
# through as a 3.2s "bout". The genuine wide shots all sit at 0.89-1.00, so there is
# real headroom above the close-ups rather than a knife-edge fit to one sample.
WIDE_FRAC = 0.80        # a frame is a wide dohyo shot if this many columns show clay
MIN_WIDE_RUN = 2.5      # sustained, to reject a wide establishing shot flash
WIDE_GAP_BRIDGE = 1.0   # bridge brief cutaways inside a bout (crowd reaction shots)

# Accept only windows that clear BOTH a duration floor and a purity floor, checked
# after selection. Verified by eye over all 68 bouts: every genuine bout window ran
# >=8s at >=90% wide frames, while every remaining junk window failed one or the
# other (close-ups at 56-86% wide, or 3-4s slivers). Two separate floors rather than
# one tuned threshold, because the two failure modes are different -- a close-up is
# impure, a mis-sliced bout is short -- and a single knob cannot express both.
#
# This deliberately trades yield for cleanliness: ~15% of bouts are dropped. A
# dropped bout costs one training sample and is visible in the count; a kept-but-
# wrong one is invisible and holds the whole model at chance. That asymmetry is the
# same reason the OCR side uses plurality voting.
MIN_ACCEPT_SPAN = 8.0
MIN_ACCEPT_WIDE = 0.90


def _decode_small(path, t0: float, t1: float) -> np.ndarray | None:
    """[frames, AH, AW, 3] float32 at BANNER_FPS, or None if too short."""
    if t1 - t0 < MIN_BANNER_RUN:
        return None
    raw = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-ss", f"{t0:.2f}",
         "-t", f"{t1 - t0:.2f}", "-i", str(path),
         "-vf", f"fps={BANNER_FPS},scale={AW}:{AH}",
         "-pix_fmt", "rgb24", "-f", "rawvideo", "-"],
        capture_output=True).stdout
    n = len(raw) // (AW * AH * 3)
    if n < BANNER_FPS * MIN_BANNER_RUN:
        return None
    return np.frombuffer(raw[: n * AW * AH * 3], dtype=np.uint8) \
             .reshape(n, AH, AW, 3).astype(np.float32)


def find_tachiai(path, start: float, caption_t: float) -> float | None:
    """Time of the charge, from the pre-bout banner disappearing. None if unclear.

    Saturation, not brightness: the banner is a coloured graphic over whatever the
    camera happens to be showing, so its signature is chroma, and that survives the
    difference between a bright arena and a dim one. Measured against frames
    checked by eye, this lands within ~1s (351.8 detected vs 351.8 observed).
    """
    frames = _decode_small(path, start, caption_t - CAPTION_GUARD)
    if frames is None:
        return None
    y0, y1 = int(BANNER_Y[0] * AH), int(BANNER_Y[1] * AH)
    reg = frames[:, y0:y1, :, :]
    sat = (1.0 - reg.min(-1) / np.maximum(reg.max(-1), 1e-6)).mean(axis=(1, 2))

    hot = sat > sat.min() + BANNER_FRAC * (sat.max() - sat.min())
    runs, i = [], 0
    while i < len(hot):
        if not hot[i]:
            i += 1
            continue
        j = i
        while j < len(hot) and hot[j]:
            j += 1
        if (j - i) >= BANNER_FPS * MIN_BANNER_RUN:
            runs.append((i, j))
        i = j
    if not runs:
        return None
    return start + runs[-1][1] / BANNER_FPS


def scene_cuts(path, t0: float, t1: float) -> list[float]:
    """Hard-cut timestamps in [t0, t1], via ffmpeg's own scene score.

    ffmpeg's filter rather than a hand-rolled frame difference: it compares
    consecutive frames in ffmpeg's colourspace and reports a normalized score, so
    one threshold covers both the 854x480 and 640x360 sources in this dataset.
    """
    if t1 <= t0:
        return []
    r = subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "info", "-ss", f"{t0:.2f}",
         "-t", f"{t1 - t0:.2f}", "-i", str(path),
         "-vf", f"scale={SCENE_SCALE[0]}:{SCENE_SCALE[1]},"
                f"select='gt(scene,{SCENE_THR})',metadata=print:file=-",
         "-an", "-f", "null", "-"],
        capture_output=True)
    txt = (r.stdout or b"").decode("utf-8", errors="replace")
    return [t0 + float(m) for m in re.findall(r"pts_time:([0-9.]+)", txt)]


def wide_dohyo_mask(frames: np.ndarray) -> np.ndarray:
    """Per-frame bool: is this a wide shot of the dohyo?

    Clay is bright, warm (R>G>B) and not very saturated -- distinguishing it from
    the red/purple of robes and seat backs, which are warm but saturated. Then the
    WIDTH test: count columns containing clay, rather than total clay pixels, so a
    close-up filling half the frame with sand does not pass.
    """
    r, g, b = frames[..., 0], frames[..., 1], frames[..., 2]
    mx, mn = frames.max(-1), frames.min(-1)
    clay = ((mx > 90) & (r > g) & (g > b)
            & ((mx - mn) < 0.55 * np.maximum(mx, 1)) & (r - b > 12))
    cols = clay.mean(axis=1)                      # [frames, AW]
    return (cols > CLAY_COL_FRAC).mean(axis=1) > WIDE_FRAC


def _runs(flags: np.ndarray, fps: float, min_len: float,
          bridge: float = 0.0) -> list[tuple[int, int]]:
    """Index runs of True at least min_len seconds long, bridging short gaps."""
    f = flags.copy()
    if bridge > 0:
        gap = int(bridge * fps)
        i = 0
        while i < len(f):
            if f[i]:
                i += 1
                continue
            j = i
            while j < len(f) and not f[j]:
                j += 1
            # Fill a gap only if it is short AND has True on both sides.
            if j - i <= gap and i > 0 and j < len(f):
                f[i:j] = True
            i = j
    out, i = [], 0
    while i < len(f):
        if not f[i]:
            i += 1
            continue
        j = i
        while j < len(f) and f[j]:
            j += 1
        if (j - i) >= fps * min_len:
            out.append((i, j))
        i = j
    return out


def find_bout_by_clay(path, start: float, caption_t: float
                      ) -> tuple[float, float] | None:
    """(t0, t1) of the longest sustained wide-dohyo stretch before the caption.

    This is the primary detector: it looks for the bout itself rather than for a
    graphic that precedes it, so it is immune to the profile-card confusion that
    defeats the banner anchor.
    """
    t_end = caption_t - CAPTION_GUARD
    frames = _decode_small(path, start, t_end)
    if frames is None:
        return None
    wide = wide_dohyo_mask(frames)
    # No gap-bridging. Bridging merged the pre-bout crouch shot, the bout and the
    # post-bout wide shots into one 57s blob; the later MAX_SPAN clamp then sliced a
    # 2.5s sliver off its start, producing "bouts" that were 0% wide. The
    # unbridged runs already track the real shots (a 34.8s run covering a bout
    # verified by eye at 176-193), so take them as they are.
    runs = _runs(wide, BANNER_FPS, MIN_WIDE_RUN)
    if not runs:
        return None
    # The LONGEST wide run, not the last. The broadcast returns to a wide dohyo shot
    # after the bout (the winner's bow, the next pairing walking on), so "last" often
    # picks a short post-bout shot. The bout is the sustained one -- the camera holds
    # the dohyo continuously from the crouch through the finish.
    i, j = max(runs, key=lambda ij: ij[1] - ij[0])
    return start + i / BANNER_FPS, start + j / BANNER_FPS


def find_bout_window(path, start: float, caption_t: float
                     ) -> tuple[float, float] | str:
    """(bout_start, bout_end) to cut, or a string explaining why not.

    A string return (rather than None) so the caller can report WHICH anchor gave
    up; the failure modes need different fixes -- 'no banner' means re-probe the
    graphics, 'gap' means the manifest window covers more than one bout.
    """
    got = find_bout_by_clay(path, start, caption_t)
    if got is None:
        return "no wide dohyo shot found before the caption"
    t, end = got

    # Cross-check with the banner anchor when it agrees. The banner marks the charge
    # precisely (measured within ~1s), whereas the clay run starts wherever the wide
    # shot does -- usually a few seconds earlier, during the crouch. When the two are
    # close, prefer the banner's sharper start; when they disagree, the banner locked
    # onto a profile card and the clay run is the one to trust.
    b = find_tachiai(path, start, caption_t)
    if b is not None and 0.0 <= b - t <= 12.0 and end - b >= MIN_SPAN:
        t = b

    # Clamp by moving the START, not the end. The finish is what identifies the
    # technique (oshidashi vs oshitaoshi differ only in whether the loser ended up
    # standing or fallen), so when a run exceeds MAX_SPAN the surplus to discard is
    # the ceremonial lead-in, not the decisive last seconds. Clamping the end instead
    # was what turned a correct 34.8s run into a 2.5s sliver of empty dohyo.
    end = min(end, caption_t - CAPTION_GUARD)
    t = max(t, end - MAX_SPAN)
    if end - t < MIN_ACCEPT_SPAN:
        return f"bout span only {end - t:.1f}s (need {MIN_ACCEPT_SPAN:.0f}s)"
    gap = caption_t - t
    if gap > MAX_CAP_GAP:
        return f"bout {gap:.0f}s before caption (likely an earlier bout)"

    # Purity check on the FINAL window: the banner cross-check above can move the
    # start earlier, pulling in close-up frames that were not in the clay run.
    frames = _decode_small(path, t, end)
    if frames is None:
        return "could not re-decode the chosen window"
    wide = wide_dohyo_mask(frames).mean()
    if wide < MIN_ACCEPT_WIDE:
        return f"only {wide:.0%} of the window is a wide dohyo shot"
    return t, end


def cut_clip(path, dest, t0: float, t1: float, mask_vf: str) -> bool:
    """Cut [t0,t1] with the leak masks burned in. Shared by labelling and re-cutting."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    return subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t0:.2f}",
         "-i", str(path), "-t", f"{t1 - t0:.2f}", "-an",
         "-vf", mask_vf, "-c:v", "libx264", "-preset", "veryfast", str(dest)],
        capture_output=True).returncode == 0


def probe_window(path, rows, out_png, tiles: int = 8) -> str:
    """Contact sheet: one row per bout, `tiles` frames spanning the detected window.

    The point of eyeballing this is that a wrong window is otherwise undetectable:
    the label is right, the clip plays, the training run completes, and the score
    just quietly sits at chance. Every row here should read crouch -> charge ->
    grapple -> finish, with wrestlers on the dohyo through the middle.
    """
    strips, notes = [], []
    for i, r in enumerate(rows, 1):
        got = find_bout_window(path, float(r["start"]), float(r["caption_t"]))
        if isinstance(got, str):
            notes.append(f"  row -- #{r['n']:<3} SKIPPED: {got}")
            continue
        t0, t1 = got
        strip = out_png.parent / f"_probe_row_{len(strips) + 1:02d}.png"
        subprocess.run(
            ["ffmpeg", "-nostdin", "-v", "error", "-y", "-ss", f"{t0:.2f}",
             "-t", f"{t1 - t0:.2f}", "-i", str(path),
             "-vf", f"fps={tiles / (t1 - t0):.4f},scale=160:90,tile={tiles}x1",
             "-frames:v", "1", str(strip)],
            capture_output=True)
        if strip.exists():
            strips.append(strip)
            notes.append(f"  row {len(strips):<2} #{r['n']:<3} "
                         f"{t0:.1f}-{t1:.1f} ({t1 - t0:.1f}s)  {r.get('kimarite','')}")
    if not strips:
        return "no window could be located for any sampled bout"
    # Renumber to a gap-free sequence so ffmpeg's image2 demuxer reads them all.
    for k, s in enumerate(strips, 1):
        s.rename(s.parent / f"_probe_seq_{k:02d}.png")
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-y",
         "-i", str(out_png.parent / "_probe_seq_%02d.png"),
         "-vf", f"tile=1x{len(strips)}", "-frames:v", "1", str(out_png)],
        capture_output=True)
    for s in out_png.parent.glob("_probe_seq_*.png"):
        s.unlink()
    return "\n".join(notes)
