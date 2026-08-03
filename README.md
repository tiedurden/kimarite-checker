# Sumo kimarite classifier

Predict the **winning technique (kimarite)** of a sumo bout from video, using a
frozen VideoMAEv2 encoder plus a small trained classifier head. No encoder
fine-tuning -- runs on a 4 GB GPU.

Labels come free: the broadcast prints the kimarite on screen after each bout.
That caption is both the label source and a forbidden model input.

```
playlist --> fetch_playlist.py --> raw/<videoid>.mp4
                                        |
                                 segment_bouts.py   (finds overlay onsets)
                                   /            \
                    crops/<vid>__<n>.png     data/_unlabeled/<vid>__<n>.mp4
                     (the caption = LABEL)    (bout, caption masked = INPUT)
                              |
                      label_clusters.py   (~15 decisions, not 675)
                              v
                     data/<kimarite>/*.mp4
                              |
                    extract_features.py   (GPU, slow, cached)
                              v
                       cache/<kimarite>/*.npy
                              |
                        train_head.py    (CPU, seconds)
                              v
                     models/head.joblib --> predict.py
```

Why this shape: the encoder is the expensive part and never changes, so it runs
once and its output is cached. All iteration -- label granularity, class merging,
hyperparameters -- happens in `train_head.py`, which re-trains in seconds.

## The trap that actually bit: clipping the wrong six seconds

Worth reading before touching the labelers, because it cost a full dataset and
looked exactly like a modelling problem.

`hb_label.py` originally cut each clip at a fixed offset behind the win caption,
assuming the caption follows the bout. **It appears during the replay.** Measured
caption-minus-tachiai over 14 bouts: **13-91 s**, clustering at 25-40. So the
6 s window landed 12-35 s after the bout ended, and **59% of the first 68-clip
dataset contained no wrestling at all** -- post-bout close-ups of the winner's
face, crowd shots, walk-offs.

Nothing failed. The labels were right, the clips played, training completed, and
the head sat at chance (balanced accuracy 0.246 vs 0.250) while the *same*
embeddings predicted the source video at 0.796 vs 0.250 chance. They were healthy
and information-rich -- encoding venue and lighting, because that is all most clips
held. Every visible symptom pointed at "too few videos".

**The lesson: verify what is IN the clips, not just that clips exist.** One contact
sheet would have caught this before four videos of labelling. Hence
`--probe-window`, which is now as mandatory as `--probe`:

```bash
python hb_label.py --probe-window <video-id>   # every row must show the bout
```

`hb_window.py` locates the bout by finding **the longest sustained wide shot of the
dohyo** before the caption -- it looks for the wrestling itself rather than for a
graphic that precedes it. Two discriminators were measured and **rejected**; don't
re-add either without re-measuring:

- **frame-to-frame motion** -- junk 9.8-35.8, good 9.0-35.5, fully overlapping. A
  close-up of a walking wrestler moves as much as a bout.
- **total clay fraction** -- junk 19-48%, good 28-57%, overlapping.

What separates them is the *shape* of the clay, not its amount: a wide dohyo shot
spreads pale clay across nearly the full frame width (89-100% of columns), a
close-up does not (43-70%).

## The two leakage traps

**1. The answer is printed in the frames.** A vision transformer reads glyphs
easily. If the caption reaches a single training frame, the model learns to read
it instead of watching the wrestling -- ~99% here, chance on uncaptioned video.
Two independent defenses: clips end *before* the overlay appears (`PAD`), and the
overlay region is painted black in every extracted frame (`MASK_OVERLAY`).

**2. Bouts from one video are not independent.** They share venue, lighting,
camera, commentary, often wrestlers. `train_head.py` splits by **source video**,
never by bout. With 45 videos your effective sample size for generalization is
**45, not 675** -- accuracy carries roughly +/-4-6 points of real uncertainty.
Don't chase a 2-point gain.

## Prerequisites

**Python 3.12**, not 3.13 -- ML wheels still lag on 3.13.

**ffmpeg** and **yt-dlp** on PATH. ffmpeg is required (frame decoding *and*
clip cutting):

```bash
winget install Gyan.FFmpeg        # then reopen the shell so PATH refreshes
ffmpeg -version                    # must print a version
```

**tesseract** on PATH, for the Herbert pipeline only (`winget install
UB-Mannheim.TesseractOCR`).

## Setup

```bash
cd ~/kimarite-checker
py -3.12 -m venv .venv
source .venv/Scripts/activate      # Git Bash;  .venv\Scripts\activate on cmd

# CUDA build if you have the VRAM and the bandwidth (2.5 GB):
pip install torch --index-url https://download.pytorch.org/whl/cu124
# CPU-only is ~200 MB and enough for a few hundred clips -- the encoder is frozen
# and each clip is embedded exactly once, so the GPU saves minutes, not hours:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Then `cp env.sh.example env.sh` and `source ./env.sh` **in every new shell** --
it puts ffmpeg/yt-dlp/tesseract and `.venv` on PATH.

> Make sure `python` is the venv's. If it resolves to a system Python while pip
> installs into `.venv`, scripts that use only stdlib run fine and hide the split,
> and then `extract_features.py` dies on `import torch` with the wheel installed.
> `env.sh` puts `.venv/Scripts` first and warns if it isn't there.

## Use

**1. Download the playlist:**

```bash
python fetch_playlist.py "https://youtube.com/playlist?list=..."
```

**2. Calibrate the overlay box, then segment.** Do the probe first -- the crop box
is the one thing that must be right, and it's video-specific:

```bash
python segment_bouts.py --probe raw/SOMEVIDEO.mp4   # writes crops/_probe/*.png
```

Open those PNGs. Each should tightly contain the kimarite caption. If not, edit
`BOX` at the top of `segment_bouts.py` (fractions of width/height) and re-probe.
Then:

```bash
python segment_bouts.py            # expect ~15 onsets/video; it warns if not
```

**3. Label by clustering.** The caption is a closed vocabulary rendered with
identical pixels, so ~675 crops collapse into ~15 visual groups:

```bash
python label_clusters.py                  # writes review/ sheets + labels.csv
#   inspect review/*.png, fill the `kimarite` column in labels.csv
python label_clusters.py --apply
```

Same name on two clusters merges them. Blank row drops a cluster.

**4. Embed** (GPU-bound; resumable):

```bash
python extract_features.py
```

**5. Train:**

```bash
python train_head.py                # top techniques + 'other'
python train_head.py --coarse       # 6 official families -- easier, good sanity check
```

**6. Predict:**

```bash
python predict.py bout.mp4
python predict.py https://youtu.be/XXXXXXXXXXX
```

## Second source: the Herbert pipeline (`hb_*.py`)

A different channel, a different broadcast, and **4.6x more footage** (110 videos,
64 h). It needs its own stages 0-1 because both NHK assumptions break: the caption
is *prose containing the winner's name* (`KINBOZAN WINS BY OSHIDASHI`), so its
pixel width varies per bout and clustering cannot group it; and the bout
boundaries are hand-written in the video descriptions, so no onset detection is
needed. See `HERBERT.md` for the full assessment.

```bash
python hb_manifest.py                     # descriptions -> hb_bouts.csv (1647 bouts)
python hb_label.py --probe <video-id>     # check WIN_BOX before a long run
python hb_label.py --resume               # OCR -> data_hb/<kimarite>/*.mp4
python hb_label.py --probe-window <id>    # then CHECK THE CLIPS HOLD THE BOUT
python hb_recut.py --dry-run              # re-cut existing labels after a fix
```

Both probes are calibration, not decoration: `--probe` checks the caption is being
read, `--probe-window` checks the footage is the bout. Skipping the second is what
produced a 59%-junk dataset.

From there it rejoins the NHK path: `extract_features.py --data data_hb` etc.

**Status as of the last session.** Manifest complete: **1647 bouts from 82
videos** (of 110; 27 have no parseable timestamps, 1 was a cross-playlist
duplicate). A 28-bout spot check across **5 tournaments and both resolutions**
(854x480 and 640x360) labeled 24/28 -- so `WIN_BOX`, calibrated on a single Haru
2026 video, transfers. **14% miss rate**, consistent across both checks.

**Labelled so far: 4 videos, 68 captions read, 60 clips** (88% of bouts locate a
usable window; the other 12% are dropped rather than guessed at). All 60 verified
to be 100% wide-dohyo footage. 10 of the 14 pilot videos remain, then 68 more in
the manifest.

Scores on those 60 clips, 4-fold grouped CV on the 6 official families:

| dataset | accuracy | balanced acc | f1_macro |
|---|---|---|---|
| caption-anchored (59% junk) | 0.460 | 0.246 | 0.202 |
| banner-anchored (15% junk) | 0.532 | 0.275 | 0.254 |
| bout-anchored (0% junk) | 0.467 | **0.301** | 0.260 |

Read `balanced_accuracy` against 0.250 chance, not `accuracy` against the 0.583
majority baseline -- the head uses `class_weight="balanced"`, so plain accuracy
penalizes it for not riding the majority class. Balanced accuracy has moved
0.246 -> 0.301 as the clips got cleaner, and the cross-video technique signal went
from p=0.70 (wrong direction) to p=0.12 (right direction). Both are still weak: 60
clips over 4 videos cannot resolve much. The next real test is more videos, now
that the ones being labelled contain the bout.

**Cost, measured: ~2.5 s/bout** single-process, so 270 bouts ~11 min and all 1647
~70 min. Do not parallelize on a laptop -- see the note in `extract_strips()`;
there is no longer much reason to.

It was 25 s/bout, of which OCR was 24 s -- and that was 135 tesseract *process
launches* per bout, not recognition. Two fixes, together 8x, verified to reproduce
all 68 existing labels exactly:

- **one batched tesseract call** instead of 135. Passed a `.txt` file it reads it as
  an image list and emits form-feed-separated pages (24 s -> 10 s).
- **an edge-energy prefilter** (`EDGE_MIN`) skipping strips too smooth to hold
  glyphs, cutting OCR to 31% of frames. Note this is *not* a clean separator -- over
  23 bouts only 7 separated cleanly. It works because the vote logic needs 2 reads
  out of ~21 caption frames, and the worst of 34 bouts still keeps 16.

**A GPU does not help here.** Labelling is ffmpeg + tesseract, neither CUDA. Only
`extract_features.py` is torch, and at ~2 s/clip embedded once, the GPU saves
minutes across the whole dataset.

**If you clone this somewhere new**, the only local state worth carrying is
`hb_labels.csv`. Videos re-download in ~10 min at 7 MB/s, and `hb_bouts.csv`
rebuilds from the descriptions.

## What to expect

**Don't attempt 82-way classification.** The distribution is brutally long-tailed:
`yorikiri` alone is ~32% of all professional bouts, while `tasukizori` went 65
years between occurrences. With ~675 bouts you'll see maybe 15-20 techniques at
all. `train_head.py` folds classes under 15 clips into `other` by default.

**Compare against the majority baseline, not zero.** Always guessing `yorikiri`
scores ~32%. `train_head.py` prints this -- beating it is the actual bar.

**Two confusions are expected and not bugs** (`kimarite.py` flags both when
present in your data):

- *Outcome pairs* -- `oshidashi`/`oshitaoshi`, `yorikiri`/`yoritaoshi`, and 5 more
  are the **same technique**, differing only in whether the opponent finished
  standing or fallen. That evidence lives in the last ~0.5s, which is why `PAD`
  is small and masking does the anti-leak work instead.
- *Grip pairs* -- `uwatenage`/`shitatenage` differ by whether the grip is outside
  or inside the opponent's arm. At 224x224, with two bodies occluding each other,
  this is often physically unresolvable. Consider merging them.

**Non-techniques are excluded.** The 7 *higi* (`koshikudake`, `tsukite`, `fusen`,
...) mean the loser lost unaided -- there is no causal motion to learn. `fusen`
means the opponent never appeared, so no bout was fought at all. This is enforced
in the labelers via `kimarite.is_technique()`, not just documented: the spot check
produced a real `fusen` read, and without the check it would have written 6 s of
an empty dohyo into the training set under a technique label.

**Judge by the grouped CV number**, not the held-out block.

## Extending

- **Add commentary**: Whisper the audio -- announcers frequently *say* the
  technique. Concatenate a text embedding to the video embedding. Likely the
  single largest available gain, though verify it isn't just re-reading the
  answer, same trap as the overlay.
- **More data**: six tournaments a year, ~600 bouts each. Past ~5k bouts, LoRA
  fine-tuning the encoder becomes worthwhile -- but not on 4 GB of VRAM.
- **Swap encoder**: change `MODEL_ID` in `extract_features.py` (V-JEPA 2,
  InternVideo2). Delete `cache/` afterwards -- embeddings are encoder-specific.
