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

## Setup

```bash
cd ~/video-categorizer
py -3.12 -m venv .venv
source .venv/Scripts/activate      # Git Bash;  .venv\Scripts\activate on cmd

# torch from the CUDA index -- plain `pip install torch` is CPU-only.
pip install torch --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements.txt

python -c "import torch; print(torch.cuda.is_available())"   # expect True
```

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
means the opponent never appeared, so no bout was fought at all.

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
