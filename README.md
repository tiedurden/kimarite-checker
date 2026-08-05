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

**The caption is not the only leak.** The Herbert broadcast also puts the winner's
shikona on a white plate at the bottom-centre of frame, and it appears *after the
bout ends but before the win caption* -- so a bout window that runs to the finish
catches it in its last seconds. The winner's name is not the technique, but it is
strongly correlated with it, and glyphs are far easier to read than wrestling. It
sat below all three original mask boxes and went unnoticed until a contact sheet of
a new video showed it. Now `LEAK_BOXES_HB` masks the full width below y=0.92.

Generalize from that: **on a new source, list every graphic the broadcast can put on
screen between the charge and the caption**, not just the caption. Check a contact
sheet of the last seconds of a window, where post-bout overlays live.

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

# Best configuration measured at the CURRENT 22 videos / 340 clips: the last 6s of
# the bout, alone. Multi-scale (below) won at 13 videos and lost the lead as the set
# grew -- so re-run hb_curve.py --versus after adding data instead of trusting this.
python train_head.py --cache cache_fin6 --scales 6 \
    --coarse --min-class 25 --out models/head_fin6.joblib

# The former best, kept because it is what most of the tables below were measured on.
python train_head.py --cache cache_fin2 cache_fin3 cache_fin6 --scales 2 3 6 \
    --coarse --min-class 15 --out models/head_multiscale.joblib
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
python hb_finish.py --len 3               # last 3s only -> data_finish/ (see below)
python hb_sweep.py --lovo                 # compare framings; --lovo is not optional
python hb_curve.py                        # does more data help THIS representation?
python hb_curve.py --versus cache_fin6    # paired A/B at every pool size
python hb_batch.py --dry-run              # plan the next download/label/embed batch
python hb_batch.py --batches 3            # run three (~20 min each, sequential)
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

**Labelled so far: the full 14-video pilot, 191 clips.** Of 270 manifest bouts: 25
OCR misses, 20 with no locatable bout window, 6 non-techniques correctly rejected
(`fusen` x4, `tsukite`, `isamiashi`). 68 videos / 1377 bouts remain in the manifest.

### The result: tripling the data changed nothing

4-fold grouped CV, 4 official families, `--coarse --min-class 15`:

| dataset | accuracy | balanced acc | f1_macro | perm p |
|---|---|---|---|---|
| 60 clips / 4 videos | 0.467 +/- 0.115 | 0.301 +/- 0.053 | 0.260 | 0.120 |
| **191 clips / 13 videos** | 0.440 +/- 0.084 | **0.302** +/- 0.033 | 0.267 | **0.305** |

3.2x the clips and 3.3x the source videos moved balanced accuracy by **+0.001**.
Accuracy loses to the 0.597 majority baseline in both. Read `balanced_accuracy`
against chance, not `accuracy` against the majority baseline -- the head uses
`class_weight="balanced"`, so plain accuracy penalizes it for not riding the
majority class.

The decisive number is the permutation test, which got *worse*: p 0.120 -> 0.305
(z = +0.34 on 6867 cross-video same-technique pairs). A real effect starved of data
sharpens when the data arrives; this collapsed toward the null. **The earlier
p = 0.120 was noise.**

**That conclusion has since been overturned, and only ever applied to this
representation.** Re-measured on multi-scale finish features, adding videos helps a
great deal -- see "Does more data help?" below. Reading the whole-bout result as a
general claim about the project is how a fixable framing bug gets mistaken for a
ceiling, which is exactly what happened here for a while.

So, with the input verified 100% bout footage and leaks masked, this is a statement
about the framing: **frozen VideoMAE embeddings of WHOLE BOUT clips do not linearly
encode kimarite.** The word "whole" turned out to be carrying the whole finding --
see below.

Most likely why: VideoMAE samples 16 frames evenly across whatever it is given, so
an 8-30s clip is sampled every ~1.25s while the technique is decided in the final
half-second. `oshidashi` vs `oshitaoshi` differ only in whether the loser ended
standing or fallen. Fixing the clip window made clips longer and more complete,
which may have diluted the decisive moment further.

### What did work: hand the encoder the finish, not the bout

`hb_finish.py` re-cuts the last N seconds from the stored `bout_start`/`bout_end`
(no OCR, no window search, ~1s/bout) into its own dataset directory. **Same 191
clips, same labels, same folds -- only the framing changes.** 4 official families,
`--min-class 15`, 5-fold grouped CV:

| window | accuracy | balanced acc | f1_macro | paired delta bal.acc | t |
|---|---|---|---|---|---|
| whole bout (8-30s) | 0.440 +/- 0.084 | 0.302 +/- 0.033 | 0.267 | -- | -- |
| 2s | 0.544 +/- 0.070 | 0.379 +/- 0.048 | 0.362 | **+0.077** | **+2.50** |
| 3s | **0.572** +/- 0.106 | 0.399 +/- 0.080 | 0.379 | +0.097 | +1.93 |
| 4s | 0.531 +/- 0.067 | 0.383 +/- 0.064 | 0.366 | +0.081 | +1.68 |
| 6s | 0.520 +/- 0.097 | **0.422** +/- 0.105 | **0.382** | +0.120 | +1.74 |

Chance 0.250; majority baseline 0.597. The deltas are **paired per-fold** against
the whole-bout run on identical splits -- the two datasets share bout ids, so that
is far more sensitive than comparing two error bars at 5 folds. 4/5 folds improve
for every window length.

For scale, against the other lever: 3.3x more source videos bought +0.004
(t = 0.40); reframing the *same* clips bought +0.077 to +0.120. **Temporal
resolution was the bottleneck, not sample size.**

**The window length is not resolved and don't claim it is.** Pairwise paired t
between 2/3/4/6s: all |t| <= 1.72. 6s has both the largest mean gain and a lower
t than 2s -- that is one fact, not two: its variance is triple. What the data
supports is "anything but the whole bout", not an optimum. Also note no window
beats the 0.597 majority baseline on plain accuracy yet; the model is now clearly
better than chance and still not better than always guessing `kihonwaza`.

### Multi-scale features: the best result at 13 videos, since overtaken

> **Superseded at 22 videos.** Everything in this section was measured on 191 clips
> from 13 videos and reproduces exactly. It no longer describes the best
> configuration: plain 6s overtook multi-scale as the set grew -- see
> [Which framing wins depends on how much data you have](#which-framing-wins-depends-on-how-much-data-you-have).
> Kept in full because the controls below are still the reason to believe multi-scale
> was doing something real, and because a superseded measurement is not a wrong one.

Concatenating the *same* finish at several lengths beats any single length. This is
free to test -- the caches are aligned clip-for-clip, so it is a `numpy.hstack`, not
another encoder pass. `hb_sweep.py` runs the whole table:

| features | dim | accuracy | balanced acc | f1_macro | paired delta vs whole | t |
|---|---|---|---|---|---|---|
| whole bout | 768 | 0.440 | 0.302 | 0.267 | -- | -- |
| 6s | 768 | 0.520 | 0.422 | 0.382 | +0.120 | +1.74 |
| whole+2s | 1536 | 0.526 | 0.319 | 0.283 | +0.016 | +0.47 |
| whole+6s | 1536 | 0.492 | 0.341 | 0.336 | +0.038 | +1.05 |
| 2s+6s | 1536 | 0.591 | 0.460 | 0.432 | +0.158 | +2.85 |
| **2s+3s+6s** | **2304** | **0.604** | **0.480** | **0.459** | **+0.178** | **+4.19, 5/5 folds** |
| 2s+3s+4s+6s | 3072 | 0.584 | 0.463 | 0.440 | +0.160 | +3.52 |
| 6s x3 (control) | 2304 | 0.525 | 0.424 | 0.384 | +0.122 | +1.78 |
| 2s x3 (control) | 2304 | 0.554 | 0.387 | 0.372 | +0.084 | +2.38 |

**The controls are the point.** A 3x wider feature block changes the regularization
budget on its own at n=191, so "wider won" is not evidence that multi-scale works.
Repeating ONE cache three times holds dim fixed at 2304 and adds exactly zero
information: `6s x3` gains **+0.002** over 6s alone, while `2s+3s+6s` gains +0.058.
The win is temporal content, not width. Keep those rows in any future sweep.

Note also that adding the **whole bout** to a finish clip barely helps (+0.016,
+0.038) while adding a *second finish length* helps a lot. The whole-bout embedding
is not contributing context; it is contributing noise.

**The permutation test finally clears.** Same 200-permutation within-video test that
showed nothing before:

| features | balanced acc | null | p | z |
|---|---|---|---|---|
| whole bout | 0.302 | 0.251 +/- 0.037 | 0.119 | +1.37 |
| **2s+3s+6s** | **0.480** | 0.253 +/- 0.034 | **0.005** | **+6.75** |

p = 0.005 is the floor at 200 permutations -- zero shuffles beat it.

**It holds on fine-grained techniques too**, not just the 4 coarse families
(`--min-class 15`, 4 technique classes, chance 0.250, majority 0.372):

| features | accuracy | balanced acc | f1_macro |
|---|---|---|---|
| whole bout | 0.302 | 0.284 | 0.243 |
| 6s | 0.356 | 0.319 | 0.309 |
| 2s+3s+6s | **0.392** | **0.367** | **0.348** |

That 0.392 is the **first configuration to beat the majority baseline** (0.372); the
whole-bout model lost to it badly. On the coarse families accuracy 0.604 still only
just edges the 0.597 baseline, so treat that as parity, not a win.

Train and use it:

```bash
python hb_finish.py --len 2 --data data_fin2      # and 3, 6
python extract_features.py --data data_fin2 --cache cache_fin2
python train_head.py --cache cache_fin2 cache_fin3 cache_fin6 --scales 2 3 6 \
    --coarse --min-class 15 --out models/head_multiscale.joblib
python predict.py bout.mp4 --model models/head_multiscale.joblib
```

`--scales` is not decoration: it records which clip length each 768-dim block came
from, so `predict.py` can cut and embed the same windows from a new video. Without
it the head trains fine and is then unusable, so `train_head.py` warns. Verified
bit-exact: `predict.py`'s reconstructed vector matches the cached training vector at
max|diff| 0.0. **The input must END at the finish** -- hand it a whole broadcast and
every window lands in the ceremony.

**Validate under leave-one-video-out before believing a best-of-N number.**
`hb_sweep.py --lovo` scores 13 deterministic folds instead of 5 random ones. This is
not optional hygiene, it caught a real mistake: `--tail 1` (trim the walk-off off
`bout_end` before cutting the finish) measured **+0.052 balanced accuracy at
t = +3.99, 5/5 folds** on GroupKFold, with a credible non-monotone shape (`--tail 2`
lost 0.068, i.e. 1s removes aftermath and 2s eats the technique). It was reported as
a win. Under LOVO the trimmed multi-scale config scores **0.451 vs 0.466 untrimmed,
t = -0.35, 4/13 folds up** -- the 5-fold result was fold-assignment luck, surfaced by
scoring 18 configurations and taking the maximum. `--tail` stays at 0.

The overshoot itself is real (a contact sheet showed one 4s clip, `3j4r7V1bzlQ#2`,
holding only aftermath), so the flag stays for a source with longer walk-offs. It
just does not pay here. `hb_sweep.py` refuses to print a table without reminding you
to re-run with `--lovo`.

Both evaluations agree on what matters -- every finish variant beats the whole bout
(LOVO t = +2.07 to +4.74) and multi-scale tops both tables:

| config | 5-fold bal acc | LOVO bal acc (13 folds) |
|---|---|---|
| whole bout | 0.302 | 0.270 |
| 6s alone | 0.422 | 0.420 |
| 2s+3s+6s | **0.480** | **0.466** |
| 2s+3s+4s+6s | 0.463 | **0.478** |
| tail-trimmed variants | 0.423-0.518 | 0.401-0.451 |

`2s+3s+6s` and `2s+3s+4s+6s` swap places between the two schemes, which is the
honest read: they are indistinguishable, and neither is "the" best.

### Does more data help? On these features, yes -- a lot for the families

`hb_curve.py`. Draw a random pool of *k* source videos, score leave-one-video-out
within the pool, repeat. Pools are **nested**, so the same draw is evaluated at every
*k* and the per-draw delta is paired -- between-video variance is enormous and
cancels this way.

Head to head at the endpoints of the original whole-bout test, 30 paired draws,
identical code and draws:

| features | 5 -> 12 videos | t | draws improving |
|---|---|---|---|
| whole bout | **-0.009** +/- 0.011 | -0.86 | 10/30 |
| **2s+3s+6s** | **+0.110** +/- 0.009 | **+11.75** | **29/30** |

The whole-bout row reproduces the earlier "no help" answer, and that is what makes
the other row believable: same method, same draws, so the difference is the
representation. Full multi-scale curve, 4 -> 13 videos:

| videos | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced acc | 0.304 | 0.317 | 0.337 | 0.371 | 0.369 | 0.384 | 0.404 | 0.418 | 0.427 | 0.466 |

Monotone apart from one step, **still climbing at 13 videos**, local slope +0.017 per
video. Do NOT extrapolate that: a linear fit puts 80 videos at 1.57 balanced
accuracy, which is impossible. Learning curves saturate; the slope says the curve has
not turned over yet, not where it lands. `hb_curve.py` prints that warning instead of
a projected number, on purpose.

Two controls rule out the obvious confound -- that 2304 dimensions simply need more
data to fill:

- a single **768-dim** finish cache also climbs steeply (+0.111, t = +10.5)
- the **2304-dim zero-information** control (6s repeated 3x) climbs no faster
  (+0.125) than the 768-dim cache it duplicates

So the curve comes from the framing, not the width. **Labelling the remaining bouts
is now justified, and it was not before.**

**Weaker at technique granularity**, and the table above is a coarse-family result, so
read it as one. `hb_curve.py --techniques`, same 30 draws:

| videos | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 |
|---|---|---|---|---|---|---|---|---|---|---|
| balanced acc | 0.322 | 0.330 | 0.339 | 0.345 | 0.355 | 0.370 | 0.361 | 0.374 | 0.379 | 0.366 |

Paired 4 -> 13: **+0.044** +/- 0.014, t = +3.15, 22/30 draws improving, slope +0.006
per video -- a third of the coarse slope, and it wobbles at the top (0.379 at 12,
0.366 at 13). Positive and significant, not the same clean climb. That is the
expected shape: the individual kimarite are long-tailed, so at 13 videos most classes
have a handful of examples and get folded into `other`. It argues for more data rather
than against it, but the honest claim is "families climb steeply, techniques climb
slowly".

### Which framing wins depends on how much data you have

The curve said labelling would pay, so `hb_batch.py` took the set from **13 videos /
191 clips to 22 / 340**. It did pay -- and it changed the answer to a question that
looked settled.

Paired on identical video pools, `--min-class 25` so the task stays the same 4
families (chance 0.250) rather than gaining a fifth as `hinerite` crosses the
threshold -- balanced accuracy is not comparable across a different class count:

```bash
python hb_curve.py --versus cache_fin6 --min-class 25 --sizes 6 10 14 18 22
```

| videos | 2s+3s+6s | 6s alone | delta | t |
|---|---|---|---|---|
| 6 | 0.324 | 0.326 | -0.002 | -0.18 |
| 10 | 0.356 | 0.362 | -0.006 | -0.63 |
| 14 | 0.367 | 0.377 | -0.010 | -1.86 |
| 18 | 0.377 | 0.399 | **-0.022** | **-3.79** |
| 22 | 0.382 | **0.427** | **-0.045** | exact* |

<sub>*at 22 the pool is every video, so there is one deterministic answer and no
spread -- `hb_curve.py` prints `exact` rather than dividing by ~0.</sub>

**A crossover**, which is why the comparison prints every size and not just the
endpoints. The two are tied to ~10 videos, then 6s pulls away. Multi-scale flattens
after 16 (0.377, 0.377, 0.382) while 6s keeps climbing -- slope **+0.0067/video vs
+0.0041**. The reading: 2304 dimensions at 191 clips bought regularization the head
no longer needs at 340, and the 2s/3s blocks are largely redundant with the 6s one.

What this does *not* overturn, because it is easy to over-correct here:

- **finish-vs-whole-bout still holds** -- 6s beats the whole bout by +0.107,
  t = +2.26 on LOVO at 22 videos
- **more data still helps** -- the curve is what says so: 6s goes 0.284 (4 videos) ->
  0.427 (22), paired **+0.143, t = +12.11, 30/30 draws**
- the multi-scale controls were still sound; multi-scale really did win at n=191

What changed is *which* framing wins, i.e. a decision to be **re-made at each data
size** rather than settled once. Treat every table above as a measurement at its own
n, and re-run `hb_curve.py --versus` after each batch.

Next experiments, cheapest first:

1. **Keep labelling** -- 59 videos / ~1240 bouts left, `python hb_batch.py`. Two
   batches done (17 min and 24 min); both curves are still climbing at 22 videos, so
   this is still the highest-value work. Re-run `hb_curve.py --versus cache_fin6`
   after each batch: the winning configuration has already changed once.
2. **Swap the encoder** (`MODEL_ID`, V-JEPA 2 / InternVideo2). The finish result says
   the *framing* was wrong; it does not say VideoMAE-base is sufficient. Now worth
   doing, because there is finally a signal to improve on rather than noise. Costs a
   re-embed of every cache (~2 s/clip) and nothing else -- `train_head.py` is
   unchanged.
3. **Audio via Whisper.** Announcers frequently *say* the technique. Likely the
   single largest available gain, and orthogonal to everything above. Verify it is
   not just re-reading the answer, same trap as the overlay -- the honest test is
   whether the *video* head improves, not the fused score.

Not worth repeating: `--tail` (above), and whole-bout+finish fusion (+0.016 to
+0.038; the whole-bout block contributes noise, not context).

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
