"""Cut the FINISH of each bout, not the whole bout. Experiment, not a replacement.

WHY. The whole-bout dataset scores at ~0.30 balanced accuracy against 0.250 chance
and its cross-video technique signal is indistinguishable from noise (permutation
p = 0.305 on 191 clips / 13 videos). Tripling the data did not move it: a paired
test over 30 random video pools measured +0.0042 +/- 0.0104 balanced accuracy going
from 5 to 12 videos (t = 0.40, 17/30 draws improving). So the bottleneck is not
sample size.

The leading hypothesis is TEMPORAL DILUTION. VideoMAE samples 16 frames evenly
across whatever it is handed, so an 8-30s clip is sampled every ~1.25s, while a
kimarite is decided in the last fraction of a second. Two measurements support it:

  * oshidashi vs oshitaoshi -- the same push, differing ONLY in whether the loser
    ended standing or fallen -- scores 0.474 against 0.500 chance. Below chance.
  * throws vs push/force-outs, the most visually distinct contrast in the sport,
    scores 0.558 against 0.500. If the representation cannot see that, it is not
    resolving body configuration at all.

So hand the encoder only the decisive moment: 16 frames over FINISH_LEN seconds is
one every ~0.25s instead of one every 1.25s, a 5x increase in temporal resolution
where the evidence actually lives.

RESULT: it works. Paired per-fold against the whole bout on identical folds, all
of 2/3/4/6s improve balanced accuracy by +0.077 to +0.120 with 4/5 folds up (2s
t = +2.50, the rest t = 1.7-1.9). The lengths do not separate from each other
(pairwise |t| <= 1.72), so this is evidence for "not the whole bout", not for any
particular length. Run hb_sweep.py to reproduce.

--tail DOES NOT HELP, and the way it looked like it did is worth keeping on record.
bout_end really does overshoot into the walk-off, so trimming 1s first is a
reasonable idea, and on 5-fold GroupKFold it measured +0.052 balanced accuracy on
the 2s clip at t = +3.99, 5/5 folds -- with a credible non-monotone shape (tail=2
lost 0.068, i.e. 1s removes aftermath, 2s eats the technique). Convincing.

It did not survive leave-one-video-out. Over 13 deterministic folds the trimmed
multi-scale config scores 0.451 against 0.466 for the untrimmed one, t = -0.35,
4/13 folds up. The 5-fold win was fold-assignment luck, found while scoring 18
configurations and reporting the maximum -- selection on noise.

So: leave --tail at 0. The flag stays because the overshoot is real and a different
source with longer walk-offs may need it, but on THIS data it buys nothing. General
lesson, and the third time this project has hit it: validate a promising number
under a different split before believing it, especially one picked as the best of
many. Multi-scale (below) survived that check; this did not.

Best measured configuration: 2s + 3s + 6s untrimmed, 0.480 balanced accuracy on
5-fold / 0.466 on leave-one-video-out, permutation p = 0.005 z = +6.75.

This writes a SEPARATE dataset directory rather than overwriting data_hb/. Three
reasons: the whole-bout numbers stay reproducible for comparison, a null result
costs nothing to walk back, and experiment #2 (concatenating whole-bout and
finish-only vectors into one 1536-dim feature) needs both to exist at once.

    python hb_finish.py --dry-run          # report spans, touch nothing
    python hb_finish.py                    # -> data_finish/<kimarite>/*.mp4
    python hb_finish.py --len 6            # sweep the window length
    python extract_features.py --data data_finish --cache cache_finish
    python train_head.py --cache cache_finish --coarse --min-class 15

Reads bout_start/bout_end straight from hb_labels.csv, so there is no OCR and no
window detection -- about 1s/bout. The labels are already validated; only the
framing changes.
"""

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

from hb_label import mask_filter, probe_dims
from hb_window import cut_clip

# 4.0s at 16 frames = one frame per 0.25s. Short enough to concentrate on the
# finish, long enough to contain the throw or the final drive rather than just the
# aftermath -- a 1s window would often hold only a wrestler already on the clay,
# which is the outcome and not the technique that produced it.
FINISH_LEN = 4.0

# Bouts shorter than this have no meaningful "finish" to isolate: the whole bout IS
# the finish, so the clip is taken as-is rather than skipped. Dropping them would
# quietly bias the set against fast techniques (hatakikomi, tsukiotoshi), which are
# exactly the ones this experiment is trying to resolve.
MIN_BOUT_FOR_TRIM = FINISH_LEN + 1.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="hb_labels.csv", type=Path)
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--data", default="data_finish", type=Path,
                    help="output dir; deliberately NOT data_hb")
    ap.add_argument("--len", type=float, default=FINISH_LEN,
                    help=f"seconds of finish to keep (default {FINISH_LEN})")
    ap.add_argument("--tail", type=float, default=0.0,
                    help="drop this many seconds from the very end (walk-off guard); "
                         "measured as NO help on this source -- see the module "
                         "docstring before using it")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    if not args.labels.exists():
        sys.exit(f"{args.labels} not found -- run hb_label.py first")
    rows = [r for r in csv.DictReader(args.labels.open(encoding="utf-8"))
            if r.get("bout_start") and r.get("bout_end") and r.get("clip")]
    if not rows:
        sys.exit(f"no rows in {args.labels} have bout_start/bout_end -- "
                 f"re-run hb_label.py or hb_recut.py first")

    print(f"{len(rows)} bouts with a located window; keeping the last "
          f"{args.len:.1f}s of each\n")

    dims: dict[str, tuple[int, int]] = {}
    cut, failed, whole = 0, 0, 0
    spans: list[float] = []
    per_class: Counter = Counter()

    for i, r in enumerate(rows, 1):
        vids = list(args.raw.glob(f"*{r['video']}*.mp4"))
        if not vids:
            print(f"[{i}/{len(rows)}] {r['video']}#{r['n']}  SKIP -- not downloaded")
            continue
        path = vids[0]
        if r["video"] not in dims:
            dims[r["video"]] = probe_dims(path)
        w, h = dims[r["video"]]

        t0, t1 = float(r["bout_start"]), float(r["bout_end"])
        end = t1 - args.tail
        if end - t0 < MIN_BOUT_FOR_TRIM:
            start = t0          # short bout: the whole thing already IS the finish
            whole += 1
        else:
            start = end - args.len
        spans.append(end - start)
        per_class[r["kimarite"]] += 1

        if args.dry_run:
            cut += 1
            continue

        dest = args.data / r["kimarite"] / f"{r['video']}__{int(r['n']):02d}.mp4"
        if cut_clip(path, dest, start, end, mask_filter(w, h)):
            cut += 1
        else:
            failed += 1
            print(f"      ffmpeg failed on {dest}")
        if i % 25 == 0:
            print(f"  [{i}/{len(rows)}] {cut} cut")

    verb = "would cut" if args.dry_run else "cut"
    print(f"\n{verb} {cut}, {failed} ffmpeg failures")
    if spans:
        s = sorted(spans)
        print(f"span: min {s[0]:.1f}s  median {s[len(s)//2]:.1f}s  max {s[-1]:.1f}s")
    print(f"{whole} bout(s) were shorter than {MIN_BOUT_FOR_TRIM:.0f}s and kept whole")
    print(f"{len(per_class)} techniques: " +
          ", ".join(f"{k} {n}" for k, n in per_class.most_common(6)))

    if args.dry_run:
        print("\ndry run -- nothing written.")
        return
    print(f"\nnext:\n"
          f"  python extract_features.py --data {args.data} "
          f"--cache cache_{args.data.name.replace('data_', '')}\n"
          f"  python train_head.py --cache cache_{args.data.name.replace('data_', '')} "
          f"--coarse --min-class 15")


if __name__ == "__main__":
    main()
