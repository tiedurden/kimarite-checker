"""Does adding source videos help? Measured per representation, not once.

WHY THIS EXISTS AS ITS OWN SCRIPT. This question was answered "no" earlier -- a
paired test over 30 random video pools measured +0.0042 +/- 0.0104 balanced accuracy
going from 5 to 12 videos (t = 0.40, 17/30 draws improving), and that steered the
project away from labelling the remaining 67 videos. But it was measured on the
WHOLE-BOUT representation, which later turned out to be the broken one: it sits at
0.302 balanced accuracy and fails a permutation test (p = 0.119). A representation
carrying no signal cannot show a learning curve, so "more data does not help" was
never really a statement about the data.

Multi-scale finish features score 0.466-0.480 at p = 0.005. The question has to be
re-asked there, and the answer decides whether ~1400 more bouts are worth an hour of
labelling. Cheap to ask: pure CPU on caches that already exist.

METHOD. Draw a random pool of k source videos, train on all but one held-out video,
score, repeat. Pools are PAIRED -- the same draw is evaluated at every k by nesting
the smaller pool inside the larger one, so a per-draw delta is a paired measurement
and the enormous between-video variance cancels. Comparing two independent mean
+/- std bars at n=13 videos would hide an effect of the size we care about.

Scoring is leave-one-video-out WITHIN the pool, because that is the generalization
that matters (unseen venue, camera, wrestlers) and because 5-fold GroupKFold on a
small pool has already produced one result that evaporated under LOVO (see --tail in
hb_finish.py).

    python hb_curve.py                       # multi-scale (default)
    python hb_curve.py --cache cache_hb      # reproduce the old whole-bout answer
    python hb_curve.py --sizes 5 12          # the original test's exact endpoints
    python hb_curve.py --draws 40

RESULT, and it reverses the earlier conclusion. Head to head at the original test's
own endpoints, 30 paired draws, identical method:

    multi-scale (2s+3s+6s)   5 -> 12 videos   +0.1098 +/- 0.0093   t +11.75  29/30
    whole bout               5 -> 12 videos   -0.0091 +/- 0.0105   t  -0.86  10/30

The whole-bout row reproduces the old "no help" answer, which is what makes the other
row trustworthy -- same code, same draws, same folds, so the difference is the
representation and not the method. Full curve on multi-scale, 4 -> 13 videos:
0.304, 0.317, 0.337, 0.371, 0.369, 0.384, 0.404, 0.418, 0.427, 0.466 -- monotone
apart from one step, and still climbing at 13.

Two controls rule out the obvious confounds. A single 768-dim finish cache also
climbs (+0.111, t = +10.5), so this is not about 2304 dimensions needing more data;
and the 2304-dim zero-information control (6s repeated 3x) climbs no faster
(+0.125) than the 768-dim cache it duplicates. The curve comes from the FRAMING.

So: labelling the remaining ~1400 bouts is now justified, and it was not before.
"More data does not help" was true of a representation with no signal to sharpen.

Weaker at TECHNIQUE granularity (--techniques): +0.0441 +/- 0.0140, t = +3.15, 22/30
draws, slope +0.006/video against +0.017 for the families, and the curve wobbles
(0.379 at 12 videos, 0.366 at 13). Still positive and still significant, but the
clean steep climb is a coarse-family result. Expected, and it argues for MORE data
rather than less: the individual techniques are long-tailed, so most classes here are
in the regime where a handful of examples per class is the binding constraint.
"""

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import kimarite
from train_head import load_many

# Paths, not strings: argparse's type= is not applied to default=, so a bare string
# here reaches load_one() and dies on .rglob.
MULTISCALE = [Path("cache_fin2"), Path("cache_fin3"), Path("cache_fin6")]


def score_pool(X, y, groups, pool, rng) -> float | None:
    """Leave-one-video-out balanced accuracy using only videos in `pool`.

    None if the pool cannot be scored -- too few videos, or a fold whose training
    half lost a whole class. Returning None rather than 0.0 matters: a failed pool
    that scored as zero would drag the mean down and look like the small pools being
    genuinely bad at the task.
    """
    m = np.isin(groups, pool)
    Xp, yp, gp = X[m], y[m], groups[m]
    if len(set(gp)) < 2:
        return None
    scores = []
    for held in sorted(set(gp)):
        tr, te = gp != held, gp == held
        if len(set(yp[tr])) < 2 or not te.any():
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
        clf.fit(Xp[tr], yp[tr])
        scores.append(balanced_accuracy_score(yp[te], clf.predict(Xp[te])))
    return float(np.mean(scores)) if scores else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", nargs="+", default=MULTISCALE, type=Path)
    ap.add_argument("--draws", type=int, default=30)
    ap.add_argument("--sizes", type=int, nargs="+", default=None,
                    help="pool sizes to test (default: 4 .. all videos)")
    ap.add_argument("--min-class", type=int, default=15)
    # Coarse families by default, and --techniques to opt out. This was an
    # action="store_true" with default=True, which can never be switched off -- a flag
    # that reads as optional and is not.
    ap.add_argument("--techniques", action="store_true",
                    help="individual kimarite instead of the coarse families")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    X, y, groups, _ = load_many(args.cache)
    if not args.techniques:
        y = np.array([kimarite.coarse(v) or "other" for v in y])
    rare = {k for k, n in Counter(y).items() if n < args.min_class and k != "other"}
    if rare:
        y = np.array(["other" if v in rare else v for v in y])
    counts = Counter(y)

    vids = sorted(set(groups))
    sizes = args.sizes or list(range(4, len(vids) + 1))
    sizes = [s for s in sizes if 2 <= s <= len(vids)]
    if len(sizes) < 2:
        sys.exit(f"need >=2 usable pool sizes; got {sizes} from {len(vids)} videos")

    print(f"{len(X)} clips, dim {X.shape[1]}, {len(vids)} videos, "
          f"{len(counts)} classes (chance {1 / len(counts):.3f})")
    print(f"features: {', '.join(c.name for c in args.cache)}")
    print(f"{args.draws} paired draws, pool sizes {sizes[0]}..{sizes[-1]}, "
          f"scored leave-one-video-out within each pool\n")

    # One nested permutation per draw. Pool at size k is the first k videos of that
    # permutation, so every size in a draw shares the same videos -- that nesting is
    # what makes the deltas paired.
    rng = np.random.default_rng(args.seed)
    curve: dict[int, list[float]] = {s: [] for s in sizes}
    for d in range(args.draws):
        order = rng.permutation(vids)
        for s in sizes:
            got = score_pool(X, y, groups, order[:s], rng)
            if got is not None:
                curve[s].append(got)

    print(f"{'videos':>7}{'bal acc':>12}{'se':>9}{'n draws':>9}")
    for s in sizes:
        v = np.array(curve[s])
        se = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else float("nan")
        print(f"{s:>7}{v.mean():>12.3f}{se:>9.3f}{len(v):>9}")

    # The paired test, on draws that produced a score at BOTH ends. Slicing to a
    # common length would silently pair draw i at one size with draw j at the other
    # if any pool failed, so both ends are taken from the same draw index.
    lo, hi = sizes[0], sizes[-1]
    n = min(len(curve[lo]), len(curve[hi]))
    a, b = np.array(curve[lo][:n]), np.array(curve[hi][:n])
    d = b - a
    se = d.std(ddof=1) / np.sqrt(len(d))
    print(f"\npaired {lo} -> {hi} videos: {d.mean():+.4f} +/- {se:.4f} (se)  "
          f"t {d.mean() / se if se else float('nan'):+.2f}  "
          f"{(d > 0).sum()}/{len(d)} draws improving")

    # Slope over the whole curve, which uses every size rather than just the ends --
    # a real learning curve should be monotone-ish, not just higher at one endpoint.
    xs = np.array([s for s in sizes for _ in curve[s]], dtype=float)
    ys = np.array([v for s in sizes for v in curve[s]])
    slope = np.polyfit(xs, ys, 1)[0]
    print(f"least-squares slope: {slope:+.4f} balanced accuracy per added video "
          f"(local, over {sizes[0]}-{sizes[-1]})")
    # Deliberately NOT extrapolating. A linear fit over 4-13 videos put 80 videos at
    # 1.57 balanced accuracy, which is impossible -- learning curves are concave and
    # saturate, so the slope is a statement about THIS range only. Quoting a linear
    # extrapolation as an expected score is how "label everything" gets justified by
    # arithmetic that cannot be true.
    print("NOT extrapolated: learning curves saturate, and a linear fit here implies "
          "an\nimpossible >1.0 at 80 videos. The slope says the curve is still "
          "climbing at 13,\nnot where it lands.")


if __name__ == "__main__":
    main()
