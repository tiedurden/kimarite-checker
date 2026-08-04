"""Score every finish-window length against the whole bout, on IDENTICAL folds.

Not a permanent script -- an experiment harness. The point is the PAIRED comparison:
all caches hold the same 191 clips with the same names and labels, so the same
GroupKFold assignment can be reused across them and the per-fold difference is a
paired measurement. Comparing two independent mean +/- std bars at 5 folds would
hide an effect this size.

Aborts if the clip name lists differ between caches, because a silent mismatch
would pair fold i of one dataset against different bouts in another and the deltas
would be meaningless.

Result on 191 clips / 13 videos (see README): every finish window beats the whole
bout by +0.077 to +0.120 balanced accuracy, 4/5 folds up in all four cases. The
lengths do NOT separate from each other -- pairwise |t| <= 1.72 -- so read this as
"not the whole bout" rather than as an optimum at any particular length.

    python hb_finish.py --len 2 --data data_fin2
    python extract_features.py --data data_fin2 --cache cache_fin2
    python hb_sweep.py
"""

import sys
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import kimarite
from train_head import group_of

CACHES = [("whole bout", "cache_hb"), ("2s", "cache_fin2"), ("3s", "cache_fin3"),
          ("4s", "cache_finish"), ("6s", "cache_fin6")]

# Feature-level fusion, free to test: the caches are aligned clip-for-clip, so
# concatenating two of them is a numpy call, not another encoder pass. Two distinct
# hypotheses in here, worth keeping apart:
#   * whole+finish -- context PLUS resolution. If the whole bout carries anything the
#     finish lacks (the grip established at the charge, the approach), this recovers
#     it. If the whole-bout half is pure noise, this should LOSE to finish alone,
#     because 768 useless dimensions at n=191 cost real regularization budget.
#   * finish+finish at two lengths -- multi-scale on the same evidence, no context
#     claim. Tests whether the sampling rate itself is the limit.
COMBOS = [("whole+6s", ["cache_hb", "cache_fin6"]),
          ("whole+2s", ["cache_hb", "cache_fin2"]),
          ("2s+6s", ["cache_fin2", "cache_fin6"]),
          ("2s+3s+6s", ["cache_fin2", "cache_fin3", "cache_fin6"]),
          ("2s+3s+4s+6s", ["cache_fin2", "cache_fin3", "cache_finish",
                           "cache_fin6"]),
          # CONTROLS, and they are load-bearing. A 3x wider feature block changes the
          # regularization budget on its own at n=191, so "wider won and therefore
          # multi-scale works" is not a valid inference without them. Repeating ONE
          # cache three times holds dimensionality fixed at 2304 and adds exactly zero
          # information. Measured: 6s x3 gains +0.002 over 6s alone while 2s+3s+6s
          # gains +0.058 -- so the win is multi-scale content, not width. Keep these
          # rows in any future sweep that adds a wider combo.
          ("6s x3 CONTROL", ["cache_fin6"] * 3),
          ("2s x3 CONTROL", ["cache_fin2"] * 3)]

MIN_CLASS = 15
N_FOLDS = 5


def load(cache: Path):
    X, y, groups, names = [], [], [], []
    for npy in sorted(cache.rglob("*.npy")):
        rel = npy.relative_to(cache)
        if len(rel.parts) < 2:
            continue
        X.append(np.load(npy))
        y.append(kimarite.coarse(rel.parts[0]) or "other")
        groups.append(group_of(npy.stem))
        names.append(npy.stem)
    return np.stack(X), np.array(y), np.array(groups), names


def fold_scores(X, y, groups, folds):
    """accuracy / balanced accuracy / f1_macro per fold, in fold order."""
    out = []
    for tr, te in folds:
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"))
        clf.fit(X[tr], y[tr])
        p = clf.predict(X[te])
        out.append((accuracy_score(y[te], p),
                    balanced_accuracy_score(y[te], p),
                    f1_score(y[te], p, average="macro", zero_division=0)))
    return np.array(out)


def main() -> None:
    loaded = []
    for label, cache in CACHES:
        d = Path(cache)
        if not d.exists():
            print(f"skip {label}: {cache}/ missing")
            continue
        loaded.append((label, cache, *load(d)))
    if len(loaded) < 2:
        sys.exit("need >=2 caches to compare")

    ref_names = loaded[0][5]
    for label, _, _, _, _, names in loaded[1:]:
        if names != ref_names:
            only_ref = sorted(set(ref_names) - set(names))[:3]
            only_this = sorted(set(names) - set(ref_names))[:3]
            sys.exit(f"clip sets differ: {loaded[0][0]} has {len(ref_names)}, "
                     f"{label} has {len(names)}.\n  only in {loaded[0][0]}: "
                     f"{only_ref}\n  only in {label}: {only_this}\n"
                     f"Paired deltas would compare different bouts -- aborting.")

    # Fold the long tail exactly as train_head does, then build the folds ONCE and
    # reuse them for every cache. Labels and groups are identical across caches
    # (verified above), so one fold assignment is valid for all of them.
    _, _, y, groups, _ = loaded[0][1:]
    c0 = Counter(y)
    rare = {k for k, n in c0.items() if n < MIN_CLASS and k != "other"}
    if rare:
        y = np.array(["other" if v in rare else v for v in y])
    counts = Counter(y)
    folds = list(GroupKFold(n_splits=N_FOLDS).split(loaded[0][2], y, groups))

    major = counts.most_common(1)[0]
    print(f"{len(y)} clips, {len(set(groups))} videos, {len(counts)} families "
          f"{dict(counts)}")
    print(f"chance (balanced) = {1/len(counts):.3f}   "
          f"majority baseline (accuracy) = {major[1]/len(y):.3f} '{major[0]}'\n")

    results = {}
    for label, _, X, _, _, _ in loaded:
        results[label] = fold_scores(X, y, groups, folds)

    # Concatenations reuse the SAME already-loaded arrays and the SAME folds, so they
    # cost no encoder time and stay paired with the single-cache rows above. Keyed off
    # the cache name carried through load, NOT off zip(CACHES, loaded) -- a skipped
    # cache would shift that zip and pair a label with the wrong array, which is the
    # same class of silent mispairing the name check above exists to prevent.
    by_cache = {cache: X for _, cache, X, *_ in loaded}
    dims = {}
    for label, parts in COMBOS:
        if any(c not in by_cache for c in parts):
            continue
        Xc = np.hstack([by_cache[c] for c in parts])
        dims[label] = Xc.shape[1]
        results[label] = fold_scores(Xc, y, groups, folds)

    print(f"{'features':<12}{'dim':>6}{'accuracy':>18}{'balanced acc':>18}"
          f"{'f1_macro':>18}")
    for label in results:
        s = results[label]
        cells = "".join(f"{s[:, k].mean():>11.3f} +/-{s[:, k].std():>5.3f}"
                        for k in range(3))
        print(f"{label:<12}{dims.get(label, 768):>6}{cells}")

    ref_label = loaded[0][0]
    ref = results[ref_label]
    print(f"\npaired per-fold balanced-accuracy delta vs {ref_label} "
          f"(same folds, same labels):")
    for label in list(results)[1:]:
        d = results[label][:, 1] - ref[:, 1]
        se = d.std(ddof=1) / np.sqrt(len(d))
        t = d.mean() / se if se else float("nan")
        print(f"  {label:<10} {d.mean():+.4f}  se {se:.4f}  t {t:+.2f}  "
              f"{(d > 0).sum()}/{len(d)} folds up   {np.round(d, 3)}")


if __name__ == "__main__":
    main()
