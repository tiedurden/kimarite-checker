"""Stage 2: train a classifier head on cached embeddings.

Reads cache/<category>/<clip>.npy. Trains in seconds on CPU -- iterate freely.
"""

import argparse
import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit, cross_val_score, GroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import joblib

# Clips cut from one source video are near-duplicates. If they straddle the
# train/test split the score is inflated -- often 30+ points. Everything before
# the first "__" (or "_clip"/"_seg" suffix) is treated as the source video id.
GROUP_RE = re.compile(r"^(.*?)(?:__|_clip\d+|_seg\d+|_part\d+)", re.IGNORECASE)


def group_of(stem: str) -> str:
    """Source-video id for leakage-safe splitting."""
    m = GROUP_RE.match(stem)
    return m.group(1) if m else stem


def load_many(caches: list[Path]):
    """Concatenate several caches of the SAME clips into one feature block.

    Multi-scale features are the single biggest measured win in this project (see
    README): 2s+3s+6s finish embeddings score 0.480 balanced accuracy against 0.302
    for the whole bout, permutation p = 0.005. Each cache holds the same 191 bouts
    embedded at a different clip length, so concatenating them hands the head several
    temporal resolutions of the same finish.

    The clip name lists must match exactly across caches. A mismatch would pair one
    bout's 2s vector with a different bout's 6s vector -- silently, since the shapes
    would still line up -- so it aborts rather than warns.
    """
    per = [load_one(c) for c in caches]
    ref_names = per[0][3]
    for cache, (_, _, _, names) in zip(caches[1:], per[1:]):
        if names != ref_names:
            missing = sorted(set(ref_names) ^ set(names))[:4]
            raise SystemExit(
                f"{caches[0]}/ and {cache}/ hold different clips "
                f"({len(ref_names)} vs {len(names)}); e.g. {missing}.\n"
                f"Concatenating them would pair one bout's features with another's. "
                f"Re-run extract_features.py so both caches cover the same clips.")
    X = np.hstack([p[0] for p in per])
    return X, per[0][1], per[0][2], ref_names


def load_one(cache: Path):
    X, y, groups, names = [], [], [], []
    for npy in sorted(cache.rglob("*.npy")):
        rel = npy.relative_to(cache)
        if len(rel.parts) < 2:
            continue  # needs cache/<category>/<clip>.npy
        X.append(np.load(npy))
        y.append(rel.parts[0])
        # Group on the video id ALONE. Prefixing the category (rel.parts[0]) makes
        # one video look like N groups, one per technique it contains -- so
        # GroupShuffleSplit happily puts oshidashi/VID in train and yorikiri/VID in
        # test, which is the same video on both sides and exactly the leak this
        # grouping exists to stop. Observed as "12 clips into 6 source videos" for
        # 12 clips that all came from a single video.
        groups.append(group_of(npy.stem))
        # Name WITHOUT the category dir, so the cross-cache check below compares
        # bouts rather than bout+label. The label comes along in y either way.
        names.append(npy.stem)
    if not X:
        raise SystemExit(f"no .npy under {cache}/ -- run extract_features.py first")
    return np.stack(X), np.array(y), np.array(groups), names


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="cache", type=Path, nargs="+",
                    help="one cache, or several of the SAME clips at different clip "
                         "lengths -- they are concatenated (multi-scale features)")
    ap.add_argument("--out", default="models/head.joblib", type=Path)
    ap.add_argument("--test-frac", type=float, default=0.25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-class", type=int, default=15,
                    help="classes with fewer clips are folded into 'other'; 0 disables")
    ap.add_argument("--coarse", action="store_true",
                    help="train on the 6 official families instead of techniques")
    ap.add_argument("--scales", type=float, nargs="+",
                    help="finish-clip length (s) each --cache was built from, in the "
                         "same order; recorded in the model so predict.py can rebuild "
                         "the same feature vector")
    args = ap.parse_args()

    if args.scales and len(args.scales) != len(args.cache):
        raise SystemExit(f"--scales has {len(args.scales)} values but --cache has "
                         f"{len(args.cache)}; they must correspond in order")
    if len(args.cache) > 1 and not args.scales:
        # Not fatal for training -- the CV score is valid either way -- but the saved
        # model cannot be used by predict.py, and finding that out later is worse than
        # a warning now.
        print("NOTE: multi-scale training without --scales. The CV numbers are valid, "
              "but predict.py\n      cannot use the saved model: nothing records which "
              "clip length each block came from.")

    X, y, groups, names = load_many(args.cache)
    if len(args.cache) > 1:
        print(f"multi-scale: {len(args.cache)} caches concatenated -> dim {X.shape[1]} "
              f"({', '.join(c.name for c in args.cache)})")

    if args.coarse:
        # 82-way on ~675 bouts is hopeless; 6-way on the official families is a
        # genuinely learnable problem and a useful sanity signal on the pipeline.
        import kimarite
        y = np.array([kimarite.coarse(v) or "other" for v in y])
        print("training on official families (--coarse)")

    # Fold AFTER coarsening, and for both modes. This used to be an `elif`, so
    # --coarse silently ignored --min-class even though the two are documented as
    # independent -- and folding matters MORE here: the families are long-tailed
    # too (kakete and hinerite turned up with 1 clip each), and one singleton class
    # is enough to disable the grouped-CV block entirely, since n_folds is bounded
    # by the smallest class. That silently removed the most trustworthy number the
    # script reports.
    if args.min_class:
        # Long tail: ~5 techniques dominate, dozens appear once. A class with 3
        # examples cannot be learned and its presence only adds label noise.
        c0 = Counter(y)
        rare = {k for k, n in c0.items() if n < args.min_class and k != "other"}
        if rare:
            y = np.array(["other" if v in rare else v for v in y])
            print(f"folded {len(rare)} class(es) with <{args.min_class} clips "
                  f"into 'other': {', '.join(sorted(rare))}")

    counts = Counter(y)
    n_groups = len(set(groups))
    print(f"{len(X)} clips, dim={X.shape[1]}, {len(counts)} categories, "
          f"{n_groups} source videos")
    for cat, n in sorted(counts.items()):
        print(f"  {cat:<28} {n:>4}")

    if len(counts) < 2:
        raise SystemExit("need >=2 categories to train a classifier")

    # A leakage-safe split needs at least two source VIDEOS -- more clips do not
    # help. Caught here with an explanation, because GroupShuffleSplit's own error
    # ("With n_samples=1, test_size=0.25 ... train set will be empty") describes
    # groups as samples and reads like a parameter problem rather than "label more
    # videos".
    if n_groups < 2:
        raise SystemExit(
            f"only {n_groups} source video in "
            f"{', '.join(str(c) for c in args.cache)} -- a leakage-safe split "
            f"needs >=2.\nLabel bouts from more videos (clips from one video are "
            f"near-duplicates: same venue,\nlighting, camera and often wrestlers, so "
            f"a within-video score means nothing).")

    # Loud warnings beat a silently misleading score.
    if n_groups < len(X):
        print(f"\ngrouping {len(X)} clips into {n_groups} source videos "
              f"(leakage-safe split)")
    else:
        print("\nNOTE: every clip looks like its own source video. If clips were "
              "cut from shared sources, name them '<video>__<n>.mp4' so they stay "
              "on one side of the split.")
    if (rare := [c for c, n in counts.items() if n < 10]):
        print(f"WARNING: thin categories {rare} -- per-class metrics will be noisy")

    # Scale first: raw VideoMAE embeddings have wildly uneven feature variance.
    # C=1.0 (not higher) because kimarite classes are severely imbalanced and
    # d >> n -- light regularization would memorize the majority class.
    # balanced weights stop 'yorikiri' (~32% of all bouts) from swallowing
    # everything; without it the model can score well by never predicting
    # anything else.
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced"),
    )

    # The number to beat. With yorikiri at ~32%, an 'always guess the majority'
    # baseline looks deceptively strong -- print it so accuracy is read in context.
    major = counts.most_common(1)[0]
    print(f"majority-class baseline: {major[1] / len(y):.1%} (always '{major[0]}')")

    splitter = GroupShuffleSplit(n_splits=1, test_size=args.test_frac,
                                random_state=args.seed)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    clf.fit(X[train_idx], y[train_idx])
    pred = clf.predict(X[test_idx])

    print(f"\n=== held-out: {len(test_idx)} clips from "
          f"{len(set(groups[test_idx]))} unseen videos ===")
    print(classification_report(y[test_idx], pred, zero_division=0))

    labels = sorted(counts)
    print("confusion (rows=true, cols=pred):")
    print(f"{'':<28}" + "".join(f"{l[:10]:>12}" for l in labels))
    for lab, row in zip(labels, confusion_matrix(y[test_idx], pred, labels=labels)):
        print(f"{lab:<28}" + "".join(f"{v:>12}" for v in row))

    # Grouped CV: one split on a few hundred clips is a high-variance estimate.
    n_folds = min(5, n_groups, min(counts.values()))
    cv_bal = None
    if n_folds >= 2:
        cv = GroupKFold(n_splits=n_folds)
        scores = cross_val_score(clf, X, y, groups=groups, cv=cv)
        print(f"\n{n_folds}-fold grouped CV accuracy: {scores.mean():.3f} "
              f"(+/- {scores.std():.3f})  {np.round(scores, 3)}")
        # Balanced accuracy too, because it is the metric this project is judged on:
        # with class_weight='balanced' the head is deliberately not riding the
        # majority class, so plain accuracy penalizes it for doing what it was told.
        # Reporting only accuracy is what made an above-chance result look like a
        # failure. Read this against 1/n_classes, and accuracy against the majority
        # baseline printed above.
        bal = cross_val_score(clf, X, y, groups=groups, cv=cv,
                              scoring="balanced_accuracy")
        cv_bal = float(bal.mean())
        print(f"{n_folds}-fold grouped CV balanced acc: {bal.mean():.3f} "
              f"(+/- {bal.std():.3f})  {np.round(bal, 3)}   "
              f"[chance {1 / len(counts):.3f}]")

    # Ship a model trained on everything; the scores above are the honest estimate.
    clf.fit(X, y)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # `scales` is what lets predict.py rebuild a multi-scale feature vector: without
    # it a 2304-dim head is unusable, because nothing records that those dims are
    # 2s|3s|6s of the finish rather than one 2304-dim encoder. Stored as None for a
    # single cache so old single-scale bundles keep working.
    joblib.dump({"model": clf, "labels": labels, "dim": int(X.shape[1]),
                 "scales": args.scales, "caches": [str(c) for c in args.cache]},
                args.out)
    args.out.with_suffix(".json").write_text(json.dumps(
        {"labels": labels, "counts": dict(counts), "n_clips": len(X),
         "n_source_videos": n_groups, "dim": int(X.shape[1]),
         "caches": [str(c) for c in args.cache], "scales": args.scales,
         "cv_balanced_accuracy": cv_bal}, indent=2))
    print(f"\nsaved -> {args.out}")


if __name__ == "__main__":
    main()
