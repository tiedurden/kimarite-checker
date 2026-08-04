"""Stage 3: categorize a new video (local file or YouTube URL).

    python predict.py clip.mp4
    python predict.py https://youtu.be/XXXXXXXXXXX
    python predict.py bout.mp4 --model models/head_multiscale.joblib

IMPORTANT for the multi-scale model (the best one, see README): it was trained on
the LAST N seconds of a bout at several N, so the input must END at the finish. Hand
it a whole broadcast and every window lands in the post-bout ceremony. The model
records its own scales, so this is automatic -- but the input framing is not.
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch

from extract_features import clip_duration, decode_frames, embed, load_encoder
from fetch_playlist import FORMAT


def resolve(target: str, tmp: Path) -> Path | None:
    """Return a local path, downloading first if `target` is a URL."""
    if not target.startswith(("http://", "https://")):
        p = Path(target)
        return p if p.exists() else None

    dest = tmp / "clip.mp4"
    print(f"downloading {target} ...")
    r = subprocess.run(
        ["yt-dlp", "--no-playlist", "-f", FORMAT, "--merge-output-format", "mp4",
         "-o", str(dest), target],
        capture_output=True, text=True)
    if r.returncode != 0 or not dest.exists():
        tail = (r.stderr or r.stdout).strip().splitlines()
        print(f"download failed: {tail[-1] if tail else '?'}", file=sys.stderr)
        return None
    return dest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("target", help="video file path or YouTube URL")
    ap.add_argument("--model", default="models/head.joblib", type=Path)
    ap.add_argument("--topk", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if not args.model.exists():
        sys.exit(f"{args.model} not found -- run train_head.py first")
    bundle = joblib.load(args.model)

    with tempfile.TemporaryDirectory() as td:
        path = resolve(args.target, Path(td))
        if path is None:
            sys.exit(f"could not read {args.target}")

        # Same loader and same pooling as extract_features.py used to build the
        # cache the head was trained on -- see load_encoder() for why that matters.
        processor, model, _ = load_encoder(args.device)

        scales = bundle.get("scales")
        if scales:
            # Multi-scale head: it was trained on the last N seconds of the bout at
            # several N, concatenated. Reproduce that here or the vector lands in a
            # different space and the head returns confident nonsense. The input is
            # assumed to END at the finish, which is what the training clips did.
            dur = clip_duration(path)
            if dur is None:
                sys.exit(f"could not read the duration of {path}, which a "
                         f"multi-scale model needs to cut its windows")
            embs = []
            for s in scales:
                t0 = max(0.0, dur - s)
                sub = decode_frames(path, t0=t0, t1=dur)
                if sub is None:
                    sys.exit(f"could not decode the last {s}s of {path}")
                embs.append(embed(sub, processor, model, args.device))
                print(f"  embedded last {s:g}s"
                      + ("  (clip is shorter -- used all of it)"
                         if dur < s else ""))
            emb = np.concatenate(embs)
        else:
            frames = decode_frames(path)
            if frames is None:
                sys.exit(f"could not decode {path}")
            emb = embed(frames, processor, model, args.device)

    if emb.shape[-1] != bundle["dim"]:
        sys.exit(f"embedding dim {emb.shape[-1]} != model's {bundle['dim']}")

    clf = bundle["model"]
    probs = clf.predict_proba(emb[None])[0]
    order = np.argsort(probs)[::-1][: args.topk]

    print()
    for rank, i in enumerate(order, 1):
        bar = "#" * int(probs[i] * 30)
        print(f"{rank}. {clf.classes_[i]:<28} {probs[i]:6.1%}  {bar}")

    # Frozen-encoder heads are poorly calibrated on small data -- treat a low
    # margin as "needs a human", not as a confident second choice.
    if len(order) > 1 and probs[order[0]] - probs[order[1]] < 0.15:
        print("\n  low margin -- ambiguous, worth reviewing by hand")


if __name__ == "__main__":
    main()
