"""Stage 3: categorize a new video (local file or YouTube URL).

    python predict.py clip.mp4
    python predict.py https://youtu.be/XXXXXXXXXXX
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

from extract_features import MODEL_ID, decode_frames
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

        frames = decode_frames(path)
        if frames is None:
            sys.exit(f"could not decode {path}")

        processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
        model = AutoModel.from_pretrained(
            MODEL_ID,
            torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
        ).to(args.device).eval()

        inputs = processor(list(frames), return_tensors="pt").to(args.device)
        if args.device == "cuda":
            inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                      for k, v in inputs.items()}
        with torch.inference_mode():
            emb = model(**inputs).last_hidden_state.mean(1).squeeze(0)
        emb = emb.float().cpu().numpy()

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
