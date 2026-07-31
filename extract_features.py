"""Stage 1: decode clips -> VideoMAEv2 embeddings, cached to disk.

Run once per dataset. Slow (GPU-bound); everything downstream reads the cache.
Re-running skips clips whose .npy already exists, so it is safe to interrupt.
"""

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModel, AutoVideoProcessor

# 16 frames is what VideoMAE was pretrained with; deviating hurts more than it helps.
NUM_FRAMES = 16
RESIZE = 224
MODEL_ID = "OpenGVLab/VideoMAEv2-Base"

VIDEO_EXTS = {".mp4", ".mkv", ".webm", ".mov", ".avi", ".m4v"}


def decode_frames(path: Path, num_frames: int = NUM_FRAMES) -> np.ndarray | None:
    """Decode `num_frames` evenly-spaced RGB frames via ffmpeg.

    ffmpeg instead of decord/av: no build toolchain, no Python-version wheel
    roulette, and it handles the long tail of broken containers that torchvision
    chokes on. Cost is one subprocess per clip, which is noise next to the GPU.
    """
    # fps filter can't hit an exact count, so oversample then pick indices.
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"scale={RESIZE}:{RESIZE}:force_original_aspect_ratio=increase,"
               f"crop={RESIZE}:{RESIZE}",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    try:
        raw = subprocess.run(cmd, capture_output=True, timeout=300).stdout
    except subprocess.TimeoutExpired:
        print(f"  timeout decoding {path.name}", file=sys.stderr)
        return None

    frame_bytes = RESIZE * RESIZE * 3
    total = len(raw) // frame_bytes
    if total == 0:
        print(f"  no frames in {path.name}", file=sys.stderr)
        return None

    frames = np.frombuffer(raw[: total * frame_bytes], dtype=np.uint8)
    frames = frames.reshape(total, RESIZE, RESIZE, 3)

    # Evenly spaced sample. Short clips get frames repeated rather than dropped,
    # which is what the model expects over zero-padding.
    idx = np.linspace(0, total - 1, num_frames).round().astype(int)
    return frames[idx]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data", type=Path,
                    help="root dir; expects data/<category>/<clip>.mp4")
    ap.add_argument("--cache", default="cache", type=Path)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    clips = sorted(p for p in args.data.rglob("*")
                   if p.suffix.lower() in VIDEO_EXTS and p.is_file())
    if not clips:
        sys.exit(f"no video files under {args.data}/ "
                 f"(expected data/<category>/<clip>.mp4)")

    print(f"loading {MODEL_ID} on {args.device} ...")
    processor = AutoVideoProcessor.from_pretrained(MODEL_ID)
    model = AutoModel.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16 if args.device == "cuda" else torch.float32,
    ).to(args.device).eval()

    done = skipped = failed = 0
    for i, clip in enumerate(clips, 1):
        # Mirror data/ layout inside cache/ so labels stay implicit in the path.
        out = args.cache / clip.relative_to(args.data).with_suffix(".npy")
        if out.exists():
            skipped += 1
            continue

        print(f"[{i}/{len(clips)}] {clip.relative_to(args.data)}")
        frames = decode_frames(clip)
        if frames is None:
            failed += 1
            continue

        inputs = processor(list(frames), return_tensors="pt").to(args.device)
        if args.device == "cuda":
            inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                      for k, v in inputs.items()}

        with torch.inference_mode():
            hidden = model(**inputs).last_hidden_state
            # Mean-pool patch tokens: VideoMAE has no CLS token to lean on.
            emb = hidden.mean(dim=1).squeeze(0).float().cpu().numpy()

        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb)
        done += 1

    print(f"\nembedded {done}, skipped {skipped} cached, {failed} failed")
    if done or skipped:
        print(f"dim={emb.shape[-1] if done else 'see cache'} -> next: python train_head.py")


if __name__ == "__main__":
    main()
