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
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from transformers import VideoMAEImageProcessor, VideoMAEModel

# 16 frames is what VideoMAE was pretrained with; deviating hurts more than it helps.
NUM_FRAMES = 16
RESIZE = 224

# MCG-NJU/videomae-base, not OpenGVLab/VideoMAEv2-Base. The v2 repo ships its
# architecture as custom code (auto_map -> modeling_videomaev2.VideoMAEv2), so
# loading it requires trust_remote_code=True -- executing a third party's Python
# at import time. Its preprocessor_config.json also still declares the
# VideoMAEFeatureExtractor class, which transformers 5.x removed, so
# AutoVideoProcessor cannot infer a type from it at all.
#
# v1 base is a first-party transformers architecture: no remote code, same input
# pipeline, same 768-dim output. If you want v2's stronger features later, pin
# transformers ~4.46 in a separate venv rather than enabling remote code here.
MODEL_ID = "MCG-NJU/videomae-base"

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


def restore_qv_bias(model: VideoMAEModel, model_id: str) -> int:
    """Copy the checkpoint's q_bias/v_bias into transformers 5.x's query/value.bias.

    VideoMAE's original attention stores a q bias and a v bias and deliberately has
    NO k bias (a k bias is redundant under softmax). transformers 5.x renamed these
    to the standard query/key/value.bias triple, so loading a VideoMAE checkpoint
    prints "newly initialized" for all three and silently drops the real q/v values
    -- they initialize to ZERO, so nothing crashes and the embeddings are quietly
    wrong. Measured norms of the discarded tensors: q 17.54, v 1.80.

    Returns the number of tensors restored; the caller checks it is the expected
    2 per layer, because a silent zero here is exactly the failure mode that made
    this necessary in the first place.
    """
    sd = load_file(hf_hub_download(model_id, "model.safetensors"))
    n = 0
    for i, layer in enumerate(model.encoder.layer):
        attn = layer.attention.attention
        for src, dst in (("q_bias", "query"), ("v_bias", "value")):
            for key in (f"videomae.encoder.layer.{i}.attention.attention.{src}",
                        f"encoder.layer.{i}.attention.attention.{src}"):
                if key in sd:
                    getattr(attn, dst).bias.data.copy_(sd[key])
                    n += 1
                    break
    return n


def load_encoder(device: str):
    """(processor, model) ready for inference. Shared with predict.py.

    Inference MUST build the encoder exactly as training did -- a processor with
    different normalization, or a model missing the q/v biases, produces
    embeddings in a different space than the cached ones the head was fit on. The
    head would still return confident-looking probabilities. So this lives in one
    place and both callers use it.
    """
    processor = VideoMAEImageProcessor.from_pretrained(MODEL_ID)
    model = VideoMAEModel.from_pretrained(
        MODEL_ID,
        dtype=torch.bfloat16 if device == "cuda" else torch.float32,
    )
    restored = restore_qv_bias(model, MODEL_ID)
    expected = 2 * len(model.encoder.layer)
    if restored != expected:
        sys.exit(f"restored {restored} q/v bias tensors, expected {expected} -- "
                 f"checkpoint layout changed; embeddings would be silently wrong")
    return processor, model.to(device).eval(), restored


def embed(frames: np.ndarray, processor, model, device: str) -> np.ndarray:
    """Frames -> one pooled embedding. Shared so training and inference agree."""
    inputs = processor(list(frames), return_tensors="pt").to(device)
    if device == "cuda":
        inputs = {k: v.to(torch.bfloat16) if v.is_floating_point() else v
                  for k, v in inputs.items()}
    with torch.inference_mode():
        hidden = model(**inputs).last_hidden_state
        # Mean-pool patch tokens: VideoMAE has no CLS token to lean on.
        return hidden.mean(dim=1).squeeze(0).float().cpu().numpy()


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
    processor, model, restored = load_encoder(args.device)
    print(f"  restored {restored} q/v attention biases "
          f"(transformers renamed them; they load as zeros otherwise)")

    done = skipped = failed = 0
    dim = None
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

        emb = embed(frames, processor, model, args.device)

        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, emb)
        dim = emb.shape[-1]
        done += 1

    print(f"\nembedded {done}, skipped {skipped} cached, {failed} failed")
    if dim is None and skipped:
        # Everything was cached: read a dim from disk rather than referencing the
        # loop variable, which is unbound on a fully-cached re-run (NameError).
        cached = next(args.cache.rglob("*.npy"), None)
        dim = np.load(cached).shape[-1] if cached is not None else None
    if dim:
        print(f"dim={dim} -> next: python train_head.py")


if __name__ == "__main__":
    main()
