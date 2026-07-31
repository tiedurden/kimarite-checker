"""Stage 0: download a whole YouTube playlist as raw videos.

    python fetch_playlist.py "https://youtube.com/playlist?list=..."

Whole videos, not sections -- bout boundaries are unknown until segment_bouts.py
finds the overlays. Output: raw/<videoid>.mp4
"""

import argparse
import subprocess
import sys
from pathlib import Path

# 480p: the encoder centre-crops to 224x224 anyway, and the overlay stays legible
# for label extraction. ~10x smaller than 1080p.
FORMAT = "bv*[height<=480]+ba/b[height<=480]/bv*+ba/b"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url", help="playlist or single-video URL")
    ap.add_argument("--raw", default="raw", type=Path)
    args = ap.parse_args()

    for tool in ("yt-dlp", "ffmpeg"):
        if subprocess.run(["which", tool], capture_output=True).returncode != 0:
            sys.exit(f"{tool} not on PATH -- see README.md (Prerequisites)")

    args.raw.mkdir(parents=True, exist_ok=True)

    # %(id)s only: the video id is the group key for leakage-safe splitting, and
    # sumo video titles are full of characters that break shell/path handling.
    # --download-archive makes re-runs resumable after an interrupted download.
    cmd = [
        "yt-dlp", "--yes-playlist", "-f", FORMAT,
        "--merge-output-format", "mp4",
        "--download-archive", str(args.raw / ".archive.txt"),
        "--ignore-errors",  # one deleted/private video shouldn't kill 29 others
        "-o", str(args.raw / "%(id)s.%(ext)s"),
        args.url,
    ]
    print(f"downloading to {args.raw}/ ...")
    rc = subprocess.run(cmd).returncode

    got = sorted(p for p in args.raw.glob("*.mp4"))
    print(f"\n{len(got)} videos in {args.raw}/")
    if rc != 0:
        print("(yt-dlp reported errors -- some videos may be unavailable)",
              file=sys.stderr)
    if got:
        print(f"next: python segment_bouts.py --probe {got[0].name}")


if __name__ == "__main__":
    main()
