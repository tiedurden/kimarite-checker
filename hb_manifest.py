"""Stage 0 (Herbert): build a bout manifest from YouTube descriptions.

Herbert hand-writes bout timestamps into every 2026-tournament description:

    01:55 - Ryuden vs Kotoeiho
    03:11 - Fujiryoga vs Kinbozan

That is human-accurate segmentation for free -- no overlay-onset detection, and
we get both wrestlers' names per bout as a bonus. 2025 tournaments have NO
timestamps (verified on Kyushu/Aki/Nagoya 2025); those fall back to
segment_bouts.py onset detection.

A bout runs from its timestamp to the NEXT one, so the window is bounded on both
sides. The kimarite caption appears near the END of that window, which is what
hb_label.py OCRs.

    python hb_manifest.py                      # all playlists in PLAYLISTS_HB
    python hb_manifest.py --playlist PLxxxx    # just one
"""

import argparse
import csv
import json
import re
import subprocess
import sys
from pathlib import Path

# 7 tournaments, 110 videos. 2026 sets carry timestamps; 2025 sets do not.
PLAYLISTS_HB = {
    "nagoya2026": "PLOezJEjOULdE",
    "natsu2026": "PLcj4Cg-53LKrjl9GIvlEuoM809L5rhK76",
    "haru2026": "PLcj4Cg-53LKoZW6b3dXURwFrwEnB3MAs7",
    "hatsu2026": "PLcj4Cg-53LKrEFlLhfoBAuwSBWoY7jE34",
    "kyushu2025": "PLcj4Cg-53LKqYuORNB2lelOHK5YIeBEka",
    "aki2025": "PLcj4Cg-53LKrEHB5rbJBE69qnbY1hEgeL",
    "nagoya2025": "PLcj4Cg-53LKoX_iGkZrmoWbWc24PRgWYN",
}

# Tournaments also covered by the NHK playlists. If NHK is the held-out test set,
# these MUST be excluded from training -- same bouts, different camera angle, so
# including them contaminates the test.
OVERLAPS_NHK = {"nagoya2026", "natsu2026"}

# Longest plausible bout window: pre-bout ritual (shikiri) can run 4 min in the
# top division, plus the bout and replay. Measured windows: min 38s, median 86s,
# max 292s. Anything past this is a description artefact, not a bout.
MAX_BOUT = 300

# Herbert uses (at least) two description formats, and the separator is NOT
# reliable -- an earlier version required one and silently dropped every bout in
# the no-separator videos, which then looked like "no timestamps" and got routed
# to onset detection for nothing:
#
#   "01:55 - Ryuden vs Kotoeiho"   2026 sets: dash (or en/em-dash, or colon)
#   "01:09 Tamashoho vs Shishi"    2025 sets: whitespace only
#
# So the separator is optional. The "A vs B" part is what actually identifies a
# bout line, and it stays mandatory -- that's what keeps "00:00 - Intro / Start"
# and "01:09 Leader board entering" out.
TS_RE = re.compile(
    r"^\s*(?P<ts>(?:\d{1,2}:)?\d{1,2}:\d{2})\s*[-–—:]?\s+"
    r"(?P<a>[A-Za-z][\w'\-\.]*)\s+(?:vs?\.?|VS)\s+(?P<b>[A-Za-z][\w'\-\.]*)",
    re.IGNORECASE,
)


def to_seconds(ts: str) -> int:
    parts = [int(p) for p in ts.split(":")]
    if len(parts) == 2:
        return parts[0] * 60 + parts[1]
    return parts[0] * 3600 + parts[1] * 60 + parts[2]


def playlist_videos(pl: str) -> list[tuple[str, int]]:
    """(video_id, duration) for a playlist, via one flat-playlist call."""
    r = subprocess.run(
        ["yt-dlp", "--flat-playlist", "--print", "%(id)s|%(duration)s",
         f"https://www.youtube.com/playlist?list={pl}"],
        capture_output=True, text=True)
    out = []
    for line in r.stdout.splitlines():
        if "|" not in line:
            continue
        vid, dur = line.split("|", 1)
        try:
            out.append((vid.strip(), int(float(dur))))
        except ValueError:
            out.append((vid.strip(), 0))
    return out


def bouts_from_description(vid: str) -> list[dict]:
    """Parse '<ts> - A vs B' lines out of one video's description."""
    r = subprocess.run(
        ["yt-dlp", "--no-playlist", "--skip-download", "--print", "%(description)s",
         f"https://www.youtube.com/watch?v={vid}"],
        capture_output=True, text=True)
    found = []
    for line in r.stdout.splitlines():
        m = TS_RE.match(line)
        if m:
            found.append({
                "start": to_seconds(m.group("ts")),
                "east": m.group("a"),
                "west": m.group("b"),
            })
    # Descriptions are hand-written; a stray out-of-order line would produce a
    # negative-length bout. Sort and drop duplicate starts.
    found.sort(key=lambda b: b["start"])
    dedup, seen = [], set()
    for b in found:
        if b["start"] not in seen:
            seen.add(b["start"])
            dedup.append(b)
    return dedup


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--playlist", help="single playlist id (default: all)")
    ap.add_argument("--out", default="hb_bouts.csv", type=Path)
    ap.add_argument("--exclude-nhk-overlap", action="store_true",
                    help="skip nagoya2026/natsu2026 (contaminate an NHK test set)")
    args = ap.parse_args()

    if subprocess.run(["which", "yt-dlp"], capture_output=True).returncode != 0:
        sys.exit("yt-dlp not on PATH -- source ./env.sh")

    sets = ({"custom": args.playlist} if args.playlist
            else {k: v for k, v in PLAYLISTS_HB.items()
                  if not (args.exclude_nhk_overlap and k in OVERLAPS_NHK)})
    if args.exclude_nhk_overlap:
        print(f"excluding NHK-overlapping tournaments: {sorted(OVERLAPS_NHK)}\n")

    rows, no_ts, over = [], [], {}
    for name, pl in sets.items():
        vids = playlist_videos(pl)
        print(f"=== {name} ({len(vids)} videos)")
        for vid, dur in vids:
            bouts = bouts_from_description(vid)
            if not bouts:
                no_ts.append((name, vid))
                print(f"  {vid}  NO timestamps -> needs segment_bouts.py")
                continue
            for i, b in enumerate(bouts):
                # End = next bout's start, or video end for the last one. The
                # kimarite caption lands in the last few seconds of this window.
                end = bouts[i + 1]["start"] if i + 1 < len(bouts) else dur
                if end <= b["start"]:
                    continue
                # A real bout window (ritual + bout + replay) runs ~40-300s.
                # Anything longer means the description was truncated or a
                # timestamp is missing, so this "bout" is actually the rest of
                # the video -- observed at 29 min. Clamp it: hb_label.py searches
                # only the window TAIL, so an over-long window makes it look for
                # the caption in the wrong place entirely and the bout is lost.
                if end - b["start"] > MAX_BOUT:
                    end = b["start"] + MAX_BOUT
                    over[name] = over.get(name, 0) + 1
                rows.append({
                    "tournament": name, "video": vid, "n": i + 1,
                    "start": b["start"], "end": end,
                    "east": b["east"], "west": b["west"],
                })
            print(f"  {vid}  {len(bouts)} bouts")

    if not rows:
        sys.exit("no bouts parsed -- check TS_RE against a real description")

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["tournament", "video", "n", "start",
                                          "end", "east", "west"])
        w.writeheader()
        w.writerows(rows)

    by_t = {}
    for r in rows:
        by_t[r["tournament"]] = by_t.get(r["tournament"], 0) + 1
    print(f"\n{len(rows)} bouts -> {args.out}")
    for t, n in sorted(by_t.items()):
        print(f"  {t:<14} {n:>4}")
    if over:
        print(f"\nclamped {sum(over.values())} over-long window(s) to {MAX_BOUT}s "
              f"(truncated description / missing timestamp): "
              + ", ".join(f"{k}:{v}" for k, v in sorted(over.items())))
    if no_ts:
        Path("hb_no_timestamps.json").write_text(json.dumps(no_ts, indent=2))
        print(f"\n{len(no_ts)} video(s) without timestamps -> hb_no_timestamps.json")
        print("  These need overlay-onset detection (segment_bouts.py) instead.")
    print("\nnext: python hb_label.py --probe <video-id>")


if __name__ == "__main__":
    main()
