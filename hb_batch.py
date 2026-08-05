"""Stages 0-2 in batches: download, label, cut and embed N videos at a time.

WHY BATCHES. Doing all 68 remaining videos in one run is ~13 GB of download and,
on CPU, about 4.5 h of embedding (measured 2.8 s/clip x ~1380 bouts x 4 caches).
A single long run is the wrong shape for that: it pins the machine for an evening
with no usable output until the end, and anything that goes wrong -- a dead video,
a source whose graphics the OCR box does not fit -- is discovered at hour four.

One batch of --videos is ~30 min and leaves the project in a fully consistent
state: labels written, clips cut, every cache in step. So progress is checkpointed
in units that are individually worth having, and stopping is free.

    python hb_batch.py                     # one batch of 5 videos
    python hb_batch.py --dry-run           # show the plan, download nothing
    python hb_batch.py --batches 4         # four batches back to back (~2 h)
    python hb_batch.py --videos 3          # smaller batch

Deliberately SEQUENTIAL, one ffmpeg/tesseract/torch process at a time. Running six
labelling shards in parallel once pinned this laptop at 12 concurrent transcodes
(see extract_strips() in hb_label.py); the batching here is about bounding elapsed
time per checkpoint, not about using more cores.

VIDEO ORDER is round-robin across tournaments, not the manifest order. Every batch
then adds venues/cameras rather than more of one venue, which is what the learning
curve is measured over (hb_curve.py scores leave-one-VIDEO-out) -- and it means a
batch is never confounded with a single tournament.

Tournaments that overlap the NHK playlists go LAST. Those are the same bouts from a
different camera, so if an NHK set is ever built as the held-out test, training on
them contaminates it. There is no NHK dataset today (data/ and cache/ are empty),
so excluding them outright would give up 30 videos for a hypothetical -- ordering
them last preserves the option for free: stop before those batches and the test set
stays clean. --exclude-nhk-overlap drops them entirely.
"""

import argparse
import csv
import shutil
import subprocess
import sys
import time
from collections import Counter, OrderedDict
from pathlib import Path

from fetch_playlist import FORMAT
from hb_manifest import OVERLAPS_NHK

# The measured-best configuration (README): finish windows at 2/3/6 s, concatenated.
SCALES = [2.0, 3.0, 6.0]

# 4 s was cut into data_finish/cache_finish before the sweep existed, so it does not
# follow the fin<N> pattern. Mapped rather than renamed: the 4 s numbers in the README
# were measured against those directory names.
SCALE_DIRS = {4.0: "finish"}

# Videos yt-dlp could not fetch (deleted, private, region-locked). Recorded so the
# next batch does not pick the same dead video again and stall forever on it.
FAILED_FILE = Path("_dl_failed.txt")

# Videos already handed to hb_label.py, whether or not any bout survived. Needed
# because "has rows in hb_labels.csv" is NOT the same as "has been processed":
# T1L6xIHl91I has 2 manifest bouts and both missed OCR, so it wrote no rows and the
# next batch picked it again. A video that yields nothing yields nothing every time,
# so without this it is retried forever and every later batch is one video short.
ATTEMPTED_FILE = Path("_batch_attempted.txt")

# Batch bouts manifest handed to hb_label.py. Underscore-prefixed, so .gitignore's
# `_*` rule keeps this scratch file out of commits.
BATCH_BOUTS = Path("_batch_bouts.csv")

# Measured: 3.3 GB of raw/ for 18 videos at 480p.
GB_PER_VIDEO = 0.19
FREE_GB_FLOOR = 5.0


def mmss(sec: float) -> str:
    return f"{int(sec) // 60}m{int(sec) % 60:02d}s"


def run(cmd: list[str], label: str) -> int:
    """Run a child stage, streaming its output straight through.

    Not captured: these stages print per-bout progress and a batch takes ~30 min,
    so swallowing it until the end would leave the user watching a silent process.
    """
    print(f"\n=== {label}\n$ {' '.join(cmd)}", flush=True)
    t = time.monotonic()
    rc = subprocess.run(cmd).returncode
    print(f"--- {label}: {mmss(time.monotonic() - t)}"
          + (f"  (exit {rc})" if rc else ""), flush=True)
    return rc


def scale_dirs(s: float) -> tuple[Path, Path]:
    name = SCALE_DIRS.get(s, f"fin{s:g}")
    return Path(f"data_{name}"), Path(f"cache_{name}")


def round_robin(groups: dict[str, list[str]]) -> list[str]:
    """Interleave per-tournament lists: one from each, then the next from each."""
    out, lists = [], [list(v) for v in groups.values()]
    for i in range(max((len(v) for v in lists), default=0)):
        for v in lists:
            if i < len(v):
                out.append(v[i])
    return out


def plan(bouts: Path, labels: Path, raw: Path, exclude_nhk: bool
         ) -> tuple[list[str], dict[str, str], dict[str, int], set[str]]:
    """(ordered pending video ids, video->tournament, video->bout count, on disk).

    Pending = no rows in hb_labels.csv AND not in _batch_attempted.txt. Row count is
    deliberately not compared against the manifest: bouts legitimately drop out (OCR
    miss, higi, no bout window) and never come back, so "fewer rows than bouts" would
    mark every finished video as unfinished. hb_label.py --resume handles the
    within-video bookkeeping; this only decides which videos to hand it.
    """
    rows = list(csv.DictReader(bouts.open(encoding="utf-8")))
    tour: dict[str, str] = {}
    count: Counter = Counter()
    for r in rows:
        tour.setdefault(r["video"], r["tournament"])
        count[r["video"]] += 1

    labelled = set()
    if labels.exists():
        labelled = {r["video"]
                    for r in csv.DictReader(labels.open(encoding="utf-8"))}
    dead = set(FAILED_FILE.read_text().split()) if FAILED_FILE.exists() else set()
    tried = (set(ATTEMPTED_FILE.read_text().split())
             if ATTEMPTED_FILE.exists() else set())

    groups: OrderedDict[str, list[str]] = OrderedDict()
    nhk: OrderedDict[str, list[str]] = OrderedDict()
    for v, t in tour.items():
        if v in labelled or v in dead or v in tried:
            continue
        (nhk if t in OVERLAPS_NHK else groups).setdefault(t, []).append(v)

    ordered = round_robin(groups)
    if not exclude_nhk:
        ordered += round_robin(nhk)

    # "On disk" must mean what hb_label.py means by it. That globs *<id>*.mp4, so a
    # file named _probe_<id>.mp4 IS the video to it -- and one existed: a complete
    # 36-min download left over from WIN_BOX calibration. Testing p.stem == id here
    # would have re-downloaded 235 MB and then handed labelling whichever file the
    # glob hit first (the underscore sorts earlier), i.e. a duplicate of the same
    # video under two names. Same substring rule on both sides, so the two agree.
    on_disk = {v for v in ordered if next(raw.glob(f"*{v}*.mp4"), None)}
    return ordered, tour, count, on_disk


def download(vid: str, raw: Path) -> bool:
    """One video. Same FORMAT and archive file as fetch_playlist.py."""
    rc = subprocess.run(
        ["yt-dlp", "--no-playlist", "-f", FORMAT, "--merge-output-format", "mp4",
         "--download-archive", str(raw / ".archive.txt"),
         "-o", str(raw / "%(id)s.%(ext)s"),
         f"https://www.youtube.com/watch?v={vid}"]).returncode
    # The archive makes yt-dlp exit 0 without downloading if the id is already
    # recorded, so success is "the file is there", not the return code. Globbed the
    # same way hb_label.py looks for it -- see the note in plan().
    if next(raw.glob(f"*{vid}*.mp4"), None):
        return True
    print(f"  {vid}: download produced no mp4 (exit {rc}) -- recording as dead",
          file=sys.stderr)
    with FAILED_FILE.open("a") as f:
        f.write(vid + "\n")
    return False


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--videos", type=int, default=5,
                    help="videos per batch (default 5, ~100 bouts, ~30 min)")
    ap.add_argument("--batches", type=int, default=1,
                    help="how many batches to run back to back (default 1)")
    ap.add_argument("--scales", type=float, nargs="+", default=SCALES,
                    help=f"finish window lengths to cut and embed "
                         f"(default {' '.join(f'{s:g}' for s in SCALES)})")
    ap.add_argument("--skip-whole-bout", action="store_true",
                    help="do not embed data_hb. Leaves cache_hb holding fewer clips "
                         "than the finish caches, which makes hb_sweep.py ABORT on "
                         "the clip-set check rather than mispair bouts -- so the "
                         "whole-bout control goes offline until it is caught up.")
    ap.add_argument("--exclude-nhk-overlap", action="store_true",
                    help="drop nagoya2026/natsu2026 entirely instead of ordering "
                         "them last (see the module docstring)")
    ap.add_argument("--bouts", default="hb_bouts.csv", type=Path)
    ap.add_argument("--labels", default="hb_labels.csv", type=Path)
    ap.add_argument("--raw", default="raw", type=Path)
    ap.add_argument("--dry-run", action="store_true",
                    help="print the batch plan and exit")
    args = ap.parse_args()

    if not args.bouts.exists():
        sys.exit(f"{args.bouts} not found -- run hb_manifest.py first")
    args.raw.mkdir(parents=True, exist_ok=True)

    for b in range(1, args.batches + 1):
        # Re-planned every batch rather than sliced once up front: labelling the
        # previous batch is what removes its videos from the pool, and a dead video
        # is only discovered mid-run.
        ordered, tour, count, on_disk = plan(
            args.bouts, args.labels, args.raw, args.exclude_nhk_overlap)
        if not ordered:
            print("nothing left to label -- every manifest video has labels "
                  "(or is recorded as undownloadable).")
            return

        batch = ordered[: args.videos]
        need = [v for v in batch if v not in on_disk]
        bouts_here = sum(count[v] for v in batch)
        left = len(ordered) - len(batch)

        print(f"\n{'=' * 72}\nBATCH {b}/{args.batches}: {len(batch)} video(s), "
              f"~{bouts_here} bouts.  {left} video(s) left after this one.")
        for v in batch:
            mark = "" if v in on_disk else "  (needs download)"
            print(f"  {v}  {tour[v]:<12} {count[v]:>3} bouts{mark}")

        free = shutil.disk_usage(".").free / 1e9
        proj = free - len(need) * GB_PER_VIDEO
        print(f"disk: {free:.1f} GB free, ~{proj:.1f} GB after downloading "
              f"{len(need)}")
        if proj < FREE_GB_FLOOR:
            sys.exit(f"under {FREE_GB_FLOOR:.0f} GB would be left -- free space or "
                     f"delete raw/ videos that are already labelled "
                     f"(hb_recut.py needs them, nothing else does)")

        if args.dry_run:
            print("\ndry run -- nothing downloaded, labelled or embedded.")
            return

        t_batch = time.monotonic()
        for i, vid in enumerate(need, 1):
            print(f"\n=== download {i}/{len(need)}: {vid}", flush=True)
            download(vid, args.raw)
        got = [v for v in batch if next(args.raw.glob(f"*{v}*.mp4"), None)]
        if not got:
            print("no videos in this batch could be downloaded -- stopping.",
                  file=sys.stderr)
            return

        # Hand hb_label.py ONLY this batch's bouts. Passing the full manifest works
        # (it skips rows whose video is absent) but it counts all ~1300 of them as
        # OCR misses, which trips the "MISS RATE 80%" warning and buries the real
        # numbers for the batch.
        src = list(csv.DictReader(args.bouts.open(encoding="utf-8")))
        keep = [r for r in src if r["video"] in set(got)]
        with BATCH_BOUTS.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(src[0]))
            w.writeheader()
            w.writerows(keep)

        # --resume appends to hb_labels.csv and skips bouts already done, so a batch
        # interrupted halfway costs only the bout in flight.
        rc = run([sys.executable, "hb_label.py", "--bouts", str(BATCH_BOUTS),
                  "--raw", str(args.raw), "--out", str(args.labels), "--resume"],
                 f"label {len(keep)} bouts from {len(got)} video(s)")
        # Recorded only on a clean exit: a batch killed midway (machine load, Ctrl-C)
        # must be retried, not skipped, and hb_label.py --resume makes the retry cheap.
        # On a clean exit the video is done even if it produced no rows at all.
        if rc == 0:
            with ATTEMPTED_FILE.open("a") as f:
                f.write("".join(f"{v}\n" for v in got))
        else:
            print(f"  hb_label.py exited {rc} -- not marking these videos as "
                  f"attempted, so the next batch retries them", file=sys.stderr)

        # Cut and embed every scale. Both stages are incremental -- hb_finish.py
        # --skip-existing leaves earlier clips alone, extract_features.py skips any
        # .npy already on disk -- so this touches only the new bouts even though it
        # is pointed at the whole growing dataset.
        for s in args.scales:
            data, cache = scale_dirs(s)
            run([sys.executable, "hb_finish.py", "--len", f"{s:g}",
                 "--data", str(data), "--skip-existing"], f"cut {s:g}s finishes")
            run([sys.executable, "extract_features.py", "--data", str(data),
                 "--cache", str(cache)], f"embed {s:g}s -> {cache}")
        if not args.skip_whole_bout:
            # data_hb was already written by hb_label.py; only the embed is missing.
            run([sys.executable, "extract_features.py", "--data", "data_hb",
                 "--cache", "cache_hb"], "embed whole bouts -> cache_hb")

        n_lab = sum(1 for _ in csv.DictReader(args.labels.open(encoding="utf-8")))
        vids_lab = len({r["video"]
                        for r in csv.DictReader(args.labels.open(encoding="utf-8"))})
        print(f"\n{'=' * 72}\nBATCH {b} done in {mmss(time.monotonic() - t_batch)}. "
              f"{args.labels}: {n_lab} bouts / {vids_lab} videos.")

    print("\nnext:\n"
          "  python hb_sweep.py --lovo          # has the best config changed?\n"
          "  python hb_curve.py                 # is the curve saturating yet?\n"
          "  python hb_batch.py                 # another batch")


if __name__ == "__main__":
    main()
