# sumowithherbert channel — assessment

7 playlists, **110 videos, 64 h** of Makuuchi footage (4.6x the NHK set), 854x480.

| Tournament | Playlist ID | Videos | Overlaps NHK set? |
|---|---|---|---|
| Nagoya 2026 | `PLOezJEjOULdE` | 16 | **YES** — same basho as `PLA3aUoW7UWbQ` |
| Natsu 2026 | `PLcj4Cg-53LKrjl9GIvlEuoM809L5rhK76` | 16 | **YES** — same basho as `PLeLZ23ieCfrbMECZiukp-UYdfOObYeAq_` |
| Haru 2026 | `PLcj4Cg-53LKoZW6b3dXURwFrwEnB3MAs7` | 16 | no |
| Hatsu 2026 | `PLcj4Cg-53LKrEFlLhfoBAuwSBWoY7jE34` | 16 | no |
| Kyushu 2025 | `PLcj4Cg-53LKqYuORNB2lelOHK5YIeBEka` | 15 | no |
| Aki 2025 | `PLcj4Cg-53LKrEHB5rbJBE69qnbY1hEgeL` | 15 | no |
| Nagoya 2025 | `PLcj4Cg-53LKoX_iGkZrmoWbWc24PRgWYN` | 16 | no |

## Measured bout coverage (all 7 playlists scraped)

`hb_manifest.py` on the full set: **1647 bouts from 82 videos**, median window 88s
(p25 76s, p75 109s). Of the 110 videos, 27 have no parseable timestamps and 1 was
a cross-playlist duplicate.

| Tournament | Bouts | Videos w/o timestamps |
|---|---:|---:|
| hatsu2026 | 314 | 1 |
| nagoya2026 | 307 | 1 |
| haru2026 | 298 | 1 |
| natsu2026 | 286 | 1 |
| kyushu2025 | 274 | 2 |
| aki2025 | 146 | 8 |
| nagoya2025 | 22 | 13 |

Corrections to earlier assumptions in this file, both found by scraping all 7
playlists rather than extrapolating from Haru 2026:

- **2025 tournaments DO carry timestamps** — Kyushu 2025 and Aki 2025 are nearly
  complete. They just use a different format (`01:09 Tamashoho vs Shishi`, no
  separator) than the 2026 sets (`01:55 - Ryuden vs Kotoeiho`). The original
  `TS_RE` required the separator and silently dropped every bout in those videos.
- **Nagoya 2025 is the real gap**, not "all of 2025": 13 of its 16 videos have
  genuinely empty descriptions. Those need `segment_bouts.py`.
- **One video sits in two playlists.** `ykaQBUI-T6w` is a Nagoya 2026 upload that
  is also filed under Nagoya 2025. It must be counted once, under 2026 — see the
  dedupe note in `hb_manifest.py` for why this one is a leak risk and not just
  double-counting.

## Why the NHK pipeline does NOT transfer

Different broadcast entirely (this is Abema/大相撲 live coverage, not NHK):

- **Caption is prose, not a plate.** `"ASAKORYU WINS BY TSUKIOTOSHI"` / `"URA WINS
  BY OSHIDASHI"` as small white text, bottom-left, with the winner's NAME in it.
  NHK used a fixed-geometry plate with romaji + English gloss.
- **Variable width.** Because the caption embeds the wrestler's name, its pixel
  width changes per bout. Clustering on raw pixels (`label_clusters.py`) assumes
  identical renders — that assumption is now broken. Same kimarite with different
  winners will NOT cluster together.
- **Full bouts, not highlights.** ~35 min/video vs NHK's ~12-25 min, with LIVE
  badge, torikumi boards, rank/record tables, flags, and a persistent REPLAY
  marker bottom-left — in the SAME region as the caption.
- **More graphics clutter.** Kanji shikona boards top-left, per-wrestler stat
  tables mid-frame, sponsor/venue furniture.

## Recommended plan

**1. Do NOT mix these into the NHK training set.** Two of the seven playlists are
the *same tournaments* the NHK set covers — the same bouts from a second camera.
Mixed in, those become near-duplicates that straddle any train/test split and
inflate accuracy, exactly the leak `train_head.py`'s grouping exists to prevent.
Group key would need to be (tournament, day), not video id.

**2. Use the 5 non-overlapping tournaments as a held-out generalization set.**
Haru 2026, Hatsu 2026, Kyushu 2025, Aki 2025, Nagoya 2025 = 78 videos, no bout
overlap with NHK. This is the honest answer to "does it work on footage it has
never seen the style of" — far more valuable than more same-style training data.

**3. Label them by OCR, not clustering — implemented in `hb_label.py`.**
The caption is prose containing the kimarite as a word, so variable-width prose
defeats pixel clustering. Tesseract on a 4x-upscaled grayscale crop works; the
vocabulary is closed (`kimarite.py normalize()`), so OCR output is fuzzy-matched
against the 82 names and misses are rejected rather than guessed at.

`WIN_BOX` was calibrated by OCR bisection on `nnSWd2TFUVo` (854x480), NOT by
eyeballing a contact sheet — the first eyeballed box was ~15 px too high and
sliced the glyph tops. Verified reading `KINBOZAN WINS BY OSHIDASHI`.

Two things that cost accuracy and are already handled:
- OCR intermittently drops inter-word spaces (`WINSBYOSHIDASHINe`). `normalize()`'s
  fuzzy match absorbs it, but that means a misread can also land on a *real but
  wrong* technique — so `MIN_VOTES=2` frames must agree before a label is accepted.
  A silently wrong label is worse than a miss; a miss at least shows in the count.
- The caption appears during the **replay**, after the bout, so clips are cut
  `CLIP_BACK=4s` before it and the whole strip is masked (`LEAK_BOXES_HB`).

**4. Fold the winner's name out of the label.** `"X WINS BY Y"` gives you the
winner too. Tempting, but it makes the caption region a name leak as well —
mask the whole strip, don't try to keep part of it.

## Cost if you do process them

- download 110 videos: ~25 min (measured 11 s/video, but these are ~3x longer)
- overlay scan 64 h @ 48x realtime: **~80 min**
- separate `BOX` calibration required — REPLAY badge shares the caption region,
  so onset detection will fire on replays without a tighter box or a colour filter
