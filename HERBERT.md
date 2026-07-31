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

**3. If you later want them as TRAINING data, label them by OCR, not clustering.**
The caption is prose containing the kimarite as a word. Tesseract or a small VLM
on the cropped strip is the right tool; variable-width prose defeats pixel
clustering. Expect to hand-verify: the vocabulary is closed (see `kimarite.py`
`normalize()`), so fuzzy-match OCR output against the 82 names and reject misses.

**4. Fold the winner's name out of the label.** `"X WINS BY Y"` gives you the
winner too. Tempting, but it makes the caption region a name leak as well —
mask the whole strip, don't try to keep part of it.

## Cost if you do process them

- download 110 videos: ~25 min (measured 11 s/video, but these are ~3x longer)
- overlay scan 64 h @ 48x realtime: **~80 min**
- separate `BOX` calibration required — REPLAY badge shares the caption region,
  so onset detection will fire on replays without a tighter box or a colour filter
