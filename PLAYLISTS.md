# Data sources

Feed these to `fetch_playlist.py` one at a time (it appends to `raw/` and keeps a
download archive, so re-running is safe and resumable).

## Verified — NHK GRAND SUMO channel, 854x480, identical graphics package

| Set | Playlist ID | Videos | Division | Tournament |
|-----|-------------|--------|----------|------------|
| Juryo Nagoya | `PLfAfwgMLoCqs` | 15 | Juryo | July 2026 Nagoya |
| Makuuchi Nagoya | `PLA3aUoW7UWbQ` | 15 | Makuuchi | July 2026 Nagoya |
| Makuuchi Natsu | `PLeLZ23ieCfrbMECZiukp-UYdfOObYeAq_` | 15 | Makuuchi | May 2026 Natsu |

45 videos total. Days 1-15 of each, except Natsu Days 1-2 which are a "TOP 10
BOUTS" and a "BOUTS ANALYSIS" edit -- different structure, expect odd onset
counts from those two.

Overlay geometry calibrated against `xHQqS4zIeAA` (Juryo Day 1) and spot-checked
against `bopkt9_roXI` (Makuuchi Nagoya Day 1): same package, same resolution, so
one `BOX` covers all three sets.

## Grouping caveat

Two of the three sets are the same tournament (July 2026 Nagoya), and all three
are one channel with one graphics package. So the 45 CV groups are NOT 45
independent conditions:

- 2 tournaments (Nagoya, Natsu), not 3
- 1 broadcaster, 1 camera setup, 1 overlay style
- Nagoya Juryo and Nagoya Makuuchi share a venue, lighting, and dohyo

A model trained on this will generalize across *bouts* and *wrestlers*, but has
no way to learn invariance to broadcaster or graphics style. Expect a drop on
footage from any other source -- which is what the 5 other-channel videos are
useful for: hold them out entirely as a true generalization test rather than
mixing them into training.

## Pending

5 videos from a second channel -- URLs not yet supplied. Re-run
`segment_bouts.py --probe` on one of these before segmenting: a different channel
almost certainly means a different overlay position, so `BOX` will need a second
calibration (and probably a per-channel override).
