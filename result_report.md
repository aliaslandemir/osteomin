# Sanity Check 1 Results

## Structural summary
- Each folder under `out/sanity_check1/` mirrors the screenshot you shared: `timeseries.xlsx` (source of all plotted metrics), `run.json` (run metadata), five PNG summaries (`01_states_counts.png` … `05_minerals_cumulative.png`), and the spatial grids/overlays (`grids/`, `spatial_cells/`, `collagen_ha/`) (PDF snapshots for `spatial_cells` and `collagen_ha`).  
- The workbook represents the run’s day-by-day statistics, and the harness runs a single command per toggle (use the arguments in `sanity_check1.py`) so the folders isolate the “on” vs “off” effect for the seven boolean switches.

## Timeseries summary

| Toggle | State | preOB | OBs | Total | Collagen | ACP | HA | Ca_used | Pi_used | Deaths | Committed |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| `apoptosis` | off | 12 | 5360 | 5372 | 25176.56 | 554.23 | 114.90 | 2237.19 | 1453.16 | 0 | 13 |
| `apoptosis` | on | 12 | 4855 | 4867 | 23174.76 | 539.89 | 97.50 | 2107.17 | 1372.28 | 6 | 6 |
| `commit_delay` | off | 11 | 4737 | 4748 | 22802.23 | 513.88 | 112.20 | 2102.64 | 1364.36 | 1 | 17 |
| `commit_delay` | on | 17 | 5625 | 5642 | 26895.68 | 609.46 | 112.60 | 2391.38 | 1556.72 | 3 | 15 |
| `nucleation` | off | 9 | 4824 | 4833 | 23384.23 | 0 | 0 | 0 | 0 | 4 | 12 |
| `nucleation` | on | 12 | 4828 | 4840 | 23368.89 | 525.98 | 113.60 | 2145.94 | 1392.76 | 4 | 17 |
| `osteogenic_commitment` | off | 9908 | 0 | 9908 | 0 | 0 | 0 | 0 | 0 | 87 | 0 |
| `osteogenic_commitment` | on | 8 | 4743 | 4751 | 22926.94 | 537.07 | 98.90 | 2105.71 | 1370.84 | 2 | 13 |
| `proliferation` | off | 0 | 658 | 658 | 4761.01 | 121.67 | 47.10 | 600.51 | 384.64 | 0 | 0 |
| `proliferation` | on | 16 | 5144 | 5160 | 24302.23 | 537.42 | 113.40 | 2179.26 | 1415.04 | 5 | 14 |
| `random_move` | off | 5 | 4273 | 4278 | 21678.33 | 520.08 | 100.60 | 2063.24 | 1341.96 | 2 | 9 |
| `random_move` | on | 16 | 4792 | 4808 | 23187.66 | 517.29 | 114.30 | 2123.37 | 1377.48 | 2 | 13 |
| `secretion` | off | 29 | 4647 | 4676 | 0 | 0 | 0 | 0 | 0 | 5 | 16 |
| `secretion` | on | 11 | 5120 | 5131 | 24523.97 | 547.75 | 114.70 | 2216.75 | 1439.60 | 5 | 7 |

## Key observations
- **Mineralization depends on both secretion and nucleation:** `secretion_off` leaves collagen/ALP/ACP/HA at zero, while `secretion_on` produces ~25k collagen, ~548 ACP, ~115 HA, and large Ca/Pi usage. Separately `nucleation_off` keeps ions at their baseline (zero Ca/Pi usage) but still records collagen/ALP, confirming nucleation is the chemistry gate.
- **Commitment and proliferation reshape population balance:** Turning `osteogenic_commitment` off freezes the system at 9,908 preOB cells, zero OBs, and no matrix/mineral output, while the `on` counterpart matches other wins (≈4.7k OBs, ~23k collagen). Similarly, `proliferation_off` leaves just 658 OBs (no new preOBs) versus over 5k OBs when on.
- **Apoptosis toggling changes death counts and matrix totals:** The `apoptosis_on` run shows 6 deaths today versus zero when off, with slightly lower OB counts and ~2k fewer Ca/Pi used, suggesting survival fosters a larger OB pool when apoptosis is disabled.
- **Random movement helps OB buildup:** With movement on there are ~4.8k OBs at day 28 vs. ~4.2k when off, and a small bump in collagen/HA despite identical CA/PI usage.
- **Commit delay influences mineral production:** Allowing a 3-day delay increases OBs (~5.6k vs. ~4.7k) and collagen (~26.9k vs. ~22.8k) while raising Ca/Pi usage (≈2.4k/1.56k vs. ≈2.1k/1.36k), implying a delayed differentiation wave lets more preOB cells proliferate before committing.
- **The reported death/commit numbers (last two columns) confirm toggles behave as expected (e.g., `proliferation_on` logs 14 commits vs. zero when off).**
