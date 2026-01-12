# Sanity Check 1 Results (snapshot)

## What’s in each run folder
- `timeseries.xlsx` with day-by-day metrics; `run.json` with CLI/seed; PNG summaries (`01_states_counts.png` … `05_minerals_cumulative.png`); spatial grids/overlays (`grids/`, `spatial_cells/`, `collagen_ha/`).
- Each folder under `out/sanity_check1/` corresponds to the positive control or one toggle-off run built from `sanity_check1.py`.

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

## Key observations (paired comparisons)
- **Secretion off → zero matrix/mineral:** Collagen/ALP/HA all zero; secretion on restores ~25k collagen and ~115 HA with matching Ca/Pi use.
- **Nucleation off → ions at baseline:** Collagen/ALP still accumulate, but Ca/Pi usage stays at zero, confirming nucleation gates mineral formation.
- **Commitment off → no OBs:** ~9,900 preOB, zero OB, no matrix/mineral. Commitment on yields ~4.7k OBs and ~23k collagen.
- **Proliferation off → small population:** ~658 OBs vs. >5k when on; matrix/mineral scale with OB count.
- **Apoptosis off → larger pool:** Fewer deaths, more OBs, higher matrix/mineral totals than apoptosis on.
- **Random move on → modest OB boost:** ~4.8k vs. ~4.2k OBs and slight collagen/HA gains.
- **Commit delay on (72 h) → higher output:** More OBs and collagen/mineral vs. immediate commitment, consistent with extra proliferation before differentiation.
