# Sanity Check 1 Report

Generated: 2025-11-15T21:08:34

## Positive control

- Command template: `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed <random> --out <dir>`
- All toggles are enabled in the positive control run.

## Toggle definitions

| Toggle | Description |
| --- | --- |
| `random_move` | Active cells explore neighboring lattice sites before division. |
| `osteogenic_commitment` | ACTIVE cells differentiate into osteoblasts when enabled. |
| `apoptosis` | Per-cell death events are sampled each day. |
| `proliferation` | Cells attempt to divide once their interval elapses. |
| `secretion` | Osteoblasts deposit collagen/ALP that fuels the field manager. |
| `nucleation` | Pi+Ca form ACP, which can mature into HA. |
| `commit_delay` | Delay (days) before commit probability turns on. |

## Run matrix

| Toggle | State | Seed | Output directory | Status | Command |
| --- | --- | --- | --- | --- | --- |
| `positive_control` | positive | 331289 | `out/sanity_check1/positive_control` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 331289 --out out/sanity_check1/positive_control` |
| `random_move` | negative | 643406 | `out/sanity_check1/random_move_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 643406 --no-random-move --out out/sanity_check1/random_move_negative` |
| `osteogenic_commitment` | negative | 860627 | `out/sanity_check1/osteogenic_commitment_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 860627 --no-osteogenic-commitment --out out/sanity_check1/osteogenic_commitment_negative` |
| `apoptosis` | negative | 21165 | `out/sanity_check1/apoptosis_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 21165 --no-apoptosis --out out/sanity_check1/apoptosis_negative` |
| `proliferation` | negative | 922576 | `out/sanity_check1/proliferation_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 922576 --no-proliferation --out out/sanity_check1/proliferation_negative` |
| `secretion` | negative | 389235 | `out/sanity_check1/secretion_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 389235 --no-secretion --out out/sanity_check1/secretion_negative` |
| `nucleation` | negative | 890895 | `out/sanity_check1/nucleation_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 890895 --no-nucleation --out out/sanity_check1/nucleation_negative` |
| `commit_delay` | negative | 378045 | `out/sanity_check1/commit_delay_negative` | ✅ success | `/home/aliad/projects/osteomin/.venv/bin/python main.py --days 30 --cells 400 --w 33 --h 33 --boundary-mode periodic --media-interval 2 --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 --random-move --move-prob 1.0 --max-moves 1 --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.001 --osteogenic-commitment --commit-growth 0.01 --commit-delay 3 --secretion --nucleation --enable-nodules --spatial-every 1 --snapshots --xlsx --nodule-window 4 --k-ha 0.04 --k-nuc 0.01 --s-thresh 0.01 --seed 378045 --commit-delay 0 --out out/sanity_check1/commit_delay_negative` |

## Interpretation notes

- Each directory under `out/sanity_check1/` contains the model outputs for the positive control or a toggle-specific negative run.
- Compare the `positive_control` directory against each `<toggle>_negative` directory (Excel, collagen/HA fields, cell snapshots) to quantify the contribution of that behavior.
- The `commit_delay` toggle keeps all other booleans fixed while changing the day ACTIVE cells are allowed to commit.

## Next steps

1. Open relevant `timeseries.xlsx` sheets for each pair to compare deaths/commits/mineral totals.
2. Use the `_snapshots` folders to verify that collagen/HA overlays obey the expected toggle.
3. Adjust `--nucleation`/`--no-nucleation` runs for more than one mineralization scenario if needed.