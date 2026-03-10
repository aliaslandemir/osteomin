# Result Report



## Reference cases

Two baseline cases are useful for routine comparison:

- `osteogenic baseline`: commitment enabled, AA present at refresh, secretion on
- `basal baseline`: commitment disabled, no AA pulse, secretion still on

These two runs give a stable contrast between a culture allowed to progress into the osteogenic program and a restrained baseline condition with the same lattice, seeding, movement, and death settings.

## Osteogenic baseline

```bash
python3 main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed 12345 \
  --w 33 --h 33 \
  --nodules 40n \
  --dex-conc 0 \
  --mk4-conc 0 \
  --alp-rb 0.001 \
  --alp-ro 0.002 \
  --alp-a-d 0.08 \
  --alp-b-d 0.08 \
  --alp-a-k 0.06 \
  --alp-b-k 0.06 \
  --collagen-rb 0.001 \
  --collagen-ro 0.002 \
  --collagen-a-d 0.08 \
  --collagen-b-d 0.08 \
  --collagen-a-k 0.06 \
  --collagen-b-k 0.06 \
  --ha-model minimal \
  --ha-ro 0.075 \
  --ha-a-d 0.08 \
  --ha-b-d 0.08 \
  --ha-a-k 0.06 \
  --ha-b-k 0.06 \
  --ha-tau 384 \
  --ha-hill-n 4 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 0.27 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 \
  --osteogenic-commitment \
  --osteo-induction-day 3 \
  --commit-delay 48 \
  --commit-k 0.08 \
  --commit-k-growth 0.0008 \
  --aa0 1.0 \
  --aa-half-life 6 \
  --collagen-aa-km 1.0 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion \
  --spatial-every 1 \
  --out main_out \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --runtime
```

## Basal baseline

```bash
python3 main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed 12345 \
  --w 33 --h 33 \
  --nodules 40n \
  --dex-conc 0 \
  --mk4-conc 0 \
  --alp-rb 0.001 \
  --alp-ro 0.002 \
  --alp-a-d 0.08 \
  --alp-b-d 0.08 \
  --alp-a-k 0.06 \
  --alp-b-k 0.06 \
  --collagen-rb 0.001 \
  --collagen-ro 0.002 \
  --collagen-a-d 0.08 \
  --collagen-b-d 0.08 \
  --collagen-a-k 0.06 \
  --collagen-b-k 0.06 \
  --ha-model minimal \
  --ha-ro 0.075 \
  --ha-a-d 0.08 \
  --ha-b-d 0.08 \
  --ha-a-k 0.06 \
  --ha-b-k 0.06 \
  --ha-tau 384 \
  --ha-hill-n 4 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 0.27 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 \
  --no-osteogenic-commitment \
  --osteo-induction-day 3 \
  --commit-delay 48 \
  --commit-k 0.08 \
  --commit-k-growth 0.0008 \
  --aa0 0.0 \
  --aa-half-life 6 \
  --collagen-aa-km 1.0 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion \
  --spatial-every 1 \
  --out main_out \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --runtime
```

## Shared simulation frame

Both reference cases use:

- `672` simulated hours
- `72` initial cells
- a `33 x 33` periodic lattice
- random seeding with fixed seed `12345`
- movement enabled with `--move-prob 0.27`
- proliferation enabled
- pre-osteoblast apoptosis enabled
- secretion enabled
- daily spatial export through `--spatial-every 1`

## Main distinctions between the two baselines

- osteogenic baseline uses `--osteogenic-commitment`
- basal baseline uses `--no-osteogenic-commitment`
- osteogenic baseline uses `--aa0 1.0`
- basal baseline uses `--aa0 0.0`

Everything else is held fixed so that differences can be traced to the osteogenic program rather than unrelated run settings.

## Output checklist

For each reference run, the following files are the first ones worth checking:

- `run.json`
- `timeseries.xlsx`
- `01_states_counts.png`
- `05_minerals_cumulative.png`
- `06_alp_total.png`
- `07_collagen_total.png`
- `08_nodule_count.png`
- `09_field_totals_absolute.png`

If snapshots are enabled, the grid and overlay folders provide the spatial readout for the same run.

## Related folders

- `endpointaad/`: endpoint image and assay analysis
- `parameterfit/`: parameter calibration and dose-response fitting
- `sanity/`: sanity and regression-style scenario checks

