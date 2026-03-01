# OsteoMin

OsteoMin is a 2D lattice-based cellular automaton for MC3T3-E1 monolayers with coupled cumulative field dynamics for ALP, collagen, and HAp. The model advances in hourly steps and supports stochastic movement, proliferation, osteogenic commitment, apoptosis, and local field deposition.

## Layout

```text
.
├── main.py
├── README.md
├── result_report.md
├── sanity_check1.py
├── requirements.txt
└── src/
    ├── abm_manager.py
    ├── agent.py
    ├── config.py
    ├── datafit.py
    ├── field_manager.py
    ├── io_utils.py
    ├── model.py
    └── viz.py
```

## Run

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Reference run:

```bash
python3 main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed 12345 \
  --w 33 --h 33 \
  --nodules 24n \
  --dex-conc 0 \
  --mk4-conc 0 \
  --alp-rb 0.001 \
  --alp-ro 0.002 \
  --collagen-rb 0.001 \
  --collagen-ro 0.002 \
  --ha-model minimal \
  --ha-ro 0.075 \
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
  --commit-k 0.07 \
  --commit-k-growth 0.0007 \
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
  --no-hourly-exports
```

## Model summary

- Time step: `1 h`
- Domain: `w x h` lattice, periodic or clamped boundaries
- Cell states: `0 empty`, `1 preOB`, `2 OB`, `3 apoptotic`
- Update order each hour: movement, proliferation, commitment, apoptosis, field deposition
- Field outputs: cumulative ALP, collagen, HAp
- Collagen osteogenic term: AA pulse/decay with Michaelis-Menten gating
- HAp term: post-commitment Hill activation with `--ha-tau` and `--ha-hill-n`

## Important CLI groups

Core:

- `--hours`
- `--cells`
- `--seed-mode`
- `--seed`
- `--w`, `--h`
- `--boundary-mode`
- `--media-interval`

Movement and division:

- `--random-move`, `--no-random-move`
- `--move-prob`
- `--max-moves`
- `--proliferation`, `--no-proliferation`
- `--min-div-hours`, `--max-div-hours`
- `--ob-div-prob`

Commitment and death:

- `--osteogenic-commitment`, `--no-osteogenic-commitment`
- `--osteo-induction-day`
- `--commit-delay`
- `--commit-k`
- `--commit-k-growth`
- `--apoptosis`, `--no-apoptosis`
- `--apoptosis-prob`

Field dynamics:

- `--alp-rb`, `--alp-ro`
- `--collagen-rb`, `--collagen-ro`
- `--collagen-aa-km`
- `--aa0`, `--aa-half-life`
- `--dex-conc`, `--mk4-conc`
- `--ha-model`
- `--ha-ro`
- `--ha-tau`
- `--ha-hill-n`

Outputs:

- `--out`
- `--spatial-every`
- `--hourly-exports`, `--no-hourly-exports`
- `--snapshots`, `--no-snapshots`
- `--csv`
- `--xlsx`, `--no-xlsx`
- `--unique-id`, `--no-unique-id`

## Outputs

Main run products include:

- `run.json`
- `timeseries.xlsx` or `timeseries.csv`
- summary figures in the output root
- PDF copies in `figures_pdf/`
- spatial overlays in `spatial_cells/` and `collagen_ha/`
- raw grids in `grids/` when snapshots are enabled

Current summary figures include:

- cell counts and daily events
- normalized ALP, collagen, and HAp accumulation
- nodule count
- cell motility summary
- absolute field totals
- ascorbate schedule
- ascorbate-gated collagen activation
- cell cycle summary

## Sanity checks

Run:

```bash
python3 sanity_check1.py
```

This writes scenario folders under `out/sanity_check1/` and generates a markdown report with:

- final-state metrics for each toggle run
- control-vs-perturbation contrasts
- monotonicity checks for cumulative fields
- structural checks for counts and nodule history

## Notes

- `--datafit` remains available as a legacy path, but the default workflow uses direct field-law parameters from the CLI.
- OB apoptosis is intentionally disabled in the current implementation.
- Internal variable names still use `ha` in some files, but user-facing text and plots refer to `HAp`.
