# OsteoMin

OsteoMin is a 2D lattice-based model for osteogenic cell culture simulations. It couples cell-state updates on an hourly grid with accumulated field production for ALP, collagen, and hydroxyapatite-like mineral output.

The codebase is organized around three practical tasks:

- running the spatial model from `main.py`
- analyzing experimental endpoint data in `endpointaad/`
- fitting rate and dose-response parameters in `parameterfit/`

## Repository layout

```text
.
├── main.py
├── README.md
├── result_report.md
├── src/
├── endpointaad/
├── parameterfit/
└── sanity/
```

### Main folders

- `src/`: simulation code, configuration objects, model logic, field dynamics, and plotting.
- `endpointaad/`: experimental endpoint analysis scripts and associated image/data assets.
- `parameterfit/`: scripts and metadata used to estimate rate parameters and dose modifiers.
- `sanity/`: scenario-based sanity checks and figure extraction helpers.
- `main_out/`: generated model outputs. This is run output, not source.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running the model

The entry point is `main.py`. A run writes tables, figures, and metadata to the directory passed with `--out`.

### Baseline osteogenic run

```bash
python main.py \
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
  --osteo-induction-hours 72 \
  --commit-delay 72 \
  --commit-k 0.0125 \
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

### Basal run

```bash
python main.py \
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
  --osteo-induction-hours 72 \
  --commit-delay 72 \
  --commit-k 0.0125 \
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

The two commands above provide a simple starting pair: one with osteogenic commitment and AA support, and one without osteogenic commitment and with `aa0=0`.

## What the model produces

Typical outputs include:

- `run.json`
- `timeseries.xlsx` and optional CSV output
- summary figures in PNG and PDF
- daily spatial cell maps
- collagen/mineral overlays
- raw grid snapshots when enabled

The default figure set covers cell counts, daily and cumulative events, motility, field totals, nodules, ascorbate scheduling, and mineral accumulation.

## Analysis folders

### `endpointaad/`

This folder contains scripts for experimental endpoint analysis. It includes analysis code for ALP, ARS, DAPI, and overlay workflows, together with image data used in those pipelines.

### `parameterfit/`

This folder contains parameter estimation scripts for:

- ALP rates
- collagen and AA-linked collagen rates
- mineral or ARS-linked rates
- dexamethasone modulation
- MK-4 modulation

Supporting metadata tables and condition maps are stored under `parameterfit/metadata/`.

### `sanity/`

This folder contains model sanity checks and helper scripts for extracting figures from those runs. The sanity checks are useful after changing transition rules, rates, or output logic.

Run a check with:

```bash
python3 sanity/sanity_check1.py
```

## Notes for a clean GitHub repository

- Keep generated outputs such as `main_out/` out of commits unless they are intentionally being shared.
- Treat `endpointaad/` as the home for endpoint analysis code and raw analysis assets.
- Treat `parameterfit/` as the source of fitted parameters, not as runtime output.
- Use `result_report.md` for the current reference run description rather than burying run details in commit messages.
