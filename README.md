# OsteoMin

OsteoMin is a Python cellular automaton model of osteogenic differentiation
and mineralization in cell culture. It simulates how pre-osteoblasts move,
divide, and differentiate into osteoblasts on a 2D grid. At hourly intervals,
the model tracks alkaline phosphatase (ALP), collagen deposition, and
hydroxyapatite-like mineral accumulation.

- [Paper — A cellular automaton model of osteogenic differentiation reveals identifiability limits of endpoint assays](https://doi.org/10.3389/fsysb.2026.1872774)
- [Experimental data — DataverseNO](https://doi.org/10.18710/FWDTNY)
- [Simulation code — `main.py`](./main.py) · [Model source — `src/`](./src/)

## What the study found

Different simulated trajectories of osteogenic differentiation and matrix
accumulation can produce similar measurements at the assay endpoints.
This limits what endpoint ALP, collagen, and mineralization measurements
can tell us about when the underlying processes occurred.

## What you can do with this repository

- Run and compare simulated osteogenic culture protocols with `main.py`.
- Analyze ALP, collagen, Alizarin Red S (ARS), and DAPI data with `endpointaad/`.
- Fit model parameters and dose–response terms with `parameterfit/`.

The model lets you vary the timing of osteogenic induction and medium
changes, ascorbic acid decay, and the effects of dexamethasone (Dex) and
menaquinone-4 (MK-4) on individual assay readouts.

## Repository layout

```text
.
├── main.py
├── src/
├── endpointaad/
├── parameterfit/
├── sanity/
├── result_report.md
└── README.md
```

### Folder guide

- `src/`: core simulation code (ABM engine, field manager, configuration, visualization, I/O).
- `endpointaad/`: endpoint analysis pipelines (ALP, collagen, ARS, DAPI, overlays).
- `parameterfit/`: parameter estimation scripts, fitted outputs, and metadata.
- `sanity/`: sanity/regression scenarios for model-change checks.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Default osteogenic run (from `main.py`)

The true defaults are already configured in `main.py`, so the default run is simply:

```bash
python main.py
```

For reproducibility, the expanded equivalent defaults are:

```bash
python main.py \
  --out main_out \
  --seed 12345 \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --w 33 \
  --h 33 \
  --boundary-mode periodic \
  --update-order random \
  --media-interval 48 \
  --dex-conc 0.0 \
  --mk4-conc 0.0 \
  --alp-rb 0.0949 \
  --alp-ro 0.810 \
  --alp-p0 0.8 \
  --alp-dex-ref 2.51535 \
  --collagen-dex-ref 1.05263 \
  --ars-dex-ref 2.85714 \
  --alp-mk4-ref 0.78289 \
  --collagen-mk4-ref 0.347951 \
  --ars-mk4-ref 0.669239 \
  --collagen-rb 0.00336 \
  --collagen-ro 0.0130 \
  --aa0 50.0 \
  --aa-half-life 8.0 \
  --collagen-aa-km 5.0 \
  --ha-model ha \
  --ha-ro 0.102 \
  --ha-h0 2.34 \
  --ha-tau 384 \
  --ha-hill-n 2 \
  --ha-collagen-maturation-tau 48 \
  --ha-collagen-km 1.0 \
  --random-move \
  --move-prob 0.27 \
  --max-moves 1 \
  --proliferation \
  --ob-div-prob 0.0 \
  --min-div-hours 38 \
  --max-div-hours 52 \
  --osteogenic-commitment \
  --osteo-induction-hours 72 \
  --commit-delay 48 \
  --commit-k 0.0125 \
  --apoptosis \
  --apoptosis-prob 0.0010 \
  --apoptosis-osteo 0.0 \
  --secretion \
  --spatial-every 24 \
  --xlsx \
  --uniqueID \
  --snapshots \
  --no-hourly-exports \
  --alp-snapshots \
  --runtime \
  --runtime-every 1 \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672
```

## Run outputs

A run writes to `--out` (default `main_out/`). Core artifacts include:

- `run.json`: complete runtime metadata and parameter configuration.
- `timeseries.xlsx`: daily simulation summary metrics.
- `cell_history.xlsx`: per-cell history export (when unique-ID tracking is enabled).
- figure panel (`*.png` + `figures_pdf/*.pdf`): counts, events, motility, field totals, modulation summary.
- spatial assets: daily `.npy` grids, collagen/HA overlays, and ALP snapshots (depending on export flags).

## Quick validation

To verify that major model pathways remain stable after changes:

```bash
python sanity/sanity_check1.py
```

## Parameter fitting and endpoint analysis

- `parameterfit/` contains scripts and fitted outputs for rate calibration and Dex/MK-4 endpoint modifiers.
- `endpointaad/` contains endpoint-focused pipelines for ALP, collagen, ARS, DAPI, and overlay analyses.

These modules are separate from runtime simulation execution and support model calibration and experimental comparison workflows.


