# OsteoMin

OsteoMin is a 2D lattice-based simulator for osteogenic culture dynamics. It couples an on-lattice agent-based model (ABM) of cell behavior with field-level accumulation of ALP, collagen, and hydroxyapatite-like mineral output.

The repository is designed for three complementary tasks:

- running osteogenic simulations (`main.py`)
- analyzing endpoint assay datasets (`endpointaad/`)
- fitting model parameters and dose-response terms (`parameterfit/`)

## Scientific scope

OsteoMin represents osteogenic progression at hourly resolution on a fixed lattice.

- Cell dynamics: pre-osteoblast seeding, motility, proliferation, commitment to osteoblast state, and apoptosis.
- Matrix dynamics: ALP and collagen deposition plus mineral accumulation through the HA model.
- Environmental control: osteogenic induction timing, media refresh interval, ascorbic acid (AA) decay, and endpoint-specific Dex/MK-4 modulation.

The model is intended for comparative in silico experiments under controlled protocol-like conditions rather than direct patient-level prediction.

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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Default osteogenic run (public reference)

Use the following run as the default osteogenic reference condition:

```bash
python main.py \
  --out main_out \
  --seed 12345 \
  --runtime \
  \
  --hours 672 \
  --cells 72 \
  \
  --w 33 \
  --h 33 \
  --boundary-mode periodic \
  --seed-mode random \
  --nodules 40n \
  \
  --media-interval 48 \
  --secretion \
  --aa0 50.0 \
  --aa-half-life 8 \
  --collagen-aa-km 5.0 \
  \
  --random-move \
  --move-prob 0.27 \
  --max-moves 1 \
  --proliferation \
  --ob-div-prob 0.0 \
  --min-div-hours 38 \
  --max-div-hours 52 \
  --apoptosis \
  --apoptosis-prob 0.0010 \
  --apoptosis-osteo 0.0 \
  \
  --osteogenic-commitment \
  --osteo-induction-hours 72 \
  --commit-delay 48 \
  --commit-k 0.0125 \
  \
  --dex-conc 0 \
  --mk4-conc 0 \
  \
  --alp-rb 0.0949 \
  --alp-ro 0.810 \
  --collagen-rb 0.00336 \
  --collagen-ro 0.0130 \
  --ha-ro 0.102 \
  \
  --alp-dex-ref 22.7 \
  --collagen-dex-ref 6.22 \
  --ars-dex-ref 46.6 \
  --alp-a-d 3.72 \
  --alp-b-d 1.75 \
  --collagen-a-d 2.69 \
  --collagen-b-d 2.04 \
  --ha-a-d 1.22 \
  --ha-b-d 0.679 \
  \
  --alp-mk4-ref 0.784 \
  --collagen-mk4-ref 0.349 \
  --ars-mk4-ref 0.667 \
  --alp-a-k 1.10 \
  --alp-b-k 1.06 \
  --collagen-a-k 4.66 \
  --collagen-b-k 2.15 \
  --ha-a-k 4.66 \
  --ha-b-k 2.15 \
  \
  --ha-model ha \
  --ha-collagen-maturation-tau 48 \
  --ha-collagen-km 1.0 \
  --ha-tau 384 \
  --ha-hill-n 2 \
  \
  --spatial-every 24 \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --alp-snapshots
```

## Run outputs

A run writes to `--out` (default `main_out/`). Core artifacts include:

- `run.json`: complete runtime metadata and parameter configuration.
- `timeseries.xlsx`: daily simulation summary metrics.
- `cell_history.xlsx`: per-cell history export (when unique-ID tracking is enabled).
- figure panel (`*.png` + `figures_pdf/*.pdf`): counts, events, motility, field totals, nodules, modulation summary.
- spatial assets: daily `.npy` grids, collagen/HA overlays, and ALP snapshots (depending on export flags).

## Quick validation

To verify that major model pathways remain stable after changes:

```bash
python3 sanity/sanity_check1.py
```

## Parameter fitting and endpoint analysis

- `parameterfit/` contains scripts and fitted outputs for rate calibration and Dex/MK-4 endpoint modifiers.
- `endpointaad/` contains endpoint-focused pipelines for ALP, collagen, ARS, DAPI, and overlay analyses.

These modules are separate from runtime simulation execution and support model calibration and experimental comparison workflows.
