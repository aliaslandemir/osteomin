# Result Report

This repository currently runs the coupled hourly OsteoMin model with direct CLI-controlled field dynamics rather than relying on fitted `datafit` schedules by default.

## Current reference configuration

Reference runs use:

- hourly updates over `672 h`
- `72` initial cells on a `33 x 33` lattice
- periodic boundaries
- movement enabled with `--move-prob 0.27`
- proliferation enabled
- preOB apoptosis enabled
- osteogenic commitment enabled after induction at day `3` with a `48 h` delay
- direct ALP, collagen, and HAp rate constants from the CLI
- AA pulse-decay control via `--aa0`, `--aa-half-life`, `--collagen-aa-km`

## Field model in use

The current non-legacy workflow is:

- ALP: basal preOB contribution plus osteogenic OB contribution
- Collagen: basal preOB contribution plus AA-gated osteogenic OB contribution
- HAp: osteogenic OB contribution with Hill activation based on post-commitment time

The corresponding controls exposed in `main.py` are:

- `--alp-rb`, `--alp-ro`
- `--collagen-rb`, `--collagen-ro`
- `--aa0`, `--aa-half-life`, `--collagen-aa-km`
- `--ha-ro`, `--ha-tau`, `--ha-hill-n`
- `--dex-conc`, `--mk4-conc`

## Main run products

Each run can produce:

- `run.json`
- `timeseries.xlsx` and/or `timeseries.csv`
- publication-style summary figures in PNG and PDF
- spatial cell maps
- collagen/HAp overlays
- raw grids when snapshots are enabled

## Summary plots currently generated

- `01_states_counts`
- `02_events_cumulative`
- `03_cell_motility`
- `04_events_daily`
- `05_minerals_cumulative`
- `06_alp_total`
- `07_collagen_total`
- `08_nodule_count`
- `09_field_totals_absolute`
- `10_ascorbate_schedule`
- `11_ascorbate_collagen_gate`
- `12_cell_cycle_summary`

## Timeseries fields worth checking

Population and events:

- `n_preob`
- `n_osteoblast`
- `n_total`
- `dead_preob_today`
- `dead_ob_today`
- `committed_today`

Field totals:

- `alp_total`
- `collagen_total`
- `ha_total`
- `alp_preob_total`
- `alp_ob_total`
- `collagen_preob_total`
- `collagen_ob_total`
- `mineral_preob_total`
- `mineral_ob_total`

Additional summary metrics:

- `msd_effective`
- `avg_move_count`
- `avg_division_interval`
- `avg_live_age_hours`
- `nodule_count_current`
- `nodule_count_ever`

## Sanity workflow

`sanity_check1.py` is the first-line regression harness for the current model. It runs a positive control and toggle-specific perturbations, exports CSV summaries, and writes a markdown report under `out/sanity_check1/`.

Use it to confirm:

- cumulative fields remain monotone
- counts remain non-negative
- nodule history is non-decreasing
- disabling key mechanisms produces the expected directional change relative to the positive control
