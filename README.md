# OsteoMin – On-Lattice ABM with Spatial Field Manager

## Project Structure
```
.
├── main.py             # default run, CLI ...
├── README.md           # default run arguments summary
├── requirements.txt    # freezed on 27.11.25
├── endpointanalysis/   # scripts used to process endpoint assays
├── out/                # default output directory for the simulation
└── src/
    ├── __init__.py
    ├── abm_manager.py
    ├── agent.py
    ├── config.py
    ├── field_manager.py
    ├── io_utils.py
    ├── model.py
    └── viz.py
```

`main.py` imports `OsteoModel` from `src/model.py`, which wires the ABM (`abm_manager.py`) to the chemistry/field layer (`field_manager.py`). All module code lives under `src/`.

---

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# run with defaults (see below)
python main.py
```

---

## Default CLI / Config Values

### Core simulation
| Flag | Default |
|------|---------|
| `--days` | 30 |
| `--cells` | 72 |
| `--w`, `--h` | 33 each |
| `--boundary-mode` | `periodic` |
| `--media-interval` | 2 |
| `--seed` | None (random) |

### Cell behaviors
| Behavior | Default |
|----------|---------|
| Random movement | enabled (`--random-move`), `--move-prob 1.0`, `--max-moves 1` |
| Proliferation | enabled (`--proliferation`), `--proliferation-chance 1.0`, `--ob-div-prob 0.0` |
| Apoptosis | enabled (`--apoptosis`), `--apoptosis-prob 0.01`, `--apoptosis-osteo 0.001` |
| Osteogenic commitment | enabled, `--commit-growth 0.01`, `--commit-delay 3` |
| Secretion | enabled (`--secretion`) |
| Division interval | `--min-div-hours 38`, `--max-div-hours 52` |

### Data-driven inputs
| Flag | Default |
|------|---------|
| `--datafit` | `datafitting/datafit.npz` (hourly basal/osteo ALP, collagen, mineral curves) |

### Output
| Flag | Default |
|------|---------|
| `--out` | `out` |
| `--spatial-every` | 1 (daily overlays) |
| `--spatial-days` | `` (auto adds last day) |
| `--snapshots` | on (`--no-snapshots` disables raw grids) |
| `--xlsx` | on (`--no-xlsx` skips Excel export) |
| `--csv` | off |
| `--uniqueID` | on (`--no-unique-id` skips per-cell history export) |

### Plotting windows
| Flag | Default |
|------|---------|
| `--alp-plot-hours` | 240 |
| `--collagen-plot-hours` | 336 |
| `--mineral-plot-hours` | 672 |

---

## Recommended Run (example)
```bash
python main.py \
  --hours 720 \
  --cells 72 \
  --w 33 --h 33 \
  --boundary-mode periodic \
  --seed $(shuf -i 1-999999 -n 1) \
  --media-interval 48 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 --apoptosis-osteo 0.000042 \
  --osteogenic-commitment --commit-growth 0.00001736 --commit-delay 72 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion \
  --spatial-every 1 --out out \
  --datafit datafitting/datafit.npz \
  --no-hourly-exports
```

This command matches the configuration used in the paper; feel free to omit flags you are not changing. Snapshots, Excel, and unique-cell history exports still run by default, so add the `--no-*` variants only when you want to turn them off.

---

## Key CLI Controls

- **Geometry & time:** `--days`, `--cells`, `--w`, `--h`, `--boundary-mode`, `--media-interval`, `--seed`.
- **Datafit forcing:** `--datafit` points to `datafitting/datafit.npz`; preOB use basal curves, OB use osteogenic curves, and hourly increments are blended by the live mix.
- **Cell events:** toggles for movement, proliferation, apoptosis, secretion, and commitment determine whether the ABM executes the corresponding steps each day. Per-process probabilities are adjustable via the associated flags.
- All per-day flags listed above are internally scaled to hourly values so the CLI stays familiar even though the solver now runs on 1-hour ticks.

---

## Time resolution

- Simulation now executes 24 hourly updates for each reported day so that the ABM and chemistry layers evolve continuously while snapshots and exports remain keyed to day indices `0..N-1`.
- The CLI still describes parameters in day-based units, but they are converted to internal hourly rates before each solver tick (commitment growth, per-day probabilities, media interval, etc.).
- Hourly spatial snapshots are saved inside `out/day_hor/dayX/` (e.g., `day_hor/day1/spatial_cells/hour_00.pdf` and `day_hor/day1/collagen_ha/hour_00.pdf`), giving per-hour PDFs in addition to the daily overlays and grids.
- Enable hourly exports with `--hourly-exports` (and disable them with `--no-hourly-exports`) if you want those per-hour PDFs; the default is off to keep output size manageable.

## Notes
- The entire chemistry configuration is centralized in `FieldConfig` (`src/config.py`), so any parameter you expose via CLI will propagate through `main.py` → `FieldConfig` → `FieldManager` without local edits.
- Overlay images (`out/collagen_ha/collagen_ha_day_XXX.pdf`) show collagen (yellow-green) and HA (magenta/blue), with live-cell/death stats in the header.

Enjoy experimenting with the osteoblast ABM! Feel free to open issues or PRs if you need additional CLI knobs or output formats.

## Endpoint analysis

The `endpointanalysis/` folder contains the scripts necessary to run the endpoint analyses referenced in the report; run them from this repository to process simulation outputs and reproduce the figures and metrics.
