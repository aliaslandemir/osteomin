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

### Supplements
| Flag | Default |
|------|---------|
| `--aa` | 1.0 |
| `--bgp` | 1.0 |
| `--mk4` | 0.0 (none) |

### Cell behaviors
| Behavior | Default |
|----------|---------|
| Random movement | enabled (`--random-move`), `--move-prob 1.0`, `--max-moves 1` |
| Proliferation | enabled (`--proliferation`), `--proliferation-chance 1.0`, `--ob-div-prob 0.0` |
| Apoptosis | enabled (`--apoptosis`), `--apoptosis-prob 0.01`, `--apoptosis-osteo 0.001` |
| Osteogenic commitment | enabled, `--commit-growth 0.01`, `--commit-delay 3` |
| Secretion | enabled (`--secretion`) |
| Dex effects | off by default (omit `--dex`; strength via `--dex-k`, default 1.0) |

### Field / chemistry
| Parameter | Default |
|-----------|---------|
| `--s-thresh` | 0.01 (Ca×Pi supersaturation for ACP) |
| `--k-nuc` | 0.025 (ACP nucleation rate) |
| `--k-ha` | 0.06 (ACP→HA conversion rate) |
| `--nucleation` | enabled |
| `--enable-nodules` | enabled (runs the sliding-window nodule scanner) |
| `--nodule-window` | 4 (lattice cells per side) |
| Nodule logic | every site in the window must be OB with HA ≥ `ACP_TO_HA_THRESHOLD` (0.20 a.u.) |

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
  --aa 1.0 --bgp 1.0 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 --apoptosis-osteo 0.000042 \
  --osteogenic-commitment --commit-growth 0.00001736 --commit-delay 72 \
  --secretion --nucleation --enable-nodules \
  --spatial-every 1 --out out \
  --nodule-window 4 --k-ha 0.0025 --k-nuc 0.0010417 \
  --no-hourly-exports
```

This command matches the configuration used in the paper; feel free to omit flags you are not changing. Snapshots, Excel, and unique-cell history exports still run by default, so add the `--no-*` variants only when you want to turn them off.

---

## Key CLI Controls

- **Geometry & time:** `--days`, `--cells`, `--w`, `--h`, `--boundary-mode`, `--media-interval`, `--seed`.
- **Supplements:** `--aa`, `--bgp`, optional `--mk4` (0–1) seed the media pools; MK-4 scales collagen rate and boosts `K_NUC`/`K_HA`. Dex is separate (`--dex`, `--dex-k`) and affects commitment/apoptosis only.
- **Cell events:** toggles for movement, proliferation, apoptosis, secretion, and commitment determine whether the ABM executes the corresponding steps each day. Per-process probabilities are adjustable via the associated flags.
- **Nucleation & nodules:** `--s-thresh`, `--k-nuc`, `--k-ha`, `--nodule-window`, and `--enable-nodules` control ACP/HA kinetics and how nodules are detected. A nodule is counted if every site in the window is an osteoblast with HA ≥ `ACP_TO_HA_THRESHOLD`.
-- All per-day flags listed above are internally scaled to hourly values so the CLI stays familiar even though the solver now runs on 1-hour ticks.

---

## Time resolution

- Simulation now executes 24 hourly updates for each reported day so that the ABM and chemistry layers evolve continuously while snapshots and exports remain keyed to day indices `0..N-1`.
- The CLI still describes parameters in day-based units, but they are converted to internal hourly rates before each solver tick (commitment growth, per-day probabilities, media interval, `k_nuc`, `k_ha`, etc.).
- Hourly spatial snapshots are saved inside `out/day_hor/dayX/` (e.g., `day_hor/day1/spatial_cells/hour_00.png` and `day_hor/day1/collagen_ha/hour_00.png`), giving per-hour visualizations in addition to the daily overlays and grids.
- Enable hourly exports with `--hourly-exports` (and disable them with `--no-hourly-exports`) if you want those per-hour PNGs; the default is off to keep output size manageable.

## Notes
- The entire chemistry configuration is centralized in `FieldConfig` (`src/config.py`), so any parameter you expose via CLI will propagate through `main.py` → `FieldConfig` → `FieldManager` without local edits.
- Overlay images (`out/collagen_ha/collagen_ha_day_XXX.png`) show collagen (yellow-green) and HA (magenta/blue). Purple stars mark cumulative nodule centers derived from `out/grids/nodules_day_XXX.npz`.
- Timeseries (`out/timeseries.xlsx`) always includes a `nodule_count` column (0 when nodule detection is disabled).

Enjoy experimenting with the osteoblast ABM! Feel free to open issues or PRs if you need additional CLI knobs or output formats.

## Endpoint analysis

The `endpointanalysis/` folder contains the scripts necessary to run the endpoint analyses referenced in the report; run them from this repository to process simulation outputs and reproduce the figures and metrics.
