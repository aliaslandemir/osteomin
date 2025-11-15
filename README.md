# OsteoMin – On-Lattice ABM with Spatial Field Manager

## Project Structure
```
.
├── main.py
├── README.md
├── requirements.txt
├── out/                # default output directory
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
| `--cells` | 1000 |
| `--w`, `--h` | 100 each |
| `--boundary-mode` | `periodic` |
| `--media-interval` | 2 |
| `--seed` | None (random) |

### Supplements
| Flag | Default |
|------|---------|
| `--aa` | 1.0 |
| `--bgp` | 1.0 |
| `--mk4` | 1.0 |
| `--dexa` | 1.0 |

### Cell behaviors
| Behavior | Default |
|----------|---------|
| Random movement | enabled (`--random-move`), `--move-prob 1.0`, `--max-moves 1` |
| Proliferation | enabled (`--proliferation`), `--proliferation-chance 1.0`, `--ob-div-prob 0.0` |
| Apoptosis | enabled (`--apoptosis`), `--apoptosis-prob 0.01`, `--apoptosis-osteo 0.0005` |
| Osteogenic commitment | enabled, `--commit-growth 0.02`, `--commit-delay 3` |
| Secretion | enabled (`--secretion`) |

### Field / chemistry
| Parameter | Default |
|-----------|---------|
| `--s-thresh` | 0.01 (Ca×Pi supersaturation for ACP) |
| `--k-nuc` | 0.08 (ACP nucleation rate) |
| `--k-ha` | 0.04 (ACP→HA conversion rate) |
| `--nucleation` | enabled |
| `--nodule-window` | 5 (lattice cells per side) |
| Nodule logic | every site in the window must be OB with HA ≥ `ACP_TO_HA_THRESHOLD` (0.20 a.u.) |

### Output
| Flag | Default |
|------|---------|
| `--out` | `out` |
| `--spatial-every` | 1 (daily overlays) |
| `--spatial-days` | `` (auto adds last day) |
| `--snapshots` | on |
| `--xlsx` | on (writes `out/timeseries.xlsx`) |
| `--csv` | off |

---

## Recommended Run (matches defaults)
```bash
python main.py \
  --days 30 \
  --cells 1000 \
  --w 100 --h 100 \
  --boundary-mode periodic \
  --seed $(shuf -i 1-999999 -n 1) \
  --media-interval 2 \
  --aa 1.0 --bgp 1.0 --mk4 1.0 --dexa 1.0 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --proliferation-chance 1.0 --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.01 --apoptosis-osteo 0.0005 \
  --osteogenic-commitment --commit-growth 0.02 --commit-delay 3 \
  --secretion --nucleation --enable-nodules \
  --nodule-window 5 --k-nuc 0.08 --k-ha 0.04 --s-thresh 0.01 \
  --spatial-every 1 --snapshots --xlsx --out out
```

This command mirrors the baked-in defaults; feel free to omit flags you are not changing.

---

## Key CLI Controls

- **Geometry & time:** `--days`, `--cells`, `--w`, `--h`, `--boundary-mode`, `--media-interval`, `--seed`.
- **Supplements:** `--aa`, `--bgp`, `--mk4`, `--dexa` seed the media pools; they deplete via collagen/ALP rules in `field_manager.py`.
- **Cell events:** toggles for movement, proliferation, apoptosis, secretion, and commitment determine whether the ABM executes the corresponding steps each day. Per-process probabilities are adjustable via the associated flags.
- **Nucleation & nodules:** `--s-thresh`, `--k-nuc`, `--k-ha`, `--nodule-window`, and `--enable-nodules` control ACP/HA kinetics and how nodules are detected. A nodule is counted if every site in the window is an osteoblast with HA ≥ `ACP_TO_HA_THRESHOLD`.
- **Exports:** `--spatial-every`, `--spatial-days`, `--snapshots`, `--xlsx`, `--csv`, and `--out` cover visualization, raw grid dumps, and timeseries formats.

---

## Notes
- The entire chemistry configuration is centralized in `FieldConfig` (`src/config.py`), so any parameter you expose via CLI will propagate through `main.py` → `FieldConfig` → `FieldManager` without local edits.
- Overlay images (`out/collagen_ha/collagen_ha_day_XXX.png`) show collagen (green), HA (blue), and min(Collagen,HA) (red). Purple stars mark cumulative nodule centers derived from `out/grids/nodules_day_XXX.npz`.
- Timeseries (`out/timeseries.xlsx`) always includes a `nodule_count` column (0 when nodule detection is disabled).

Enjoy experimenting with the osteoblast ABM! Feel free to open issues or PRs if you need additional CLI knobs or output formats.

