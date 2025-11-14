# OsteoMin – Agent-Based Pre-Osteoblast Differentiation Model (Field Manager)

## 📂 Project Structure
```
.
├── main.py           # Main control
├── README.md         # Read
├── requirements.txt  # Dependencies
├── out/              # Default output folder
└── src/              # Python handling
    ├── __init__.py
    ├── abm_manager.py
    ├── agent.py
    ├── config.py
    ├── io_utils.py
    ├── model.py
    ├── field_manager.py
    └── viz.py
```

> ✅ **Imports:** `main.py` uses `from src.model import OsteoModel`. Keep modules in `src/` as shown.

---

## ▶️ Quick Start

```bash
# Create venv & install deps
python -m venv .venv
source .venv/bin/activate   # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt

# Run with defaults
python main.py
```

---

## ⚙️ Defaults (match code)

| Parameter            | Default                                                             |
| -------------------- | ------------------------------------------------------------------- |
| width (`--w`)        | 100                                                                 |
| height (`--h`)       | 100                                                                 |
| boundary mode (`--boundary-mode`) | clamp (hard edges; periodic wraps around)               |
| output dir (`--out`) | out                                                                 |
| days                 | 21                                                                  |
| media interval       | 2                                                                   |
| switches             | movement ✅, commitment ✅, apoptosis ✅, proliferation ✅, secretion ✅ |
| commitment delay (`--commit-delay`) | 0 (days before ACTIVE cells are allowed to turn into osteoblasts) |
| output filename      | timeseries.xlsx                                                     |
| apoptosis probability (`--apoptosis-prob`) | 0.01 (per-cell death chance each day)                         |
| proliferation chance (`--proliferation-chance`) | 1.0 (per-day division attempt multiplier after the interval gate) |
| ALP→Pi rate (`--alp-to-pi-rate`) | 0.01 (β-GP limited; Pi₀ = 0 so Pi only appears through ALP) |
| supersaturation threshold (`--s-thresh`) | 0.01 (Ca×Pi required locally to nucleate ACP) |
| nucleation switch (`--nucleation`/`--no-nucleation`) | enabled (Pi powers ACP nucleation and HA maturation) |

## 🛠️ CLI Control Overview

- **Simulation size/time:** `--days`, `--cells`, `--w`, `--h`, `--media-interval`, `--seed` let you set duration, population, lattice dimensions, media cadence, and RNG seed.
- **Supplements:** `--aa`, `--bgp`, `--mk4`, `--dexa` seed the media pools. AA is spent directly to make collagen while β-GP is consumed when ALP elevates Pi, and both pools refill whenever `--media-interval` triggers a scrub. Dexa and MK-4 remain logged without downstream effects.
  - **Cellular switches:** `--random-move/--no-random-move` (aliases `--cell-movement/--no-cell-movement`), `--osteogenic-commitment/--no-osteogenic-commitment`, `--apoptosis/--no-apoptosis`, `--proliferation/--no-proliferation`, `--secretion/--no-secretion`, plus `--ob-div-prob`, `--proliferation-chance`, `--apoptosis-prob`, and `--apoptosis-osteo` control random_move, fate, death, and division rules.
- **Movement tuning:** `--move-prob`, `--max-moves` adjust random_move probability/steps per day; both random_move and division now sample the Moore (8-neighbor) grid (axial ~0.0333, diagonal ~0.0667) with a 60 % stay chance before attempting another direction.
- **Boundary handling:** `--boundary-mode clamp|periodic` keeps agents inside the lattice (`clamp`) or wraps neighbors and daughters across opposite edges (`periodic`).
- **Mineralization knobs:** `--s-thresh`, `--k-nuc`, `--k-ha`, and the new `--nucleation/--no-nucleation` switch control supersaturation, formation kinetics, and whether Pi is allowed to create ACP/HA; the field manager no longer enforces a collagen threshold or neighborhood dilation.
- **Commitment tuning:** `--commit-delay` pauses ACTIVE→OB differentiation for the first N days (default 0) so early expansion stays in the proliferative/active regime; once the delay passes the usual `base_commit_prob + day * commit_prob_growth` roller coaster resumes, letting you space out the differentiation wave without touching the base coefficient.
- **Output/export:** `--spatial-every`, `--spatial-days`, `--out`, `--xlsx`, `--csv`, `--snapshots` configure PNG/Excel/CSV exports and snapshot cadence.

---

## 🧪 Example Runs

### Linux / WSL Positive Control

```bash
python main.py \
  # --- Simulation setup ---
  --days 29 \                     # total simulation duration
  --cells 700 \                   # initial number of cells
  --w 100 --h 100 \               # grid size (width × height)
  --boundary-mode periodic \      # periodic boundary conditions
  --seed $(shuf -i 1-999999 -n 1) # random seed for reproducibility
  --media-interval 2 \            # media refresh interval (days)

  # --- Biochemical environment ---
  --aa 1.0 \                      # ascorbic acid level
  --bgp 1.0 \                     # β-glycerophosphate level
  --mk4 1.0 \                     # vitamin K2 (MK-4) level
  --dexa 1.0 \                    # dexamethasone level

  # --- Cell behaviors ---
  --random-move \                 # enable random cell movement
  --move-prob 1.0 \               # movement probability per day
  --max-moves 1 \                 # max steps per cell per day
  --proliferation \               # enable cell proliferation
  --proliferation-chance 1.0 \    # daily proliferation chance
  --ob-div-prob 0.0 \             # osteoblast division probability
  --apoptosis \                   # enable apoptosis
  --apoptosis-prob 0.01 \         # base apoptosis probability
  --apoptosis-osteo 0.0005 \      # apoptosis for osteoblasts
  --osteogenic-commitment \       # enable OB differentiation
  --commit-growth 0.02 \          # commitment growth rate
  --secretion \                   # enable matrix secretion

  # --- Output and logging ---
  --spatial-every 1 \             # save spatial maps every day
  --snapshots \                   # store visual snapshots
  --xlsx \                        # export summary data to Excel
  --out out                       # output directory


```


### Linux / WSL Negative Control

```bash
python main.py \
   --days 21 \
   --cells 300 \
   --w 100 \
   --h 100 \
   --media-interval 2 \
   --seed $(shuf -i 1-999999 -n 1) \
   --aa 1.0 \
   --bgp 1.0 \
   --mk4 1.0 \
   --no-random-move \
   --no-osteogenic-commitment \
   --no-apoptosis \
   --no-proliferation \
   --no-secretion \
   --move-prob 0.5 \
   --max-moves 1 \
   --ob-div-prob 0.0 \
   --apoptosis-prob 0.01 \
   --spatial-every 5 \
   --snapshots \
   --xlsx \
   --out out
```

---

## 📊 Output

* `out/timeseries.xlsx` – Time-series metrics.
* PNG snapshots (if `--snapshots`) under `out/`.
* Spatial heatmaps at intervals specified by `--spatial-every`.

### Figures & Visualization
- Exports (300 DPI, tight margins for papers):
  - `01_states_counts.png` – Active/OB/Total counts vs. day
  - `02_events_cumulative.png` – Cumulative deaths by class
  - `03_matrix_biochem.png` – Free Ca/Pi (left axis) plus cumulative Ca/Pi usage (right axis)
  - `04_events_daily.png` – Daily commitments and deaths
  - `05_supplements.png` – Reported supplement scaling factors (AA, β-GP, MK-4, Dexa)
  - `spatial_cells/spatial_day_XXX.png` – Cell-state map at day `XXX`
  - `collagen_ha/collagen_ha_day_XXX.png` – Collagen–HA overlay at day `XXX`
  - Color scheme (colorblind‑friendly):
  - Active `#0072B2` (blue), Osteoblast `#D55E00` (vermillion)
  - Died while active `#E69F00`, died while osteoblast `#CC79A7`
  - Collagen overlay uses `magma`; HA overlay uses `Blues`; both alpha‑scaled with robust p99 normalization.
- Layout notes:
  - Spatial maps and collagen–HA overlays are square and fill the figure; side colorbars are narrow and height‑matched for consistent subfigure sizing.
  - All daily spatial frames live in `out/spatial_cells/`, and collagen/HA overlays live in `out/collagen_ha/` to keep the root folder tidy.
  - Default overlays are exported every 5 days and on the last day of the run (with `--snapshots` enabled); spatial maps follow `--spatial-every` or explicit `--spatial-days`, and every overlay now uses the global collagen/HA range so early days preserve the same color scale.

---

## 🧬 Notes on Biology Consistency

* AA and β-GP pools now deplete between media changes in proportion to collagen/ALP activity, then reset when media is refreshed. Tunable `--aa`, `--bgp`, or `--media-interval` let you probe supply-limited versus saturation regimes.
* Fresh media contains Ca (`CA0 = 1`) but no inorganic phosphate (`PI0 = 0`). Pi is only generated when ALP cleaves β-GP, and ACP nucleation requires the local product Ca×Pi to exceed `S_THRESH`, so mineralization truly depends on both living osteoblasts and supplement supply.
* Calcium is pulled from the media and logged via `ca_used_total` (with matching `pi_used_total`), so `03_matrix_biochem.png` now shows both the free ionic reservoir and how much has been incorporated into ACP/HA.
* Osteoblast apoptosis now uses `--apoptosis-osteo` (default 0.001), so committed cells die ten times less frequently than ACTIVEs and stay longer to build matrix.
* Dexa and MK-4 are recorded but still inert; hook them into commitment or secretion logic if you need downstream pharmacology.
* Overlays require `--snapshots` so the raw grids exist; without snapshots the `collagen_ha/collagen_ha_day_*` exports will be skipped and only the time series/spatial maps are written.
