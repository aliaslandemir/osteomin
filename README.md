# 🧬 **Osteogenic Potential of MC3T3-E1 (osteomin)**
*A 2-D, object-oriented simulator of MC3T3-E1 osteogenic differentiation on coverslips*

---

## 📖 Overview
**osteomin** is a lightweight agent-based + field simulator for **bone biomineralization** studies.  
It models **MC3T3-E1 pre-osteoblasts** on **2-D coverslips** in osteogenic medium containing:

- 🍊 **Ascorbic Acid (AA)** — stimulates collagen deposition  
- 🧴 **β-Glycerophosphate (β-GP)** — phosphate reservoir (ALP-hydrolyzed)  
- 💊 **Dexamethasone (Dexa)** — biases osteogenic commitment & ALP secretion  
- 🟨 **Vitamin K2 (MK-4)** — accelerates **ACP → HA** maturation

Supplements are **present from day 0-2**, **consumed** as cells use them, and **replenished** on each **media change**.

---

## ✨ Features
- 📐 **Object-oriented design** — clear modules in `src/` (ABM, PDE/kinetics, I/O, viz)
- 🧫 **ABM** — proliferation, migration, apoptosis, osteoblast commitment
- 🔬 **Local kinetics** — β-GP → PO₄³⁻ (ALP), collagen-guided ACP nucleation, MK-4-scaled HA maturation
- 🧪 **Supplement dynamics** — AA, Dexa, MK-4 **deplete with use** and **reset** on media change
- 📊 **Outputs**
  - `results.xlsx` (when `--xlsx` is set) — daily metrics (ions, ECM/mineral, cell states, supplements)
  - PNG time-series: `01_states_counts.png`, `02_events_cumulative.png`, `03_matrix_biochem.png`, `04_events_daily.png`
  - Spatial panels for selected days: `spatial_day_###.png`
  - Raw grids (for overlays): `out/grids/state_day_###.npy`, `deaths_day_###.npy` (and optionally split masks)

---

## 📂 Repository Structure
```
osteomin/
├─ main.py                 # CLI entry point
├─ requirements.txt
├─ README.md
└─ src/
   ├─ agent.py             # Cell object & state enum
   ├─ abm_manager.py       # Cell behavior logic (prolif/move/commit/apoptosis)
   ├─ pde_manager.py       # Local chemistry/kinetics (no spatial diffusion)
   ├─ model.py             # ABM↔fields coupling, supplements, run loop
   ├─ config.py            # Tunable parameters
   ├─ io_utils.py          # Excel/CSV writers + optional snapshots
   └─ viz.py               # Matplotlib summary & per-day panel plotting
```

---

## ⚙️ Command-line Options
| Flag              | Default      | Description                                                       |
|-------------------|--------------|-------------------------------------------------------------------|
| `--w`, `--h`      | 120          | Grid width & height                                               |
| `--days`          | 21           | Total simulation days                                             |
| `--cells`         | 300          | Initial cell count                                                |
| `--aa`            | 1.0          | Initial AA level [0–1]                                            |
| `--bgp`           | 1.0          | Initial β-GP reservoir level [0–1]                                 |
| `--dexa`          | 1.0          | Initial Dexa level [0–1]                                          |
| `--mk4`           | 1.0          | Initial MK-4 level [0–1]                                          |
| `--media-interval`| 2            | Days between medium changes                                       |
| `--cell-movement` | (off)        | Enable random cell movement                                       |
| `--osteogenic-commitment`| (off) | Enable osteoblast commitment                                      |
| `--apoptosis`     | (off)        | Enable apoptosis                                                  |
| `--proliferation` | (off)        | Enable proliferation                                              |
| `--secretion`     | (off)        | Enable ALP secretion & ECM deposition                              |
| `--xlsx`          | (off)        | Write results.xlsx                                                |
| `--csv`           | (off)        | Also write results.csv                                            |
| `--snapshots`     | (off)        | Save per-day grids and PNG panels                                 |
| `--out`           | results      | Output directory                                                  |

---

## ▶ How to Run

### PowerShell (Windows)
```powershell
 python main.py `
   --days 21 `
   --cells 300 `
   --aa 1.0 `
   --bgp 1.0 `
   --dexa 1.0 `
   --mk4 1.0 `
   --media-interval 2 `
   --cell-movement `
   --osteogenic-commitment `
   --apoptosis `
   --proliferation `
   --secretion `
   --xlsx `
   --snapshots `
   --out out
```
*(Use backticks ` for line continuation in PowerShell.)*

### macOS/Linux
```bash
 python main.py --days 21 --cells 300 --aa 1.0 --bgp 1.0 --dexa 1.0 --mk4 1.0 --media-interval 2 --snapshots --xlsx
```

---

## 📊 Outputs
| File / Folder  | Description                                                        |
|----------------|------------------------------------------------------------------|
| results.xlsx   | Daily metrics for easy plotting                                   |
| results.csv    | (Optional) CSV copy if --csv is specified                         |
| 01_states_counts.png, etc.| Matplotlib time-series summaries                        |
| spatial_day_###.png | Spatial panels with split-death overlays                      |
| grids/         | Raw NumPy grids for post-processing                               |

---

## 🧪 Biological Assumptions
- Supplements start at **day 0**, decrease with cellular use, and reset on media change.  
- β-GP is hydrolyzed by ALP to release PO₄³⁻.  
- Dexa biases osteoblast commitment.  
- ACP nucleates on collagen scaffold and matures to HA accelerated by MK-4.  
