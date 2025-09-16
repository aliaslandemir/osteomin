# 🧬 **Osteogenic Potential of MC3T3-E1 (osteomin)**
*A 2-D, object-oriented simulator of MC3T3-E1 osteogenic differentiation on coverslips*


---

## 📖 Overview
**osteomin** is a lightweight, agent-based + field simulator for **bone biomineralization** studies.  
It models **MC3T3-E1 pre-osteoblasts** on **2-D coverslips** in osteogenic medium containing:

- 🍊 **Ascorbic Acid (AA)** — stimulates collagen deposition  
- 🧴 **β-Glycerophosphate (β-GP)** — phosphate reservoir  
- 💊 **Dexamethasone (Dexa)** — promotes osteoblast commitment & ALP secretion  
- 🟨 **Vitamin K2 (MK-4)** — accelerates **ACP → HA** maturation  

Supplements are **present from day 0**, **consumed** as cells use them, and **replenished** at each medium change.  

---

## ✨ Features
- 📐 **Object-oriented design** — clean modules (`src/`): ABM, PDE, I/O, viz  
- 🧫 **Agent-Based Model** — proliferation (early), migration, apoptosis, osteoblast commitment  
- 🔬 **Field kinetics** — Ca²⁺/PO₄³⁻ chemistry, β-GP hydrolysis (ALP), ACP nucleation, HA maturation (MK-4 scaled)  
- 🧪 **Supplement dynamics** — AA, Dexa, MK-4 **deplete with use** and **reset on media change**  
- 📊 **Outputs**
  - `results.xlsx` (+ optional `results.csv`) — daily metrics (ions, ECM/mineral, nodules, cell counts, supplement levels)
  - `summary.png` — 3×3 time-series figure
  - `panels_pub/` — per-day PNGs when `--snapshots` is enabled

---

## 📂 Repository Structure
```
osteomin/
├─ main_fast.py          # CLI entry point
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ README.md
└─ src/
   ├─ agent.py           # Cell object & state enum
   ├─ abm_manager.py     # Cell behavior logic
   ├─ pde_manager.py     # Local chemistry/kinetics (no diffusion)
   ├─ model.py           # ABM↔PDE coupling, supplements, run loop
   ├─ config.py          # Tunable parameters
   ├─ io_utils.py        # Excel/CSV writers + optional snapshots
   └─ viz.py             # Summary & per-day panel plotting
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
| `--snapshots`     | (off)        | Save arrays/CSV each week & final day; enables per-day PNG panels |
| `--csv`           | (off)        | Also write results.csv alongside results.xlsx                     |
| `--snapdir`       | snapshots    | Snapshot directory                                                |
| `--out`           | results.xlsx | Output Excel path                                                 |
| `--seed`          | 42           | RNG seed for reproducibility                                      |

---

## ▶ How to Run

### PowerShell (Windows)
```powershell
python main_fast.py --days 21 --cells 300 --aa 1.0 --bgp 1.0 --dexa 1.0 --mk4 1.0 --media-interval 2 --snapshots --csv
```
*(Use backticks ` for line continuation in PowerShell if splitting lines.)*

### macOS/Linux
```bash
python main_fast.py --days 21 --cells 300 --aa 1.0 --bgp 1.0 --dexa 1.0 --mk4 1.0 --media-interval 2 --snapshots --csv
```

---

## 📊 Outputs
| File / Folder  | Description                                                        |
|----------------|------------------------------------------------------------------|
| results.xlsx   | Daily metrics for easy Excel plotting                             |
| results.csv    | (Optional) CSV copy if --csv is specified                          |
| summary.png    | Multi-panel time-series summary                                   |
| snapshots/     | Raw arrays (ha_DAY.npy, etc.) and cells_DAY.csv per saved day      |
| panels_pub/    | Publication-ready PNGs for chosen days                             |

---

## 🧪 Biological Assumptions
- Supplements start at **day 0**, decrease with cellular use, and reset on media change.  
- β-GP is hydrolyzed by ALP to release PO₄³⁻.  
- Local supersaturation and Dexa bias osteoblast commitment.  
- ACP nucleates on collagen scaffold and matures to HA accelerated by MK-4.  

---

