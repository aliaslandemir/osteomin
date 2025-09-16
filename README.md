# 🧬 **Osteogenic Potential of MC3T3-E1 Cell Lines**  
*A 2-D, object-oriented for MC3T3-E1 osteogenic differentiation on coverslips*

---

## 📖 Overview
**osteomin** is a lightweight, agent-based + field simulator for **bone biomineralization** experiments.  
It reproduces **MC3T3-E1 pre-osteoblast behavior** on **2-D coverslips** exposed to **osteogenic medium** containing:

- 🍊 **Ascorbic Acid (AA)** — stimulates collagen deposition  
- 🧴 **β-Glycerophosphate (β-GP)** — phosphate reservoir  
- 💊 **Dexamethasone (Dexa)** — promotes osteoblast commitment and ALP secretion  
- 🟨 **Vitamin K2 (MK-4)** — accelerates ACP → HA maturation  

Supplements are **introduced on day 0**, **consumed** as cells use them, and **replenished** at each medium change.  

---

## ✨ Features
- 📐 **Object-oriented design** for easy extension (`src/` modules: ABM, PDE, I/O, viz).  
- 🧫 **Agent-Based Model (ABM)**: MC3T3-E1 cells proliferate, migrate, undergo apoptosis, and commit to osteoblasts.  
- 🌊 **Field kinetics**: Local Ca²⁺/PO₄³⁻ chemistry, β-GP hydrolysis, ACP nucleation, HA maturation.  
- 🧪 **Supplement dynamics**: AA, Dexa, and MK-4 deplete with use and refresh with medium changes.  
- 📊 **Outputs**:  
  - `results.parquet` + `results.csv`: per-day values of all tracked variables  
  - `summary.png`: multi-panel time-series figure  
  - `panels_pub/`: single-panel PNGs for cell states, nodule map, HA, collagen, density  

---

## 📂 Repository Structure
osteomin/
├─ main_fast.py # CLI entry point
├─ requirements.txt
├─ .gitignore
├─ LICENSE
├─ README.md
└─ src/
    ├─ agent.py # Cell object & state enum
    ├─ abm_manager.py # Cell behavior logic
    ├─ pde_manager.py # Local chemistry/kinetics manager
    ├─ model.py # Coupling of ABM + PDE, supplement handling
    ├─ config.py # Tunable parameters
    ├─ io_utils.py # Snapshot & CSV/Parquet writers
    └─ viz.py # Plotting utilities




## ⚙️ Command-line Options
Flag	Default	Description
--w, --h	120	Grid width & height
--days	21	Total simulation days
--cells	300	Initial cell count
--aa	1.0	Initial AA level (0–1)
--bgp	1.0	Initial β-GP reservoir level (0–1)
--dexa	1.0	Initial Dexa level (0–1)
--mk4	1.0	Initial MK-4 level (0–1)
--media-interval	2	Days between medium changes
--snapdir	snapshots	Snapshot directory
--out	results.parquet	Output Parquet path
--seed	42	RNG seed for reproducibility


## How to run?

python main_fast.py --days 21 --cells 300 --aa 1.0 --bgp 1.0 --dexa 1.0 --mk4 1.0 --media-interval 2

