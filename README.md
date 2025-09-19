# OsteoMin – Agent-Based & PDE Bone Remodeling Simulator

## 📂 Project Structure

```
.
├── main.py           # Entry point (run from here)
├── README.md         # This file
├── requirements.txt  # Dependencies (include openpyxl, numpy, matplotlib, etc.)
├── out/              # Default output folder
└── src/              # Python package
    ├── __init__.py
    ├── abm_manager.py
    ├── agent.py
    ├── config.py
    ├── io_utils.py
    ├── model.py
    ├── pde_manager.py
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
| output dir (`--out`) | out                                                                 |
| days                 | 21                                                                  |
| media interval       | 2                                                                   |
| switches             | movement ✅, commitment ✅, apoptosis ✅, proliferation ✅, secretion ✅ |
| output filename      | timeseries.xlsx                                                     |

---

## 🧪 Example Runs

### PowerShell Positive Control (Windows)

```powershell
python main.py `
   --days 21 `
   --cells 300 `
   --w 100 `
   --h 100 `
   --media-interval 2 `
   --seed $(Get-Random) `
   --aa 1.0 `
   --bgp 1.0 `
   --dexa 1.0 `
   --mk4 1.0 `
   --cell-movement `
   --osteogenic-commitment `
   --apoptosis `
   --proliferation `
   --secretion `
   --move-prob 0.5 `
   --max-moves 1 `
   --ob-div-prob 0.0 `
   --spatial-every 5 `
   --snapshots `
   --xlsx `
   --out out
```

### PowerShell Negative Control (Windows)

```powershell
python main.py `
   --days 21 `
   --cells 300 `
   --w 100 `
   --h 100 `
   --media-interval 2 `
   --seed $(Get-Random) `
   --aa 1.0 `
   --bgp 1.0 `
   --dexa 1.0 `
   --mk4 1.0 `
   --no-cell-movement `
   --no-osteogenic-commitment `
   --no-apoptosis `
   --no-proliferation `
   --no-secretion `
   --move-prob 0.5 `
   --max-moves 1 `
   --ob-div-prob 0.0 `
   --spatial-every 5 `
   --snapshots `
   --xlsx `
   --out out
```

---

## 📊 Output

* `out/timeseries.xlsx` – Time-series metrics.
* PNG snapshots (if `--snapshots`) under `out/`.
* Spatial heatmaps at intervals specified by `--spatial-every`.

---

## 🧬 Notes on Biology Consistency

* Supplements (AA, β-GP, Dexa, MK-4) currently stay constant between media changes. Implement depletion if needed.
* Dexa only nudges Pi in code—extend to influence commitment probability and ALP secretion for closer biological realism.
* MK-4 acts as a Pi sink proxy; to fully model ACP→HA conversion, add explicit ACP/HA kinetics or update docs.

