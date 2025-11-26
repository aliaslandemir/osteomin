### `alp_analysis.py` — ALP Quantification and Visualization
Processes **alkaline phosphatase (ALP)** absorbance data from a 24-well plate to compute activity and generate plots.

**Input:** `alp_raw_data.xlsx` — rows **B–D** are samples; row **D** includes controls (`pNPP+lysate`, `Lysate`). Experimental groups: *Basal*, *Osteo*, and combinatorial **Dex/MK-4** doses.

**Blank & Calibration:** subtracts assay blank (default `pNPP+lysate = 0.006 AU`), fits an origin-constrained line through three **CIP standards** (0.00005–0.00020 U/well), and reports slope (`net A = m × U`) and **R²**. (not added into final plot)

**Quantification:** converts absorbance to **activity (mU/mL)**, computes *n, mean, median, SD, SEM, 95 % CI*, and formats two-line condition labels (e.g. `A+β-GP` / `D_HK_L`).

**Visualization:** generates (1) a dual-panel figure (CIP calibration + sample violins) and (2) a standalone publication panel (violins only). Amber violins show distributions; blue dots = replicates; black ▢ = median; teal ◆ = mean ± SD ± 95 % CI; numeric means annotated.

**Outputs (→ `results/figures/`):**
- `ALP_dualplot_pub.png/svg` — calibration + activity violins  
- `ALP_activity_only_pub.png/svg` — publication-ready violins with legend “blank corrected = −(pNPP+lysate 0.006)”

**Run:**  
```bash
python alp_analysis.py
