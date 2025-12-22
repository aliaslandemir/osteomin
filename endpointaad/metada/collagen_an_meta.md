### `collagen_analysis.py` — Collagen vs DAPI Normalization & Violin Plots
Quantifies collagen burden relative to nuclei on confocal fields, applies matched QC filters, and renders publication-ready violins with annotated means.

**Inputs:** `--input` folder (e.g., `input_images/*.tif(f)` stacks). DAPI channel index `0`, collagen index `2` (editable near the top). Condition metadata comes from filename prefixes (`C1_…C8_`) which map to the Basal/Osteo ± Dex/MK-4 panel used in the other assays.

**Image normalization:** Each channel undergoes Gaussian background subtraction, percentile clipping (`1–99 %`), and scaling to the `95th` percentile. This minimizes exposure differences before segmentation or ratio measurements.

**Nuclei segmentation:** Runs Otsu thresholding on the normalized DAPI image, removes small artifacts, then optionally watersheds (`H_MAXIMA = 0.5`) to split touching nuclei. Areas are filtered via `MIN_NUC_AREA_UM2`/`MAX_NUC_AREA_UM2` (converted to pixels with `PIXEL_SIZE_UM = 0.21`), producing final masks plus per-image nuclei counts and confluence.

**Derived metrics (per image):**
- `collagen_per_cell_norm` — total normalized collagen signal / nuclei count.
- `ecm_density_outside_norm` — collagen intensity outside the nuclear mask, divided by its area.
- `collagen_over_dapi_norm` — whole-field collagen / DAPI ratio.

**Outputs (`--out`, default `collagen_norm_output/`):**
- `image_metrics.csv` — rows per image, columns for the metrics above, condition label, nuclei count, and confluence.
- `figs/violin_<metric>.pdf` — violin plots in the unified house style (amber violins, blue jitter points, teal diamonds for means with SD + 95 % CI bars, teal mean text labels offset to the right, mathtext two-line x-labels).
- `figs/qc_scatter_<metric>_vs_nuclei.pdf` — scatter QC panels showing each metric versus nuclei count for visualizing sampling consistency.

**Run:**  
```bash
python collagen_analysis.py \
    --input input_images \
    --out collagen_out
```
Edit the constants at the top (channel indices, percentile cutoffs, watershed toggle, or size thresholds) before re-running if your acquisition settings differ.
