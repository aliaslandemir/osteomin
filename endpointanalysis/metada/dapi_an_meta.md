### `confocal/dapi_analysis.py` — DAPI Nuclei Segmentation & Morphometry
Segments DAPI-stained nuclei, extracts morphology metrics, filters by quality thresholds, and builds publication-ready violin plots with annotated means.

**Inputs:** `input_images/*.tif(f)` volumes (channel index 0 used for DAPI). Condition labels inferred from filename prefixes (`C1_…C8_`) and mapped to Basal/Osteo ± Dex & MK-4 combinations.

**Segmentation pipeline:** Gaussian blur → CLAHE equalization → Otsu thresholding → hole filling → distance-transform markers → watershed → border clearing + small-object removal. Saves uint16 label maps per image (`dapi_analysis_output/labelmaps/*_labels.tif`) plus grayscale overlays with **red contours** for all nuclei.

**Shape features & metrics:** Uses `skimage.measure.regionprops_table` to compute area, perimeter, eccentricity, solidity, axis lengths, etc., then derives `aspect_ratio`, `circularity`, and equivalent diameters. Pixel-scale conversions use `PIXEL_SIZE_UM = 0.21` to report µm and µm² units.

**Post-filters / analyzed nuclei:** Keeps nuclei with `MIN_AREA_UM2 ≤ area ≤ MAX_AREA_UM2` and `MIN_CIRCULARITY ≤ circularity ≤ MAX_CIRCULARITY` (defaults 100–600 µm² and 0.5–1.2). Prints retention stats and stores kept measurements in `dapi_analysis_output/nuclei_measurements_summary.csv`. Generates QC overlays with **blue contours** for kept nuclei only (`overlays_analyzed/`).

**Visualization:** For each feature in `["area_um2","equivalent_diameter_um","aspect_ratio","circularity","eccentricity","solidity"]`, builds orange violin plots with jittered blue points, black ▢ medians, teal ◆ means ± SD ± 95 % CI, and inline mean labels. X-axis uses two-line mathtext condition names with `(n=…)`. Figures saved to `dapi_analysis_output/figs_violin/violin_<feature>_all_conditions.png`.

**Run:**  
```bash
python confocal/dapi_analysis.py
```
