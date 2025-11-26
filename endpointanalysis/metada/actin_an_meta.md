### `confocal/actin_analysis.py` — F-Actin Coherency & Length Density
Quantifies actin alignment per field, normalizes by DAPI coverage, and generates violin plots consistent with the DAPI/ALP/collagen figure style.

**Inputs:** `confocal/input_images/` containing multichannel `.tif/.tiff/.png/.jpg`. Defaults assume actin channel `1`, DAPI channel `0`, pixel size `0.21 µm/px`. Condition codes derive from filename prefixes (`C1_…C8_`), mapping to Basal/Osteo ± Dex/MK-4 combinations.

**Processing pipeline per image:**
1. Read both channels, perform percentile-based scaling and CLAHE (actin) / linear normalization (DAPI).
2. Compute the structure-tensor **coherency** map (`sigma=1.5` by default).
3. Threshold coherency (`coh_t=0.6`) and intensity (`i_t=0.12`), remove small components, and skeletonize to estimate **length density** (µm/mm²) using fast downsampled thinning.
4. Compute summary metrics:
   - `mean_coherency` and `length_density_um_per_mm2`
   - DAPI mask fraction (for normalization)
   - DAPI-normalized counterparts (`*_normDAPI`).
5. Optionally save cyan-on-gray overlays of high-coherency fibers (`actin_results/overlays/*_overlay.png`).

**Outputs (`actin_results/`):**
- `actin_coherency_summary.csv` — per-image table of the metrics above plus condition labels.
- `figs/violin_<metric>.png` — amber violins with blue jitter dots, black medians, teal diamonds for means ± SD ± 95 % CI, and teal numeric mean labels offset to the right (same treatment as ALP/DAPI/collagen).

**Run:**  
```bash
python confocal/actin_analysis.py \
    --input input_images \
    --out actin_results
```
Command-line flags let you tweak channel indices, coherency thresholds, skeleton downsampling (`--len_ds`), and overlay parameters to match different acquisitions.
