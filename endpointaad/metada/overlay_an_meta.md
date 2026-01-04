# `overlay_analysis.py` — Confocal overlay exporter

## Purpose
`overlay_analysis.py` reads multichannel microscopy input (DAPI, Actin, Collagen) and creates publication-ready RGB overlays in `--out/overlays`.

## Workflow

1. **Channel extraction**: Each TIFF/PNG/JPEG is inspected for a `C1..C8` prefix, and the DAPI, Actin, and Collagen channels are pulled via `pick_channel` (handles common HWC/CHW/4D layouts).
2. **Channel prep** (`prepare_channel`):
   * Channel-specific Gaussian smoothing (lighter on DAPI/Actin, heavier on Collagen) to preserve dotty nuclear signal while reducing Collagen noise.
   * Percentile stretch from P1 to P99 for contrast.
   * Per-channel gamma plus unsharp sharpening (always stronger for DAPI) so nuclei/actin filaments stay crisp without oversmoothing.
   * Background floor reduction (`BACKGROUND_DROP=0.04`) plus scaling that boosts DAPI/Actin intensity and dampens Collagen before compositing.
3. **Compose RGB**: Each normalized channel is tinted (`COLOR_DAPI`, `COLOR_ACTIN`, `COLOR_COLLAGEN`), converted to single intensities, and combined via screen blending; the merged RGB is slightly dimmed to keep the overall background darker and the stained structures popping.
4. **Annotation**:
   * Scale bar appears in the lower-left, auto-sized based on `--pix` (defaults to 0.21 µm/px); optionally force `--scalebar_um`.
   * Channel legend + stretch text are drawn across the top banner with outlined white text for readability.
   * Condition text uses predefined two-line ALP labels (e.g., `A+β-GP` / `D_HK_L`) drawn at the top-left.
5. **Export**: Saves overlays at 100 DPI into `overlay_output/overlays/<basename>_overlay.png`; set `--pdf` (plus optional `--pdf-dpi`) to also emit 300 DPI PDF copies with the same basename.

## Usage
```sh
python overlay_analysis.py \
    --input input_images \
    --out overlay_output \
    --pix 0.21 \
    --scalebar_um 200 \
    --pdf
```

Set `--pdf-dpi 300` (default) to keep the secondary export at publication resolution.

Input hints: filenames should start with `C1_…C8_` or live in a parent named `c1…c8`.
