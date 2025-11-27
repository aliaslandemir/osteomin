# `ars_analysis.py` — Condition Reference

## Overview

`ars_analysis.py` quantifies Alizarin Red S–stained area for images grouped into eight experimental conditions (C1–C8). Each image is assigned a code by matching `C1..C8` in the filename or in parent folders (`c1..c8`). The processing pipeline computes a threshold for the red channel over green (`R-G`) and applies an L* cap to focus on darker (stained) pixels. Results include per-image statistics, global or per-image thresholds, and two figures: a violin plot (`figs/violin_percent_area.pdf`) and a bar plot with mean±SD (`figs/bar_mean_sd_percent_area.pdf`). Provide `--input`/`--out` (or `--output`) to match your dataset root.

## Condition codes

The script converts each code to a descriptive label that appears in the output tables and on the plots:

| Code | Condition label          | Plot label (two lines)         |
|------|--------------------------|--------------------------------|
| C1   | Basal                    | `Basal`                        |
| C2   | Osteo                    | `A+β-GP`                       |
| C3   | Osteo+D/MK4 (L/L)        | `A+β-GP` / `D_LK_L`            |
| C4   | Osteo+D/MK4 (L/H)        | `A+β-GP` / `D_LK_H`            |
| C5   | Osteo+D/MK4 (H/L)        | `A+β-GP` / `D_HK_L`            |
| C6   | Osteo+D/MK4 (H/H)        | `A+β-GP` / `D_HK_H`            |
| C7   | Osteo+Dex (H)            | `A+β-GP` / `D_HK_0`            |
| C8   | Osteo+MK-4 (H)           | `A+β-GP` / `D_0K_H`            |

The two-line labels inject the ALP-specific notation (first line `A+β-GP`, second line the treatment codes) so violin and bar plots display uniform, aligned legends.

## Plot styling notes

- **Violin plot** (`violin_with_median`):
  - Violins are tinted `#E69F00` with light transparency and a black outline.
  - Dots represent individual image measurements and are jittered horizontally with a dark blue fill and black edge.
  - Means (diamond marker) and medians (square) are overlaid alongside standard deviation and 95% confidence intervals.

- **Bar plot** (`bar_mean_sd_with_text`):
  - Bars use the same orange fill, with a diamond marker for the mean and each bar annotated with the mean value.
  - X-axis tick labels reuse the two-line mapping.

## Usage recap

```sh
python ars_analysis.py \
    --input data/ars \
    --out ars_out \
    --lcap 65
```

Key options:

| Flag         | Purpose |
|--------------|---------|
| `--mode`     | `global` uses median threshold, `perimg` re-computes Otsu per image |
| `--thumb`    | Downsample factor for threshold estimation (lower → more aggressive) |
| `--lcap`     | L* cap (default 85) filters out lighter pixels |
| `--save-scale` | Downscale factor for saved masks/overlays |

Processed data writes to `ars_out` with masks, overlays, tables (including `ars_area_by_image.csv`), and figures.
