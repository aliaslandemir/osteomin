#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collagen vs DAPI — robust normalization + scientific violin plots + QC scatter.
- Unified style with DAPI/Actin/DAPI scripts.
- Two-line x-labels e.g., "A+$\\beta$-GP\n$D_LK_H$".
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tf

from scipy import ndimage as ndi
from skimage import filters, morphology, measure, segmentation

# ------------------ STYLE ------------------
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "figure.autolayout": True,
})

VIOLIN_COLOR = "#E69F00"
DOT_COLOR    = "#0072B2"
MEAN_COLOR   = "#009E73"
MEAN_TEXT_FMT = "{:.2f}"
MEAN_TEXT_OFFSET = 0.14

# ========================== CONFIG ==========================
INPUT_DIR = Path("input_images")
OUTPUT_ROOT = Path("collagen_norm_output")

DAPI_CHANNEL_INDEX = 0
COLLAGEN_CHANNEL_INDEX = 2

PIXEL_SIZE_UM = 0.21
MIN_NUC_AREA_UM2 = 50.0
MAX_NUC_AREA_UM2 = 800.0

BG_SIGMA_PX = 15.0
LOW_PCT = 1.0
HIGH_PCT = 99.0
SCALE_PCT = 95.0

USE_WATERSHED = True
H_MAXIMA = 0.5

# Condition parsing (C1_..C8_ → label)
COND_CODE = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo+D/MK4 (L/L)",
    "C4": "Osteo+D/MK4 (L/H)",
    "C5": "Osteo+D/MK4 (H/L)",
    "C6": "Osteo+D/MK4 (H/H)",
    "C7": "Osteo+Dex (H)",
    "C8": "Osteo+MK-4 (H)",
}
ORDER_C = list(COND_CODE.values())

def two_line_label(pretty: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": r"A+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"A+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"A+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"A+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)":       r"A+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)":      r"A+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(pretty, pretty)

# ========================== I/O HELPERS ==========================
def setup_dirs():
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figs").mkdir(exist_ok=True)

def get_condition_from_stem(stem: str):
    m = re.match(r'^(C[1-8])[_-]', stem, flags=re.IGNORECASE)
    code = m.group(1).upper() if m else None
    return COND_CODE.get(code, "Unknown")

def read_channels(path: Path):
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()
    if arr.ndim == 2:
        dapi = arr.astype(np.float64, copy=False)
        col  = arr.astype(np.float64, copy=False)
    elif arr.ndim == 3:
        dapi = arr[DAPI_CHANNEL_INDEX].astype(np.float64, copy=False)
        col  = arr[COLLAGEN_CHANNEL_INDEX].astype(np.float64, copy=False)
    elif arr.ndim == 4:
        dapi = arr[0, DAPI_CHANNEL_INDEX].astype(np.float64, copy=False)
        col  = arr[0, COLLAGEN_CHANNEL_INDEX].astype(np.float64, copy=False)
    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")
    return dapi, col

# ========================== PROCESSING ==========================
def robust_normalize(img: np.ndarray) -> np.ndarray:
    bg = ndi.gaussian_filter(img, BG_SIGMA_PX) if BG_SIGMA_PX and BG_SIGMA_PX > 0 else 0.0
    x = img - bg
    lo, hi = np.percentile(x, [LOW_PCT, HIGH_PCT])
    x = np.clip(x, lo, hi)
    scale = np.percentile(x, SCALE_PCT)
    if not np.isfinite(scale) or scale <= 0:
        scale = hi - lo if hi > lo else 1.0
    x = x / scale
    x = np.clip(x, 0.0, 1.0)
    return x

def segment_nuclei(dapi_norm: np.ndarray):
    sm = ndi.gaussian_filter(dapi_norm, 1.0)
    thr = filters.threshold_otsu(sm)
    bw = sm > thr
    bw = morphology.remove_small_holes(bw, area_threshold=64)
    bw = morphology.remove_small_objects(bw, 64)

    if USE_WATERSHED:
        dist = ndi.distance_transform_edt(bw)
        peaks = morphology.h_maxima(dist, H_MAXIMA)
        markers = measure.label(peaks)
        lab = segmentation.watershed(-dist, markers, mask=bw)
    else:
        lab = measure.label(bw)

    area_per_px_um2 = (PIXEL_SIZE_UM ** 2)
    min_px = int(np.round(MIN_NUC_AREA_UM2 / area_per_px_um2))
    max_px = int(np.round(MAX_NUC_AREA_UM2 / area_per_px_um2))
    keep = np.zeros_like(lab, dtype=bool)
    for pr in measure.regionprops(lab):
        if min_px <= pr.area <= max_px:
            keep[lab == pr.label] = True
    lab = measure.label(keep)
    mask = lab > 0
    return lab, mask

# ========================== PLOTTING ==========================
def jittered(ax, x_center, y, width=0.06, s=12, alpha=0.8):
    if len(y) == 0:
        return
    x = np.random.normal(x_center, width, size=len(y))
    ax.scatter(x, y, s=s, alpha=alpha, color=DOT_COLOR, edgecolor="none", zorder=5)

def violin_all_conditions(df: pd.DataFrame, feature: str, outdir: Path):
    df = df.copy()
    present = [g for g in ORDER_C if g in df["condition"].unique()]
    if not present:
        print(f"[skip] {feature} (no valid conditions)")
        return

    data = [df.loc[df["condition"] == g, feature].dropna().values for g in present]

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for b in parts["bodies"]:
        b.set_facecolor(VIOLIN_COLOR)
        b.set_edgecolor("black")
        b.set_alpha(0.55)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    for i, y in enumerate(data, start=1):
        jittered(ax, i, y)
        if len(y) == 0:
            continue
        mu = float(np.mean(y))
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
        ci = 1.96 * sd / np.sqrt(len(y)) if len(y) > 1 else np.nan
        ax.plot([i], [mu], "D", color=MEAN_COLOR, ms=6, mec="black", mew=0.5, zorder=7)
        if np.isfinite(ci):
            ax.errorbar([i], [mu], yerr=ci, fmt="none", ecolor="#444", elinewidth=2, capsize=3, zorder=6)
        if np.isfinite(sd):
            ax.errorbar([i], [mu], yerr=sd, fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=5)
        if np.isfinite(mu):
            label = MEAN_TEXT_FMT.format(mu)
            ax.text(
                i + MEAN_TEXT_OFFSET,
                mu,
                label,
                color=MEAN_COLOR,
                fontsize=8.5,
                fontweight="bold",
                ha="left",
                va="center",
                zorder=8,
            )

    xticks = [two_line_label(c) for c in present]
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(xticks, ha="center")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_title("")
    fig.tight_layout()
    out = outdir / f"violin_{feature}.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

def qc_scatter(df: pd.DataFrame, feature: str, outdir: Path):
    fig, ax = plt.subplots(figsize=(6.6, 5.2))
    for c in [g for g in ORDER_C if g in df["condition"].unique()]:
        sub = df[df["condition"] == c]
        ax.scatter(sub["nuclei_count"], sub[feature], label=two_line_label(c), s=24, alpha=0.85)
    ax.set_xlabel("Nuclei count (per image)")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_title("")
    ax.legend(fontsize=7, frameon=False, ncol=2)
    fig.tight_layout()
    out = outdir / f"qc_scatter_{feature}_vs_nuclei.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out}")

# ========================== MAIN ==========================
if __name__ == "__main__":
    setup_dirs()
    figdir = OUTPUT_ROOT / "figs"
    csv_out = OUTPUT_ROOT / "image_metrics.csv"

    paths = sorted(list(INPUT_DIR.glob("*.tif")) + list(INPUT_DIR.glob("*.tiff")))
    if not paths:
        print(f"No images found in {INPUT_DIR}")
        raise SystemExit(0)

    area_per_px_um2 = (PIXEL_SIZE_UM ** 2)
    min_px = int(np.round(MIN_NUC_AREA_UM2 / area_per_px_um2))
    max_px = int(np.round(MAX_NUC_AREA_UM2 / area_per_px_um2))

    rows = []
    for p in paths:
        try:
            with tf.TiffFile(p) as tif:
                arr = tif.asarray()
            # --- read channels ---
            if arr.ndim == 2:
                dapi_raw = arr.astype(np.float64, copy=False)
                col_raw  = arr.astype(np.float64, copy=False)
            elif arr.ndim == 3:
                dapi_raw = arr[DAPI_CHANNEL_INDEX].astype(np.float64, copy=False)
                col_raw  = arr[COLLAGEN_CHANNEL_INDEX].astype(np.float64, copy=False)
            elif arr.ndim == 4:
                dapi_raw = arr[0, DAPI_CHANNEL_INDEX].astype(np.float64, copy=False)
                col_raw  = arr[0, COLLAGEN_CHANNEL_INDEX].astype(np.float64, copy=False)
            else:
                raise ValueError(f"Unsupported image shape: {arr.shape}")

            # --- robust per-image normalization ---
            dapi = robust_normalize(dapi_raw)
            col  = robust_normalize(col_raw)

            # --- nuclei segmentation on normalized DAPI ---
            lab, nuc_mask = segment_nuclei(dapi)

            nuclei_count = int(lab.max())
            field_area_px = dapi.shape[0] * dapi.shape[1]
            nuclear_area_px = int(nuc_mask.sum())
            confluence = nuclear_area_px / field_area_px if field_area_px > 0 else np.nan

            # --- normalized metrics ---
            sum_dapi_norm = float(np.nansum(dapi))
            sum_col_norm  = float(np.nansum(col))

            collagen_per_cell_norm = (sum_col_norm / nuclei_count) if nuclei_count > 0 else np.nan

            outside_mask = ~nuc_mask
            outside_area = int(outside_mask.sum())
            ecm_outside_sum_norm = float(np.nansum(col[outside_mask]))
            ecm_density_outside_norm = (ecm_outside_sum_norm / outside_area) if outside_area > 0 else np.nan

            rows.append({
                "image_filename": p.name,
                "condition": get_condition_from_stem(p.stem),
                "nuclei_count": nuclei_count,
                "confluence": confluence,
                "collagen_per_cell_norm": collagen_per_cell_norm,
                "ecm_density_outside_norm": ecm_density_outside_norm,
                "collagen_over_dapi_norm": (sum_col_norm / sum_dapi_norm) if sum_dapi_norm > 0 else np.nan,
            })
        except Exception as e:
            print(f"[warn] {p.name}: {e}")

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid images processed.")
        raise SystemExit(0)

    present = [c for c in ORDER_C if c in df["condition"].unique()]
    df = df[df["condition"].isin(present)].copy()

    df.to_csv(csv_out, index=False)
    print(f"[saved] {csv_out}")

    for feat in ["collagen_per_cell_norm", "ecm_density_outside_norm", "collagen_over_dapi_norm"]:
        violin_all_conditions(df, feat, figdir)
        qc_scatter(df, feat, figdir)

    print("✅ Done.")
