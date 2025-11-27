#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPI nuclei segmentation + shape analysis + publication-grade violin plots.
- Detects condition from filename *prefix* (C1_, C2_, ...).
- Uses unified plotting style across all figures.
- Two-line x-labels using mathtext for β and subscripts (e.g., "A+$\\beta$-GP\n$D_LK_H$").
- Overlays: red contours (all nuclei) and blue contours (kept nuclei after filters).
- Post-filter nuclei by area (µm²) and circularity.
"""

import warnings
warnings.filterwarnings('ignore', category=UserWarning)

import argparse
import matplotlib
matplotlib.use("Agg")  # safe for headless
import matplotlib.pyplot as plt
from matplotlib import rcParams

from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tf
from scipy import ndimage
from skimage import exposure, filters, measure, util, segmentation, morphology, feature

# ========================== GLOBAL STYLE (shared) ==========================
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

# Colors
VIOLIN_COLOR = "#E69F00"
DOT_COLOR    = "#0072B2"
MEAN_COLOR   = "#009E73"
MEAN_TEXT_FMT = "{:.2f}"
MEAN_TEXT_OFFSET = 0.14

# ========================== CONFIG ==========================

INPUT_DIR = Path("input_images")
OUTPUT_ROOT = Path("dapi_analysis_output")
PIXEL_SIZE_UM = 0.21  # µm/px (reporting only)

# Segmentation
DAPI_CHANNEL_INDEX = 0
MIN_NUCLEUS_SIZE_PX = 100
CLAHE_CLIP_LIMIT = 0.02
GAUSSIAN_BLUR_SIGMA = 3.0
WATERSHED_PEAK_MIN_DIST = 10

SHAPE_PROPERTIES = [
    "label", "area", "perimeter", "minor_axis_length",
    "major_axis_length", "eccentricity", "solidity", "mean_intensity"
]

FEATURES_TO_PLOT = [
    "area_um2", "equivalent_diameter_um", "aspect_ratio",
    "circularity", "eccentricity", "solidity"
]

# Condition mapping/order
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
    """Return standardized two-line label with mathtext."""
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

# Post-filter thresholds
MIN_AREA_UM2 = 100.0
MAX_AREA_UM2 = 600.0
MIN_CIRCULARITY = 0.5
MAX_CIRCULARITY = 1.2

# ========================== HELPERS ==========================

def setup_output_directories(output_root: Path):
    global OUTPUT_ROOT, OUTPUT_PNG_DIR, OUTPUT_PNG_ANALYZED_DIR, OUTPUT_LABEL_DIR, OUTPUT_CSV_PATH, OUTPUT_FIG_DIR
    OUTPUT_ROOT = output_root
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_PNG_DIR = OUTPUT_ROOT / "overlays"                # red all-nuclei
    OUTPUT_PNG_DIR.mkdir(exist_ok=True)
    OUTPUT_PNG_ANALYZED_DIR = OUTPUT_ROOT / "overlays_analyzed"  # blue kept-nuclei
    OUTPUT_PNG_ANALYZED_DIR.mkdir(exist_ok=True)
    OUTPUT_LABEL_DIR = OUTPUT_ROOT / "labelmaps"
    OUTPUT_LABEL_DIR.mkdir(exist_ok=True)
    OUTPUT_FIG_DIR = OUTPUT_ROOT / "figs_violin"
    OUTPUT_FIG_DIR.mkdir(exist_ok=True)
    OUTPUT_CSV_PATH = OUTPUT_ROOT / "nuclei_measurements_summary.csv"

def get_condition_code_from_stem(stem: str):
    m = re.match(r'^(C[1-8])[_-]', stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def get_condition_pretty_from_path(p: Path) -> str:
    code = get_condition_code_from_stem(p.stem)
    if code and code in COND_CODE:
        return COND_CODE[code]
    return "Unknown"

def load_dapi_channel(image_path: Path) -> np.ndarray:
    with tf.TiffFile(image_path) as tif:
        data = tif.asarray()
    if data.ndim == 4:  # Z, C, Y, X
        dapi = data[0, DAPI_CHANNEL_INDEX, :, :]
    elif data.ndim == 3:  # C, Y, X
        dapi = data[DAPI_CHANNEL_INDEX, :, :]
    else:  # Y, X
        dapi = data
    return dapi.squeeze()

def derive_shape_metrics(df: pd.DataFrame) -> pd.DataFrame:
    eps = 1e-12
    df = df.copy()
    df["aspect_ratio"] = (df["major_axis_length"] + eps) / (df["minor_axis_length"] + eps)
    df["circularity"] = (4 * np.pi * df["area"]) / (np.maximum(df["perimeter"], eps) ** 2)
    df["equivalent_diameter_px"] = 2 * np.sqrt(df["area"] / np.pi)
    if PIXEL_SIZE_UM:
        s = PIXEL_SIZE_UM
        df["area_um2"] = df["area"] * s**2
        df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * s
    return df

def jittered_scatter(ax, x_center, y, width=0.06, s=10, alpha=0.75):
    if len(y) == 0:
        return
    x = np.random.normal(x_center, width, size=len(y))
    ax.scatter(x, y, s=s, alpha=alpha, color=DOT_COLOR, edgecolor="none", zorder=5)

# ========================== SEGMENTATION ==========================

def process_single_image(image_path: Path):
    try:
        dapi_original = load_dapi_channel(image_path)
        dapi_float = util.img_as_float(dapi_original)

        dapi_blurred = filters.gaussian(dapi_float, sigma=GAUSSIAN_BLUR_SIGMA)
        dapi_eq = exposure.equalize_adapthist(dapi_blurred, clip_limit=CLAHE_CLIP_LIMIT)
        thresh = filters.threshold_otsu(dapi_eq)
        binary = dapi_eq > thresh
        binary_filled = ndimage.binary_fill_holes(binary)

        dist = ndimage.distance_transform_edt(binary_filled)
        coords = feature.peak_local_max(
            dist, min_distance=WATERSHED_PEAK_MIN_DIST,
            labels=binary_filled, exclude_border=False
        )
        mask_markers = np.zeros(dist.shape, dtype=bool)
        if coords.size > 0:
            mask_markers[tuple(coords.T)] = True
        markers = measure.label(mask_markers)
        labeled = segmentation.watershed(-dist, markers, mask=binary_filled)
        labeled = segmentation.clear_border(labeled)
        labeled = morphology.remove_small_objects(labeled, min_size=MIN_NUCLEUS_SIZE_PX)
        labeled_final, num = measure.label(labeled > 0, return_num=True)
        if num == 0:
            print(f"> No nuclei found in {image_path.name}")
            return None

        # Save label map
        label_path = OUTPUT_LABEL_DIR / f"{image_path.stem}_labels.tif"
        tf.imwrite(label_path, labeled_final.astype(np.uint16), photometric='minisblack')

        props = measure.regionprops_table(labeled_final, intensity_image=dapi_original,
                                          properties=SHAPE_PROPERTIES)
        df = pd.DataFrame(props)
        df["image_filename"] = image_path.name
        df["condition"] = get_condition_pretty_from_path(image_path)
        # overlay with RED contours (all)
        disp = exposure.rescale_intensity(dapi_original, out_range=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray")
        ax.contour(labeled_final > 0, levels=[0.5], colors="red", linewidths=0.9)
        ax.set_axis_off()
        fig.savefig(OUTPUT_PNG_DIR / f"{image_path.stem}_overlay.png",
                    dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return df
    except Exception as e:
        print(f"!! ERROR {image_path.name}: {e}")
        return None

# ========================== PLOTTING ==========================

def violin_all_conditions(df: pd.DataFrame, feature: str, outdir: Path):
    df = df.copy()
    present = [g for g in ORDER_C if g in df['condition'].unique()]
    if not present:
        print(f"[skip] {feature} (no valid conditions)")
        return
    data = [df.loc[df['condition'] == g, feature].dropna().values for g in present]

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    parts = ax.violinplot(data, showmeans=False, showmedians=True, showextrema=False)

    for b in parts['bodies']:
        b.set_facecolor(VIOLIN_COLOR)
        b.set_edgecolor("black")
        b.set_alpha(0.55)
    if "cmedians" in parts:
        parts["cmedians"].set_color("black")
        parts["cmedians"].set_linewidth(1.2)

    for i, y in enumerate(data, start=1):
        jittered_scatter(ax, i, y)
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
            ax.text(i + MEAN_TEXT_OFFSET, mu, label,
                    color=MEAN_COLOR, fontsize=8.5, fontweight="bold",
                    ha="left", va="center", zorder=8)

    counts = [len(y) for y in data]
    xticks = [f"{two_line_label(g)}\n(n={n})" for g, n in zip(present, counts)]
    ax.set_xticks(range(1, len(present) + 1))
    ax.set_xticklabels(xticks, ha="center")
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_title("")  # minimalist
    fig.tight_layout()

    fig.savefig(
        outdir / f"violin_{feature}_all_conditions.pdf",
        bbox_inches="tight",
        dpi=300,
    )
    plt.close(fig)

def parse_args():
    parser = argparse.ArgumentParser(description="DAPI segmentation and violin plots.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=INPUT_DIR,
        help="Folder with confocal input images (.*tif/.*tiff)",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=OUTPUT_ROOT,
        help="Root directory for overlays, label maps, and CSV output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    input_dir = args.input
    setup_output_directories(args.out)

    paths = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not paths:
        print(f"No images found in {input_dir}")
        raise SystemExit(0)

    all_data = []
    try:
        from tqdm import tqdm

        iterator = tqdm(paths, desc="Processing Images")
    except ImportError:
        iterator = paths

    for p in iterator:
        df = process_single_image(p)
        if df is not None:
            all_data.append(df)

    if not all_data:
        print("No nuclei processed.")
        raise SystemExit(0)

    final_df = pd.concat(all_data, ignore_index=True)
    final_df = derive_shape_metrics(final_df)

    # ---------- POST FILTERS ----------
    if "area_um2" not in final_df.columns:
        raise RuntimeError("area_um2 not found — ensure PIXEL_SIZE_UM is set.")

    before_n = len(final_df)
    keep_mask = (
        final_df["area_um2"].between(MIN_AREA_UM2, MAX_AREA_UM2, inclusive="both")
        & final_df["circularity"].between(MIN_CIRCULARITY, MAX_CIRCULARITY, inclusive="both")
    )
    filtered_df = final_df.loc[keep_mask].copy()
    after_n = len(filtered_df)
    kept_pct = (after_n / before_n * 100.0) if before_n else 0.0
    print(
        f"Filtered nuclei: kept {after_n}/{before_n} ({kept_pct:.1f}%) "
        f"[area {MIN_AREA_UM2}-{MAX_AREA_UM2} µm², circularity {MIN_CIRCULARITY}-{MAX_CIRCULARITY}]"
    )

    filtered_df.to_csv(OUTPUT_CSV_PATH, index=False)
    print(f"[saved] {OUTPUT_CSV_PATH}")

    # === analyzed-only overlays (BLUE edges) ===
    kept_by_img = (
        filtered_df.groupby("image_filename")["label"]
        .apply(lambda s: set(map(int, s.values)))
        .to_dict()
    )
    for img_name, kept_labels in kept_by_img.items():
        if not kept_labels:
            continue
        img_path = next((p for p in paths if p.name == img_name), None)
        if img_path is None:
            continue
        dapi_original = load_dapi_channel(img_path)
        label_path = OUTPUT_LABEL_DIR / f"{Path(img_name).stem}_labels.tif"
        if not label_path.exists():
            print(f"[warn] Missing labelmap for {img_name}, skip analyzed overlay.")
            continue
        labeled_final = tf.imread(label_path).astype(np.int32)
        kept_mask_img = np.isin(labeled_final, list(kept_labels))

        disp = exposure.rescale_intensity(dapi_original, out_range=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray")
        ax.contour(
            kept_mask_img,
            levels=[0.5],
            colors="deepskyblue",
            linewidths=1.0,
        )
        ax.set_axis_off()
        fig.savefig(
            OUTPUT_PNG_ANALYZED_DIR / f"{Path(img_name).stem}_overlay_analyzed.png",
            dpi=150,
            bbox_inches="tight",
            pad_inches=0,
        )
        plt.close(fig)

    # Violin plots on KEPT (filtered) data
    for feat in FEATURES_TO_PLOT:
        if feat in filtered_df.columns:
            violin_all_conditions(filtered_df, feat, OUTPUT_FIG_DIR)

    print("\n✅ Done.")
    print(" - All-nuclei overlays (red):", OUTPUT_PNG_DIR)
    print(" - Analyzed-only overlays (blue):", OUTPUT_PNG_ANALYZED_DIR)
    print(" - Label maps:", OUTPUT_LABEL_DIR)
    print(" - Figures:", OUTPUT_FIG_DIR)
    print(" - Table:", OUTPUT_CSV_PATH)


if __name__ == "__main__":
    main()
