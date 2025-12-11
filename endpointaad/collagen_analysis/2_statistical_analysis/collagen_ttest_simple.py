#!/usr/bin/env python3

"""
Simple collagen/DAPI stats/plot assuming images are technical replicates
and parametric tests.

- Reads multichannel TIFFs, extracts DAPI + collagen channels
- Gaussian-smooths both, computes sum(collagen_bg) / sum(dapi_bg)
- Computes mean ± SEM for each condition
- Runs Welch's t-tests vs Basal (C1), treating images as independent
- Plots jittered individual points + mean ± SEM and marks * for p < 0.05

NOTE: This is a "simple" parametric pipeline; it ignores
      (i) non-normality, (ii) tiny n, and (iii) multiple-testing correction.
      Use mainly as a quick visual / sanity check.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tifffile as tf
from scipy import ndimage as ndi
from scipy import stats

# ---------------- configurable ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2
BG_SIGMA = 10.0

# -------------------- CONDITIONS & LABELS --------------------
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
CONDITIONS = list(COND_CODE.keys())


def two_line_label(name: str) -> str:
    """Two-line labels for conditions."""
    m = {
        "Basal": "Basal",
        "Osteo": r"AA+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)": r"AA+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


def save_figures(fig, pdf_path: Optional[Path], png_path: Optional[Path]):
    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if png_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")


# -------- image helpers --------
def get_condition_from_name(stem: str) -> str:
    import re

    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_channels(path: Path):
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    if arr.ndim == 2:
        raise ValueError("2D TIFF found but channel count unknown.")

    if arr.ndim == 3:
        # assume C,H,W
        if arr.shape[0] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError(f"Image {path} does not contain enough channels.")
        dapi = arr[DAPI_CHANNEL].astype(np.float64)
        col = arr[COLLAGEN_CHANNEL].astype(np.float64)

    elif arr.ndim == 4:
        # assume T,C,H,W or Z,C,H,W — take first T/Z plane
        if arr.shape[1] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError(f"Image {path} does not contain enough channels.")
        dapi = arr[0, DAPI_CHANNEL].astype(np.float64)
        col = arr[0, COLLAGEN_CHANNEL].astype(np.float64)

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")

    return dapi, col


def bg_smooth(img: np.ndarray) -> np.ndarray:
    return ndi.gaussian_filter(img, BG_SIGMA)


def compute_metric(path: Path) -> dict:
    dapi_raw, col_raw = read_channels(path)
    dapi_bg = bg_smooth(dapi_raw)
    col_bg = bg_smooth(col_raw)

    sum_dapi = float(np.sum(dapi_bg))
    sum_col = float(np.sum(col_bg))
    ratio = sum_col / sum_dapi if sum_dapi > 0 else np.nan

    return {
        "image": path.name,
        "condition": get_condition_from_name(path.stem),
        "collagen_over_dapi_bg": ratio,
    }


# ----------------- ARG PARSING ----------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute collagen/DAPI (Gaussian-smoothed sum ratio) from TIFF images "
            "and run simple Welch t-tests vs basal (technical replicates)."
        )
    )
    p.add_argument("-input", type=Path, default=INPUT_DIR)
    p.add_argument("--out", help="Optional output PDF path")
    p.add_argument(
        "--png",
        help="Optional PNG path; defaults to the same stem as --out with .png.",
    )
    p.add_argument(
        "--basal",
        default="C1",
        help="Code of basal condition (default: C1).",
    )
    p.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=BG_SIGMA,
        help="Gaussian sigma for background smoothing.",
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for stars (default: 0.05).",
    )
    return p.parse_args()


# ----------------- MAIN ----------------- #
def main():
    global BG_SIGMA
    args = parse_args()
    BG_SIGMA = args.sigma

    input_dir = args.input
    paths = sorted(list(input_dir.glob("*.tif")) + list(input_dir.glob("*.tiff")))
    if not paths:
        print("No images found in", input_dir)
        return

    # compute metrics per image
    rows = []
    for pth in paths:
        try:
            rows.append(compute_metric(pth))
        except Exception as e:
            print(f"[warn] {pth}: {e}")

    df = pd.DataFrame(rows)
    df = df[df["condition"].isin(CONDITIONS)]

    if df.empty:
        print("No valid C1–C8 images found after filtering.")
        return

    # save raw metrics
    out_dir = OUTPUT_DIR if args.out is None and args.png is None else Path(".")
    out_dir.mkdir(exist_ok=True)
    csv_path = out_dir / f"collagen_image_metrics_BG{BG_SIGMA}.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved metrics CSV: {csv_path}")

    # build cond_to_vals dict
    cond_to_ratio: Dict[str, List[float]] = {}
    for cond in CONDITIONS:
        vals = (
            df.loc[df["condition"] == cond, "collagen_over_dapi_bg"]
            .dropna()
            .astype(float)
            .tolist()
        )
        if vals:
            cond_to_ratio[cond] = vals

    conds_c = [c for c in CONDITIONS if c in cond_to_ratio]

    if not conds_c:
        print("No conditions with data, exiting.")
        return

    # ---------- Simple Welch t-tests vs basal ----------
    if args.basal not in cond_to_ratio or len(cond_to_ratio[args.basal]) == 0:
        raise ValueError(f"Basal condition '{args.basal}' has no data.")

    basal_vals = np.asarray(cond_to_ratio[args.basal], dtype=float)
    mean_dict: Dict[str, float] = {}
    sem_dict: Dict[str, float] = {}
    pvals: Dict[str, float] = {}

    for cond in conds_c:
        vals = np.asarray(cond_to_ratio.get(cond, []), dtype=float)
        if vals.size == 0:
            mean_dict[cond] = np.nan
            sem_dict[cond] = np.nan
            pvals[cond] = np.nan
            continue

        mean = float(np.mean(vals))
        sem = (
            float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            if vals.size > 1
            else np.nan
        )
        mean_dict[cond] = mean
        sem_dict[cond] = sem

        if cond == args.basal:
            pvals[cond] = np.nan
        else:
            _t_stat, p = stats.ttest_ind(basal_vals, vals, equal_var=False)
            pvals[cond] = p

    # ---------- Print summary ----------
    print("\n===== Simple Welch t-tests vs Basal (collagen_over_dapi_bg) =====")
    print(f"Basal condition: {args.basal}")
    for cond in conds_c:
        mean = mean_dict[cond]
        sem = sem_dict[cond]
        p = pvals[cond]
        n = len(cond_to_ratio.get(cond, []))
        if cond == args.basal:
            print(f"{cond} (Basal): mean = {mean:.4g}, SEM = {sem:.4g}, n = {n}")
        else:
            p_str = "NA" if np.isnan(p) else f"{p:.4g}"
            print(
                f"{cond}: mean = {mean:.4g}, SEM = {sem:.4g}, n = {n}, "
                f"p (Welch vs Basal) = {p_str}"
            )

    # ---------- Plot: publication-style scatter + mean/SEM ----------

    x_positions = np.arange(1, len(conds_c) + 1)
    means = [mean_dict[c] for c in conds_c]
    sems = [sem_dict[c] for c in conds_c]

    # all values for y-range
    all_vals = []
    for c in conds_c:
        all_vals.extend(cond_to_ratio.get(c, []))

    if all_vals:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
    else:
        y_min, y_max = 0.0, 1.0
    yr = max(y_max - y_min, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 4))

    # Style similar to ALP plot
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", width=1)

    rng = np.random.default_rng(42)
    jitter_width = 0.12

    # Dots (images as technical replicates) – no hard-coded palette
    for x, cond in zip(x_positions, conds_c):
        vals = np.asarray(cond_to_ratio.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
        ax.scatter(
            np.full_like(vals, x, dtype=float) + jitter,
            vals,
            s=28,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

    # Mean + SEM whiskers (black for clarity)
    for x, m, se in zip(x_positions, means, sems):
        if np.isnan(m):
            continue
        ax.hlines(
            m,
            x - 0.22,
            x + 0.22,
            color="black",
            linewidth=2.0,
            zorder=3,
        )
        if not np.isnan(se):
            ax.vlines(x, m - se, m + se, color="black", linewidth=1.0)
            ax.hlines(m - se, x - 0.10, x + 0.10, color="black", linewidth=1.0)
            ax.hlines(m + se, x - 0.10, x + 0.10, color="black", linewidth=1.0)

    # X labels
    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)

    ax.set_ylabel("Collagen/DAPI (Gaussian-smoothed sum ratio)", fontsize=10)

    # Stars for p < alpha vs basal
    for x, cond in zip(x_positions, conds_c):
        if cond == args.basal:
            continue
        p = pvals[cond]
        if p is None or np.isnan(p) or p >= args.alpha:
            continue
        vals = np.asarray(cond_to_ratio.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        group_max = float(np.max(vals))
        star_y = group_max + 0.09 * yr
        ax.text(
            x,
            star_y,
            "*",
            ha="center",
            va="bottom",
            fontsize=14,
        )

    ax.set_ylim(y_min - 0.05 * yr, y_max + 0.25 * yr)

    fig.tight_layout()

    # save / show
    if args.out or args.png:
        pdf_path = Path(args.out) if args.out else None
        png_path = (
            Path(args.png)
            if args.png
            else pdf_path.with_suffix(".png") if pdf_path else None
        )
        save_figures(fig, pdf_path, png_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()


# To Run

"""
Example usage:

python collagen_ttest_simple.py \
  -input input_images \
  --basal C1 \
  --alpha 0.05 \
  --out collagen_ttest_simple.pdf
"""
