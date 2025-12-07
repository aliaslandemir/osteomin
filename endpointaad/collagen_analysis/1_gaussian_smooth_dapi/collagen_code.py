#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tf
from scipy import ndimage as ndi

import matplotlib.pyplot as plt

# ---------------- configurable ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2
BG_SIGMA = 10.0  # default; can be overridden by --sigma


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


# -------- helper functions --------
def get_condition(stem: str) -> str:
    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_channels(path: Path):
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    if arr.ndim == 2:
        dapi = arr.astype(np.float64)
        col = arr.astype(np.float64)
    elif arr.ndim == 3:
        dapi = arr[DAPI_CHANNEL].astype(np.float64)
        col = arr[COLLAGEN_CHANNEL].astype(np.float64)
    elif arr.ndim == 4:
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
        "condition": get_condition(path.stem),
        "collagen_over_dapi_bg": ratio,
    }


# -------- plotting --------
def plot_scatter(df: pd.DataFrame, outdir: Path):
    fig, ax = plt.subplots(figsize=(8, 4))

    x_pos = {cond: i for i, cond in enumerate(CONDITIONS, start=1)}
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].values
        xpos = x_pos[cond]
        if len(vals) > 0:
            ax.scatter([xpos] * len(vals), vals, s=30)

    #ax.set_xlabel("Condition")
    ax.set_ylabel("collagen_over_dapi_bg")
    ax.set_xticks([x_pos[c] for c in CONDITIONS])
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in CONDITIONS])
    ax.set_title("")
    fig.tight_layout()

    fname = f"{int(BG_SIGMA)}_scatter_collagen_over_dapi_bg.png"
    png_out = outdir / fname
    pdf_out = png_out.with_suffix(".pdf")
    fig.savefig(png_out, dpi=300)
    fig.savefig(pdf_out, dpi=300)
    plt.close(fig)
    print(f"[saved] {png_out}")
    print(f"[saved] {pdf_out}")


# -------- main --------
def parse_args():
    p = argparse.ArgumentParser(description="Collagen/DAPI scatter.")
    p.add_argument("-i", "--input", type=Path, default=INPUT_DIR)
    p.add_argument("-o", "--out", type=Path, default=OUTPUT_DIR)
    p.add_argument(
        "-s",
        "--sigma",
        type=float,
        default=BG_SIGMA,
        help="Gaussian sigma for smoothing (e.g. 5, 10, 15).",
    )
    return p.parse_args()


def main():
    global BG_SIGMA

    args = parse_args()
    BG_SIGMA = args.sigma  # override default with CLI value

    in_dir = args.input
    out_dir = args.out
    out_dir.mkdir(exist_ok=True, parents=True)

    paths = sorted(list(in_dir.glob("*.tif")) + list(in_dir.glob("*.tiff")))
    if not paths:
        print("No images found.")
        return

    rows = []
    for p in paths:
        try:
            rows.append(compute_metric(p))
        except Exception as e:
            print(f"[warn] {p}: {e}")

    df = pd.DataFrame(rows)
    df = df[df["condition"].isin(CONDITIONS)]

    sigma_str = f"{BG_SIGMA:g}"
    csv_path = out_dir / f"image_metrics_BG{sigma_str}.csv"
    df.to_csv(csv_path, index=False)

    print(f"[saved] {csv_path}")
    plot_scatter(df, out_dir)


if __name__ == "__main__":
    main()

# To run:
# python collagen_code.py --input input_images --out collagen_out --sigma 10
