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
BG_SIGMA = 15.0  # 5, 10, 15, 20 etc.
CONDITIONS = [f"C{i}" for i in range(1, 9)]  # C1..C8


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


def bg_subtract(img: np.ndarray) -> np.ndarray:
    bg = ndi.gaussian_filter(img, BG_SIGMA)
    out = img - bg
    out[out < 0] = 0
    return out


def compute_metric(path: Path) -> dict:
    dapi_raw, col_raw = read_channels(path)
    dapi_bg = bg_subtract(dapi_raw)
    col_bg = bg_subtract(col_raw)

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
    x_vals, y_vals = [], []

    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].values
        for v in vals:
            x_vals.append(cond)
            y_vals.append(v)

    ax.scatter(x_vals, y_vals, s=30)
    ax.set_xlabel("Condition")
    ax.set_ylabel("collagen_over_dapi_bg")
    ax.set_title("")
    fig.tight_layout()

    out = outdir / "scatter_collagen_over_dapi_bg.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out}")


# -------- main --------
def parse_args():
    p = argparse.ArgumentParser(description="Collagen/DAPI scatter.")
    p.add_argument("-i", "--input", type=Path, default=INPUT_DIR)
    p.add_argument("-o", "--out", type=Path, default=OUTPUT_DIR)
    return p.parse_args()


def main():
    args = parse_args()
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
    df.to_csv(out_dir / "image_metrics.csv", index=False)
    print(f"[saved] {out_dir/'image_metrics.csv'}")

    plot_scatter(df, out_dir)


if __name__ == "__main__":
    main()



# To run

# python collagen_code.py --input input_images --out collagen_out