#!/usr/bin/env python3

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tf
from scipy import ndimage as ndi
from scipy import stats
import matplotlib.pyplot as plt

# ---------------- configurable ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2
BG_SIGMA = 10.0
BASAL_COND = "C1"

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
    m = {
        "Basal": "Basal",
        "Osteo": r"A+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"A+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"A+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"A+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)":    r"A+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)":   r"A+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }

    return m.get(name, name)


# -------- helper functions --------
def get_condition(stem: str) -> str:
    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_channels(path: Path):
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    # FIX 2 — safer handling
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
        "condition": get_condition(path.stem),
        "collagen_over_dapi_bg": ratio,
    }


# -------- stats --------
def run_kruskal(df: pd.DataFrame, outdir: Path):
    groups = []
    labels = []
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(cond)

    if len(groups) < 2:
        print("Not enough groups for Kruskal–Wallis.")
        return np.nan, np.nan

    H, p = stats.kruskal(*groups)

    sigma_str = f"{BG_SIGMA:g}"
    out_txt = outdir / f"kruskal_BG{sigma_str}.txt"
    with open(out_txt, "w") as f:
        f.write("Kruskal–Wallis test on collagen_over_dapi_bg\n")
        f.write("Conditions: " + ", ".join(labels) + "\n")
        f.write(f"H: {H:.4f}\n")
        f.write(f"p: {p:.4e}\n")

    print(f"[saved] {out_txt}")
    return H, p


def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def run_pairwise_vs_basal(df: pd.DataFrame, outdir: Path):
    basal_vals = df.loc[df["condition"] == BASAL_COND, "collagen_over_dapi_bg"].dropna().values
    if len(basal_vals) == 0:
        print("No basal data.")
        return pd.DataFrame(), {}

    records = []
    p_raw_list = []
    cond_list = []

    for cond in CONDITIONS:
        if cond == BASAL_COND:
            continue
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].dropna().values
        if len(vals) == 0:
            continue

        U, p_raw = stats.mannwhitneyu(basal_vals, vals, alternative="two-sided")
        records.append({
            "basal": BASAL_COND,
            "condition": cond,
            "n_basal": len(basal_vals),
            "n_cond": len(vals),
            "median_basal": float(np.median(basal_vals)),
            "median_cond": float(np.median(vals)),
            "median_diff": float(np.median(vals) - np.median(basal_vals)),
            "U": U,
            "p_raw": p_raw,
        })
        p_raw_list.append(p_raw)
        cond_list.append(cond)

    p_adj_arr = benjamini_hochberg(np.array(p_raw_list))

    for rec, p_adj in zip(records, p_adj_arr):
        rec["p_adj_BH"] = float(p_adj)

    results_df = pd.DataFrame(records)
    out_csv = outdir / f"pairwise_vs_{BASAL_COND}_BG{BG_SIGMA}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    sig_dict = {cond: (p_adj < 0.05) for cond, p_adj in zip(cond_list, p_adj_arr)}
    return results_df, sig_dict


# -------- plotting --------
def plot_scatter(df: pd.DataFrame, outdir: Path, p_value: float, sig_dict: dict):
    fig, ax = plt.subplots(figsize=(8, 4))

    x_pos = {cond: i for i, cond in enumerate(CONDITIONS)}

    all_vals = []
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].dropna().values
        all_vals.extend(vals)
        xpos = x_pos[cond]
        if len(vals) > 0:
            ax.scatter([xpos]*len(vals), vals, s=30)
            ax.hlines(np.median(vals), xpos-0.2, xpos+0.2, linewidth=1, color="0.2")

    #ax.set_xlabel("Condition")
    ax.set_ylabel("Collagen/DAPI (smoothed sum ratio)")

    if not all_vals:
        ax.set_title("Collagen/DAPI per field of view\nStars = BH-adjusted p < 0.05 vs basal")
        fig.savefig(outdir / "scatter_empty.pdf", dpi=300)
        return

    y_min, y_max = min(all_vals), max(all_vals)
    yr = y_max - y_min if y_max > y_min else 1.0

    ax.set_xticks([x_pos[c] for c in CONDITIONS])
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in CONDITIONS])

    basal_vals = df.loc[df["condition"] == BASAL_COND, "collagen_over_dapi_bg"].dropna().values
    median_basal = float(np.median(basal_vals))

    label_y = y_max + 0.05 * yr
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, "collagen_over_dapi_bg"].dropna().values
        if not len(vals):
            continue
        xpos = x_pos[cond]
        median_cond = float(np.median(vals))
        if cond == BASAL_COND:
            label = "baseline"
        else:
            diff = median_cond - median_basal
            ad = abs(diff)
            if ad == 0:
                label = "0"
            else:
                level = "strong" if ad > 2 else "mod" if ad >= 1 else "mild"
                arrow = "↑" if diff > 0 else "↓"
                label = f"{diff:+.2f} {arrow} {level}"
        ax.text(xpos, label_y, label, ha="center", va="bottom", fontsize=7)

    star_y = y_max + 0.10 * yr
    for cond in CONDITIONS:
        if cond != BASAL_COND and sig_dict.get(cond, False):
            ax.text(x_pos[cond], star_y, "*", ha="center", va="bottom", fontsize=12)

    # FIX 1 — move definition outside loop
    title_main = (
        "Collagen/DAPI per field of view\n"
        "Stars indicate Benjamini–Hochberg–adjusted p < 0.05 vs basal"
    )

    ax.set_title(title_main)

    ax.set_ylim(y_min - 0.05*yr, y_max + 0.20*yr)
    fig.tight_layout()
    out = outdir / f"{int(BG_SIGMA)}_scatter_collagen_over_dapi_bg.pdf"
    fig.savefig(out, dpi=300)
    plt.close(fig)
    print(f"[saved] {out}")


# -------- main --------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("-i", "--input", type=Path, default=INPUT_DIR)
    p.add_argument("-o", "--out", type=Path, default=OUTPUT_DIR)
    p.add_argument("-s", "--sigma", type=float, default=BG_SIGMA)
    return p.parse_args()


def main():
    global BG_SIGMA
    args = parse_args()
    BG_SIGMA = args.sigma

    out_dir = args.out
    out_dir.mkdir(exist_ok=True)

    paths = sorted(list(args.input.glob("*.tif")) + list(args.input.glob("*.tiff")))
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

    csv_path = out_dir / f"image_metrics_BG{BG_SIGMA}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    _, sig_dict = run_pairwise_vs_basal(df, out_dir)
    plot_scatter(df, out_dir, None, sig_dict)


if __name__ == "__main__":
    main()


# Example:
# python collagen_code.py --input ../input_images --out collagen_out --sigma 10
