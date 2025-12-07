#!/usr/bin/env python3
"""
dapi_stat_code.py

Quantify DAPI signal per field of view and test for differences
between conditions (Kruskal–Wallis + MWU vs basal + BH FDR).

Usage:
  python dapi_stat_code.py --input input_images --out dapi_output --sigma 10
"""

import argparse
from pathlib import Path
import re
import numpy as np
import pandas as pd
import tifffile as tf
from scipy import ndimage as ndi
from scipy import stats
import matplotlib.pyplot as plt

# ---------------- configurable defaults ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("dapi_output")

DAPI_CHANNEL = 0
BG_SIGMA = 10.0  # Gaussian sigma for background-smoothed sums
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
    """Map condition names to 1–2 line labels for plotting."""
    m = {
        "Basal": "Basal",
        "Osteo": r"A+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"A+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"A+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"A+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)": r"A+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


# -------------------- I/O + image helpers --------------------
def get_condition(stem: str) -> str:
    """
    Parse 'C1_', 'C2-', etc. from filename stem.
    Returns 'C1'..'C8' or 'Unknown'.
    """
    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_dapi_channel(path: Path) -> np.ndarray:
    """
    Read TIFF and extract DAPI channel as float64.
    Supports (C,H,W) or (T,C,H,W)/(Z,C,H,W) → uses first T/Z plane.
    """
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    if arr.ndim == 2:
        raise ValueError(f"2D TIFF found but channel count unknown: {path}")

    if arr.ndim == 3:
        # assume C,H,W
        if arr.shape[0] <= DAPI_CHANNEL:
            raise ValueError(
                f"Image {path} does not contain DAPI channel index {DAPI_CHANNEL}."
            )
        dapi = arr[DAPI_CHANNEL].astype(np.float64)

    elif arr.ndim == 4:
        # assume T,C,H,W or Z,C,H,W — take first T/Z plane
        if arr.shape[1] <= DAPI_CHANNEL:
            raise ValueError(
                f"Image {path} does not contain DAPI channel index {DAPI_CHANNEL}."
            )
        dapi = arr[0, DAPI_CHANNEL].astype(np.float64)

    else:
        raise ValueError(f"Unsupported image shape {arr.shape} in {path}")

    return dapi


def bg_smooth(img: np.ndarray) -> np.ndarray:
    """Gaussian smooth for robust total-intensity measure."""
    return ndi.gaussian_filter(img, BG_SIGMA)


def compute_metric(path: Path) -> dict:
    """
    Compute DAPI metrics for a single image:
    - sum_dapi_raw: raw total DAPI intensity
    - sum_dapi_bg:  total DAPI intensity after Gaussian smoothing
    """
    dapi_raw = read_dapi_channel(path)
    dapi_bg = bg_smooth(dapi_raw)

    sum_dapi_raw = float(np.sum(dapi_raw))
    sum_dapi_bg = float(np.sum(dapi_bg))

    return {
        "image": path.name,
        "condition": get_condition(path.stem),
        "sum_dapi_raw": sum_dapi_raw,
        "sum_dapi_bg": sum_dapi_bg,
    }


# -------------------- stats helpers --------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction."""
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


def run_kruskal(df: pd.DataFrame, outdir: Path, metric: str = "sum_dapi_bg"):
    """Global Kruskal–Wallis across conditions."""
    groups = []
    labels = []
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, metric].dropna().values
        if len(vals) > 0:
            groups.append(vals)
            labels.append(cond)

    if len(groups) < 2:
        print("Not enough groups for Kruskal–Wallis.")
        return np.nan, np.nan

    H, p = stats.kruskal(*groups)

    sigma_str = f"{BG_SIGMA:g}"
    out_txt = outdir / f"kruskal_{metric}_BG{sigma_str}.txt"
    with open(out_txt, "w") as f:
        f.write(f"Kruskal–Wallis test on {metric}\n")
        f.write("Conditions: " + ", ".join(labels) + "\n")
        f.write(f"H: {H:.4f}\n")
        f.write(f"p: {p:.4e}\n")
    print(f"[saved] {out_txt}")

    return H, p


def run_pairwise_vs_basal(df: pd.DataFrame, outdir: Path, metric: str = "sum_dapi_bg"):
    """Pairwise Mann–Whitney vs basal + BH correction."""
    basal_vals = df.loc[df["condition"] == BASAL_COND, metric].dropna().values
    if len(basal_vals) == 0:
        print("No basal data.")
        return pd.DataFrame(), {}

    records = []
    p_raw_list = []
    cond_list = []

    for cond in CONDITIONS:
        if cond == BASAL_COND:
            continue
        vals = df.loc[df["condition"] == cond, metric].dropna().values
        if len(vals) == 0:
            continue

        U, p_raw = stats.mannwhitneyu(basal_vals, vals, alternative="two-sided")
        rec = {
            "basal": BASAL_COND,
            "condition": cond,
            "metric": metric,
            "n_basal": len(basal_vals),
            "n_cond": len(vals),
            "mean_basal": float(np.mean(basal_vals)),
            "mean_cond": float(np.mean(vals)),
            "mean_diff": float(np.mean(vals) - np.mean(basal_vals)),
            "U": U,
            "p_raw": p_raw,
        }
        records.append(rec)
        p_raw_list.append(p_raw)
        cond_list.append(cond)

    p_adj_arr = benjamini_hochberg(np.array(p_raw_list))

    for rec, p_adj in zip(records, p_adj_arr):
        rec["p_adj_BH"] = float(p_adj)

    results_df = pd.DataFrame(records)
    sigma_str = f"{BG_SIGMA:g}"
    out_csv = outdir / f"pairwise_vs_{BASAL_COND}_{metric}_BG{sigma_str}.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[saved] {out_csv}")

    # store BH-adjusted p for quick plotting
    sig_dict = {cond: float(p_adj) for cond, p_adj in zip(cond_list, p_adj_arr)}
    return results_df, sig_dict


# -------------------- plotting --------------------
def plot_scatter(
    df: pd.DataFrame,
    outdir: Path,
    metric: str,
    p_kw: float,
    sig_dict: dict,
):
    """Scatter + mean + Δ-mean labels + BH stars (gated by KW)."""
    fig, ax = plt.subplots(figsize=(8, 4))

    x_pos = {cond: i for i, cond in enumerate(CONDITIONS, start=1)}

    all_vals = []
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, metric].dropna().values
        all_vals.extend(vals)
        xpos = x_pos[cond]
        if len(vals) > 0:
            ax.scatter([xpos] * len(vals), vals, s=30)
            mean_val = float(np.mean(vals))
            ax.hlines(mean_val, xpos - 0.2, xpos + 0.2, linewidth=1, color="0.2")

    if not all_vals:
        ax.set_title("No data for plotting.")
        fig.savefig(
            outdir / f"{metric}_scatter_empty.pdf", dpi=300, bbox_inches="tight"
        )
        plt.close(fig)
        return

    y_min, y_max = min(all_vals), max(all_vals)
    yr = y_max - y_min if y_max > y_min else 1.0

    ax.set_xticks([x_pos[c] for c in CONDITIONS])
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in CONDITIONS])

    # Basal statistics
    basal_vals = df.loc[df["condition"] == BASAL_COND, metric].dropna().values
    mean_basal = float(np.mean(basal_vals)) if len(basal_vals) > 0 else np.nan

    # Δ-mean labels (same style as collagen)
    label_y = y_max + 0.05 * yr
    for cond in CONDITIONS:
        vals = df.loc[df["condition"] == cond, metric].dropna().values
        if not len(vals):
            continue
        xpos = x_pos[cond]
        mean_cond = float(np.mean(vals))

        if cond == BASAL_COND or np.isnan(mean_basal):
            label = "baseline"
        else:
            diff = mean_cond - mean_basal
            ad = abs(diff)
            if ad == 0:
                label = "0"
            else:
                level = "strong" if ad > 2.0 else "mod" if ad >= 1.0 else "mild"
                arrow = "↑" if diff > 0 else "↓"
                label = f"{diff:+.2f} {arrow} {level}"

        ax.text(xpos, label_y, label, ha="center", va="bottom", fontsize=7)

    # Stars: only if KW significant AND BH-adjusted p < 0.05 vs basal
    star_y = y_max + 0.12 * yr
    if p_kw is not None and not np.isnan(p_kw) and p_kw < 0.05:
        for cond in CONDITIONS:
            if cond == BASAL_COND:
                continue
            p_adj = sig_dict.get(cond, np.nan)
            if not np.isnan(p_adj) and p_adj < 0.05:
                ax.text(x_pos[cond], star_y, "*", ha="center", va="bottom", fontsize=12)

    # Axis labels + title
    ax.set_ylabel("Total DAPI intensity (smoothed sum)")
    title_lines = [
        "DAPI total signal per field of view",
        "Stars: BH-adjusted p < 0.05 vs basal (only if Kruskal–Wallis p < 0.05)",
    ]
    if p_kw is not None and not np.isnan(p_kw):
        title_lines.append(f"Kruskal–Wallis p = {p_kw:.2e}")
    ax.set_title("\n".join(title_lines))

    ax.set_ylim(y_min - 0.05 * yr, y_max + 0.25 * yr)
    fig.tight_layout()

    out_pdf = outdir / f"{metric}_BG{BG_SIGMA}_scatter.pdf"
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {out_pdf}")


# -------------------- main --------------------
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

    # keep only known conditions (C1..C8)
    df = df[df["condition"].isin(CONDITIONS)]

    # Save raw metrics
    sigma_str = f"{BG_SIGMA:g}"
    csv_path = out_dir / f"dapi_image_metrics_BG{sigma_str}.csv"
    df.to_csv(csv_path, index=False)
    print(f"[saved] {csv_path}")

    metric = "sum_dapi_bg"

    # Quick summary table (n, mean, sd per condition)
    summary = (
        df.groupby("condition")[metric]
        .agg(n="count", mean="mean", sd="std")
        .reset_index()
    )
    summary_path = out_dir / f"dapi_summary_{metric}_BG{sigma_str}.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[saved] {summary_path}")

    # 1) Global Kruskal–Wallis
    H_kw, p_kw = run_kruskal(df, out_dir, metric=metric)

    # 2) Pairwise vs basal + BH — only used for stars if KW is significant
    if not np.isnan(p_kw) and p_kw < 0.05:
        print(f"Kruskal–Wallis significant (p = {p_kw:.4e}); running post-hoc tests.")
        _, sig_dict = run_pairwise_vs_basal(df, out_dir, metric=metric)
    else:
        print(
            f"Kruskal–Wallis not significant or invalid (p = {p_kw:.4e}); "
            "skipping post-hoc significance."
        )
        sig_dict = {}

    # 3) Plot
    plot_scatter(df, out_dir, metric=metric, p_kw=p_kw, sig_dict=sig_dict)


if __name__ == "__main__":
    main()
