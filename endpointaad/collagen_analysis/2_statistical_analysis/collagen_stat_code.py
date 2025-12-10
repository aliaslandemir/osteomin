#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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


# ----------------- STATS HELPERS ----------------- #

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (non-parametric effect size)."""
    a = np.asarray(a)
    b = np.asarray(b)
    greater = 0
    lesser = 0
    for x in a:
        greater += np.sum(x > b)
        lesser += np.sum(x < b)
    n_tot = a.size * b.size
    if n_tot == 0:
        return np.nan
    return (greater - lesser) / n_tot


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH FDR correction, returns list of adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []

    order = np.argsort(p)
    ranked_p = p[order]
    adj = ranked_p * m / (np.arange(m) + 1)

    # enforce monotone decreasing adjustment
    adj_rev = np.minimum.accumulate(adj[::-1])[::-1]
    adj_final = np.clip(adj_rev, 0.0, 1.0)

    p_adj = np.empty_like(adj_final)
    p_adj[order] = adj_final
    return p_adj.tolist()


def mannwhitney_z(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate z-score for Mann–Whitney U (no tie correction)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    mean_U = n1 * n2 / 2.0
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_U == 0:
        return np.nan
    return (U - mean_U) / std_U


def collagen_stats(
    cond_to_ratio: Dict[str, List[float]],
    conds_order: List[str],
    basal: str = "C1",
    alpha: float = 0.05,
) -> Tuple[float, float, Dict[str, bool], Dict[str, Dict[str, float]]]:
    """
    Kruskal–Wallis across all conditions, then Dunn-style
    Mann–Whitney vs basal (BH-corrected).
    """

    # Build groups for KW in the requested order
    groups = []
    conds_used = []
    for c in conds_order:
        vals = np.asarray(cond_to_ratio.get(c, []), dtype=float)
        if vals.size > 0:
            groups.append(vals)
            conds_used.append(c)

    if len(groups) < 2:
        return np.nan, np.nan, {}, {}

    H, p_kw = stats.kruskal(*groups)

    if basal not in cond_to_ratio or len(cond_to_ratio[basal]) == 0:
        raise ValueError(f"Basal condition '{basal}' has no data.")

    basal_vals = np.asarray(cond_to_ratio[basal], dtype=float)

    # pairwise vs basal
    names = []
    p_raws = []
    zs = []
    deltas = []
    ns = []
    means = []

    diff_dict: Dict[str, Dict[str, float]] = {}
    diff_dict[basal] = {
        "mean": float(np.mean(basal_vals)),
        "n": int(basal_vals.size),
        "p_raw": np.nan,
        "p_adj": np.nan,
        "z": np.nan,
        "delta": np.nan,
    }

    for c in conds_order:
        vals = np.asarray(cond_to_ratio.get(c, []), dtype=float)
        if c == basal:
            continue
        if vals.size == 0:
            diff_dict[c] = {
                "mean": np.nan,
                "n": 0,
                "p_raw": np.nan,
                "p_adj": np.nan,
                "z": np.nan,
                "delta": np.nan,
            }
            continue

        _, p_raw = stats.mannwhitneyu(basal_vals, vals, alternative="two-sided")
        z = mannwhitney_z(basal_vals, vals)
        delta = cliffs_delta(basal_vals, vals)

        names.append(c)
        p_raws.append(p_raw)
        zs.append(z)
        deltas.append(delta)
        ns.append(int(vals.size))
        means.append(float(np.mean(vals)))

    p_adjs = benjamini_hochberg(p_raws) if p_raws else []

    sig_dict: Dict[str, bool] = {}
    for i, c in enumerate(names):
        diff_dict[c] = {
            "mean": means[i],
            "n": ns[i],
            "p_raw": p_raws[i],
            "p_adj": p_adjs[i],
            "z": zs[i],
            "delta": deltas[i],
        }
        sig_dict[c] = p_adjs[i] < alpha

    return H, p_kw, sig_dict, diff_dict


# ----------------- ARG PARSING ----------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=(
            "Compute collagen/DAPI (Gaussian-smoothed sum ratio) from TIFF images "
            "and run Kruskal–Wallis + Dunn-style vs basal with BH correction."
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

    xs, ys = [], []
    for i, cond in enumerate(conds_c, start=1):
        for v in cond_to_ratio.get(cond, []):
            xs.append(i)
            ys.append(v)

    if not xs or not ys:
        print("No numeric values for collagen_over_dapi_bg, exiting.")
        return

    # ---------- STATS ----------
    H, p_kw, sig_dict, diff_dict = collagen_stats(
        cond_to_ratio, conds_c, basal=args.basal
    )

    print("\n===== Kruskal–Wallis across all conditions (collagen/DAPI) =====")
    print(f"H = {H:.3f}, p = {p_kw:.4g}")

    print("\n===== Per-condition stats vs basal (collagen/DAPI) =====")
    basal_mean = diff_dict.get(args.basal, {}).get("mean", np.nan)
    for cond in conds_c:
        info = diff_dict.get(cond, {})
        mean_val = info.get("mean", np.nan)
        n = int(info.get("n", 0))
        delta = info.get("delta", np.nan)
        if cond == args.basal:
            print(f"{cond} (Basal): mean = {mean_val:.4g}, n = {n}")
        else:
            p_raw = info.get("p_raw", np.nan)
            p_adj = info.get("p_adj", np.nan)
            z = info.get("z", np.nan)
            print(
                f"{cond}: mean = {mean_val:.4g}, n = {n}, "
                f"Δmean (vs basal) = {mean_val - basal_mean:+.4g}, "
                f"Cliff δ = {delta:+.3f}, "
                f"z = {z:.3f}, p_raw = {p_raw:.4g}, p_BH = {p_adj:.4g}, "
                f"significant = {sig_dict.get(cond, False)}"
            )

    means = {c: diff_dict.get(c, {}).get("mean", np.nan) for c in conds_c}
    mean_basal = means.get(args.basal, np.nan)

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(8, 4))

    # scatter + mean lines (with jitter)
    for i, cond in enumerate(conds_c, start=1):
        vals = cond_to_ratio.get(cond, [])
        if len(vals) > 0:
            x_jitter = np.random.normal(loc=i, scale=0.05, size=len(vals))
            ax.scatter(x_jitter, vals, s=30)
            mean_val = np.mean(vals)
            ax.hlines(mean_val, i - 0.2, i + 0.2, linewidth=1, color="0.2")

    ax.set_xticks(range(1, len(conds_c) + 1))
    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
    ax.set_xticklabels(labels)

    ax.set_ylabel("Collagen/DAPI (Gaussian-smoothed sum ratio)")

    # y-range + annotation heights
    y_min, y_max = min(ys), max(ys)
    yr = y_max - y_min if y_max > y_min else 1.0
    label_y = y_max + 0.05 * yr
    star_y = y_max + 0.12 * yr

    # Δ-mean labels + stars vs basal
    for i, cond in enumerate(conds_c, start=1):
        if cond == args.basal:
            ax.text(i, label_y, "baseline", ha="center", fontsize=7)
            continue

        diff = means[cond] - mean_basal
        if np.isnan(diff):
            lbl = "NA"
        else:
            arrow = "↑" if diff > 0 else "↓"
            lbl = f"{diff:+.2f} {arrow}"

        ax.text(i, label_y, lbl, ha="center", fontsize=7)

        if sig_dict.get(cond, False):
            ax.text(i, star_y, "*", ha="center", fontsize=12)

    ax.set_title(
        "Collagen/DAPI (Gaussian-smoothed sum ratio)\n"
        "Stars mark BH-corrected Mann–Whitney significance vs Basal"
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

python collagen_stat_code.py \
  -input input_images \
  --basal C1 \
  --out collagen_stats.pdf

"""
