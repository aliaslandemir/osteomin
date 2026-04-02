#!/usr/bin/env python3
"""Compute collagen image metrics and generate summary plots."""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats


# ---------------- defaults ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2

COND_CODE = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo + D_LK_L",
    "C4": "Osteo + D_LK_H",
    "C5": "Osteo + D_HK_L",
    "C6": "Osteo + D_HK_H",
    "C7": "Osteo + D_HK_0",
    "C8": "Osteo + D_0K_H",
}
CONDITIONS = list(COND_CODE.keys())

FAM_A_REF = "C2"
FAM_A_TARGETS = ["C1", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C2

# Assay colors (journal style, colorblind-safe)
BAR_COLOR = "#009E73"
POINT_COLOR = "#0A6F55"

# Plot colors
MEAN_COLOR = "#B22222"
SD_COLOR = "#5A5A5A"
STAR_COLOR_A = "#000000"  # vs C2

# =========================
# GLOBAL PLOT SETTINGS
# =========================
def apply_pub_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12.5,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


# ---------------- labels ----------------
def two_line_label(name: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": "Osteo",
        "Osteo + D_LK_L": r"Osteo" + "\n" + r"$D_LK_L$",
        "Osteo + D_LK_H": r"Osteo" + "\n" + r"$D_LK_H$",
        "Osteo + D_HK_L": r"Osteo" + "\n" + r"$D_HK_L$",
        "Osteo + D_HK_H": r"Osteo" + "\n" + r"$D_HK_H$",
        "Osteo + D_HK_0": r"Osteo" + "\n" + r"$D_HK_0$",
        "Osteo + D_0K_H": r"Osteo" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


def cond_id_to_pretty_name(c: str) -> str:
    return COND_CODE.get(c, c)


def cond_id_to_tag(c: str) -> str:
    return two_line_label(cond_id_to_pretty_name(c))

def _mean_sd(vals: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean, np.nan
    sd = float(np.std(vals, ddof=1))
    return mean, sd


# ---------------- IO ----------------
def get_condition_from_name(stem: str) -> str:
    import re

    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_channels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    if arr.ndim == 2:
        raise ValueError("2D TIFF found; expected multichannel.")

    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError("Not enough channels.")
        dapi = arr[DAPI_CHANNEL].astype(np.float64)
        col = arr[COLLAGEN_CHANNEL].astype(np.float64)
    elif arr.ndim == 4:
        # (T/Z,C,H,W) -> first plane
        if arr.shape[1] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError("Not enough channels.")
        dapi = arr[0, DAPI_CHANNEL].astype(np.float64)
        col = arr[0, COLLAGEN_CHANNEL].astype(np.float64)
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")

    return dapi, col


def compute_row(path: Path) -> dict:
    dapi_raw, col_raw = read_channels(path)

    A_total = float(dapi_raw.size)
    C_sum = float(np.sum(col_raw))
    D_sum = float(np.sum(dapi_raw))
    D_area = float(np.count_nonzero(dapi_raw > 0))

    M1_field = (C_sum / A_total) if A_total > 0 else np.nan
    M2_cov = (C_sum / D_area) if D_area > 0 else np.nan
    f_cov = (D_area / A_total) if A_total > 0 else np.nan
    M3_dsum = (C_sum / D_sum) if D_sum > 0 else np.nan

    return {
        "image": path.name,
        "condition": get_condition_from_name(path.stem),
        "C_sum": C_sum,
        "A_total": A_total,
        "D_area": D_area,
        "D_sum": D_sum,
        "M1_field": M1_field,
        "M2_cov": M2_cov,
        "M3_dsum": M3_dsum,
        "f_cov": f_cov,
    }


# ---------------- stats ----------------
def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def star_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""  # blank = ns


def kw_all_conditions(
    cond_to_vals: Dict[str, np.ndarray], conds_order: List[str]
) -> Tuple[float, float]:
    groups = []
    for c in conds_order:
        v = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            groups.append(v)
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = stats.kruskal(*groups)
    return float(H), float(p)


def pairwise_vs_ref_BH(
    cond_to_vals: Dict[str, np.ndarray],
    ref: str,
    targets: List[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    ref_vals = np.asarray(cond_to_vals.get(ref, np.array([])), dtype=float)
    ref_vals = ref_vals[np.isfinite(ref_vals)]

    out: Dict[str, Dict[str, float]] = {}
    names: List[str] = []
    p_raws: List[float] = []

    for c in targets:
        vals = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]

        if ref_vals.size == 0 or vals.size == 0:
            out[c] = {"p_raw": np.nan, "p_adj": np.nan, "sig": False}
            continue

        _, p_raw = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
        out[c] = {"p_raw": float(p_raw)}
        names.append(c)
        p_raws.append(float(p_raw))

    p_adjs = benjamini_hochberg(p_raws)
    for c, padj in zip(names, p_adjs):
        out[c]["p_adj"] = float(padj)
        out[c]["sig"] = bool(padj < alpha)
    return out


# =========================
# ALP-STYLE FIGURE (generic)
# =========================
def plot_tags_bar_delta_sig(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    suptitle: str,
    xlab: str,
    seed: int = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)

    data = [
        df.loc[df["condition"] == c, value_col].dropna().astype(float).values
        for c in conds
    ]

    fig, axB = plt.subplots(figsize=(6.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.305, top=0.885)

    x_step = 0.56
    x = np.arange(len(conds), dtype=float) * x_step

    means: List[float] = []
    sds: List[float] = []
    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        mean, sd = _mean_sd(vals)
        means.append(mean)
        sds.append(sd)
        axB.bar(
            x[i],
            mean,
            width=0.40,
            color=BAR_COLOR,
            edgecolor="#444444",
            linewidth=0.8,
            alpha=0.28,
            zorder=2,
        )
        if np.isfinite(mean):
            if np.isfinite(sd) and sd > 0.0:
                axB.errorbar(
                    x[i],
                    mean,
                    yerr=sd,
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.0,
                    capsize=4.0,
                    capthick=1.0,
                    zorder=4,
                )
            xj = x[i] + rng.normal(0, 0.020, size=vals.size)
            axB.scatter(
                xj,
                vals,
                s=32,
                marker="o",
                color=POINT_COLOR,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                zorder=5,
            )
            axB.scatter(
                x[i],
                mean,
                s=55,
                marker="o",
                color="#111111",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )

    finite_vals = np.concatenate([vals[np.isfinite(vals)] for vals in data if vals.size])
    finite_means = np.asarray([v for v in means if np.isfinite(v)], dtype=float)
    finite_sds = np.asarray([v for v in sds if np.isfinite(v)], dtype=float)
    ymax = 1.0
    if finite_vals.size and finite_means.size:
        ymax = max(
            float(np.max(finite_vals)),
            float(np.max(finite_means) + np.max(finite_sds) if finite_sds.size else np.max(finite_means)),
        )
    ypad_top = max(ymax * 0.22, 0.12)
    axB.set_ylim(0.0, ymax + ypad_top)
    axB.set_xlim(x[0] - 0.25, x[-1] + 0.25)

    labels = [two_line_label(cond_id_to_pretty_name(c)) for c in conds]
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=0, ha="center")
    axB.set_ylabel(xlab, fontsize=15.5)
    axB.set_xlabel("")
    axB.grid(False)
    axB.tick_params(axis="x", labelsize=12.0, pad=8)
    axB.tick_params(axis="y", labelsize=13.5)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_linewidth(0.8)
    axB.spines["bottom"].set_linewidth(0.8)

    y_sig = ymax + ypad_top * 0.60
    for i, c in enumerate(conds):
        st = "ref" if c == "C2" else star_from_p(famA.get(c, {}).get("p_adj", np.nan))
        axB.text(x[i], y_sig, st, ha="center", va="center", fontsize=13.8, color=STAR_COLOR_A)

    fig.suptitle(suptitle, fontsize=20, y=0.965)
    return fig


# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Compute collagen metrics and summary plots."
    )
    p.add_argument(
        "-input", type=Path, default=INPUT_DIR, help="Folder containing TIFF images."
    )
    p.add_argument(
        "--alpha", type=float, default=0.05, help="BH threshold (default 0.05)."
    )
    p.add_argument(
        "--out_prefix", type=str, default="collagen_raw", help="Output filename prefix."
    )
    p.add_argument("--out_dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    return p.parse_args()


def main():
    apply_pub_rcparams()

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(list(args.input.glob("*.tif")) + list(args.input.glob("*.tiff")))
    if paths:
        rows = []
        for pth in paths:
            try:
                rows.append(compute_row(pth))
            except Exception as e:
                print(f"[warn] {pth.name}: {e}")
        df = pd.DataFrame(rows)
        df = df[df["condition"].isin(CONDITIONS)].copy()
        if df.empty:
            print("No valid C1–C8 images after filtering.")
            return
    else:
        csv_path = args.out_dir / f"{args.out_prefix}_metrics_RAW.csv"
        if not csv_path.exists():
            print("No images found in", args.input)
            print("No fallback CSV found at", csv_path)
            return
        df = pd.read_csv(csv_path)
        df = df[df["condition"].isin(CONDITIONS)].copy()
        if df.empty:
            print("Fallback CSV has no valid C1–C8 rows:", csv_path)
            return
        print(f"Loaded fallback metrics CSV: {csv_path}")

    conds = [c for c in CONDITIONS if (df["condition"] == c).any()]
    print("Conditions present:", conds)

    if "C2" not in conds:
        raise ValueError("Missing C2 (Osteo).")

    # save metrics (RAW)
    csv_path = args.out_dir / f"{args.out_prefix}_metrics_RAW.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    def cond_to_vals(col: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for c in conds:
            v = df.loc[df["condition"] == c, col].dropna().astype(float).values
            if v.size:
                out[c] = v
        return out

    plot_specs = [
        ("M1_field", "Collagen density per field / Day 14", "Mean collagen intensity per pixel"),
        (
            "M2_cov",
            "Collagen per DAPI-positive area",
            "Collagen intensity / DAPI-positive pixel",
        ),
        (
            "M3_dsum",
            "Collagen per total DAPI intensity",
            "Collagen sum / DAPI sum (a.u.)",
        ),
        ("f_cov", "Coverage: DAPI-positive fraction", "DAPI-positive area fraction"),
    ]

    for col, title, xlab in plot_specs:
        cmap = cond_to_vals(col)

        H, pkw = kw_all_conditions(cmap, conds)

        famA = pairwise_vs_ref_BH(
            cmap,
            ref=FAM_A_REF,
            targets=[c for c in FAM_A_TARGETS if c in conds],
            alpha=args.alpha,
        )
        famB: Dict[str, Dict[str, float]] = {}

        # stats report per metric
        rpt = args.out_dir / f"{args.out_prefix}_{col}_RAW_stats.txt"
        with open(rpt, "w", encoding="utf-8") as f:
            f.write(f"{title} (RAW)\n")
            f.write(f"Metric column: {col}\n")
            f.write("Test: Mann–Whitney U (two-sided)\n")
            f.write("Multiple testing: Benjamini–Hochberg across C2-vs-target comparisons\n")
            f.write(f"Alpha: {args.alpha}\n\n")
            f.write(f"Kruskal–Wallis across all conditions: H={H:.4g}, p={pkw:.4g}\n\n")

            f.write("Pairwise comparisons vs C2\n")
            for c in [x for x in FAM_A_TARGETS if x in conds]:
                f.write(
                    f"  {c} vs C2: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                    f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n"
                )

        fig = plot_tags_bar_delta_sig(
            df=df,
            value_col=col,
            conds=conds,
            famA=famA,
            famB=famB,
            suptitle=title,
            xlab=xlab,
            seed=args.seed,
        )

        pdf = args.out_dir / f"{args.out_prefix}_{col}_RAW.pdf"
        png = args.out_dir / f"{args.out_prefix}_{col}_RAW.png"
        fig.savefig(pdf)
        fig.savefig(png)
        plt.close(fig)

        means_path = args.out_dir / f"{args.out_prefix}_{col}_RAW_means.txt"
        with means_path.open("w", encoding="utf-8") as f:
            f.write("Condition\tMean\tSD\tN\n")
            for c in conds:
                vals = cmap.get(c, np.array([]))
                if vals.size:
                    mean_val = float(np.mean(vals))
                    sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
                    n_val = int(vals.size)
                else:
                    mean_val = float("nan")
                    sd_val = float("nan")
                    n_val = 0
                f.write(f"{c}\t{mean_val:.6g}\t{sd_val:.6g}\t{n_val}\n")

        print(f"Saved: {pdf}")
        print(f"Saved: {png}")
        print(f"Saved: {rpt}")
        print(f"Saved: {means_path}")


if __name__ == "__main__":
    main()
