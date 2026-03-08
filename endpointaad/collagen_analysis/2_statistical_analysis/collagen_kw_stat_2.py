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
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter


# ---------------- defaults ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2

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

FAM_A_REF = "C2"
FAM_A_TARGETS = ["C1", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C2

# Colors
COND_COLORS = {
    "C1": "#BDCC16",
    "C2": "#1F5AA6",
    "C3": "#1B7F5A",
    "C4": "#C97A00",
    "C5": "#A65C8A",
    "C6": "#B24C1A",
    "C7": "#6AAED6",
    "C8": "#7A6A3A",
}

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
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


# ---------------- labels ----------------
def two_line_label(name: str) -> str:
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
def plot_tags_violin_delta_sig(
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

    nonempty = [v for v in data if v.size > 0]
    if nonempty:
        allv = np.concatenate(nonempty)
        xmin = float(np.min(allv))
        xmax = float(np.max(allv))
        xr = (xmax - xmin) if xmax > xmin else (abs(xmax) if xmax != 0 else 1.0)
        pad = 0.10 * xr
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = (0.0, 1.0)

    fig = plt.figure(figsize=(12.8, 6.0))
    fig.subplots_adjust(left=0.04, right=0.995, bottom=0.12, top=0.82, wspace=0.16)

    gs = GridSpec(
        1,
        3,
        width_ratios=[1.05, 6.10, 0.95],
        wspace=0.12,
        figure=fig,
    )

    axT = fig.add_subplot(gs[0, 0])
    axV = fig.add_subplot(gs[0, 1], sharey=axT)
    axS = fig.add_subplot(gs[0, 2], sharey=axT)

    y = np.arange(len(conds))
    axT.set_ylim(len(conds) - 0.5, -0.5)

    row_color, row_alpha, row_lw = "#000000", 0.10, 0.8
    for yy in y:
        for ax in (axT, axV, axS):
            ax.axhline(yy, color=row_color, alpha=row_alpha, linewidth=row_lw, zorder=0)

    title_pad = 6

    axT.set_xlim(0, 1)
    axT.set_xticks([])
    axT.set_yticks(y)
    axT.set_yticklabels([])
    axT.set_title("Condition", pad=title_pad)

    for i, c in enumerate(conds):
        axT.text(
            0.5,
            i,
            cond_id_to_tag(c),
            ha="center",
            va="center",
            fontsize=11,
            color="#1a1a1a",
            linespacing=1.05,
        )
    for s in axT.spines.values():
        s.set_visible(False)

    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            continue
        yj = i + rng.normal(0, 0.022, size=vals.size)
        axV.scatter(
            vals,
            yj,
            s=24,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.9,
            zorder=4,
        )
        mean, sd = _mean_sd(vals)
        if np.isfinite(mean):
            if np.isfinite(sd):
                lo = mean - sd
                hi = mean + sd
                axV.hlines(i, lo, hi, color=SD_COLOR, linewidth=1.2, alpha=0.24, zorder=2)
                axV.scatter(
                    [lo, hi],
                    [i, i],
                    marker="|",
                    s=105,
                    color=SD_COLOR,
                    alpha=0.24,
                    linewidths=1.4,
                    zorder=2,
                )
            axV.scatter(
                mean,
                i,
                s=96,
                marker="o",
                color=COND_COLORS.get(c, MEAN_COLOR),
                edgecolor="black",
                linewidth=0.9,
                zorder=5,
            )

    axV.set_xlim(*xlim)
    axV.set_xlabel("")
    axV.set_title(xlab, pad=title_pad)
    axV.grid(axis="x", alpha=0.05, linewidth=0.45)
    axV.set_yticks([])
    axV.yaxis.set_major_formatter(NullFormatter())
    axV.tick_params(axis="y", length=0, labelsize=11)
    for s in axV.spines.values():
        s.set_visible(False)

    axS.set_xlim(-0.6, 0.6)
    axS.set_yticks([])
    axS.yaxis.set_major_formatter(NullFormatter())
    axS.tick_params(axis="y", length=0)
    axS.set_xticks([])

    axS.set_title("Significance", pad=title_pad)
    axS.axvspan(-0.6, 0.6, color="0.0", alpha=0.008, zorder=0)

    axS.text(
        0.5,
        -0.09,
        "vs C2",
        transform=axS.transAxes,
        ha="center",
        va="top",
        fontsize=9,
        color="0.45",
    )

    for i, c in enumerate(conds):
        if c == "C1":
            axS.text(0.0, i, "ref", ha="center", va="center", fontsize=12, color=STAR_COLOR_A)
        else:
            st = star_from_p(famA.get(c, {}).get("p_adj", np.nan))
            axS.text(
                0.0, i, st, ha="center", va="center", fontsize=10, color=STAR_COLOR_A
            )

    for s in axS.spines.values():
        s.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=0.955)
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
    if not paths:
        print("No images found in", args.input)
        return

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
        ("M1_field", "Collagen density per field / 336 hours", "Mean collagen intensity per pixel"),
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

        fig = plot_tags_violin_delta_sig(
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
