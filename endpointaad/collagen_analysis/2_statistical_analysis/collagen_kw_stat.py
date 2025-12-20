#!/usr/bin/env python3
"""
Publication-ready collagen metrics — RAW / ABSOLUTE (NO fold normalization)
Dual-family stats + collagen/ALP-style plot

Based on your current collagen script :contentReference[oaicite:0]{index=0}, but UPDATED to:
- REMOVE fold vs C1 median completely
- Run stats on RAW metrics (absolute)
- Plot RAW metrics (absolute) with same violin + points + mean±SEM + dual star rows + Δmean row

Per image (RAW):
- C_sum   = sum(collagen intensities)
- A_total = total # pixels
- D_area  = # pixels where DAPI > 0  (assumes true background pixels are exactly 0)

RAW metrics:
- M1_field = C_sum / A_total   (collagen density per field; intensity/pixel)
- M2_cov   = C_sum / D_area    (collagen per DAPI-positive area; intensity/pixel in nuclei+cell regions proxy)
- f_cov    = D_area / A_total  (coverage proxy; fraction of pixels DAPI-positive)
- M3_dsum  = C_sum / D_sum     (collagen per total DAPI intensity; intensity/intensity)

Stats (per RAW metric):
- Global Kruskal–Wallis across all present conditions (reported)
- Family A (lower, black): C1 vs (C2–C8), Mann–Whitney U + BH
- Family B (upper, red):   C2 vs (C3–C8), Mann–Whitney U + BH (excluding C1)

Outputs:
- CSV metrics (RAW)
- one stats report TXT per RAW metric
- figures (PDF + PNG): M1_field, M2_cov, M3_dsum, f_cov

Run:
python collagen_publishable_dualref_RAW_pro.py \
  -input ../../../confocal \
  --alpha 0.05 \
  --out_prefix collagen_raw \
  --out_dir collagen_output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tf
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


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

# Comparison families
FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C1
FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]        # vs C2 (exclude C1)

# ---- Publication-safe palette ----
COND_COLORS = {
    "C1": "#2B2B2B",  # Basal: charcoal
    "C2": "#0072B2",  # Osteo ref: blue
    "C3": "#009E73",  # green
    "C4": "#E69F00",  # orange
    "C5": "#CC79A7",  # purple/pink
    "C6": "#D55E00",  # vermillion
    "C7": "#56B4E9",  # light blue
    "C8": "#8C6D31",  # brown/olive
}
VIOLIN_FACE = "#E6E6E6"
VIOLIN_EDGE = "#B3B3B3"

STAR_COLOR_A = "#000000"   # black (vs C1)
STAR_COLOR_B = "#B22222"   # dark red (vs C2)

SEPARATOR_LINE = "#F2F2F2"


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
    return ""


def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))


def kw_all_conditions(cond_to_vals: Dict[str, np.ndarray], conds_order: List[str]) -> Tuple[float, float]:
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


# ---------------- plotting ----------------
def make_positions_with_gaps(conds: List[str]) -> Tuple[List[float], List[float]]:
    xs = []
    base = 0.0
    for c in conds:
        base += 1.0
        if c == "C2":
            base += 0.30  # gap after C1
        if c == "C7":
            base += 0.35  # gap after C6
        xs.append(base)

    idx = {c: i for i, c in enumerate(conds)}
    seps = []
    if "C1" in idx and "C2" in idx:
        seps.append((xs[idx["C1"]] + xs[idx["C2"]]) / 2.0)
    if "C6" in idx and "C7" in idx:
        seps.append((xs[idx["C6"]] + xs[idx["C7"]]) / 2.0)
    return xs, seps


def mean_diff_labels_vs_c1(df: pd.DataFrame, value_col: str, conds: List[str]) -> List[str]:
    """Mean difference (RAW) vs C1 mean, with arrow."""
    ref_mean = df.loc[df["condition"] == "C1", value_col].mean()
    out = []
    for c in conds:
        if c == "C1":
            out.append("baseline")
        else:
            m = df.loc[df["condition"] == c, value_col].mean()
            d = m - ref_mean
            arrow = "↑" if d > 0 else "↓"
            # formatting: keep it readable across metrics
            if np.isnan(d):
                out.append("NA")
            else:
                if abs(d) < 0.005:
                    d = 0.0
                if abs(d) >= 1000:
                    out.append(f"{d:+.0f} {arrow}")
                elif abs(d) >= 10:
                    out.append(f"{d:+.1f} {arrow}")
                else:
                    out.append(f"{d:+.2f} {arrow}")
    return out


def add_stats_header(ax, fs=8.6, grey="0.35", red="#B22222"):
    t0 = TextArea("Stars: * p<0.05, ** p<0.01, *** p<0.001 (BH)",
                  textprops=dict(color=grey, fontsize=fs))
    s0 = TextArea("   |   ", textprops=dict(color=grey, fontsize=fs))
    t1 = TextArea("Lower (black): vs C1 (MWU + BH)",
                  textprops=dict(color=grey, fontsize=fs))
    s1 = TextArea("   |   ", textprops=dict(color=grey, fontsize=fs))
    t2 = TextArea("Upper (red): vs C2 (MWU + BH, excl. C1)",
                  textprops=dict(color=red, fontsize=fs))

    row = HPacker(children=[t0, s0, t1, s1, t2], align="center", pad=0, sep=0)
    box = AnchoredOffsetbox(
        loc="upper center",
        child=row,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    box.set_clip_on(False)
    ax.add_artist(box)


def plot_violin_pro(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    title: str,
    ylab: str,
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    seed: int = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    xs, seps = make_positions_with_gaps(conds)
    data = [df.loc[df["condition"] == c, value_col].dropna().astype(float).values for c in conds]

    allv = np.concatenate([v for v in data if v.size > 0]) if any(v.size > 0 for v in data) else np.array([0.0])
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    yr = (y_max - y_min) if y_max > y_min else 1.0

    yA = y_max + 0.045 * yr   # black stars
    yB = y_max + 0.080 * yr   # red stars

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    vp = ax.violinplot(
        data,
        positions=xs,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_alpha(1.0)
        body.set_linewidth(1.0)

    for s in seps:
        ax.axvline(s, color="white", linewidth=12, zorder=1)
        ax.axvline(s, color=SEPARATOR_LINE, linewidth=1.5, zorder=2)

    for x, c, vals in zip(xs, conds, data):
        if vals.size == 0:
            continue
        jitter = rng.normal(0, 0.055, size=vals.size)
        ax.scatter(
            x + jitter,
            vals,
            s=40,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        m = float(np.mean(vals))
        s = sem(vals)
        ax.errorbar(
            [x], [m], yerr=[s],
            fmt="o",
            color="black",
            markersize=4.5,
            capsize=3.2,
            linewidth=1.1,
            zorder=4,
        )

    for x, c in zip(xs, conds):
        if c in FAM_A_TARGETS:
            st = star_from_p(famA.get(c, {}).get("p_adj", np.nan))
            if st:
                ax.text(x, yA + 0.015 * yr, st, ha="center", va="bottom",
                        fontsize=12, color=STAR_COLOR_A, zorder=6)
        if c in FAM_B_TARGETS:
            st = star_from_p(famB.get(c, {}).get("p_adj", np.nan))
            if st:
                ax.text(x, yB + 0.015 * yr, st, ha="center", va="bottom",
                        fontsize=12, color=STAR_COLOR_B, zorder=6)

    delta_labels = mean_diff_labels_vs_c1(df, value_col, conds)
    y_delta = y_min - 0.11 * yr
    for x, lab in zip(xs, delta_labels):
        ax.text(x, y_delta, lab, ha="center", va="top", fontsize=8.2, color="0.35")

    ax.text(xs[0] - 0.75, y_delta, r"$\Delta$mean", ha="left", va="top",
            fontsize=8.2, color="0.35", clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in conds], fontsize=9)

    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(y_min - 0.17 * yr, y_max + 0.26 * yr)

    fig.tight_layout()
    add_stats_header(ax)
    return fig


# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Collagen RAW metrics + dual-family MWU+BH + pro plots (no fold).")
    p.add_argument("-input", type=Path, default=INPUT_DIR, help="Folder containing TIFF images.")
    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")
    p.add_argument("--out_prefix", type=str, default="collagen_raw", help="Output filename prefix.")
    p.add_argument("--out_dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    return p.parse_args()


def main():
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

    if "C1" not in conds:
        raise ValueError("Missing C1 (Basal).")
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

    # ---- RAW plots ----
    # NOTE: units are "intensity per pixel" for M1/M2, "ratio" for f_cov, "intensity/intensity" for M3
    plot_specs = [
        ("M1_field", "Collagen density per field", "Mean collagen intensity per pixel"),
        ("M2_cov", "Collagen per DAPI-positive area", "Collagen intensity / DAPI-positive pixel"),
        ("M3_dsum", "Collagen per total DAPI intensity", "Collagen sum / DAPI sum (a.u.)"),
        ("f_cov", "Coverage: DAPI-positive fraction", "DAPI-positive area fraction"),
    ]

    for col, title, ylab in plot_specs:
        cmap = cond_to_vals(col)

        H, pkw = kw_all_conditions(cmap, conds)

        famA = pairwise_vs_ref_BH(
            cmap,
            ref=FAM_A_REF,
            targets=[c for c in FAM_A_TARGETS if c in conds],
            alpha=args.alpha,
        )
        famB = pairwise_vs_ref_BH(
            cmap,
            ref=FAM_B_REF,
            targets=[c for c in FAM_B_TARGETS if c in conds],
            alpha=args.alpha,
        )

        # stats report per plot
        rpt = args.out_dir / f"{args.out_prefix}_{col}_RAW_stats.txt"
        with open(rpt, "w", encoding="utf-8") as f:
            f.write(f"{title} (RAW)\n")
            f.write(f"Metric column: {col}\n")
            f.write("Test: Mann–Whitney U (two-sided)\n")
            f.write("Multiple testing: Benjamini–Hochberg within each family\n")
            f.write(f"Alpha: {args.alpha}\n\n")
            f.write(f"Kruskal–Wallis across all conditions: H={H:.4g}, p={pkw:.4g}\n\n")

            f.write("Family A (black): C1 vs targets\n")
            for c in [x for x in FAM_A_TARGETS if x in conds]:
                f.write(
                    f"  {c} vs C1: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                    f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n"
                )

            f.write("\nFamily B (red): C2 vs targets (excluding C1)\n")
            for c in [x for x in FAM_B_TARGETS if x in conds]:
                f.write(
                    f"  {c} vs C2: p_raw={famB.get(c, {}).get('p_raw', np.nan):.4g}, "
                    f"p_adj={famB.get(c, {}).get('p_adj', np.nan):.4g}\n"
                )

        fig = plot_violin_pro(
            df=df,
            value_col=col,
            conds=conds,
            title=title,
            ylab=ylab,
            famA=famA,
            famB=famB,
            seed=args.seed,
        )

        pdf = args.out_dir / f"{args.out_prefix}_{col}_RAW.pdf"
        png = args.out_dir / f"{args.out_prefix}_{col}_RAW.png"
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {pdf}")
        print(f"Saved: {png}")
        print(f"Saved: {rpt}")


if __name__ == "__main__":
    main()


"""
Run:

python collagen_kw_stat.py \
  -input ../../../confocal \
  --alpha 0.05 \
  --out_prefix collagen_raw \
  --out_dir collagen_output
  
"""
