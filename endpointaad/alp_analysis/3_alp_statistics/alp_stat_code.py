#!/usr/bin/env python3
"""
Publication-ready ALP (technical replicates) — RAW / ABSOLUTE mU/mL
PLOT STYLE UPDATED to match your reference style:
- violins (light grey)
- small colored jittered points
- mean marker (red circle) + median marker (dark diamond)
- mean/median legend
- dual-family stars (black vs C1; red vs C2)
- separators (C2–C3 and C6–C7)
- Δmean row (mean difference vs C1) at bottom (IN mU/mL)
- descriptive stats header

Keeps the SAME IO:
- parse_metadata(metadata.txt)  -> well_to_cond, cond_order
- read_values_xlsx(data.xlsx, well_to_cond) -> cond_to_values (raw absorbance)

Pipeline:
1) blank-correct + convert to mU/mL (clamped to >=0)
2) Stats on RAW ALP_mU_mL (absolute):
   - Global Kruskal–Wallis across all present conditions (reported)
   - Family A (lower black stars): C1 vs C2–C8, Mann–Whitney U (two-sided) + BH
   - Family B (upper red stars):   C2 vs C3–C8, Mann–Whitney U (two-sided) + BH (excluding C1)
3) Plot as above

Outputs:
- CSV: per-well absorbance + ALP_mU_mL
- TXT: stats report
- Figure: PDF + PNG (RAW)

Run:
python alp_stat_code.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output
"""

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from matplotlib.lines import Line2D

from pipeline import parse_metadata, read_values_xlsx


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

# Comparison families
FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C1
FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]        # vs C2 (exclude C1)

# ---- Publication-safe palette (Okabe–Ito-ish; distinct) ----
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

# mean/median markers (match reference style)
MEAN_COLOR   = "#B22222"   # red mean circle
MEDIAN_COLOR = "#2B2B2B"   # dark median diamond


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


# -------------------- stats helpers --------------------

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


# -------------------- plotting helpers --------------------

def make_positions_with_gaps(conds: List[str]) -> Tuple[List[float], List[float]]:
    xs = []
    base = 0.0
    for c in conds:
        base += 1.0
        if c == "C3":
            base += 0.30   # gap between C2–C3
        if c == "C7":
            base += 0.35   # gap between C6–C7
        xs.append(base)

    idx = {c: i for i, c in enumerate(conds)}
    seps = []
    if "C2" in idx and "C3" in idx:
        seps.append((xs[idx["C2"]] + xs[idx["C3"]]) / 2.0)
    if "C6" in idx and "C7" in idx:
        seps.append((xs[idx["C6"]] + xs[idx["C7"]]) / 2.0)
    return xs, seps


def mean_diff_labels_vs_c1(df: pd.DataFrame, value_col: str, conds: List[str]) -> List[str]:
    ref_mean = df.loc[df["condition"] == "C1", value_col].mean()
    out = []
    for c in conds:
        if c == "C1":
            out.append("baseline")
        else:
            m = df.loc[df["condition"] == c, value_col].mean()
            d = m - ref_mean
            arrow = "↑" if d > 0 else "↓"
            if abs(d) < 0.005:
                d = 0.0
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


def add_mean_median_legend(ax):
    handles = [
        Line2D([], [], marker="o", linestyle="None",
               markerfacecolor=MEAN_COLOR, markeredgecolor="white",
               markersize=7, label="Mean"),
        Line2D([], [], marker="D", linestyle="None",
               markerfacecolor=MEDIAN_COLOR, markeredgecolor="white",
               markersize=7, label="Median"),
    ]
    ax.legend(handles=handles, loc="upper left", fontsize=8.5,
              frameon=False, handletextpad=0.4, borderaxespad=0.3)


def plot_violin(
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

    any_nonempty = any(v.size > 0 for v in data)
    allv = np.concatenate([v for v in data if v.size > 0]) if any_nonempty else np.array([0.0])
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    yr = (y_max - y_min) if y_max > y_min else 1.0

    yA = y_max + 0.045 * yr   # black stars
    yB = y_max + 0.080 * yr   # red stars

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.12)

    vp = ax.violinplot(
        data,
        positions=xs,
        widths=0.72,
        showmeans=False, showmedians=False, showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_alpha(1.0)
        body.set_linewidth(1.0)

    for s in seps:
        ax.axvline(s, color="white", linewidth=12, zorder=1)
        ax.axvline(s, color=SEPARATOR_LINE, linewidth=1.5, zorder=2)

    # points + mean/median markers
    for x, c, vals in zip(xs, conds, data):
        if vals.size == 0:
            continue

        jitter = rng.normal(0, 0.055, size=vals.size)
        ax.scatter(
            x + jitter, vals,
            s=22,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )

        mean_val = float(np.mean(vals))
        median_val = float(np.median(vals))

        ax.scatter(
            x, mean_val,
            s=55, marker="o",
            color=MEAN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.60, 
            zorder=5,
        )
        ax.scatter(
            x, median_val,
            s=60, marker="D",
            color=MEDIAN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.60, 
            zorder=5,
        )

    # stars (two rows)
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

    # Δmean row (vs C1)
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

    add_stats_header(ax)
    add_mean_median_legend(ax)
    fig.tight_layout()
    return fig


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="ALP RAW mU/mL stats + pro violin plot (MWU+BH dual-family).")
    p.add_argument("--metadata", required=True, help="Metadata text file (same format as before).")
    p.add_argument("--data", required=True, help="Raw ALP data xlsx (same as before).")
    p.add_argument("--blank", type=float, default=0.066, help="Blank absorbance (default: 0.066).")
    p.add_argument("--conversion", type=float, required=True,
                   help="Conversion factor from net absorbance to mU/mL.")
    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")
    p.add_argument("--out_prefix", type=str, default="alp_raw", help="Output filename prefix.")
    p.add_argument("--out_dir", type=Path, default=Path("alp_output"), help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read metadata and raw values (UNCHANGED)
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)  # raw absorbance

    # 2) Convert to per-well mU/mL
    rows = []
    for cond, vals in cond_to_values.items():
        if cond not in CONDITIONS:
            continue
        for i, v in enumerate(vals):
            try:
                v = float(v)
            except Exception:
                continue
            v_net = v - float(args.blank)
            mU = max(v_net * float(args.conversion), 0.0)
            rows.append({
                "condition": cond,
                "well_index": i + 1,
                "abs_raw": v,
                "abs_net": v_net,
                "ALP_mU_mL": mU,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid ALP values found for conditions C1–C8.")
        return

    # 3) Restrict to conditions present in metadata order
    conds = [c for c in cond_order if (c in CONDITIONS and (df["condition"] == c).any())]
    if not conds:
        print("No conditions C1–C8 found in metadata order; exiting.")
        return

    if "C1" not in conds:
        raise ValueError("Missing C1 (Basal).")
    if "C2" not in conds:
        raise ValueError("Missing C2 (Osteo).")

    # Save CSV (per-well, RAW)
    csv_path = args.out_dir / f"{args.out_prefix}_alp_metrics_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Convenience: map condition->values for RAW column
    cond_to_vals: Dict[str, np.ndarray] = {}
    for c in conds:
        v = df.loc[df["condition"] == c, "ALP_mU_mL"].dropna().astype(float).values
        if v.size:
            cond_to_vals[c] = v

    # 4) Global KW + pairwise MWU+BH families
    H, pkw = kw_all_conditions(cond_to_vals, conds)

    famA = pairwise_vs_ref_BH(
        cond_to_vals,
        ref=FAM_A_REF,
        targets=[c for c in FAM_A_TARGETS if c in conds],
        alpha=args.alpha,
    )
    famB = pairwise_vs_ref_BH(
        cond_to_vals,
        ref=FAM_B_REF,
        targets=[c for c in FAM_B_TARGETS if c in conds],
        alpha=args.alpha,
    )

    # 5) Stats report
    rpt = args.out_dir / f"{args.out_prefix}_ALP_mU_mL_stats.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("ALP activity — RAW mU/mL (30-min pNPP assay)\n")
        f.write("Metric column: ALP_mU_mL\n")
        f.write("Per-well values treated as independent technical replicates.\n")
        f.write("Test: Mann–Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini–Hochberg within each family\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Blank absorbance: {args.blank}\n")
        f.write(f"Conversion factor: {args.conversion} (net abs → mU/mL)\n\n")
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

    print(f"Saved stats: {rpt}")

    # 6) Plot (RAW mU/mL) — updated style
    fig = plot_violin(
        df=df,
        value_col="ALP_mU_mL",
        conds=conds,
        title="ALP activity (30-min pNPP assay)",
        ylab=r"ALP activity (mU·mL$^{-1}$)",
        famA=famA,
        famB=famB,
        seed=args.seed,
    )

    pdf = args.out_dir / f"{args.out_prefix}_ALP_mU_mL.pdf"
    png = args.out_dir / f"{args.out_prefix}_ALP_mU_mL.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()



"""
Run:

python alp_stat_code.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output
  
"""
