#!/usr/bin/env python3
"""
Figure 2 ONLY — ALP: condition tags + distribution + Δmean + significance

Publication-readiness pass (minimal changes):
- Fix top overlaps (suptitle + axes titles + annotation)
- Keep sizes, style, and analysis intact

Run:
python alp_stat_code2.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output2
"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

from pipeline import parse_metadata, read_values_xlsx


# =========================
# GLOBAL PLOT SETTINGS (PLOS-ish)
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


# =========================
# CONDITIONS
# =========================
CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C1
FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]  # vs C2

COND_COLORS = {
    "C1": "#7a7a7a",
    "C2": "#3b6fb6",
    "C3": "#2a9d8f",
    "C4": "#e0a100",
    "C5": "#b565a7",
    "C6": "#c84c3a",
    "C7": "#6aaed6",
    "C8": "#8c6d31",
}

MEAN_COLOR = "#8b1e1e"
SD_COLOR = "#5a5a5a"

STAR_COLOR_A = "#000000"
STAR_COLOR_B = "#8b1e1e"


# =========================
# CONDITION TAGS
# =========================
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
    m = {
        "C1": "Basal",
        "C2": "Osteo",
        "C3": "Osteo+D/MK4 (L/L)",
        "C4": "Osteo+D/MK4 (L/H)",
        "C5": "Osteo+D/MK4 (H/L)",
        "C6": "Osteo+D/MK4 (H/H)",
        "C7": "Osteo+Dex (H)",
        "C8": "Osteo+MK-4 (H)",
    }
    return m.get(c, c)


def cond_id_to_tag(c: str) -> str:
    return two_line_label(cond_id_to_pretty_name(c))


# =========================
# STATS HELPERS
# =========================
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
        return "-"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


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
# FIGURE 2
# =========================
def _mean_sd(
    vals: np.ndarray,
) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean, np.nan
    sd = float(np.std(vals, ddof=1))
    return mean, sd


def plot_figure2_tags_violin_delta_sig(
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

    ref_vals = df.loc[df["condition"] == "C1", value_col].dropna().astype(float).values
    ref_mean = float(np.mean(ref_vals)) if ref_vals.size else np.nan

    dmeans: List[float] = []
    for c in conds:
        vals = df.loc[df["condition"] == c, value_col].dropna().astype(float).values
        if vals.size == 0 or not np.isfinite(ref_mean):
            dmeans.append(np.nan)
        else:
            dmeans.append(float(np.mean(vals) - ref_mean))

    def fmt_dmean(d: float) -> str:
        if not np.isfinite(d):
            return ""
        if abs(d) < 0.005:
            d = 0.0
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        return f"{d:+.2f} {arrow}"

    # ---- layout: REMOVE constrained_layout to prevent title collisions ----
    fig = plt.figure(figsize=(15.0, 6.3))
    # manual margins: more top room so nothing overlaps
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.12, top=0.86, wspace=0.16)

    gs = GridSpec(
        1,
        5,
        width_ratios=[1.00, 0.50, 5.40, 0.80, 1.00],
        wspace=0.12,
        figure=fig,
    )

    axT = fig.add_subplot(gs[0, 0])
    axID = fig.add_subplot(gs[0, 1], sharey=axT)
    axV = fig.add_subplot(gs[0, 2], sharey=axT)
    axD = fig.add_subplot(gs[0, 3], sharey=axT)
    axS = fig.add_subplot(gs[0, 4], sharey=axT)

    y = np.arange(len(conds))
    axT.set_ylim(len(conds) - 0.5, -0.5)

    # row guides
    row_color, row_alpha, row_lw = "#000000", 0.10, 0.8
    for yy in y:
        for ax in (axT, axID, axV, axD, axS):
            ax.axhline(yy, color=row_color, alpha=row_alpha, linewidth=row_lw, zorder=0)

    title_pad = 10  # smaller pad avoids crowding near top

    # ======================
    # Tag column
    # ======================
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
            fontsize=12,
            color="#1a1a1a",
            linespacing=1.10,
        )
    for s in axT.spines.values():
        s.set_visible(False)

    # ======================
    # ID column
    # ======================
    axID.set_xlim(0, 1)
    axID.set_xticks([])
    axID.set_yticks(y)
    axID.set_yticklabels([])
    axID.set_title("ID", pad=title_pad)

    for i, c in enumerate(conds):
        axID.text(
            0.5,
            i,
            c,
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#1a1a1a",
        )
    for s in axID.spines.values():
        s.set_visible(False)

    # ======================
    # Mean ± SD + raw points
    # ======================
    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            continue
        yj = i + rng.normal(0, 0.035, size=vals.size)
        axV.scatter(
            vals,
            yj,
            s=20,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.75,
            zorder=3,
        )
        mean, sd = _mean_sd(vals)
        if np.isfinite(mean):
            if np.isfinite(sd):
                lo = mean - sd
                hi = mean + sd
                axV.hlines(
                    i, lo, hi, color=SD_COLOR, linewidth=2.2, alpha=0.45, zorder=4
                )
                axV.scatter(
                    [lo, hi],
                    [i, i],
                    marker="|",
                    s=170,
                    color=SD_COLOR,
                    alpha=0.45,
                    linewidths=2.2,
                    zorder=4,
                )
                axV.scatter(
                    mean,
                    i,
                    s=70,
                    marker="o",
                    color=COND_COLORS.get(c, MEAN_COLOR),
                    edgecolor="black",
                    linewidth=0.7,
                    zorder=5,
                )
            else:
                axV.scatter(
                    mean,
                    i,
                    s=60,
                    marker="o",
                    color=COND_COLORS.get(c, MEAN_COLOR),
                    edgecolor="black",
                    linewidth=0.7,
                    alpha=0.90,
                    zorder=5,
                )

    axV.set_xlabel(xlab)
    axV.set_title("ALP Mean ± SD (pNPP assay 30 min / mU·mL$^{-1}$)", pad=14)
    axV.grid(axis="x", alpha=0.15)
    axV.set_yticks([])
    axV.yaxis.set_major_formatter(NullFormatter())
    axV.tick_params(axis="y", length=0, labelsize=11)
    for s in axV.spines.values():
        s.set_visible(False)

    # ======================
    # Δmean column
    # ======================
    axD.set_xlim(0, 1)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.yaxis.set_major_formatter(NullFormatter())
    axD.tick_params(axis="y", length=0)
    axD.set_title("Δmean", pad=title_pad)

    axD.text(
        0.5,
        -0.085,
        "(vs C1)",
        transform=axD.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="0.65",
    )

    for i, (c, d) in enumerate(zip(conds, dmeans)):
        if c == "C1":
            txt = "baseline"
            col = "0.45"
        else:
            txt = fmt_dmean(d)
            col = "0.25"
        axD.text(0.5, i, txt, ha="center", va="center", fontsize=10, color=col)

    for s in axD.spines.values():
        s.set_visible(False)

    # ======================
    # Significance
    # ======================
    axS.set_xlim(-0.6, 1.6)
    axS.set_yticks([])
    axS.yaxis.set_major_formatter(NullFormatter())
    axS.tick_params(axis="y", length=0)
    axS.set_xticks([])

    # Put "blank = ns" inside the title to avoid top collisions
    axS.set_title("Significance\n(blank = ns)", pad=title_pad - 2)

    axS.axvspan(-0.6, 0.5, color="0.0", alpha=0.015, zorder=0)
    axS.axvline(0.5, color="0.0", alpha=0.18, lw=1.0, zorder=1)

    axS.text(
        0.25,
        -0.09,
        "vs C1",
        transform=axS.transAxes,
        ha="center",
        va="top",
        fontsize=10,
    )
    axS.text(
        0.75,
        -0.09,
        "vs C2",
        transform=axS.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#8b1e1e",
    )

    for i, c in enumerate(conds):
        if c == "C1":
            axS.text(
                0.0, i, "ref", ha="center", va="center", fontsize=12, color=STAR_COLOR_A
            )
        else:
            st = star_from_p(famA.get(c, {}).get("p_adj", np.nan))
            axS.text(
                0.0, i, st, ha="center", va="center", fontsize=10, color=STAR_COLOR_A
            )

        if c == "C2":
            axS.text(
                1.0, i, "ref", ha="center", va="center", fontsize=12, color=STAR_COLOR_B
            )
        elif c != "C1":
            st = star_from_p(famB.get(c, {}).get("p_adj", np.nan))
            axS.text(
                1.0, i, st, ha="center", va="center", fontsize=10, color=STAR_COLOR_B
            )

    for s in axS.spines.values():
        s.set_visible(False)

    # Safe global title placement (NO overlap)
    fig.suptitle(suptitle, fontsize=16, y=0.980)
    return fig


# =========================
# CLI / MAIN
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="ALP Figure 2 only (condition tags + violins + Δmean + significance)."
    )
    p.add_argument("--metadata", required=True, help="Metadata text file.")
    p.add_argument("--data", required=True, help="Raw ALP data xlsx.")
    p.add_argument("--blank", type=float, default=0.066, help="Blank absorbance.")
    p.add_argument(
        "--conversion", type=float, required=True, help="Conversion net abs -> mU/mL."
    )
    p.add_argument("--alpha", type=float, default=0.05, help="BH alpha.")
    p.add_argument("--out_prefix", type=str, default="alp_raw", help="Output prefix.")
    p.add_argument(
        "--out_dir", type=Path, default=Path("alp_output"), help="Output directory."
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for jitter.")
    return p.parse_args()


def main():
    apply_pub_rcparams()

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

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
            alp = max(v_net * float(args.conversion), 0.0)
            rows.append({"condition": cond, "well_index": i + 1, "ALP_mU_mL": alp})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid ALP values found for conditions C1–C8.")

    conds = [c for c in cond_order if c in CONDITIONS and (df["condition"] == c).any()]
    if "C1" not in conds or "C2" not in conds:
        raise ValueError("C1 and C2 must be present.")

    cond_to_vals: Dict[str, np.ndarray] = {
        c: df.loc[df["condition"] == c, "ALP_mU_mL"].dropna().astype(float).values
        for c in conds
    }

    means_txt = []
    for c in conds:
        vals = cond_to_vals.get(c, np.array([]))
        if vals.size:
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            n_val = int(vals.size)
        else:
            mean_val = float("nan")
            sd_val = float("nan")
            n_val = 0
        means_txt.append((c, mean_val, sd_val, n_val))

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

    fig2 = plot_figure2_tags_violin_delta_sig(
        df=df,
        value_col="ALP_mU_mL",
        conds=conds,
        famA=famA,
        famB=famB,
        suptitle="Alkaline Phosphatase Activity - Day 10",
        xlab=r"ALP activity (mU·mL$^{-1}$)",
        seed=args.seed,
    )

    out_pdf = args.out_dir / f"{args.out_prefix}_ALP_fig2_tags_violin_delta_sig.pdf"
    out_png = args.out_dir / f"{args.out_prefix}_ALP_fig2_tags_violin_delta_sig.png"
    fig2.savefig(out_pdf)
    fig2.savefig(out_png)
    plt.close(fig2)

    out_txt = args.out_dir / f"{args.out_prefix}_ALP_means.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Condition\tMean\tSD\tN\n")
        for c, mean_val, sd_val, n_val in means_txt:
            f.write(f"{c}\t{mean_val:.6g}\t{sd_val:.6g}\t{n_val}\n")

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
