#!/usr/bin/env python3
"""
Shared plotting / stats helpers for the publication-ready violin figures used across the
endpointaad finalcodes modules. Centralizes the DAPI-style palette, layout, and family-based
stats helpers so the other scripts can stay clean and consistent.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import matplotlib
# Backend must be set before pyplot imports
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
import numpy as np
import pandas as pd
from scipy import stats

# Backend + global style (matches dapi_stat_code.py)
matplotlib.use("Agg")
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "figure.autolayout": True,
})

# Shared condition metadata
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

FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]
FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]

# Palette + markers
COND_COLORS = {
    "C1": "#6B6B5E",
    "C2": "#1F5AA6",
    "C3": "#1B7F5A",
    "C4": "#C97A00",
    "C5": "#A65C8A",
    "C6": "#B24C1A",
    "C7": "#6AAED6",
    "C8": "#7A6A3A",
}
VIOLIN_FACE = "#E6E6E6"
VIOLIN_EDGE = "#B3B3B3"
STAR_COLOR_A = "#000000"
STAR_COLOR_B = "#B22222"
SEPARATOR_LINE = "#F2F2F2"
MEAN_COLOR = "#B22222"
MEDIAN_COLOR = "#2B2B2B"


def two_line_label(name: str) -> str:
    mapping = {
        "Basal": "Basal",
        "Osteo": r"AA+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)": r"AA+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return mapping.get(name, name)


def benjamini_hochberg(pvals: Sequence[float]) -> List[float]:
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


def kw_all_conditions(cond_to_vals: Dict[str, np.ndarray], conds_order: Sequence[str]) -> Tuple[float, float]:
    groups = []
    for c in conds_order:
        vals = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            groups.append(vals)
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = stats.kruskal(*groups)
    return float(H), float(p)


def pairwise_vs_ref_BH(
    cond_to_vals: Dict[str, np.ndarray],
    ref: str,
    targets: Sequence[str],
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


def make_positions_with_gaps(conds: Sequence[str]) -> Tuple[List[float], List[float]]:
    xs: List[float] = []
    base = 0.0
    for c in conds:
        base += 1.0
        if c == "C3":
            base += 0.30
        if c == "C7":
            base += 0.35
        xs.append(base)

    idx = {c: i for i, c in enumerate(conds)}
    seps: List[float] = []
    if "C2" in idx and "C3" in idx:
        seps.append((xs[idx["C2"]] + xs[idx["C3"]]) / 2.0)
    if "C6" in idx and "C7" in idx:
        seps.append((xs[idx["C6"]] + xs[idx["C7"]]) / 2.0)
    return xs, seps


def mean_diff_labels_vs_c1(
    df: pd.DataFrame,
    value_col: str,
    conds: Sequence[str],
    condition_col: str = "condition",
) -> List[str]:
    ref_vals = df.loc[df[condition_col] == "C1", value_col].dropna().astype(float)
    ref_mean = float(ref_vals.mean()) if not ref_vals.empty else 0.0

    out: List[str] = []
    for c in conds:
        if c == "C1":
            out.append("baseline")
            continue
        vals = df.loc[df[condition_col] == c, value_col].dropna().astype(float)
        d = float(vals.mean()) - ref_mean if not vals.empty else np.nan
        if not np.isfinite(d):
            out.append("")
            continue
        if abs(d) < 0.0005:
            d = 0.0
        arrow = "↑" if d > 0 else "↓"
        out.append(f"{d:+.2g} {arrow}")
    return out


def add_stats_header(ax, fs: float = 8.6, grey: str = "0.35", red: str = "#B22222"):
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
    conds: Sequence[str],
    title: str,
    ylab: str,
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    seed: int = 0,
    condition_col: str = "condition",
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    xs, seps = make_positions_with_gaps(conds)
    data = [df.loc[df[condition_col] == c, value_col].dropna().astype(float).values for c in conds]

    valid = [v for v in data if v.size > 0]
    allv = np.concatenate(valid) if valid else np.array([0.0])
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    yr = (y_max - y_min) if y_max > y_min else 1.0

    yA = y_max + 0.045 * yr
    yB = y_max + 0.080 * yr

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.12)

    vp = ax.violinplot(
        data, positions=xs, widths=0.72,
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
            zorder=5,
        )
        ax.scatter(
            x, median_val,
            s=60, marker="D",
            color=MEDIAN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            zorder=5,
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

    delta_labels = mean_diff_labels_vs_c1(df, value_col, conds, condition_col=condition_col)
    y_delta = y_min - 0.11 * yr
    for x, lab in zip(xs, delta_labels):
        ax.text(x, y_delta, lab, ha="center", va="top", fontsize=8.2, color="0.35")
    ax.text(xs[0] - 0.75, y_delta, r"$\Delta$mean", ha="left", va="top",
            fontsize=8.2, color="0.35", clip_on=False)

    ax.set_xticks(xs)
    labels = [two_line_label(COND_CODE.get(c, c)) for c in conds]
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(y_min - 0.17 * yr, y_max + 0.26 * yr)

    add_stats_header(ax)
    add_mean_median_legend(ax)
    fig.tight_layout()
    return fig
