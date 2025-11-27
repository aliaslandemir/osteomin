#!/usr/bin/env python3
"""
ALP assay — raw A405 (left) + CIP calibration (right),
and a separate panel of CIP-normalised ALP activities.

The script reads an ALP Excel sheet (default `alp_raw_data.xlsx`) containing
rows B–D, maps wells to conditions, performs blank correction, fits a CIP
calibration curve, and converts absorbance → activity (U/well → mU/mL).

Outputs:
  • Raw A405 (including D1–D6 blanks) + CIP standard curve
  • CIP-normalised ALP activity distributions (violin + replicates + mean)
  • No statistical testing is performed.

Assay volumes per well:
  • 25 µL lysate   → contains the measurable ALP activity
  • 75 µL pNPP     → substrate generating the 405-nm chromophore
  • 25 µL NaOH     → stop solution stabilising the final colour

Total reaction volume = 125 µL (read by the plate reader).
Activity is reported per mL of original lysate, so U_per_well
is divided by 0.025 mL (lysate volume per well).
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -------------------- STYLE --------------------
rcParams.update(
    {
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
    }
)

VIOLIN_COLOR = "#E5E7EB"  # light neutral grey
DOT_COLOR = "#1D4ED8"  # deep blue (replicates)
MEAN_COLOR = "#B91C1C"  # dark red (mean + label)
LINE_COLOR = "#111827"  # almost black (CIP fit)

# -------------------- CONFIG --------------------
BLANK_MODE = "pnpp"  # "pnpp" → D1–D3 ; "lysate" → D4–D6
SHOW_UNIT = "mU/mL"  # "U/mL", "mU/mL", "U/µL", or "U/well"
MEAN_LABEL_FMT = "{:.3f}" if "mu" in SHOW_UNIT.lower() else "{:.4f}"

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
ORDER_C = list(COND_CODE.values())  # for activity only


def two_line_label(name: str) -> str:
    """Two-line labels for conditions."""
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


def raw_xtick_label(name: str) -> str:
    """Labels for raw A405 plot (blanks + conditions)."""
    if name == "pNPP+lysate":
        return "pNPP+lysate\nblank"
    if name == "Lysate":
        return "Lysate\nblank"
    return two_line_label(name)


def well(r, c) -> str:
    return f"{r}{c}"


def parse_args():
    parser = argparse.ArgumentParser(description="ALP assay plots and CIP calibration.")
    parser.add_argument(
        "-i",
        "--input",
        type=Path,
        default=Path("alp_raw_data.xlsx"),
        help="Path to alp_raw_data.xlsx (or equivalent) file.",
    )
    parser.add_argument(
        "-o",
        "--out",
        type=Path,
        default=Path("alp_results"),
        help="Output directory root for generated figures.",
    )
    return parser.parse_args()


# plate layout for conditions
id_map = {}
for cid, row, start in [
    ("C1", "B", 1),
    ("C2", "B", 4),
    ("C3", "B", 7),
    ("C4", "B", 10),
    ("C5", "C", 1),
    ("C6", "C", 4),
    ("C7", "C", 7),
    ("C8", "C", 10),
]:
    for k in range(3):
        id_map[f"{row}{start + k}"] = COND_CODE[cid]

# blanks
ctrl_map = {well("D", i): "pNPP+lysate" for i in [1, 2, 3]}
ctrl_map.update({well("D", i): "Lysate" for i in [4, 5, 6]})


def assign_group(w):
    if w in id_map:
        return id_map[w]
    if w in ctrl_map:
        return ctrl_map[w]
    return None


def get_row(raw_df, letter, ncols=12):
    row_vals = (
        raw_df.loc[raw_df[0] == letter].iloc[0, 1 : 1 + ncols].to_numpy(dtype=float)
    )
    if row_vals.size < ncols:
        row_vals = np.concatenate([row_vals, np.full(ncols - row_vals.size, np.nan)])
    return row_vals


def row_to_long(vals, row_label):
    return pd.DataFrame(
        {
            "Well": [well(row_label, i) for i in range(1, 13)],
            "Row": row_label,
            "Col": range(1, 13),
            "Value": vals,
        }
    )


def a405_to_activity(a405, blank, slope, unit=SHOW_UNIT):
    A_net_s = a405 - blank
    U_per_well = np.maximum(A_net_s / slope, 0.0)
    if unit == "U/well":
        return U_per_well
    if unit in ("U/µL", "U/uL"):
        return U_per_well / 25.0
    if unit == "mU/mL":
        return (U_per_well / 0.025) * 1000.0
    return U_per_well / 0.025  # default U/mL


def main():
    global df_all, df, plot_data_act, plot_data_raw, ORDER_RAW, A_blank, blank_label, m

    args = parse_args()
    data_file = Path(args.input)
    if data_file.is_dir():
        data_file = data_file / "alp_raw_data.xlsx"
    raw = pd.read_excel(data_file, header=None)
    raw[0] = raw[0].astype(str).str.strip().str.upper()  # B/C/D labels in col 0

    row_B = get_row(raw, "B")  # B1–B12
    row_C = get_row(raw, "C")  # C1–C12
    row_D = get_row(raw, "D")  # D1–D6 (blanks)

    assert len(row_B) == len(row_C) == len(row_D) == 12

    df_all = pd.concat(
        [row_to_long(row_B, "B"), row_to_long(row_C, "C"), row_to_long(row_D, "D")],
        ignore_index=True,
    )
    df_all["Group"] = df_all["Well"].map(assign_group)

    # -------------------- Blank & standards --------------------
    A_pnpp = np.nanmean(row_D[0:3])  # D1–D3
    A_lys = np.nanmean(row_D[3:6])  # D4–D6
    A_blank = A_pnpp if BLANK_MODE.lower() == "pnpp" else A_lys
    blank_label = "pNPP+lysate" if BLANK_MODE.lower() == "pnpp" else "Lysate"

    standards = np.array(
        [
            [0.00005, 0.20],
            [0.00010, 0.40],
            [0.00020, 0.80],
        ],
        dtype=float,
    )
    U = standards[:, 0]
    A = standards[:, 1]
    A_net = A - A_blank

    # origin-constrained fit: A_net = m * U
    m = (U @ A_net) / (U @ U)
    yhat = m * U
    ss_res = np.sum((A_net - yhat) ** 2)
    ss_tot = np.sum((A_net - np.mean(A_net)) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Quantify only C1–C8 for activity
    df = df_all[df_all["Group"].isin(ORDER_C)].copy()
    df["Activity"] = df["Value"].map(
        lambda a: a405_to_activity(a, blank=A_blank, slope=m, unit=SHOW_UNIT)
    )

    stats = (
        df.groupby("Group")["Activity"]
        .agg(
            n=lambda x: np.isfinite(x).sum(),
            mean=lambda x: np.nanmean(x),
            sd=lambda x: np.nanstd(x, ddof=1),
            median=lambda x: np.nanmedian(x),
        )
        .reset_index()
    )
    stats["sem"] = stats["sd"] / np.sqrt(stats["n"])
    stats["ci95"] = 1.96 * stats["sem"]
    stats["Group"] = pd.Categorical(stats["Group"], ORDER_C)
    stats = stats.sort_values("Group")

    output_root = Path(args.out)
    figdir = output_root / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Using input file: {data_file}")
    print(f"[INFO] Output directory: {figdir}")

    pdf_cip_raw = figdir / "ALP_CIP_plus_raw_pub.pdf"
    pdf_activity = figdir / "ALP_activity_pub.pdf"

    ORDER_RAW = ["pNPP+lysate", "Lysate"] + ORDER_C

    plot_data_act = [
        df.loc[df["Group"] == g, "Activity"].dropna().values for g in ORDER_C
    ]
    valid_act = (
        np.concatenate([v for v in plot_data_act if v.size])
        if any(v.size for v in plot_data_act)
        else np.array([0.0])
    )
    ymin_a, ymax_a = float(valid_act.min()), float(valid_act.max())
    span_a = ymax_a - ymin_a
    pad_a = 0.15 * span_a if span_a > 0 else 0.15 * max(ymax_a, 1.0)
    y_limits_act = (max(ymin_a - pad_a, 0.0), ymax_a + pad_a)

    plot_data_raw = [
        df_all.loc[df_all["Group"] == g, "Value"].dropna().values for g in ORDER_RAW
    ]
    valid_raw = (
        np.concatenate([v for v in plot_data_raw if v.size])
        if any(v.size for v in plot_data_raw)
        else np.array([0.0])
    )
    ymin_r, ymax_r = float(valid_raw.min()), float(valid_raw.max())
    span_r = ymax_r - ymin_r
    pad_r = 0.15 * span_r if span_r > 0 else 0.15 * max(ymax_r, 0.2)
    y_limits_raw = (max(ymin_r - pad_r, 0.0), ymax_r + pad_r)

    fig_cip_raw, (ax_raw, ax1) = plt.subplots(
        1, 2, figsize=(12.0, 4.8), width_ratios=[1.8, 1.0]
    )
    fig_cip_raw.subplots_adjust(wspace=0.35)

    render_raw_axis(ax_raw, "Raw $A_{405}$ per condition", y_limits_raw)
    add_violin_legend(ax_raw, loc="upper left")

    ax1.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
    ax1.scatter(
        U,
        A_net,
        s=40,
        facecolor=DOT_COLOR,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    U_fit = np.linspace(0, U.max() * 1.05, 200)
    ax1.plot(U_fit, m * U_fit, color=LINE_COLOR, lw=2.0, zorder=2)

    y_top = max(A_net.max(), (m * U_fit).max())

    ax1.text(
        0.03 * U.max(),
        0.95 * y_top,
        rf"$\mathrm{{Net}}\,A_{{405}} = {m:.2e}\cdot\mathrm{{U}}$"
        + "\n"
        + rf"$R^2 = {r2:.4f}$",
        ha="left",
        va="top",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#666666", alpha=0.9),
    )

    ax1.text(
        0.03 * U.max(),
        0.15 * y_top,
        rf"Blank: {blank_label}" + "\n" + rf"$A_{{405}} = {A_blank:.3f}$",
        ha="left",
        va="bottom",
        fontsize=8.5,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9),
    )

    ax1.set_xlabel("CIP activity (U/well)")
    ax1.set_ylabel(r"Net absorbance ($A_{405}$ − blank)")
    ax1.set_title("CIP standard curve")

    fig_cip_raw.savefig(pdf_cip_raw, dpi=300, bbox_inches="tight")

    fig_activity, ax_activity = plt.subplots(figsize=(7.0, 4.5))
    render_activity_axis(ax_activity, "ALP activity (CIP-normalised)", y_limits_act)
    add_violin_legend(ax_activity, loc="upper left")
    add_blank_note(ax_activity, bbox=(1.0, 0.96))

    fig_activity.savefig(pdf_activity, dpi=300, bbox_inches="tight")

    print(f"[OK] Saved CIP+raw figure → {pdf_cip_raw}")
    print(f"[OK] Saved activity figure → {pdf_activity}")
    print(f"[INFO] Slope m = {m:.2e} (net A per U/well); unit displayed: {SHOW_UNIT}")




# ---------- helpers ----------
def add_violin_legend(ax, loc="upper left"):
    elements = [
        Patch(
            facecolor=VIOLIN_COLOR,
            edgecolor="#9CA3AF",
            alpha=0.9,
            label="Distribution",
        ),
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=DOT_COLOR,
            markeredgecolor="black",
            label="Replicates",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            linestyle="none",
            markerfacecolor=MEAN_COLOR,
            markeredgecolor="black",
            label="Mean",
        ),
    ]
    leg = ax.legend(handles=elements, loc=loc, frameon=True, fontsize=8)
    leg.get_frame().set_alpha(0.95)
    leg.get_frame().set_edgecolor("#9CA3AF")


def render_activity_axis(ax, title_suffix, ylim):
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)

    parts = ax.violinplot(
        plot_data_act, showmeans=False, showmedians=False, showextrema=False
    )
    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor("#9CA3AF")
        body.set_alpha(0.9)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(42)
    for i, y in enumerate(plot_data_act, start=1):
        if y.size == 0:
            continue
        x = np.full_like(y, i, dtype=float) + rng.uniform(-0.07, 0.07, size=y.size)
        ax.plot(
            x,
            y,
            "o",
            ms=4.0,
            mfc=DOT_COLOR,
            mec="black",
            mew=0.3,
            alpha=0.9,
            zorder=3,
        )

    for i, g in enumerate(ORDER_C, start=1):
        y = df.loc[df["Group"] == g, "Activity"].dropna()
        if y.empty:
            continue
        mu = float(np.mean(y))

        ax.plot(
            [i],
            [mu],
            "D",
            ms=6.0,
            mfc=MEAN_COLOR,
            mec="black",
            mew=0.5,
            zorder=5,
        )
        ax.text(
            i + 0.10,
            mu,
            MEAN_LABEL_FMT.format(mu),
            ha="left",
            va="center",
            fontsize=8.5,
            color=MEAN_COLOR,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xticks(range(1, len(ORDER_C) + 1))
    ax.set_xticklabels([two_line_label(g) for g in ORDER_C])
    ax.set_ylabel(f"ALP activity ({SHOW_UNIT})")
    ax.set_title(title_suffix)
    ax.set_ylim(*ylim)
    return ax


def render_raw_axis(ax, title_suffix, ylim):
    ax.yaxis.grid(True, linestyle="--", alpha=0.35, linewidth=0.6)

    parts = ax.violinplot(
        plot_data_raw, showmeans=False, showmedians=False, showextrema=False
    )
    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor("#9CA3AF")
        body.set_alpha(0.9)
        body.set_linewidth(0.8)

    rng = np.random.default_rng(43)
    for i, y in enumerate(plot_data_raw, start=1):
        if y.size == 0:
            continue
        x = np.full_like(y, i, dtype=float) + rng.uniform(-0.07, 0.07, size=y.size)
        ax.plot(
            x,
            y,
            "o",
            ms=4.0,
            mfc=DOT_COLOR,
            mec="black",
            mew=0.3,
            alpha=0.9,
            zorder=3,
        )

    # mean markers + labels
    for i, g in enumerate(ORDER_RAW, start=1):
        y = df_all.loc[df_all["Group"] == g, "Value"].dropna()
        if y.empty:
            continue
        mu = float(np.mean(y))
        ax.plot(
            [i],
            [mu],
            "D",
            ms=6.0,
            mfc=MEAN_COLOR,
            mec="black",
            mew=0.5,
            zorder=5,
        )
        ax.text(
            i + 0.10,
            mu,
            f"{mu:.3f}",
            ha="left",
            va="center",
            fontsize=8.5,
            color=MEAN_COLOR,
            fontweight="bold",
            zorder=6,
        )

    ax.set_xticks(range(1, len(ORDER_RAW) + 1))
    ax.set_xticklabels([raw_xtick_label(g) for g in ORDER_RAW])
    ax.set_ylabel(r"$A_{405}$ (raw, not blank-corrected)")
    ax.set_title(title_suffix)
    ax.set_ylim(*ylim)
    return ax


def add_blank_note(ax, bbox=(1.0, 0.92)):
    txt = rf"Blank for activity: {blank_label}  ($A_{{405}}={A_blank:.3f}$)"
    leg = ax.legend(
        handles=[Patch(facecolor="none", edgecolor="none", label=txt)],
        loc="upper right",
        bbox_to_anchor=bbox,
        frameon=True,
        fontsize=8,
        handlelength=0,
        handletextpad=0.0,
        borderaxespad=0.3,
    )
    leg.get_frame().set_facecolor("white")
    leg.get_frame().set_alpha(0.9)
    leg.get_frame().set_edgecolor("#888888")


# ---------- Figure 1: raw A405 (left) + CIP curve (right) ----------
fig_cip_raw, (ax_raw, ax1) = plt.subplots(
    1, 2, figsize=(12.0, 4.8), width_ratios=[1.8, 1.0]
)
fig_cip_raw.subplots_adjust(wspace=0.35)

# Left: raw A405 including blanks
render_raw_axis(ax_raw, "Raw $A_{405}$ per condition", y_limits_raw)
add_violin_legend(ax_raw, loc="upper left")

# Right: CIP standard curve (net A vs U)
ax1.yaxis.grid(True, linestyle="--", alpha=0.4, linewidth=0.6)
ax1.scatter(
    U,
    A_net,
    s=40,
    facecolor=DOT_COLOR,
    edgecolor="black",
    linewidth=0.8,
    zorder=3,
)

U_fit = np.linspace(0, U.max() * 1.05, 200)
ax1.plot(U_fit, m * U_fit, color=LINE_COLOR, lw=2.0, zorder=2)

y_top = max(A_net.max(), (m * U_fit).max())

ax1.text(
    0.03 * U.max(),
    0.95 * y_top,
    rf"$\mathrm{{Net}}\,A_{{405}} = {m:.2e}\cdot\mathrm{{U}}$"
    + "\n"
    + rf"$R^2 = {r2:.4f}$",
    ha="left",
    va="top",
    fontsize=9,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#666666", alpha=0.9),
)

ax1.text(
    0.03 * U.max(),
    0.15 * y_top,
    rf"Blank: {blank_label}" + "\n" + rf"$A_{{405}} = {A_blank:.3f}$",
    ha="left",
    va="bottom",
    fontsize=8.5,
    bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#999999", alpha=0.9),
)

ax1.set_xlabel("CIP activity (U/well)")
ax1.set_ylabel(r"Net absorbance ($A_{405}$ − blank)")
ax1.set_title("CIP standard curve")

fig_cip_raw.savefig(png_cip_raw, dpi=300, bbox_inches="tight")

# ---------- Figure 2: CIP-normalised ALP activity ----------
fig_activity, ax_activity = plt.subplots(figsize=(7.0, 4.5))
render_activity_axis(ax_activity, "ALP activity (CIP-normalised)", y_limits_act)
add_violin_legend(ax_activity, loc="upper left")
add_blank_note(ax_activity, bbox=(1.0, 0.96))

fig_activity.savefig(png_activity, dpi=300, bbox_inches="tight")

print(f"[OK] Saved CIP+raw figure → {png_cip_raw}")
    print(f"[OK] Saved activity figure → {png_activity}")
    print(f"[INFO] Slope m = {m:.2e} (net A per U/well); unit displayed: {SHOW_UNIT}")


if __name__ == "__main__":
    main()
