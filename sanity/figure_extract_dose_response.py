#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
TEXT_SCALE = 1.5

COLORS = {
    "ink": "#111827",
    "grid": "#d1d5db",
    "bg": "#f8fafc",
    "panel": "#ffffff",
    "dex": "#6b7280",
    "mk4": "#9ca3af",
    "alp": "#b07d39",
    "collagen": "#4c956c",
    "ha": "#4c78a8",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12 * TEXT_SCALE,
        "axes.titleweight": "semibold",
        "axes.labelsize": 9.5 * TEXT_SCALE,
        "xtick.labelsize": 8.5 * TEXT_SCALE,
        "ytick.labelsize": 8.5 * TEXT_SCALE,
        "legend.fontsize": 8.5 * TEXT_SCALE,
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Two creative dose-response figures: A(single-dose) and B(combined vs additive)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--sanity3",
        type=Path,
        default=SCRIPT_DIR / "out" / "sanity_check3" / "figure_extract_3",
        help="sanity3 extract folder or condition_summary.csv",
    )
    p.add_argument(
        "--sanity4",
        type=Path,
        default=SCRIPT_DIR / "out" / "sanity_check4" / "figure_extract_4",
        help="sanity4 extract folder or condition_summary.csv",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=SCRIPT_DIR / "out" / "figure_extract_dose_response",
        help="output directory",
    )
    return p.parse_args()


def resolve_summary_csv(path_like: Path, check_num: int) -> Path:
    if path_like.is_file():
        return path_like
    candidates = [
        path_like / "condition_summary.csv",
        path_like / f"figure_extract_{check_num}" / "condition_summary.csv",
    ]
    for c in candidates:
        if c.exists():
            return c
    raise SystemExit(f"condition_summary.csv not found under {path_like}")


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, facecolor: str | None = None) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fc = facecolor if facecolor is not None else fig.get_facecolor()
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=fc)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight", pad_inches=0.02, facecolor=fc)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(direction="out", length=3.8, width=1.0)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.9, alpha=0.6)


def split_branches(df4: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dex = df4[df4["branch"] == "dex"].copy().sort_values("dose").reset_index(drop=True)
    mk4 = df4[df4["branch"] == "mk4"].copy().sort_values("dose").reset_index(drop=True)
    if dex.empty or mk4.empty:
        raise SystemExit("sanity4 needs both dex and mk4 branches")
    if not bool((mk4["dose"] == 0.0).any()):
        v = dex[dex["dose"] == 0.0]
        if not v.empty:
            add = v.iloc[[0]].copy()
            add.loc[:, "branch"] = "mk4"
            add.loc[:, "dose"] = 0.0
            add.loc[:, "mk4_conc_uM"] = 0.0
            mk4 = pd.concat([add, mk4], ignore_index=True).sort_values("dose").reset_index(drop=True)
    return dex, mk4


def interp_log2fc(branch_df: pd.DataFrame, col: str, dose: float) -> float:
    if dose <= 0.0:
        z = branch_df[branch_df["dose"] == 0.0]
        return float(z[col].iloc[0]) if not z.empty else 0.0
    d = branch_df.sort_values("dose")
    x = np.clip(d["dose"].to_numpy(dtype=float), 1e-9, None)
    y = d[col].to_numpy(dtype=float)
    return float(np.interp(np.log10(max(dose, 1e-9)), np.log10(x), y))


def make_synergy(df3: pd.DataFrame, dex: pd.DataFrame, mk4: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df3.iterrows():
        d = float(r["dex_conc_nM"])
        m = float(r["mk4_conc_uM"])
        obs = float(r["assay_ha_log2fc_vs_control"])
        exp = interp_log2fc(dex, "assay_ha_log2fc_vs_vehicle", d) + interp_log2fc(
            mk4, "assay_ha_log2fc_vs_vehicle", m
        )
        rows.append(
            {
                "dex_conc_nM": d,
                "mk4_conc_uM": m,
                "obs_ha_log2fc": obs,
                "expected_additive_log2fc": exp,
                "synergy_log2fc": obs - exp,
                "assay_collagen_fc_vs_control": float(r["assay_collagen_fc_vs_control"]),
            }
        )
    return pd.DataFrame(rows)


def add_letter(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.13,
        1.06,
        letter,
        transform=ax.transAxes,
        fontsize=14 * TEXT_SCALE,
        fontweight="bold",
        va="bottom",
        ha="left",
        color=COLORS["ink"],
    )


def export_figure_a(df4: pd.DataFrame, out_dir: Path) -> None:
    dex, mk4 = split_branches(df4)
    metrics = [
        ("assay_alp_log2fc_vs_vehicle", "assay_alp_log2fc_sd_vs_vehicle", "ALP", COLORS["alp"]),
        ("assay_collagen_log2fc_vs_vehicle", "assay_collagen_log2fc_sd_vs_vehicle", "Collagen", COLORS["collagen"]),
        ("assay_ha_log2fc_vs_vehicle", "assay_ha_log2fc_sd_vs_vehicle", "Mineral", COLORS["ha"]),
    ]

    fig, ax = plt.subplots(figsize=(9.8, 6.0))
    style_axes(ax, "y")

    # One panel: normalized dose rank so both agents share an x-axis.
    # 0 = vehicle, 1 = highest tested dose in each branch.
    for bdf, branch_name, marker, ls in [
        (dex, "Dex", "o", "-"),
        (mk4, "MK-4", "s", "--"),
    ]:
        dose = bdf["dose"].to_numpy(dtype=float)
        x_rank = np.zeros_like(dose)
        if np.nanmax(dose) > 0:
            x_rank = np.log10(np.clip(dose, 1e-9, None)) / np.log10(np.nanmax(dose))
            x_rank[dose <= 0] = 0.0
        for ycol, sdcol, label, color in metrics:
            y = bdf[ycol].to_numpy(dtype=float)
            sd = bdf[sdcol].to_numpy(dtype=float)
            ax.plot(
                x_rank,
                y,
                color=color,
                linestyle=ls,
                marker=marker,
                linewidth=2.0,
                markersize=5.4,
                alpha=0.95 if branch_name == "Dex" else 0.82,
                label=f"{label} ({branch_name})",
            )
            ax.fill_between(x_rank, y - sd, y + sd, color=color, alpha=0.08)

    ax.axhline(0.0, color=COLORS["ink"], linewidth=1.0, alpha=0.75)
    ax.set_xlim(-0.02, 1.02)
    ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("Normalized log-dose rank (0 = vehicle, 1 = highest tested)")
    ax.set_ylabel("log2 fold-change vs vehicle")
    ax.legend(frameon=False, ncol=2, loc="upper left")
    add_letter(ax, "A")
    save_figure(fig, out_dir, "figure_A_single_dose_creative")


def export_figure_b(df3: pd.DataFrame, df4: pd.DataFrame, out_dir: Path) -> None:
    dex, mk4 = split_branches(df4)
    syn = make_synergy(df3, dex, mk4)
    score = (
        0.60 * df3["assay_ha_log2fc_vs_control"]
        + 0.25 * df3["assay_collagen_log2fc_vs_control"]
        + 0.15 * df3["assay_alp_log2fc_vs_control"]
    )
    top = df3.assign(poster_score=score).nlargest(6, "poster_score")

    fig, ax0 = plt.subplots(figsize=(7.8, 5.8))
    fig.patch.set_facecolor("white")

    # Only positive-dose combinations for true log-scale contour maps.
    combo = df3[(df3["dex_conc_nM"] > 0.0) & (df3["mk4_conc_uM"] > 0.0)].copy()
    # Left: combined-dose HAp landscape
    style_axes(ax0, "")
    pivot = (
        combo.pivot(index="mk4_conc_uM", columns="dex_conc_nM", values="assay_ha_log2fc_vs_control")
        .sort_index()
        .sort_index(axis=1)
    )
    xv = pivot.columns.to_numpy(dtype=float)
    yv = pivot.index.to_numpy(dtype=float)
    z = pivot.to_numpy(dtype=float)
    # Sequential, colorblind-safe style for strictly positive values.
    cmap = LinearSegmentedColormap.from_list(
        "plos_cb_seq",
        ["#fff7bc", "#c7e9b4", "#7fcdbb", "#41b6c4", "#1d91c0", "#225ea8", "#0c2c84"],
        N=256,
    )
    im = ax0.contourf(xv, yv, z, levels=22, cmap=cmap, alpha=0.98, antialiased=True)
    cs = ax0.contour(xv, yv, z, levels=9, colors="#334155", linewidths=0.9, alpha=0.8)
    # Label alternate levels to avoid crowding/overlap in tight regions.
    label_levels = cs.levels[::2]
    ax0.clabel(
        cs,
        levels=label_levels,
        inline=True,
        inline_spacing=6,
        fontsize=6.6 * TEXT_SCALE,
        fmt="%.1f",
        colors="black",
    )
    ax0.set_xscale("log")
    ax0.set_yscale("log")
    ax0.set_xlim(xv.min() * 0.85, xv.max() * 1.15)
    ax0.set_ylim(yv.min() * 0.85, yv.max() * 1.15)
    xticks = list(xv)
    # Simple anti-overlap for crowded high-end log ticks (e.g., 80 and 100).
    if len(xticks) >= 2 and (xticks[-1] / xticks[-2]) < 1.35:
        xticks = xticks[:-2] + [xticks[-1]]
    ax0.set_xticks(xticks)
    ax0.set_yticks(yv)
    xfmt = ScalarFormatter()
    xfmt.set_scientific(False)
    xfmt.set_useOffset(False)
    yfmt = ScalarFormatter()
    yfmt.set_scientific(False)
    yfmt.set_useOffset(False)
    ax0.xaxis.set_major_formatter(xfmt)
    ax0.yaxis.set_major_formatter(yfmt)
    ax0.set_xlabel("Dex (nM)")
    ax0.set_ylabel("MK-4 (uM)")
    ax0.set_title("Combined Dose Response", fontsize=(12 * TEXT_SCALE) / 1.3)
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.03)
    cbar.set_label("Mineral log2FC", fontsize=8.5 * TEXT_SCALE)
    cbar.ax.tick_params(labelsize=8 * TEXT_SCALE)
    # Keep the contour-only panel for a cleaner publication-style view.

    fig.subplots_adjust(left=0.12, right=0.90, bottom=0.12, top=0.97)
    save_figure(fig, out_dir, "figure_B_combined_vs_single_creative", facecolor="white")

    syn.sort_values("synergy_log2fc", ascending=False).to_csv(out_dir / "figure_B_synergy_table.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    s3_path = resolve_summary_csv(args.sanity3, 3)
    s4_path = resolve_summary_csv(args.sanity4, 4)
    df3 = pd.read_csv(s3_path)
    df4 = pd.read_csv(s4_path)

    req3 = {
        "dex_conc_nM",
        "mk4_conc_uM",
        "assay_alp_log2fc_vs_control",
        "assay_collagen_log2fc_vs_control",
        "assay_ha_log2fc_vs_control",
        "assay_collagen_fc_vs_control",
    }
    req4 = {
        "branch",
        "dose",
        "assay_alp_log2fc_vs_vehicle",
        "assay_collagen_log2fc_vs_vehicle",
        "assay_ha_log2fc_vs_vehicle",
        "assay_alp_log2fc_sd_vs_vehicle",
        "assay_collagen_log2fc_sd_vs_vehicle",
        "assay_ha_log2fc_sd_vs_vehicle",
    }
    miss3 = sorted(req3 - set(df3.columns))
    miss4 = sorted(req4 - set(df4.columns))
    if miss3:
        raise SystemExit(f"sanity3 summary missing: {miss3}")
    if miss4:
        raise SystemExit(f"sanity4 summary missing: {miss4}")

    export_figure_a(df4, out_dir)
    export_figure_b(df3, df4, out_dir)

    print(f"[figure_extract_dose_response] used sanity3: {s3_path}")
    print(f"[figure_extract_dose_response] used sanity4: {s4_path}")
    print(f"[figure_extract_dose_response] wrote two figures to: {out_dir}")


if __name__ == "__main__":
    main()
