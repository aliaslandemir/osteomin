#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sanity5_figure_extract import (
    ASSAY_SPECS,
    CONDITION_ORDER,
    CONDITION_TICK_LABELS,
    build_all_assays,
)

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_DEFAULT = SCRIPT_DIR / "out" / "sanity_check5"

COLORS = {
    "ink": "#111827",
    "grid": "#d7dee8",
    "panel": "#ffffff",
    "bg": "#f8fafc",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.labelsize": 12,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Poster-style single figure for sanity_check5 sim-vs-exp comparison.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input",
        type=Path,
        default=INPUT_DEFAULT,
        help="sanity_check5 root (same convention as sanity5_figure_extract.py)",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder (default: <input>/figure_extract_5/comparison_extract_3styles)",
    )
    return p.parse_args()


def save_figure(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    *,
    png_dpi: int = 300,
    pdf_dpi: int = 300,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=png_dpi, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=pdf_dpi, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def condition_label(code: str) -> str:
    return CONDITION_TICK_LABELS.get(code, code).replace("\n", " ")


def build_tidy_from_assays(assay_frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for assay in ("alp", "collagen", "ars"):
        t = ASSAY_SPECS[assay]["title"]
        f = assay_frames[assay].copy()
        f["assay"] = assay
        f["assay_title"] = t
        rows.append(
            f[
                [
                    "assay",
                    "assay_title",
                    "condition_code",
                    "sim_rel_effect",
                    "exp_rel_effect",
                    "is_reference",
                ]
            ]
        )
    out = pd.concat(rows, ignore_index=True)
    out["condition_code"] = pd.Categorical(out["condition_code"], categories=CONDITION_ORDER, ordered=True)
    return out.sort_values(["condition_code", "assay_title"]).reset_index(drop=True)


def export_style1_lollipop(tidy: pd.DataFrame, out_dir: Path) -> None:
    text_scale = 1.5
    exp_color = "#1f4e8c"
    sim_color = "#c47e2f"
    marker_size = 58 * 1.4 * 1.3
    fig, axes = plt.subplots(1, 3, figsize=(15.5 * 1.3, 7.6), sharey=True)
    assays = ["ALP", "Collagen", "ARS"]
    y_pos = np.arange(len(CONDITION_ORDER))

    for ax, assay in zip(axes, assays):
        sub = tidy[tidy["assay_title"] == assay].copy()
        sub["condition_code"] = pd.Categorical(sub["condition_code"], categories=CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition_code").reset_index(drop=True)
        exp_v = sub["exp_rel_effect"].to_numpy(dtype=float)
        sim_v = sub["sim_rel_effect"].to_numpy(dtype=float)
        for y, xv, yv in zip(y_pos, exp_v, sim_v):
            ax.plot([xv, yv], [y, y], color="black", linewidth=2.2, alpha=0.9, zorder=1)
        ax.scatter(exp_v, y_pos, s=marker_size, color=exp_color, marker="o", label="Experiment", zorder=3)
        ax.scatter(sim_v, y_pos, s=marker_size, color=sim_color, marker="D", label="Simulation", zorder=3)
        ax.axvline(0.0, color="#6b7280", linewidth=1.1, linestyle="--", alpha=0.7)
        ax.set_title(assay, fontsize=14 * text_scale, fontweight="semibold")
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.9, alpha=0.65)
        ax.set_facecolor(COLORS["panel"])
        ax.set_xlabel("Relative effect vs C2", fontsize=12 * text_scale)
        ax.tick_params(direction="out", length=4, labelsize=11 * text_scale)
        if assay == "ALP":
            ax.set_yticks(y_pos)
            ax.set_yticklabels([condition_label(c) for c in CONDITION_ORDER])
            ax.set_ylabel("Condition", fontsize=12 * text_scale, labelpad=44)
            # Add condition codes as a vertical label column next to condition names.
            for y, code in zip(y_pos, CONDITION_ORDER):
                ax.text(
                    -0.095,
                    y,
                    code,
                    transform=ax.get_yaxis_transform(),
                    rotation=-90,
                    va="center",
                    ha="center",
                    fontsize=9.6 * text_scale,
                    color=COLORS["ink"],
                    clip_on=False,
                )
        else:
            ax.set_yticks(y_pos)
            ax.set_yticklabels([])
        lim = max(
            0.12,
            float(
                np.nanpercentile(
                    np.abs(np.concatenate([exp_v, sim_v])),
                    99,
                )
            ),
        )
        ax.set_xlim(-1.08 * lim, 1.08 * lim)
        ax.invert_yaxis()

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles[:2],
        labels[:2],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.02),
        fontsize=10.5 * text_scale,
    )
    save_figure(fig, out_dir, "style1_lollipop_condition_comparison")


def export_style2_dual_heatmap(tidy: pd.DataFrame, out_dir: Path) -> None:
    assays = ["ALP", "Collagen", "ARS"]
    conds = [c for c in CONDITION_ORDER]
    exp_mat = np.zeros((len(conds), len(assays)), dtype=float)
    sim_mat = np.zeros((len(conds), len(assays)), dtype=float)
    for i, c in enumerate(conds):
        for j, a in enumerate(assays):
            r = tidy[(tidy["condition_code"] == c) & (tidy["assay_title"] == a)].iloc[0]
            exp_mat[i, j] = float(r["exp_rel_effect"])
            sim_mat[i, j] = float(r["sim_rel_effect"])
    vals = np.concatenate([exp_mat.ravel(), sim_mat.ravel()])
    vmax = max(0.16, float(np.nanpercentile(np.abs(vals[np.isfinite(vals)]), 97)))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 7.4), sharey=True)
    mats = [(exp_mat, "Experiment"), (sim_mat, "Simulation")]
    for ax, (mat, title) in zip(axes, mats):
        im = ax.imshow(mat, cmap=cmap, norm=norm, aspect="auto")
        ax.set_title(title, fontsize=14, fontweight="semibold")
        ax.set_xticks(np.arange(len(assays)))
        ax.set_xticklabels(assays)
        ax.set_yticks(np.arange(len(conds)))
        ax.set_yticklabels([condition_label(c) for c in conds])
        ax.set_xlabel("Assay")
        ax.set_facecolor(COLORS["panel"])
        for i in range(len(conds)):
            for j in range(len(assays)):
                val = mat[i, j]
                txt_col = "white" if abs(val) > 0.55 * vmax else "black"
                ax.text(j, i, f"{val:+.2f}", ha="center", va="center", fontsize=9.5, color=txt_col)
        ax.set_xticks(np.arange(-0.5, len(assays), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(conds), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=1.6)
        ax.tick_params(which="minor", length=0)

    axes[0].set_ylabel("Condition")
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Relative effect vs C2", fontsize=11)
    cbar.ax.tick_params(labelsize=10)
    save_figure(fig, out_dir, "style2_dual_heatmap_experiment_vs_simulation")


def export_style3_assay_scatter(tidy: pd.DataFrame, out_dir: Path) -> None:
    assay_colors = {"ALP": "#b07d39", "Collagen": "#4c956c", "ARS": "#4c78a8"}
    fig, ax = plt.subplots(figsize=(8.9, 8.2))
    ax.set_facecolor(COLORS["panel"])

    xmin, xmax = [], []
    for assay in ["ALP", "Collagen", "ARS"]:
        sub = tidy[tidy["assay_title"] == assay].copy()
        x = sub["exp_rel_effect"].to_numpy(dtype=float)
        y = sub["sim_rel_effect"].to_numpy(dtype=float)
        xmin.extend(x.tolist())
        xmax.extend(y.tolist())
        ax.scatter(
            x,
            y,
            s=95,
            color=assay_colors[assay],
            edgecolor="white",
            linewidth=0.9,
            alpha=0.95,
            label=assay,
        )
        for _, r in sub.iterrows():
            ax.annotate(
                str(r["condition_code"]),
                (float(r["exp_rel_effect"]), float(r["sim_rel_effect"])),
                textcoords="offset points",
                xytext=(4, 4),
                fontsize=8.5,
                color="black",
            )
    lo = min(xmin + xmax)
    hi = max(xmin + xmax)
    pad = 0.08 * max(abs(lo), abs(hi), 0.2)
    lo2, hi2 = lo - pad, hi + pad
    ax.plot([lo2, hi2], [lo2, hi2], linestyle="--", color="#4b5563", linewidth=1.2)
    ax.axhline(0.0, color="#9ca3af", linewidth=0.9, alpha=0.8)
    ax.axvline(0.0, color="#9ca3af", linewidth=0.9, alpha=0.8)
    ax.set_xlim(lo2, hi2)
    ax.set_ylim(lo2, hi2)
    ax.grid(color=COLORS["grid"], linewidth=0.9, alpha=0.55)
    ax.set_xlabel("Experimental relative effect vs C2")
    ax.set_ylabel("Simulation relative effect vs C2")
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, out_dir, "style3_agreement_scatter_by_assay")


def export_creative1_butterfly(tidy: pd.DataFrame, out_dir: Path) -> None:
    assay_colors = {"ALP": "#bc6c25", "Collagen": "#2d6a4f", "ARS": "#2a6f97"}
    fig, axes = plt.subplots(1, 3, figsize=(16.2, 8.1), sharey=True)
    y = np.arange(len(CONDITION_ORDER))
    cond_labels = [condition_label(c) for c in CONDITION_ORDER]

    for ax, assay in zip(axes, ["ALP", "Collagen", "ARS"]):
        sub = tidy[tidy["assay_title"] == assay].copy()
        sub["condition_code"] = pd.Categorical(sub["condition_code"], categories=CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition_code").reset_index(drop=True)
        exp_v = sub["exp_rel_effect"].to_numpy(dtype=float)
        sim_v = sub["sim_rel_effect"].to_numpy(dtype=float)

        ax.barh(y, -exp_v, height=0.64, color="#374151", alpha=0.90, label="Experiment")
        ax.barh(y, sim_v, height=0.46, color=assay_colors[assay], alpha=0.92, label="Simulation")
        ax.axvline(0.0, color="#111827", linewidth=1.2)
        ax.set_title(assay, fontsize=15, fontweight="semibold")
        ax.set_facecolor(COLORS["panel"])
        ax.grid(axis="x", color=COLORS["grid"], linewidth=0.9, alpha=0.65)
        ax.tick_params(direction="out", length=4)

        lim = max(0.12, float(np.nanpercentile(np.abs(np.concatenate([exp_v, sim_v])), 98)))
        ax.set_xlim(-1.18 * lim, 1.18 * lim)
        ax.set_xlabel("Left: experiment   |   Right: simulation")
        if assay == "ALP":
            ax.set_yticks(y)
            ax.set_yticklabels(cond_labels, fontsize=10.5)
            ax.set_ylabel("Condition")
        else:
            ax.set_yticks(y)
            ax.set_yticklabels([])
        ax.invert_yaxis()

    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.01), fontsize=12)
    save_figure(fig, out_dir, "creative1_butterfly_mirrored_bars")


def export_creative2_radar_triptych(tidy: pd.DataFrame, out_dir: Path) -> None:
    text_scale = 1.55
    assays = ["ALP", "Collagen", "ARS"]
    # One cohesive palette for all panels; distinction is by style.
    exp_col = "#1f2937"
    sim_col = "#0f766e"
    n = len(CONDITION_ORDER)
    angles = np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    angles_closed = np.r_[angles, angles[0]]

    fig, axes = plt.subplots(1, 3, figsize=(17.2, 7.0), subplot_kw={"projection": "polar"})
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, assay in zip(axes, assays):
        sub = tidy[tidy["assay_title"] == assay].copy()
        sub["condition_code"] = pd.Categorical(sub["condition_code"], categories=CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition_code").reset_index(drop=True)
        exp_v = sub["exp_rel_effect"].to_numpy(dtype=float)
        sim_v = sub["sim_rel_effect"].to_numpy(dtype=float)

        lo = min(float(np.min(exp_v)), float(np.min(sim_v)))
        hi = max(float(np.max(exp_v)), float(np.max(sim_v)))
        pad = 0.08 * max(abs(lo), abs(hi), 0.25)
        lo -= pad
        hi += pad
        zero_r = 0.0 - lo
        exp_r = exp_v - lo
        sim_r = sim_v - lo

        exp_closed = np.r_[exp_r, exp_r[0]]
        sim_closed = np.r_[sim_r, sim_r[0]]
        ax.plot(
            angles_closed,
            exp_closed,
            color=exp_col,
            linewidth=2.8,
            marker="o",
            markersize=5.2,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Experiment",
        )
        ax.fill(angles_closed, exp_closed, color=exp_col, alpha=0.10)
        ax.plot(
            angles_closed,
            sim_closed,
            color=sim_col,
            linewidth=2.8,
            linestyle="--",
            marker="s",
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=1.2,
            label="Simulation",
        )
        ax.fill(angles_closed, sim_closed, color=sim_col, alpha=0.18)
        ax.plot(angles_closed, np.full_like(angles_closed, zero_r), linestyle="--", color="#6b7280", linewidth=1.0, alpha=0.9)

        ax.set_theta_offset(np.pi / 2.0)
        ax.set_theta_direction(-1)
        ax.set_xticks(angles)
        ax.set_xticklabels(CONDITION_ORDER, fontsize=10.5 * text_scale, color="black")
        ax.set_ylim(0, hi - lo)
        ax.set_yticklabels([])
        ax.grid(color=COLORS["grid"], alpha=0.72, linewidth=0.9)
        ax.set_facecolor(COLORS["panel"])
        ax.set_title(assay, y=1.14, fontsize=15 * text_scale, fontweight="semibold")

    legend_handles = [
        plt.Line2D([0], [0], color=exp_col, lw=2.8, marker="o", markersize=6.2, markerfacecolor="white", label="Experiment"),
        plt.Line2D([0], [0], color=sim_col, lw=2.8, ls="--", marker="s", markersize=6.2, markerfacecolor="white", label="Simulation"),
    ]
    fig.legend(
        legend_handles,
        ["Experiment", "Simulation"],
        frameon=False,
        ncol=2,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.07),
        fontsize=12 * text_scale,
    )
    save_figure(fig, out_dir, "creative2_radar_triptych_exp_vs_sim", png_dpi=600, pdf_dpi=300)


def export_creative3_ribbon_flow(tidy: pd.DataFrame, out_dir: Path) -> None:
    assays = ["ALP", "Collagen", "ARS"]
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 7.6), sharey=False)
    fig.patch.set_facecolor(COLORS["bg"])

    for ax, assay in zip(axes, assays):
        sub = tidy[tidy["assay_title"] == assay].copy()
        sub["condition_code"] = pd.Categorical(sub["condition_code"], categories=CONDITION_ORDER, ordered=True)
        sub = sub.sort_values("condition_code").reset_index(drop=True)
        exp_v = sub["exp_rel_effect"].to_numpy(dtype=float)
        sim_v = sub["sim_rel_effect"].to_numpy(dtype=float)
        codes = sub["condition_code"].astype(str).tolist()

        all_vals = np.concatenate([exp_v, sim_v])
        lim = max(0.12, float(np.nanpercentile(np.abs(all_vals), 98)))
        ax.set_xlim(-0.08, 1.08)
        ax.set_ylim(-1.15 * lim, 1.15 * lim)
        ax.axhline(0.0, color="#6b7280", linewidth=1.0, linestyle="--", alpha=0.8)
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.9, alpha=0.65)
        ax.set_facecolor(COLORS["panel"])

        for e, s, c in zip(exp_v, sim_v, codes):
            d = s - e
            col = "#2563eb" if d >= 0 else "#b45309"
            lw = 1.0 + 8.0 * min(abs(d) / (lim + 1e-9), 1.0)
            ax.plot([0, 1], [e, s], color=col, alpha=0.34, linewidth=lw, solid_capstyle="round", zorder=1)
            ax.scatter([0], [e], s=42, color="#111827", edgecolor="white", linewidth=0.8, zorder=3)
            ax.scatter([1], [s], s=42, color=col, edgecolor="white", linewidth=0.8, zorder=3)
            ax.text(1.02, s, c, fontsize=8.7, va="center", ha="left", color="black")

        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Experiment", "Simulation"])
        ax.set_title(assay, fontsize=15, fontweight="semibold")
        if assay == "ALP":
            ax.set_ylabel("Relative effect vs C2")
        ax.tick_params(direction="out", length=4)

    save_figure(fig, out_dir, "creative3_ribbon_flow_exp_to_sim")


def main() -> None:
    args = parse_args()
    out_dir = args.out if args.out is not None else args.input / "figure_extract_5" / "comparison_extract_3styles"

    assay_frames, _metrics_df = build_all_assays(args.input)
    tidy = build_tidy_from_assays(assay_frames)
    export_style1_lollipop(tidy, out_dir)
    export_style2_dual_heatmap(tidy, out_dir)
    export_style3_assay_scatter(tidy, out_dir)
    creative_out = out_dir / "creative_alternatives"
    export_creative1_butterfly(tidy, creative_out)
    export_creative2_radar_triptych(tidy, creative_out)
    export_creative3_ribbon_flow(tidy, creative_out)

    out_dir.mkdir(parents=True, exist_ok=True)
    tidy.to_csv(out_dir / "comparison_extract_3styles_tidy.csv", index=False)
    tidy.to_csv(creative_out / "creative_alternatives_tidy.csv", index=False)
    print(f"[comparison_extract] processed sanity5 input from: {args.input}")
    print(f"[comparison_extract] wrote 3 style figures to: {out_dir}")
    print(f"[comparison_extract] wrote creative alternatives to: {creative_out}")


if __name__ == "__main__":
    main()
