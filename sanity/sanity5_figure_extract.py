#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DEFAULT = SCRIPT_DIR / "out" / "sanity_check5"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 12345

CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
CONDITION_LABELS = {
    "C1": "Basal",
    "C2": "AA+beta-GP",
    "C3": "D_L K_L",
    "C4": "D_L K_H",
    "C5": "D_H K_L",
    "C6": "D_H K_H",
    "C7": "D_H K_0",
    "C8": "D_0 K_H",
}

ASSAY_SPECS = {
    "alp": {
        "sim_col": "alp_240h",
        "exp_file": "alp/alp_raw_alp_metrics_raw.csv",
        "exp_col": "ALP_mU_mL",
        "exp_group_col": "condition",
        "title": "ALP",
        "sim_endpoint": "240 h ALP total",
        "exp_endpoint": "experimental ALP_mU_mL",
        "profile_ylabel_full": "ALP activity (a.u.)",
        "color": "#D55E00",
        "marker": "o",
    },
    "collagen": {
        "sim_col": "collagen_336h",
        "exp_file": "col/collagen_raw_metrics_RAW.csv",
        "exp_col": "M1_field",
        "exp_group_col": "condition",
        "title": "Collagen",
        "sim_endpoint": "336 h collagen total",
        "exp_endpoint": "experimental M1_field",
        "profile_ylabel_full": "collagen intensity (a.u.)",
        "color": "#009E73",
        "marker": "s",
    },
    "ars": {
        "sim_col": "ars_672h",
        "exp_file": "ars/ars_ratio1_ars_metrics_absolute.csv",
        "exp_col": "fraction_percent",
        "exp_group_col": "cond",
        "title": "ARS",
        "sim_endpoint": "672 h HAp total",
        "exp_endpoint": "experimental fraction_percent",
        "profile_ylabel_full": "ARS+ area (%)",
        "color": "#0072B2",
        "marker": "^",
    },
}

COLORS = {
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#dde3ea",
    "sim": "#111827",
    "sim_line": "#4b5563",
    "bg": "#ffffff",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract predictive-comparison figures and tables for sanity_check5.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=INPUT_DEFAULT,
        help="root folder containing 'simulation' and 'experimental' subfolders",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder; defaults to <input>/figure_extract_5",
    )
    parser.add_argument(
        "--main-figure",
        action="store_true",
        help="write the main 2x3 predictive-comparison figure",
    )
    parser.add_argument(
        "--supplement-figure",
        action="store_true",
        help="write supplemental calibration and pooled comparison figure",
    )
    parser.add_argument(
        "--metrics-table",
        action="store_true",
        help="write tidy assay-comparison and metrics CSV tables",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=BOOTSTRAP_SAMPLES,
        help="number of bootstrap resamples over conditions for correlation confidence intervals",
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=BOOTSTRAP_SEED,
        help="bootstrap RNG seed",
    )
    args = parser.parse_args()
    if not (args.main_figure or args.supplement_figure or args.metrics_table):
        args.main_figure = True
        args.supplement_figure = True
        args.metrics_table = True
    return args


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def style_axes(
    ax: plt.Axes, grid_axis: str | None = "y", grid_alpha: float = 0.45
) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.5, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=grid_alpha)


def condition_index(code: str) -> int:
    return int(str(code).replace("C", ""))


def zscore(values: pd.Series) -> pd.Series:
    vals = values.astype(float)
    sd = float(vals.std(ddof=0))
    if sd <= 0.0:
        return pd.Series(np.zeros(len(vals), dtype=float), index=values.index)
    return (vals - float(vals.mean())) / sd


def raw_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        df["sim_mean"].to_numpy(dtype=float),
        df["exp_mean"].to_numpy(dtype=float),
    )


def standardized_values(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    return (
        df["sim_z"].to_numpy(dtype=float),
        df["exp_z"].to_numpy(dtype=float),
    )


def bootstrap_corr_ci(
    x: np.ndarray,
    y: np.ndarray,
    kind: str,
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    samples: list[float] = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        xb = x[idx]
        yb = y[idx]
        if np.allclose(xb, xb[0]) or np.allclose(yb, yb[0]):
            continue
        try:
            if kind == "spearman":
                val = float(spearmanr(xb, yb).statistic)
            else:
                val = float(pearsonr(xb, yb).statistic)
        except Exception:
            continue
        if np.isfinite(val):
            samples.append(val)
    if not samples:
        return float("nan"), float("nan")
    return float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def compute_metrics(
    df: pd.DataFrame, *, n_boot: int, seed: int
) -> dict[str, float | int]:
    sim_raw, exp_raw = raw_values(df)
    spearman_val = float(spearmanr(sim_raw, exp_raw).statistic)
    pearson_val = float(pearsonr(sim_raw, exp_raw).statistic)
    rmse_val = float(np.sqrt(np.mean((sim_raw - exp_raw) ** 2)))
    mae_val = float(np.mean(np.abs(sim_raw - exp_raw)))
    spearman_ci_low, spearman_ci_high = bootstrap_corr_ci(
        sim_raw, exp_raw, "spearman", n_boot=n_boot, seed=seed
    )
    pearson_ci_low, pearson_ci_high = bootstrap_corr_ci(
        sim_raw, exp_raw, "pearson", n_boot=n_boot, seed=seed + 1
    )
    return {
        "spearman_rho": spearman_val,
        "spearman_ci_low": spearman_ci_low,
        "spearman_ci_high": spearman_ci_high,
        "pearson_r": pearson_val,
        "pearson_ci_low": pearson_ci_low,
        "pearson_ci_high": pearson_ci_high,
        "rmse": rmse_val,
        "mae": mae_val,
        "n_conditions": int(len(df)),
        "sim_n": int(df["sim_n"].iloc[0]),
        "exp_n": (
            int(df["exp_n"].min())
            if int(df["exp_n"].min()) == int(df["exp_n"].max())
            else -1
        ),
        "exp_n_min": int(df["exp_n"].min()),
        "exp_n_max": int(df["exp_n"].max()),
    }


def load_simulation(input_dir: Path) -> pd.DataFrame:
    candidates = [
        input_dir / "simulation" / "condition_endpoint_summary.csv",
        input_dir / "condition_endpoint_summary.csv",
        PROJECT_ROOT
        / "out"
        / "sanity_check5"
        / "simulation"
        / "condition_endpoint_summary.csv",
        PROJECT_ROOT / "out" / "sanity_check5" / "condition_endpoint_summary.csv",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise SystemExit(
            f"missing simulation summary: tried {', '.join(str(p) for p in candidates)}"
        )
    df = pd.read_csv(path)
    df["condition_code"] = df["condition_code"].astype(str)
    return df


def resolve_experimental_dir(input_dir: Path) -> Path:
    candidates = [
        input_dir / "experimental",
        PROJECT_ROOT / "endpointaad" / "experimentresults",
        PROJECT_ROOT / "out" / "sanity_check5" / "experimental",
    ]
    path = next((p for p in candidates if p.exists()), None)
    if path is None:
        raise SystemExit(
            f"missing experimental folder: tried {', '.join(str(p) for p in candidates)}"
        )
    return path


def load_experimental_assay(exp_dir: Path, assay: str) -> pd.DataFrame:
    spec = ASSAY_SPECS[assay]
    path = exp_dir / spec["exp_file"]
    if not path.exists():
        raise SystemExit(f"missing experimental file: {path}")

    if assay == "alp":
        raw = pd.read_csv(path)
        out = (
            raw.groupby(spec["exp_group_col"], as_index=False)[spec["exp_col"]]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    spec["exp_group_col"]: "condition_code",
                    "mean": "exp_mean",
                    "std": "exp_sd",
                    "count": "exp_n",
                }
            )
        )
    else:
        raw = pd.read_csv(path)
        out = (
            raw.groupby(spec["exp_group_col"], as_index=False)[spec["exp_col"]]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(
                columns={
                    spec["exp_group_col"]: "condition_code",
                    "mean": "exp_mean",
                    "std": "exp_sd",
                    "count": "exp_n",
                }
            )
        )
    out["condition_code"] = out["condition_code"].astype(str)
    return out


def build_assay_frame(
    input_dir: Path,
    assay: str,
    bootstrap_samples: int,
    seed_offset: int,
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    sim = load_simulation(input_dir)
    exp = load_experimental_assay(resolve_experimental_dir(input_dir), assay)
    spec = ASSAY_SPECS[assay]

    merged = (
        sim[
            [
                "condition_code",
                "n_replicates",
                f"{spec['sim_col']}_mean",
                f"{spec['sim_col']}_sd",
            ]
        ]
        .rename(
            columns={
                "n_replicates": "sim_n",
                f"{spec['sim_col']}_mean": "sim_mean",
                f"{spec['sim_col']}_sd": "sim_sd",
            }
        )
        .merge(exp, on="condition_code", how="inner")
        .copy()
    )
    merged["condition"] = pd.Categorical(
        merged["condition_code"], categories=CONDITION_ORDER, ordered=True
    )
    merged = merged.sort_values("condition").reset_index(drop=True)
    merged["condition_label"] = merged["condition_code"].map(CONDITION_LABELS)
    merged["sim_z"] = zscore(merged["sim_mean"])
    merged["exp_z"] = zscore(merged["exp_mean"])
    sim_scale = float(merged["sim_mean"].astype(float).std(ddof=0))
    exp_scale = float(merged["exp_mean"].astype(float).std(ddof=0))
    merged["sim_sd_z"] = (
        merged["sim_sd"].astype(float) / sim_scale if sim_scale > 1e-12 else 0.0
    )
    merged["exp_sd_z"] = (
        merged["exp_sd"].astype(float) / exp_scale if exp_scale > 1e-12 else 0.0
    )
    seed = seed_offset + 100 * condition_index(merged["condition_code"].iloc[0])
    metrics = compute_metrics(merged, n_boot=bootstrap_samples, seed=seed)
    return merged, metrics


def build_all_assays(
    input_dir: Path, bootstrap_samples: int, bootstrap_seed: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    assay_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, float | int | str]] = []
    for i, assay in enumerate(("alp", "collagen", "ars")):
        frame, metrics = build_assay_frame(
            input_dir, assay, bootstrap_samples, bootstrap_seed + i
        )
        assay_frames[assay] = frame
        metric_rows.append(
            {
                "assay": assay,
                "title": ASSAY_SPECS[assay]["title"],
                **metrics,
            }
        )
    metrics_df = pd.DataFrame(metric_rows)
    return assay_frames, metrics_df


def write_tables(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    tidy_rows: list[pd.DataFrame] = []
    for assay, frame in assay_frames.items():
        tidy = frame[
            [
                "condition_code",
                "sim_mean",
                "sim_sd",
                "exp_mean",
                "exp_sd",
                "sim_z",
                "exp_z",
            ]
        ].copy()
        tidy.insert(0, "assay", assay)
        tidy = tidy.rename(columns={"condition_code": "condition"})
        tidy_rows.append(tidy)
    tidy_df = pd.concat(tidy_rows, ignore_index=True)
    tidy_df.to_csv(out_dir / "condition_comparison_tidy.csv", index=False)
    metrics_df.to_csv(out_dir / "assay_metrics.csv", index=False)


def main_panel_metric_text(metrics: pd.Series) -> str:
    return (
        f"Spearman ρ = {metrics['spearman_rho']:.2f}\n"
        f"95% CI = [{metrics['spearman_ci_low']:.2f}, {metrics['spearman_ci_high']:.2f}]"
    )


def supplement_metric_text(metrics: pd.Series) -> str:
    return (
        f"Pearson r = {metrics['pearson_r']:.2f}\n"
        f"RMSE = {metrics['rmse']:.2g}\n"
        f"MAE = {metrics['mae']:.2g}"
    )


def scatter_label_offsets(points: np.ndarray) -> list[tuple[float, float]]:
    offsets: list[tuple[float, float]] = []
    for i, (x, y) in enumerate(points):
        dx = 0.035 if i % 2 == 0 else -0.10
        dy = 0.055 if i % 3 == 0 else (-0.06 if i % 3 == 1 else 0.015)
        for ox, oy in offsets:
            if abs((x + dx) - ox) < 0.12 and abs((y + dy) - oy) < 0.12:
                dy += 0.08 if dy <= 0 else -0.08
        offsets.append((x + dx, y + dy))
    return [(ox - x, oy - y) for (x, y), (ox, oy) in zip(points, offsets)]


def plot_profile_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metrics: pd.Series,
    assay: str,
    letter: str,
) -> None:
    spec = ASSAY_SPECS[assay]
    style_axes(ax, "y", grid_alpha=0.35)
    x = np.arange(len(frame), dtype=float)
    exp_vals = frame["exp_z"].to_numpy(dtype=float)
    exp_sd = frame["exp_sd_z"].to_numpy(dtype=float)
    sim_vals = frame["sim_z"].to_numpy(dtype=float)
    sim_sd = frame["sim_sd_z"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        exp_vals,
        yerr=exp_sd,
        color=spec["color"],
        marker=spec["marker"],
        ms=5.0,
        lw=1.8,
        capsize=2.0,
        capthick=0.8,
        elinewidth=0.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        linestyle="-",
        zorder=4,
    )
    ax.errorbar(
        x,
        sim_vals,
        yerr=sim_sd,
        color=COLORS["sim_line"],
        marker="D",
        ms=4.8,
        lw=1.4,
        ls=(0, (4, 2)),
        capsize=1.8,
        capthick=0.7,
        elinewidth=0.7,
        markerfacecolor="white",
        markeredgecolor=COLORS["sim_line"],
        markeredgewidth=0.7,
        alpha=0.9,
        zorder=3,
    )

    ax.text(
        0.0,
        1.13,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=12.0,
        fontweight="semibold",
    )
    ax.text(
        0.0,
        1.04,
        spec["title"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        fontweight="normal",
        color=COLORS["ink"],
    )
    ax.text(
        0.98,
        1.015,
        main_panel_metric_text(metrics),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.4,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    ax.set_xticks(x, frame["condition_code"])
    ax.set_xlim(-0.5, len(frame) - 0.5)
    ax.set_ylabel(spec["profile_ylabel_full"])
    ax.set_ylabel("Standardized endpoint (z)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ymin = float(min(np.min(exp_vals - exp_sd), np.min(sim_vals - sim_sd)))
    ymax = float(max(np.max(exp_vals + exp_sd), np.max(sim_vals + sim_sd)))
    pad = max(0.2, 0.12 * (ymax - ymin + 1e-12))
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(0.0, color=COLORS["muted"], lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.tick_params(axis="x", pad=3)


def plot_standardized_panel(
    ax: plt.Axes,
    frame: pd.DataFrame,
    metrics: pd.Series,
    assay: str,
    letter: str,
    shared_limit: float,
) -> None:
    spec = ASSAY_SPECS[assay]
    style_axes(ax, "both", grid_alpha=0.35)
    sim_z, exp_z = standardized_values(frame)
    limit = shared_limit
    ax.plot(
        [-limit, limit],
        [-limit, limit],
        color=COLORS["muted"],
        lw=1.0,
        ls=(0, (4, 3)),
        zorder=1,
    )
    ax.scatter(
        exp_z,
        sim_z,
        s=102,
        color=spec["color"],
        marker=spec["marker"],
        edgecolors="white",
        linewidths=0.8,
        zorder=3,
    )
    offsets = scatter_label_offsets(np.column_stack([exp_z, sim_z]))
    for (_, row), (dx, dy) in zip(frame.iterrows(), offsets):
        ax.text(
            float(row["exp_z"]) + dx,
            float(row["sim_z"]) + dy,
            str(row["condition_code"]),
            fontsize=8.3,
            color=COLORS["ink"],
        )
    ax.set_title(letter, loc="left", pad=5, fontsize=11, fontweight="semibold")
    ax.text(
        0.03,
        1.02,
        spec["title"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=11,
        color=COLORS["ink"],
    )
    ax.text(
        0.97,
        0.92,
        main_panel_metric_text(metrics),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8.0,
        color=COLORS["ink"],
    )
    ax.set_xlabel("Experimental standardized response")
    ax.set_ylabel("Simulation standardized response")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def write_captions(out_dir: Path) -> None:
    main_caption = (
        "Main figure. Per-assay simulation-versus-experiment comparison across conditions C1-C8. "
        "Simulation endpoints were ALP total at 240 h, collagen total at 336 h, and HAp total at 672 h. "
        "Experimental endpoints were ALP mean activity, collagen M1_field, and ARS fraction_percent. "
        "Top-row profiles show raw endpoint values with dual y-axes (left: experiment, right: simulation); no linear rescaling is used in the main panels. "
        "The primary analysis is assay-specific trend recovery across conditions, quantified by Spearman rho with 95% bootstrap confidence intervals. "
        "Pearson r and raw-unit calibration metrics are supplementary."
    )
    supplement_caption = (
        "Supplementary figure. Raw-unit calibration views for each assay and a pooled standardized comparison across all assay-condition points. "
        "These views are supplementary only. The pooled panel is descriptive and is not the primary basis for any claim because the central analysis is assay-specific trend recovery."
    )
    (out_dir / "main_figure_caption.txt").write_text(
        main_caption + "\n", encoding="utf-8"
    )
    (out_dir / "supplement_figure_caption.txt").write_text(
        supplement_caption + "\n", encoding="utf-8"
    )


def export_main_figure(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    # V1: clean dual-axis raw endpoint profiles (a.u.), condition-wise.
    fig = plt.figure(figsize=(12.8, 4.9))
    gs = fig.add_gridspec(
        1,
        3,
        width_ratios=[1.0, 1.0, 1.0],
        top=0.80,
        bottom=0.18,
        left=0.055,
        right=0.97,
        wspace=0.60,
    )
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for idx, assay in enumerate(("alp", "collagen", "ars")):
        spec = ASSAY_SPECS[assay]
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        ax = axes[idx]
        style_axes(ax, "y", grid_alpha=0.34)
        ax2 = ax.twinx()
        ax2.grid(False)
        ax2.spines["top"].set_visible(False)
        ax2.spines["left"].set_visible(False)
        ax2.spines["right"].set_color(COLORS["sim_line"])
        ax2.tick_params(
            axis="y", colors=COLORS["sim_line"], length=4.0, width=0.8, labelsize=8.8
        )

        x = np.arange(len(frame), dtype=float)
        x_exp = x - 0.08
        x_sim = x + 0.08
        exp_vals = frame["exp_mean"].to_numpy(dtype=float)
        exp_sd = frame["exp_sd"].to_numpy(dtype=float)
        sim_vals = frame["sim_mean"].to_numpy(dtype=float)
        sim_sd = frame["sim_sd"].to_numpy(dtype=float)
        ax.errorbar(
            x_exp,
            exp_vals,
            yerr=exp_sd,
            color=spec["color"],
            marker=spec["marker"],
            ms=5.1,
            lw=1.8,
            capsize=2.0,
            elinewidth=0.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            zorder=4,
        )
        ax2.errorbar(
            x_sim,
            sim_vals,
            yerr=sim_sd,
            color=COLORS["sim_line"],
            marker="D",
            ms=4.8,
            lw=1.2,
            ls=(0, (4, 2)),
            capsize=1.8,
            elinewidth=0.7,
            markerfacecolor="white",
            markeredgecolor=COLORS["sim_line"],
            markeredgewidth=0.7,
            alpha=0.85,
            zorder=3,
        )
        ax.text(
            0.0,
            1.13,
            chr(65 + idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="semibold",
        )
        ax.text(
            0.0,
            1.04,
            spec["title"],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.2,
        )
        ax.text(
            0.98,
            1.015,
            main_panel_metric_text(metrics),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
        )
        ax.set_xticks(x, frame["condition_code"])
        ax.set_xlim(-0.5, len(frame) - 0.5)
        ax.set_ylabel(spec["profile_ylabel_full"])
        if idx == 2:
            ax2.set_ylabel(
                "Simulation endpoint (a.u.)",
                color=COLORS["sim_line"],
                fontsize=8.8,
                labelpad=4,
            )
        exp_rng = max(
            float(np.max(exp_vals + exp_sd) - np.min(exp_vals - exp_sd)), 1e-9
        )
        sim_rng = max(
            float(np.max(sim_vals + sim_sd) - np.min(sim_vals - sim_sd)), 1e-9
        )
        ax.set_ylim(
            max(0.0, float(np.min(exp_vals - exp_sd) - 0.08 * exp_rng)),
            float(np.max(exp_vals + exp_sd) + 0.10 * exp_rng),
        )
        ax2.set_ylim(
            max(0.0, float(np.min(sim_vals - sim_sd) - 0.08 * sim_rng)),
            float(np.max(sim_vals + sim_sd) + 0.10 * sim_rng),
        )
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax2.yaxis.set_major_locator(MaxNLocator(nbins=4))

    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["muted"],
            marker="o",
            linestyle="-",
            linewidth=1.4,
            markersize=5.5,
            markerfacecolor=COLORS["muted"],
            markeredgecolor=COLORS["muted"],
            label="Experiment",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["sim_line"],
            marker="D",
            linestyle=(0, (4, 2)),
            linewidth=1.0,
            markersize=5,
            markerfacecolor="white",
            markeredgecolor=COLORS["sim_line"],
            label="Simulation",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.52, 0.975),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=9.0,
        ncol=2,
    )
    fig.suptitle(
        "Simulation vs Experimental Endpoints Across Conditions (Raw a.u.)",
        fontsize=12.5,
        fontweight="semibold",
        y=0.99,
    )
    save_figure(fig, out_dir, "sanity5_predictive_comparison_main")

    # V2: stacked raw profiles (Experiment top row, Simulation bottom row), clearer scale reading.
    fig2, axes2 = plt.subplots(
        2, 3, figsize=(12.6, 6.4), sharex="col", constrained_layout=True
    )
    for c, assay in enumerate(("alp", "collagen", "ars")):
        spec = ASSAY_SPECS[assay]
        frame = assay_frames[assay]
        x = np.arange(len(frame), dtype=float)
        ax_top = axes2[0, c]
        style_axes(ax_top, "y", grid_alpha=0.32)
        ax_top.errorbar(
            x,
            frame["exp_mean"].to_numpy(dtype=float),
            yerr=frame["exp_sd"].to_numpy(dtype=float),
            color=spec["color"],
            marker=spec["marker"],
            lw=1.8,
            ms=5.0,
            capsize=2.0,
        )
        ax_top.set_title(spec["title"], pad=8, fontsize=11.5)
        ax_top.set_ylabel("Experiment (a.u.)")
        ax_top.set_xticks(x, frame["condition_code"])
        ax_bot = axes2[1, c]
        style_axes(ax_bot, "y", grid_alpha=0.32)
        ax_bot.errorbar(
            x,
            frame["sim_mean"].to_numpy(dtype=float),
            yerr=frame["sim_sd"].to_numpy(dtype=float),
            color=COLORS["sim_line"],
            marker="D",
            markerfacecolor="white",
            lw=1.3,
            ls=(0, (4, 2)),
            ms=4.8,
            capsize=1.8,
        )
        ax_bot.set_ylabel("Simulation (a.u.)")
        ax_bot.set_xlabel("Condition")
        ax_bot.set_xticks(x, frame["condition_code"])
    fig2.suptitle(
        "Raw Endpoint Profiles by Assay", fontsize=12.4, fontweight="semibold"
    )
    save_figure(fig2, out_dir, "sanity5_predictive_comparison_main_v2")

    # V3: raw calibration scatter (experiment vs simulation) with uncertainty and identity line.
    fig3, axes3 = plt.subplots(1, 3, figsize=(12.4, 4.1), constrained_layout=True)
    for i, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes3[i]
        spec = ASSAY_SPECS[assay]
        frame = assay_frames[assay]
        style_axes(ax, "both", grid_alpha=0.30)
        x = frame["exp_mean"].to_numpy(dtype=float)
        y = frame["sim_mean"].to_numpy(dtype=float)
        xerr = frame["exp_sd"].to_numpy(dtype=float)
        yerr = frame["sim_sd"].to_numpy(dtype=float)
        lo = min(float(np.min(x - xerr)), float(np.min(y - yerr)))
        hi = max(float(np.max(x + xerr)), float(np.max(y + yerr)))
        pad = 0.06 * (hi - lo + 1e-12)
        lo = max(0.0, lo - pad)
        hi = hi + pad
        ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)))
        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            fmt=spec["marker"],
            color=spec["color"],
            markeredgecolor="white",
            markeredgewidth=0.8,
            ms=6.0,
            elinewidth=0.85,
            capsize=2.0,
            lw=0,
        )
        for _, row in frame.iterrows():
            ax.text(
                float(row["exp_mean"]) + 0.01 * (hi - lo),
                float(row["sim_mean"]),
                str(row["condition_code"]),
                fontsize=7.5,
            )
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_title(spec["title"], pad=8, fontsize=11.5)
        ax.set_xlabel("Experimental endpoint (a.u.)")
        if i == 0:
            ax.set_ylabel("Simulation endpoint (a.u.)")
    fig3.suptitle(
        "Raw Endpoint Calibration by Assay", fontsize=12.4, fontweight="semibold"
    )
    save_figure(fig3, out_dir, "sanity5_predictive_comparison_main_v3")


def export_discrete_condition_figure(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(12.4, 4.4), sharey=True, constrained_layout=True
    )
    sim_color = "#B2182B"
    for idx, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes[idx]
        style_axes(ax, "y", grid_alpha=0.32)
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spec = ASSAY_SPECS[assay]
        x = np.arange(len(frame), dtype=float)
        x_exp = x - 0.10
        x_sim = x + 0.10

        ax.axhline(0.0, color=COLORS["muted"], lw=0.9, ls=(0, (4, 3)), zorder=1)
        # light connecting trends for condition ordering
        ax.plot(
            x_exp,
            frame["exp_z"].to_numpy(dtype=float),
            color=spec["color"],
            lw=1.5,
            alpha=0.55,
            ls=(0, (4, 2)),
            zorder=2,
        )
        ax.plot(
            x_sim,
            frame["sim_z"].to_numpy(dtype=float),
            color=sim_color,
            lw=1.8,
            alpha=0.9,
            ls="-",
            zorder=2,
        )
        ax.errorbar(
            x_exp,
            frame["exp_z"].to_numpy(dtype=float),
            yerr=frame["exp_sd_z"].to_numpy(dtype=float),
            fmt=spec["marker"],
            color=spec["color"],
            markersize=5.4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            elinewidth=0.9,
            capsize=2.0,
            capthick=0.8,
            lw=0,
            zorder=3,
            label="Experiment",
        )
        ax.errorbar(
            x_sim,
            frame["sim_z"].to_numpy(dtype=float),
            yerr=frame["sim_sd_z"].to_numpy(dtype=float),
            fmt="D",
            color=sim_color,
            markersize=5.0,
            markerfacecolor="white",
            markeredgewidth=0.8,
            markeredgecolor=sim_color,
            elinewidth=0.9,
            capsize=1.8,
            capthick=0.7,
            lw=0,
            zorder=2,
            alpha=0.95,
            label="Simulation",
        )

        ax.set_title(spec["title"], pad=8, fontsize=11.5)
        ax.text(
            0.98,
            0.96,
            main_panel_metric_text(metrics),
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=7.3,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
        ax.set_xticks(x, frame["condition_code"])
        ax.set_xlim(-0.5, len(frame) - 0.5)
        if idx == 0:
            ax.set_ylabel("Standardized response (z)")
        ax.set_xlabel("Condition")

    handles = [
        Line2D(
            [0],
            [0],
            color="gray",
            marker="o",
            lw=1.5,
            alpha=0.65,
            ls=(0, (4, 2)),
            markersize=5.8,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="Experiment",
        ),
        Line2D(
            [0],
            [0],
            color=sim_color,
            marker="D",
            lw=1.8,
            ls="-",
            markersize=5.4,
            markerfacecolor="white",
            markeredgecolor=sim_color,
            markeredgewidth=0.9,
            label="Simulation",
        ),
    ]
    fig.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.985, 1.045),
        ncol=1,
        frameon=False,
        fontsize=9.0,
    )
    fig.suptitle(
        "Condition-Wise Endpoint Comparison",
        fontsize=12.5,
        fontweight="semibold",
        y=1.07,
    )
    save_figure(fig, out_dir, "sanity5_discrete_condition_comparison")


def export_supplement_figure(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax in axes.ravel():
        style_axes(ax, "both", grid_alpha=0.35)

    for idx, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes.ravel()[idx]
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spec = ASSAY_SPECS[assay]
        sim_raw, exp_raw = raw_values(frame)
        exp_plot = np.log10(np.maximum(exp_raw, 0.0) + 1.0)
        sim_plot = np.log10(np.maximum(sim_raw, 0.0) + 1.0)
        lo = min(np.min(sim_plot), np.min(exp_plot))
        hi = max(np.max(sim_plot), np.max(exp_plot))
        ax.scatter(
            exp_plot,
            sim_plot,
            s=78,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.8,
        )
        ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)))
        for _, row in frame.iterrows():
            ex = float(np.log10(max(float(row["exp_mean"]), 0.0) + 1.0))
            sy = float(np.log10(max(float(row["sim_mean"]), 0.0) + 1.0))
            ax.text(
                ex + 0.01 * (hi - lo + 1e-9), sy, row["condition_code"], fontsize=7.8
            )
        ax.set_title(
            f"{chr(65 + idx)}  {spec['title']}", loc="left", pad=8, fontsize=11.5
        )
        ax.text(
            0.02,
            0.96,
            supplement_metric_text(metrics),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9.0,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
        )
        ax.set_xlabel("log10(experimental + 1)")
        ax.set_ylabel("log10(simulation + 1)")

    pooled_ax = axes[1, 1]
    pooled_rows = []
    for assay, frame in assay_frames.items():
        tmp = frame[["exp_z", "sim_z"]].copy()
        tmp["assay"] = assay
        tmp["condition_code"] = frame["condition_code"]
        pooled_rows.append(tmp)
    pooled = pd.concat(pooled_rows, ignore_index=True)
    limit = max(
        1.1,
        float(np.max(np.abs(pooled[["exp_z", "sim_z"]].to_numpy(dtype=float)))) * 1.15,
    )
    pooled_ax.plot(
        [-limit, limit], [-limit, limit], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3))
    )
    for assay in ("alp", "collagen", "ars"):
        spec = ASSAY_SPECS[assay]
        sub = pooled[pooled["assay"] == assay]
        pooled_ax.scatter(
            sub["exp_z"],
            sub["sim_z"],
            s=72,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.8,
            label=spec["title"],
        )
    pooled_r = float(pearsonr(pooled["exp_z"], pooled["sim_z"]).statistic)
    pooled_rho = float(spearmanr(pooled["exp_z"], pooled["sim_z"]).statistic)
    pooled_ax.text(
        0.02,
        0.96,
        f"Pooled standardized points\nPearson r = {pooled_r:.2f}\nSpearman rho = {pooled_rho:.2f}\nSupplementary view only",
        transform=pooled_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    pooled_ax.set_title(
        "D  Pooled standardized comparison", loc="left", pad=8, fontsize=11.5
    )
    pooled_ax.set_xlabel("Experimental standardized response")
    pooled_ax.set_ylabel("Simulation standardized response")
    pooled_ax.set_xlim(-limit, limit)
    pooled_ax.set_ylim(-limit, limit)
    pooled_ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.suptitle(
        "Supplementary Calibration and Pooled Comparison",
        fontsize=13.5,
        fontweight="semibold",
        y=0.985,
    )
    fig.subplots_adjust(
        left=0.09, right=0.985, bottom=0.08, top=0.92, wspace=0.30, hspace=0.34
    )
    save_figure(fig, out_dir, "sanity5_predictive_comparison_supplement")


def main() -> None:
    out_dir = args_global.out or (args_global.input / "figure_extract_5")
    out_dir.mkdir(parents=True, exist_ok=True)
    assay_frames, metrics_df = build_all_assays(
        args_global.input, args_global.bootstrap_samples, args_global.bootstrap_seed
    )

    if args_global.metrics_table:
        write_tables(assay_frames, metrics_df, out_dir)
    if args_global.main_figure:
        export_main_figure(assay_frames, metrics_df, out_dir)
        export_discrete_condition_figure(assay_frames, metrics_df, out_dir)
    if args_global.supplement_figure:
        export_supplement_figure(assay_frames, metrics_df, out_dir)

    write_captions(out_dir)
    print(f"Saved outputs to: {out_dir}")


args_global = parse_args()


if __name__ == "__main__":
    main()
