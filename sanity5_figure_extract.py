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


INPUT_DEFAULT = Path("out") / "sanity_check5"
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
        "exp_file": "alp_raw_ALP_means.txt",
        "exp_col": "Mean",
        "exp_group_col": "Condition",
        "title": "ALP",
        "sim_endpoint": "240 h ALP total",
        "exp_endpoint": "experimental ALP mean",
        "profile_ylabel_full": "ALP activity (a.u.)",
        "color": "#c75a1b",
        "marker": "o",
    },
    "collagen": {
        "sim_col": "collagen_336h",
        "exp_file": "collagen_raw_metrics_RAW.csv",
        "exp_col": "M1_field",
        "exp_group_col": "condition",
        "title": "Collagen",
        "sim_endpoint": "336 h collagen total",
        "exp_endpoint": "experimental M1_field",
        "profile_ylabel_full": "collagen intensity (a.u.)",
        "color": "#149670",
        "marker": "s",
    },
    "ars": {
        "sim_col": "ars_672h",
        "exp_file": "ars_ratio1_ars_metrics_absolute.csv",
        "exp_col": "fraction_percent",
        "exp_group_col": "cond",
        "title": "ARS",
        "sim_endpoint": "672 h HAp total",
        "exp_endpoint": "experimental fraction_percent",
        "profile_ylabel_full": "ARS+ area (%)",
        "color": "#2a5f9e",
        "marker": "^",
    },
}

COLORS = {
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#d7dee7",
    "sim": "#111827",
    "sim_line": "#6b7280",
    "bg": "#ffffff",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.5,
        "xtick.labelsize": 10.5,
        "ytick.labelsize": 10.5,
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


def style_axes(ax: plt.Axes, grid_axis: str | None = "y", grid_alpha: float = 0.45) -> None:
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


def compute_metrics(df: pd.DataFrame, *, n_boot: int, seed: int) -> dict[str, float | int]:
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
        "exp_n": int(df["exp_n"].min()) if int(df["exp_n"].min()) == int(df["exp_n"].max()) else -1,
        "exp_n_min": int(df["exp_n"].min()),
        "exp_n_max": int(df["exp_n"].max()),
    }


def load_simulation(input_dir: Path) -> pd.DataFrame:
    sim_dir = input_dir / "simulation"
    path = sim_dir / "condition_endpoint_summary.csv"
    if not path.exists():
        raise SystemExit(f"missing simulation summary: {path}")
    df = pd.read_csv(path)
    df["condition_code"] = df["condition_code"].astype(str)
    return df


def load_experimental_assay(exp_dir: Path, assay: str) -> pd.DataFrame:
    spec = ASSAY_SPECS[assay]
    path = exp_dir / spec["exp_file"]
    if not path.exists():
        raise SystemExit(f"missing experimental file: {path}")

    if assay == "alp":
        raw = pd.read_csv(path, sep="\t")
        out = pd.DataFrame(
            {
                "condition_code": raw["Condition"].astype(str),
                "exp_mean": raw["Mean"].astype(float),
                "exp_sd": raw["SD"].astype(float),
                "exp_n": raw["N"].astype(int),
            }
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
    exp = load_experimental_assay(input_dir / "experimental", assay)
    spec = ASSAY_SPECS[assay]

    merged = (
        sim[["condition_code", "n_replicates", f"{spec['sim_col']}_mean", f"{spec['sim_col']}_sd"]]
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
    merged["condition"] = pd.Categorical(merged["condition_code"], categories=CONDITION_ORDER, ordered=True)
    merged = merged.sort_values("condition").reset_index(drop=True)
    merged["condition_label"] = merged["condition_code"].map(CONDITION_LABELS)
    merged["sim_z"] = zscore(merged["sim_mean"])
    merged["exp_z"] = zscore(merged["exp_mean"])
    seed = seed_offset + 100 * condition_index(merged["condition_code"].iloc[0])
    metrics = compute_metrics(merged, n_boot=bootstrap_samples, seed=seed)
    return merged, metrics

def build_all_assays(
    input_dir: Path, bootstrap_samples: int, bootstrap_seed: int
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    assay_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, float | int | str]] = []
    for i, assay in enumerate(("alp", "collagen", "ars")):
        frame, metrics = build_assay_frame(input_dir, assay, bootstrap_samples, bootstrap_seed + i)
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


def write_tables(assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path) -> None:
    tidy_rows: list[pd.DataFrame] = []
    for assay, frame in assay_frames.items():
        tidy = frame[["condition_code", "sim_mean", "sim_sd", "exp_mean", "exp_sd", "sim_z", "exp_z"]].copy()
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



def plot_profile_panel(ax: plt.Axes, frame: pd.DataFrame, metrics: pd.Series, assay: str, letter: str) -> None:
    spec = ASSAY_SPECS[assay]
    style_axes(ax, "y", grid_alpha=0.35)
    ax_sim = ax.twinx()
    ax_sim.grid(False)
    ax_sim.spines["top"].set_visible(False)
    ax_sim.spines["left"].set_visible(False)
    ax_sim.spines["right"].set_color(COLORS["sim_line"])
    ax_sim.tick_params(axis="y", colors=COLORS["sim_line"], length=4.0, width=0.8, labelsize=9.0)
    x = np.arange(len(frame), dtype=float)
    x_exp = x - 0.08
    x_sim = x + 0.08
    sim_vals = frame["sim_mean"].to_numpy(dtype=float)
    sim_sd = frame["sim_sd"].to_numpy(dtype=float)
    exp_vals = frame["exp_mean"].to_numpy(dtype=float)
    exp_sd = frame["exp_sd"].to_numpy(dtype=float)

    ax.errorbar(
        x_exp,
        exp_vals,
        yerr=exp_sd,
        color=spec["color"],
        marker=spec["marker"],
        ms=5.4,
        lw=1.8,
        capsize=2.0,
        capthick=0.8,
        elinewidth=0.8,
        markeredgecolor="white",
        markeredgewidth=0.8,
        zorder=4,
    )
    ax_sim.errorbar(
        x_sim,
        sim_vals,
        yerr=sim_sd,
        color=COLORS["sim_line"],
        marker="D",
        ms=5.2,
        lw=1.4,
        ls=(0, (4, 2)),
        capsize=1.8,
        capthick=0.7,
        elinewidth=0.7,
        markerfacecolor="white",
        markeredgecolor=COLORS["sim_line"],
        markeredgewidth=0.7,
        alpha=0.7,
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
        1.02,
        main_panel_metric_text(metrics),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=7.6,
        color=COLORS["ink"],
    )
    ax.set_xticks(x, frame["condition_code"])
    ax.set_xlim(-0.5, len(frame) - 0.5)
    ax.set_ylabel(spec["profile_ylabel_full"])
    ax_sim.set_ylabel(f"{spec['title']} endpoint (a.u.)", color=COLORS["sim_line"])
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    exp_range = max(float(np.max(exp_vals + exp_sd) - np.min(exp_vals - exp_sd)), 1e-9)
    exp_ymin = float(np.min(exp_vals - exp_sd)) - 0.08 * exp_range
    exp_ymax = float(np.max(exp_vals + exp_sd)) + 0.10 * exp_range
    ax.set_ylim(exp_ymin, exp_ymax)

    sim_range = max(float(np.max(sim_vals + sim_sd) - np.min(sim_vals - sim_sd)), 1e-9)
    sim_ymin = float(np.min(sim_vals - sim_sd)) - 0.08 * sim_range
    sim_ymax = float(np.max(sim_vals + sim_sd)) + 0.10 * sim_range
    ax_sim.set_ylim(sim_ymin, sim_ymax)
    ax_sim.yaxis.set_major_locator(MaxNLocator(nbins=5))
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
    ax.plot([-limit, limit], [-limit, limit], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
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
    (out_dir / "main_figure_caption.txt").write_text(main_caption + "\n", encoding="utf-8")
    (out_dir / "supplement_figure_caption.txt").write_text(supplement_caption + "\n", encoding="utf-8")


def export_main_figure(assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig = plt.figure(figsize=(12.4, 4.2))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.0], top=0.78, bottom=0.2, left=0.05, right=0.96, wspace=0.60)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    for idx, assay in enumerate(("alp", "collagen", "ars")):
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        plot_profile_panel(axes[idx], assay_frames[assay], metrics, assay, chr(65 + idx))
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
        loc="upper right",
        bbox_to_anchor=(0.98, 0.99),
        bbox_transform=fig.transFigure,
        frameon=False,
        fontsize=9.0,
        ncol=1,
    )
    fig.suptitle(
        "Simulation-experiment comparison by conditions",
        fontsize=11.5,
        fontweight="normal",
        y=0.97,
    )
    save_figure(fig, out_dir, "sanity5_predictive_comparison_main")


def export_supplement_figure(assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.5, 7.0))
    for ax in axes.ravel():
        style_axes(ax, "both", grid_alpha=0.35)

    for idx, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes.ravel()[idx]
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spec = ASSAY_SPECS[assay]
        sim_raw, exp_raw = raw_values(frame)
        lo = min(np.min(sim_raw), np.min(exp_raw))
        hi = max(np.max(sim_raw), np.max(exp_raw))
        ax.scatter(
            exp_raw,
            sim_raw,
            s=78,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.8,
        )
        ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)))
        for _, row in frame.iterrows():
            ax.text(float(row["exp_mean"]) + 0.01 * (hi - lo + 1e-9), float(row["sim_mean"]), row["condition_code"], fontsize=7.8)
        ax.set_title(f"{chr(65 + idx)}  {spec['title']}", loc="left", pad=8)
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
        ax.set_xlabel("Experimental raw value")
        ax.set_ylabel("Simulation raw value")

    pooled_ax = axes[1, 1]
    pooled_rows = []
    for assay, frame in assay_frames.items():
        tmp = frame[["exp_z", "sim_z"]].copy()
        tmp["assay"] = assay
        tmp["condition_code"] = frame["condition_code"]
        pooled_rows.append(tmp)
    pooled = pd.concat(pooled_rows, ignore_index=True)
    limit = max(1.1, float(np.max(np.abs(pooled[["exp_z", "sim_z"]].to_numpy(dtype=float)))) * 1.15)
    pooled_ax.plot([-limit, limit], [-limit, limit], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)))
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
    pooled_ax.set_title("D  Pooled standardized comparison", loc="left", pad=8)
    pooled_ax.set_xlabel("Experimental standardized response")
    pooled_ax.set_ylabel("Simulation standardized response")
    pooled_ax.set_xlim(-limit, limit)
    pooled_ax.set_ylim(-limit, limit)
    pooled_ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.suptitle("Supplementary calibration and pooled comparison views", fontsize=14.0, fontweight="semibold", y=0.985)
    fig.subplots_adjust(left=0.10, right=0.985, bottom=0.08, top=0.92, wspace=0.32, hspace=0.36)
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
    if args_global.supplement_figure:
        export_supplement_figure(assay_frames, metrics_df, out_dir)

    write_captions(out_dir)
    print(f"Saved outputs to: {out_dir}")


args_global = parse_args()


if __name__ == "__main__":
    main()
