#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import kendalltau, pearsonr, spearmanr

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
INPUT_DEFAULT = SCRIPT_DIR / "out" / "sanity_check5"
BOOTSTRAP_SAMPLES = 5000
BOOTSTRAP_SEED = 12345
REFERENCE_PREFERENCE = ("C2", "C1")
RELATIVE_EPS = 1e-9

JOURNAL_AXIS_LABEL_PT = 11
JOURNAL_TICK_PT = 10
JOURNAL_LEGEND_PT = 10
JOURNAL_LINE_PT = 1.2
JOURNAL_AXIS_LINE_PT = 1.0
JOURNAL_MARKER_PT = 6.2
JOURNAL_PANEL_LABEL_PT = 14

CONDITION_ORDER = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
CONDITION_LABELS = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo + D_LK_L",
    "C4": "Osteo + D_LK_H",
    "C5": "Osteo + D_HK_L",
    "C6": "Osteo + D_HK_H",
    "C7": "Osteo + D_HK_0",
    "C8": "Osteo + D_0K_H",
}

CONDITION_TICK_LABELS = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo\n$D_LK_L$",
    "C4": "Osteo\n$D_LK_H$",
    "C5": "Osteo\n$D_HK_L$",
    "C6": "Osteo\n$D_HK_H$",
    "C7": "Osteo\n$D_HK_0$",
    "C8": "Osteo\n$D_0K_H$",
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
    "exp_line": "#111111",
    "bg": "#ffffff",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": JOURNAL_AXIS_LINE_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14.0,
        "axes.titleweight": "semibold",
        "axes.labelsize": JOURNAL_AXIS_LABEL_PT,
        "xtick.labelsize": JOURNAL_TICK_PT,
        "ytick.labelsize": JOURNAL_TICK_PT,
        "legend.fontsize": JOURNAL_LEGEND_PT,
        "lines.linewidth": JOURNAL_LINE_PT,
        "lines.markersize": JOURNAL_MARKER_PT,
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
    ax.tick_params(direction="out", length=4.5, width=JOURNAL_AXIS_LINE_PT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=JOURNAL_LINE_PT, alpha=grid_alpha)


def condition_tick_label(code: str) -> str:
    return CONDITION_TICK_LABELS.get(str(code), str(code))


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


def choose_reference_condition(df: pd.DataFrame) -> str:
    codes = set(df["condition_code"].astype(str))
    for code in REFERENCE_PREFERENCE:
        if code in codes:
            return code
    return str(df["condition_code"].iloc[0])


def relative_effect(
    values: pd.Series, ref_value: float, eps: float = RELATIVE_EPS
) -> pd.Series:
    vals = values.astype(float)
    denom = vals.abs() + abs(float(ref_value)) + eps
    return (vals - float(ref_value)) / denom


def relative_effect_sd(
    values: pd.Series, sds: pd.Series, ref_value: float, ref_sd: float, eps: float = RELATIVE_EPS
) -> pd.Series:
    vals = values.astype(float).to_numpy()
    sd_vals = sds.astype(float).fillna(0.0).to_numpy()
    ref = float(ref_value)
    ref_sd = float(ref_sd)
    denom = np.abs(vals) + abs(ref) + eps
    deriv_x = (denom - (vals - ref) * np.sign(vals)) / (denom ** 2)
    deriv_r = (-denom - (vals - ref) * np.sign(ref if ref != 0 else 1.0)) / (denom ** 2)
    propagated = np.sqrt((deriv_x * sd_vals) ** 2 + (deriv_r * ref_sd) ** 2)
    return pd.Series(propagated, index=values.index)


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
    nonref = df.loc[~df["is_reference"]].copy()
    sim_rel = nonref["sim_rel_effect"].to_numpy(dtype=float)
    exp_rel = nonref["exp_rel_effect"].to_numpy(dtype=float)
    direction_match = (
        np.sign(sim_rel) == np.sign(exp_rel)
        if len(nonref) > 0
        else np.array([], dtype=bool)
    )
    kendall_val = float(kendalltau(sim_rel, exp_rel).statistic) if len(nonref) > 1 else float("nan")
    spearman_val = float(spearmanr(sim_rel, exp_rel).statistic) if len(nonref) > 1 else float("nan")
    pearson_val = float(pearsonr(sim_rel, exp_rel).statistic) if len(nonref) > 1 else float("nan")
    rmse_val = float(np.sqrt(np.mean((sim_rel - exp_rel) ** 2))) if len(nonref) else float("nan")
    mae_val = float(np.mean(np.abs(sim_rel - exp_rel))) if len(nonref) else float("nan")
    spearman_ci_low, spearman_ci_high = bootstrap_corr_ci(
        sim_rel, exp_rel, "spearman", n_boot=n_boot, seed=seed
    )
    pearson_ci_low, pearson_ci_high = bootstrap_corr_ci(
        sim_rel, exp_rel, "pearson", n_boot=n_boot, seed=seed + 1
    )
    rank_sim = nonref.sort_values("sim_rel_effect", ascending=False)["condition_code"].tolist()
    rank_exp = nonref.sort_values("exp_rel_effect", ascending=False)["condition_code"].tolist()
    top_condition_match = int(bool(rank_sim and rank_exp and rank_sim[0] == rank_exp[0]))
    top3_recovery = len(set(rank_sim[:3]).intersection(rank_exp[:3]))
    return {
        "reference_condition": str(df.loc[df["is_reference"], "condition_code"].iloc[0]),
        "direction_agreement": (
            float(np.mean(direction_match)) if len(direction_match) else float("nan")
        ),
        "direction_agreement_n": int(len(direction_match)),
        "kendall_tau": kendall_val,
        "spearman_rho": spearman_val,
        "spearman_ci_low": spearman_ci_low,
        "spearman_ci_high": spearman_ci_high,
        "pearson_r": pearson_val,
        "pearson_ci_low": pearson_ci_low,
        "pearson_ci_high": pearson_ci_high,
        "rmse_rel_effect": rmse_val,
        "mae_rel_effect": mae_val,
        "top_condition_match": top_condition_match,
        "top3_recovery": int(top3_recovery),
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
    reference_condition = choose_reference_condition(merged)
    merged["is_reference"] = merged["condition_code"].eq(reference_condition)
    ref_row = merged.loc[merged["is_reference"]].iloc[0]
    merged["sim_z"] = zscore(merged["sim_mean"])
    merged["exp_z"] = zscore(merged["exp_mean"])
    merged["sim_rel_effect"] = relative_effect(merged["sim_mean"], float(ref_row["sim_mean"]))
    merged["exp_rel_effect"] = relative_effect(merged["exp_mean"], float(ref_row["exp_mean"]))
    merged["sim_rel_effect_sd"] = relative_effect_sd(
        merged["sim_mean"],
        merged["sim_sd"],
        float(ref_row["sim_mean"]),
        float(ref_row["sim_sd"]),
    )
    merged["exp_rel_effect_sd"] = relative_effect_sd(
        merged["exp_mean"],
        merged["exp_sd"],
        float(ref_row["exp_mean"]),
        float(ref_row["exp_sd"]),
    )
    sim_scale = float(merged["sim_mean"].astype(float).std(ddof=0))
    exp_scale = float(merged["exp_mean"].astype(float).std(ddof=0))
    merged["sim_sd_z"] = (
        merged["sim_sd"].astype(float) / sim_scale if sim_scale > 1e-12 else 0.0
    )
    merged["exp_sd_z"] = (
        merged["exp_sd"].astype(float) / exp_scale if exp_scale > 1e-12 else 0.0
    )
    metrics = compute_metrics(merged, n_boot=bootstrap_samples, seed=seed_offset)
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
                "sim_rel_effect",
                "sim_rel_effect_sd",
                "exp_rel_effect",
                "exp_rel_effect_sd",
                "sim_z",
                "exp_z",
                "is_reference",
            ]
        ].copy()
        tidy.insert(0, "assay", assay)
        tidy = tidy.rename(columns={"condition_code": "condition"})
        tidy_rows.append(tidy)
    tidy_df = pd.concat(tidy_rows, ignore_index=True)
    tidy_df.to_csv(out_dir / "condition_comparison_tidy.csv", index=False)
    metrics_df.to_csv(out_dir / "assay_metrics.csv", index=False)

    spearman_parts = []
    for assay in ("alp", "collagen", "ars"):
        row = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spearman_parts.append(
            f"{ASSAY_SPECS[assay]['title']} {float(row['spearman_rho']):.2f}"
        )
    summary = (
        "Condition-wise assay agreement was summarized by comparing the eight "
        "experimental condition means (C1-C8) against the corresponding eight "
        "simulated condition means and computing Spearman rank correlation across "
        "conditions. For the current sanity5 output, Spearman rho was "
        f"{spearman_parts[0]}, {spearman_parts[1]}, and {spearman_parts[2]} "
        "(n = 8 conditions for each assay). These values are most appropriately "
        "interpreted as descriptive trend-agreement metrics across conditions.\n"
    )
    (out_dir / "assay_spearman_validation.txt").write_text(summary, encoding="utf-8")


def main_panel_metric_text(metrics: pd.Series) -> str:
    return (
        f"Direction agreement = {metrics['direction_agreement']:.0%}\n"
        f"Kendall τ = {metrics['kendall_tau']:.2f}\n"
        f"Top-3 recovery = {int(metrics['top3_recovery'])}/3"
    )


def supplement_metric_text(metrics: pd.Series) -> str:
    return (
        f"Spearman ρ = {metrics['spearman_rho']:.2f}\n"
        f"Pearson r = {metrics['pearson_r']:.2f}\n"
        f"RMSE(rel) = {metrics['rmse_rel_effect']:.2g}\n"
        f"MAE(rel) = {metrics['mae_rel_effect']:.2g}"
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
    exp_vals = frame["exp_rel_effect"].to_numpy(dtype=float)
    exp_sd = frame["exp_rel_effect_sd"].to_numpy(dtype=float)
    sim_vals = frame["sim_rel_effect"].to_numpy(dtype=float)
    sim_sd = frame["sim_rel_effect_sd"].to_numpy(dtype=float)

    ax.errorbar(
        x,
        exp_vals,
        yerr=exp_sd,
        color=COLORS["exp_line"],
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
        0.04,
        main_panel_metric_text(metrics),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.8,
        color=COLORS["ink"],
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.82, "pad": 1.0},
    )
    tick_labels = [condition_tick_label(c) for c in frame["condition_code"]]
    ax.set_xticks(x, tick_labels)
    ax.set_xlim(-0.5, len(frame) - 0.5)
    ax.set_ylabel("Reference-centered effect: (x - x_ref) / (x + x_ref)")
    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
    ymin = float(min(np.min(exp_vals - exp_sd), np.min(sim_vals - sim_sd), -0.6))
    ymax = float(max(np.max(exp_vals + exp_sd), np.max(sim_vals + sim_sd), 0.6))
    pad = max(0.06, 0.10 * (ymax - ymin + 1e-12))
    ax.set_ylim(max(-1.0, ymin - pad), min(1.0, ymax + pad))
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
        color=COLORS["exp_line"],
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
        0.05,
        main_panel_metric_text(metrics),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.0,
        color=COLORS["ink"],
    )
    ax.set_xlabel("Experimental standardized response")
    ax.set_ylabel("Simulation standardized response")
    ax.set_xlim(-limit, limit)
    ax.set_ylim(-limit, limit)


def write_captions(out_dir: Path) -> None:
    main_caption = (
        "Main figure. Per-assay validation is shown as condition-wise relative effect profiles, expressed as a bounded reference-centered effect size "
        "(x - x_ref) / (x + x_ref), using C2 when available and otherwise C1 as the reference condition. Simulation and experiment are compared on the same "
        "relative scale across conditions C1-C8. "
        "Primary validation focuses on recovery of condition-wise relative trends across media conditions, quantified by direction-of-effect agreement, Kendall tau, "
        "and top-condition recovery. Direction-of-effect agreement is defined as the fraction of non-reference conditions for which simulation and experiment show the "
        "same sign relative to the reference condition (increase or decrease versus reference). The reference condition itself is omitted from agreement scoring. "
        "Simulation error bars reflect stochastic replicate variability only and do not represent full predictive uncertainty. "
        "Correlation metrics are reported as secondary descriptive summaries."
    )
    supplement_caption = (
        "Supplementary figure. Raw-value calibration views for each assay and a pooled standardized comparison across assay-condition points. "
        "These views are descriptive only. Raw-unit calibration metrics are not the primary basis for model validation because experimental and simulated units differ; "
        "the main validation emphasizes relative responses versus the reference condition."
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
    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.6), sharey=True, constrained_layout=True)
    for idx, assay in enumerate(("alp", "collagen", "ars")):
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        plot_profile_panel(axes[idx], frame, metrics, assay, chr(65 + idx))
        axes[idx].set_xlabel("")
    legend_handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["exp_line"],
            marker=ASSAY_SPECS["alp"]["marker"],
            linestyle="-",
            linewidth=1.8,
            markersize=5.5,
            markeredgecolor="white",
            label="Experiment",
        ),
        Line2D(
            [0],
            [0],
            color=COLORS["sim_line"],
            marker="D",
            linestyle=(0, (4, 2)),
            linewidth=1.4,
            markersize=5.0,
            markerfacecolor="white",
            markeredgecolor=COLORS["sim_line"],
            label="Simulation",
        ),
    ]
    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.97), frameon=False, fontsize=10.5, ncol=2)
    fig.suptitle(
        "Condition-Wise Relative Effect Profiles vs Reference",
        fontsize=14.2,
        fontweight="semibold",
        y=1.03,
    )
    save_figure(fig, out_dir, "sanity5_predictive_comparison_main")


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
            frame["exp_rel_effect"].to_numpy(dtype=float),
            color=COLORS["exp_line"],
            lw=1.5,
            alpha=0.55,
            ls=(0, (4, 2)),
            zorder=2,
        )
        ax.plot(
            x_sim,
            frame["sim_rel_effect"].to_numpy(dtype=float),
            color=sim_color,
            lw=1.8,
            alpha=0.9,
            ls="-",
            zorder=2,
        )
        ax.errorbar(
            x_exp,
            frame["exp_rel_effect"].to_numpy(dtype=float),
            yerr=frame["exp_rel_effect_sd"].to_numpy(dtype=float),
            fmt=spec["marker"],
            color=COLORS["exp_line"],
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
            frame["sim_rel_effect"].to_numpy(dtype=float),
            yerr=frame["sim_rel_effect_sd"].to_numpy(dtype=float),
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

        ax.set_title(spec["title"], pad=8, fontsize=13.2)
        ax.text(
            0.98,
            0.05,
            main_panel_metric_text(metrics),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10.1,
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.0},
        )
        tick_labels = [condition_tick_label(c) for c in frame["condition_code"]]
        ax.set_xticks(x, tick_labels)
        ax.set_xlim(-0.5, len(frame) - 0.5)
        if idx == 0:
            ax.set_ylabel("Reference-centered effect: (x - x_ref) / (x + x_ref)")
        ax.set_xlabel("")

    handles = [
        Line2D(
            [0],
            [0],
            color=COLORS["exp_line"],
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
        fontsize=10.5,
    )
    fig.suptitle(
        "Condition-Wise Relative Effect Comparison",
        fontsize=14.2,
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

    fig.suptitle("Supplementary Raw Calibration Views", fontsize=13.5, fontweight="semibold", y=0.985)
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
