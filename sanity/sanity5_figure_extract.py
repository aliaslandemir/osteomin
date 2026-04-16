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
REFERENCE_CONDITION = "C2"

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
        "color": "#1f77b4",
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
        "color": "#1f77b4",
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
        "color": "#1f77b4",
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
        ax.grid(
            axis=grid_axis,
            color=COLORS["grid"],
            linewidth=JOURNAL_LINE_PT,
            alpha=grid_alpha,
        )


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
    if REFERENCE_CONDITION not in codes:
        raise SystemExit(
            f"Required reference condition '{REFERENCE_CONDITION}' is missing from merged assay table."
        )
    return REFERENCE_CONDITION


def relative_effect(values: pd.Series, ref_value: float) -> pd.Series:
    vals = values.astype(float)
    denom = vals.abs() + abs(float(ref_value))
    out = pd.Series(np.zeros(len(vals), dtype=float), index=values.index)
    valid = denom > 0.0
    out.loc[valid] = (vals.loc[valid] - float(ref_value)) / denom.loc[valid]
    return out


def relative_effect_sd(
    values: pd.Series,
    sds: pd.Series,
    ref_value: float,
    ref_sd: float,
) -> pd.Series:
    vals = values.astype(float).to_numpy()
    sd_vals = sds.astype(float).fillna(0.0).to_numpy()
    ref = float(ref_value)
    ref_sd = float(ref_sd)
    denom = np.abs(vals) + abs(ref)
    deriv_x = np.zeros_like(vals, dtype=float)
    deriv_r = np.zeros_like(vals, dtype=float)
    valid = denom > 0.0
    deriv_x[valid] = (
        denom[valid] - (vals[valid] - ref) * np.sign(vals[valid])
    ) / (denom[valid] ** 2)
    deriv_r[valid] = (
        -denom[valid] - (vals[valid] - ref) * np.sign(ref if ref != 0 else 1.0)
    ) / (denom[valid] ** 2)
    propagated = np.zeros_like(vals, dtype=float)
    propagated[valid] = np.sqrt(
        (deriv_x[valid] * sd_vals[valid]) ** 2 + (deriv_r[valid] * ref_sd) ** 2
    )
    return pd.Series(propagated, index=values.index)


def compute_metrics(df: pd.DataFrame) -> dict[str, float | int]:
    nonref = df.loc[~df["is_reference"]].copy()
    sim_rel = nonref["sim_rel_effect"].to_numpy(dtype=float)
    exp_rel = nonref["exp_rel_effect"].to_numpy(dtype=float)
    direction_match = (
        np.sign(sim_rel) == np.sign(exp_rel)
        if len(nonref) > 0
        else np.array([], dtype=bool)
    )
    kendall_val = (
        float(kendalltau(sim_rel, exp_rel).statistic)
        if len(nonref) > 1
        else float("nan")
    )
    spearman_val = (
        float(spearmanr(sim_rel, exp_rel).statistic)
        if len(nonref) > 1
        else float("nan")
    )
    pearson_val = (
        float(pearsonr(sim_rel, exp_rel).statistic) if len(nonref) > 1 else float("nan")
    )
    rmse_val = (
        float(np.sqrt(np.mean((sim_rel - exp_rel) ** 2)))
        if len(nonref)
        else float("nan")
    )
    mae_val = float(np.mean(np.abs(sim_rel - exp_rel))) if len(nonref) else float("nan")
    rank_sim = nonref.sort_values("sim_rel_effect", ascending=False)[
        "condition_code"
    ].tolist()
    rank_exp = nonref.sort_values("exp_rel_effect", ascending=False)[
        "condition_code"
    ].tolist()
    top_condition_match = int(
        bool(rank_sim and rank_exp and rank_sim[0] == rank_exp[0])
    )
    top3_recovery = len(set(rank_sim[:3]).intersection(rank_exp[:3]))
    direction_sd = (
        float(direction_match.astype(float).std(ddof=0))
        if len(direction_match)
        else float("nan")
    )
    if len(nonref) > 1:
        rng = np.random.default_rng(12345)
        n = len(nonref)
        boot_vals: list[float] = []
        for _ in range(512):
            idx = rng.integers(0, n, size=n)
            tau = kendalltau(sim_rel[idx], exp_rel[idx]).statistic
            if np.isfinite(float(tau)):
                boot_vals.append(float(tau))
        kendall_sd = (
            float(np.std(np.asarray(boot_vals, dtype=float), ddof=0))
            if boot_vals
            else float("nan")
        )
    else:
        kendall_sd = float("nan")
    return {
        "reference_condition": str(
            df.loc[df["is_reference"], "condition_code"].iloc[0]
        ),
        "direction_agreement": (
            float(np.mean(direction_match)) if len(direction_match) else float("nan")
        ),
        "direction_agreement_sd": direction_sd,
        "direction_agreement_n": int(len(direction_match)),
        "kendall_tau": kendall_val,
        "kendall_tau_sd": kendall_sd,
        "spearman_rho": spearman_val,
        "pearson_r": pearson_val,
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
    merged["sim_rel_effect"] = relative_effect(
        merged["sim_mean"], float(ref_row["sim_mean"])
    )
    merged["exp_rel_effect"] = relative_effect(
        merged["exp_mean"], float(ref_row["exp_mean"])
    )
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
    metrics = compute_metrics(merged)
    return merged, metrics


def build_all_assays(input_dir: Path) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    assay_frames: dict[str, pd.DataFrame] = {}
    metric_rows: list[dict[str, float | int | str]] = []
    for assay in ("alp", "collagen", "ars"):
        frame, metrics = build_assay_frame(input_dir, assay)
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
    write_relative_effect_condition_text(assay_frames, metrics_df, out_dir)


def _relative_state_label(value: float, *, tol: float = 1e-9) -> str:
    if value > tol:
        return "higher than reference"
    if value < -tol:
        return "lower than reference"
    return "approximately equal to reference"


def write_relative_effect_condition_text(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    lines: list[str] = []
    lines.append(
        "Relative effect condition-by-condition comparison (simulation vs experiment)"
    )
    lines.append(
        "Reference-centered effect is defined as (x - x_ref) / (|x| + |x_ref|), with x_ref fixed to C2."
    )
    lines.append("")

    for assay in ("alp", "collagen", "ars"):
        frame = assay_frames[assay].copy()
        frame["condition"] = pd.Categorical(
            frame["condition_code"], categories=CONDITION_ORDER, ordered=True
        )
        frame = frame.sort_values("condition").reset_index(drop=True)
        mrow = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        ref_code = str(mrow["reference_condition"])
        lines.append(
            f"{ASSAY_SPECS[assay]['title']} (reference={ref_code}, directionality={float(mrow['direction_agreement']):.0%})"
        )
        for _, row in frame.iterrows():
            code = str(row["condition_code"])
            exp_rel = float(row["exp_rel_effect"])
            sim_rel = float(row["sim_rel_effect"])
            exp_state = _relative_state_label(exp_rel)
            sim_state = _relative_state_label(sim_rel)
            if bool(row["is_reference"]):
                lines.append(
                    f"  {code}: reference condition; exp_rel={exp_rel:+.3f}, sim_rel={sim_rel:+.3f}."
                )
                continue
            direction_match = np.sign(exp_rel) == np.sign(sim_rel)
            lines.append(
                f"  {code}: experiment {exp_state} (exp_rel={exp_rel:+.3f}); "
                f"simulation {sim_state} (sim_rel={sim_rel:+.3f}); "
                f"direction_match={'yes' if direction_match else 'no'}."
            )
        lines.append("")

    (out_dir / "relative_effect_condition_comparison.txt").write_text(
        "\n".join(lines).rstrip() + "\n",
        encoding="utf-8",
    )


def main_panel_metric_text(metrics: pd.Series) -> str:
    return (
        f"Sign agreement = {metrics['direction_agreement']:.0%}\n"
        f"Kendall τ = {metrics['kendall_tau']:.2f}\n"
        f"Top-3 recovery = {int(metrics['top3_recovery'])}/3"
    )


def supplement_metric_text(metrics: pd.Series) -> str:
    return (
        f"RMSE(rel) = {metrics['rmse_rel_effect']:.2g}\n"
        f"MAE = {metrics['mae_rel_effect']:.2f}"
    )


def scatter_label_offsets(
    points: np.ndarray, axis_limit: float | None = None
) -> list[tuple[float, float]]:
    if points.size == 0:
        return []
    x_span = max(float(np.ptp(points[:, 0])), 1e-6)
    y_span = max(float(np.ptp(points[:, 1])), 1e-6)
    scale = max(x_span, y_span, float(axis_limit) if axis_limit else 0.65, 0.65)

    r1 = 0.050 * scale
    r2 = 0.072 * scale
    angles = np.deg2rad(np.arange(0, 360, 30, dtype=float))
    candidates = [
        (r * float(np.cos(a)), r * float(np.sin(a))) for r in (r1, r2) for a in angles
    ]

    n = len(points)
    pairwise = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    np.fill_diagonal(pairwise, np.inf)
    crowd_order = np.argsort(np.min(pairwise, axis=1))

    chosen = np.zeros((n, 2), dtype=float)
    placed_xy: list[tuple[float, float]] = []
    lim = float(axis_limit) if axis_limit else None
    edge_margin = 0.03 * scale

    for idx in crowd_order:
        x, y = float(points[idx, 0]), float(points[idx, 1])
        best_dx, best_dy = candidates[0]
        best_score = -np.inf
        for dx, dy in candidates:
            lx = x + dx
            ly = y + dy
            score = 0.0
            if placed_xy:
                min_label_dist = min(np.hypot(lx - px, ly - py) for px, py in placed_xy)
                score += 2.4 * min_label_dist
            # keep labels away from nearby points (other than self)
            point_dist = np.hypot(lx - points[:, 0], ly - points[:, 1])
            point_dist[idx] = np.inf
            score += 1.8 * float(np.min(point_dist))
            # avoid excessive offset
            score -= 0.25 * np.hypot(dx, dy)
            if lim is not None:
                if lx > (lim - edge_margin):
                    score -= 8.0 * (lx - (lim - edge_margin))
                if lx < (-lim + edge_margin):
                    score -= 8.0 * ((-lim + edge_margin) - lx)
                if ly > (lim - edge_margin):
                    score -= 8.0 * (ly - (lim - edge_margin))
                if ly < (-lim + edge_margin):
                    score -= 8.0 * ((-lim + edge_margin) - ly)
            if score > best_score:
                best_score = score
                best_dx, best_dy = dx, dy
        chosen[idx, :] = (best_dx, best_dy)
        placed_xy.append((x + best_dx, y + best_dy))
    return [(float(dx), float(dy)) for dx, dy in chosen]


def parity_metric_text(metrics: pd.Series) -> str:
    directionality = metrics.get("direction_agreement", float("nan"))
    kendall_tau = metrics.get("kendall_tau", float("nan"))
    directionality_txt = (
        f"{float(directionality):.0%}" if np.isfinite(float(directionality)) else "NA"
    )
    kendall_txt = f"{float(kendall_tau):.2f}" if np.isfinite(float(kendall_tau)) else "NA"
    return (
        f"Kendall τ = {kendall_txt}\n"
        f"Sign agreement = {directionality_txt}"
    )


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
        ms=5.8,
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
        ms=5.6,
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
    ax.set_ylabel("Reference-centered effect: (x - x_ref) / (|x| + |x_ref|)")
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
        s=116,
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
        "(x - x_ref) / (|x| + |x_ref|), with C2 fixed as the reference condition. Simulation and experiment are compared on the same "
        "relative scale across conditions C1-C8. "
        "Primary validation focuses on recovery of condition-wise relative trends across media conditions, quantified by direction-of-effect agreement, Kendall tau, "
        "and top-condition recovery. Direction-of-effect agreement is defined as the fraction of non-reference conditions for which simulation and experiment show the "
        "same sign relative to the reference condition (increase or decrease versus reference). The reference condition itself is omitted from agreement scoring. "
        "Simulation error bars reflect stochastic replicate variability only and do not represent full predictive uncertainty. "
        "Correlation metrics are reported as secondary descriptive summaries."
    )
    supplement_caption = (
        "Supplementary figure. Per-assay and pooled parity views in the same C2-centered relative-effect space used in the primary analysis. "
        "All assay-condition comparisons are shown as reference-centered effect values, enabling direct direction and ranking checks without unit mismatch."
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
    fig, axes = plt.subplots(
        1, 3, figsize=(12.6, 4.6), sharey=True, constrained_layout=True
    )
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
            markersize=6.3,
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
            markersize=5.8,
            markerfacecolor="white",
            markeredgecolor=COLORS["sim_line"],
            label="Simulation",
        ),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        frameon=False,
        fontsize=10.5,
        ncol=2,
    )
    fig.suptitle(
        "Condition-Wise Relative Effect Profiles vs C2",
        fontsize=14.2,
        fontweight="semibold",
        y=1.03,
    )
    save_figure(fig, out_dir, "sanity5_predictive_comparison_main")


def export_discrete_condition_figure(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(
        1, 3, figsize=(13.6, 4.8), constrained_layout=True, sharey=True
    )
    panel_titles = {
        "alp": "ALP",
        "collagen": "Collagen",
        "ars": "Mineralization",
    }
    # Match endpoint palette used in endpoint figures.
    panel_colors = {
        "alp": "#1f77b4",
        "collagen": "#1f77b4",
        "ars": "#1f77b4",
    }
    global_values = np.concatenate(
        [
            assay_frames["alp"]["exp_rel_effect"].to_numpy(dtype=float),
            assay_frames["alp"]["sim_rel_effect"].to_numpy(dtype=float),
            assay_frames["collagen"]["exp_rel_effect"].to_numpy(dtype=float),
            assay_frames["collagen"]["sim_rel_effect"].to_numpy(dtype=float),
            assay_frames["ars"]["exp_rel_effect"].to_numpy(dtype=float),
            assay_frames["ars"]["sim_rel_effect"].to_numpy(dtype=float),
        ]
    )
    shared_limit = max(0.35, 1.05 * float(np.max(np.abs(global_values))))

    for idx, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes[idx]
        style_axes(ax, None, grid_alpha=0.0)
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spec = ASSAY_SPECS[assay]
        exp_rel = frame["exp_rel_effect"].to_numpy(dtype=float)
        sim_rel = frame["sim_rel_effect"].to_numpy(dtype=float)

        ax.plot(
            [-shared_limit, shared_limit],
            [-shared_limit, shared_limit],
            color="0.65",
            lw=1.0,
            ls="--",
            zorder=1,
        )
        ax.axhline(0.0, lw=0.6, color="0.88", zorder=0)
        ax.axvline(0.0, lw=0.6, color="0.88", zorder=0)
        ax.scatter(
            exp_rel,
            sim_rel,
            s=110,
            color=panel_colors[assay],
            marker=spec["marker"],
            edgecolors="#111111",
            linewidths=0.8,
            zorder=3,
        )
        offsets = scatter_label_offsets(
            np.column_stack([exp_rel, sim_rel]), axis_limit=shared_limit
        )
        for (_, row), (dx, dy) in zip(frame.iterrows(), offsets):
            ax.text(
                float(row["exp_rel_effect"]) + dx,
                float(row["sim_rel_effect"]) + dy,
                str(row["condition_code"]),
                fontsize=14.5,
                color=COLORS["ink"],
                zorder=4,
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.8,
                    "pad": 0.15,
                },
            )

        ax.text(
            -0.16,
            1.06,
            chr(65 + idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=24.0,
            fontweight="bold",
            color=COLORS["ink"],
        )
        ax.text(
            0.02,
            1.03,
            panel_titles[assay],
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=22.0,
            fontweight="semibold",
            color=COLORS["ink"],
        )
        ax.text(
            0.97,
            0.05,
            parity_metric_text(metrics),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=12.0,
            color=COLORS["ink"],
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.9, "pad": 1.0},
        )
        ax.set_xlim(-shared_limit, shared_limit)
        ax.set_ylim(-shared_limit, shared_limit)
        ax.set_aspect("equal", adjustable="box")
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        ax.set_xlabel("Experimental relative effect", fontsize=19.0)
        if idx == 0:
            ax.set_ylabel("Simulated relative effect", fontsize=19.0)
        else:
            ax.set_ylabel("")
            ax.tick_params(axis="y", labelleft=False)
        ax.yaxis.labelpad = 6
        ax.tick_params(axis="both", labelsize=14.5, length=7.0, width=1.2)

    save_figure(fig, out_dir, "sanity5_discrete_condition_comparison")


def export_supplement_figure(
    assay_frames: dict[str, pd.DataFrame], metrics_df: pd.DataFrame, out_dir: Path
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 7.2))
    for ax in axes.ravel():
        style_axes(ax, "both", grid_alpha=0.35)

    pooled_rows = []
    for idx, assay in enumerate(("alp", "collagen", "ars")):
        ax = axes.ravel()[idx]
        frame = assay_frames[assay]
        metrics = metrics_df.loc[metrics_df["assay"] == assay].iloc[0]
        spec = ASSAY_SPECS[assay]
        exp_plot = frame["exp_rel_effect"].to_numpy(dtype=float)
        sim_plot = frame["sim_rel_effect"].to_numpy(dtype=float)
        lo = min(np.min(sim_plot), np.min(exp_plot), -0.1)
        hi = max(np.max(sim_plot), np.max(exp_plot), 0.1)
        pad = max(0.08, 0.12 * (hi - lo + 1e-12))
        lo -= pad
        hi += pad
        ax.scatter(
            exp_plot,
            sim_plot,
            s=90,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.8,
        )
        ax.plot([lo, hi], [lo, hi], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)))
        for _, row in frame.iterrows():
            ex = float(row["exp_rel_effect"])
            sy = float(row["sim_rel_effect"])
            ax.text(ex + 0.01 * (hi - lo + 1e-9), sy, row["condition_code"], fontsize=7.8)
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
        ax.set_xlabel("Experimental relative effect vs C2")
        ax.set_ylabel("Simulated relative effect vs C2")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)

        tmp = frame[["exp_rel_effect", "sim_rel_effect"]].copy()
        tmp["assay"] = assay
        tmp["condition_code"] = frame["condition_code"]
        pooled_rows.append(tmp)

    pooled_ax = axes[1, 1]
    pooled = pd.concat(pooled_rows, ignore_index=True)
    limit = max(
        0.4,
        float(
            np.max(
                np.abs(pooled[["exp_rel_effect", "sim_rel_effect"]].to_numpy(dtype=float))
            )
        )
        * 1.15,
    )
    pooled_ax.plot(
        [-limit, limit], [-limit, limit], color=COLORS["muted"], lw=1.0, ls=(0, (4, 3))
    )
    for assay in ("alp", "collagen", "ars"):
        spec = ASSAY_SPECS[assay]
        sub = pooled[pooled["assay"] == assay]
        pooled_ax.scatter(
            sub["exp_rel_effect"],
            sub["sim_rel_effect"],
            s=84,
            color=spec["color"],
            marker=spec["marker"],
            edgecolors="white",
            linewidths=0.8,
            label=spec["title"],
        )
    pooled_ax.text(
        0.02,
        0.96,
        "Pooled relative-effect points\n(all assays; C2 reference)",
        transform=pooled_ax.transAxes,
        ha="left",
        va="top",
        fontsize=8.8,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.8, "pad": 1.2},
    )
    pooled_ax.set_title(
        "D  Pooled relative-effect comparison", loc="left", pad=8, fontsize=11.5
    )
    pooled_ax.set_xlabel("Experimental relative effect vs C2")
    pooled_ax.set_ylabel("Simulated relative effect vs C2")
    pooled_ax.set_xlim(-limit, limit)
    pooled_ax.set_ylim(-limit, limit)
    pooled_ax.legend(frameon=False, fontsize=8.8, loc="lower right")

    fig.suptitle(
        "Supplementary Relative-Effect Parity Views (Reference = C2)",
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
    assay_frames, metrics_df = build_all_assays(args_global.input)

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
