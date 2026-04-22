#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

JOURNAL_AXIS_LABEL_PT = 9
JOURNAL_TICK_PT = 8
JOURNAL_LEGEND_PT = 8
JOURNAL_PANEL_LABEL_PT = 12
JOURNAL_LINE_PT = 1.2
JOURNAL_AXIS_LINE_PT = 1.0
JOURNAL_MARKER_PT = 5.5

COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "control": "#1d4ed8",
    "collagen": "#059669",
    "ha": "#b45309",
    "alp": "#c2410c",
    "neutral": "#94a3b8",
    "bg": "#f8fafc",
}

PRIMARY_METRICS = [
    "assay_alp",
    "assay_collagen",
    "assay_ha",
    "final_live",
    "final_preob",
    "final_ob",
]

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": JOURNAL_AXIS_LINE_PT,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": JOURNAL_AXIS_LABEL_PT,
        "xtick.labelsize": JOURNAL_TICK_PT,
        "ytick.labelsize": JOURNAL_TICK_PT,
        "legend.fontsize": JOURNAL_LEGEND_PT,
        "lines.linewidth": JOURNAL_LINE_PT,
        "lines.markersize": JOURNAL_MARKER_PT,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Publication-ready omics-style figure extraction for sanity_check2 outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "out" / "sanity_check2",
        help="sanity_check2 output folder containing replicate_summary.csv",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder for extracted figures; defaults to <input>/figure_extract_2",
    )
    parser.add_argument("--alp-hours", type=int, default=None, help="hour used for ALP comparison")
    parser.add_argument("--collagen-hours", type=int, default=None, help="hour used for collagen comparison")
    parser.add_argument("--hap-hours", type=int, default=None, help="hour used for HAp comparison")
    parser.add_argument(
        "--figure3-geometry-mode",
        choices=["classic", "patch_density"],
        default="classic",
        help="geometry panel mode for figure_3; 'patch_density' replaces Radius/Edge with N_patch and n=N_patch/A",
    )
    parser.add_argument(
        "--domain-area-mm2",
        type=float,
        default=1.0,
        help="domain area A used for patch-density calculation n=N_patch/A (clusters per mm^2)",
    )
    return parser.parse_args()


def load_inputs(input_dir: Path) -> tuple[pd.DataFrame, dict]:
    replicate_path = input_dir / "replicate_summary.csv"
    if not replicate_path.exists():
        raise SystemExit(f"replicate_summary.csv not found under {input_dir}")
    replicate_df = pd.read_csv(replicate_path)
    manifest_path = input_dir / "manifest.json"
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return replicate_df, manifest


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=320, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=JOURNAL_AXIS_LINE_PT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=JOURNAL_LINE_PT, alpha=0.6)


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.02,
        label,
        transform=ax.transAxes,
        fontsize=JOURNAL_PANEL_LABEL_PT,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def response_cmap() -> LinearSegmentedColormap:
    return matplotlib.colormaps["coolwarm"]


def hap_field_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "hap_field",
        ["#120c2c", "#4c1d6f", "#9c2963", "#e16462", "#f6c667", "#fff3bf"],
    )


def assay_hours_from_args(args: argparse.Namespace, manifest: dict) -> dict[str, int]:
    return {
        "alp": int(args.alp_hours if args.alp_hours is not None else manifest.get("alp_hours", 240)),
        "collagen": int(args.collagen_hours if args.collagen_hours is not None else manifest.get("collagen_hours", 336)),
        "ha": int(args.hap_hours if args.hap_hours is not None else manifest.get("mineral_hours", 672)),
    }


def assay_label(name: str, hours: int) -> str:
    pretty = {"alp": "ALP", "collagen": "Collagen", "ha": "Mineral"}[name]
    return f"{pretty} at {int(hours)} h"


def row_at_hours(df: pd.DataFrame, hours: int) -> pd.Series:
    target_day = float(hours) / 24.0
    idx = int((df["day"] - target_day).abs().idxmin())
    return df.loc[idx]


def estimate_autocorr_length_xi(field: np.ndarray) -> float:
    arr = np.asarray(field, dtype=float)
    if arr.ndim != 2:
        return float("nan")
    centered = arr - float(np.mean(arr))
    var = float(np.mean(centered * centered))
    if var <= 1e-12:
        return 0.0

    fft = np.fft.fftn(centered)
    power = np.abs(fft) ** 2
    ac = np.fft.ifftn(power).real
    ac /= max(float(ac[0, 0]), 1e-12)

    h, w = centered.shape
    yy = np.minimum(np.arange(h), h - np.arange(h))
    xx = np.minimum(np.arange(w), w - np.arange(w))
    rr = np.sqrt(yy[:, None] ** 2 + xx[None, :] ** 2)
    r_max = int(np.floor(min(h, w) / 2))
    if r_max < 2:
        return float("nan")
    bins = np.arange(r_max + 1, dtype=float)
    radial = np.full(r_max, np.nan, dtype=float)
    for i in range(r_max):
        mask = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(mask):
            radial[i] = float(np.mean(ac[mask]))
    if not np.isfinite(radial[0]):
        radial[0] = 1.0
    target = float(np.exp(-1.0))
    for i in range(1, len(radial)):
        c0 = float(radial[i - 1])
        c1 = float(radial[i])
        if np.isfinite(c0) and np.isfinite(c1) and (c0 >= target) and (c1 <= target):
            denom = max(abs(c0 - c1), 1e-12)
            frac = (c0 - target) / denom
            return float((i - 1) + np.clip(frac, 0.0, 1.0))
    return float("nan")


def _find_grid_for_day(rep_dir: Path, prefix: str, preferred_day: int) -> Path:
    preferred = rep_dir / "grids" / f"{prefix}_day_{preferred_day:03d}.npy"
    if preferred.exists():
        return preferred
    candidates = sorted((rep_dir / "grids").glob(f"{prefix}_day_*.npy"))
    if not candidates:
        raise SystemExit(f"No {prefix} grids found in {rep_dir / 'grids'}")
    return candidates[-1]


def enrich_replicate_summary(replicate_summary: pd.DataFrame, assay_hours: dict[str, int]) -> pd.DataFrame:
    df = replicate_summary.copy()
    assay_values = {"assay_alp": [], "assay_collagen": [], "assay_ha": [], "spatial_autocorr_xi": []}
    for _, row in df.iterrows():
        out_dir = Path(str(row["out_dir"]))
        ts_path = out_dir / "timeseries.csv"
        if not ts_path.exists():
            raise SystemExit(f"timeseries.csv not found for replicate output: {ts_path}")
        ts = pd.read_csv(ts_path)
        assay_values["assay_alp"].append(float(row_at_hours(ts, assay_hours["alp"])["alp_total"]))
        assay_values["assay_collagen"].append(float(row_at_hours(ts, assay_hours["collagen"])["collagen_total"]))
        assay_values["assay_ha"].append(float(row_at_hours(ts, assay_hours["ha"])["ha_total"]))
        target_day = int(round(float(assay_hours["ha"]) / 24.0))
        ha_grid_path = _find_grid_for_day(out_dir, "ha", target_day)
        ha_grid = np.load(ha_grid_path).astype(float)
        assay_values["spatial_autocorr_xi"].append(float(estimate_autocorr_length_xi(ha_grid)))
    for key, values in assay_values.items():
        df[key] = values
    return df


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3g}"


def significance_symbol(p_adj: float) -> str:
    if pd.isna(p_adj):
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def build_condition_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition_index", "slug", "title", "description"]
    numeric_cols = [
        "seed_centroid_x",
        "seed_centroid_y",
        "seed_std_x",
        "seed_std_y",
        "seed_mean_radius",
        "seed_edge_fraction",
        "seed_center_fraction",
        "seed_left_fraction",
        "seed_top_fraction",
        "seed_mean_nearest_neighbor",
        "assay_alp",
        "assay_collagen",
        "assay_ha",
        "spatial_autocorr_xi",
        "final_alp",
        "final_collagen",
        "final_ha",
        "final_preob",
        "final_ob",
        "final_live",
        "final_msd",
    ]
    grouped = replicate_summary.groupby(group_cols, sort=True)
    mean_df = grouped[numeric_cols].mean().reset_index()
    std_df = grouped[numeric_cols].std(ddof=1).reset_index().fillna(0.0)
    n_df = grouped.size().reset_index(name="n_replicates")
    first_out = grouped["out_dir"].first().reset_index(name="representative_out_dir")
    summary = mean_df.merge(std_df, on=group_cols, suffixes=("_mean", "_sd"))
    summary = summary.merge(n_df, on=group_cols)
    summary = summary.merge(first_out, on=group_cols)
    return summary.sort_values("condition_index").reset_index(drop=True)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def metric_label(metric: str, assay_hours: dict[str, int]) -> str:
    labels = {
        "assay_alp": assay_label("alp", assay_hours["alp"]),
        "assay_collagen": assay_label("collagen", assay_hours["collagen"]),
        "assay_ha": assay_label("ha", assay_hours["ha"]),
        "final_live": "Final live cells",
        "final_preob": "Final preOB",
        "final_ob": "Final OB",
    }
    return labels.get(metric, metric)


def compute_statistics(replicate_summary: pd.DataFrame, assay_hours: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    control_idx = int(replicate_summary["condition_index"].min())
    for metric in PRIMARY_METRICS:
        groups = [
            grp[metric].astype(float).to_numpy()
            for _, grp in replicate_summary.groupby("condition_index", sort=True)
        ]
        if len(groups) >= 2:
            try:
                h_stat, p_global = stats.kruskal(*groups)
            except ValueError:
                h_stat, p_global = 0.0, 1.0
        else:
            h_stat, p_global = np.nan, np.nan

        pairwise_rows = []
        control_vals = replicate_summary.loc[replicate_summary["condition_index"] == control_idx, metric].astype(float).to_numpy()
        for condition_index, grp in replicate_summary.groupby("condition_index", sort=True):
            vals = grp[metric].astype(float).to_numpy()
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if int(condition_index) == control_idx:
                pairwise_rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label(metric, assay_hours),
                        "condition_index": int(condition_index),
                        "slug": str(grp["slug"].iloc[0]),
                        "title": str(grp["title"].iloc[0]),
                        "n": int(len(vals)),
                        "mean": mean_val,
                        "sd": sd_val,
                        "delta_vs_control": 0.0,
                        "pct_vs_control": 0.0,
                        "u_stat": np.nan,
                        "p_value": np.nan,
                        "p_adj": np.nan,
                        "significant_vs_control": False,
                        "global_h": float(h_stat) if not pd.isna(h_stat) else np.nan,
                        "global_p": float(p_global) if not pd.isna(p_global) else np.nan,
                    }
                )
                continue
            try:
                u_stat, p_value = stats.mannwhitneyu(vals, control_vals, alternative="two-sided")
            except ValueError:
                u_stat, p_value = np.nan, 1.0
            pairwise_rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label(metric, assay_hours),
                    "condition_index": int(condition_index),
                    "slug": str(grp["slug"].iloc[0]),
                    "title": str(grp["title"].iloc[0]),
                    "n": int(len(vals)),
                    "mean": mean_val,
                    "sd": sd_val,
                    "delta_vs_control": mean_val - float(np.mean(control_vals)),
                    "pct_vs_control": 100.0 * (mean_val - float(np.mean(control_vals))) / max(float(np.mean(control_vals)), 1e-12),
                    "u_stat": float(u_stat) if not pd.isna(u_stat) else np.nan,
                    "p_value": float(p_value),
                    "p_adj": np.nan,
                    "significant_vs_control": False,
                    "global_h": float(h_stat) if not pd.isna(h_stat) else np.nan,
                    "global_p": float(p_global) if not pd.isna(p_global) else np.nan,
                }
            )
        pvals = [float(row["p_value"]) for row in pairwise_rows if not pd.isna(row["p_value"])]
        adj = holm_adjust(pvals) if pvals else []
        adj_iter = iter(adj)
        for row in pairwise_rows:
            if pd.isna(row["p_value"]):
                row["p_adj"] = np.nan
            else:
                row["p_adj"] = float(next(adj_iter))
                row["significant_vs_control"] = bool(row["p_adj"] < 0.05)
        rows.extend(pairwise_rows)
    return pd.DataFrame(rows).sort_values(["metric", "condition_index"]).reset_index(drop=True)


def prep_summary(summary: pd.DataFrame, domain_area_mm2: float) -> pd.DataFrame:
    df = summary.copy().sort_values("condition_index").reset_index(drop=True)
    base = df.iloc[0]
    for mean_src, sd_src, pct_dest, sd_dest in [
        ("assay_collagen_mean", "assay_collagen_sd", "assay_collagen_pct", "assay_collagen_pct_sd"),
        ("final_collagen_mean", "final_collagen_sd", "final_collagen_pct", "final_collagen_pct_sd"),
        ("assay_ha_mean", "assay_ha_sd", "assay_ha_pct", "assay_ha_pct_sd"),
        ("final_ha_mean", "final_ha_sd", "final_ha_pct", "final_ha_pct_sd"),
        ("assay_alp_mean", "assay_alp_sd", "assay_alp_pct", "assay_alp_pct_sd"),
        ("final_live_mean", "final_live_sd", "final_live_pct", "final_live_pct_sd"),
    ]:
        denom = max(float(base[mean_src]), 1e-12)
        df[pct_dest] = 100.0 * (df[mean_src] - float(base[mean_src])) / denom
        df[sd_dest] = 100.0 * df[sd_src].astype(float) / denom
    xi_denom = max(float(base["spatial_autocorr_xi_mean"]), 1e-12)
    df["spatial_autocorr_xi_pct"] = 100.0 * (df["spatial_autocorr_xi_mean"].astype(float) - float(base["spatial_autocorr_xi_mean"])) / xi_denom
    df["spatial_autocorr_xi_pct_sd"] = 100.0 * df["spatial_autocorr_xi_sd"].astype(float) / xi_denom
    area = float(domain_area_mm2)
    if area <= 0.0:
        raise SystemExit(f"--domain-area-mm2 must be > 0 (got {area})")
    df["domain_area_mm2"] = area
    df["condition_label"] = [f"C{int(i):02d}" for i in df["condition_index"]]
    df["row_label"] = df["condition_label"] + "  " + df["slug"]
    return df


def write_statistics_tables(stats_df: pd.DataFrame, out_dir: Path) -> None:
    stats_df.to_csv(out_dir / "statistics.csv", index=False)
    lines = [
        "# Sanity Check 2 Statistics",
        "",
        "| Metric | Condition | Mean +/- SD | Delta vs control | Percent vs control | U | p | p_adj | Significant |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in stats_df.iterrows():
        u_text = _fmt(float(row["u_stat"])) if not pd.isna(row["u_stat"]) else "NA"
        p_text = _fmt_p(float(row["p_value"])) if not pd.isna(row["p_value"]) else "NA"
        p_adj_text = _fmt_p(float(row["p_adj"])) if not pd.isna(row["p_adj"]) else "NA"
        mean_pm_sd = f"{_fmt(float(row['mean']))} +/- {_fmt(float(row['sd']))}"
        lines.append(
            f"| {row['metric_label']} | C{int(row['condition_index']):02d} `{row['slug']}` | {mean_pm_sd} | {_fmt(float(row['delta_vs_control']))} | {float(row['pct_vs_control']):+.1f}% | {u_text} | {p_text} | {p_adj_text} | {'yes' if bool(row['significant_vs_control']) else 'no'} |"
        )
    (out_dir / "statistics.md").write_text("\n".join(lines), encoding="utf-8")


def write_report(
    condition_summary: pd.DataFrame,
    replicate_summary: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    assay_hours: dict[str, int],
) -> None:
    base = condition_summary.iloc[0]
    lines = [
        "# Sanity Check 2 Report",
        "",
        f"This extraction stage evaluates {len(condition_summary)} initialization modes with {int(replicate_summary['replicate_index'].max())} replicates each.",
        "",
        "## Endpoint table",
        "",
        f"| Condition | {assay_label('alp', assay_hours['alp'])} | {assay_label('collagen', assay_hours['collagen'])} | {assay_label('ha', assay_hours['ha'])} | Final live | Autocorrelation (xi) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in condition_summary.iterrows():
        xi_mean = float(row["spatial_autocorr_xi_mean"])
        xi_sd = float(row["spatial_autocorr_xi_sd"])
        lines.append(
            f"| C{int(row['condition_index']):02d} | {_fmt(row['assay_alp_mean'])} +/- {_fmt(row['assay_alp_sd'])} | {_fmt(row['assay_collagen_mean'])} +/- {_fmt(row['assay_collagen_sd'])} | {_fmt(row['assay_ha_mean'])} +/- {_fmt(row['assay_ha_sd'])} | {_fmt(row['final_live_mean'])} +/- {_fmt(row['final_live_sd'])} | {_fmt(xi_mean)} +/- {_fmt(xi_sd)} |"
        )
    lines.extend(["", "## Statistical summary", "", "| Metric | Omnibus p | Significant vs control after Holm |", "| --- | ---: | --- |"])
    for metric in PRIMARY_METRICS:
        subset = stats_df[stats_df["metric"] == metric]
        global_p = subset["global_p"].dropna()
        global_p_text = _fmt_p(float(global_p.iloc[0])) if not global_p.empty else "NA"
        sig_conditions = [f"C{int(idx):02d}" for idx in subset.loc[subset["significant_vs_control"], "condition_index"].tolist()]
        lines.append(f"| {metric_label(metric, assay_hours)} | {global_p_text} | {', '.join(sig_conditions) if sig_conditions else 'none'} |")
    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def reorder_rows(df: pd.DataFrame, sort_col: str, ascending: bool = False) -> pd.DataFrame:
    return df.sort_values([sort_col, "condition_index"], ascending=[ascending, True]).reset_index(drop=True)


def export_assay_triptych(
    df: pd.DataFrame,
    stats_df: pd.DataFrame,
    out_dir: Path,
    assay_hours: dict[str, int],
) -> None:
    score = 0.45 * df["assay_ha_pct"] + 0.35 * df["assay_collagen_pct"] + 0.20 * df["assay_alp_pct"]
    ordered = df.assign(_score=score).sort_values(["_score", "condition_index"], ascending=[False, True]).reset_index(drop=True)
    assay_cols = ["assay_alp_pct", "assay_collagen_pct", "assay_ha_pct"]
    assay_titles = [
        f"A  ALP {assay_hours['alp']} h",
        f"B  Collagen {assay_hours['collagen']} h",
        f"C  Mineral {assay_hours['ha']} h",
    ]
    xticklabels = ["ALP", "Collagen", "Mineral"]
    values = ordered[assay_cols].to_numpy(dtype=float)
    vmax = max(3.0, float(np.max(np.abs(values))))
    cmap = response_cmap()

    fig = plt.figure(figsize=(11.2, 9.6))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1.2, 1.2, 1.2, 0.08], wspace=0.12)
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    cax = fig.add_subplot(gs[0, 3])
    metric_names = ["assay_alp", "assay_collagen", "assay_ha"]

    for idx, (ax, col, title, xtick, metric_name) in enumerate(
        zip(axes, assay_cols, assay_titles, xticklabels, metric_names)
    ):
        mat = ordered[[col]].to_numpy(dtype=float)
        im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
        ax.set_xticks([0], [xtick])
        ax.set_yticks(np.arange(len(ordered)))
        if idx == 0:
            ax.set_yticklabels(ordered["row_label"])
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="y", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, loc="left", pad=10, fontsize=12.0)
        metric_stats = stats_df[stats_df["metric"] == metric_name].set_index("condition_index")
        for i in range(mat.shape[0]):
            condition_index = int(ordered.iloc[i]["condition_index"])
            p_adj = float(metric_stats.loc[condition_index, "p_adj"]) if condition_index in metric_stats.index and not pd.isna(metric_stats.loc[condition_index, "p_adj"]) else np.nan
            star = significance_symbol(p_adj)
            stat_text = star if star else ("ns" if condition_index != 1 else "ctrl")
            ax.text(0, i - 0.12, f"{mat[i, 0]:+.1f}", ha="center", va="center", fontsize=8.4, color=COLORS["ink"])
            ax.text(0, i + 0.18, stat_text, ha="center", va="center", fontsize=6.7, color=COLORS["muted"])
        control_row = int(np.where(ordered["condition_index"].to_numpy() == 1)[0][0])
        ax.axhline(control_row - 0.5, color=COLORS["control"], lw=1.1)
        ax.axhline(control_row + 0.5, color=COLORS["control"], lw=1.1)

    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("% difference vs C01", rotation=90)
    fig.suptitle("Assay endpoint heatmaps across initialization modes", fontsize=16, fontweight="semibold", y=0.955)
    fig.subplots_adjust(top=0.86)
    save_figure(fig, out_dir, "figure_1_assay_triptych")


def export_simple_heatmap(
    df: pd.DataFrame,
    value_cols: list[str],
    value_labels: list[str],
    title: str,
    subtitle: str,
    stem: str,
    out_dir: Path,
    *,
    cmap_colors: tuple[str, str, str],
    sort_col: str,
) -> None:
    ordered = reorder_rows(df, sort_col)
    values = ordered[value_cols].to_numpy(dtype=float)
    vmax = max(2.0, float(np.max(np.abs(values))))
    cmap = LinearSegmentedColormap.from_list(stem, list(cmap_colors))

    fig, ax = plt.subplots(figsize=(7.4, 9.4), constrained_layout=True)
    im = ax.imshow(values, aspect="auto", cmap=cmap, vmin=-vmax, vmax=vmax)
    ax.set_xticks(np.arange(len(value_labels)), value_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ordered)), ordered["row_label"])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=8.1, color=COLORS["ink"])

    control_row = int(np.where(ordered["condition_index"].to_numpy() == 1)[0][0])
    ax.axhline(control_row - 0.5, color=COLORS["control"], lw=1.2)
    ax.axhline(control_row + 0.5, color=COLORS["control"], lw=1.2)
    ax.set_title(title, pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("% difference vs C01")
    save_figure(fig, out_dir, stem)


def export_ecm_overview_heatmap(
    df: pd.DataFrame,
    out_dir: Path,
    assay_hours: dict[str, int],
    geometry_mode: str = "classic",
) -> None:
    ordered = reorder_rows(df, "final_ha_pct")
    value_cols = ["assay_alp_pct", "assay_collagen_pct", "assay_ha_pct", "spatial_autocorr_xi_pct"]
    value_sd_cols = [
        "assay_alp_pct_sd",
        "assay_collagen_pct_sd",
        "assay_ha_pct_sd",
        "spatial_autocorr_xi_pct_sd",
    ]
    value_labels = [
        f"ALP {assay_hours['alp']} h",
        f"Collagen {assay_hours['collagen']} h",
        f"Mineral {assay_hours['ha']} h",
        "Autocorrelation",
    ]
    values = ordered[value_cols].to_numpy(dtype=float)
    values_sd = ordered[value_sd_cols].to_numpy(dtype=float)
    vmax = max(2.0, float(np.max(np.abs(values))))
    cmap = response_cmap()
    fs_scale = 1.2
    value_text_scale = 1.2

    fig, ax_main = plt.subplots(figsize=(13.8, 12.4))

    im = ax_main.imshow(
        values,
        aspect="auto",
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        extent=(-0.5, 3.5, len(ordered) - 0.5, -0.5),
    )

    ax_main.set_xlim(-0.5, 3.5)
    ax_main.set_xticks(np.arange(len(value_labels)), value_labels, rotation=0, ha="center")
    ax_main.set_yticks(np.arange(len(ordered)), ordered["condition_label"])
    ax_main.tick_params(axis="x", labelsize=15.5 * fs_scale)
    ax_main.tick_params(axis="y", length=0, labelsize=15.5 * fs_scale)
    for spine in ax_main.spines.values():
        spine.set_visible(False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax_main.text(
                j,
                i,
                f"{values[i, j]:+.1f} ± {values_sd[i, j]:.1f}",
                ha="center",
                va="center",
                fontsize=10.6 * fs_scale * value_text_scale,
                color=COLORS["ink"],
            )

    control_row = int(np.where(ordered["condition_index"].to_numpy() == 1)[0][0])
    ax_main.axhline(control_row - 0.5, color=COLORS["control"], lw=1.2)
    ax_main.axhline(control_row + 0.5, color=COLORS["control"], lw=1.2)

    fig.suptitle("Relative endpoint response heatmap", fontsize=25.0, fontweight="semibold", y=0.968)
    cbar = fig.colorbar(im, ax=ax_main, fraction=0.028, pad=0.016)
    cbar.set_label("% difference vs C01", fontsize=14.0 * fs_scale)
    cbar.ax.tick_params(labelsize=13.0 * fs_scale)
    fig.subplots_adjust(top=0.93, right=0.95, bottom=0.18, left=0.12)
    save_figure(fig, out_dir, "figure_3_ecm_overview_heatmap")


def export_ecm_fingerprint(df: pd.DataFrame, replicate_df: pd.DataFrame | None, out_dir: Path, assay_hours: dict[str, int]) -> None:
    order = (
        df.sort_values(["final_ha_pct", "final_collagen_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    labels = order["condition_label"].tolist()
    y = np.arange(len(order))

    fig = plt.figure(figsize=(13.6, 9.2))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.55, 1.55, 1.05], wspace=0.18)
    ax_ha = fig.add_subplot(gs[0, 0])
    ax_col = fig.add_subplot(gs[0, 1], sharey=ax_ha)
    ax_side = fig.add_subplot(gs[0, 2], sharey=ax_ha)

    ha_vals = order["final_ha_pct"].to_numpy(dtype=float)
    col_vals = order["final_collagen_pct"].to_numpy(dtype=float)
    ha_672 = order["assay_ha_pct"].to_numpy(dtype=float)
    col_336 = order["assay_collagen_pct"].to_numpy(dtype=float)
    ha_err = order["final_ha_pct_sd"].to_numpy(dtype=float)
    col_err = order["final_collagen_pct_sd"].to_numpy(dtype=float)
    ha_672_err = order["assay_ha_pct_sd"].to_numpy(dtype=float)
    col_336_err = order["assay_collagen_pct_sd"].to_numpy(dtype=float)

    ax_ha.axvline(0.0, color=COLORS["neutral"], lw=0.9)
    ax_ha.hlines(y, 0.0, ha_vals, color=COLORS["ha"], lw=2.1, alpha=0.85)
    ax_ha.errorbar(ha_vals, y, xerr=ha_err, fmt="none", ecolor=COLORS["muted"], elinewidth=0.8, alpha=0.75, capsize=2, zorder=2)
    ax_ha.scatter(ha_vals, y, s=56, color=COLORS["ha"], edgecolors="white", linewidths=0.55, zorder=3)
    ax_ha.errorbar(ha_672, y, xerr=ha_672_err, fmt="none", ecolor=COLORS["muted"], elinewidth=0.65, alpha=0.65, capsize=1.5, zorder=2)
    ax_ha.scatter(ha_672, y, s=24, color="#d97706", edgecolors="white", linewidths=0.45, zorder=3)
    style_axes(ax_ha, "x")
    ax_ha.set_title("HAp response")
    ax_ha.set_xlabel("% difference vs C01")
    ax_ha.set_yticks(y, labels)
    ax_ha.invert_yaxis()

    ax_col.axvline(0.0, color=COLORS["neutral"], lw=0.9)
    ax_col.hlines(y, 0.0, col_vals, color=COLORS["collagen"], lw=2.1, alpha=0.85)
    ax_col.errorbar(col_vals, y, xerr=col_err, fmt="none", ecolor=COLORS["muted"], elinewidth=0.8, alpha=0.75, capsize=2, zorder=2)
    ax_col.scatter(col_vals, y, s=56, color=COLORS["collagen"], edgecolors="white", linewidths=0.55, zorder=3)
    ax_col.errorbar(col_336, y, xerr=col_336_err, fmt="none", ecolor=COLORS["muted"], elinewidth=0.65, alpha=0.65, capsize=1.5, zorder=2)
    ax_col.scatter(col_336, y, s=24, color="#0f9f6e", edgecolors="white", linewidths=0.45, zorder=3)
    style_axes(ax_col, "x")
    ax_col.set_title("Collagen response")
    ax_col.set_xlabel("% difference vs C01")
    ax_col.tick_params(axis="y", labelleft=False)

    side_cols = ["seed_mean_radius_mean", "seed_edge_fraction_mean", "seed_mean_nearest_neighbor_mean"]
    side_labels = ["Radius", "Edge", "NN"]
    side = order[side_cols].to_numpy(dtype=float)
    side_scaled = np.zeros_like(side)
    for i in range(side.shape[1]):
        col = side[:, i]
        denom = max(float(np.max(col) - np.min(col)), 1e-12)
        side_scaled[:, i] = (col - np.min(col)) / denom
    side_im = ax_side.imshow(side_scaled, aspect="auto", cmap=LinearSegmentedColormap.from_list("geo_grey", ["#f8fafc", "#4b5563"]), vmin=0.0, vmax=1.0)
    ax_side.set_xticks(np.arange(len(side_labels)), side_labels, rotation=35, ha="right")
    ax_side.tick_params(axis="y", labelleft=False, length=0)
    ax_side.set_title("Initialization geometry")
    for i in range(side_scaled.shape[0]):
        for j in range(side_scaled.shape[1]):
            ax_side.text(j, i, f"{side[i,j]:.2f}", ha="center", va="center", fontsize=7.0, color=COLORS["ink"])
    for s in ax_side.spines.values():
        s.set_visible(False)

    norm = Normalize(vmin=0.0, vmax=1.0)
    cbar = fig.colorbar(ScalarMappable(norm=norm, cmap=LinearSegmentedColormap.from_list("geo_grey", ["#f8fafc", "#4b5563"])), ax=ax_side, fraction=0.05, pad=0.04)
    cbar.set_label("Scaled geometry descriptor")

    add_panel_label(ax_ha, "D")
    fig.suptitle(
        "ECM fingerprint ranking across initialization modes",
        fontsize=15,
        fontweight="semibold",
        y=0.98,
    )
    save_figure(fig, out_dir, "figure_4_ecm_fingerprint")


def export_hap_collagen_map(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    order = df.sort_values(["final_ha_pct", "final_collagen_pct"], ascending=[False, False]).reset_index(drop=True)
    x = order["final_collagen_pct"].to_numpy(dtype=float)
    y = order["final_ha_pct"].to_numpy(dtype=float)
    alp = order["assay_alp_pct"].to_numpy(dtype=float)
    edge = order["seed_edge_fraction_mean"].to_numpy(dtype=float)
    size = 70.0 + 170.0 * (
        (edge - float(np.min(edge))) / max(float(np.max(edge) - np.min(edge)), 1e-12)
    )
    cmap = response_cmap()
    vmax = max(3.0, float(np.max(np.abs(alp))))

    fig = plt.figure(figsize=(9.8, 7.2))
    gs = gridspec.GridSpec(1, 2, width_ratios=[5.2, 0.18], wspace=0.2)
    ax = fig.add_subplot(gs[0, 0])
    cax = fig.add_subplot(gs[0, 1])

    xmin = min(float(np.min(x)), -1.0) - 0.5
    xmax = max(float(np.max(x)), 1.0) + 0.5
    ymin = min(float(np.min(y)), -1.0) - 0.7
    ymax = max(float(np.max(y)), 1.0) + 0.7

    ax.axvspan(xmin, 0.0, ymin=0.5, ymax=1.0, color="#e7effa", alpha=0.7, zorder=0)
    ax.axvspan(0.0, xmax, ymin=0.5, ymax=1.0, color="#fff1e6", alpha=0.7, zorder=0)
    ax.axvspan(xmin, 0.0, ymin=0.0, ymax=0.5, color="#f1f5f9", alpha=0.9, zorder=0)
    ax.axvspan(0.0, xmax, ymin=0.0, ymax=0.5, color="#edf7f2", alpha=0.75, zorder=0)
    ax.axvline(0.0, color="#64748b", lw=1.0, zorder=1)
    ax.axhline(0.0, color="#64748b", lw=1.0, zorder=1)

    sc = ax.scatter(
        x,
        y,
        c=alp,
        s=size,
        cmap=cmap,
        vmin=-vmax,
        vmax=vmax,
        edgecolors="white",
        linewidths=0.8,
        alpha=0.97,
        zorder=3,
    )

    radial = np.hypot(x, y)
    label_idx = set(np.argsort(radial)[-7:].tolist())
    control_idx = int(order.index[order["condition_index"] == 1][0])
    label_idx.add(control_idx)
    for idx, row in order.iterrows():
        if idx not in label_idx:
            continue
        dx = 0.08 if float(row["final_collagen_pct"]) >= 0 else -0.18
        dy = 0.08 if float(row["final_ha_pct"]) >= 0 else -0.14
        ax.text(
            float(row["final_collagen_pct"]) + dx,
            float(row["final_ha_pct"]) + dy,
            row["condition_label"],
            fontsize=8.0,
            color=COLORS["ink"],
            zorder=4,
            bbox={"boxstyle": "round,pad=0.12", "fc": (1, 1, 1, 0.72), "ec": "none"},
        )

    ctrl = order.loc[order["condition_index"] == 1].iloc[0]
    ax.scatter(
        [float(ctrl["final_collagen_pct"])],
        [float(ctrl["final_ha_pct"])],
        s=260,
        facecolors="none",
        edgecolors=COLORS["control"],
        linewidths=1.4,
        zorder=5,
    )

    style_axes(ax, "both")
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.set_xlabel("Final collagen difference vs C01 (%)")
    ax.set_ylabel("Final HAp difference vs C01 (%)")
    ax.set_title(f"ECM response map with ALP encoding ({assay_hours['alp']} h)", pad=18)
    cbar = fig.colorbar(sc, cax=cax)
    cbar.set_label("Selected ALP difference vs C01 (%)")
    cbar.ax.yaxis.set_label_position("right")
    cbar.ax.yaxis.tick_right()

    legend_sizes = [0.0, 0.4, 0.8]
    size_handles = [
        plt.scatter([], [], s=70.0 + 170.0 * frac, color="#94a3b8", edgecolors="white", linewidths=0.8)
        for frac in legend_sizes
    ]
    ax.legend(
        size_handles,
        [f"Edge fraction {frac:.1f}" for frac in legend_sizes],
        frameon=False,
        loc="lower right",
        fontsize=8.2,
        title="Point size",
        title_fontsize=8.6,
    )

    fig.subplots_adjust(top=0.86, right=0.93)
    save_figure(fig, out_dir, "figure_2_ecm_response_map")


def export_geometry_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    ordered = reorder_rows(df, "seed_mean_radius_mean")
    geo_cols = ["seed_mean_radius_mean", "seed_edge_fraction_mean", "seed_mean_nearest_neighbor_mean"]
    geo_labels = ["Mean radius", "Edge fraction", "NN distance"]
    geo = ordered[geo_cols].to_numpy(dtype=float)
    scaled = np.zeros_like(geo)
    for i in range(geo.shape[1]):
        col = geo[:, i]
        denom = max(float(np.max(col) - np.min(col)), 1e-12)
        scaled[:, i] = (col - np.min(col)) / denom
    fig, ax = plt.subplots(figsize=(7.8, 9.0))
    im = ax.imshow(scaled, aspect="auto", cmap=LinearSegmentedColormap.from_list("geo_grey", ["#f8fafc", "#4b5563"]), vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(geo_labels)), geo_labels, rotation=25, ha="right")
    ax.set_yticks(np.arange(len(ordered)), ordered["row_label"])
    ax.tick_params(axis="y", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(geo.shape[0]):
        for j in range(geo.shape[1]):
            ax.text(j, i, f"{geo[i, j]:.2f}", ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
    ax.set_title("Initialization geometry reference", pad=10)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Scaled descriptor value")
    fig.subplots_adjust(top=0.9)
    save_figure(fig, out_dir, "figure_5_geometry_reference")


def _find_grid(rep_dir: Path, prefix: str, preferred_day: int) -> Path:
    preferred = rep_dir / "grids" / f"{prefix}_day_{preferred_day:03d}.npy"
    if preferred.exists():
        return preferred
    candidates = sorted((rep_dir / "grids").glob(f"{prefix}_day_*.npy"))
    if not candidates:
        raise SystemExit(f"No {prefix} grids found in {rep_dir / 'grids'}")
    return candidates[-1]


def _spatial_signature(ha: np.ndarray, state: np.ndarray) -> np.ndarray:
    state_mask = (state > 0).astype(float)
    ob_mask = (state == 2).astype(float)
    ha_scaled = ha.astype(float)
    if float(np.max(ha_scaled)) > 1e-12:
        ha_scaled = ha_scaled / float(np.max(ha_scaled))
    signature = np.concatenate(
        [
            ha_scaled.ravel(),
            0.35 * state_mask.ravel(),
            0.20 * ob_mask.ravel(),
        ]
    )
    signature = signature - float(np.mean(signature))
    norm = float(np.linalg.norm(signature))
    if norm > 1e-12:
        signature = signature / norm
    return signature


def _render_state_thumb(
    ax: plt.Axes,
    ha: np.ndarray,
    vmax: float,
    title: str | None = None,
    title_fontsize: float = 8.2,
) -> None:
    ax.imshow(ha, origin="upper", cmap="viridis", vmin=0.0, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(False)
    if title:
        ax.set_title(title, loc="left", fontsize=title_fontsize, pad=2.5)


def _render_initial_state_thumb(
    ax: plt.Axes,
    state: np.ndarray,
    title: str | None = None,
    title_fontsize: float = 11.0,
) -> None:
    ys, xs = np.nonzero(state > 0)
    h, w = state.shape
    ax.set_facecolor("#f3f7fc")
    if xs.size:
        ax.scatter(
            xs,
            ys,
            s=22.0,
            c="#1d67c9",
            edgecolors="#eef5ff",
            linewidths=0.45,
            alpha=1.0,
        )
    ax.set_xlim(-0.5, w - 0.5)
    ax.set_ylim(h - 0.5, -0.5)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.8)
        spine.set_edgecolor("#e2e8f0")
    if title:
        ax.set_title(title, loc="left", fontsize=title_fontsize, pad=4.0)


def export_spatial_gallery(
    df: pd.DataFrame,
    out_dir: Path,
    *,
    target_day: int,
    title: str,
    stem: str,
    mode: str = "final",
) -> None:
    ordered = df.sort_values(["assay_ha_pct", "condition_index"], ascending=[False, True]).reset_index(drop=True)
    labels = ordered["condition_label"].tolist()
    has: list[np.ndarray] = []
    states: list[np.ndarray] = []
    for _, row in ordered.iterrows():
        rep_dir = Path(str(row["representative_out_dir"]))
        if mode == "initial":
            state_path = _find_grid(rep_dir, "state", 0)
            state = np.load(state_path)
            states.append(state)
        else:
            ha_path = _find_grid(rep_dir, "ha", target_day)
            ha = np.load(ha_path).astype(float)
            has.append(ha)
    vmax_ha = max((max(float(np.max(ha)) for ha in has) if has else 0.0), 1e-9)

    n_conditions = len(labels)
    ncols = 5
    nrows = int(np.ceil(n_conditions / ncols))
    fig_h = 3.25 * nrows
    fig = plt.figure(figsize=(18.8, fig_h))
    gallery_gs = gridspec.GridSpec(nrows, ncols, figure=fig, wspace=0.12, hspace=0.18)
    tile_label_fontsize = 16.8 if mode == "initial" else 8.2

    for idx, label in enumerate(labels):
        r = idx // ncols
        c = idx % ncols
        gax = fig.add_subplot(gallery_gs[r, c])
        if mode == "initial":
            _render_initial_state_thumb(
                gax, states[idx], title=label, title_fontsize=tile_label_fontsize
            )
        else:
            _render_state_thumb(
                gax, has[idx], vmax_ha, title=label, title_fontsize=tile_label_fontsize
            )
        if int(ordered.iloc[idx]["condition_index"]) == 1:
            for spine in gax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1.8)
                spine.set_edgecolor("#2563eb")
    for idx in range(n_conditions, nrows * ncols):
        r = idx // ncols
        c = idx % ncols
        gax = fig.add_subplot(gallery_gs[r, c])
        gax.axis("off")

    title_size = 19.5 if mode == "initial" else 18
    fig.suptitle(title, fontsize=title_size, fontweight="semibold", y=0.992)
    fig.subplots_adjust(top=0.94, right=0.99, left=0.025, bottom=0.03)
    save_figure(fig, out_dir, stem)


def main() -> None:
    args = parse_args()
    out_dir = args.out if args.out is not None else (args.input / "figure_extract_2")
    replicate_df, manifest = load_inputs(args.input)
    assay_hours = assay_hours_from_args(args, manifest)
    replicate_df = enrich_replicate_summary(replicate_df, assay_hours)
    condition_summary = build_condition_summary(replicate_df)
    stats_df = compute_statistics(replicate_df, assay_hours)
    condition_summary.to_csv(args.input / "summary.csv", index=False)
    write_statistics_tables(stats_df, args.input)
    write_report(
        condition_summary,
        replicate_df,
        stats_df,
        args.input,
        assay_hours,
    )

    summary = prep_summary(condition_summary, domain_area_mm2=float(args.domain_area_mm2))
    # Keep the original heatmap-style outputs.
    export_assay_triptych(summary, stats_df, out_dir, assay_hours)
    export_hap_collagen_map(summary, out_dir, assay_hours)
    export_ecm_overview_heatmap(
        summary,
        out_dir,
        assay_hours,
        geometry_mode=args.figure3_geometry_mode,
    )
    export_ecm_fingerprint(summary, replicate_df, out_dir, assay_hours)
    export_geometry_heatmap(summary, out_dir)
    export_spatial_gallery(
        summary,
        out_dir,
        target_day=0,
        title="Spatial Seed Initialization Gallery",
        stem="supplementary_figure_s1_spatial_gallery_initial",
        mode="initial",
    )
    export_spatial_gallery(
        summary,
        out_dir,
        target_day=max(0, int(round(float(assay_hours["ha"]) / 24.0))),
        title=f"Spatial HAp Heatmap Gallery ({assay_hours['ha']} h)",
        stem="supplementary_figure_s1_spatial_gallery",
        mode="final",
    )
    print(f"Wrote figures to {out_dir}")


if __name__ == "__main__":
    main()
