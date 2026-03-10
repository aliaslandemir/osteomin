#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.ticker import ScalarFormatter

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent

COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "dex": "#d55e00",
    "mk4": "#0072b2",
    "alp": "#c2410c",
    "collagen": "#059669",
    "ha": "#1d4e89",
    "neutral": "#94a3b8",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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
        description="Figure extraction for sanity_check4 single-drug dose runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "out" / "sanity_check4",
        help="sanity_check4 output folder containing condition_manifest and per-condition outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder for extracted figures; defaults to <input>/figure_extract_4",
    )
    parser.add_argument("--alp-hours", type=int, default=240, help="hour used for ALP assay extraction")
    parser.add_argument("--collagen-hours", type=int, default=336, help="hour used for collagen assay extraction")
    parser.add_argument("--hap-hours", type=int, default=672, help="hour used for HAp assay extraction")
    return parser.parse_args()


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=320, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y", grid_alpha: float = 0.7) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=grid_alpha)


def response_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("plos_cb_div", ["#2b6c8f", "#f8f5ef", "#b44d2a"])


def row_at_hours(df: pd.DataFrame, hours: int) -> pd.Series:
    target_day = float(hours) / 24.0
    idx = int((df["day"] - target_day).abs().idxmin())
    return df.loc[idx]


def find_condition_timeseries(condition_out_dir: Path) -> list[Path]:
    direct = condition_out_dir / "timeseries.csv"
    if direct.exists():
        return [direct]
    replicate_paths = sorted(condition_out_dir.glob("r*/timeseries.csv"))
    if replicate_paths:
        return replicate_paths
    nested = sorted(condition_out_dir.rglob("timeseries.csv"))
    return nested


def load_manifest(input_dir: Path) -> pd.DataFrame:
    manifest_csv = input_dir / "condition_manifest.csv"
    manifest_json = input_dir / "condition_manifest.json"
    if manifest_csv.exists():
        return pd.read_csv(manifest_csv)
    if manifest_json.exists():
        return pd.DataFrame(json.loads(manifest_json.read_text(encoding="utf-8")))
    raise SystemExit(f"condition_manifest.[csv|json] not found under {input_dir}")


def build_summary(input_dir: Path, assay_hours: dict[str, int]) -> pd.DataFrame:
    manifest = load_manifest(input_dir).copy()
    rows: list[dict[str, float | int | str]] = []
    for _, row in manifest.iterrows():
        out_dir = Path(str(row["out_dir"]))
        ts_paths = find_condition_timeseries(out_dir)
        if not ts_paths:
            raise SystemExit(f"timeseries.csv not found under condition output: {out_dir}")
        metrics_by_run: list[dict[str, float]] = []
        for ts_path in ts_paths:
            ts = pd.read_csv(ts_path)
            alp_row = row_at_hours(ts, assay_hours["alp"])
            col_row = row_at_hours(ts, assay_hours["collagen"])
            ha_row = row_at_hours(ts, assay_hours["ha"])
            final_row = ts.iloc[-1]
            metrics_by_run.append(
                {
                    "assay_alp": float(alp_row["alp_total"]),
                    "assay_collagen": float(col_row["collagen_total"]),
                    "assay_ha": float(ha_row["ha_total"]),
                    "final_live": float(final_row["n_total"]),
                    "final_preob": float(final_row["n_preob"]),
                    "final_ob": float(final_row["n_osteoblast"]),
                }
            )
        run_df = pd.DataFrame(metrics_by_run)
        dex = float(row["dex_conc_nM"])
        mk4 = float(row["mk4_conc_uM"])
        branch = "mk4" if dex == 0.0 and mk4 > 0.0 else "dex"
        dose = mk4 if branch == "mk4" else dex
        rows.append(
            {
                "condition_index": int(row["index"]),
                "slug": str(row["slug"]),
                "branch": branch,
                "dose": dose,
                "dose_units": "uM" if branch == "mk4" else "nM",
                "dex_conc_nM": dex,
                "mk4_conc_uM": mk4,
                "n_replicates": int(len(run_df)),
                "assay_alp": float(run_df["assay_alp"].mean()),
                "assay_collagen": float(run_df["assay_collagen"].mean()),
                "assay_ha": float(run_df["assay_ha"].mean()),
                "final_live": float(run_df["final_live"].mean()),
                "final_preob": float(run_df["final_preob"].mean()),
                "final_ob": float(run_df["final_ob"].mean()),
                "assay_alp_sd": float(run_df["assay_alp"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "assay_collagen_sd": float(run_df["assay_collagen"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "assay_ha_sd": float(run_df["assay_ha"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "final_live_sd": float(run_df["final_live"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "final_preob_sd": float(run_df["final_preob"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "final_ob_sd": float(run_df["final_ob"].std(ddof=1)) if len(run_df) > 1 else 0.0,
            }
        )
    df = pd.DataFrame(rows).sort_values(["branch", "dose"]).reset_index(drop=True)
    for branch in ("dex", "mk4"):
        mask = df["branch"] == branch
        branch_df = df.loc[mask].copy().sort_values("dose")
        vehicle_mask = branch_df["dose"] == 0.0
        for col, sd_col in [
            ("assay_alp", "assay_alp_sd"),
            ("assay_collagen", "assay_collagen_sd"),
            ("assay_ha", "assay_ha_sd"),
            ("final_live", "final_live_sd"),
            ("final_preob", "final_preob_sd"),
            ("final_ob", "final_ob_sd"),
        ]:
            if bool(vehicle_mask.any()):
                base = float(branch_df.loc[vehicle_mask, col].iloc[0])
            else:
                base = float(branch_df.iloc[0][col])
            df.loc[mask, f"{col}_pct_vs_vehicle"] = 100.0 * (
                branch_df[col].to_numpy(dtype=float) - base
            ) / max(base, 1e-12)
            df.loc[mask, f"{col}_pct_sd_vs_vehicle"] = 100.0 * branch_df[sd_col].to_numpy(dtype=float) / max(base, 1e-12)
    return df


def split_branches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dex_df = df[df["branch"] == "dex"].sort_values("dose").reset_index(drop=True)
    mk4_df = df[df["branch"] == "mk4"].sort_values("dose").reset_index(drop=True)
    if dex_df.empty or mk4_df.empty:
        raise SystemExit("sanity_check4 requires both Dex-only and MK-4-only branches")
    return dex_df, mk4_df


def peak_window_for_metric(branch_df: pd.DataFrame, peak_col: str) -> tuple[float, float]:
    branch_sorted = branch_df.sort_values("dose").reset_index(drop=True)
    local_idx = int(branch_sorted[peak_col].astype(float).idxmax())
    doses = branch_sorted["dose"].to_numpy(dtype=float)
    peak_low = float(doses[local_idx - 1]) if local_idx > 0 else float(doses[local_idx])
    peak_high = (
        float(doses[local_idx + 1])
        if local_idx < len(doses) - 1
        else float(doses[local_idx])
    )
    return peak_low, peak_high


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "condition_summary.csv", index=False)
    (out_dir / "condition_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def export_assay_dose_curves(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.1, 9.5), sharex="col", sharey="row")
    axes[0, 0].set_title("Dex only", pad=4)
    axes[0, 1].set_title("MK-4 only", pad=4)
    rows = [
        ("assay_alp_pct_vs_vehicle", "assay_alp_pct_sd_vs_vehicle", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_vehicle", "assay_collagen_pct_sd_vs_vehicle", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_vehicle", "assay_ha_pct_sd_vs_vehicle", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    dex_pos = dex_df[dex_df["dose"] > 0.0].copy()
    mk4_pos = mk4_df[mk4_df["dose"] > 0.0].copy()
    dex_ticks = dex_pos["dose"].to_numpy(dtype=float)
    mk4_ticks = mk4_pos["dose"].to_numpy(dtype=float)

    def add_peak_marker(ax: plt.Axes, peak_low: float, peak_high: float, shade_color: str, y_top: float) -> None:
        ax.axvspan(peak_low, peak_high, color=shade_color, alpha=0.08, zorder=0)
        ax.axvline(peak_low, color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.axvline(peak_high, color=COLORS["muted"], lw=1.0, ls=(0, (4, 3)), zorder=1)
        ax.text(
            np.sqrt(peak_low * peak_high),
            y_top,
            "peak window",
            ha="center",
            va="bottom",
            fontsize=8.5,
            color=COLORS["muted"],
        )

    for ridx, (col, err_col, title, color) in enumerate(rows):
        row_vals = np.concatenate(
            [
                dex_df[col].to_numpy(dtype=float),
                dex_df[err_col].to_numpy(dtype=float),
                mk4_df[col].to_numpy(dtype=float),
                mk4_df[err_col].to_numpy(dtype=float),
            ]
        )
        y_abs = float(np.max(np.abs(row_vals))) if row_vals.size else 1.0
        y_pad = max(0.8, 0.14 * y_abs)
        y_lim = (-y_abs - y_pad, y_abs + y_pad)
        y_peak = y_lim[1] - 0.08 * (y_lim[1] - y_lim[0])
        line_width = 2.5 if "ha" in col else 1.9
        dex_peak_low, dex_peak_high = peak_window_for_metric(dex_pos, col)
        mk4_peak_low, mk4_peak_high = peak_window_for_metric(mk4_pos, col)

        ax = axes[ridx, 0]
        style_axes(ax, "both", grid_alpha=0.3)
        ax.errorbar(
            dex_pos["dose"],
            dex_pos[col],
            yerr=dex_pos[err_col],
            color=color,
            marker="o",
            lw=line_width,
            ms=5.2,
            capsize=2,
            alpha=0.9,
            elinewidth=0.8,
            capthick=0.8,
            markeredgewidth=0.6,
        )
        ax.set_xscale("log")
        ax.axhline(0.0, color=COLORS["neutral"], lw=0.9, ls=(0, (4, 3)))
        ax.set_ylim(*y_lim)
        ax.set_xticks(dex_ticks, [f"{d:.4g}" for d in dex_ticks])
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_ylabel(f"{title}\n% vs vehicle")
        add_peak_marker(ax, dex_peak_low, dex_peak_high, COLORS["dex"], y_peak)
        if ridx == 2:
            ax.set_xlabel("Dex concentration (nM)")

        ax = axes[ridx, 1]
        style_axes(ax, "both", grid_alpha=0.3)
        ax.errorbar(
            mk4_pos["dose"],
            mk4_pos[col],
            yerr=mk4_pos[err_col],
            color=color,
            marker="o",
            lw=line_width,
            ms=5.2,
            capsize=2,
            alpha=0.9,
            elinewidth=0.8,
            capthick=0.8,
            markeredgewidth=0.6,
        )
        ax.set_xscale("log")
        ax.axhline(0.0, color=COLORS["neutral"], lw=0.9, ls=(0, (4, 3)))
        ax.set_ylim(*y_lim)
        ax.set_xticks(mk4_ticks, [f"{d:.4g}" for d in mk4_ticks])
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        add_peak_marker(ax, mk4_peak_low, mk4_peak_high, COLORS["mk4"], y_peak)
        if ridx == 2:
            ax.set_xlabel("MK-4 concentration (uM)")
    fig.subplots_adjust(left=0.11, right=0.985, bottom=0.11, top=0.91, wspace=0.12, hspace=0.18)
    fig.suptitle("Single-drug dose-response curves", fontsize=15, fontweight="semibold", y=0.975)
    fig.text(
        0.5,
        0.035,
        (
            f"Mean ± SD (n=5 replicates); responses shown as % vs branch vehicle. "
            "Each panel uses its own assay-specific peak window from the local sampled-dose neighborhood."
        ),
        ha="center",
        fontsize=9,
        color=COLORS["muted"],
    )
    save_figure(fig, out_dir, "figure_1_dose_response_curves")


def export_relative_effect_heatmap(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.6), constrained_layout=True)
    metric_cols = [
        ("assay_alp_pct_vs_vehicle", f"ALP {assay_hours['alp']} h"),
        ("assay_collagen_pct_vs_vehicle", f"Collagen {assay_hours['collagen']} h"),
        ("assay_ha_pct_vs_vehicle", f"HAp {assay_hours['ha']} h"),
        ("final_live_pct_vs_vehicle", "Final live"),
    ]
    metric_sd_cols = [c.replace("_pct_vs_vehicle", "_pct_sd_vs_vehicle") for c, _ in metric_cols]
    vmax = max(
        0.02,
        float(
            np.max(
                np.abs(
                    np.concatenate(
                        [
                            dex_df[[c for c, _ in metric_cols]].to_numpy(dtype=float).ravel(),
                            mk4_df[[c for c, _ in metric_cols]].to_numpy(dtype=float).ravel(),
                        ]
                    )
                )
            )
        ),
    )
    for ax, branch_df, title in zip(axes, [dex_df, mk4_df], ["Dex only", "MK-4 only"]):
        style_axes(ax, None)
        values = branch_df[[c for c, _ in metric_cols]].to_numpy(dtype=float).T
        errors = branch_df[metric_sd_cols].to_numpy(dtype=float).T
        im = ax.imshow(
            values,
            aspect="auto",
            cmap=response_cmap(),
            norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        )
        ax.set_xticks(np.arange(len(branch_df)), [f"{d:.4g}" for d in branch_df["dose"]], rotation=0)
        ax.set_yticks(np.arange(len(metric_cols)), [label for _, label in metric_cols])
        ax.set_xlabel("Dose")
        ax.set_title(title, pad=10)
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                ax.text(j, i, f"{values[i, j]:+.2f}\n±{errors[i, j]:.2f}", ha="center", va="center", fontsize=6.6, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("Mean % difference vs branch vehicle")
    fig.suptitle("Relative effect heatmaps", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_2_relative_effect_heatmap")


def export_local_sensitivity(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    metric_defs = [
        ("assay_alp_pct_vs_vehicle", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_vehicle", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_vehicle", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    for panel_idx, (ax, branch_df, title, xlabel) in enumerate(zip(
        axes,
        [dex_df, mk4_df],
        ["Dex adjacent-dose sensitivity", "MK-4 adjacent-dose sensitivity"],
        ["Dex concentration (nM, log scale)", "MK-4 concentration (uM, log scale)"],
    )):
        style_axes(ax, "both")
        dose = branch_df["dose"].to_numpy(dtype=float)
        pos_mask = dose > 0.0
        dose_pos = dose[pos_mask]
        if dose_pos.size < 2:
            continue
        x_mid = np.sqrt(dose_pos[:-1] * dose_pos[1:])
        denom = np.diff(np.log10(dose_pos))
        for col, label, color in metric_defs:
            values = branch_df[col].to_numpy(dtype=float)[pos_mask]
            slope = np.diff(values) / np.maximum(denom, 1e-12)
            ax.plot(x_mid, slope, color=color, marker="o", lw=2.0, ms=5.0, label=label)
        ax.set_xscale("log")
        ax.axhline(0.0, color=COLORS["neutral"], lw=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Adjacent slope (% per log10 dose)")
        if panel_idx == 1:
            ax.legend(frameon=False, loc="upper right")
    fig.suptitle("Adjacent-dose sensitivity analysis", fontsize=15, fontweight="semibold", y=1.03)
    save_figure(fig, out_dir, "figure_3_local_sensitivity")


def export_peak_summary(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    metric_defs = [
        ("assay_alp", f"ALP {assay_hours['alp']} h", COLORS["alp"], "assay_alp_pct_vs_vehicle"),
        ("assay_collagen", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"], "assay_collagen_pct_vs_vehicle"),
        ("assay_ha", f"HAp {assay_hours['ha']} h", COLORS["ha"], "assay_ha_pct_vs_vehicle"),
    ]
    rows: list[dict[str, float | str]] = []
    for branch_name, branch_df in [("Dex", dex_df), ("MK-4", mk4_df)]:
        for col, label, color, pct_col in metric_defs:
            branch_sorted = branch_df.sort_values("dose").reset_index(drop=True)
            local_idx = int(branch_sorted[col].astype(float).idxmax())
            row = branch_sorted.loc[local_idx]
            sd_col = pct_col.replace("_pct_vs_vehicle", "_pct_sd_vs_vehicle")
            max_idx = int(branch_sorted[pct_col].astype(float).idxmax())
            min_idx = int(branch_sorted[pct_col].astype(float).idxmin())
            range_sd = float(
                np.sqrt(
                    float(branch_sorted.iloc[max_idx][sd_col]) ** 2
                    + float(branch_sorted.iloc[min_idx][sd_col]) ** 2
                )
            )
            peak_val = float(row[col])
            doses = branch_sorted["dose"].to_numpy(dtype=float)
            if local_idx > 0:
                peak_low = float(doses[local_idx - 1])
            else:
                peak_low = float(doses[local_idx])
            if local_idx < len(doses) - 1:
                peak_high = float(doses[local_idx + 1])
            else:
                peak_high = float(doses[local_idx])
            rows.append(
                {
                    "branch": branch_name,
                    "metric": label,
                    "color": color,
                    "peak_value": peak_val,
                    "peak_dose": float(row["dose"]),
                    "peak_dose_low": peak_low,
                    "peak_dose_high": peak_high,
                    "dynamic_range_pct": float(branch_df[pct_col].max() - branch_df[pct_col].min()),
                    "dynamic_range_sd": range_sd,
                }
            )
    summary = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8), constrained_layout=True)
    ax = axes[0]
    style_axes(ax, "y")
    x = np.arange(len(metric_defs))
    width = 0.35
    dex_rng = summary[summary["branch"] == "Dex"]["dynamic_range_pct"].to_numpy(dtype=float)
    mk4_rng = summary[summary["branch"] == "MK-4"]["dynamic_range_pct"].to_numpy(dtype=float)
    dex_err = summary[summary["branch"] == "Dex"]["dynamic_range_sd"].to_numpy(dtype=float)
    mk4_err = summary[summary["branch"] == "MK-4"]["dynamic_range_sd"].to_numpy(dtype=float)
    ax.bar(x - width / 2.0, dex_rng, width=width, color=COLORS["dex"], yerr=dex_err, capsize=2, label="Dex")
    ax.bar(x + width / 2.0, mk4_rng, width=width, color=COLORS["mk4"], yerr=mk4_err, capsize=2, label="MK-4")
    ax.set_xticks(x, [label for _, label, _, _ in metric_defs], rotation=20, ha="right")
    ax.set_ylabel("Dynamic range (% vs vehicle)")
    ax.set_title("Dynamic range by drug branch", pad=10)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    style_axes(ax, "y")
    branch_pos = {"Dex": 0.0, "MK-4": 1.0}
    metric_offsets = {metric_defs[0][1]: -0.18, metric_defs[1][1]: 0.0, metric_defs[2][1]: 0.18}
    for _, row in summary.iterrows():
        xpos = branch_pos[str(row["branch"])] + metric_offsets[str(row["metric"])]
        y = float(row["peak_dose"])
        low = float(row["peak_dose_low"])
        high = float(row["peak_dose_high"])
        ax.scatter(
            xpos,
            y,
            s=90,
            color=str(row["color"]),
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.errorbar(
            xpos,
            y,
            yerr=[[max(y - low, 0.0)], [max(high - y, 0.0)]],
            fmt="none",
            ecolor=COLORS["muted"],
            elinewidth=1.0,
            capsize=2,
            zorder=2,
        )
        ax.text(xpos, y, f"{row['peak_dose']:.4g}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0.0, 1.0], ["Dex", "MK-4"])
    ax.set_ylabel("Peak dose (native units)")
    ax.set_title("Peak dose by assay endpoint", pad=10)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for _, label, color, _ in metric_defs
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    fig.suptitle("Peak-response summary", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_4_peak_summary")


def main() -> None:
    args = parse_args()
    out_dir = args.out if args.out is not None else args.input / "figure_extract_4"
    assay_hours = {
        "alp": int(args.alp_hours),
        "collagen": int(args.collagen_hours),
        "ha": int(args.hap_hours),
    }

    df = build_summary(args.input, assay_hours)
    dex_df, mk4_df = split_branches(df)
    write_tables(df, out_dir)

    export_assay_dose_curves(dex_df, mk4_df, out_dir, assay_hours)
    export_relative_effect_heatmap(dex_df, mk4_df, out_dir, assay_hours)
    export_local_sensitivity(dex_df, mk4_df, out_dir, assay_hours)
    export_peak_summary(dex_df, mk4_df, out_dir, assay_hours)

    print(f"[sanity4_figure_extract] processed {len(df)} conditions")
    print(f"[sanity4_figure_extract] figures written to {out_dir}")


if __name__ == "__main__":
    main()
