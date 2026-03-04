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


COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "dex": "#d55e00",
    "mk4": "#009e73",
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
        default=Path("out") / "sanity_check4",
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
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.7)


def response_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("plos_cb_div", ["#2b6c8f", "#f8f5ef", "#b44d2a"])


def row_at_hours(df: pd.DataFrame, hours: int) -> pd.Series:
    target_day = float(hours) / 24.0
    idx = int((df["day"] - target_day).abs().idxmin())
    return df.loc[idx]


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
        ts_path = out_dir / "timeseries.csv"
        if not ts_path.exists():
            raise SystemExit(f"timeseries.csv not found for condition output: {ts_path}")
        ts = pd.read_csv(ts_path)
        alp_row = row_at_hours(ts, assay_hours["alp"])
        col_row = row_at_hours(ts, assay_hours["collagen"])
        ha_row = row_at_hours(ts, assay_hours["ha"])
        final_row = ts.iloc[-1]
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
                "assay_alp": float(alp_row["alp_total"]),
                "assay_collagen": float(col_row["collagen_total"]),
                "assay_ha": float(ha_row["ha_total"]),
                "final_live": float(final_row["n_total"]),
                "final_preob": float(final_row["n_preob"]),
                "final_ob": float(final_row["n_osteoblast"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["branch", "dose"]).reset_index(drop=True)
    for branch in ("dex", "mk4"):
        mask = df["branch"] == branch
        branch_df = df.loc[mask].copy().sort_values("dose")
        for col in ("assay_alp", "assay_collagen", "assay_ha", "final_live", "final_preob", "final_ob"):
            base = float(branch_df.iloc[0][col])
            df.loc[mask, f"{col}_pct_vs_branch_min"] = 100.0 * (
                branch_df[col].to_numpy(dtype=float) - base
            ) / max(base, 1e-12)
    return df


def split_branches(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    dex_df = df[df["branch"] == "dex"].sort_values("dose").reset_index(drop=True)
    mk4_df = df[df["branch"] == "mk4"].sort_values("dose").reset_index(drop=True)
    if dex_df.empty or mk4_df.empty:
        raise SystemExit("sanity_check4 requires both Dex-only and MK-4-only branches")
    return dex_df, mk4_df


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "condition_summary.csv", index=False)
    (out_dir / "condition_summary.json").write_text(
        json.dumps(df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )


def export_assay_dose_curves(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(11.6, 10.2), sharex="col", constrained_layout=True)
    rows = [
        ("assay_alp_pct_vs_branch_min", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_branch_min", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_branch_min", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    for ridx, (col, title, color) in enumerate(rows):
        ax = axes[ridx, 0]
        style_axes(ax, "both")
        ax.plot(dex_df["dose"], dex_df[col], color=color, marker="o", lw=2.1, ms=5.5)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_title(title if ridx == 0 else "", pad=8)
        ax.set_ylabel("% vs branch minimum")
        if ridx == 2:
            ax.set_xlabel("Dex concentration (nM)")
        ax.text(0.02, 0.90, "Dex only", transform=ax.transAxes, fontsize=8.7, color=COLORS["muted"])

        ax = axes[ridx, 1]
        style_axes(ax, "both")
        ax.plot(mk4_df["dose"], mk4_df[col], color=color, marker="o", lw=2.1, ms=5.5)
        ax.yaxis.set_major_formatter(ScalarFormatter(useOffset=False))
        ax.set_title(title if ridx == 0 else "", pad=8)
        if ridx == 2:
            ax.set_xlabel("MK-4 concentration (uM)")
        ax.text(0.02, 0.90, "MK-4 only", transform=ax.transAxes, fontsize=8.7, color=COLORS["muted"])

    fig.suptitle("Figure 1. Single-drug dose-response curves", fontsize=15, fontweight="semibold", y=1.01)
    save_figure(fig, out_dir, "figure_1_dose_response_curves")


def export_relative_effect_heatmap(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.2, 6.6), constrained_layout=True)
    metric_cols = [
        ("assay_alp_pct_vs_branch_min", f"ALP {assay_hours['alp']} h"),
        ("assay_collagen_pct_vs_branch_min", f"Collagen {assay_hours['collagen']} h"),
        ("assay_ha_pct_vs_branch_min", f"HAp {assay_hours['ha']} h"),
        ("final_live_pct_vs_branch_min", "Final live"),
    ]
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
                ax.text(j, i, f"{values[i, j]:+.3f}", ha="center", va="center", fontsize=7.4, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=axes, fraction=0.028, pad=0.02)
    cbar.set_label("% difference vs branch minimum dose")
    fig.suptitle("Figure 2. Relative effect heatmaps", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_2_relative_effect_heatmap")


def export_local_sensitivity(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.2), constrained_layout=True)
    metric_defs = [
        ("assay_alp_pct_vs_branch_min", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_branch_min", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_branch_min", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    for ax, branch_df, title, xlabel in zip(
        axes,
        [dex_df, mk4_df],
        ["Dex local sensitivity", "MK-4 local sensitivity"],
        ["Dex concentration (nM)", "MK-4 concentration (uM)"],
    ):
        style_axes(ax, "both")
        dose = branch_df["dose"].to_numpy(dtype=float)
        for col, label, color in metric_defs:
            values = branch_df[col].to_numpy(dtype=float)
            slope = np.gradient(values, dose)
            ax.plot(dose, slope, color=color, marker="o", lw=2.0, ms=5.0, label=label)
        ax.axhline(0.0, color=COLORS["neutral"], lw=0.9)
        ax.set_title(title, pad=10)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Local slope (% per dose unit)")
        ax.legend(frameon=False, loc="best")
    fig.suptitle("Figure 3. Local sensitivity analysis", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_3_local_sensitivity")


def export_peak_summary(dex_df: pd.DataFrame, mk4_df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    metric_defs = [
        ("assay_alp", f"ALP {assay_hours['alp']} h", COLORS["alp"], "assay_alp_pct_vs_branch_min"),
        ("assay_collagen", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"], "assay_collagen_pct_vs_branch_min"),
        ("assay_ha", f"HAp {assay_hours['ha']} h", COLORS["ha"], "assay_ha_pct_vs_branch_min"),
    ]
    rows: list[dict[str, float | str]] = []
    for branch_name, branch_df in [("Dex", dex_df), ("MK-4", mk4_df)]:
        for col, label, color, pct_col in metric_defs:
            idx = int(branch_df[col].astype(float).idxmax())
            row = branch_df.loc[idx]
            rows.append(
                {
                    "branch": branch_name,
                    "metric": label,
                    "color": color,
                    "peak_value": float(row[col]),
                    "peak_dose": float(row["dose"]),
                    "dynamic_range_pct": float(branch_df[pct_col].max() - branch_df[pct_col].min()),
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
    ax.bar(x - width / 2.0, dex_rng, width=width, color=COLORS["dex"], label="Dex")
    ax.bar(x + width / 2.0, mk4_rng, width=width, color=COLORS["mk4"], label="MK-4")
    ax.set_xticks(x, [label for _, label, _, _ in metric_defs], rotation=20, ha="right")
    ax.set_ylabel("Dynamic range (% vs branch minimum)")
    ax.set_title("Dynamic range by drug branch", pad=10)
    ax.legend(frameon=False, loc="best")

    ax = axes[1]
    style_axes(ax, "y")
    branch_pos = {"Dex": 0.0, "MK-4": 1.0}
    metric_offsets = {metric_defs[0][1]: -0.18, metric_defs[1][1]: 0.0, metric_defs[2][1]: 0.18}
    for _, row in summary.iterrows():
        xpos = branch_pos[str(row["branch"])] + metric_offsets[str(row["metric"])]
        ax.scatter(
            xpos,
            float(row["peak_dose"]),
            s=90,
            color=str(row["color"]),
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.text(xpos, float(row["peak_dose"]), f"{row['peak_dose']:.4g}", ha="center", va="center", fontsize=7.5, color="white")
    ax.set_xlim(-0.45, 1.45)
    ax.set_xticks([0.0, 1.0], ["Dex", "MK-4"])
    ax.set_ylabel("Peak dose (native units)")
    ax.set_title("Peak dose by assay endpoint", pad=10)
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="white", markersize=8, label=label)
        for _, label, color, _ in metric_defs
    ]
    ax.legend(handles=legend_handles, frameon=False, loc="best")

    fig.suptitle("Figure 4. Peak-response summary", fontsize=15, fontweight="semibold", y=1.03)
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
