#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize, PowerNorm, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
LN2 = float(np.log(2.0))

JOURNAL_AXIS_LABEL_PT = 9
JOURNAL_TICK_PT = 8
JOURNAL_LEGEND_PT = 8
JOURNAL_LINE_PT = 1.2
JOURNAL_AXIS_LINE_PT = 1.0
JOURNAL_MARKER_PT = 5.5

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
        description="Figure extraction for sanity_check3 Dex/MK-4 combinatory dose runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=SCRIPT_DIR / "out" / "sanity_check3",
        help="sanity_check3 output folder containing condition manifests and per-condition outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder for extracted figures; defaults to <input>/figure_extract_3",
    )
    parser.add_argument("--alp-hours", type=int, default=240, help="hour used for ALP assay extraction")
    parser.add_argument(
        "--collagen-hours",
        type=int,
        default=336,
        help="hour used for collagen assay extraction",
    )
    parser.add_argument("--hap-hours", type=int, default=672, help="hour used for HAp assay extraction")
    return parser.parse_args()


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=320, bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=JOURNAL_AXIS_LINE_PT)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=JOURNAL_LINE_PT, alpha=0.7)


def response_cmap() -> LinearSegmentedColormap:
    return matplotlib.colormaps["cividis"]


def sequential_cmap(name: str, base: str) -> LinearSegmentedColormap:
    return matplotlib.colormaps["cividis"]


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
    def _safe_log10(value: float) -> float:
        v = float(value)
        return float(np.log10(v)) if v > 0.0 else float("nan")

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
                    "final_alp": float(final_row["alp_total"]),
                    "final_collagen": float(final_row["collagen_total"]),
                    "final_ha": float(final_row["ha_total"]),
                    "final_preob": float(final_row["n_preob"]),
                    "final_ob": float(final_row["n_osteoblast"]),
                    "final_live": float(final_row["n_total"]),
                    "final_nodule_count": float(final_row["nodule_count_ever"])
                    if "nodule_count_ever" in final_row
                    else np.nan,
                }
            )
        run_df = pd.DataFrame(metrics_by_run)
        rows.append(
            {
                "condition_index": int(row["index"]),
                "slug": str(row["slug"]),
                "out_dir": str(out_dir),
                "dex_conc_nM": float(row["dex_conc_nM"]),
                "mk4_conc_uM": float(row["mk4_conc_uM"]),
                "log10_dex_nM": _safe_log10(float(row["dex_conc_nM"])),
                "log10_mk4_uM": _safe_log10(float(row["mk4_conc_uM"])),
                "n_replicates": int(len(run_df)),
                "assay_alp": float(run_df["assay_alp"].mean()),
                "assay_collagen": float(run_df["assay_collagen"].mean()),
                "assay_ha": float(run_df["assay_ha"].mean()),
                "final_alp": float(run_df["final_alp"].mean()),
                "final_collagen": float(run_df["final_collagen"].mean()),
                "final_ha": float(run_df["final_ha"].mean()),
                "final_preob": float(run_df["final_preob"].mean()),
                "final_ob": float(run_df["final_ob"].mean()),
                "final_live": float(run_df["final_live"].mean()),
                "final_nodule_count": float(run_df["final_nodule_count"].mean()),
                "assay_alp_sd": float(run_df["assay_alp"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "assay_collagen_sd": float(run_df["assay_collagen"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "assay_ha_sd": float(run_df["assay_ha"].std(ddof=1)) if len(run_df) > 1 else 0.0,
                "final_live_sd": float(run_df["final_live"].std(ddof=1)) if len(run_df) > 1 else 0.0,
            }
        )
    df = pd.DataFrame(rows).sort_values("condition_index").reset_index(drop=True)
    for col in [
        "assay_alp",
        "assay_collagen",
        "assay_ha",
        "final_alp",
        "final_collagen",
        "final_ha",
        "final_live",
        "final_preob",
        "final_ob",
    ]:
        control_mask = (df["dex_conc_nM"] == 0.0) & (df["mk4_conc_uM"] == 0.0)
        if bool(control_mask.any()):
            baseline = float(df.loc[control_mask, col].iloc[0])
        else:
            baseline = float(df[col].min())
        df[f"{col}_pct_vs_control"] = 100.0 * (df[col] - baseline) / max(baseline, 1e-12)
    for col, sd_col in [
        ("assay_alp", "assay_alp_sd"),
        ("assay_collagen", "assay_collagen_sd"),
        ("assay_ha", "assay_ha_sd"),
        ("final_live", "final_live_sd"),
    ]:
        control_mask = (df["dex_conc_nM"] == 0.0) & (df["mk4_conc_uM"] == 0.0)
        if bool(control_mask.any()):
            baseline = float(df.loc[control_mask, col].iloc[0])
        else:
            baseline = float(df[col].min())
        df[f"{col}_pct_sd_vs_control"] = 100.0 * df[sd_col].astype(float) / max(baseline, 1e-12)
    for col, sd_col in [
        ("assay_alp", "assay_alp_sd"),
        ("assay_collagen", "assay_collagen_sd"),
        ("assay_ha", "assay_ha_sd"),
    ]:
        control_mask = (df["dex_conc_nM"] == 0.0) & (df["mk4_conc_uM"] == 0.0)
        if bool(control_mask.any()):
            baseline = float(df.loc[control_mask, col].iloc[0])
        else:
            baseline = float(df[col].min())
        fc = df[col].astype(float) / max(baseline, 1e-12)
        df[f"{col}_fc_vs_control"] = fc
        df[f"{col}_log2fc_vs_control"] = np.log2(np.clip(fc.to_numpy(dtype=float), 1e-12, None))
        mean_safe = np.clip(df[col].astype(float).to_numpy(dtype=float), 1e-12, None)
        # Delta-method SD for log2(x / baseline): sd_log2 ~= sd_x / (x * ln(2))
        df[f"{col}_log2fc_sd_vs_control"] = df[sd_col].astype(float).to_numpy(dtype=float) / (mean_safe * LN2)
    return df


def pivot_metric(df: pd.DataFrame, value_col: str) -> pd.DataFrame:
    table = df.pivot(index="mk4_conc_uM", columns="dex_conc_nM", values=value_col)
    return table.sort_index().sort_index(axis=1)


def heatmap_with_labels(
    ax: plt.Axes,
    matrix: np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    *,
    cmap,
    norm,
    text_fmt: str,
) -> any:
    im = ax.imshow(matrix, aspect="auto", cmap=cmap, norm=norm, origin="lower")
    ax.set_xticks(np.arange(len(col_labels)), col_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            ax.text(j, i, text_fmt.format(val), ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
    return im


def export_assay_triptych(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    metrics = [
        ("assay_alp", f"ALP {assay_hours['alp']} h", sequential_cmap("alp", COLORS["alp"])),
        (
            "assay_collagen",
            f"Collagen {assay_hours['collagen']} h",
            sequential_cmap("collagen", COLORS["collagen"]),
        ),
        ("assay_ha", f"HAp {assay_hours['ha']} h", sequential_cmap("ha", COLORS["ha"])),
    ]
    dex_vals = np.array(sorted(df["dex_conc_nM"].unique()), dtype=float)
    mk4_vals = np.array(sorted(df["mk4_conc_uM"].unique()), dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 5.2), constrained_layout=True)
    for ax, (metric, title, cmap) in zip(axes, metrics):
        table = (
            df.pivot(index="mk4_conc_uM", columns="dex_conc_nM", values=metric)
            .reindex(index=mk4_vals, columns=dex_vals)
        )
        matrix = table.to_numpy(dtype=float)
        vals = matrix[np.isfinite(matrix)]
        norm = Normalize(vmin=float(np.min(vals)), vmax=float(np.max(vals)))
        im = ax.imshow(np.ma.array(matrix, mask=~np.isfinite(matrix)), aspect="auto", cmap=cmap, norm=norm, origin="lower")
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                val = matrix[i, j]
                if np.isfinite(val):
                    ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=8.0, color=COLORS["ink"])
                else:
                    ax.text(j, i, "-", ha="center", va="center", fontsize=8.0, color=COLORS["muted"])
        ax.set_xticks(np.arange(len(dex_vals)), [f"{v:.4g}" for v in dex_vals])
        ax.set_yticks(np.arange(len(mk4_vals)), [f"{v:.4g}" for v in mk4_vals])
        ax.set_title(title, pad=10)
        ax.set_xlabel("Dex (nM)")
        ax.set_ylabel("MK-4 (uM)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Absolute response (a.u.)", fontsize=9)

    fig.suptitle(
        "Assay triptych across the Dex/MK-4 dose grid",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
    )
    save_figure(fig, out_dir, "figure_1_assay_triptych")


def export_response_map(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    x = df["assay_collagen_pct_vs_control"].to_numpy(dtype=float)
    y = df["assay_ha_pct_vs_control"].to_numpy(dtype=float)
    c = df["assay_alp_pct_vs_control"].to_numpy(dtype=float)
    size = np.full_like(x, 130.0, dtype=float)
    vmax = max(1.0, float(np.max(np.abs(c))))

    fig, ax = plt.subplots(figsize=(8.6, 6.6))
    sc = ax.scatter(
        x,
        y,
        c=c,
        s=size,
        cmap=response_cmap(),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
        edgecolors="white",
        linewidths=0.8,
        alpha=0.96,
    )
    radial = np.hypot(x, y)
    label_idx = set(np.argsort(radial)[-5:].tolist())
    control_hits = np.where((df["dex_conc_nM"].to_numpy(dtype=float) == 0.0) & (df["mk4_conc_uM"].to_numpy(dtype=float) == 0.0))[0]
    if control_hits.size > 0:
        label_idx.add(int(control_hits[0]))
    xspan = max(float(np.max(x) - np.min(x)), 1e-12)
    yspan = max(float(np.max(y) - np.min(y)), 1e-12)
    placed: list[tuple[float, float]] = []
    for idx, row in df.reset_index(drop=True).iterrows():
        if idx not in label_idx:
            continue
        dx = (0.018 * xspan) * (1.0 if float(row["assay_collagen_pct_vs_control"]) >= 0 else -1.0)
        dy = (0.022 * yspan) * (1.0 if float(row["assay_ha_pct_vs_control"]) >= 0 else -1.0)
        tx = float(row["assay_collagen_pct_vs_control"]) + dx
        ty = float(row["assay_ha_pct_vs_control"]) + dy
        if any((tx - px) ** 2 + (ty - py) ** 2 < (0.045 * xspan) ** 2 for px, py in placed):
            continue
        ax.text(
            tx,
            ty,
            f"C{int(row['condition_index']):02d}",
            fontsize=7.8,
            color=COLORS["ink"],
            bbox={"boxstyle": "round,pad=0.12", "fc": (1, 1, 1, 0.68), "ec": "none"},
        )
        placed.append((tx, ty))
    style_axes(ax, "both")
    ax.axhline(0.0, color=COLORS["neutral"], lw=1.0)
    ax.axvline(0.0, color=COLORS["neutral"], lw=1.0)
    ax.set_xlabel(f"Collagen {assay_hours['collagen']} h vs control (%)")
    ax.set_ylabel(f"HAp {assay_hours['ha']} h vs control (%)")
    ax.set_title("Dex/MK-4 response map", pad=14)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(f"ALP {assay_hours['alp']} h vs control (%)")
    save_figure(fig, out_dir, "figure_2_ecm_response_map")


def export_overview_heatmap(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    metrics = [
        ("assay_alp_log2fc_vs_control", f"ALP {assay_hours['alp']} h"),
        ("assay_collagen_log2fc_vs_control", f"Collagen {assay_hours['collagen']} h"),
        ("assay_ha_log2fc_vs_control", f"Mineral {assay_hours['ha']} h"),
    ]
    metric_sd_cols = [
        "assay_alp_log2fc_sd_vs_control",
        "assay_collagen_log2fc_sd_vs_control",
        "assay_ha_log2fc_sd_vs_control",
    ]
    # Rank by HAp response and keep only the top 10 conditions.
    ranked = df.copy()
    ranked_sorted = (
        ranked.sort_values(
            ["assay_ha_log2fc_vs_control", "assay_collagen_log2fc_vs_control", "assay_alp_log2fc_vs_control"],
            ascending=[False, False, False],
        )
        .reset_index(drop=True)
    )
    ordered = ranked_sorted.head(10).copy()
    # Always keep baseline condition (Dex=0, MK-4=0) in the displayed top-10 table.
    baseline = ranked[(ranked["dex_conc_nM"] == 0.0) & (ranked["mk4_conc_uM"] == 0.0)]
    if not baseline.empty:
        baseline_row = baseline.iloc[[0]].copy()
        baseline_cid = int(baseline_row.iloc[0]["condition_index"])
        already_in = bool((ordered["condition_index"] == baseline_cid).any())
        if not already_in:
            ordered = pd.concat([ordered.iloc[:-1], baseline_row], ignore_index=True)

    ordered.to_csv(out_dir / "figure_3_top10_conditions.csv", index=False)

    values = ordered[[m[0] for m in metrics]].to_numpy(dtype=float)
    values_sd = ordered[metric_sd_cols].to_numpy(dtype=float)
    # Improve color discrimination among high positive conditions by
    # excluding the baseline row from normalization and applying power scaling.
    control_mask = (ordered["dex_conc_nM"].to_numpy(dtype=float) == 0.0) & (
        ordered["mk4_conc_uM"].to_numpy(dtype=float) == 0.0
    )
    color_values = values.copy()
    if np.any(control_mask):
        color_values[control_mask, :] = np.nan
    finite_vals = color_values[np.isfinite(color_values)]
    if finite_vals.size:
        vmin = float(np.nanpercentile(finite_vals, 5.0))
        vmax = float(np.nanpercentile(finite_vals, 98.0))
    else:
        vmin, vmax = 0.0, 1.0
    if vmax <= vmin:
        vmax = vmin + 1.0

    # Muted sequential scale (PLOS-style): low contrast near baseline, restrained high-end saturation.
    plos_cmap = LinearSegmentedColormap.from_list(
        "plos_muted_warm",
        ["#faf7f5", "#f2ddd5", "#e6bfb3", "#d69484", "#bc6754", "#8e3326"],
        N=256,
    )
    plos_cmap.set_bad("#ececec")
    norm = PowerNorm(gamma=1.55, vmin=vmin, vmax=vmax)

    row_labels = [
        f"D={row.dex_conc_nM:.4g} nM | K={row.mk4_conc_uM:.4g} uM"
        for row in ordered.itertuples()
    ]

    # Smarter split layout: dedicated label panel + wide heatmap panel.
    fig = plt.figure(figsize=(10.2, 7.1))
    gs = gridspec.GridSpec(1, 2, width_ratios=[0.72, 2.32], wspace=0.004)
    ax_labels = fig.add_subplot(gs[0, 0])
    ax = fig.add_subplot(gs[0, 1])

    im = ax.imshow(np.ma.array(color_values, mask=~np.isfinite(color_values)), aspect="auto", cmap=plos_cmap, norm=norm)
    ax.set_xticks(
        np.arange(len(metrics)),
        [
            f"ALP\n{assay_hours['alp']} h",
            f"Collagen\n{assay_hours['collagen']} h",
            f"Mineral\n{assay_hours['ha']} h",
        ],
    )
    ax.tick_params(axis="x", pad=9, labelsize=13.2)
    ax.set_yticks([])

    # Subtle cell boundaries improve scanning in dense matrices.
    ax.set_xticks(np.arange(-0.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, values.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#ffffff", linewidth=0.9, alpha=0.75)
    ax.tick_params(which="minor", bottom=False, left=False)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(
                j,
                i,
                f"{values[i, j]:+.2f} ± {values_sd[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=10.6,
                color=COLORS["ink"],
            )

    # Long condition labels in a separate axis keep columns wide and readable.
    ax_labels.set_xlim(0.0, 1.0)
    ax_labels.set_ylim(values.shape[0] - 0.5, -0.5)
    ax_labels.axis("off")
    for i, lbl in enumerate(row_labels):
        ax_labels.text(
            0.99,
            i,
            lbl,
            ha="right",
            va="center",
            fontsize=13.6,
            color=COLORS["ink"],
            clip_on=False,
        )

    ax.set_title("Top-10 dose response heatmap", pad=14, fontsize=17.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.038, pad=0.035)
    cbar.set_label("log2 fold change vs control", fontsize=13.2)
    cbar.ax.tick_params(labelsize=12.0)
    fig.subplots_adjust(left=0.01, right=0.965, bottom=0.13, top=0.90)
    save_figure(fig, out_dir, "figure_3_ecm_overview_heatmap")


def export_fingerprint(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    ordered = df.sort_values(["assay_ha_pct_vs_control", "assay_collagen_pct_vs_control"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 8.8), sharey=True, constrained_layout=True)
    cols = [
        ("assay_alp_pct_vs_control", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_control", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_control", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    err_cols = {
        "assay_alp_pct_vs_control": "assay_alp_pct_sd_vs_control",
        "assay_collagen_pct_vs_control": "assay_collagen_pct_sd_vs_control",
        "assay_ha_pct_vs_control": "assay_ha_pct_sd_vs_control",
    }
    labels = [
        f"C{int(row.condition_index):02d} | D={row.dex_conc_nM:.4g} | K={row.mk4_conc_uM:.4g}"
        for row in ordered.itertuples()
    ]

    for ax, (col, title, color) in zip(axes, cols):
        vals = ordered[col].to_numpy(dtype=float)
        errs = ordered[err_cols[col]].to_numpy(dtype=float)
        ax.axvline(0.0, color=COLORS["neutral"], lw=0.9)
        ax.hlines(y, 0.0, vals, color=color, lw=2.0, alpha=0.9)
        ax.scatter(vals, y, s=42, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        ax.errorbar(vals, y, xerr=errs, fmt="none", ecolor=COLORS["muted"], elinewidth=0.8, alpha=0.75, capsize=2, zorder=2)
        style_axes(ax, "x")
        ax.set_title(title, pad=10)
        ax.set_xlabel("% vs control")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=8.2)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle("ECM fingerprint across Dex/MK-4 conditions", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_4_ecm_fingerprint")


def export_geometry_reference(df: pd.DataFrame, out_dir: Path) -> None:
    dex_vals = np.sort(df.loc[df["dex_conc_nM"] > 0.0, "dex_conc_nM"].unique())
    mk4_vals = np.sort(df.loc[df["mk4_conc_uM"] > 0.0, "mk4_conc_uM"].unique())
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for _, row in df.iterrows():
        dex = float(row["dex_conc_nM"])
        mk4 = float(row["mk4_conc_uM"])
        if dex <= 0.0 or mk4 <= 0.0:
            continue
        dex_log = np.log10(dex)
        mk4_log = np.log10(mk4)
        ax.scatter(
            dex_log,
            mk4_log,
            s=95,
            color=COLORS["neutral"],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.text(
            dex_log + 0.015,
            mk4_log + 0.015,
            f"C{int(row['condition_index']):02d}",
            fontsize=8.2,
            color=COLORS["ink"],
        )
    style_axes(ax, "both")
    if dex_vals.size > 0:
        ax.set_xticks(np.log10(dex_vals), [f"{v:.4g}" for v in dex_vals])
    if mk4_vals.size > 0:
        ax.set_yticks(np.log10(mk4_vals), [f"{v:.4g}" for v in mk4_vals])
    ax.set_xlabel("Dex concentration (nM, log10-spaced)")
    ax.set_ylabel("MK-4 concentration (uM, log10-spaced)")
    ax.set_title("Dose grid reference", pad=12)
    save_figure(fig, out_dir, "figure_5_geometry_reference")


def export_hap_pairwise_heatmap(df: pd.DataFrame, out_dir: Path) -> None:
    ordered = df.sort_values(["assay_ha", "condition_index"], ascending=[False, True]).reset_index(drop=True)
    hap = ordered["assay_ha"].to_numpy(dtype=float)
    mean_pair = np.maximum((hap[:, None] + hap[None, :]) * 0.5, 1e-12)
    pairwise = 100.0 * (hap[:, None] - hap[None, :]) / mean_pair
    mask = np.tril(np.ones_like(pairwise, dtype=bool), k=-1)
    pairwise_masked = np.ma.array(pairwise, mask=mask)
    vmax = max(1.0, float(np.max(np.abs(pairwise))))

    fig, ax = plt.subplots(figsize=(9.4, 8.0))
    im = ax.imshow(
        pairwise_masked,
        aspect="auto",
        cmap=response_cmap(),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    labels = [
        f"C{int(row.condition_index):02d}"
        for row in ordered.itertuples()
    ]
    ax.set_xticks(np.arange(len(labels)), labels, rotation=90)
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xlabel("Reference condition")
    ax.set_ylabel("Compared condition")
    ax.set_title("HAp pairwise heatmap", pad=12)
    for i in range(pairwise.shape[0]):
        for j in range(pairwise.shape[1]):
            if mask[i, j]:
                continue
            if i == j:
                ax.text(j, i, "0", ha="center", va="center", fontsize=6.8, color=COLORS["ink"])
            elif abs(pairwise[i, j]) >= 8.0:
                ax.text(j, i, f"{pairwise[i, j]:+.0f}", ha="center", va="center", fontsize=6.4, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.03)
    cbar.set_label("Pairwise HAp difference (%)")
    save_figure(fig, out_dir, "figure_6_hap_pairwise_heatmap")


def export_dose_response_ranges(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    unique_dex = np.array(sorted(df["dex_conc_nM"].unique()), dtype=float)
    unique_mk4 = np.array(sorted(df["mk4_conc_uM"].unique()), dtype=float)

    dex_summary = (
        df.groupby("dex_conc_nM", sort=True)
        .agg(
            assay_alp_mean=("assay_alp", "mean"),
            assay_alp_sd=("assay_alp", "std"),
            assay_collagen_mean=("assay_collagen", "mean"),
            assay_collagen_sd=("assay_collagen", "std"),
            assay_ha_mean=("assay_ha", "mean"),
            assay_ha_sd=("assay_ha", "std"),
        )
        .fillna(0.0)
        .reset_index()
    )
    mk4_summary = (
        df.groupby("mk4_conc_uM", sort=True)
        .agg(
            assay_alp_mean=("assay_alp", "mean"),
            assay_alp_sd=("assay_alp", "std"),
            assay_collagen_mean=("assay_collagen", "mean"),
            assay_collagen_sd=("assay_collagen", "std"),
            assay_ha_mean=("assay_ha", "mean"),
            assay_ha_sd=("assay_ha", "std"),
        )
        .fillna(0.0)
        .reset_index()
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(12.6, 8.4),
        gridspec_kw={"height_ratios": [0.8, 2.4]},
        constrained_layout=True,
    )

    ax = axes[0, 0]
    dex_pos = np.arange(len(unique_dex), dtype=float)
    ax.scatter(dex_pos, np.ones_like(dex_pos), s=84, color=COLORS["dex"], edgecolors="white", linewidths=0.8)
    ax.set_xticks(dex_pos, [f"{v:.4g}" for v in unique_dex])
    ax.set_yticks([])
    ax.set_ylim(0.92, 1.08)
    ax.set_xlim(-0.5, len(unique_dex) - 0.5)
    ax.set_xlabel("Dex concentrations sampled (nM)")
    ax.set_title("Dex range from condition_manifest", pad=8)
    style_axes(ax, "x")

    ax = axes[0, 1]
    mk4_pos = np.arange(len(unique_mk4), dtype=float)
    ax.scatter(mk4_pos, np.ones_like(mk4_pos), s=84, color=COLORS["mk4"], edgecolors="white", linewidths=0.8)
    ax.set_xticks(mk4_pos, [f"{v:.4g}" for v in unique_mk4])
    ax.set_yticks([])
    ax.set_ylim(0.92, 1.08)
    ax.set_xlim(-0.5, len(unique_mk4) - 0.5)
    ax.set_xlabel("MK-4 concentrations sampled (uM)")
    ax.set_title("MK-4 range from condition_manifest", pad=8)
    style_axes(ax, "x")

    ax = axes[1, 0]
    style_axes(ax, "both")
    ax.errorbar(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_alp_mean"],
        yerr=dex_summary["assay_alp_sd"],
        color=COLORS["alp"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"ALP {assay_hours['alp']} h",
    )
    ax.errorbar(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_collagen_mean"],
        yerr=dex_summary["assay_collagen_sd"],
        color=COLORS["collagen"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"Collagen {assay_hours['collagen']} h",
    )
    ax.errorbar(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_ha_mean"],
        yerr=dex_summary["assay_ha_sd"],
        color=COLORS["ha"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"HAp {assay_hours['ha']} h",
    )
    ax.set_xlabel("Dex concentration (nM)")
    ax.set_ylabel("Mean assay response (a.u.)")
    ax.set_title("Dex dose response averaged across sampled MK-4 values", pad=10)
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    style_axes(ax, "both")
    ax.errorbar(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_alp_mean"],
        yerr=mk4_summary["assay_alp_sd"],
        color=COLORS["alp"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"ALP {assay_hours['alp']} h",
    )
    ax.errorbar(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_collagen_mean"],
        yerr=mk4_summary["assay_collagen_sd"],
        color=COLORS["collagen"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"Collagen {assay_hours['collagen']} h",
    )
    ax.errorbar(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_ha_mean"],
        yerr=mk4_summary["assay_ha_sd"],
        color=COLORS["ha"],
        marker="o",
        lw=2.0,
        capsize=2,
        label=f"HAp {assay_hours['ha']} h",
    )
    ax.set_xlabel("MK-4 concentration (uM)")
    ax.set_ylabel("Mean assay response (a.u.)")
    ax.set_title("MK-4 dose response averaged across sampled Dex values", pad=10)
    ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Dose ranges and marginal dose-response trends",
        fontsize=15,
        fontweight="semibold",
        y=1.01,
    )
    save_figure(fig, out_dir, "figure_7_dose_response_ranges")


def export_mineralization_dose_plots(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    dose_df = (
        df[["condition_index", "dex_conc_nM", "mk4_conc_uM", "assay_ha_log2fc_vs_control"]]
        .dropna(subset=["assay_ha_log2fc_vs_control"])
        .copy()
    )
    dose_df = dose_df[
        (dose_df["dex_conc_nM"] >= 0.0)
        & (dose_df["dex_conc_nM"] <= 100.0)
        & (dose_df["mk4_conc_uM"] >= 0.0)
        & (dose_df["mk4_conc_uM"] <= 10.0)
    ].copy()
    if dose_df.empty:
        return

    dex_vals = np.sort(dose_df["dex_conc_nM"].unique())
    mk4_vals = np.sort(dose_df["mk4_conc_uM"].unique())
    grid = (
        dose_df.pivot(index="mk4_conc_uM", columns="dex_conc_nM", values="assay_ha_log2fc_vs_control")
        .reindex(index=mk4_vals, columns=dex_vals)
        .to_numpy(dtype=float)
    )
    masked_grid = np.ma.array(grid, mask=~np.isfinite(grid))
    finite_vals = masked_grid.compressed()
    if finite_vals.size == 0:
        return
    vmax = max(0.25, float(np.max(np.abs(finite_vals))))
    norm = TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax)

    # Figure 8A: mineralization dose map on explicit X/Y dose axes.
    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    im = ax.imshow(
        masked_grid,
        origin="lower",
        aspect="auto",
        cmap="viridis",
        norm=norm,
    )
    ax.set_xticks(np.arange(len(dex_vals)), [f"{v:.4g}" for v in dex_vals])
    ax.set_yticks(np.arange(len(mk4_vals)), [f"{v:.4g}" for v in mk4_vals])
    ax.set_xlabel("Dex concentration (nM)")
    ax.set_ylabel("MK-4 concentration (uM)")
    ax.set_title(f"Mineralization log2FC dose map ({assay_hours['ha']} h HAp)", pad=12)
    ax.set_xlim(-0.5, len(dex_vals) - 0.5)
    ax.set_ylim(-0.5, len(mk4_vals) - 0.5)
    style_axes(ax, "both")
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Mineralization response (log2FC vs control)")
    save_figure(fig, out_dir, "figure_8_mineralization_dose_xy")

    # Figure 8B: contour map with 20 highlighted points for readability.
    x = dose_df["dex_conc_nM"].to_numpy(dtype=float)
    y = dose_df["mk4_conc_uM"].to_numpy(dtype=float)
    z = dose_df["assay_ha_log2fc_vs_control"].to_numpy(dtype=float)
    order = np.lexsort((dose_df["mk4_conc_uM"].to_numpy(dtype=float), dose_df["dex_conc_nM"].to_numpy(dtype=float)))
    x_sorted = x[order]
    y_sorted = y[order]
    z_sorted = z[order]
    points_n = min(20, len(z_sorted))
    mark_idx = np.linspace(0, len(z_sorted) - 1, num=points_n, dtype=int)

    fig, ax = plt.subplots(figsize=(8.8, 6.8))
    tric = ax.tricontourf(x_sorted, y_sorted, z_sorted, levels=14, cmap="viridis", norm=norm)
    ax.tricontour(x_sorted, y_sorted, z_sorted, levels=8, colors="#334155", linewidths=0.8, alpha=0.65)
    ax.scatter(
        x_sorted[mark_idx],
        y_sorted[mark_idx],
        s=50,
        c="white",
        edgecolors="#0f172a",
        linewidths=0.9,
        zorder=3,
    )
    style_axes(ax, "both")
    ax.set_xticks(dex_vals, [f"{v:.4g}" for v in dex_vals])
    ax.set_yticks(mk4_vals, [f"{v:.4g}" for v in mk4_vals])
    ax.set_xlim(0.0, 100.0)
    ax.set_ylim(0.0, 10.0)
    ax.set_xlabel("Dex concentration (nM)")
    ax.set_ylabel("MK-4 concentration (uM)")
    ax.set_title("Mineralization contour with 20 marked dose points", pad=12)
    cbar = fig.colorbar(tric, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Mineralization response (log2FC vs control)")
    save_figure(fig, out_dir, "figure_8_mineralization_contour_20pts")


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "condition_summary.csv", index=False)
    summary_json = []
    for rec in df.to_dict(orient="records"):
        clean = {}
        for key, value in rec.items():
            if isinstance(value, (float, np.floating)) and (not np.isfinite(float(value))):
                clean[key] = None
            else:
                clean[key] = value
        summary_json.append(clean)
    (out_dir / "condition_summary.json").write_text(
        json.dumps(summary_json, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    out_dir = args.out if args.out is not None else args.input / "figure_extract_3"
    assay_hours = {
        "alp": int(args.alp_hours),
        "collagen": int(args.collagen_hours),
        "ha": int(args.hap_hours),
    }

    df = build_summary(args.input, assay_hours)
    write_tables(df, out_dir)

    export_assay_triptych(df, out_dir, assay_hours)
    export_response_map(df, out_dir, assay_hours)
    export_overview_heatmap(df, out_dir, assay_hours)
    export_fingerprint(df, out_dir, assay_hours)
    export_geometry_reference(df, out_dir)
    export_hap_pairwise_heatmap(df, out_dir)
    export_dose_response_ranges(df, out_dir, assay_hours)
    export_mineralization_dose_plots(df, out_dir, assay_hours)

    print(f"[sanity3_figure_extract] processed {len(df)} conditions")
    print(f"[sanity3_figure_extract] figures written to {out_dir}")


if __name__ == "__main__":
    main()
