#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, TwoSlopeNorm
from matplotlib.cm import ScalarMappable

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
        description="Figure extraction for sanity_check3 Dex/MK-4 combinatory dose runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("out") / "sanity_check3",
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
    fig.savefig(out_dir / f"{stem}.png", dpi=320, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=320, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y") -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.7)


def response_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list("dose_div", ["#2b6c8f", "#f8f5ef", "#b44d2a"])


def sequential_cmap(name: str, base: str) -> LinearSegmentedColormap:
    if name == "alp":
        return LinearSegmentedColormap.from_list(name, ["#fff4ed", base])
    if name == "collagen":
        return LinearSegmentedColormap.from_list(name, ["#effcf5", base])
    return LinearSegmentedColormap.from_list(name, ["#eef4ff", base])


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
        rows.append(
            {
                "condition_index": int(row["index"]),
                "slug": str(row["slug"]),
                "out_dir": str(out_dir),
                "dex_conc_nM": float(row["dex_conc_nM"]),
                "mk4_conc_uM": float(row["mk4_conc_uM"]),
                "log10_dex_nM": float(np.log10(float(row["dex_conc_nM"]))),
                "log10_mk4_uM": float(np.log10(float(row["mk4_conc_uM"]))),
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
        baseline = float(df[col].min())
        df[f"{col}_pct_vs_min"] = 100.0 * (df[col] - baseline) / max(baseline, 1e-12)
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


def sparse_dose_grid(
    ax: plt.Axes,
    df: pd.DataFrame,
    value_col: str,
    *,
    cmap,
    norm,
    text_fmt: str,
) -> any:
    dex_vals = np.array(sorted(df["dex_conc_nM"].unique()), dtype=float)
    mk4_vals = np.array(sorted(df["mk4_conc_uM"].unique()), dtype=float)
    points = df[["dex_conc_nM", "mk4_conc_uM", value_col]].copy()

    ax.set_facecolor("white")
    for x in dex_vals:
        ax.axvline(x, color=COLORS["grid"], lw=0.8, alpha=0.6, zorder=0)
    for y in mk4_vals:
        ax.axhline(y, color=COLORS["grid"], lw=0.8, alpha=0.6, zorder=0)

    sc = ax.scatter(
        points["dex_conc_nM"],
        points["mk4_conc_uM"],
        c=points[value_col],
        cmap=cmap,
        norm=norm,
        s=880,
        marker="s",
        edgecolors="white",
        linewidths=1.2,
        zorder=3,
    )
    for row in points.itertuples():
        ax.text(
            float(row.dex_conc_nM),
            float(row.mk4_conc_uM),
            text_fmt.format(float(getattr(row, value_col))),
            ha="center",
            va="center",
            fontsize=8.6,
            color=COLORS["ink"],
            zorder=4,
        )

    ax.set_xticks(dex_vals, [f"{v:.4g}" for v in dex_vals])
    ax.set_yticks(mk4_vals, [f"{v:.4g}" for v in mk4_vals])
    ax.set_xlim(float(dex_vals.min()) - 5.0, float(dex_vals.max()) + 5.0)
    ax.set_ylim(float(mk4_vals.min()) - 0.5, float(mk4_vals.max()) + 0.5)
    return sc


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
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 4.8), constrained_layout=True)
    for ax, (metric, title, cmap) in zip(axes, metrics):
        vals = df[metric].to_numpy(dtype=float)
        norm = Normalize(vmin=float(np.min(vals)), vmax=float(np.max(vals)))
        im = sparse_dose_grid(
            ax,
            df,
            metric,
            cmap=cmap,
            norm=norm,
            text_fmt="{:.1f}",
        )
        ax.set_title(title, pad=10)
        ax.set_xlabel("Dex (nM)")
        ax.set_ylabel("MK-4 (uM)")
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label("Absolute response (a.u.)", fontsize=9)

    fig.suptitle(
        "Figure 1. Assay triptych across the Dex/MK-4 dose grid",
        fontsize=15,
        fontweight="semibold",
        y=1.02,
    )
    save_figure(fig, out_dir, "figure_1_assay_triptych")


def export_response_map(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    x = df["assay_collagen_pct_vs_min"].to_numpy(dtype=float)
    y = df["assay_ha_pct_vs_min"].to_numpy(dtype=float)
    c = df["assay_alp_pct_vs_min"].to_numpy(dtype=float)
    size = 70.0 + 170.0 * (
        (df["dex_conc_nM"].to_numpy(dtype=float) - df["dex_conc_nM"].min())
        / max(float(df["dex_conc_nM"].max() - df["dex_conc_nM"].min()), 1e-12)
    )
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
    for _, row in df.iterrows():
        ax.text(
            float(row["assay_collagen_pct_vs_min"]) + 0.7,
            float(row["assay_ha_pct_vs_min"]) + 0.7,
            f"C{int(row['condition_index']):02d}",
            fontsize=7.8,
            color=COLORS["ink"],
        )
    style_axes(ax, "both")
    ax.axhline(0.0, color=COLORS["neutral"], lw=1.0)
    ax.axvline(0.0, color=COLORS["neutral"], lw=1.0)
    ax.set_xlabel(f"Collagen {assay_hours['collagen']} h vs minimum condition (%)")
    ax.set_ylabel(f"HAp {assay_hours['ha']} h vs minimum condition (%)")
    ax.set_title("Figure 2. Dex/MK-4 response map", pad=14)
    cbar = fig.colorbar(sc, ax=ax, fraction=0.04, pad=0.03)
    cbar.set_label(f"ALP {assay_hours['alp']} h vs minimum condition (%)")
    fig.text(
        0.125,
        0.94,
        "Point size increases with Dex concentration. Point color encodes the ALP assay response.",
        fontsize=9.1,
        color=COLORS["muted"],
        ha="left",
    )
    save_figure(fig, out_dir, "figure_2_ecm_response_map")


def export_overview_heatmap(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    metrics = [
        ("assay_alp_pct_vs_min", f"ALP {assay_hours['alp']} h"),
        ("assay_collagen_pct_vs_min", f"Collagen {assay_hours['collagen']} h"),
        ("assay_ha_pct_vs_min", f"HAp {assay_hours['ha']} h"),
        ("final_live_pct_vs_min", "Final live"),
    ]
    ordered = df.sort_values(["mk4_conc_uM", "dex_conc_nM"]).reset_index(drop=True)
    values = ordered[[m[0] for m in metrics]].to_numpy(dtype=float)
    vmax = max(1.0, float(np.max(np.abs(values))))

    fig, ax = plt.subplots(figsize=(8.8, 8.4))
    im = ax.imshow(
        values,
        aspect="auto",
        cmap=response_cmap(),
        norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax),
    )
    ax.set_xticks(np.arange(len(metrics)), [m[1] for m in metrics], rotation=25, ha="right")
    row_labels = [
        f"C{int(row.condition_index):02d} | D={row.dex_conc_nM:.4g} nM | K={row.mk4_conc_uM:.4g} uM"
        for row in ordered.itertuples()
    ]
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="y", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            ax.text(j, i, f"{values[i, j]:+.1f}", ha="center", va="center", fontsize=7.6, color=COLORS["ink"])
    ax.set_title("Figure 3. Dose overview heatmap", pad=12)
    cbar = fig.colorbar(im, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("% difference vs minimum condition")
    save_figure(fig, out_dir, "figure_3_ecm_overview_heatmap")


def export_fingerprint(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    ordered = df.sort_values(["assay_ha_pct_vs_min", "assay_collagen_pct_vs_min"], ascending=[False, False]).reset_index(drop=True)
    y = np.arange(len(ordered))

    fig, axes = plt.subplots(1, 3, figsize=(15.2, 8.8), sharey=True, constrained_layout=True)
    cols = [
        ("assay_alp_pct_vs_min", f"ALP {assay_hours['alp']} h", COLORS["alp"]),
        ("assay_collagen_pct_vs_min", f"Collagen {assay_hours['collagen']} h", COLORS["collagen"]),
        ("assay_ha_pct_vs_min", f"HAp {assay_hours['ha']} h", COLORS["ha"]),
    ]
    labels = [
        f"C{int(row.condition_index):02d} | D={row.dex_conc_nM:.4g} | K={row.mk4_conc_uM:.4g}"
        for row in ordered.itertuples()
    ]

    for ax, (col, title, color) in zip(axes, cols):
        vals = ordered[col].to_numpy(dtype=float)
        ax.axvline(0.0, color=COLORS["neutral"], lw=0.9)
        ax.hlines(y, 0.0, vals, color=color, lw=2.0, alpha=0.9)
        ax.scatter(vals, y, s=42, color=color, edgecolors="white", linewidths=0.6, zorder=3)
        style_axes(ax, "x")
        ax.set_title(title, pad=10)
        ax.set_xlabel("% vs minimum")
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels(labels, fontsize=8.2)
        else:
            ax.tick_params(axis="y", labelleft=False)
    axes[0].invert_yaxis()
    fig.suptitle("Figure 4. ECM fingerprint across Dex/MK-4 conditions", fontsize=15, fontweight="semibold", y=1.02)
    save_figure(fig, out_dir, "figure_4_ecm_fingerprint")


def export_geometry_reference(df: pd.DataFrame, out_dir: Path) -> None:
    dex_vals = np.sort(df["dex_conc_nM"].unique())
    mk4_vals = np.sort(df["mk4_conc_uM"].unique())
    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for _, row in df.iterrows():
        ax.scatter(
            np.log10(float(row["dex_conc_nM"])),
            np.log10(float(row["mk4_conc_uM"])),
            s=95,
            color=COLORS["neutral"],
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
        ax.text(
            np.log10(float(row["dex_conc_nM"])) + 0.015,
            np.log10(float(row["mk4_conc_uM"])) + 0.015,
            f"C{int(row['condition_index']):02d}",
            fontsize=8.2,
            color=COLORS["ink"],
        )
    style_axes(ax, "both")
    ax.set_xticks(np.log10(dex_vals), [f"{v:.4g}" for v in dex_vals])
    ax.set_yticks(np.log10(mk4_vals), [f"{v:.4g}" for v in mk4_vals])
    ax.set_xlabel("Dex concentration (nM, log10-spaced)")
    ax.set_ylabel("MK-4 concentration (uM, log10-spaced)")
    ax.set_title("Figure 5. Dose grid reference", pad=12)
    fig.text(
        0.125,
        0.94,
        "Reference layout of the 20 combinatory conditions used in sanity_check3.",
        fontsize=9.1,
        color=COLORS["muted"],
        ha="left",
    )
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
    ax.set_title("Figure 6. HAp pairwise heatmap", pad=12)
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
    fig.text(
        0.125,
        0.94,
        "Conditions are ordered by decreasing HAp response; only the upper triangle is shown to reduce redundancy.",
        fontsize=9.0,
        color=COLORS["muted"],
        ha="left",
    )
    save_figure(fig, out_dir, "figure_6_hap_pairwise_heatmap")


def export_dose_response_ranges(df: pd.DataFrame, out_dir: Path, assay_hours: dict[str, int]) -> None:
    unique_dex = np.array(sorted(df["dex_conc_nM"].unique()), dtype=float)
    unique_mk4 = np.array(sorted(df["mk4_conc_uM"].unique()), dtype=float)

    dex_summary = (
        df.groupby("dex_conc_nM", sort=True)[["assay_alp", "assay_collagen", "assay_ha"]]
        .mean()
        .reset_index()
    )
    mk4_summary = (
        df.groupby("mk4_conc_uM", sort=True)[["assay_alp", "assay_collagen", "assay_ha"]]
        .mean()
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
    ax.scatter(unique_dex, np.ones_like(unique_dex), s=84, color=COLORS["dex"], edgecolors="white", linewidths=0.8)
    for x in unique_dex:
        ax.text(x, 1.07, f"{x:.4g}", ha="center", va="bottom", fontsize=8.3, color=COLORS["ink"])
    ax.set_xscale("linear")
    ax.set_yticks([])
    ax.set_ylim(0.9, 1.18)
    ax.set_xlabel("Dex concentrations sampled (nM)")
    ax.set_title("Dex range from condition_manifest", pad=8)
    style_axes(ax, "x")

    ax = axes[0, 1]
    ax.scatter(unique_mk4, np.ones_like(unique_mk4), s=84, color=COLORS["mk4"], edgecolors="white", linewidths=0.8)
    for x in unique_mk4:
        ax.text(x, 1.07, f"{x:.4g}", ha="center", va="bottom", fontsize=8.3, color=COLORS["ink"])
    ax.set_xscale("linear")
    ax.set_yticks([])
    ax.set_ylim(0.9, 1.18)
    ax.set_xlabel("MK-4 concentrations sampled (uM)")
    ax.set_title("MK-4 range from condition_manifest", pad=8)
    style_axes(ax, "x")

    ax = axes[1, 0]
    style_axes(ax, "both")
    ax.plot(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_alp"],
        color=COLORS["alp"],
        marker="o",
        lw=2.0,
        label=f"ALP {assay_hours['alp']} h",
    )
    ax.plot(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_collagen"],
        color=COLORS["collagen"],
        marker="o",
        lw=2.0,
        label=f"Collagen {assay_hours['collagen']} h",
    )
    ax.plot(
        dex_summary["dex_conc_nM"],
        dex_summary["assay_ha"],
        color=COLORS["ha"],
        marker="o",
        lw=2.0,
        label=f"HAp {assay_hours['ha']} h",
    )
    ax.set_xlabel("Dex concentration (nM)")
    ax.set_ylabel("Mean assay response (a.u.)")
    ax.set_title("Dex dose response averaged across sampled MK-4 values", pad=10)
    ax.legend(frameon=False, loc="best")

    ax = axes[1, 1]
    style_axes(ax, "both")
    ax.plot(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_alp"],
        color=COLORS["alp"],
        marker="o",
        lw=2.0,
        label=f"ALP {assay_hours['alp']} h",
    )
    ax.plot(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_collagen"],
        color=COLORS["collagen"],
        marker="o",
        lw=2.0,
        label=f"Collagen {assay_hours['collagen']} h",
    )
    ax.plot(
        mk4_summary["mk4_conc_uM"],
        mk4_summary["assay_ha"],
        color=COLORS["ha"],
        marker="o",
        lw=2.0,
        label=f"HAp {assay_hours['ha']} h",
    )
    ax.set_xlabel("MK-4 concentration (uM)")
    ax.set_ylabel("Mean assay response (a.u.)")
    ax.set_title("MK-4 dose response averaged across sampled Dex values", pad=10)
    ax.legend(frameon=False, loc="best")

    fig.suptitle(
        "Figure 7. Dose ranges and marginal dose-response trends",
        fontsize=15,
        fontweight="semibold",
        y=1.01,
    )
    fig.text(
        0.125,
        0.958,
        "Top panels list all concentration values present in condition_manifest. Bottom panels summarize marginal Dex and MK-4 responses across the sampled combinatory design.",
        fontsize=9.0,
        color=COLORS["muted"],
        ha="left",
    )
    save_figure(fig, out_dir, "figure_7_dose_response_ranges")


def write_tables(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "condition_summary.csv", index=False)
    summary_json = df.to_dict(orient="records")
    (out_dir / "condition_summary.json").write_text(
        json.dumps(summary_json, indent=2),
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

    export_overview_heatmap(df, out_dir, assay_hours)
    export_fingerprint(df, out_dir, assay_hours)

    print(f"[sanity3_figure_extract] processed {len(df)} conditions")
    print(f"[sanity3_figure_extract] figures written to {out_dir}")


if __name__ == "__main__":
    main()
