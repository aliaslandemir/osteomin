#!/usr/bin/env python3
from __future__ import annotations

import argparse
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle
from scipy.stats import mannwhitneyu

METRIC_COLS = [
    "alp_day10",
    "collagen_day14",
    "hap_day28",
    "cell_count_day28",
]
METRIC_LABELS = ["ALP", "Collagen", "Mineral", "Cells"]

VARIANT_ORDER = [
    ("full_model", "Spatial (reference)"),
    ("no_commit_delay", "No commitment delay"),
    ("no_collagen_maturation", "No collagen delay"),
    ("no_osteogenic_commitment", "No commitment"),
    ("no_apoptosis", "No apoptosis"),
    ("nonspatial_model", "Nonspatial model"),
]

LINE_COLORS = {
    "full_model": "#0072B2",
    "no_osteogenic_commitment": "#009E73",
}

VARIANT_LABELS = {
    "full_model": "Spatial model (reference)",
    "no_osteogenic_commitment": "No osteogenic commitment",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot sanity-check-8 ablation outputs.")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "ablation_out")
    p.add_argument(
        "--spatial-out",
        type=Path,
        default=Path(__file__).resolve().parents[2] / "main_out",
        help="kept for CLI compatibility (not used directly; full_model rows are the reference)",
    )
    p.add_argument("--band-type", choices=("prediction", "confidence"), default="prediction")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _significance_stars(p_or_q: float) -> str:
    if not np.isfinite(p_or_q):
        return ""
    if p_or_q < 0.001:
        return "***"
    if p_or_q < 0.01:
        return "**"
    if p_or_q < 0.05:
        return "*"
    return ""


def _bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=float))
    ranked = np.asarray(pvals, dtype=float)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        q[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def _compute_significance(df_raw: pd.DataFrame) -> pd.DataFrame:
    cols = ["variant", *METRIC_COLS]
    if df_raw.empty or not set(cols).issubset(df_raw.columns):
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    full = df_raw[df_raw["variant"] == "full_model"]
    if full.empty:
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    variant_ids = [v for v, _ in VARIANT_ORDER if v != "full_model"]

    rows: list[dict[str, float | str | bool]] = []
    pvals: list[float] = []
    for variant in variant_ids:
        a = df_raw[df_raw["variant"] == variant]
        if a.empty:
            continue
        for metric in METRIC_COLS:
            x = a[metric].to_numpy(dtype=float)
            y = full[metric].to_numpy(dtype=float)
            x = x[np.isfinite(x)]
            y = y[np.isfinite(y)]
            if x.size < 2 or y.size < 2:
                continue
            if np.allclose(x, y, atol=1e-12, rtol=0.0):
                p = 1.0
                rbc = 0.0
            else:
                mw = mannwhitneyu(x, y, alternative="two-sided", method="auto")
                n1 = max(1, x.size)
                n2 = max(1, y.size)
                rbc = float((2.0 * mw.statistic) / (n1 * n2) - 1.0)
                p = float(mw.pvalue)
            rows.append({"variant": variant, "metric": metric, "p": p, "rbc": rbc})
            pvals.append(p)

    if not rows:
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    qvals = _bh_adjust(pvals)
    metric_label_map = dict(zip(METRIC_COLS, METRIC_LABELS))
    out_rows = []
    for i, r in enumerate(rows):
        q = float(qvals[i])
        out_rows.append(
            {
                "variant": r["variant"],
                "metric": r["metric"],
                "p": float(r["p"]),
                "q": q,
                "rbc": float(r["rbc"]),
                "significant": bool(q < 0.05),
                "stars": _significance_stars(q),
                "metric_label": metric_label_map.get(str(r["metric"]), str(r["metric"])),
            }
        )
    return pd.DataFrame(out_rows)


def _build_comparison_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    needed = {"variant", "label", *METRIC_COLS}
    if not needed.issubset(df_raw.columns):
        missing = ", ".join(sorted(needed - set(df_raw.columns)))
        raise ValueError(f"ablation_raw_metrics_runs.csv missing columns: {missing}")

    agg = (
        df_raw.groupby(["variant", "label"], as_index=False)[METRIC_COLS]
        .median(numeric_only=True)
        .reset_index(drop=True)
    )

    order = [v for v, _ in VARIANT_ORDER]
    agg["_order"] = agg["variant"].map({v: i for i, v in enumerate(order)}).fillna(999).astype(int)
    agg = agg.sort_values(["_order", "variant"], kind="stable").drop(columns=["_order"]).reset_index(drop=True)

    full_row = agg[agg["variant"] == "full_model"]
    if full_row.empty:
        raise ValueError("Missing full_model rows in ablation_raw_metrics_runs.csv")

    stats_rows: list[dict[str, float | str]] = []
    for c in METRIC_COLS:
        ref_val = float(full_row[c].iloc[0])
        ratio_col = f"{c}_ratio"
        log2_col = f"{c}_log2fc"
        ratios = []
        log2s = []
        for val in agg[c].to_numpy(dtype=float):
            if (not np.isfinite(val)) or (not np.isfinite(ref_val)) or abs(ref_val) < 1e-12:
                ratios.append(float("nan"))
                log2s.append(float("nan"))
            else:
                ratio = float(val / ref_val)
                if ratio < 0.0:
                    ratio = float("nan")
                ratios.append(ratio)
                log2s.append(float(np.log2(max(ratio, 1e-12))))
        agg[ratio_col] = ratios
        agg[log2_col] = log2s
        for _, r in agg.iterrows():
            stats_rows.append(
                {
                    "variant": str(r["variant"]),
                    "label": str(r["label"]),
                    "metric": c,
                    "reference_variant": "full_model",
                    "value": float(r[c]),
                    "reference_value": ref_val,
                    "ratio": float(r[ratio_col]) if pd.notna(r[ratio_col]) else float("nan"),
                    "log2_fold_change": float(r[log2_col]) if pd.notna(r[log2_col]) else float("nan"),
                }
            )

    agg.attrs["heatmap_stats"] = pd.DataFrame(stats_rows)
    return agg


def plot_heatmap(
    df_cmp: pd.DataFrame,
    out: Path,
    sig_df: pd.DataFrame | None = None,
) -> None:
    def fold_label(ratio: float) -> str:
        if (not np.isfinite(ratio)) or ratio < 0.0:
            return "N/A"
        return f"{ratio:.2f}×"

    cmap = LinearSegmentedColormap.from_list("plos_cb_clean", ["#4C72B0", "#F7F7F7", "#C44E52"], N=256)
    row_order = df_cmp["variant"].tolist()
    label_map = dict(VARIANT_ORDER)
    row_labels = [label_map.get(v, str(v)) for v in row_order]
    mat = np.full((len(row_order), len(METRIC_COLS)), np.nan, dtype=float)
    annot = [["" for _ in METRIC_COLS] for _ in row_order]

    for i, var in enumerate(row_order):
        for j, c in enumerate(METRIC_COLS):
            ratio = df_cmp.loc[df_cmp["variant"] == var, f"{c}_ratio"]
            if ratio.empty:
                continue
            ratio_val = float(ratio.iloc[0]) if pd.notna(ratio.iloc[0]) else float("nan")
            if np.isfinite(ratio_val) and ratio_val >= 0.0:
                mat[i, j] = ratio_val
            annot[i][j] = fold_label(ratio_val)

    star_map: dict[tuple[str, str], str] = {}
    if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
        for _, r in sig_df.iterrows():
            v = str(r.get("variant", ""))
            m = str(r.get("metric", ""))
            s = str(r.get("stars", "")) if pd.notna(r.get("stars", "")) else ""
            if v and m and s:
                star_map[(v, m)] = s

    finite_vals = mat[np.isfinite(mat)]
    finite_vals = finite_vals[finite_vals >= 0.0]
    if finite_vals.size:
        lo = float(np.percentile(finite_vals, 8))
        hi = float(np.percentile(finite_vals, 92))
        vmin = min(1.0, lo)
        vmax = max(1.0, hi)
    else:
        vmin = 0.8
        vmax = 1.2
    vmin = max(0.0, vmin)
    if not (vmin < 1.0):
        vmin = 0.8
    if not (vmax > 1.0):
        vmax = 1.2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=1.0, vmax=vmax)
    cmap = cmap.copy()
    cmap.set_bad("#EEF2F7")

    plt.rcParams.update({"font.size": 10.5, "xtick.labelsize": 10.5, "ytick.labelsize": 10.8})
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    max_label_len = max((len(str(x)) for x in row_labels), default=24)
    left_margin = min(max(0.26, 0.16 + 0.0065 * max_label_len), 0.36)
    fig.subplots_adjust(left=left_margin, right=0.88, top=0.84, bottom=0.10)
    ax.set_facecolor("#F8FAFC")

    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, norm=norm, aspect="auto")

    ax.set_xticks(np.arange(len(METRIC_LABELS)), METRIC_LABELS)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="y", pad=8)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labelsize=10.5)

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    # De-emphasize the reference row while keeping it visible.
    ref_idx = row_order.index("full_model") if "full_model" in row_order else None
    if ref_idx is not None:
        ax.add_patch(
            Rectangle(
                (-0.5, ref_idx - 0.5),
                mat.shape[1],
                1.0,
                facecolor="white",
                edgecolor="#CBD5E1",
                linewidth=0.8,
                alpha=0.20,
                zorder=2,
            )
        )
        ylabels = ax.get_yticklabels()
        if 0 <= ref_idx < len(ylabels):
            ylabels[ref_idx].set_color("#6B7280")

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = annot[i][j]
            if not txt:
                continue
            sig_txt = star_map.get((row_order[i], METRIC_COLS[j]), "")
            if sig_txt:
                txt = f"{txt}\n{sig_txt}"
            if row_order[i] == "full_model":
                color = "#4B5563"
                weight = "normal"
            else:
                norm_v = norm(v) if np.isfinite(v) else 0.5
                color = "white" if (np.isfinite(v) and (norm_v < 0.28 or norm_v > 0.72)) else "black"
                weight = "semibold"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=10.1, fontweight=weight, zorder=6)

    ax.axvline(2.5, color="#B8C1CC", lw=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.050, pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label("Fold change vs spatial reference (×)", fontsize=10.0, labelpad=6)

    fig.suptitle("Ablation Effects Relative to Spatial Model", y=0.965, fontsize=13.2, fontweight="semibold")

    fig.savefig(out / "ablation_heatmap_relative_change.png", dpi=300, bbox_inches="tight", pad_inches=0.04)
    fig.savefig(out / "ablation_heatmap_relative_change.pdf", bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    stats_df = df_cmp.attrs.get("heatmap_stats", pd.DataFrame())
    if isinstance(stats_df, pd.DataFrame):
        stats_df.to_csv(out / "ablation_heatmap_stats.csv", index=False)


def _build_publication_summary(
    df_raw: pd.DataFrame,
    df_cmp: pd.DataFrame,
    sig_df: pd.DataFrame,
) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame(
            columns=[
                "variant",
                "label",
                "metric",
                "metric_label",
                "n_runs",
                "median",
                "q1",
                "q3",
                "ratio_to_full",
                "log2_fc_to_full",
                "q_value",
                "effect_rbc",
                "significant_q05",
            ]
        )

    metric_to_ratio_col = {m: f"{m}_ratio" for m in METRIC_COLS}
    metric_to_log2_col = {m: f"{m}_log2fc" for m in METRIC_COLS}
    cmp_lookup = df_cmp.set_index("variant", drop=False)
    sig_lookup = sig_df.set_index(["variant", "metric"], drop=False) if not sig_df.empty else None

    rows: list[dict[str, float | int | str | bool]] = []
    for variant, label in VARIANT_ORDER:
        d = df_raw[df_raw["variant"] == variant]
        if d.empty:
            continue
        for metric, metric_label in zip(METRIC_COLS, METRIC_LABELS):
            vals = d[metric].to_numpy(dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                med = float("nan")
                q1 = float("nan")
                q3 = float("nan")
            else:
                med = float(np.median(vals))
                q1 = float(np.percentile(vals, 25))
                q3 = float(np.percentile(vals, 75))
            ratio = float("nan")
            log2fc = float("nan")
            if variant in cmp_lookup.index:
                ratio_col = metric_to_ratio_col[metric]
                log2_col = metric_to_log2_col[metric]
                ratio = float(cmp_lookup.loc[variant, ratio_col]) if pd.notna(cmp_lookup.loc[variant, ratio_col]) else float("nan")
                log2fc = float(cmp_lookup.loc[variant, log2_col]) if pd.notna(cmp_lookup.loc[variant, log2_col]) else float("nan")

            q_value = float("nan")
            effect_rbc = float("nan")
            significant = False
            if sig_lookup is not None and (variant, metric) in sig_lookup.index:
                rec = sig_lookup.loc[(variant, metric)]
                q_value = float(rec["q"]) if pd.notna(rec["q"]) else float("nan")
                effect_rbc = float(rec["rbc"]) if pd.notna(rec["rbc"]) else float("nan")
                significant = bool(rec["significant"])

            rows.append(
                {
                    "variant": variant,
                    "label": label,
                    "metric": metric,
                    "metric_label": metric_label,
                    "n_runs": int(vals.size),
                    "median": med,
                    "q1": q1,
                    "q3": q3,
                    "ratio_to_full": ratio,
                    "log2_fc_to_full": log2fc,
                    "q_value": q_value,
                    "effect_rbc": effect_rbc,
                    "significant_q05": significant,
                }
            )

    return pd.DataFrame(rows)


def _write_publication_legend(
    out: Path,
    df_raw: pd.DataFrame,
    band_type: str,
    endpoint_condition: str,
) -> None:
    if df_raw.empty:
        n_runs = 0
    else:
        grouped = df_raw.groupby("variant", as_index=False)["run_id"].nunique()
        n_runs = int(grouped["run_id"].max()) if not grouped.empty else 0

    band_label = "95% confidence interval of the median (bootstrap)" if band_type == "confidence" else "95% prediction band (2.5-97.5 percentiles)"
    text = textwrap.dedent(
        f"""
        Figure legend (sanity check 8: structural ablation study)

        Endpoint condition: {endpoint_condition}
        Replicates per ablation case: n={n_runs}

        Panel A (ablation_heatmap_relative_change): Median endpoint fold-change relative to the full spatial model.
        Values are shown as fold-change in color and fold-change text in each cell.
        Asterisks denote FDR-adjusted significance (Benjamini-Hochberg q<0.05) from two-sided Mann-Whitney U tests versus full spatial model.

        Panel B (dose_response_robustness): Mineral day-28 dose-response robustness for C2 and C3 across selected spatial variants.
        Curves show median fold-change with {band_label}.

        Panel C (conclusion_matrix): Categorical maintenance summary across endpoints (Yes/Partial/No) based on ratio thresholds.
        This panel is intended as a visual summary; quantitative interpretation should rely on Panel A and ablation_publication_summary.csv.
        """
    ).strip()
    (out / "ablation_figure_legend.txt").write_text(text + "\n", encoding="utf-8")


def _bootstrap_ci_for_median(samples: np.ndarray, rng: np.random.Generator, n_boot: int = 700) -> tuple[np.ndarray, np.ndarray]:
    n_runs, n_cond = samples.shape
    boot = np.empty((n_boot, n_cond), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[b, :] = np.nanmedian(samples[idx, :], axis=0)
    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)
    return lo, hi


def plot_dose_robustness(df_dose: pd.DataFrame, out: Path, band_type: str, seed: int) -> pd.DataFrame:
    cond_order = (
        df_dose["condition"]
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    cond_order = sorted(cond_order, key=lambda c: int(c[1:]) if c.startswith("C") and c[1:].isdigit() else 999)
    if not cond_order:
        raise ValueError("ablation_dose_runs.csv has no conditions")
    x = np.arange(len(cond_order))
    variants = ["full_model", "no_osteogenic_commitment"]
    variants = [v for v in variants if (df_dose["variant"] == v).any()]
    fold_col = "hap_fold_vs_ref" if "hap_fold_vs_ref" in df_dose.columns else "hap_fold_vs_c1"
    ref_cond = "C1"
    if "dose_reference_condition" in df_dose.columns:
        uniq_ref = (
            df_dose["dose_reference_condition"]
            .dropna()
            .astype(str)
            .drop_duplicates()
            .tolist()
        )
        if uniq_ref:
            ref_cond = uniq_ref[0]
    rng = np.random.default_rng(seed)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
    ax_top, ax_bot, ax_dyn, ax_peak = axes.ravel()

    rows = []
    raw_arrays: dict[str, np.ndarray] = {}
    fold_arrays: dict[str, np.ndarray] = {}

    for v in variants:
        d = df_dose[df_dose["variant"] == v]
        fold_pivot = d.pivot(index="run_id", columns="condition", values=fold_col).reindex(columns=cond_order)
        raw_pivot = d.pivot(index="run_id", columns="condition", values="hap_raw").reindex(columns=cond_order)
        arr_fold = fold_pivot.to_numpy(dtype=float)
        arr_raw = raw_pivot.to_numpy(dtype=float)

        raw_arrays[v] = arr_raw
        fold_arrays[v] = arr_fold

        med = np.nanmedian(arr_fold, axis=0)
        if band_type == "confidence":
            lo, hi = _bootstrap_ci_for_median(arr_fold, rng)
        else:
            lo = np.nanpercentile(arr_fold, 2.5, axis=0)
            hi = np.nanpercentile(arr_fold, 97.5, axis=0)

        ax_top.fill_between(x, lo, hi, color=LINE_COLORS.get(v, "#4B5563"), alpha=0.20, linewidth=0.0)
        ax_top.plot(
            x,
            med,
            lw=2.5,
            marker="o",
            markersize=5,
            color=LINE_COLORS.get(v, "#4B5563"),
            label=VARIANT_LABELS.get(v, v),
        )

        for i, c in enumerate(cond_order):
            rows.append(
                {
                    "variant": v,
                    "condition": c,
                    "median": float(med[i]) if np.isfinite(med[i]) else float("nan"),
                    "lower": float(lo[i]) if np.isfinite(lo[i]) else float("nan"),
                    "upper": float(hi[i]) if np.isfinite(hi[i]) else float("nan"),
                    "panel": "fold_vs_ref",
                    "band_type": band_type,
                }
            )

    full_raw = raw_arrays.get("full_model")
    if full_raw is not None:
        for v in [vv for vv in variants if vv != "full_model"]:
            if v not in raw_arrays:
                continue
            ratio = raw_arrays[v] / np.maximum(1e-12, full_raw)
            med = np.nanmedian(ratio, axis=0)
            if band_type == "confidence":
                lo, hi = _bootstrap_ci_for_median(ratio, rng)
            else:
                lo = np.nanpercentile(ratio, 2.5, axis=0)
                hi = np.nanpercentile(ratio, 97.5, axis=0)

            ax_bot.fill_between(x, lo, hi, color=LINE_COLORS.get(v, "#4B5563"), alpha=0.20, linewidth=0.0)
            ax_bot.plot(
                x,
                med,
                lw=2.5,
                marker="o",
                markersize=5,
                color=LINE_COLORS.get(v, "#4B5563"),
                label=f"{VARIANT_LABELS.get(v, v)} / Full",
            )

            for i, c in enumerate(cond_order):
                rows.append(
                    {
                        "variant": v,
                        "condition": c,
                        "median": float(med[i]) if np.isfinite(med[i]) else float("nan"),
                        "lower": float(lo[i]) if np.isfinite(lo[i]) else float("nan"),
                        "upper": float(hi[i]) if np.isfinite(hi[i]) else float("nan"),
                        "panel": "ratio_to_full",
                        "band_type": band_type,
                    }
                )

    ylabel = (
        f"Mineral d28 fold vs {ref_cond} (median ± 95% CI)"
        if band_type == "confidence"
        else f"Mineral d28 fold vs {ref_cond} (median ± 95% band)"
    )
    ax_top.set_ylabel(ylabel)
    ax_top.set_title("Dose-response curves")
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.grid(False)

    ax_bot.set_xlabel("Condition")
    ax_bot.set_ylabel("Ablation / Full (median ± 95%)")
    ax_bot.axhline(1.0, color="#6B7280", lw=1.2)
    ax_bot.legend(loc="upper right", frameon=False)
    ax_bot.grid(False)
    ax_bot.set_xticks(x, cond_order)

    dyn_vals = []
    dyn_labels = []
    dyn_keys = []
    for v in variants:
        arr = fold_arrays.get(v)
        if arr is None or arr.size == 0:
            continue
        dyn = np.nanmax(arr, axis=1) - np.nanmin(arr, axis=1)
        dyn = dyn[np.isfinite(dyn)]
        if dyn.size == 0:
            continue
        dyn_vals.append(dyn)
        dyn_labels.append(VARIANT_LABELS.get(v, v))
        dyn_keys.append(v)

    if dyn_vals:
        vp = ax_dyn.violinplot(dyn_vals, showmeans=False, showmedians=True, widths=0.8)
        for i, b in enumerate(vp["bodies"]):
            color = LINE_COLORS.get(dyn_keys[i], "#4B5563")
            b.set_facecolor(color)
            b.set_alpha(0.28)
            b.set_edgecolor(color)
        vp["cmedians"].set_color("#111827")
        vp["cmedians"].set_linewidth(1.5)
        ax_dyn.set_xticks(np.arange(1, len(dyn_labels) + 1), dyn_labels, rotation=12, ha="right")
    ax_dyn.set_ylabel("Dose dynamic range")
    ax_dyn.set_title("Across-run dynamic range")
    ax_dyn.grid(False)

    peak_prob = np.zeros((len(variants), len(cond_order)), dtype=float)
    for i, v in enumerate(variants):
        arr = fold_arrays.get(v)
        if arr is None or arr.size == 0:
            continue
        valid_rows = 0
        for row in arr:
            if not np.isfinite(row).any():
                continue
            idx = int(np.nanargmax(row))
            peak_prob[i, idx] += 1.0
            valid_rows += 1
        if valid_rows > 0:
            peak_prob[i, :] /= float(valid_rows)

    im = ax_peak.imshow(peak_prob, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax_peak.set_xticks(np.arange(len(cond_order)), cond_order)
    ax_peak.set_yticks(np.arange(len(variants)), [VARIANT_LABELS.get(v, v) for v in variants])
    ax_peak.set_title("Peak-condition probability")
    for i in range(len(variants)):
        for j in range(len(cond_order)):
            txt = f"{peak_prob[i, j] * 100:.0f}%"
            color = "white" if peak_prob[i, j] > 0.55 else "black"
            ax_peak.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax_peak, fraction=0.046, pad=0.04)
    cbar.set_label("Run fraction")

    for ax, letter in zip([ax_top, ax_bot, ax_dyn, ax_peak], ["A", "B", "C", "D"]):
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig.savefig(out / "dose_response_robustness.png", dpi=300)
    fig.savefig(out / "dose_response_robustness.pdf")
    plt.close(fig)

    bands = pd.DataFrame(rows)
    bands.to_csv(out / "dose_response_robustness_bands.csv", index=False)
    return bands


def plot_conclusion_matrix(df_cmp: pd.DataFrame, out: Path) -> None:
    def status(ratio: float) -> float:
        if not np.isfinite(ratio):
            return float("nan")
        if 0.80 <= ratio <= 1.20:
            return 2.0
        if 0.50 <= ratio <= 1.50:
            return 1.0
        return 0.0

    rows = []
    for _, r in df_cmp.iterrows():
        rows.append(
            {
                "label": r["label"],
                "ALP maintained": status(float(r["alp_day10_ratio"])),
                "Collagen maintained": status(float(r["collagen_day14_ratio"])),
                "Mineral maintained": status(float(r["hap_day28_ratio"])),
                "Cell count maintained": status(float(r["cell_count_day28_ratio"])),
            }
        )
    mat_df = pd.DataFrame(rows)
    cols = ["ALP maintained", "Collagen maintained", "Mineral maintained", "Cell count maintained"]
    mat = mat_df[cols].to_numpy(dtype=float)

    cmap = ListedColormap(["#D55E00", "#F0E442", "#009E73", "#D1D5DB"])
    mat_show = np.where(np.isnan(mat), 3.0, mat)
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    ax.imshow(mat_show, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="equal")
    ax.set_xticks(np.arange(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(mat_df)), mat_df["label"])
    ax.set_title("Conclusion matrix")

    txt_map = {0: "No", 1: "Partial", 2: "Yes", 3: "N/A"}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            key = 3 if np.isnan(mat[i, j]) else int(mat[i, j])
            ax.text(j, i, txt_map[key], ha="center", va="center", fontsize=9)

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(out / "conclusion_matrix.png", dpi=300)
    fig.savefig(out / "conclusion_matrix.pdf")
    mat_df.to_csv(out / "conclusion_matrix_table.csv", index=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.out.resolve()

    dose_csv = out / "ablation_dose_runs.csv"
    raw_csv = out / "ablation_raw_metrics_runs.csv"
    if not dose_csv.exists():
        raise SystemExit("Missing ablation_dose_runs.csv. Run ablation_study_run.py first.")
    if not raw_csv.exists():
        raise SystemExit("Missing ablation_raw_metrics_runs.csv. Run ablation_study_run.py first.")

    df_dose = pd.read_csv(dose_csv)
    df_raw = pd.read_csv(raw_csv)
    endpoint_condition = (
        str(df_raw["condition"].dropna().astype(str).iloc[0])
        if ("condition" in df_raw.columns and not df_raw["condition"].dropna().empty)
        else "unknown"
    )

    df_cmp = _build_comparison_table(df_raw)
    df_cmp.to_csv(out / "ablation_comparison_table.csv", index=False)

    sig_df = _compute_significance(df_raw)
    sig_df.to_csv(out / "ablation_significance_table.csv", index=False)

    pub_df = _build_publication_summary(df_raw, df_cmp, sig_df)
    pub_df.to_csv(out / "ablation_publication_summary.csv", index=False)

    plot_heatmap(df_cmp, out, sig_df=sig_df)
    plot_dose_robustness(df_dose, out, band_type=args.band_type, seed=int(args.seed))
    plot_conclusion_matrix(df_cmp, out)
    _write_publication_legend(out, df_raw=df_raw, band_type=args.band_type, endpoint_condition=endpoint_condition)

    print(f"Saved: {out / 'ablation_heatmap_relative_change.png'}")
    print(f"Saved: {out / 'dose_response_robustness.png'}")
    print(f"Saved: {out / 'conclusion_matrix.png'}")
    print(f"Saved: {out / 'dose_response_robustness_bands.csv'}")
    print(f"Saved: {out / 'ablation_heatmap_stats.csv'}")
    print(f"Saved: {out / 'ablation_comparison_table.csv'}")
    print(f"Saved: {out / 'ablation_significance_table.csv'}")
    print(f"Saved: {out / 'ablation_publication_summary.csv'}")
    print(f"Saved: {out / 'ablation_figure_legend.txt'}")


if __name__ == "__main__":
    main()
