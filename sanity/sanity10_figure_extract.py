#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "osteomin_mplconfig").resolve()))

import matplotlib
import numpy as np
import pandas as pd

import sanity_check10 as sc10

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, Polygon, Rectangle

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "out" / "sanity_check10"


def _resolve_input(input_path: Path | None, out_dir: Path) -> Path:
    if input_path is None:
        return out_dir / "simulated_observables_long.csv"
    if input_path.is_dir():
        return input_path / "simulated_observables_long.csv"
    return input_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity 10 figure extractor. Reads an existing long-format table and "
            "regenerates the figure outputs without rerunning the simulations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory for regenerated figures")
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Long-format CSV/XLSX input, or a directory containing simulated_observables_long.csv",
    )
    parser.add_argument("--sheet", type=str, default="", help="Optional sheet for XLSX input")
    parser.add_argument("--days", type=int, nargs="+", default=list(sc10.DEFAULT_DAYS), help="Day panels")
    parser.add_argument("--scaling", choices=("global", "per-day"), default="global")
    parser.add_argument("--pca-mode", choices=("global", "per-day"), default="global")
    parser.add_argument("--equivalence-threshold", type=float, default=1.0)
    parser.add_argument("--threshold-sweep", type=float, nargs="*", default=None, help="Optional list of equivalence thresholds to rerun analysis with")
    parser.add_argument("--family-col", type=str, default="")
    parser.add_argument("--spatial-col", type=str, default="spatial_class")
    parser.add_argument("--observables-only", action="store_true", help="Use only ALP/collagen/mineral for PCA and collapse analysis")
    parser.add_argument("--figures", type=str, nargs="+", default=["all"], choices=("all", "composite", "collapse", "trajectory", "distance"), help="Select figure families to generate")
    parser.add_argument("--save-tag", type=str, default="", help="Optional suffix appended to output directory")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA outputs and PCA composite figure")
    parser.add_argument("--export-summary-json", action="store_true", help="Export analysis configuration summary JSON")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument(
        "--skip-ident-robustness-figure",
        action="store_true",
        help="Skip the composite identifiability robustness/information-gain figure export",
    )
    args = parser.parse_args()
    if args.save_tag:
        args.out = args.out / str(args.save_tag)
    return args


def _prepare_analysis_raw(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    frame = raw.copy()
    day_col = sc10.resolve_column(frame, ("day",), required=True)
    cond_col = sc10.resolve_column(frame, ("condition_id", "condition", "conditionid"), required=True)
    frame["condition_id"] = frame[cond_col].astype(str)
    frame["day"] = pd.to_numeric(frame[day_col], errors="coerce").round().astype(int)
    for target, names in {
        "ALP": ("ALP", "alp", "alp_total"),
        "collagen": ("collagen", "collagen_total"),
        "mineral": ("mineral", "ha", "ha_total", "ars", "hap"),
    }.items():
        col = sc10.resolve_column(frame, names, required=True)
        frame[target] = pd.to_numeric(frame[col], errors="coerce")
    frame["perturbation_family"] = sc10.infer_family(frame, args.family_col)
    # Legacy guard: older sanity10 runs may still include the old dose family.
    frame = frame.loc[frame["perturbation_family"].astype(str) != "3_dex_mk4_dose"].copy()
    frame = frame.loc[
        frame["perturbation_family"].astype(str)
        != getattr(sc10, "TYPE_SAMPLING", "6_sampling_time")
    ].copy()
    frame = frame.loc[
        frame["perturbation_family"].astype(str)
        != getattr(sc10, "TYPE_BOUNDARY", "5_boundary_scale")
    ].copy()
    frame = frame.loc[frame["day"].isin(args.days)].copy()
    grouped = frame.groupby(["condition_id", "day"], observed=True).agg(
        perturbation_family=("perturbation_family", "first"),
        ALP=("ALP", "mean"),
        collagen=("collagen", "mean"),
        mineral=("mineral", "mean"),
    )
    return grouped.reset_index()


def _prepare_family_cond_day_full(raw: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    fam_keep = {
        getattr(sc10, "TYPE_BASELINE", "0_baseline"),
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    }
    composite_path = args.out / "composite_conditions.csv"
    runs_dir = args.out / "runs"
    if composite_path.exists() and runs_dir.exists():
        meta = pd.read_csv(composite_path)
        meta = meta.loc[meta["perturbation_family"].astype(str).isin(fam_keep)].copy()
        fam_map = {
            str(row["condition_id"]): str(row["perturbation_family"])
            for _, row in meta.iterrows()
        }
        recs: list[pd.DataFrame] = []
        for cond_id, fam in fam_map.items():
            cond_dir = runs_dir / cond_id
            if not cond_dir.exists():
                continue
            for ts_path in sorted(cond_dir.glob("r*/timeseries.csv")):
                try:
                    ts = pd.read_csv(
                        ts_path,
                        usecols=["day", "alp_total", "collagen_total", "ha_total"],
                    )
                except Exception:
                    continue
                ts["day"] = pd.to_numeric(ts["day"], errors="coerce").round().astype("Int64")
                ts = ts.loc[ts["day"].notna()].copy()
                ts["day"] = ts["day"].astype(int)
                ts = ts.loc[(ts["day"] >= 0) & (ts["day"] <= 28)].copy()
                if ts.empty:
                    continue
                ts = ts.rename(
                    columns={
                        "alp_total": "ALP",
                        "collagen_total": "collagen",
                        "ha_total": "mineral",
                    }
                )
                # Backward-compatibility correction: older outputs may write day-0 mineral
                # as 0.0 even though the field starts at width*height*ha_initial.
                d0 = ts.loc[ts["day"].astype(int) == 0, "mineral"]
                if not d0.empty and float(d0.iloc[0]) <= 0.0:
                    run_json = ts_path.with_name("run.json")
                    if run_json.exists():
                        try:
                            payload = json.loads(run_json.read_text(encoding="utf-8"))
                            fc = payload.get("field_cfg", {})
                            w = float(fc.get("width", 0.0))
                            h = float(fc.get("height", 0.0))
                            h0 = float(fc.get("ha_initial", 0.0))
                            init_ha = w * h * h0
                            if init_ha > 0.0:
                                ts.loc[ts["day"].astype(int) == 0, "mineral"] = init_ha
                        except Exception:
                            pass
                ts["condition_id"] = str(cond_id)
                ts["perturbation_family"] = str(fam)
                recs.append(ts[["condition_id", "day", "perturbation_family", "ALP", "collagen", "mineral"]])
        if recs:
            frame = pd.concat(recs, ignore_index=True)
            grouped = frame.groupby(["condition_id", "day"], observed=True).agg(
                perturbation_family=("perturbation_family", "first"),
                ALP=("ALP", "mean"),
                collagen=("collagen", "mean"),
                mineral=("mineral", "mean"),
            )
            return grouped.reset_index()

    # Fallback to provided long-format table if full run timeseries are unavailable.
    frame = raw.copy()
    day_col = sc10.resolve_column(frame, ("day",), required=True)
    cond_col = sc10.resolve_column(frame, ("condition_id", "condition", "conditionid"), required=True)
    frame["condition_id"] = frame[cond_col].astype(str)
    frame["day"] = pd.to_numeric(frame[day_col], errors="coerce").round().astype(int)
    for target, names in {
        "ALP": ("ALP", "alp", "alp_total"),
        "collagen": ("collagen", "collagen_total"),
        "mineral": ("mineral", "ha", "ha_total", "ars", "hap"),
    }.items():
        col = sc10.resolve_column(frame, names, required=True)
        frame[target] = pd.to_numeric(frame[col], errors="coerce")
    frame["perturbation_family"] = sc10.infer_family(frame, args.family_col)
    frame = frame.loc[frame["perturbation_family"].astype(str) != "3_dex_mk4_dose"].copy()
    frame = frame.loc[frame["perturbation_family"].astype(str).isin(fam_keep)].copy()
    frame = frame.loc[(frame["day"].astype(int) >= 0) & (frame["day"].astype(int) <= 28)].copy()
    grouped = frame.groupby(["condition_id", "day"], observed=True).agg(
        perturbation_family=("perturbation_family", "first"),
        ALP=("ALP", "mean"),
        collagen=("collagen", "mean"),
        mineral=("mineral", "mean"),
    )
    return grouped.reset_index()


def _max_centroid_distance(cond_day: pd.DataFrame, features: list[str], scaling: str) -> float:
    by = "day" if scaling == "per-day" else None
    work, _ = sc10.standardize(cond_day.copy(), features, by=by)
    z_cols = [f"z_{feature}" for feature in features]
    base_id = str(sc10._baseline_condition_id(work))
    max_dist = 0.0
    for _, day_df in work.groupby("day", observed=True):
        base = day_df.loc[day_df["condition_id"].astype(str) == base_id]
        if base.empty:
            continue
        ref = base[z_cols].iloc[0].to_numpy(dtype=float)
        comp = day_df.loc[day_df["condition_id"].astype(str) != base_id, z_cols]
        if comp.empty:
            continue
        d = np.linalg.norm(comp.to_numpy(dtype=float) - ref[None, :], axis=1)
        max_dist = max(max_dist, float(np.nanmax(d)))
    return max(max_dist, 1.0)


def _threshold_grid(cond_day: pd.DataFrame, args: argparse.Namespace) -> np.ndarray:
    sweep = [float(x) for x in (args.threshold_sweep or []) if np.isfinite(float(x))]
    max_dist = _max_centroid_distance(cond_day, list(sc10.FEATURES), args.scaling)
    hi = max_dist * 1.05
    if sweep:
        hi = max(hi, max(sweep) * 1.10)
    dense = np.linspace(0.0, hi, 121)
    if sweep:
        return np.unique(np.concatenate([dense, np.array(sorted(set(sweep)), dtype=float)]))
    return dense


def _distance_to_reference_table(
    cond_day: pd.DataFrame,
    features: list[str],
    scaling: str,
) -> pd.DataFrame:
    by = "day" if scaling == "per-day" else None
    work, _ = sc10.standardize(cond_day.copy(), features, by=by)
    z_cols = [f"z_{feature}" for feature in features]
    base_id = str(sc10._baseline_condition_id(work))
    rows: list[dict[str, float | int | str]] = []
    for day, day_df in work.groupby("day", observed=True):
        base = day_df.loc[day_df["condition_id"].astype(str) == base_id]
        if base.empty:
            continue
        ref = base[z_cols].iloc[0].to_numpy(dtype=float)
        comp = day_df.loc[day_df["condition_id"].astype(str) != base_id].copy()
        if comp.empty:
            continue
        dist = np.linalg.norm(comp[z_cols].to_numpy(dtype=float) - ref[None, :], axis=1)
        comp["distance_to_C2"] = dist
        for _, row in comp.iterrows():
            rows.append(
                {
                    "condition_id": str(row["condition_id"]),
                    "perturbation_family": str(row["perturbation_family"]),
                    "day": int(day),
                    "distance_to_C2": float(row["distance_to_C2"]),
                }
            )
    return pd.DataFrame(rows)


def _collapse_curves(
    cond_day: pd.DataFrame,
    thresholds: np.ndarray,
    features: list[str],
    scaling: str,
    subset_label: str,
) -> pd.DataFrame:
    dist_df = _distance_to_reference_table(cond_day, features, scaling)
    if dist_df.empty:
        return pd.DataFrame(columns=["subset", "day", "threshold", "collapsed_fraction", "n_conditions"])
    rows: list[dict[str, float | int | str]] = []
    for thr in thresholds:
        for day, day_df in dist_df.groupby("day", observed=True):
            y = day_df["distance_to_C2"].to_numpy(dtype=float)
            rows.append(
                {
                    "subset": subset_label,
                    "day": int(day),
                    "threshold": float(thr),
                    "collapsed_fraction": float(np.mean(y <= float(thr))),
                    "n_conditions": int(len(y)),
                    "median_distance": float(np.nanmedian(y)),
                }
            )
    return pd.DataFrame(rows)


def _make_identifiability_summary_figure(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    def _nn_eval(cond: pd.DataFrame, features: list[str], subset: str) -> tuple[pd.DataFrame, pd.DataFrame]:
        by = "day" if args.scaling == "per-day" else None
        work, _ = sc10.standardize(cond.copy(), features, by=by)
        z_cols = [f"z_{f}" for f in features]
        work = work.loc[work["perturbation_family"].astype(str) != sc10.TYPE_BASELINE].copy()
        rows: list[dict[str, float | int | str]] = []
        fam_rows: list[dict[str, float | int | str]] = []
        for day, day_df in work.groupby("day", observed=True):
            x = day_df[z_cols].to_numpy(dtype=float)
            labels = day_df["perturbation_family"].astype(str).to_numpy()
            n = len(labels)
            if n < 3:
                continue
            d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
            np.fill_diagonal(d, np.inf)
            nn_idx = np.argmin(d, axis=1)
            pred = labels[nn_idx]
            ok = (pred == labels)
            acc = float(np.mean(ok))
            counts = pd.Series(labels).value_counts().astype(float)
            p = counts / float(n)
            random_acc = float(np.sum(np.square(p.to_numpy(dtype=float))))
            rows.append(
                {
                    "subset": subset,
                    "day": int(day),
                    "n_conditions": int(n),
                    "nn_consistency": acc,
                    "nn_ambiguity": float(1.0 - acc),
                    "random_consistency": random_acc,
                    "random_ambiguity": float(1.0 - random_acc),
                }
            )
            fam_acc = (
                pd.DataFrame({"true_family": labels, "ok": ok})
                .groupby("true_family", observed=True)["ok"]
                .mean()
                .reset_index()
            )
            for _, r in fam_acc.iterrows():
                fam_rows.append(
                    {
                        "subset": subset,
                        "day": int(day),
                        "true_family": str(r["true_family"]),
                        "nn_consistency": float(r["ok"]),
                        "nn_ambiguity": float(1.0 - float(r["ok"])),
                    }
                )
        return pd.DataFrame(rows), pd.DataFrame(fam_rows)

    cond_day = _prepare_analysis_raw(raw, args)
    if cond_day.empty:
        return
    day_order = sorted(int(day) for day in args.days)
    subset_order = ["ALP only", "ALP+Collagen", "ALP+Collagen+Mineral"]
    all_metrics: list[pd.DataFrame] = []
    fam_metrics: list[pd.DataFrame] = []
    for subset, feats in [
        ("ALP only", ["ALP"]),
        ("ALP+Collagen", ["ALP", "collagen"]),
        ("ALP+Collagen+Mineral", ["ALP", "collagen", "mineral"]),
    ]:
        s, f = _nn_eval(cond_day, feats, subset)
        all_metrics.append(s)
        fam_metrics.append(f)
    summary = pd.concat(all_metrics, ignore_index=True) if all_metrics else pd.DataFrame()
    family_df = pd.concat(fam_metrics, ignore_index=True) if fam_metrics else pd.DataFrame()
    if summary.empty:
        return

    summary.to_csv(args.out / "ident_nn_consistency_summary.csv", index=False)
    family_df.to_csv(args.out / "ident_nn_family_consistency.csv", index=False)

    day_last = max(day_order) if day_order else int(summary["day"].max())
    all3_last = cond_day.loc[
        (cond_day["day"].astype(int) == int(day_last))
        & (cond_day["perturbation_family"].astype(str) != sc10.TYPE_BASELINE)
    ].copy()
    work_last, _ = sc10.standardize(all3_last.copy(), ["ALP", "collagen", "mineral"], by="day" if args.scaling == "per-day" else None)
    z_cols = ["z_ALP", "z_collagen", "z_mineral"]
    x = work_last[z_cols].to_numpy(dtype=float)
    labels = work_last["perturbation_family"].astype(str).to_numpy()
    d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn_idx = np.argmin(d, axis=1)
    pred = labels[nn_idx]
    conf = pd.crosstab(pd.Series(labels, name="true"), pd.Series(pred, name="pred"), normalize="index")
    conf.to_csv(args.out / "ident_nn_confusion_day_last_all3.csv")

    family_label = {
        sc10.TYPE_SPATIAL: "Spatial",
        sc10.TYPE_CELLS: "Cell #",
        sc10.TYPE_DYNAMICS: "Dynamics",
    }
    conf = conf.reindex(index=list(family_label.keys()), columns=list(family_label.keys()), fill_value=0.0)
    conf.index = [family_label.get(x, x) for x in conf.index]
    conf.columns = [family_label.get(x, x) for x in conf.columns]

    fig = plt.figure(figsize=(12.6, 7.1), constrained_layout=False)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.15, 1.0], hspace=0.34, wspace=0.20)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, :])

    all3 = summary.loc[summary["subset"].astype(str) == "ALP+Collagen+Mineral"].sort_values("day")
    ax_a.plot(all3["day"].to_numpy(dtype=float), all3["nn_ambiguity"].to_numpy(dtype=float), color="#1f77b4", marker="o", linewidth=2.3, label="Observed ambiguity")
    ax_a.plot(all3["day"].to_numpy(dtype=float), all3["random_ambiguity"].to_numpy(dtype=float), color="#9ca3af", linestyle="--", linewidth=1.6, label="Random-label baseline")
    ax_a.set_ylim(0.0, 1.0)
    ax_a.set_xticks(day_order)
    ax_a.set_xlabel("Day")
    ax_a.set_ylabel("NN ambiguity rate (lower = better)")
    ax_a.set_title("A. Threshold-free collapse metric (all 3 markers)")
    ax_a.grid(axis="y", color="#d8dee9", linewidth=0.8, alpha=0.5)
    ax_a.legend(loc="upper right", frameon=False, fontsize=8.8)

    im = ax_b.imshow(conf.to_numpy(dtype=float), vmin=0.0, vmax=1.0, cmap="Blues")
    ax_b.set_xticks(np.arange(conf.shape[1]))
    ax_b.set_yticks(np.arange(conf.shape[0]))
    ax_b.set_xticklabels(conf.columns.tolist())
    ax_b.set_yticklabels(conf.index.tolist())
    ax_b.set_xlabel("Nearest-neighbor family")
    ax_b.set_ylabel("True family")
    ax_b.set_title(f"B. Family confusion at day {int(day_last)} (all 3 markers)")
    for i in range(conf.shape[0]):
        for j in range(conf.shape[1]):
            val = float(conf.iloc[i, j])
            txt = f"{100.0 * val:.0f}%"
            color = "white" if val >= 0.55 else "#111827"
            ax_b.text(j, i, txt, ha="center", va="center", fontsize=9.0, color=color)
    cbar = fig.colorbar(im, ax=ax_b, fraction=0.046, pad=0.04)
    cbar.ax.set_ylabel("Row-normalized fraction", rotation=90)

    colors = {"ALP only": "#9e9e9e", "ALP+Collagen": "#5d88c1", "ALP+Collagen+Mineral": "#1f77b4"}
    styles = {"ALP only": "--", "ALP+Collagen": "-.", "ALP+Collagen+Mineral": "-"}
    for subset in subset_order:
        s = summary.loc[summary["subset"].astype(str) == subset].sort_values("day")
        if s.empty:
            continue
        ax_c.plot(
            s["day"].to_numpy(dtype=float),
            s["nn_ambiguity"].to_numpy(dtype=float),
            color=colors[subset],
            linestyle=styles[subset],
            marker="o",
            linewidth=2.1 if subset == "ALP+Collagen+Mineral" else 1.9,
            label=subset,
        )
    ax_c.set_ylim(0.0, 1.0)
    ax_c.set_xticks(day_order)
    ax_c.set_xlabel("Day")
    ax_c.set_ylabel("NN ambiguity rate (lower = better)")
    ax_c.set_title("C. Observable subset test")
    ax_c.grid(axis="y", color="#d8dee9", linewidth=0.8, alpha=0.5)
    ax_c.legend(loc="upper right", frameon=False, ncol=1, fontsize=9.0)

    fig.suptitle("Identifiability Summary", fontsize=14.0, y=0.99)
    fig.savefig(args.out / "sanity10_identifiability_robustness.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_identifiability_robustness.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_identifiability_robustness.svg", bbox_inches="tight")
    plt.close(fig)


def _make_confusion_abc_figure(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    text_scale = 1.2
    cond_day = _prepare_family_cond_day_full(raw, args)
    if cond_day.empty:
        return
    panel_day_end_by_marker = {"ALP": 10, "collagen": 14, "mineral": 28}
    available_days = set(cond_day["day"].astype(int).tolist())
    if not any(any(0 <= int(d) <= end for d in available_days) for end in panel_day_end_by_marker.values()):
        return

    work = cond_day.copy()
    work = work.loc[work["perturbation_family"].astype(str) != sc10.TYPE_BASELINE].copy()
    preferred_order = [
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    ]
    present = set(work["perturbation_family"].astype(str).unique().tolist())
    family_order = [fam for fam in preferred_order if fam in present]
    family_label = {
        sc10.TYPE_SPATIAL: "Spatial",
        sc10.TYPE_CELLS: "Cell #",
        sc10.TYPE_DYNAMICS: "Dynamics",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "Ascorbic Acid",
    }

    marker_order = ["ALP", "collagen", "mineral"]
    panel_specs = [
        {"letter": "A", "marker": "ALP", "title": "ALP"},
        {"letter": "B", "marker": "collagen", "title": "Collagen"},
        {"letter": "C", "marker": "mineral", "title": "Mineral"},
    ]

    def _panel_days(mode: str, marker: str) -> list[int]:
        end = int(panel_day_end_by_marker[marker])
        if mode == "endpoint":
            return [end] if end in available_days else []
        return sorted(int(d) for d in available_days if 0 <= int(d) <= end)

    def _compute_panel(mode: str, marker: str) -> tuple[pd.DataFrame, float, list[int]] | None:
        days = _panel_days(mode, marker)
        if not days:
            return None
        sub = work.loc[work["day"].astype(int).isin(days)].copy()
        if sub.empty:
            return None
        marker_df = (
            sub[["condition_id", "day", "perturbation_family", marker]]
            .copy()
            .rename(columns={marker: "value"})
        )
        pivot = (
            marker_df.pivot_table(
                index=["condition_id", "perturbation_family"],
                columns="day",
                values="value",
                aggfunc="mean",
            )
            .reindex(columns=days)
            .dropna(axis=0, how="any")
        )
        if pivot.empty:
            return None
        x_raw = pivot.to_numpy(dtype=float)
        # Standardize each time coordinate (or endpoint singleton) to avoid scale dominance.
        mu = np.nanmean(x_raw, axis=0)
        sd = np.nanstd(x_raw, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        x = (x_raw - mu[None, :]) / sd[None, :]
        labels = pivot.index.get_level_values("perturbation_family").astype(str).to_numpy()
        if len(labels) < 3:
            return None
        dmat = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        pred = labels[np.argmin(dmat, axis=1)]
        true_s = pd.Series(labels, name="true")
        pred_s = pd.Series(pred, name="pred")
        cmat = pd.crosstab(true_s, pred_s).reindex(index=family_order, columns=family_order, fill_value=0)
        f1_vals: list[float] = []
        for fam in family_order:
            tp = float(cmat.loc[fam, fam])
            fp = float(cmat[fam].sum() - tp)
            fn = float(cmat.loc[fam].sum() - tp)
            denom = 2.0 * tp + fp + fn
            f1_vals.append((2.0 * tp / denom) if denom > 0.0 else 0.0)
        macro_f1 = float(np.mean(f1_vals)) if f1_vals else float("nan")
        conf = pd.crosstab(true_s, pred_s, normalize="index")
        conf = conf.reindex(index=family_order, columns=family_order, fill_value=0.0)
        return conf, macro_f1, days

    conf_by_panel_endpoint: dict[str, pd.DataFrame] = {}
    conf_by_panel_trajectory: dict[str, pd.DataFrame] = {}
    f1_by_panel_endpoint: dict[str, float] = {}
    f1_by_panel_trajectory: dict[str, float] = {}
    long_rows: list[dict[str, float | int | str]] = []
    for panel in panel_specs:
        key = str(panel["letter"])
        marker = str(panel["marker"])
        result_ep = _compute_panel("endpoint", marker)
        result_tr = _compute_panel("trajectory", marker)
        if result_ep is None or result_tr is None:
            continue
        conf_ep, macro_f1_ep, days_ep = result_ep
        conf_tr, macro_f1_tr, days_tr = result_tr
        conf_by_panel_endpoint[key] = conf_ep
        conf_by_panel_trajectory[key] = conf_tr
        f1_by_panel_endpoint[key] = macro_f1_ep
        f1_by_panel_trajectory[key] = macro_f1_tr
        for mode, conf, days_used in [("endpoint", conf_ep, days_ep), ("trajectory", conf_tr, days_tr)]:
            for true_key in family_order:
                for pred_key in family_order:
                    long_rows.append(
                        {
                            "panel": key,
                            "mode": mode,
                            "marker": marker,
                            "days_used": ",".join(str(int(dy)) for dy in days_used),
                            "true_family": str(true_key),
                            "pred_family": str(pred_key),
                            "fraction": float(conf.loc[true_key, pred_key]),
                        }
                    )

    if not conf_by_panel_endpoint:
        return
    pd.DataFrame(long_rows).to_csv(args.out / "ident_nn_confusion_endpoint_vs_trajectory.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.out / "ident_nn_confusion_day10_14_28_all3.csv", index=False)

    fig, axes = plt.subplots(1, 3, figsize=(16.4, 6.1), constrained_layout=False)
    fig.patch.set_facecolor("white")
    fig.subplots_adjust(left=0.065, right=0.992, top=0.86, bottom=0.135, wspace=0.12)
    # Colorblind-safe, value-dependent split fills.
    endpoint_cmap = plt.get_cmap("Blues")
    trajectory_cmap = plt.get_cmap("Oranges")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for i, panel in enumerate(panel_specs):
        ax = axes[i]
        key = str(panel["letter"])
        conf_ep = conf_by_panel_endpoint.get(key)
        conf_tr = conf_by_panel_trajectory.get(key)
        if conf_ep is None or conf_tr is None:
            ax.axis("off")
            continue
        arr_ep = conf_ep.to_numpy(dtype=float)
        arr_tr = conf_tr.to_numpy(dtype=float)
        n = arr_ep.shape[0]
        for r in range(n):
            for c in range(n):
                x0, y0 = float(c), float(r)
                v_ep = float(arr_ep[r, c])
                v_tr = float(arr_tr[r, c])
                c_ep = endpoint_cmap(0.25 + 0.70 * norm(v_ep))
                c_tr = trajectory_cmap(0.25 + 0.70 * norm(v_tr))
                tri_bl = Polygon(
                    [[x0 - 0.5, y0 - 0.5], [x0 - 0.5, y0 + 0.5], [x0 + 0.5, y0 + 0.5]],
                    closed=True,
                    facecolor=c_ep,
                    edgecolor="none",
                )
                tri_tr = Polygon(
                    [[x0 - 0.5, y0 - 0.5], [x0 + 0.5, y0 - 0.5], [x0 + 0.5, y0 + 0.5]],
                    closed=True,
                    facecolor=c_tr,
                    edgecolor="none",
                )
                ax.add_patch(tri_bl)
                ax.add_patch(tri_tr)
                # Dark diagonal split line for clearer endpoint/trajectory separation.
                ax.plot(
                    [x0 - 0.5, x0 + 0.5],
                    [y0 - 0.5, y0 + 0.5],
                    color="#111111",
                    linewidth=1.05,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=4,
                )
                # Strong white box borders to visually separate each split cell.
                ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 1.0, 1.0, fill=False, edgecolor="#ffffff", linewidth=2.16))
                txt_ep = f"{100.0 * v_ep:.0f}%"
                txt_tr = f"{100.0 * v_tr:.0f}%"
                col_ep = "#ffffff" if v_ep >= 0.62 else "#0f172a"
                col_tr = "#ffffff" if v_tr >= 0.62 else "#7c2d12"
                if r == c:
                    col_ep = "#ffffff"
                    col_tr = "#ffffff"
                ax.text(x0 - 0.16, y0 + 0.16, txt_ep, ha="center", va="center", fontsize=8.8 * text_scale, color=col_ep)
                ax.text(x0 + 0.16, y0 - 0.16, txt_tr, ha="center", va="center", fontsize=8.8 * text_scale, color=col_tr)

        x_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        y_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        if i == 0:
            ax.set_yticklabels(y_labels)
            ax.set_ylabel("True family", fontsize=11.0 * text_scale)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Nearest-neighbor family", fontsize=11.0 * text_scale)
        f1_ep = f1_by_panel_endpoint.get(key, float("nan"))
        f1_tr = f1_by_panel_trajectory.get(key, float("nan"))
        _ = (f1_ep, f1_tr)  # kept for summary title only
        ax.set_title(str(panel["title"]), fontsize=12.6 * text_scale, fontweight="semibold", pad=10)
        ax.tick_params(axis="both", labelsize=10.1 * text_scale, length=3.0, width=0.8, color="#444444")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#c9cdd2")
        ax.set_aspect("equal")
        ax.set_facecolor("white")

    endpoint_f1 = [f1_by_panel_endpoint.get(k, float("nan")) for k in ("A", "B", "C")]
    endpoint_f1 = [float(v) for v in endpoint_f1 if np.isfinite(float(v))]
    trajectory_f1 = [f1_by_panel_trajectory.get(k, float("nan")) for k in ("A", "B", "C")]
    trajectory_f1 = [float(v) for v in trajectory_f1 if np.isfinite(float(v))]
    endpoint_txt = f"{float(np.mean(endpoint_f1)):.2f}" if endpoint_f1 else "NA"
    trajectory_txt = f"{float(np.mean(trajectory_f1)):.2f}" if trajectory_f1 else "NA"
    # One-line centered title with colored triangle symbols and no overlap.
    title_fs = 13.0 * text_scale
    y_top = 0.975
    x_mid = 0.5
    fig.text(
        x_mid,
        y_top,
        "Family Confusion Matrices: Endpoint (",
        ha="right",
        va="center",
        fontsize=title_fs,
        fontweight="semibold",
        color="#0f172a",
    )
    fig.text(x_mid, y_top, "▲", ha="left", va="center", fontsize=title_fs, fontweight="bold", color=endpoint_cmap(0.82))
    fig.text(
        x_mid + 0.010,
        y_top,
        ") vs Trajectory (",
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="semibold",
        color="#0f172a",
    )
    fig.text(
        x_mid + 0.136,
        y_top,
        "▲",
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="bold",
        color=trajectory_cmap(0.82),
    )
    fig.text(
        x_mid + 0.146,
        y_top,
        f") | Macro-F1: endpoint={endpoint_txt}, trajectory={trajectory_txt}",
        ha="left",
        va="center",
        fontsize=title_fs,
        fontweight="semibold",
        color="#0f172a",
    )
    fig.savefig(args.out / "sanity10_family_confusion_abc.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_confusion_abc.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    # Variant figure: same split-triangle layout, but one unified value-intensity palette.
    figv, axesv = plt.subplots(1, 3, figsize=(16.4, 6.1), constrained_layout=False)
    figv.patch.set_facecolor("white")
    figv.subplots_adjust(left=0.065, right=0.992, top=0.86, bottom=0.135, wspace=0.12)
    mono_cmap = plt.get_cmap("YlGnBu")
    mono_norm = mcolors.Normalize(vmin=0.0, vmax=1.0)

    for i, panel in enumerate(panel_specs):
        ax = axesv[i]
        key = str(panel["letter"])
        conf_ep = conf_by_panel_endpoint.get(key)
        conf_tr = conf_by_panel_trajectory.get(key)
        if conf_ep is None or conf_tr is None:
            ax.axis("off")
            continue
        arr_ep = conf_ep.to_numpy(dtype=float)
        arr_tr = conf_tr.to_numpy(dtype=float)
        n = arr_ep.shape[0]
        for r in range(n):
            for c in range(n):
                x0, y0 = float(c), float(r)
                v_ep = float(arr_ep[r, c])
                v_tr = float(arr_tr[r, c])
                c_ep = mono_cmap(0.18 + 0.76 * mono_norm(v_ep))
                c_tr = mono_cmap(0.18 + 0.76 * mono_norm(v_tr))
                tri_bl = Polygon(
                    [[x0 - 0.5, y0 - 0.5], [x0 - 0.5, y0 + 0.5], [x0 + 0.5, y0 + 0.5]],
                    closed=True,
                    facecolor=c_ep,
                    edgecolor="none",
                )
                tri_tr = Polygon(
                    [[x0 - 0.5, y0 - 0.5], [x0 + 0.5, y0 - 0.5], [x0 + 0.5, y0 + 0.5]],
                    closed=True,
                    facecolor=c_tr,
                    edgecolor="none",
                )
                ax.add_patch(tri_bl)
                ax.add_patch(tri_tr)
                ax.plot(
                    [x0 - 0.5, x0 + 0.5],
                    [y0 - 0.5, y0 + 0.5],
                    color="#111111",
                    linewidth=1.05,
                    alpha=0.9,
                    solid_capstyle="round",
                    zorder=4,
                )
                ax.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 1.0, 1.0, fill=False, edgecolor="#ffffff", linewidth=2.16))
                # Curved arrow per cell to show endpoint -> trajectory change.
                d = float(v_tr - v_ep)
                if abs(d) >= 0.005:
                    rad = -0.26 if d > 0 else 0.26
                    arr_col = "#065f46" if d > 0 else "#7f1d1d"
                    arr_alpha = 0.75 + 0.20 * min(1.0, abs(d) / 0.30)
                    arrow = FancyArrowPatch(
                        (x0 - 0.38, y0 + 0.08),
                        (x0 + 0.08, y0 - 0.38),
                        connectionstyle=f"arc3,rad={rad}",
                        arrowstyle="-|>",
                        mutation_scale=3.9,
                        linewidth=0.975,
                        color=arr_col,
                        alpha=arr_alpha,
                        zorder=5,
                    )
                    ax.add_patch(arrow)
                txt_ep = f"{100.0 * v_ep:.0f}%"
                txt_tr = f"{100.0 * v_tr:.0f}%"
                col_ep = "#ffffff" if v_ep >= 0.62 else "#0f172a"
                col_tr = "#ffffff" if v_tr >= 0.62 else "#0f172a"
                if r == c:
                    col_ep = "#ffffff"
                    col_tr = "#ffffff"
                ax.text(x0 - 0.16, y0 + 0.16, txt_ep, ha="center", va="center", fontsize=8.8 * text_scale, color=col_ep)
                ax.text(x0 + 0.16, y0 - 0.16, txt_tr, ha="center", va="center", fontsize=8.8 * text_scale, color=col_tr)

        # Highlight true positives (diagonal) with inset red borders.
        for d in range(n):
            x0, y0 = float(d), float(d)
            ax.add_patch(
                Rectangle(
                    (x0 - 0.46, y0 - 0.46),
                    0.92,
                    0.92,
                    fill=False,
                    edgecolor="#b91c1c",
                    linewidth=4.2,
                    zorder=12,
                    joinstyle="round",
                )
            )

        x_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        y_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        if i == 0:
            ax.set_yticklabels(y_labels)
            ax.set_ylabel("True family", fontsize=11.0 * text_scale)
        else:
            ax.set_yticklabels([])
        ax.set_xlabel("Nearest-neighbor family", fontsize=11.0 * text_scale)
        ax.set_title(str(panel["title"]), fontsize=12.6 * text_scale, fontweight="semibold", pad=10)
        ax.tick_params(axis="both", labelsize=10.1 * text_scale, length=3.0, width=0.8, color="#444444")
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#c9cdd2")
        ax.set_aspect("equal")
        ax.set_facecolor("white")

    figv.suptitle(
        f"Family Confusion Matrices | Macro-F1: endpoint={endpoint_txt}, trajectory={trajectory_txt}",
        fontsize=13.0 * text_scale,
        y=0.975,
        fontweight="semibold",
        color="#0f172a",
    )
    figv.savefig(args.out / "sanity10_familyconfusion2cba.png", dpi=args.dpi, bbox_inches="tight")
    figv.savefig(args.out / "sanity10_familyconfusion2cba.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(figv)


def _make_family_trajectory_abc_figure(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    cond_day = _prepare_analysis_raw(raw, args)
    if cond_day.empty:
        return

    preferred_order = [
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    ]
    family_label = {
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "Spatial",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "Cell #",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "Dynamics",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "Ascorbic Acid",
    }
    # Okabe-Ito inspired, colorblind-friendly set.
    family_color = {
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "#0072B2",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "#E69F00",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "#009E73",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "#CC79A7",
    }
    present = set(cond_day["perturbation_family"].astype(str).unique().tolist())
    family_order = [fam for fam in preferred_order if fam in present]
    if not family_order:
        return

    panel_specs = [
        {"letter": "A", "feature": "ALP", "title": "ALP (Day 0-10)", "day_end": 10},
        {"letter": "B", "feature": "collagen", "title": "Collagen (Day 0-14)", "day_end": 14},
        {"letter": "C", "feature": "mineral", "title": "Mineral (Day 0-28)", "day_end": 28},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 4.7), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.19, wspace=0.20)

    for ax, panel in zip(axes, panel_specs):
        day_end = int(panel["day_end"])
        feature = str(panel["feature"])
        sub = cond_day.loc[
            (cond_day["day"].astype(int) >= 0) & (cond_day["day"].astype(int) <= day_end)
        ].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        for fam in family_order:
            fam_df = sub.loc[sub["perturbation_family"].astype(str) == fam].copy()
            if fam_df.empty:
                continue
            q = (
                fam_df.groupby("day", observed=True)[feature]
                .quantile([0.25, 0.50, 0.75])
                .unstack()
                .sort_index()
            )
            days = q.index.to_numpy(dtype=float)
            q1 = q[0.25].to_numpy(dtype=float)
            q2 = q[0.50].to_numpy(dtype=float)
            q3 = q[0.75].to_numpy(dtype=float)
            color = family_color[fam]
            ax.fill_between(days, q1, q3, color=color, alpha=0.14, linewidth=0.0, zorder=1)
            ax.plot(days, q2, color=color, linewidth=2.1, marker="o", markersize=4.2, zorder=2)

        ax.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Day", fontsize=10.6)
        if feature == "ALP":
            ax.set_ylabel("ALP (a.u.)", fontsize=10.6)
        elif feature == "collagen":
            ax.set_ylabel("Collagen (a.u.)", fontsize=10.6)
        else:
            ax.set_ylabel("Mineral (a.u.)", fontsize=10.6)
        if day_end >= 24:
            ax.set_xticks(np.arange(0, day_end + 1, 4))
        elif day_end > 10:
            ax.set_xticks(np.arange(0, day_end + 1, 2))
        else:
            ax.set_xticks(np.arange(0, day_end + 1, 2))
        ax.set_xlim(-0.3, day_end + 0.3)
        ax.tick_params(axis="both", labelsize=9.4, length=3.0, width=0.8, color="#444444")
        ax.set_title(panel["title"], fontsize=11.8, fontweight="semibold", pad=8)
        ax.text(
            -0.12,
            1.06,
            panel["letter"],
            transform=ax.transAxes,
            fontsize=13.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111111",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_edgecolor("#c9cdd2")

    legend_handles = [
        Line2D([0], [0], color=family_color[f], lw=2.2, marker="o", markersize=4.6, label=family_label[f])
        for f in family_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.98),
        ncol=min(4, len(legend_handles)),
        frameon=False,
        fontsize=9.8,
    )
    fig.suptitle("Family-Specific Trajectories from Day 0 to Assay Endpoint", fontsize=13.8, y=0.965, fontweight="semibold")
    fig.savefig(args.out / "sanity10_family_trajectory_abc.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_trajectory_abc.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_trajectory_abc.svg", bbox_inches="tight")
    plt.close(fig)


def _make_identify_test_new_figures(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    cond_day = _prepare_family_cond_day_full(raw, args)
    if cond_day.empty:
        return
    work = cond_day.loc[cond_day["perturbation_family"].astype(str) != sc10.TYPE_BASELINE].copy()
    if work.empty:
        return

    family_order = [
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    ]
    present = set(work["perturbation_family"].astype(str).unique().tolist())
    family_order = [fam for fam in family_order if fam in present]
    if len(family_order) < 2:
        return
    family_label = {
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "Spatial",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "Cell #",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "Dynamics",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "Ascorbic Acid",
    }
    marker_specs = [
        ("ALP", 10, "A"),
        ("collagen", 14, "B"),
        ("mineral", 28, "C"),
    ]

    def _build_nn(marker: str, day_end: int, mode: str) -> tuple[np.ndarray, np.ndarray, pd.DataFrame] | None:
        if mode == "endpoint":
            days = [day_end]
        else:
            days = sorted(int(d) for d in work["day"].astype(int).unique().tolist() if 0 <= int(d) <= day_end)
        if not days:
            return None
        sub = work.loc[work["day"].astype(int).isin(days), ["condition_id", "day", "perturbation_family", marker]].copy()
        if sub.empty:
            return None
        piv = (
            sub.pivot_table(
                index=["condition_id", "perturbation_family"],
                columns="day",
                values=marker,
                aggfunc="mean",
            )
            .reindex(columns=days)
            .dropna(axis=0, how="any")
        )
        if piv.empty:
            return None
        x_raw = piv.to_numpy(dtype=float)
        mu = np.nanmean(x_raw, axis=0)
        sd = np.nanstd(x_raw, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        x = (x_raw - mu[None, :]) / sd[None, :]
        y = piv.index.get_level_values("perturbation_family").astype(str).to_numpy()
        if len(y) < 3:
            return None
        d = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(d, np.inf)
        pred = y[np.argmin(d, axis=1)]
        conf = pd.crosstab(
            pd.Series(y, name="true"),
            pd.Series(pred, name="pred"),
            normalize="index",
        ).reindex(index=family_order, columns=family_order, fill_value=0.0)
        return y.astype(str), pred.astype(str), conf

    def _macro_f1(conf: pd.DataFrame) -> float:
        c = conf.reindex(index=family_order, columns=family_order, fill_value=0.0)
        vals: list[float] = []
        for fam in family_order:
            tp = float(c.loc[fam, fam])
            fp = float(c[fam].sum() - tp)
            fn = float(c.loc[fam].sum() - tp)
            den = 2.0 * tp + fp + fn
            vals.append((2.0 * tp / den) if den > 0.0 else 0.0)
        return float(np.mean(vals)) if vals else float("nan")

    results: dict[str, dict[str, object]] = {}
    metric_rows: list[dict[str, float | str]] = []
    for marker, day_end, panel_letter in marker_specs:
        ep = _build_nn(marker, day_end, "endpoint")
        tr = _build_nn(marker, day_end, "trajectory")
        if ep is None or tr is None:
            continue
        y_ep, p_ep, conf_ep = ep
        y_tr, p_tr, conf_tr = tr
        # Align to common conditions by family order only (both derived from same table shape typically).
        f1_ep = _macro_f1(conf_ep)
        f1_tr = _macro_f1(conf_tr)
        results[marker] = {
            "panel": panel_letter,
            "day_end": int(day_end),
            "conf_endpoint": conf_ep,
            "conf_trajectory": conf_tr,
            "true_endpoint": y_ep,
            "pred_endpoint": p_ep,
            "true_trajectory": y_tr,
            "pred_trajectory": p_tr,
            "macro_f1_endpoint": f1_ep,
            "macro_f1_trajectory": f1_tr,
        }
        metric_rows.append(
            {
                "marker": marker,
                "day_end": int(day_end),
                "macro_f1_endpoint": float(f1_ep),
                "macro_f1_trajectory": float(f1_tr),
                "delta_macro_f1": float(f1_tr - f1_ep),
            }
        )
    if not results:
        return
    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(args.out / "sanity10_identify_test_metrics.csv", index=False)

    # Figure 1: Endpoint-only confusion A-C (limited separability).
    fig1, axes = plt.subplots(1, 3, figsize=(15.8, 5.6), constrained_layout=False)
    fig1.subplots_adjust(left=0.07, right=0.92, top=0.84, bottom=0.12, wspace=0.22)
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    mappable = None
    for ax, (marker, _, letter) in zip(axes, marker_specs):
        if marker not in results:
            ax.axis("off")
            continue
        conf_ep = results[marker]["conf_endpoint"]  # type: ignore[index]
        f1_ep = float(results[marker]["macro_f1_endpoint"])  # type: ignore[index]
        arr = conf_ep.to_numpy(dtype=float)
        mappable = ax.imshow(arr, cmap=cmap, norm=norm, interpolation="nearest")
        lbls = [family_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in family_order]
        ax.set_xticks(np.arange(len(family_order)), lbls)
        ax.set_yticks(np.arange(len(family_order)), lbls if letter == "A" else [])
        if letter == "A":
            ax.set_ylabel("True family", fontsize=11.0)
        ax.set_xlabel("Nearest-neighbor family", fontsize=11.0)
        pretty = "Collagen" if marker == "collagen" else ("Mineral" if marker == "mineral" else "ALP")
        day_end = int(results[marker]["day_end"])  # type: ignore[index]
        ax.set_title(f"{pretty} @ Day {day_end} (Macro-F1={f1_ep:.2f})", fontsize=12.3, fontweight="semibold", pad=9)
        ax.text(-0.14, 1.06, letter, transform=ax.transAxes, fontsize=13.2, fontweight="bold", ha="left", va="bottom")
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                v = float(arr[i, j])
                ax.text(j, i, f"{100.0*v:.0f}%", ha="center", va="center", fontsize=9.8, color="#111111")
        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#edf2f7", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#c9cdd2")
        ax.set_aspect("equal")
    if mappable is not None:
        cax = fig1.add_axes([0.935, 0.22, 0.013, 0.56])
        cbar = fig1.colorbar(mappable, cax=cax)
        cbar.set_label("Row-normalized fraction", fontsize=10.6)
        cbar.ax.tick_params(labelsize=9.2)
    fig1.suptitle("Endpoint Readouts Show Limited Family Separability", fontsize=16.5, y=0.95, fontweight="semibold")
    fig1.savefig(args.out / "sanity10_identify_test_fig1_endpoint_confusion.png", dpi=args.dpi, bbox_inches="tight")
    fig1.savefig(args.out / "sanity10_identify_test_fig1_endpoint_confusion.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig1)

    # Figure 2: Macro-F1 lift from endpoint -> trajectory.
    fig2, ax2 = plt.subplots(figsize=(8.8, 4.8), constrained_layout=False)
    marker_pretty = []
    ep_vals = []
    tr_vals = []
    for marker, _, _ in marker_specs:
        if marker not in results:
            continue
        marker_pretty.append("Collagen" if marker == "collagen" else ("Mineral" if marker == "mineral" else "ALP"))
        ep_vals.append(float(results[marker]["macro_f1_endpoint"]))  # type: ignore[index]
        tr_vals.append(float(results[marker]["macro_f1_trajectory"]))  # type: ignore[index]
    y = np.arange(len(marker_pretty), dtype=float)
    endpoint_color = "#a8c4d7"
    trajectory_color = "#e3a3a3"
    for i in range(len(marker_pretty)):
        ax2.plot([ep_vals[i], tr_vals[i]], [y[i], y[i]], color="#6b7280", linewidth=1.5, zorder=1)
    ax2.scatter(ep_vals, y, color=endpoint_color, s=84, edgecolors="#4b5563", linewidths=0.7, label="Endpoint", zorder=2)
    ax2.scatter(tr_vals, y, color=trajectory_color, s=84, edgecolors="#4b5563", linewidths=0.7, label="Trajectory", zorder=3)
    for i in range(len(marker_pretty)):
        ax2.text(max(ep_vals[i], tr_vals[i]) + 0.012, y[i], f"Delta {tr_vals[i]-ep_vals[i]:+.2f}", va="center", ha="left", fontsize=9.4, color="#111111")
    ax2.set_yticks(y, marker_pretty)
    ax2.set_xlabel("Macro-F1 (LOO 1-NN)")
    ax2.set_xlim(0.0, min(1.0, float(max(max(ep_vals), max(tr_vals)) + 0.20)))
    ax2.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.65)
    ax2.set_title("Trajectory Features Increase Class Separability", fontsize=13.2, fontweight="semibold", pad=9)
    ax2.legend(frameon=False, fontsize=9.1, loc="lower right")
    fig2.savefig(args.out / "sanity10_identify_test_fig2_macrof1_lift.png", dpi=args.dpi, bbox_inches="tight")
    fig2.savefig(args.out / "sanity10_identify_test_fig2_macrof1_lift.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)

    def _recall_and_se(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        recalls: list[float] = []
        ses: list[float] = []
        for fam in family_order:
            mask = y_true == fam
            n = int(np.sum(mask))
            if n <= 0:
                recalls.append(0.0)
                ses.append(0.0)
                continue
            p = float(np.mean(y_pred[mask] == fam))
            recalls.append(p)
            ses.append(float(np.sqrt(max(p * (1.0 - p), 0.0) / n)))
        return np.array(recalls, dtype=float), np.array(ses, dtype=float)

    # Figure 3: Family-level recall (diagonal) endpoint vs trajectory by marker.
    fig3, axes3 = plt.subplots(1, 3, figsize=(15.8, 4.9), constrained_layout=False)
    fig3.subplots_adjust(left=0.06, right=0.99, top=0.86, bottom=0.22, wspace=0.23)
    x = np.arange(len(family_order), dtype=float)
    w = 0.36
    ep_bar = "#4C78A8"
    tr_bar = "#E69F00"
    edge = "#1f2937"
    err_kw = {"elinewidth": 1.2, "ecolor": "#111827", "capsize": 3, "capthick": 1.2}
    for ax, (marker, day_end, letter) in zip(axes3, marker_specs):
        if marker not in results:
            ax.axis("off")
            continue
        conf_ep = results[marker]["conf_endpoint"]  # type: ignore[index]
        conf_tr = results[marker]["conf_trajectory"]  # type: ignore[index]
        diag_ep = np.diag(conf_ep.to_numpy(dtype=float))
        diag_tr = np.diag(conf_tr.to_numpy(dtype=float))
        y_ep = np.asarray(results[marker]["true_endpoint"], dtype=str)  # type: ignore[index]
        p_ep = np.asarray(results[marker]["pred_endpoint"], dtype=str)  # type: ignore[index]
        y_tr = np.asarray(results[marker]["true_trajectory"], dtype=str)  # type: ignore[index]
        p_tr = np.asarray(results[marker]["pred_trajectory"], dtype=str)  # type: ignore[index]
        rec_ep, se_ep = _recall_and_se(y_ep, p_ep)
        rec_tr, se_tr = _recall_and_se(y_tr, p_tr)
        b1 = ax.bar(
            x - w / 2,
            100.0 * rec_ep,
            yerr=100.0 * se_ep,
            width=w,
            color=ep_bar,
            edgecolor=edge,
            linewidth=0.9,
            label="Endpoint",
            error_kw=err_kw,
            zorder=3,
        )
        b2 = ax.bar(
            x + w / 2,
            100.0 * rec_tr,
            yerr=100.0 * se_tr,
            width=w,
            color=tr_bar,
            edgecolor=edge,
            linewidth=0.9,
            label="Trajectory",
            error_kw=err_kw,
            zorder=3,
        )
        ax.set_xticks(x, [family_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in family_order])
        ax.set_ylim(0.0, 100.0)
        ax.grid(axis="y", color="#cbd5e1", linewidth=0.8, alpha=0.85, zorder=0)
        ax.set_facecolor("#ffffff")
        if letter == "A":
            ax.set_ylabel("Per-family recall (%)")
        pretty = "Collagen" if marker == "collagen" else ("Mineral" if marker == "mineral" else "ALP")
        ax.set_title(f"{pretty} (Day 0-{day_end} vs Day {day_end})", fontsize=11.9, fontweight="semibold")
        ax.text(-0.15, 1.05, letter, transform=ax.transAxes, fontsize=12.8, fontweight="bold", ha="left", va="bottom")
    axes3[0].legend(frameon=False, fontsize=8.9, loc="upper left")
    fig3.suptitle("Family-Specific Recovery Improves with Trajectory Windows", fontsize=14.2, y=0.97, fontweight="semibold")
    fig3.savefig(args.out / "sanity10_identify_test_fig3_family_recall_lift.png", dpi=args.dpi, bbox_inches="tight")
    fig3.savefig(args.out / "sanity10_identify_test_fig3_family_recall_lift.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig3)

    # Extra creative figure: per-family improvement (trajectory - endpoint) in recall.
    rows: list[dict[str, object]] = []
    marker_label = {"ALP": "ALP", "collagen": "Collagen", "mineral": "Mineral"}
    for marker, _, _ in marker_specs:
        if marker not in results:
            continue
        y_ep = np.asarray(results[marker]["true_endpoint"], dtype=str)  # type: ignore[index]
        p_ep = np.asarray(results[marker]["pred_endpoint"], dtype=str)  # type: ignore[index]
        y_tr = np.asarray(results[marker]["true_trajectory"], dtype=str)  # type: ignore[index]
        p_tr = np.asarray(results[marker]["pred_trajectory"], dtype=str)  # type: ignore[index]
        rec_ep, se_ep = _recall_and_se(y_ep, p_ep)
        rec_tr, se_tr = _recall_and_se(y_tr, p_tr)
        for fam, rep, rtr, sep, str_ in zip(family_order, rec_ep, rec_tr, se_ep, se_tr):
            rows.append(
                {
                    "marker": marker_label.get(marker, marker),
                    "family": family_label.get(fam, fam),
                    "delta_pct": 100.0 * (rtr - rep),
                    "se_delta_pct": 100.0 * np.sqrt(sep * sep + str_ * str_),
                }
            )
    if rows:
        lift_df = pd.DataFrame(rows)
        figx, axx = plt.subplots(figsize=(11.2, 5.8), constrained_layout=False)
        figx.patch.set_facecolor("white")
        axx.set_facecolor("white")
        marker_colors = {"ALP": "#4C78A8", "Collagen": "#009E73", "Mineral": "#E69F00"}
        fams = [family_label[f] for f in family_order]
        x0 = np.arange(len(fams), dtype=float)
        offsets = {"ALP": -0.22, "Collagen": 0.0, "Mineral": 0.22}
        for m in ["ALP", "Collagen", "Mineral"]:
            sub = lift_df[lift_df["marker"] == m].copy()
            if sub.empty:
                continue
            sub["family"] = pd.Categorical(sub["family"], categories=fams, ordered=True)
            sub = sub.sort_values("family")
            x = x0 + offsets[m]
            y = sub["delta_pct"].to_numpy(dtype=float)
            e = sub["se_delta_pct"].to_numpy(dtype=float)
            axx.plot(x, y, color=marker_colors[m], linewidth=2.0, alpha=0.95, zorder=2)
            axx.errorbar(
                x,
                y,
                yerr=e,
                fmt="o",
                markersize=7.5,
                color=marker_colors[m],
                ecolor="#111827",
                elinewidth=1.1,
                capsize=3,
                capthick=1.1,
                markeredgecolor="#1f2937",
                markeredgewidth=0.6,
                zorder=3,
                label=m,
            )
        axx.axhline(0.0, color="#334155", linewidth=1.2, linestyle="--", alpha=0.9)
        axx.set_xticks(x0, [f.replace("Ascorbic Acid", "Ascorbic\nAcid") for f in fams])
        axx.set_ylabel("Recall lift (percentage points)")
        axx.set_xlabel("Perturbation family")
        axx.set_title("Trajectory Recall Gain by Family", fontsize=14.0, fontweight="semibold", pad=10)
        axx.grid(axis="y", color="#cbd5e1", linewidth=0.9, alpha=0.85)
        axx.legend(frameon=False, ncol=3, loc="upper left")
        figx.savefig(args.out / "sanity10_identify_test_extraplot1famtra.png", dpi=args.dpi, bbox_inches="tight")
        figx.savefig(args.out / "sanity10_identify_test_extraplot1famtra.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(figx)


def _make_option2_distance_to_baseline_abc(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    cond_day = _prepare_family_cond_day_full(raw, args)
    if cond_day.empty:
        return

    family_order = [
        getattr(sc10, "TYPE_BASELINE", "0_baseline"),
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    ]
    family_label = {
        getattr(sc10, "TYPE_BASELINE", "0_baseline"): "Baseline",
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "Spatial",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "Cell #",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "Dynamics",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "Ascorbic Acid",
    }
    family_color = {
        getattr(sc10, "TYPE_BASELINE", "0_baseline"): "#111111",
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "#0072B2",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "#E69F00",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "#009E73",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "#CC79A7",
    }
    present = set(cond_day["perturbation_family"].astype(str).unique().tolist())
    family_order = [fam for fam in family_order if fam in present]
    if not family_order:
        return

    panel_specs = [
        {"letter": "A", "feature": "ALP", "title": "Day 0-10 ALP", "day_end": 10},
        {"letter": "B", "feature": "collagen", "title": "Day 0-14 Collagen", "day_end": 14},
        {"letter": "C", "feature": "mineral", "title": "Day 0-28 Mineral", "day_end": 28},
    ]

    fig, axes = plt.subplots(1, 3, figsize=(13.8, 5.0), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.99, top=0.84, bottom=0.26, wspace=0.20)

    for ax, panel in zip(axes, panel_specs):
        day_end = int(panel["day_end"])
        feature = str(panel["feature"])
        sub = cond_day.loc[(cond_day["day"].astype(int) >= 0) & (cond_day["day"].astype(int) <= day_end)].copy()
        if sub.empty:
            ax.set_visible(False)
            continue

        y = sub[["condition_id", "day", "perturbation_family", feature]].copy().rename(columns={feature: "value"})

        for fam in family_order:
            fam_df = y.loc[y["perturbation_family"].astype(str) == fam].copy()
            if fam_df.empty:
                continue
            q = (
                fam_df.groupby("day", observed=True)["value"]
                .quantile([0.25, 0.50, 0.75])
                .unstack()
                .sort_index()
            )
            days = q.index.to_numpy(dtype=float)
            q1 = q[0.25].to_numpy(dtype=float)
            q2 = q[0.50].to_numpy(dtype=float)
            q3 = q[0.75].to_numpy(dtype=float)
            color = family_color[fam]
            if str(fam) == getattr(sc10, "TYPE_BASELINE", "0_baseline"):
                ax.plot(days, q2, color=color, linewidth=1.6, marker="o", markersize=3.8, zorder=3)
            else:
                ax.fill_between(days, q1, q3, color=color, alpha=0.14, linewidth=0.0, zorder=1)
                ax.plot(days, q2, color=color, linewidth=2.1, marker="o", markersize=4.1, zorder=2)

        ax.set_xlabel("Day", fontsize=10.6)
        if feature == "ALP":
            ax.set_ylabel("ALP (a.u.)", fontsize=10.4)
        elif feature == "collagen":
            ax.set_ylabel("Collagen (a.u.)", fontsize=10.4)
        else:
            ax.set_ylabel("Mineral (a.u.)", fontsize=10.4)
        if day_end >= 24:
            ax.set_xticks(np.arange(0, day_end + 1, 4))
        elif day_end > 10:
            ax.set_xticks(np.arange(0, day_end + 1, 2))
        else:
            ax.set_xticks(np.arange(0, day_end + 1, 2))
        ax.set_xlim(-0.3, day_end + 0.3)
        ax.tick_params(axis="both", labelsize=9.3, length=3.0, width=0.8, color="#444444")
        ax.set_title(panel["title"], fontsize=11.7, fontweight="semibold", pad=8)
        ax.text(
            -0.12,
            1.06,
            panel["letter"],
            transform=ax.transAxes,
            fontsize=13.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111111",
        )
        for spine in ax.spines.values():
            spine.set_linewidth(0.85)
            spine.set_edgecolor("#c9cdd2")

    legend_handles = [
        Line2D([0], [0], color=family_color[f], lw=2.2, marker="o", markersize=4.6, label=family_label[f])
        for f in family_order
    ]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.06),
        ncol=max(1, len(legend_handles)),
        frameon=False,
        fontsize=9.8,
    )
    fig.suptitle("Endpoint-Value Trajectories by Family", fontsize=13.8, y=0.965, fontweight="semibold")
    fig.savefig(args.out / "sanity10_option2_distance_to_baseline.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_option2_distance_to_baseline.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def _make_trajectory_added_value_figures(raw: pd.DataFrame, args: argparse.Namespace) -> None:
    cond_day = _prepare_family_cond_day_full(raw, args)
    if cond_day.empty:
        return

    work = cond_day.loc[cond_day["perturbation_family"].astype(str) != sc10.TYPE_BASELINE].copy()
    if work.empty:
        return

    family_order = [
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"),
        getattr(sc10, "TYPE_CELLS", "2_cell_number"),
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"),
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"),
    ]
    present = set(work["perturbation_family"].astype(str).unique().tolist())
    family_order = [fam for fam in family_order if fam in present]
    if len(family_order) < 2:
        return

    family_label = {
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "Spatial",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "Cell #",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "Dynamics",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "Ascorbic Acid",
    }
    family_color = {
        getattr(sc10, "TYPE_SPATIAL", "1_spatial_pattern"): "#0072B2",
        getattr(sc10, "TYPE_CELLS", "2_cell_number"): "#E69F00",
        getattr(sc10, "TYPE_DYNAMICS", "3_cell_dynamics"): "#009E73",
        getattr(sc10, "TYPE_AA", "4_ascorbic_acid"): "#CC79A7",
    }

    marker_end = {"ALP": 10, "collagen": 14, "mineral": 28}

    base = (
        work[["condition_id", "perturbation_family"]]
        .drop_duplicates(subset=["condition_id"])
        .set_index("condition_id")
    )
    endpoint_feats = base.copy()
    traj_feats = base.copy()

    for marker, day_end in marker_end.items():
        sub = work.loc[(work["day"].astype(int) >= 0) & (work["day"].astype(int) <= day_end), ["condition_id", "day", marker]].copy()
        if sub.empty:
            continue
        pivot = sub.pivot_table(index="condition_id", columns="day", values=marker, aggfunc="mean").sort_index(axis=1)
        if day_end not in pivot.columns:
            continue
        endpoint_feats[f"{marker}_end"] = pd.to_numeric(pivot[day_end], errors="coerce")

        def _traj_stats(row: pd.Series) -> pd.Series:
            s = row.dropna()
            if s.empty:
                return pd.Series(
                    {
                        f"{marker}_end": np.nan,
                        f"{marker}_auc_norm": np.nan,
                        f"{marker}_slope": np.nan,
                        f"{marker}_peak": np.nan,
                        f"{marker}_tpeak": np.nan,
                    }
                )
            x = s.index.to_numpy(dtype=float)
            y = s.to_numpy(dtype=float)
            span = max(float(x[-1] - x[0]), 1.0)
            auc_norm = float(np.trapezoid(y, x) / span) if len(y) > 1 else float(y[-1])
            slope = float((y[-1] - y[0]) / span) if len(y) > 1 else 0.0
            peak_idx = int(np.argmax(y))
            return pd.Series(
                {
                    f"{marker}_end": float(y[-1]),
                    f"{marker}_auc_norm": auc_norm,
                    f"{marker}_slope": slope,
                    f"{marker}_peak": float(y[peak_idx]),
                    f"{marker}_tpeak": float(x[peak_idx]),
                }
            )

        stats = pivot.apply(_traj_stats, axis=1)
        traj_feats = traj_feats.join(stats, how="left")

    endpoint_cols = [c for c in endpoint_feats.columns if c != "perturbation_family"]
    traj_cols = [c for c in traj_feats.columns if c != "perturbation_family"]
    endpoint_df = endpoint_feats.dropna(subset=endpoint_cols).copy()
    traj_df = traj_feats.dropna(subset=traj_cols).copy()
    common_ids = sorted(set(endpoint_df.index.astype(str)).intersection(set(traj_df.index.astype(str))))
    if len(common_ids) < 8:
        return
    endpoint_df = endpoint_df.loc[common_ids].copy()
    traj_df = traj_df.loc[common_ids].copy()

    y_labels = endpoint_df["perturbation_family"].astype(str).to_numpy()

    def _zscore_frame(df_in: pd.DataFrame, cols: list[str]) -> np.ndarray:
        x = df_in[cols].to_numpy(dtype=float)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd = np.where(sd > 1e-12, sd, 1.0)
        return (x - mu[None, :]) / sd[None, :]

    def _loo_knn(x: np.ndarray, labels: np.ndarray, classes: list[str], k: int = 5) -> tuple[np.ndarray, np.ndarray]:
        n = len(labels)
        dmat = np.linalg.norm(x[:, None, :] - x[None, :, :], axis=2)
        np.fill_diagonal(dmat, np.inf)
        k_eff = max(1, min(k, n - 1))
        nn_idx = np.argsort(dmat, axis=1)[:, :k_eff]
        pred = np.empty(n, dtype=object)
        p_true = np.zeros(n, dtype=float)
        class_to_i = {c: i for i, c in enumerate(classes)}
        for i in range(n):
            w = 1.0 / np.maximum(dmat[i, nn_idx[i]], 1e-9)
            probs = np.zeros(len(classes), dtype=float)
            for j, idx in enumerate(nn_idx[i]):
                cls = str(labels[idx])
                probs[class_to_i[cls]] += w[j]
            total = float(np.sum(probs))
            if total > 0.0:
                probs = probs / total
            pred[i] = classes[int(np.argmax(probs))]
            p_true[i] = float(probs[class_to_i[str(labels[i])]])
        return pred.astype(str), p_true

    x_endpoint = _zscore_frame(endpoint_df, endpoint_cols)
    x_traj = _zscore_frame(traj_df, traj_cols)
    pred_end, ptrue_end = _loo_knn(x_endpoint, y_labels, family_order, k=5)
    pred_trj, ptrue_trj = _loo_knn(x_traj, y_labels, family_order, k=5)

    shift_df = pd.DataFrame(
        {
            "condition_id": common_ids,
            "true_family": y_labels,
            "pred_endpoint": pred_end,
            "pred_trajectory": pred_trj,
            "p_true_endpoint": ptrue_end,
            "p_true_trajectory": ptrue_trj,
            "delta_p_true": ptrue_trj - ptrue_end,
        }
    )
    shift_df.to_csv(args.out / "sanity10_trajectory_added_value_shift.csv", index=False)

    conf_end = pd.crosstab(
        pd.Series(y_labels, name="true"),
        pd.Series(pred_end, name="pred"),
        normalize="index",
    ).reindex(index=family_order, columns=family_order, fill_value=0.0)
    conf_trj = pd.crosstab(
        pd.Series(y_labels, name="true"),
        pd.Series(pred_trj, name="pred"),
        normalize="index",
    ).reindex(index=family_order, columns=family_order, fill_value=0.0)
    conf_delta = conf_trj - conf_end
    conf_delta.to_csv(args.out / "sanity10_trajectory_added_value_confusion_delta.csv")

    def _fisher_score(x: np.ndarray, labels: np.ndarray) -> float:
        mu = float(np.nanmean(x))
        num = 0.0
        den = 0.0
        for fam in family_order:
            vals = x[labels == fam]
            if vals.size == 0:
                continue
            m = float(np.nanmean(vals))
            v = float(np.nanvar(vals))
            n = float(vals.size)
            num += n * (m - mu) ** 2
            den += n * v
        return float(num / max(den, 1e-12))

    gain_rows: list[dict[str, float | str]] = []
    gain_detail_rows: list[dict[str, float | str]] = []
    for marker in ("ALP", "collagen", "mineral"):
        end_col = f"{marker}_end"
        traj_set = [f"{marker}_end", f"{marker}_auc_norm", f"{marker}_slope", f"{marker}_peak", f"{marker}_tpeak"]
        if end_col not in endpoint_df.columns:
            continue
        endpoint_score = _fisher_score(endpoint_df[end_col].to_numpy(dtype=float), y_labels)
        traj_scores = []
        for col in traj_set:
            if col in traj_df.columns:
                sc = _fisher_score(traj_df[col].to_numpy(dtype=float), y_labels)
                traj_scores.append(sc)
                gain_detail_rows.append(
                    {
                        "marker": marker,
                        "trajectory_feature": col,
                        "feature_score": float(sc),
                        "endpoint_score": float(endpoint_score),
                        "delta_vs_endpoint": float(sc - endpoint_score),
                    }
                )
        if not traj_scores:
            continue
        trajectory_score = float(np.mean(traj_scores))
        gain_rows.append(
            {
                "marker": marker,
                "endpoint_score": float(endpoint_score),
                "trajectory_score": trajectory_score,
                "delta_score": float(trajectory_score - endpoint_score),
            }
        )
    gain_df = pd.DataFrame(gain_rows)
    gain_detail_df = pd.DataFrame(gain_detail_rows)
    gain_df.to_csv(args.out / "sanity10_trajectory_added_value_feature_gain.csv", index=False)
    gain_detail_df.to_csv(args.out / "sanity10_trajectory_added_value_feature_gain_detail.csv", index=False)

    # Plot 1: confidence-lift slopegraph (professional, direct message).
    shift_sorted = shift_df.sort_values("delta_p_true", ascending=True).reset_index(drop=True)
    x_idx = np.arange(len(shift_sorted), dtype=float)
    fig1, ax0 = plt.subplots(figsize=(13.8, 5.2), constrained_layout=False)
    for i, row in shift_sorted.iterrows():
        y0 = float(row["p_true_endpoint"])
        y1 = float(row["p_true_trajectory"])
        fam = str(row["true_family"])
        c = family_color.get(fam, "#4b5563")
        ax0.plot([x_idx[i], x_idx[i]], [y0, y1], color=c, alpha=0.33, linewidth=1.15, zorder=1)
    for fam in family_order:
        m = shift_sorted["true_family"].astype(str) == fam
        ax0.scatter(x_idx[m], shift_sorted.loc[m, "p_true_endpoint"].to_numpy(dtype=float), s=20, color="#9ca3af", alpha=0.9, edgecolors="none", zorder=2)
        ax0.scatter(x_idx[m], shift_sorted.loc[m, "p_true_trajectory"].to_numpy(dtype=float), s=24, color=family_color.get(fam, "#4b5563"), alpha=0.95, edgecolors="white", linewidths=0.4, zorder=3, label=family_label.get(fam, fam))
    ax0.axhline(0.5, color="#94a3b8", linestyle="--", linewidth=1.0, alpha=0.7)
    ax0.set_xlim(-1.0, len(shift_sorted))
    ax0.set_ylim(0.0, 1.0)
    ax0.set_xlabel("Conditions (sorted by confidence gain from trajectory)")
    ax0.set_ylabel("P(true family)")
    ax0.set_title("Confidence Lift per Condition: Endpoint-only vs Endpoint+Trajectory", pad=11)
    ax0.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#9ca3af", markersize=5.5, label="Endpoint-only"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#111827", markersize=5.5, label="Endpoint+trajectory"),
    ]
    fam_handles = [Line2D([0], [0], marker="o", color="none", markerfacecolor=family_color[f], markersize=5.8, label=family_label[f]) for f in family_order]
    leg1 = ax0.legend(handles=handles, frameon=False, loc="upper left", fontsize=8.8)
    ax0.add_artist(leg1)
    ax0.legend(handles=fam_handles, frameon=False, loc="lower right", ncol=2, fontsize=8.6)
    mean_delta = float(np.nanmean(shift_df["delta_p_true"].to_numpy(dtype=float)))
    median_delta = float(np.nanmedian(shift_df["delta_p_true"].to_numpy(dtype=float)))
    ax0.text(
        0.01,
        0.02,
        f"Mean DeltaP={mean_delta:+.3f} | Median DeltaP={median_delta:+.3f}",
        transform=ax0.transAxes,
        fontsize=9.2,
        color="#111827",
        ha="left",
        va="bottom",
    )
    fig1.savefig(args.out / "sanity10_trajectory_added_value_shiftplot.png", dpi=args.dpi, bbox_inches="tight")
    fig1.savefig(args.out / "sanity10_trajectory_added_value_shiftplot.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig1)

    # Plot 2: single diagonal-split confusion matrix (endpoint vs trajectory in one panel).
    fig2, ax1 = plt.subplots(figsize=(7.4, 6.2), constrained_layout=False)
    labels_txt = [family_label.get(f, f).replace("Ascorbic Acid", "Ascorbic\nAcid") for f in family_order]
    ncls = len(family_order)
    cmap_endpoint = plt.get_cmap("Blues")
    cmap_trajectory = plt.get_cmap("Oranges")
    norm01 = mcolors.Normalize(vmin=0.0, vmax=1.0)
    for i in range(ncls):
        for j in range(ncls):
            x0, y0 = float(j), float(i)
            v_end = float(conf_end.iloc[i, j])
            v_trj = float(conf_trj.iloc[i, j])
            tri_bl = Polygon(
                [[x0 - 0.5, y0 - 0.5], [x0 - 0.5, y0 + 0.5], [x0 + 0.5, y0 + 0.5]],
                closed=True,
                facecolor=cmap_endpoint(norm01(v_end)),
                edgecolor="none",
            )
            tri_tr = Polygon(
                [[x0 - 0.5, y0 - 0.5], [x0 + 0.5, y0 - 0.5], [x0 + 0.5, y0 + 0.5]],
                closed=True,
                facecolor=cmap_trajectory(norm01(v_trj)),
                edgecolor="none",
            )
            ax1.add_patch(tri_bl)
            ax1.add_patch(tri_tr)
            ax1.add_patch(Rectangle((x0 - 0.5, y0 - 0.5), 1.0, 1.0, fill=False, edgecolor="#e2e8f0", linewidth=0.8))
            ax1.text(
                x0 - 0.16,
                y0 + 0.16,
                f"{100.0 * v_end:.0f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="#0f172a",
            )
            ax1.text(
                x0 + 0.16,
                y0 - 0.16,
                f"{100.0 * v_trj:.0f}",
                ha="center",
                va="center",
                fontsize=8.8,
                color="#7c2d12",
            )
    ax1.set_xlim(-0.5, ncls - 0.5)
    ax1.set_ylim(ncls - 0.5, -0.5)
    ax1.set_xticks(np.arange(ncls), labels_txt)
    ax1.set_yticks(np.arange(ncls), labels_txt)
    ax1.set_xlabel("Predicted family")
    ax1.set_ylabel("True family")
    ax1.set_title("Single Family Confusion Matrix: Endpoint (blue) vs Trajectory (orange)")
    # mini legend showing triangle semantics
    ax1.text(1.01, 0.98, "Top-left: Endpoint", transform=ax1.transAxes, fontsize=8.8, color="#1e3a8a", ha="left", va="top")
    ax1.text(1.01, 0.93, "Bottom-right: Trajectory", transform=ax1.transAxes, fontsize=8.8, color="#9a3412", ha="left", va="top")
    fig2.savefig(args.out / "sanity10_trajectory_added_value_confusion_split.png", dpi=args.dpi, bbox_inches="tight")
    fig2.savefig(args.out / "sanity10_trajectory_added_value_confusion_split.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig2)

    # Plot 3: mechanistic feature-gain decomposition (trajectory descriptors vs endpoint).
    if not gain_detail_df.empty:
        plot_df = gain_detail_df.copy()
        feature_label = {
            "auc_norm": "AUC/Span",
            "slope": "Slope",
            "peak": "Peak",
            "tpeak": "Peak Time",
            "end": "Endpoint",
        }
        plot_df["marker_pretty"] = plot_df["marker"].replace({"ALP": "ALP", "collagen": "Collagen", "mineral": "Mineral"})
        plot_df["feat_short"] = plot_df["trajectory_feature"].str.split("_").str[-1].map(feature_label).fillna(plot_df["trajectory_feature"])
        plot_df["group"] = plot_df["marker_pretty"] + " - " + plot_df["feat_short"]
        plot_df = plot_df.sort_values(["marker_pretty", "delta_vs_endpoint"], ascending=[True, False]).reset_index(drop=True)
        fig3, ax2 = plt.subplots(figsize=(12.8, 5.6), constrained_layout=False)
        x = np.arange(len(plot_df), dtype=float)
        colors = []
        palette = {"ALP": "#2563eb", "Collagen": "#059669", "Mineral": "#b45309"}
        for _, r in plot_df.iterrows():
            colors.append(palette.get(str(r["marker_pretty"]), "#475569"))
        bars = ax2.bar(x, plot_df["delta_vs_endpoint"].to_numpy(dtype=float), color=colors, alpha=0.9)
        ax2.axhline(0.0, color="#334155", linewidth=0.9)
        ax2.set_xticks(x, plot_df["group"].tolist(), rotation=45, ha="right")
        ax2.set_ylabel("Delta separability vs endpoint (Fisher score)")
        ax2.set_title("Mechanistic Gain from Trajectory Descriptors", pad=11)
        ax2.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)
        for b in bars:
            h = float(b.get_height())
            ax2.text(
                float(b.get_x() + b.get_width() / 2),
                h + (0.01 if h >= 0 else -0.01),
                f"{h:+.2f}",
                ha="center",
                va="bottom" if h >= 0 else "top",
                fontsize=8.2,
                color="#111827",
            )
        leg = [
            Line2D([0], [0], color=palette["ALP"], lw=4, label="ALP family"),
            Line2D([0], [0], color=palette["Collagen"], lw=4, label="Collagen family"),
            Line2D([0], [0], color=palette["Mineral"], lw=4, label="Mineral family"),
        ]
        ax2.legend(handles=leg, frameon=False, loc="upper right", fontsize=8.7)
        fig3.savefig(args.out / "sanity10_trajectory_added_value_feature_decomposition.png", dpi=args.dpi, bbox_inches="tight")
        fig3.savefig(args.out / "sanity10_trajectory_added_value_feature_decomposition.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig3)

    # Plot 4: per-family diagonal gain summary.
    diag_end = np.diag(conf_end.to_numpy(dtype=float))
    diag_trj = np.diag(conf_trj.to_numpy(dtype=float))
    gain = diag_trj - diag_end
    fig4, ax4 = plt.subplots(figsize=(8.4, 4.8), constrained_layout=False)
    x = np.arange(len(family_order), dtype=float)
    bars = ax4.bar(x, 100.0 * gain, color=["#16a34a" if g >= 0 else "#dc2626" for g in gain], alpha=0.9)
    ax4.axhline(0.0, color="#334155", linewidth=0.9)
    ax4.set_xticks(x, labels_txt)
    ax4.set_ylabel("Diagonal improvement (percentage points)")
    ax4.set_title("Per-family classification gain from trajectory features")
    ax4.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)
    for i, b in enumerate(bars):
        h = float(b.get_height())
        ax4.text(float(b.get_x() + b.get_width() / 2), h + (0.35 if h >= 0 else -0.35), f"{h:+.1f}", ha="center", va="bottom" if h >= 0 else "top", fontsize=8.6)
    fig4.savefig(args.out / "sanity10_trajectory_added_value_family_gain.png", dpi=args.dpi, bbox_inches="tight")
    fig4.savefig(args.out / "sanity10_trajectory_added_value_family_gain.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig4)


def _make_change_story_plots(args: argparse.Namespace) -> None:
    shift_path = args.out / "sanity10_trajectory_added_value_shift.csv"
    conf_long_path = args.out / "ident_nn_confusion_endpoint_vs_trajectory.csv"
    if (not shift_path.exists()) or (not conf_long_path.exists()):
        return

    shift = pd.read_csv(shift_path)
    conf_long = pd.read_csv(conf_long_path)
    if shift.empty or conf_long.empty:
        return

    fam_order = [
        "1_spatial_pattern",
        "2_cell_number",
        "3_cell_dynamics",
        "4_ascorbic_acid",
    ]
    fam_label = {
        "1_spatial_pattern": "Spatial",
        "2_cell_number": "Cell #",
        "3_cell_dynamics": "Dynamics",
        "4_ascorbic_acid": "Ascorbic Acid",
    }
    fam_color = {
        "1_spatial_pattern": "#1f77b4",
        "2_cell_number": "#f59e0b",
        "3_cell_dynamics": "#059669",
        "4_ascorbic_acid": "#c026d3",
    }

    # Plot 1: confidence-shift waterfall (sorted delta P(true)).
    s = shift.copy()
    s["delta_p_true"] = pd.to_numeric(s["delta_p_true"], errors="coerce")
    s = s.dropna(subset=["delta_p_true"]).sort_values("delta_p_true").reset_index(drop=True)
    if not s.empty:
        x = np.arange(len(s), dtype=float)
        fig1, ax = plt.subplots(figsize=(13.2, 5.2), constrained_layout=False)
        colors = [fam_color.get(str(v), "#64748b") for v in s["true_family"].astype(str)]
        ax.bar(x, s["delta_p_true"].to_numpy(dtype=float), color=colors, width=0.9, alpha=0.9)
        running = np.cumsum(s["delta_p_true"].to_numpy(dtype=float)) / np.maximum(x + 1.0, 1.0)
        ax.plot(x, running, color="#111827", linewidth=2.0, label="Cumulative mean DeltaP")
        ax.axhline(0.0, color="#475569", linewidth=0.9)
        ax.set_xlim(-1.0, len(s))
        ax.set_xlabel("Conditions sorted by Delta P(true)")
        ax.set_ylabel("Delta P(true family) = trajectory - endpoint")
        ax.set_title("Trajectory Information Gain Across Conditions")
        ax.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)
        handles = [Line2D([0], [0], color=fam_color[f], lw=5, label=fam_label[f]) for f in fam_order if f in set(s["true_family"].astype(str))]
        handles.append(Line2D([0], [0], color="#111827", lw=2.0, label="Cumulative mean"))
        ax.legend(handles=handles, frameon=False, fontsize=8.6, loc="lower right", ncol=2)
        mean_delta = float(np.nanmean(s["delta_p_true"].to_numpy(dtype=float)))
        p10 = float(np.nanpercentile(s["delta_p_true"].to_numpy(dtype=float), 10))
        p90 = float(np.nanpercentile(s["delta_p_true"].to_numpy(dtype=float), 90))
        ax.text(0.01, 0.97, f"Mean={mean_delta:+.3f} | P10={p10:+.3f} | P90={p90:+.3f}", transform=ax.transAxes, ha="left", va="top", fontsize=9.3, color="#111827")
        fig1.savefig(args.out / "sanity10_change_story_1_confidence_waterfall.png", dpi=args.dpi, bbox_inches="tight")
        fig1.savefig(args.out / "sanity10_change_story_1_confidence_waterfall.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig1)

    # Plot 2: transition-outcome bars by family (improved/degraded/stable).
    t = shift.copy()
    t["ok_endpoint"] = (t["pred_endpoint"].astype(str) == t["true_family"].astype(str))
    t["ok_trajectory"] = (t["pred_trajectory"].astype(str) == t["true_family"].astype(str))
    def _state(r: pd.Series) -> str:
        e = bool(r["ok_endpoint"])
        tr = bool(r["ok_trajectory"])
        if (not e) and tr:
            return "Improved (wrong->correct)"
        if e and (not tr):
            return "Degraded (correct->wrong)"
        if e and tr:
            return "Stable correct"
        return "Stable wrong"
    t["state"] = t.apply(_state, axis=1)
    state_order = [
        "Improved (wrong->correct)",
        "Degraded (correct->wrong)",
        "Stable correct",
        "Stable wrong",
    ]
    state_color = {
        "Improved (wrong->correct)": "#16a34a",
        "Degraded (correct->wrong)": "#dc2626",
        "Stable correct": "#2563eb",
        "Stable wrong": "#94a3b8",
    }
    rows = []
    for fam in fam_order:
        sub = t.loc[t["true_family"].astype(str) == fam]
        if sub.empty:
            continue
        n = float(len(sub))
        for st in state_order:
            v = float((sub["state"] == st).sum()) / max(n, 1.0)
            rows.append({"family": fam, "state": st, "fraction": v})
    trans = pd.DataFrame(rows)
    if not trans.empty:
        fig2, ax = plt.subplots(figsize=(9.4, 5.0), constrained_layout=False)
        x = np.arange(len(trans["family"].unique()), dtype=float)
        bottoms = np.zeros_like(x)
        families_present = [f for f in fam_order if f in trans["family"].unique().tolist()]
        for st in state_order:
            vals = []
            for fam in families_present:
                r = trans.loc[(trans["family"] == fam) & (trans["state"] == st), "fraction"]
                vals.append(float(r.iloc[0]) if not r.empty else 0.0)
            y = np.array(vals, dtype=float)
            ax.bar(x, y, bottom=bottoms, color=state_color[st], edgecolor="white", linewidth=0.8, label=st)
            bottoms = bottoms + y
        ax.set_xticks(x, [fam_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in families_present])
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Fraction of conditions")
        ax.set_title("Classification Transition Profile by Family")
        ax.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)
        ax.legend(frameon=False, fontsize=8.6, loc="upper center", ncol=2)
        fig2.savefig(args.out / "sanity10_change_story_2_transition_profile.png", dpi=args.dpi, bbox_inches="tight")
        fig2.savefig(args.out / "sanity10_change_story_2_transition_profile.pdf", dpi=args.dpi, bbox_inches="tight")
        plt.close(fig2)

    # Plot 3: marker x family diagonal gain heatmap (trajectory - endpoint).
    conf_long = conf_long.copy()
    conf_long = conf_long.loc[conf_long["panel"].astype(str).isin(["A", "B", "C"])].copy()
    if not conf_long.empty:
        markers = {"A": "ALP", "B": "Collagen", "C": "Mineral"}
        vals = np.full((3, len(fam_order)), np.nan, dtype=float)
        for i, p in enumerate(["A", "B", "C"]):
            for j, fam in enumerate(fam_order):
                e = conf_long.loc[
                    (conf_long["panel"].astype(str) == p)
                    & (conf_long["mode"].astype(str) == "endpoint")
                    & (conf_long["true_family"].astype(str) == fam)
                    & (conf_long["pred_family"].astype(str) == fam),
                    "fraction",
                ]
                tr = conf_long.loc[
                    (conf_long["panel"].astype(str) == p)
                    & (conf_long["mode"].astype(str) == "trajectory")
                    & (conf_long["true_family"].astype(str) == fam)
                    & (conf_long["pred_family"].astype(str) == fam),
                    "fraction",
                ]
                if (not e.empty) and (not tr.empty):
                    vals[i, j] = float(tr.iloc[0]) - float(e.iloc[0])
        if np.isfinite(vals).any():
            vmax = max(0.05, float(np.nanmax(np.abs(vals))))
            fig3, ax = plt.subplots(figsize=(8.9, 4.8), constrained_layout=False)
            im = ax.imshow(vals, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
            ax.set_xticks(np.arange(len(fam_order)), [fam_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in fam_order])
            ax.set_yticks(np.arange(3), [markers["A"], markers["B"], markers["C"]])
            ax.set_xlabel("Perturbation family")
            ax.set_ylabel("Readout")
            ax.set_title("Diagonal Classification Gain (Trajectory - Endpoint)")
            for i in range(vals.shape[0]):
                for j in range(vals.shape[1]):
                    if np.isfinite(vals[i, j]):
                        ax.text(j, i, f"{100.0*vals[i,j]:+.1f} pp", ha="center", va="center", fontsize=8.7, color="#111827")
            cbar = fig3.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
            cbar.set_label("Percentage-point change")
            fig3.savefig(args.out / "sanity10_change_story_3_marker_family_gain.png", dpi=args.dpi, bbox_inches="tight")
            fig3.savefig(args.out / "sanity10_change_story_3_marker_family_gain.pdf", dpi=args.dpi, bbox_inches="tight")
            plt.close(fig3)


def _make_change_story_pro_figure(args: argparse.Namespace) -> None:
    shift_path = args.out / "sanity10_trajectory_added_value_shift.csv"
    conf_long_path = args.out / "ident_nn_confusion_endpoint_vs_trajectory.csv"
    if (not shift_path.exists()) or (not conf_long_path.exists()):
        return
    shift = pd.read_csv(shift_path)
    conf_long = pd.read_csv(conf_long_path)
    if shift.empty or conf_long.empty:
        return

    fam_order = [
        "1_spatial_pattern",
        "2_cell_number",
        "3_cell_dynamics",
        "4_ascorbic_acid",
    ]
    fam_label = {
        "1_spatial_pattern": "Spatial",
        "2_cell_number": "Cell #",
        "3_cell_dynamics": "Dynamics",
        "4_ascorbic_acid": "Ascorbic Acid",
    }
    marker_panels = [("A", "ALP"), ("B", "Collagen"), ("C", "Mineral")]

    # 1) Marker-level Macro-F1 endpoint vs trajectory.
    f1_rows: list[dict[str, float | str]] = []
    diag_gain = np.full((3, len(fam_order)), np.nan, dtype=float)
    for i, (panel, marker_name) in enumerate(marker_panels):
        for mode in ["endpoint", "trajectory"]:
            sub = conf_long.loc[
                (conf_long["panel"].astype(str) == panel) & (conf_long["mode"].astype(str) == mode)
            ].copy()
            if sub.empty:
                continue
            mat = (
                sub.pivot(index="true_family", columns="pred_family", values="fraction")
                .reindex(index=fam_order, columns=fam_order, fill_value=0.0)
            )
            f1_vals: list[float] = []
            for fam in fam_order:
                tp = float(mat.loc[fam, fam])
                fp = float(mat[fam].sum() - tp)
                fn = float(mat.loc[fam].sum() - tp)
                den = 2.0 * tp + fp + fn
                f1_vals.append((2.0 * tp / den) if den > 0.0 else 0.0)
            f1_rows.append(
                {
                    "marker": marker_name,
                    "mode": mode,
                    "macro_f1": float(np.mean(f1_vals)),
                }
            )
            if mode == "endpoint":
                diag_ep = np.diag(mat.to_numpy(dtype=float))
            else:
                diag_tr = np.diag(mat.to_numpy(dtype=float))
        if "diag_ep" in locals() and "diag_tr" in locals():
            diag_gain[i, :] = diag_tr - diag_ep
        if "diag_ep" in locals():
            del diag_ep
        if "diag_tr" in locals():
            del diag_tr
    f1_df = pd.DataFrame(f1_rows)
    f1_df.to_csv(args.out / "sanity10_change_story_pro_macro_f1.csv", index=False)

    # 2) Family-level distribution of Delta P(true).
    shift = shift.copy()
    shift["delta_p_true"] = pd.to_numeric(shift["delta_p_true"], errors="coerce")
    shift = shift.dropna(subset=["delta_p_true"])

    fig = plt.figure(figsize=(15.8, 5.8), constrained_layout=False)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.35, 1.25], wspace=0.28)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[0, 2])

    # Panel A: Dumbbell Macro-F1.
    endpoint_color = "#4f8fbe"
    trajectory_color = "#d86b6b"
    marker_order = [m for _, m in marker_panels]
    y = np.arange(len(marker_order), dtype=float)
    ep = []
    tr = []
    for m in marker_order:
        e = f1_df.loc[(f1_df["marker"] == m) & (f1_df["mode"] == "endpoint"), "macro_f1"]
        t = f1_df.loc[(f1_df["marker"] == m) & (f1_df["mode"] == "trajectory"), "macro_f1"]
        ep.append(float(e.iloc[0]) if not e.empty else np.nan)
        tr.append(float(t.iloc[0]) if not t.empty else np.nan)
    ep_arr = np.array(ep, dtype=float)
    tr_arr = np.array(tr, dtype=float)
    for i in range(len(marker_order)):
        if np.isfinite(ep_arr[i]) and np.isfinite(tr_arr[i]):
            ax1.plot([ep_arr[i], tr_arr[i]], [y[i], y[i]], color="#7b8794", linewidth=1.6, zorder=1)
    ax1.scatter(ep_arr, y, color=endpoint_color, s=58, zorder=2, label="Endpoint")
    ax1.scatter(tr_arr, y, color=trajectory_color, s=58, zorder=3, label="Trajectory")
    for i, m in enumerate(marker_order):
        if np.isfinite(ep_arr[i]) and np.isfinite(tr_arr[i]):
            ax1.text(max(ep_arr[i], tr_arr[i]) + 0.008, y[i], f"Delta {tr_arr[i]-ep_arr[i]:+.2f}", va="center", ha="left", fontsize=9.2, color="#111111")
    ax1.set_yticks(y, marker_order)
    ax1.set_xlabel("Macro-F1")
    ax1.set_title("A. Marker-Level Classification Lift", pad=10, fontsize=12.2, fontweight="semibold")
    ax1.grid(axis="x", color="#d9dee5", linewidth=0.8, alpha=0.65)
    ax1.legend(frameon=False, fontsize=9.0, loc="lower right")
    ax1.set_xlim(0.0, min(1.0, float(np.nanmax([np.nanmax(ep_arr), np.nanmax(tr_arr)]) + 0.14)))

    # Panel B: Family-level Delta P(true) distribution (box + jitter).
    fam_positions = np.arange(len(fam_order), dtype=float)
    data = [shift.loc[shift["true_family"].astype(str) == fam, "delta_p_true"].to_numpy(dtype=float) for fam in fam_order]
    bp = ax2.boxplot(
        data,
        positions=fam_positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        medianprops={"color": "#111111", "linewidth": 1.5},
        boxprops={"linewidth": 0.9, "edgecolor": "#374151"},
        whiskerprops={"linewidth": 0.9, "color": "#6b7280"},
        capprops={"linewidth": 0.9, "color": "#6b7280"},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("#dbe9f6")
    rng = np.random.default_rng(123)
    for i, fam in enumerate(fam_order):
        vals = data[i]
        if vals.size == 0:
            continue
        jitter = rng.uniform(-0.18, 0.18, size=vals.size)
        ax2.scatter(np.full(vals.size, i) + jitter, vals, s=10, color="#4f8fbe", alpha=0.32, edgecolors="none")
        ax2.text(i, float(np.nanmedian(vals)) + 0.004, f"med {np.nanmedian(vals):+.3f}", ha="center", va="bottom", fontsize=8.5, color="#111111")
    ax2.axhline(0.0, color="#475569", linewidth=0.9)
    ax2.set_xticks(fam_positions, [fam_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in fam_order])
    ax2.set_ylabel("Delta P(true family)")
    ax2.set_title("B. Where Trajectory Helps (or Hurts)", pad=10, fontsize=12.2, fontweight="semibold")
    ax2.grid(axis="y", color="#d9dee5", linewidth=0.8, alpha=0.65)

    # Panel C: Marker x Family diagonal gain heatmap.
    vmax = max(0.03, float(np.nanmax(np.abs(diag_gain)))) if np.isfinite(diag_gain).any() else 0.05
    im = ax3.imshow(diag_gain, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax3.set_xticks(np.arange(len(fam_order)), [fam_label[f].replace("Ascorbic Acid", "Ascorbic\nAcid") for f in fam_order])
    ax3.set_yticks(np.arange(len(marker_order)), marker_order)
    ax3.set_title("C. Family-Specific Diagonal Gain", pad=10, fontsize=12.2, fontweight="semibold")
    ax3.set_xlabel("Family")
    ax3.set_ylabel("Marker")
    for i in range(diag_gain.shape[0]):
        for j in range(diag_gain.shape[1]):
            v = diag_gain[i, j]
            if np.isfinite(v):
                ax3.text(j, i, f"{100.0*v:+.1f} pp", ha="center", va="center", fontsize=8.6, color="#111111")
    cbar = fig.colorbar(im, ax=ax3, fraction=0.05, pad=0.03)
    cbar.set_label("Trajectory - Endpoint")

    mean_delta = float(np.nanmean(shift["delta_p_true"].to_numpy(dtype=float))) if not shift.empty else 0.0
    fig.suptitle(
        f"Trajectory Added-Value Summary (mean Delta P(true) = {mean_delta:+.3f})",
        fontsize=14.5,
        y=0.985,
        fontweight="semibold",
    )
    fig.savefig(args.out / "sanity10_change_story_pro_summary.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_change_story_pro_summary.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = _resolve_input(args.input, args.out)
    print(f"[extract] input={src}")
    print(f"[extract] out={args.out}")
    print(f"[extract] days={args.days}")
    print(f"[extract] threshold={args.equivalence_threshold}")
    print(f"[extract] observables_only={bool(args.observables_only)}")
    print(f"[extract] family_col={args.family_col or '<auto>'}")
    print(f"[extract] spatial_col={args.spatial_col or '<auto>'}")
    print("[extract] comparison families: baseline + spatial + cell-number + dynamics + ascorbic-acid (sampling-time and boundary/scale excluded)")
    raw = sc10.load_input(src, args.sheet)
    if args.input is None:
        args.input = src
    run_args = argparse.Namespace(**vars(args))
    run_args.figures = ["composite"]
    if args.threshold_sweep:
        for thr in args.threshold_sweep:
            thr_args = argparse.Namespace(**vars(run_args))
            thr_args.equivalence_threshold = float(thr)
            thr_args.out = args.out / f"thr_{float(thr):.2f}"
            thr_args.out.mkdir(parents=True, exist_ok=True)
            print(f"[extract] running threshold={thr_args.equivalence_threshold:.4g} out={thr_args.out}")
            sc10.run_analysis(thr_args, raw.copy())
    else:
        sc10.run_analysis(run_args, raw)
    _make_confusion_abc_figure(raw, args)
    print(f"[extract] figure={args.out / 'sanity10_family_confusion_abc.png'}")
    _make_identify_test_new_figures(raw, args)
    print(f"[extract] figure={args.out / 'sanity10_identify_test_fig1_endpoint_confusion.png'}")
    _make_trajectory_added_value_figures(raw, args)
    print(f"[extract] figure={args.out / 'sanity10_trajectory_added_value.png'}")
    _make_change_story_plots(args)
    print(f"[extract] figure={args.out / 'sanity10_change_story_1_confidence_waterfall.png'}")
    _make_change_story_pro_figure(args)
    print(f"[extract] figure={args.out / 'sanity10_change_story_pro_summary.png'}")
    _make_option2_distance_to_baseline_abc(raw, args)
    print(f"[extract] figure={args.out / 'sanity10_option2_distance_to_baseline.png'}")
    for svg in args.out.glob("sanity10_*.svg"):
        try:
            svg.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
