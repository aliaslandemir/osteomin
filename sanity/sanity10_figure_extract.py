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
        {"letter": "A", "mode": "endpoint", "marker": "ALP", "title": "ALP @ Day 10"},
        {"letter": "B", "mode": "endpoint", "marker": "collagen", "title": "Collagen @ Day 14"},
        {"letter": "C", "mode": "endpoint", "marker": "mineral", "title": "Mineral @ Day 28"},
        {"letter": "D", "mode": "trajectory", "marker": "ALP", "title": "ALP (Day 0-10)"},
        {"letter": "E", "mode": "trajectory", "marker": "collagen", "title": "Collagen (Day 0-14)"},
        {"letter": "F", "mode": "trajectory", "marker": "mineral", "title": "Mineral (Day 0-28)"},
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

    conf_by_panel: dict[str, pd.DataFrame] = {}
    f1_by_panel: dict[str, float] = {}
    long_rows: list[dict[str, float | int | str]] = []
    for panel in panel_specs:
        key = str(panel["letter"])
        mode = str(panel["mode"])
        marker = str(panel["marker"])
        result = _compute_panel(mode, marker)
        if result is None:
            continue
        conf, macro_f1, days_used = result
        conf_by_panel[key] = conf
        f1_by_panel[key] = macro_f1
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

    if not conf_by_panel:
        return
    pd.DataFrame(long_rows).to_csv(args.out / "ident_nn_confusion_endpoint_vs_trajectory.csv", index=False)
    pd.DataFrame(long_rows).to_csv(args.out / "ident_nn_confusion_day10_14_28_all3.csv", index=False)

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 10.0), constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.905, top=0.90, bottom=0.07, wspace=0.12, hspace=0.26)
    cmap = plt.get_cmap("Blues")
    norm = mcolors.Normalize(vmin=0.0, vmax=1.0)
    mappable = None
    for i, panel in enumerate(panel_specs):
        row = i // 3
        col = i % 3
        ax = axes[row, col]
        key = str(panel["letter"])
        conf = conf_by_panel.get(key)
        if conf is None:
            ax.axis("off")
            continue
        arr = conf.to_numpy(dtype=float)
        mappable = ax.imshow(arr, norm=norm, cmap=cmap, interpolation="nearest")
        ax.set_xticks(np.arange(len(family_order)))
        ax.set_yticks(np.arange(len(family_order)))
        # Wrap long family names to avoid overlap in compact multi-panel layouts.
        x_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        y_labels = [family_label[k].replace("Ascorbic Acid", "Ascorbic\nAcid") for k in family_order]
        ax.set_xticklabels(x_labels, rotation=0, ha="center")
        ax.set_yticklabels(y_labels if col == 0 else [])
        if col == 0:
            ax.set_ylabel("True family", fontsize=11.0)
        if row == 1:
            ax.set_xlabel("Nearest-neighbor family", fontsize=11.0)
        else:
            ax.set_xlabel("")
        panel_title = str(panel["title"])
        panel_f1 = f1_by_panel.get(key, float("nan"))
        if np.isfinite(panel_f1):
            panel_title = f"{panel_title} (F1={panel_f1:.2f})"
        ax.set_title(panel_title, fontsize=12.0, fontweight="semibold", pad=9)
        ax.text(
            -0.14,
            1.06,
            key,
            transform=ax.transAxes,
            fontsize=13.0,
            fontweight="bold",
            ha="left",
            va="bottom",
            color="#111111",
        )
        ax.tick_params(axis="both", labelsize=10.0, length=3.0, width=0.8, color="#444444")
        # Subtle cell boundaries for print legibility.
        ax.set_xticks(np.arange(-0.5, arr.shape[1], 1), minor=True)
        ax.set_yticks(np.arange(-0.5, arr.shape[0], 1), minor=True)
        ax.grid(which="minor", color="#f2f4f7", linestyle="-", linewidth=0.6)
        ax.tick_params(which="minor", bottom=False, left=False)
        for r in range(arr.shape[0]):
            for c in range(arr.shape[1]):
                v = float(arr[r, c])
                txt = f"{100.0 * v:.0f}%"
                color = "white" if v >= 0.55 else "#222222"
                ax.text(c, r, txt, ha="center", va="center", fontsize=10.2, color=color)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_edgecolor("#c9cdd2")
        ax.set_aspect("equal")

    if mappable is not None:
        cax = fig.add_axes([0.918, 0.20, 0.0135, 0.62])
        cbar = fig.colorbar(mappable, cax=cax)
        cbar.set_label("Row-normalized fraction", fontsize=11.0)
        cbar.ax.tick_params(labelsize=9.4)
    endpoint_f1 = [f1_by_panel.get(k, float("nan")) for k in ("A", "B", "C")]
    endpoint_f1 = [float(v) for v in endpoint_f1 if np.isfinite(float(v))]
    trajectory_f1 = [f1_by_panel.get(k, float("nan")) for k in ("D", "E", "F")]
    trajectory_f1 = [float(v) for v in trajectory_f1 if np.isfinite(float(v))]
    if endpoint_f1 or trajectory_f1:
        endpoint_txt = f"{float(np.mean(endpoint_f1)):.2f}" if endpoint_f1 else "NA"
        trajectory_txt = f"{float(np.mean(trajectory_f1)):.2f}" if trajectory_f1 else "NA"
        fig.suptitle(
            f"Family Confusion Matrices: Endpoint vs Trajectory (Macro-F1 endpoint={endpoint_txt}, trajectory={trajectory_txt})",
            fontsize=16.0,
            y=0.97,
            fontweight="semibold",
        )
    else:
        fig.suptitle("Family Confusion Matrices: Endpoint vs Trajectory", fontsize=16.0, y=0.97, fontweight="semibold")
    fig.savefig(args.out / "sanity10_family_confusion_abc.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_confusion_abc.pdf", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_confusion_abcdef.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(args.out / "sanity10_family_confusion_abcdef.pdf", dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


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
    _make_option2_distance_to_baseline_abc(raw, args)
    print(f"[extract] figure={args.out / 'sanity10_option2_distance_to_baseline.png'}")
    for svg in args.out.glob("sanity10_*.svg"):
        try:
            svg.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    main()
