#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT = SCRIPT_DIR / "out" / "sanity12"
os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "osteomin_mplconfig").resolve()))

import sanity2_figure_extract as s2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Figure extraction for sanity12 outputs (focus: figure_3_ecm_overview_heatmap).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="sanity12 output folder containing run_summary.csv")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output folder for figures; defaults to <input>/figure_extract_12",
    )
    parser.add_argument("--alp-hours", type=int, default=240, help="hour used for ALP endpoint")
    parser.add_argument("--collagen-hours", type=int, default=336, help="hour used for collagen endpoint")
    parser.add_argument("--hap-hours", type=int, default=672, help="hour used for HAp endpoint")
    return parser.parse_args()


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


def _find_grid_for_day(rep_dir: Path, prefix: str, preferred_day: int) -> Path | None:
    preferred = rep_dir / "grids" / f"{prefix}_day_{preferred_day:03d}.npy"
    if preferred.exists():
        return preferred
    candidates = sorted((rep_dir / "grids").glob(f"{prefix}_day_*.npy"))
    if not candidates:
        return None
    return candidates[-1]


def _safe_float(row: pd.Series, key: str) -> float:
    value = row.get(key, np.nan)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def build_replicate_table(run_summary: pd.DataFrame, assay_hours: dict[str, int]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    target_day = int(round(float(assay_hours["ha"]) / 24.0))

    for _, run in run_summary.iterrows():
        if not bool(run.get("success", True)):
            continue
        out_dir = Path(str(run["out_dir"]))
        ts_path = out_dir / "timeseries.csv"
        if not ts_path.exists():
            raise SystemExit(f"timeseries.csv not found for run output: {ts_path}")

        ts = pd.read_csv(ts_path)
        alp_row = row_at_hours(ts, assay_hours["alp"])
        collagen_row = row_at_hours(ts, assay_hours["collagen"])
        ha_row = row_at_hours(ts, assay_hours["ha"])
        final_row = ts.iloc[-1]

        xi = float("nan")
        ha_grid_path = _find_grid_for_day(out_dir, "ha", target_day)
        if ha_grid_path is not None:
            ha_grid = np.load(ha_grid_path).astype(float)
            xi = float(estimate_autocorr_length_xi(ha_grid))

        index = int(run["index"])
        cells = int(run["cells"])
        rows.append(
            {
                "condition_index": index,
                "cells": cells,
                "condition_label": f"C{index:02d}",
                "row_label": f"C{index:02d}  cells={cells}",
                "replicate_index": int(run["replicate_index"]),
                "seed": int(run["seed"]),
                "out_dir": str(out_dir),
                "assay_alp": float(alp_row["alp_total"]),
                "assay_collagen": float(collagen_row["collagen_total"]),
                "assay_ha": float(ha_row["ha_total"]),
                "final_collagen": _safe_float(final_row, "collagen_total"),
                "final_ha": _safe_float(final_row, "ha_total"),
                "final_live": _safe_float(final_row, "n_total"),
                "final_nodule_count": _safe_float(final_row, "nodule_count_ever"),
                "spatial_autocorr_xi": xi,
            }
        )

    if not rows:
        raise SystemExit("no successful runs found in run_summary.csv")
    return pd.DataFrame(rows).sort_values(["condition_index", "replicate_index"]).reset_index(drop=True)


def build_condition_summary(replicate_df: pd.DataFrame) -> pd.DataFrame:
    numeric = [
        "assay_alp",
        "assay_collagen",
        "assay_ha",
        "final_collagen",
        "final_ha",
        "final_live",
        "final_nodule_count",
        "spatial_autocorr_xi",
    ]
    grouped = replicate_df.groupby(["condition_index", "cells", "condition_label", "row_label"], sort=True)
    mean_df = grouped[numeric].mean().reset_index()
    sd_df = grouped[numeric].std(ddof=1).reset_index().fillna(0.0)
    n_df = grouped.size().reset_index(name="n_replicates")
    summary = mean_df.merge(
        sd_df,
        on=["condition_index", "cells", "condition_label", "row_label"],
        suffixes=("_mean", "_sd"),
    ).merge(n_df, on=["condition_index", "cells", "condition_label", "row_label"])

    summary = summary.sort_values("condition_index").reset_index(drop=True)
    base = summary.iloc[0]

    for mean_src, sd_src, pct_dest, sd_dest in [
        ("assay_alp_mean", "assay_alp_sd", "assay_alp_pct", "assay_alp_pct_sd"),
        ("assay_collagen_mean", "assay_collagen_sd", "assay_collagen_pct", "assay_collagen_pct_sd"),
        ("assay_ha_mean", "assay_ha_sd", "assay_ha_pct", "assay_ha_pct_sd"),
        ("final_ha_mean", "final_ha_sd", "final_ha_pct", "final_ha_pct_sd"),
        ("final_collagen_mean", "final_collagen_sd", "final_collagen_pct", "final_collagen_pct_sd"),
        ("final_live_mean", "final_live_sd", "final_live_pct", "final_live_pct_sd"),
    ]:
        denom = max(float(base[mean_src]), 1e-12)
        summary[pct_dest] = 100.0 * (summary[mean_src] - float(base[mean_src])) / denom
        summary[sd_dest] = 100.0 * summary[sd_src].astype(float) / denom

    summary["final_n_patch_mean"] = summary["final_nodule_count_mean"].astype(float)
    summary["final_n_patch_sd"] = summary["final_nodule_count_sd"].astype(float)
    return summary


def main() -> None:
    args = parse_args()
    out_dir = args.out if args.out is not None else (args.input / "figure_extract_12")

    run_summary_path = args.input / "run_summary.csv"
    if not run_summary_path.exists():
        raise SystemExit(f"run_summary.csv not found under {args.input}")

    run_summary = pd.read_csv(run_summary_path)
    assay_hours = {
        "alp": int(args.alp_hours),
        "collagen": int(args.collagen_hours),
        "ha": int(args.hap_hours),
    }

    replicate_df = build_replicate_table(run_summary, assay_hours)
    summary = build_condition_summary(replicate_df)

    out_dir.mkdir(parents=True, exist_ok=True)
    replicate_df.to_csv(out_dir / "replicate_table_sanity12.csv", index=False)
    summary.to_csv(out_dir / "summary_sanity12.csv", index=False)

    s2.export_ecm_overview_heatmap(
        summary,
        out_dir,
        assay_hours,
        nodule_column_mode="nn",
        geometry_mode="classic",
    )

    print(f"Wrote figure to {out_dir / 'figure_3_ecm_overview_heatmap.png'}")
    print(f"Wrote summary table to {out_dir / 'summary_sanity12.csv'}")


if __name__ == "__main__":
    main()
