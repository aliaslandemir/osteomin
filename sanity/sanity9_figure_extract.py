#!/usr/bin/env python3
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

JOURNAL_AXIS_LABEL_PT = 9
JOURNAL_TICK_PT = 8
JOURNAL_LEGEND_PT = 8
JOURNAL_LINE_PT = 1.2
JOURNAL_AXIS_LINE_PT = 1.0
JOURNAL_MARKER_PT = 5.5
JOURNAL_PANEL_LABEL_PT = 12

DEFAULT_INPUT = SCRIPT_DIR / "out" / "sanity_check9"
DEFAULT_OUT = DEFAULT_INPUT / "figure_extract_9"
DEFAULT_RUNS = 20
RANDOM_SEED_MAX = (1 << 63) - 1

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--out",
    "main_out",
    "--seed",
    "12345",
    "--hours",
    "672",
    "--cells",
    "72",
    "--w",
    "33",
    "--h",
    "33",
    "--boundary-mode",
    "periodic",
    "--seed-mode",
    "random",
    "--nodules",
    "40n",
    "--media-interval",
    "48",
    "--secretion",
    "--aa0",
    "50.0",
    "--aa-half-life",
    "8",
    "--collagen-aa-km",
    "5.0",
    "--random-move",
    "--move-prob",
    "0.27",
    "--max-moves",
    "1",
    "--proliferation",
    "--ob-div-prob",
    "0.0",
    "--min-div-hours",
    "38",
    "--max-div-hours",
    "52",
    "--apoptosis",
    "--apoptosis-prob",
    "0.0010",
    "--apoptosis-osteo",
    "0.0",
    "--osteogenic-commitment",
    "--osteo-induction-hours",
    "72",
    "--commit-delay",
    "48",
    "--commit-k",
    "0.0125",
    "--dex-conc",
    "0",
    "--mk4-conc",
    "0",
    "--alp-rb",
    "0.0949",
    "--alp-ro",
    "0.810",
    "--collagen-rb",
    "0.00336",
    "--collagen-ro",
    "0.0130",
    "--ha-ro",
    "0.102",
    "--alp-dex-ref",
    "22.7",
    "--collagen-dex-ref",
    "6.22",
    "--ars-dex-ref",
    "46.6",
    "--alp-a-d",
    "3.72",
    "--alp-b-d",
    "1.75",
    "--collagen-a-d",
    "2.69",
    "--collagen-b-d",
    "2.04",
    "--ha-a-d",
    "1.22",
    "--ha-b-d",
    "0.679",
    "--alp-mk4-ref",
    "0.784",
    "--collagen-mk4-ref",
    "0.349",
    "--ars-mk4-ref",
    "0.667",
    "--alp-a-k",
    "1.10",
    "--alp-b-k",
    "1.06",
    "--collagen-a-k",
    "4.66",
    "--collagen-b-k",
    "2.15",
    "--ha-a-k",
    "4.66",
    "--ha-b-k",
    "2.15",
    "--ha-model",
    "ha",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--spatial-every",
    "24",
    "--alp-plot-hours",
    "240",
    "--collagen-plot-hours",
    "336",
    "--mineral-plot-hours",
    "672",
    "--no-hourly-exports",
    "--alp-snapshots",
    "--csv",
    "--no-xlsx",
]

MARKERS = [
    {"column": "alp_total", "label": "ALP", "color": "#D55E00", "max_hour": 240.0},
    {"column": "collagen_total", "label": "Collagen", "color": "#009E73", "max_hour": 336.0},
    {"column": "ha_total", "label": "Mineral", "color": "#3B5BA9", "max_hour": 672.0},
]

COMPONENTS = [
    {
        "column": "alp_preob_total",
        "label": "ALP preOB",
        "parent": "alp_total",
        "color": "#E69F00",
        "max_hour": 240.0,
    },
    {
        "column": "alp_ob_total",
        "label": "ALP OB",
        "parent": "alp_total",
        "color": "#D55E00",
        "max_hour": 240.0,
    },
    {
        "column": "collagen_preob_total",
        "label": "Collagen preOB",
        "parent": "collagen_total",
        "color": "#56B4E9",
        "max_hour": 336.0,
    },
    {
        "column": "collagen_ob_total",
        "label": "Collagen OB",
        "parent": "collagen_total",
        "color": "#009E73",
        "max_hour": 336.0,
    },
    {
        "column": "mineral_ob_total",
        "label": "Mineral",
        "parent": "ha_total",
        "color": "#3B5BA9",
        "max_hour": 672.0,
    },
]

COLORS = {
    "ink": "#0f172a",
    "grid": "#d8dee9",
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
        "axes.titlesize": 12.5,
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
        description="Figure extraction for sanity_check9 differentiation trajectories.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="root folder containing run_XX/timeseries.csv outputs",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=DEFAULT_OUT,
        help="output folder for extracted figures and summary tables",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help="expected number of stochastic runs",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=12345,
        help="starting seed when --seed-mode sequence is used",
    )
    parser.add_argument(
        "--seed-mode",
        choices=["random", "sequence"],
        default="random",
        help="seed policy when launching missing runs",
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter for optional model reruns",
    )
    parser.add_argument("--jobs", type=int, default=1, help="number of parallel model runs when launching")
    parser.add_argument("--runtime", dest="runtime", action="store_true", help="pass --runtime to main.py for progress logging")
    parser.add_argument("--no-runtime", dest="runtime", action="store_false", help="disable --runtime passthrough to main.py")
    parser.set_defaults(runtime=True)
    parser.add_argument(
        "--run-missing",
        action="store_true",
        help="launch main.py for any missing run_XX/timeseries.csv outputs",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun all runs before extracting the figure",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print model commands without launching runs",
    )
    return parser.parse_args()


def style_axes(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=JOURNAL_AXIS_LINE_PT)
    ax.grid(True, color=COLORS["grid"], linewidth=JOURNAL_LINE_PT, alpha=0.22)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=400, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    idx = updated.index(flag)
    updated[idx + 1] = value
    return updated


def build_run_command(python_bin: str, run_out: Path, seed: int) -> list[str]:
    cmd = [python_bin] + list(BASE_ARGS)
    cmd = replace_flag_value(cmd, "--out", str(run_out))
    cmd = replace_flag_value(cmd, "--seed", str(seed))
    return cmd


def launch_run(
    python_bin: str,
    run_idx: int,
    run_dir: Path,
    seed: int,
    runtime: bool,
    dry_run: bool,
) -> Path:
    cmd = build_run_command(python_bin, run_dir, seed)
    if runtime:
        cmd.append("--runtime")
    print(f"[run] {run_idx:02d}: {' '.join(shlex.quote(part) for part in cmd)}")
    if not dry_run:
        run_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    return run_dir / "timeseries.csv"


def ensure_runs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [args.input / f"run_{idx:02d}" for idx in range(args.runs)]
    csv_paths = [run_dir / "timeseries.csv" for run_dir in run_dirs]
    csv_paths_by_index: dict[int, Path] = {idx: path for idx, path in enumerate(csv_paths)}
    rng = random.SystemRandom()
    run_specs: list[tuple[int, Path, int]] = []
    for idx, run_dir in enumerate(run_dirs):
        csv_path = csv_paths[idx]
        needs_run = args.overwrite or not csv_path.exists()
        if needs_run:
            if not args.run_missing and not args.overwrite:
                raise SystemExit(
                    f"missing run output: {csv_path}\n"
                    f"Use --run-missing to launch the missing model runs."
                )
            if args.seed_mode == "random":
                seed = int(rng.randrange(1, RANDOM_SEED_MAX + 1))
            else:
                seed = int(args.base_seed + idx)
            run_specs.append((idx, run_dir, seed))

    if args.jobs == 1:
        for idx, run_dir, seed in run_specs:
            csv_paths_by_index[idx] = launch_run(
                args.python, idx, run_dir, seed, args.runtime, args.dry_run
            )
    else:
        if run_specs:
            print(
                f"[sanity9_figure_extract] parallel mode: {len(run_specs)} runs with jobs={args.jobs}"
            )
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(
                    launch_run,
                    args.python,
                    idx,
                    run_dir,
                    seed,
                    args.runtime,
                    args.dry_run,
                ): idx
                for idx, run_dir, seed in run_specs
            }
            done = 0
            for fut in as_completed(futures):
                idx = futures[fut]
                csv_paths_by_index[idx] = fut.result()
                done += 1
                print(
                    f"[sanity9_figure_extract] completed run {done}/{len(run_specs)} (run_{idx:02d})"
                )
    return [csv_paths_by_index[idx] for idx in range(args.runs)]


def load_run_timeseries(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required = {
        "day",
        "alp_total",
        "collagen_total",
        "ha_total",
        "alp_preob_total",
        "alp_ob_total",
        "collagen_preob_total",
        "collagen_ob_total",
        "mineral_preob_total",
        "mineral_ob_total",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise SystemExit(f"{csv_path} is missing required columns: {', '.join(missing)}")
    out = df.loc[
        :,
        [
            "day",
            "alp_total",
            "collagen_total",
            "ha_total",
            "alp_preob_total",
            "alp_ob_total",
            "collagen_preob_total",
            "collagen_ob_total",
            "mineral_preob_total",
            "mineral_ob_total",
        ],
    ].copy()
    out["hour"] = out["day"].astype(float) * 24.0
    return out.loc[
        :,
        [
            "hour",
            "alp_total",
            "collagen_total",
            "ha_total",
            "alp_preob_total",
            "alp_ob_total",
            "collagen_preob_total",
            "collagen_ob_total",
            "mineral_preob_total",
            "mineral_ob_total",
        ],
    ]


def summarize_runs(
    run_frames: list[pd.DataFrame],
) -> tuple[
    np.ndarray,
    dict[str, dict[str, np.ndarray]],
    dict[str, dict[str, np.ndarray]],
    dict[str, np.ndarray],
]:
    if not run_frames:
        raise SystemExit("no timeseries data found")
    hours = run_frames[0]["hour"].to_numpy(dtype=float)
    for idx, frame in enumerate(run_frames[1:], start=1):
        candidate = frame["hour"].to_numpy(dtype=float)
        if candidate.shape != hours.shape or not np.allclose(candidate, hours):
            raise SystemExit(f"time grid mismatch in run index {idx}")

    summary: dict[str, dict[str, np.ndarray]] = {}
    raw_summary: dict[str, dict[str, np.ndarray]] = {}
    normalized_runs: dict[str, np.ndarray] = {}
    denom_by_marker: dict[str, float] = {}
    for marker in MARKERS:
        column = marker["column"]
        raw_matrix = np.vstack([frame[column].to_numpy(dtype=float) for frame in run_frames])
        max_hour = float(marker["max_hour"])
        visible = hours <= max_hour
        denom = (
            float(np.nanmax(raw_matrix[:, visible]))
            if np.any(visible)
            else float(np.nanmax(raw_matrix))
        )
        denom_by_marker[column] = max(denom, 1e-12)

    all_series = list(MARKERS) + COMPONENTS
    for spec in all_series:
        column = spec["column"]
        raw_matrix = np.vstack([frame[column].to_numpy(dtype=float) for frame in run_frames])
        mean_raw = raw_matrix.mean(axis=0)
        lower_raw = np.quantile(raw_matrix, 0.05, axis=0)
        upper_raw = np.quantile(raw_matrix, 0.95, axis=0)
        parent = str(spec.get("parent", column))
        denom = float(denom_by_marker[parent])
        norm_matrix = raw_matrix / denom
        normalized_runs[column] = norm_matrix.copy()
        mean_norm = norm_matrix.mean(axis=0)
        lower_norm = np.quantile(norm_matrix, 0.05, axis=0)
        upper_norm = np.quantile(norm_matrix, 0.95, axis=0)
        raw_summary[column] = {
            "mean": mean_raw,
            "lower": np.clip(lower_raw, 0.0, None),
            "upper": np.clip(upper_raw, 0.0, None),
        }
        summary[column] = {
            "mean": mean_norm,
            "lower": np.clip(lower_norm, 0.0, None),
            "upper": np.clip(upper_norm, 0.0, None),
            "scale_max_raw": float(denom),
        }
    return hours, summary, raw_summary, normalized_runs


def export_summary(hours: np.ndarray, summary: dict[str, dict[str, np.ndarray]], out_dir: Path) -> Path:
    days = hours / 24.0
    data: dict[str, np.ndarray] = {"hour": hours, "day": days}
    for marker in MARKERS:
        stem = marker["column"].replace("_total", "")
        mask = hours <= float(marker["max_hour"])
        data[f"{stem}_mean"] = np.where(mask, summary[marker["column"]]["mean"], np.nan)
        data[f"{stem}_lower"] = np.where(mask, summary[marker["column"]]["lower"], np.nan)
        data[f"{stem}_upper"] = np.where(mask, summary[marker["column"]]["upper"], np.nan)
    df = pd.DataFrame(data)
    out_path = out_dir / "representative_differentiation_trajectories_summary.csv"
    df.to_csv(out_path, index=False)
    return out_path


def export_metadata(args: argparse.Namespace, csv_paths: list[Path], out_dir: Path) -> Path:
    payload = {
        "title": "Osteogenic differentiation trajectories by cell-state contribution",
        "n_runs": int(len(csv_paths)),
        "input_root": str(args.input),
        "run_files": [str(path) for path in csv_paths],
        "main_figure": "representative_differentiation_trajectories shows normalized ALP, collagen, and mineral contributions on one shared plot with continuous 5-95% bands",
        "normalization": (
            "for each marker (ALP, collagen, mineral/ARS proxy), each run is normalized "
            "by one shared marker-specific maximum across all runs within that marker's "
            "assay window, then mean and 5-95% bands are computed from the normalized runs"
        ),
        "uncertainty_band": "5-95% prediction band computed at each time point from the empirical run distribution",
        "companion_raw_variability_figure": "representative_differentiation_trajectories_raw_variability",
        "colors": {
            "alp_preob": "#E69F00",
            "alp_ob": "#D55E00",
            "collagen_preob": "#56B4E9",
            "collagen_ob": "#009E73",
            "mineral": "#3B5BA9",
        },
    }
    out_path = out_dir / "representative_differentiation_trajectories_metadata.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def export_figure(
    hours: np.ndarray,
    summary: dict[str, dict[str, np.ndarray]],
    normalized_runs: dict[str, np.ndarray],
    out_dir: Path,
) -> Path:
    fig, ax = plt.subplots(figsize=(8.6, 5.6))
    style_axes(ax)
    days = hours / 24.0
    _ = normalized_runs  # Kept for compatibility with previous call signature.
    legend_handles: list[object] = []
    lw_curve = 1.7
    lw_mineral = 1.95
    band_alpha = 0.28
    endpoint_days = (10.0, 14.0, 28.0)
    for spec in COMPONENTS:
        stats = summary[spec["column"]]
        mask = hours <= float(spec["max_hour"])
        x = days[mask]
        mean = stats["mean"][mask]
        lower = stats["lower"][mask]
        upper = stats["upper"][mask]
        color = spec["color"]
        line_width = lw_mineral if spec["column"] == "mineral_ob_total" else lw_curve
        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=band_alpha,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=line_width,
            alpha=0.98,
            solid_capstyle="round",
            label=spec["label"],
            zorder=3,
        )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=line_width, solid_capstyle="round", label=spec["label"])
        )

    for endpoint_day in endpoint_days:
        ax.axvline(
            endpoint_day,
            color="#475569",
            linestyle="--",
            linewidth=1.2,
            alpha=0.9,
            zorder=2,
        )

    ax.set_xlim(0.0, 28.0)
    ax.set_xticks([0, 5, 10, 14, 18, 20, 25, 28])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Simulation time (days)", fontsize=11.8)
    ax.set_ylabel("Normalized marker value", fontsize=11.8)
    ax.tick_params(axis="both", labelsize=10.2)
    ax.set_title(
        "Cell-state contributions define osteogenic differentiation trajectories",
        fontsize=13.0,
        fontweight="bold",
        pad=6.0,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#475569",
            linestyle="--",
            linewidth=1.2,
            label="Endpoint assay day",
        )
    )
    legend_handles.append(
        Patch(facecolor="#475569", edgecolor="none", alpha=band_alpha, label="5-95% band")
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=4,
        frameon=False,
        columnspacing=1.0,
        handlelength=2.0,
        handletextpad=0.5,
        borderaxespad=0.0,
        fontsize=9.8,
    )
    fig.subplots_adjust(bottom=0.29, top=0.90)

    save_figure(fig, out_dir, "representative_differentiation_trajectories")
    return out_dir / "representative_differentiation_trajectories.pdf"


def export_ci_band_figure(
    hours: np.ndarray, summary: dict[str, dict[str, np.ndarray]], out_dir: Path
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 9.0), sharex=False)
    days = hours / 24.0

    for ax, marker in zip(axes, MARKERS):
        style_axes(ax)
        stats = summary[marker["column"]]
        mask = hours <= float(marker["max_hour"])
        x = days[mask]
        mean = stats["mean"][mask]
        lower = stats["lower"][mask]
        upper = stats["upper"][mask]
        color = marker["color"]

        ax.fill_between(
            x,
            lower,
            upper,
            color=color,
            alpha=0.38,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(x, lower, color=color, linewidth=1.1, alpha=0.75, zorder=2)
        ax.plot(x, upper, color=color, linewidth=1.1, alpha=0.75, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.0, zorder=3)

        ymin = float(np.nanmin(lower))
        ymax = float(np.nanmax(upper))
        pad = max(0.015, 0.10 * max(ymax - ymin, 1e-6))
        ax.set_ylim(max(0.0, ymin - pad), min(1.02, ymax + pad))
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylabel("Normalized\nvalue")
        ax.set_title(f"{marker['label']} with continuous 5-95% band", loc="left", fontsize=11.5)

    axes[-1].set_xlabel("Simulation time (days)")
    fig.suptitle("Representative differentiation trajectories", y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_dir, "representative_differentiation_trajectories_ci_bands")
    return out_dir / "representative_differentiation_trajectories_ci_bands.pdf"


def export_raw_variability_figure(
    hours: np.ndarray, raw_summary: dict[str, dict[str, np.ndarray]], out_dir: Path
) -> Path:
    fig, axes = plt.subplots(3, 1, figsize=(8.4, 9.4), sharex=False)
    days = hours / 24.0
    for ax, marker in zip(axes, MARKERS):
        style_axes(ax)
        stats = raw_summary[marker["column"]]
        mask = hours <= float(marker["max_hour"])
        x = days[mask]
        mean = stats["mean"][mask]
        lower = stats["lower"][mask]
        upper = stats["upper"][mask]
        color = marker["color"]
        ax.fill_between(x, lower, upper, color=color, alpha=0.25, linewidth=0.0, zorder=1)
        ax.plot(x, lower, color=color, linewidth=0.9, alpha=0.55, zorder=2)
        ax.plot(x, upper, color=color, linewidth=0.9, alpha=0.55, zorder=2)
        ax.plot(x, mean, color=color, linewidth=2.2, zorder=3)
        ax.set_xlim(float(x.min()), float(x.max()))
        ax.set_ylabel(f"{marker['label']}\nraw total")
        ax.set_title(f"{marker['label']} raw variability (mean with 5-95% band)", loc="left", fontsize=11.0, color=color)
    axes[-1].set_xlabel("Simulation time (days)")
    fig.suptitle("Representative differentiation trajectories: raw variability", y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_dir, "representative_differentiation_trajectories_raw_variability")
    return out_dir / "representative_differentiation_trajectories_raw_variability.pdf"


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be >= 1")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be >= 1")
    args.out.mkdir(parents=True, exist_ok=True)

    csv_paths = ensure_runs(args)
    if args.dry_run:
        return

    run_frames = [load_run_timeseries(path) for path in csv_paths]
    hours, summary, raw_summary, normalized_runs = summarize_runs(run_frames)
    summary_csv = export_summary(hours, summary, args.out)
    metadata_json = export_metadata(args, csv_paths, args.out)
    figure_pdf = export_figure(hours, summary, normalized_runs, args.out)
    ci_band_pdf = export_ci_band_figure(hours, summary, args.out)
    raw_variability_pdf = export_raw_variability_figure(hours, raw_summary, args.out)

    print(f"[ok] summary CSV: {summary_csv}")
    print(f"[ok] metadata JSON: {metadata_json}")
    print(f"[ok] figure PDF: {figure_pdf}")
    print(f"[ok] 5-95 band PDF: {ci_band_pdf}")
    print(f"[ok] raw-variability PDF: {raw_variability_pdf}")


if __name__ == "__main__":
    main()
