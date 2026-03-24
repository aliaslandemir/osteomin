#!/usr/bin/env python3
from __future__ import annotations

import argparse
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

DEFAULT_INPUT = SCRIPT_DIR / "out" / "sanity_check9"
DEFAULT_OUT = DEFAULT_INPUT / "figure_extract_9"
DEFAULT_RUNS = 10

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--out",
    "main_out",
    "--seed",
    "0",
    "--runtime",
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
    "0.094",
    "--alp-ro",
    "0.810",
    "--collagen-rb",
    "0.00355",
    "--collagen-ro",
    "0.0131",
    "--ha-ro",
    "0.106",
    "--alp-dex-ref",
    "20",
    "--collagen-dex-ref",
    "60",
    "--ars-dex-ref",
    "50",
    "--alp-a-d",
    "3.79",
    "--alp-b-d",
    "1.92",
    "--collagen-a-d",
    "2.42",
    "--collagen-b-d",
    "0.770",
    "--ha-a-d",
    "1.14",
    "--ha-b-d",
    "0.365",
    "--alp-mk4-ref",
    "0.6",
    "--collagen-mk4-ref",
    "8",
    "--ars-mk4-ref",
    "0.7",
    "--alp-a-k",
    "0.56",
    "--alp-b-k",
    "0.471",
    "--collagen-a-k",
    "0.915",
    "--collagen-b-k",
    "0.167",
    "--ha-a-k",
    "4.69",
    "--ha-b-k",
    "2.21",
    "--ha-model",
    "minimal",
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
    {"column": "alp_total", "label": "ALP", "color": "#f28e2b", "max_hour": 240.0},
    {"column": "collagen_total", "label": "Collagen", "color": "#59a14f", "max_hour": 336.0},
    {"column": "ha_total", "label": "Mineral", "color": "#4e79a7", "max_hour": 672.0},
]

COMPONENTS = [
    {
        "column": "alp_preob_total",
        "label": "ALP preOB",
        "parent": "alp_total",
        "color": "#f6b26b",
        "max_hour": 240.0,
    },
    {
        "column": "alp_ob_total",
        "label": "ALP OB",
        "parent": "alp_total",
        "color": "#f28e2b",
        "max_hour": 240.0,
    },
    {
        "column": "collagen_preob_total",
        "label": "Collagen preOB",
        "parent": "collagen_total",
        "color": "#90c987",
        "max_hour": 336.0,
    },
    {
        "column": "collagen_ob_total",
        "label": "Collagen OB",
        "parent": "collagen_total",
        "color": "#59a14f",
        "max_hour": 336.0,
    },
    {
        "column": "mineral_ob_total",
        "label": "Mineral",
        "parent": "ha_total",
        "color": "#4e79a7",
        "max_hour": 672.0,
    },
]

COLORS = {
    "ink": "#0f172a",
    "grid": "#d8dee9",
    "bg": "#ffffff",
}

CI_COLORS = {
    "alp_total": "#f8c792",
    "collagen_total": "#bfe0bc",
    "ha_total": "#bfd7ef",
    "alp_preob_total": "#f9d9b5",
    "alp_ob_total": "#f8c792",
    "collagen_preob_total": "#d4ecd0",
    "collagen_ob_total": "#bfe0bc",
    "mineral_ob_total": "#bfd7ef",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 12.5,
        "axes.titleweight": "semibold",
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
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
    ax.tick_params(direction="out", length=4.0, width=0.8)
    ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.35)


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


def ensure_runs(args: argparse.Namespace) -> list[Path]:
    run_dirs = [args.input / f"run_{idx:02d}" for idx in range(args.runs)]
    csv_paths: list[Path] = []
    rng = random.SystemRandom()
    for idx, run_dir in enumerate(run_dirs):
        csv_path = run_dir / "timeseries.csv"
        needs_run = args.overwrite or not csv_path.exists()
        if needs_run:
            if not args.run_missing and not args.overwrite:
                raise SystemExit(
                    f"missing run output: {csv_path}\n"
                    f"Use --run-missing to launch the missing model runs."
                )
            if args.seed_mode == "random":
                seed = int(rng.randrange(1, 2**31 - 1))
            else:
                seed = int(args.base_seed + idx)
            cmd = build_run_command(args.python, run_dir, seed)
            print(f"[run] {idx:02d}: {' '.join(shlex.quote(part) for part in cmd)}")
            if not args.dry_run:
                run_dir.mkdir(parents=True, exist_ok=True)
                subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        csv_paths.append(csv_path)
    return csv_paths


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
    n = len(run_frames)
    # Two-sided 95% t critical values for small n used in sanity_check9.
    t_975_by_df = {
        1: 12.706204736432095,
        2: 4.302652729696142,
        3: 3.182446305284264,
        4: 2.7764451051977987,
        5: 2.570581835636314,
        6: 2.4469118511449692,
        7: 2.3646242515927844,
        8: 2.306004135204166,
        9: 2.2621571628540993,
        10: 2.2281388519649385,
        11: 2.200985160082949,
        12: 2.1788128296634177,
    }
    t_crit = t_975_by_df.get(max(1, n - 1), 1.959963984540054)
    denom_by_marker: dict[str, float] = {}
    total_matrices: dict[str, np.ndarray] = {}
    for marker in MARKERS:
        column = marker["column"]
        raw_matrix = np.vstack([frame[column].to_numpy(dtype=float) for frame in run_frames])
        total_matrices[column] = raw_matrix
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
        if n > 1:
            sem_raw = raw_matrix.std(axis=0, ddof=1) / np.sqrt(float(n))
            delta_raw = t_crit * sem_raw
        else:
            delta_raw = np.zeros_like(mean_raw)
        parent = str(spec.get("parent", column))
        denom = float(denom_by_marker[parent])
        norm_matrix = raw_matrix / denom
        normalized_runs[column] = norm_matrix.copy()
        mean_norm = norm_matrix.mean(axis=0)
        if n > 1:
            sem_norm = norm_matrix.std(axis=0, ddof=1) / np.sqrt(float(n))
            delta_norm = t_crit * sem_norm
        else:
            delta_norm = np.zeros_like(mean_norm)
        raw_summary[column] = {
            "mean": mean_raw,
            "lower": np.clip(mean_raw - delta_raw, 0.0, None),
            "upper": np.clip(mean_raw + delta_raw, 0.0, None),
        }
        summary[column] = {
            "mean": mean_norm,
            "lower": np.clip(mean_norm - delta_norm, 0.0, None),
            "upper": np.clip(mean_norm + delta_norm, 0.0, None),
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
        "main_figure": "representative_differentiation_trajectories shows normalized ALP, collagen, and mineral contributions on one shared plot with continuous 95% CI bands",
        "normalization": (
            "for each marker (ALP, collagen, mineral/ARS proxy), each run is normalized "
            "by one shared marker-specific maximum across all runs within that marker's "
            "assay window, then mean and 95% CI are computed from the normalized runs"
        ),
        "confidence_interval": "95% CI computed continuously at each time point as mean +/- t_(0.975, n-1) * SEM across runs",
        "companion_raw_variability_figure": "representative_differentiation_trajectories_raw_variability",
        "colors": {
            "alp": "#f28e2b",
            "collagen": "#59a14f",
            "mineral": "#4e79a7",
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
    legend_handles: list[object] = []
    assay_guides = [
        (10.0, "#f28e2b"),
        (14.0, "#59a14f"),
        (28.0, "#4e79a7"),
    ]
    for xpos, color in assay_guides:
        ax.axvline(
            xpos,
            color=color,
            linewidth=0.8,
            linestyle=(0, (2.2, 2.2)),
            alpha=0.12,
            zorder=0,
        )
    for spec in COMPONENTS:
        stats = summary[spec["column"]]
        run_matrix = normalized_runs[spec["column"]]
        mask = hours <= float(spec["max_hour"])
        x = days[mask]
        mean = stats["mean"][mask]
        lower = stats["lower"][mask]
        upper = stats["upper"][mask]
        color = spec["color"]
        ci_fill = CI_COLORS.get(spec["column"], "#d7dee8")
        for run_vals in run_matrix:
            ax.plot(
                x,
                run_vals[mask],
                color="#475569",
                linewidth=0.38,
                alpha=0.16,
                solid_capstyle="round",
                zorder=1.4,
            )
        ax.fill_between(
            x,
            lower,
            upper,
            color=ci_fill,
            alpha=0.52,
            linewidth=0.0,
            zorder=1,
        )
        ax.plot(
            x,
            mean,
            color=color,
            linewidth=1.35,
            alpha=1.0,
            solid_capstyle="round",
            label=spec["label"],
            zorder=3,
        )
        if x.size:
            ax.plot(
                [float(x[-1])],
                [float(mean[-1])],
                marker="o",
                markersize=5.8,
                markerfacecolor=color,
                markeredgecolor=color,
                markeredgewidth=1.0,
                linestyle="None",
                zorder=4,
            )
        legend_handles.append(
            Line2D([0], [0], color=color, lw=2.0, solid_capstyle="round", label=spec["label"])
        )

    ax.set_xlim(float(days.min()), float(days.max()) + 0.35)
    ax.set_xticks([0, 5, 10, 14, 20, 25, 28])
    ax.set_ylim(0.0, 1.02)
    ax.set_xlabel("Simulation time (days)")
    ax.set_ylabel("Normalized marker value")
    ax.set_title("Osteogenic differentiation trajectories by cell-state contribution")
    legend_handles.append(
        Patch(facecolor="#d7dee8", edgecolor="none", alpha=0.55, label="95% CI")
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            color="#94a3b8",
            lw=1.0,
            linestyle=(0, (2.2, 2.2)),
            label="Assay day",
        )
    )
    legend_handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            markersize=6.0,
            markerfacecolor="#475569",
            markeredgecolor="#475569",
            markeredgewidth=1.0,
            linestyle="None",
            label="Endpoint",
        )
    )
    ax.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.21),
        ncol=5,
        frameon=False,
        columnspacing=1.4,
        handlelength=2.0,
        handletextpad=0.6,
        borderaxespad=0.0,
    )
    fig.subplots_adjust(bottom=0.32)

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
        ax.set_title(f"{marker['label']} with continuous 95% CI", loc="left", fontsize=11.5)

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
        ax.set_title(f"{marker['label']} raw variability (mean ± 95% CI)", loc="left", fontsize=11.0, color=color)
    axes[-1].set_xlabel("Simulation time (days)")
    fig.suptitle("Representative differentiation trajectories: raw variability", y=0.995, fontsize=14, fontweight="semibold")
    fig.tight_layout()
    save_figure(fig, out_dir, "representative_differentiation_trajectories_raw_variability")
    return out_dir / "representative_differentiation_trajectories_raw_variability.pdf"


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be >= 1")
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
    print(f"[ok] CI-band PDF: {ci_band_pdf}")
    print(f"[ok] raw-variability PDF: {raw_variability_pdf}")


if __name__ == "__main__":
    main()
