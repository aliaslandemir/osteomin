#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parent
NONSPATIAL_DIR = ROOT / "nonspatialmodel"
RUNNER = NONSPATIAL_DIR / "ablation_study_run.py"
PLOTTER = NONSPATIAL_DIR / "ablation_study_plot.py"
DEFAULT_OUT = NONSPATIAL_DIR / "ablation_out"
DEFAULT_SPATIAL_OUT = ROOT.parent / "main_out"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Sanity check 8: nonspatial ablation ensemble plus ablation figure "
            "generation against the spatial main_out reference."
        )
    )
    p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output folder for ablation CSVs and figures")
    p.add_argument(
        "--spatial-out",
        type=Path,
        default=DEFAULT_SPATIAL_OUT,
        help="spatial reference folder containing run.json and timeseries.xlsx",
    )
    p.add_argument("--python", type=str, default=sys.executable, help="Python interpreter")
    p.add_argument(
        "--replicates",
        "--n-runs",
        dest="replicates",
        type=int,
        default=5,
        help="replicates per ablation case (alias: --n-runs)",
    )
    p.add_argument(
        "--param-cv",
        type=float,
        default=0.10,
        help="replicate-level CV for nonspatial parameter uncertainty",
    )
    p.add_argument("--seed", type=int, default=42, help="seed for reproducible replicate seeds and plot bootstrap")
    p.add_argument("--jobs", type=int, default=1, help="parallel replicate workers for ablation runs")
    p.add_argument(
        "--endpoint-condition",
        type=str,
        default="C2",
        choices=[f"C{i}" for i in range(1, 9)],
        help="Dex/MK-4 condition used as reference for endpoint-effect summaries (default C2)",
    )
    p.add_argument(
        "--band-type",
        choices=("prediction", "confidence"),
        default="prediction",
        help="band type for dose-response robustness figure",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="print timing/progress information during ablation run and wrapper steps",
    )
    p.add_argument("--overwrite", action="store_true", help="force rerun even if cached per-run outputs already exist")
    p.add_argument("--run-only", action="store_true", help="run ablation ensemble only (skip plotting)")
    p.add_argument("--plot-only", action="store_true", help="plot only from existing output tables (skip run step)")
    p.add_argument("--dry-run", action="store_true", help="print commands without executing")
    return p.parse_args()


def _run_cmd(cmd: list[str], dry_run: bool, runtime: bool, step: str) -> None:
    print(f"[sanity_check8] {' '.join(shlex.quote(part) for part in cmd)}")
    if dry_run:
        return
    t0 = time.perf_counter()
    subprocess.run(cmd, check=True)
    if runtime:
        print(f"[sanity_check8] {step} finished in {time.perf_counter() - t0:.2f}s")


def main() -> None:
    args = parse_args()
    if args.run_only and args.plot_only:
        raise SystemExit("Use at most one of --run-only or --plot-only.")
    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be >= 1")

    if not RUNNER.exists():
        raise FileNotFoundError(f"Missing ablation runner: {RUNNER}")
    if not PLOTTER.exists():
        raise FileNotFoundError(f"Missing ablation plotter: {PLOTTER}")

    args.out.mkdir(parents=True, exist_ok=True)
    do_run = not args.plot_only
    do_plot = not args.run_only
    overall_t0 = time.perf_counter()

    if do_run:
        spatial_run_json = args.spatial_out / "run.json"
        spatial_timeseries = args.spatial_out / "timeseries.xlsx"
        if not spatial_run_json.exists():
            raise FileNotFoundError(f"Missing spatial reference metadata: {spatial_run_json}")
        if not spatial_timeseries.exists():
            raise FileNotFoundError(f"Missing spatial reference timeseries: {spatial_timeseries}")

        run_cmd = [
            args.python,
            str(RUNNER),
            "--out",
            str(args.out),
            "--spatial-out",
            str(args.spatial_out),
            "--replicates",
            str(args.replicates),
            "--base-seed",
            str(args.seed),
            "--param-cv",
            str(args.param_cv),
            "--jobs",
            str(args.jobs),
            "--endpoint-condition",
            str(args.endpoint_condition),
        ]
        if args.runtime:
            run_cmd += ["--runtime"]
        if args.overwrite:
            run_cmd += ["--overwrite"]
        _run_cmd(run_cmd, args.dry_run, args.runtime, "ablation runs")

    if do_plot:
        plot_cmd = [
            args.python,
            str(PLOTTER),
            "--out",
            str(args.out),
            "--spatial-out",
            str(args.spatial_out),
            "--band-type",
            str(args.band_type),
            "--seed",
            str(args.seed),
        ]
        _run_cmd(plot_cmd, args.dry_run, args.runtime, "ablation plotting")

    if args.dry_run:
        print("[sanity_check8] dry run complete")
        return

    print(f"[sanity_check8] outputs: {args.out}")
    print(f"[sanity_check8] heatmap: {args.out / 'ablation_heatmap_relative_change.png'}")
    print(f"[sanity_check8] dose robustness: {args.out / 'dose_response_robustness.png'}")
    print(f"[sanity_check8] conclusion matrix: {args.out / 'conclusion_matrix.png'}")
    if args.runtime:
        print(f"[sanity_check8] total runtime: {time.perf_counter() - overall_t0:.2f}s")


if __name__ == "__main__":
    main()
