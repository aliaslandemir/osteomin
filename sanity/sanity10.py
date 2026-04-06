#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import sanity_check10 as sc10

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "out" / "sanity_check10"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity 10 simulation runner only. Builds the condition set, runs model outputs, "
            "and writes tabular outputs without generating the figure suite."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for main.py")
    parser.add_argument("--sobol-problem", type=Path, default=sc10.DEFAULT_SOBOL_PROBLEM, help="Sobol problem JSON")
    parser.add_argument("--conditions", type=int, default=300, help="Number of composite conditions")
    parser.add_argument("--replicates", type=int, default=3, help="Replicates per condition")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel simulation workers")
    parser.add_argument("--sobol-seed", type=int, default=24680, help="Sobol scramble seed")
    parser.add_argument("--base-seed", type=int, default=12345, help="Base seed for runs")
    parser.add_argument("--layout-seed", type=int, default=12345, help="Base seed for sanity2 initialization layouts")
    parser.add_argument("--cells-range", type=int, nargs=2, default=[24, 220], help="Range for initial cells")
    parser.add_argument("--dex-range", type=float, nargs=2, default=[0.0, 100.0], help="Range for Dex concentration")
    parser.add_argument("--mk4-range", type=float, nargs=2, default=[0.0, 10.0], help="Range for MK-4 concentration")
    parser.add_argument("--grid-spacing-range", type=int, nargs=2, default=[1, 12], help="Grid spacing range")
    parser.add_argument("--days", type=int, nargs="+", default=list(sc10.DEFAULT_DAYS), help="Days extracted into the long table")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-run outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print run commands only")
    parser.add_argument("--runtime", action="store_true", help="Pass --runtime through to main.py")
    args = parser.parse_args()
    if args.conditions < 1:
        raise SystemExit("--conditions must be >= 1")
    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    return args


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sim_df = sc10.simulate_conditions(args)
    if args.dry_run:
        print("[sanity10] dry run complete; simulation commands printed only")
        return
    print(f"[sanity10] conditions: {args.out / 'composite_conditions.csv'}")
    print(f"[sanity10] manifest: {args.out / 'run_manifest.csv'}")
    print(f"[sanity10] long table: {args.out / 'simulated_observables_long.csv'}")
    if sim_df.empty:
        print("[sanity10] no rows were produced")


if __name__ == "__main__":
    main()
