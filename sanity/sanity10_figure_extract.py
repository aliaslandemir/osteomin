#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import sanity_check10 as sc10

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
    parser.add_argument("--family-col", type=str, default="")
    parser.add_argument("--spatial-col", type=str, default="spatial_class")
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    src = _resolve_input(args.input, args.out)
    raw = sc10.load_input(src, args.sheet)
    sc10.run_analysis(args, raw)


if __name__ == "__main__":
    main()
