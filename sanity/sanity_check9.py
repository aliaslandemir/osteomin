#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import random
import shlex
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_OUT = SCRIPT_DIR / "out" / "sanity_check9"
N_RUNS = 10

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sanity check 9: run-only control-condition stochastic OsteoMin simulations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=BASE_OUT, help="output directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter for main.py")
    parser.add_argument("--runs", type=int, default=N_RUNS, help="number of stochastic simulations")
    parser.add_argument(
        "--seed-mode",
        choices=["random", "sequence"],
        default="random",
        help="seed policy for stochastic runs",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=12345,
        help="starting seed when --seed-mode sequence is used",
    )
    parser.add_argument("--overwrite", action="store_true", help="rerun simulations even if timeseries.csv already exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands without launching simulations")
    return parser.parse_args()


def replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    idx = updated.index(flag)
    updated[idx + 1] = value
    return updated


def build_run_command(args: argparse.Namespace, run_out: Path, seed: int) -> list[str]:
    cmd = [args.python] + list(BASE_ARGS)
    cmd = replace_flag_value(cmd, "--out", str(run_out))
    cmd = replace_flag_value(cmd, "--seed", str(seed))
    return cmd


def run_model(args: argparse.Namespace, run_idx: int, run_out: Path, seed: int) -> Path:
    csv_path = run_out / "timeseries.csv"
    if csv_path.exists() and not args.overwrite:
        print(f"[skip] run {run_idx:02d}: {csv_path}")
        return csv_path

    cmd = build_run_command(args, run_out, seed)
    print(f"[run] {run_idx:02d}: {' '.join(shlex.quote(part) for part in cmd)}")
    if args.dry_run:
        return csv_path

    run_out.mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
    if not csv_path.exists():
        raise FileNotFoundError(f"missing expected output: {csv_path}")
    return csv_path


def write_metadata(args: argparse.Namespace, run_records: list[dict[str, object]]) -> Path:
    payload = {
        "n_runs": int(args.runs),
        "seed_mode": str(args.seed_mode),
        "base_seed": int(args.base_seed),
        "control_condition": {"dex_conc_nM": 0.0, "mk4_conc_uM": 0.0},
        "run_dirs": [str(Path(str(rec["run_dir"]))) for rec in run_records],
        "run_seeds": [
            {
                "run_index": int(rec["run_index"]),
                "seed": int(rec["seed"]),
                "run_dir": str(Path(str(rec["run_dir"]))),
            }
            for rec in run_records
        ],
        "main_py_run_style": "mirrors the default main.py command for the control condition",
        "purpose": "run-only stochastic simulation launcher; figure extraction is handled by sanity9_figure_extract.py",
    }
    out_path = args.out / "sanity_check9_metadata.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def main() -> None:
    args = parse_args()
    if args.runs <= 0:
        raise SystemExit("--runs must be >= 1")

    args.out.mkdir(parents=True, exist_ok=True)
    run_dirs = [args.out / f"run_{idx:02d}" for idx in range(args.runs)]
    rng = random.SystemRandom()
    csv_paths: list[Path] = []
    run_records: list[dict[str, object]] = []
    for idx, run_out in enumerate(run_dirs):
        if args.seed_mode == "random":
            seed = int(rng.randrange(1, 2**31 - 1))
        else:
            seed = int(args.base_seed + idx)
        csv_paths.append(run_model(args, idx, run_out, seed))
        run_records.append({"run_index": idx, "seed": seed, "run_dir": run_out})

    if args.dry_run:
        return

    metadata_json = write_metadata(args, run_records)
    print(f"[ok] generated {len(csv_paths)} run outputs under {args.out}")
    print(f"[ok] metadata: {metadata_json}")
    print("[ok] next step: python3 sanity/sanity9_figure_extract.py")


if __name__ == "__main__":
    main()
