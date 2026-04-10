#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_OUT = SCRIPT_DIR / "out" / "sanity12"
DEFAULT_CELLS = [20, 40, 80, 160, 320, 640]
RANDOM_SEED_MAX = (1 << 32) - 1

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--hours",
    "672",
    "--cells",
    "72",
    "--seed-mode",
    "random",
    "--seed",
    "12345",
    "--w",
    "33",
    "--h",
    "33",
    "--nodules",
    "40n",
    "--boundary-mode",
    "periodic",
    "--media-interval",
    "48",
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
    "--secretion",
    "--aa0",
    "50.0",
    "--aa-half-life",
    "8",
    "--collagen-aa-km",
    "5.0",
    "--alp-rb",
    "0.0949",
    "--alp-ro",
    "0.810",
    "--alp-p0",
    "0.8",
    "--collagen-rb",
    "0.00336",
    "--collagen-ro",
    "0.0130",
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
    "minimal",
    "--ha-ro",
    "0.102",
    "--ha-h0",
    "2.34",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--alp-plot-hours",
    "240",
    "--collagen-plot-hours",
    "336",
    "--mineral-plot-hours",
    "672",
    "--csv",
    "--no-xlsx",
    "--no-hourly-exports",
]


@dataclass(frozen=True)
class ConditionSpec:
    index: int
    cells: int
    out_dir: Path

    @property
    def slug(self) -> str:
        return f"c{self.index:02d}_cells_{self.cells}"


@dataclass
class RunResult:
    index: int
    cells: int
    replicate_index: int
    seed: int
    out_dir: str
    success: bool
    returncode: int
    command: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity 12 run launcher: 6 initial-cell-count conditions with stochastic "
            "replicates and random spatial initialization."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="output root directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for main.py")
    parser.add_argument("--replicates", type=int, default=3, help="replicates per initial-cell-count condition")
    parser.add_argument(
        "--cells",
        type=int,
        nargs="+",
        default=list(DEFAULT_CELLS),
        help="initial cell counts to run as separate conditions",
    )
    parser.add_argument("--jobs", type=int, default=1, help="number of parallel simulation runs")
    parser.add_argument(
        "--seed-mode",
        choices=["random", "sequence"],
        default="random",
        help="seed policy across condition/replicate runs",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=12345,
        help="base seed used when --seed-mode sequence",
    )
    parser.add_argument("--runtime", action="store_true", help="pass --runtime to main.py")
    parser.add_argument("--overwrite", action="store_true", help="rerun even if timeseries.csv exists")
    parser.add_argument("--dry-run", action="store_true", help="print commands and write manifests only")
    args = parser.parse_args()

    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be >= 1")
    if not args.cells:
        raise SystemExit("--cells must include at least one value")
    bad = [int(v) for v in args.cells if int(v) <= 0]
    if bad:
        raise SystemExit(f"all --cells values must be >= 1, got invalid: {bad}")
    return args


def replace_flag_value(args: list[str], flag: str, value: str) -> list[str]:
    updated = list(args)
    idx = updated.index(flag)
    updated[idx + 1] = value
    return updated


def build_specs(out_root: Path, cells_values: list[int]) -> list[ConditionSpec]:
    return [
        ConditionSpec(index=i + 1, cells=int(cells), out_dir=out_root / f"c{i + 1:02d}")
        for i, cells in enumerate(cells_values)
    ]


def build_command(args: argparse.Namespace, spec: ConditionSpec, run_out: Path, seed: int) -> list[str]:
    cmd = [args.python] + list(BASE_ARGS)
    cmd = replace_flag_value(cmd, "--cells", str(int(spec.cells)))
    cmd = replace_flag_value(cmd, "--seed", str(int(seed)))
    cmd = replace_flag_value(cmd, "--out", str(run_out)) if "--out" in cmd else cmd + ["--out", str(run_out)]
    if args.runtime:
        cmd.append("--runtime")
    return cmd


def run_model(
    args: argparse.Namespace,
    spec: ConditionSpec,
    replicate_index: int,
    seed: int,
) -> RunResult:
    run_out = spec.out_dir / f"r{replicate_index:02d}"
    run_out.mkdir(parents=True, exist_ok=True)
    csv_path = run_out / "timeseries.csv"
    cmd = build_command(args, spec, run_out, seed)
    command_str = shlex.join(cmd)

    if csv_path.exists() and not args.overwrite:
        return RunResult(
            index=spec.index,
            cells=spec.cells,
            replicate_index=replicate_index,
            seed=seed,
            out_dir=str(run_out),
            success=True,
            returncode=0,
            command=command_str,
            error=None,
        )

    if args.dry_run:
        print(command_str)
        return RunResult(
            index=spec.index,
            cells=spec.cells,
            replicate_index=replicate_index,
            seed=seed,
            out_dir=str(run_out),
            success=True,
            returncode=0,
            command=command_str,
            error=None,
        )

    try:
        completed = subprocess.run(cmd, check=False, cwd=PROJECT_ROOT)
    except OSError as exc:
        return RunResult(
            index=spec.index,
            cells=spec.cells,
            replicate_index=replicate_index,
            seed=seed,
            out_dir=str(run_out),
            success=False,
            returncode=-1,
            command=command_str,
            error=str(exc),
        )

    return RunResult(
        index=spec.index,
        cells=spec.cells,
        replicate_index=replicate_index,
        seed=seed,
        out_dir=str(run_out),
        success=(completed.returncode == 0),
        returncode=int(completed.returncode),
        command=command_str,
        error=None if completed.returncode == 0 else f"exit {completed.returncode}",
    )


def write_condition_manifest(out_root: Path, specs: list[ConditionSpec]) -> None:
    rows = [
        {
            "index": spec.index,
            "slug": spec.slug,
            "cells": spec.cells,
            "out_dir": str(spec.out_dir),
        }
        for spec in specs
    ]
    (out_root / "condition_manifest.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_root / "condition_manifest.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["index", "slug", "cells", "out_dir"])
        writer.writeheader()
        writer.writerows(rows)


def write_run_summary(out_root: Path, results: list[RunResult], args: argparse.Namespace) -> None:
    rows = [asdict(result) for result in results]
    (out_root / "run_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (out_root / "run_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "index",
                "cells",
                "replicate_index",
                "seed",
                "out_dir",
                "success",
                "returncode",
                "command",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metadata = {
        "out_dir": str(args.out),
        "jobs": int(args.jobs),
        "replicates": int(args.replicates),
        "seed_mode": str(args.seed_mode),
        "base_seed": int(args.base_seed),
        "cells": [int(c) for c in args.cells],
        "initialization": "fully random site placement via main.py with --seed-mode random",
        "total_runs": int(len(results)),
    }
    (out_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = build_specs(args.out, [int(v) for v in args.cells])
    write_condition_manifest(args.out, specs)

    rng = random.SystemRandom()
    used_seeds: set[int] = set()
    tasks: list[tuple[ConditionSpec, int, int]] = []
    for spec in specs:
        for rep_idx in range(1, int(args.replicates) + 1):
            if args.seed_mode == "sequence":
                seed = int(args.base_seed + (spec.index - 1) * 1000 + rep_idx)
            else:
                seed = int(rng.randrange(1, RANDOM_SEED_MAX + 1))
                while seed in used_seeds:
                    seed = int(rng.randrange(1, RANDOM_SEED_MAX + 1))
            used_seeds.add(seed)
            tasks.append((spec, rep_idx, seed))

    print(f"[sanity12] conditions: {len(specs)}")
    print(f"[sanity12] initial cells: {', '.join(str(s.cells) for s in specs)}")
    print(f"[sanity12] replicates per condition: {args.replicates}")
    print(f"[sanity12] total runs: {len(tasks)}")
    print(f"[sanity12] jobs: {args.jobs}")

    results: list[RunResult] = []
    if args.jobs == 1:
        total = len(tasks)
        for i, (spec, rep_idx, seed) in enumerate(tasks, start=1):
            print(f"[sanity12] run {i}/{total}: condition {spec.index:02d} cells={spec.cells} rep={rep_idx:02d}")
            results.append(run_model(args, spec, rep_idx, seed))
    else:
        print(f"[sanity12] parallel mode: {len(tasks)} runs with jobs={args.jobs}")
        with ThreadPoolExecutor(max_workers=args.jobs) as pool:
            futures = {
                pool.submit(run_model, args, spec, rep_idx, seed): (spec, rep_idx)
                for spec, rep_idx, seed in tasks
            }
            done = 0
            total = len(tasks)
            for fut in as_completed(futures):
                spec, rep_idx = futures[fut]
                results.append(fut.result())
                done += 1
                print(f"[sanity12] run {done}/{total}: condition {spec.index:02d} rep={rep_idx:02d}")

    results.sort(key=lambda r: (r.index, r.replicate_index))
    write_run_summary(args.out, results, args)

    failed = [r for r in results if not r.success]
    if failed:
        raise SystemExit(f"{len(failed)} run(s) failed. See {args.out / 'run_summary.csv'}")

    print(f"[sanity12] completed {len(specs)} conditions x {args.replicates} replicates")
    print(f"[sanity12] manifest: {args.out / 'condition_manifest.csv'}")
    print(f"[sanity12] summary: {args.out / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
