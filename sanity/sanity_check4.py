#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_OUT = SCRIPT_DIR / "out" / "sanity_check4"
TOTAL_CONDITIONS = 15
DEFAULT_REPLICATES = 5
BASE_SEED = 12345
DOSE_MATRIX = [
    (0.0, 0.0),
    (0.1, 0.0),
    (0.3, 0.0),
    (1.0, 0.0),
    (2.5, 0.0),
    (4.0, 0.0),
    (6.0, 0.0),
    (10.0, 0.0),
    (0.0, 1.0),
    (0.0, 3.0),
    (0.0, 6.0),
    (0.0, 10.0),
    (0.0, 30.0),
    (0.0, 60.0),
    (0.0, 100.0),
]

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
    "--no-snapshots",
    "--no-hourly-exports",
]


@dataclass(frozen=True)
class ConditionSpec:
    index: int
    dex_conc_nM: float
    mk4_conc_uM: float
    out_dir: Path

    @property
    def slug(self) -> str:
        dex_label = f"{self.dex_conc_nM:.4g}".replace(".", "p")
        mk4_label = f"{self.mk4_conc_uM:.4g}".replace(".", "p")
        return f"c{self.index:02d}_dex_{dex_label}nM_mk4_{mk4_label}uM"


@dataclass
class RunResult:
    index: int
    replicate_index: int
    seed: int
    dex_conc_nM: float
    mk4_conc_uM: float
    out_dir: str
    success: bool
    returncode: int
    command: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Single-drug dose sanity suite for Dex-only and MK-4-only runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--out", type=Path, default=BASE_OUT, help="output root directory"
    )
    parser.add_argument(
        "--python",
        type=str,
        default=sys.executable,
        help="Python interpreter used to launch main.py",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun conditions even if timeseries.csv already exists",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=DEFAULT_REPLICATES,
        help="replicates per condition",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=BASE_SEED,
        help="base RNG seed used to derive replicate seeds",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands and write manifests without launching simulations",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="forward --runtime to main.py for progress logging",
    )
    args = parser.parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    return args


def build_condition_specs(out_root: Path) -> list[ConditionSpec]:
    specs: list[ConditionSpec] = []
    for index, (mk4_conc, dex_conc) in enumerate(DOSE_MATRIX, start=1):
        out_dir = out_root / f"c{index:02d}"
        specs.append(
            ConditionSpec(
                index=index,
                dex_conc_nM=float(dex_conc),
                mk4_conc_uM=float(mk4_conc),
                out_dir=out_dir,
            )
        )
    if len(specs) != TOTAL_CONDITIONS:
        raise RuntimeError(
            f"expected {TOTAL_CONDITIONS} conditions, built {len(specs)}"
        )
    return specs


def build_command(
    spec: ConditionSpec,
    python_exe: str,
    runtime: bool,
    seed: int,
    out_dir: Path,
) -> list[str]:
    cmd = [python_exe, *BASE_ARGS]
    cmd += ["--seed", str(int(seed))]
    cmd += ["--dex-conc", f"{spec.dex_conc_nM:.10g}"]
    cmd += ["--mk4-conc", f"{spec.mk4_conc_uM:.10g}"]
    cmd += ["--out", str(out_dir)]
    if runtime:
        cmd += ["--runtime"]
    return cmd


def run_condition(
    spec: ConditionSpec,
    replicate_index: int,
    seed: int,
    *,
    python_exe: str,
    overwrite: bool,
    dry_run: bool,
    runtime: bool,
) -> RunResult:
    run_out_dir = spec.out_dir / f"r{replicate_index:02d}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = run_out_dir / "timeseries.csv"
    cmd = build_command(spec, python_exe, runtime, seed, run_out_dir)
    command_str = shlex.join(cmd)

    if timeseries_path.exists() and not overwrite:
        return RunResult(
            index=spec.index,
            replicate_index=replicate_index,
            seed=seed,
            dex_conc_nM=spec.dex_conc_nM,
            mk4_conc_uM=spec.mk4_conc_uM,
            out_dir=str(run_out_dir),
            success=True,
            returncode=0,
            command=command_str,
            error=None,
        )

    if dry_run:
        print(command_str)
        return RunResult(
            index=spec.index,
            replicate_index=replicate_index,
            seed=seed,
            dex_conc_nM=spec.dex_conc_nM,
            mk4_conc_uM=spec.mk4_conc_uM,
            out_dir=str(run_out_dir),
            success=True,
            returncode=0,
            command=command_str,
            error=None,
        )

    try:
        completed = subprocess.run(cmd, check=False)
    except OSError as exc:
        return RunResult(
            index=spec.index,
            replicate_index=replicate_index,
            seed=seed,
            dex_conc_nM=spec.dex_conc_nM,
            mk4_conc_uM=spec.mk4_conc_uM,
            out_dir=str(run_out_dir),
            success=False,
            returncode=-1,
            command=command_str,
            error=str(exc),
        )

    return RunResult(
        index=spec.index,
        replicate_index=replicate_index,
        seed=seed,
        dex_conc_nM=spec.dex_conc_nM,
        mk4_conc_uM=spec.mk4_conc_uM,
        out_dir=str(run_out_dir),
        success=(completed.returncode == 0),
        returncode=int(completed.returncode),
        command=command_str,
        error=None if completed.returncode == 0 else f"exit {completed.returncode}",
    )


def write_manifest(out_root: Path, specs: list[ConditionSpec]) -> None:
    rows = [
        {
            "index": spec.index,
            "slug": spec.slug,
            "dex_conc_nM": spec.dex_conc_nM,
            "mk4_conc_uM": spec.mk4_conc_uM,
            "out_dir": str(spec.out_dir),
        }
        for spec in specs
    ]
    with (out_root / "condition_manifest.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    with (out_root / "condition_manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["index", "slug", "dex_conc_nM", "mk4_conc_uM", "out_dir"],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_run_summary(out_root: Path, results: list[RunResult]) -> None:
    rows = [asdict(result) for result in results]
    with (out_root / "run_summary.json").open("w", encoding="utf-8") as fh:
        json.dump(rows, fh, indent=2)

    with (out_root / "run_summary.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "index",
                "replicate_index",
                "seed",
                "dex_conc_nM",
                "mk4_conc_uM",
                "out_dir",
                "success",
                "returncode",
                "command",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = build_condition_specs(args.out)
    write_manifest(args.out, specs)

    print(
        f"[sanity_check4] prepared {len(specs)} single-drug conditions from the explicit Dex/MK-4 matrix plus no-drug baseline"
    )
    print(
        "[sanity_check4] MK-4-only branch: 0.1 to 10 uM at Dex = 0 (dense sampling below 1 uM)"
    )
    print(
        "[sanity_check4] Dex-only branch: 1 to 100 nM at MK-4 = 0 (dense sampling below 10 nM)"
    )
    print("[sanity_check4] Baseline branch: 0 nM Dex and 0 uM MK-4")
    print(f"[sanity_check4] replicates per condition: {args.replicates}")

    results: list[RunResult] = []
    for spec in specs:
        print(
            f"[sanity_check4] condition {spec.index:02d}/{len(specs)}: "
            f"Dex={spec.dex_conc_nM:.4g} nM, MK-4={spec.mk4_conc_uM:.4g} uM"
        )
        for rep_idx in range(1, args.replicates + 1):
            rep_seed = int(args.base_seed + spec.index * 1000 + rep_idx)
            result = run_condition(
                spec,
                rep_idx,
                rep_seed,
                python_exe=args.python,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                runtime=args.runtime,
            )
            results.append(result)
            if not result.success:
                write_run_summary(args.out, results)
                raise SystemExit(
                    f"condition {spec.index:02d} replicate {rep_idx:02d} failed: {result.error or 'unknown error'}"
                )

    write_run_summary(args.out, results)
    print(
        f"[sanity_check4] completed {len(specs)} conditions x {args.replicates} replicates"
    )
    print(f"[sanity_check4] manifest: {args.out / 'condition_manifest.csv'}")
    print(f"[sanity_check4] summary: {args.out / 'run_summary.csv'}")


if __name__ == "__main__":
    main()
