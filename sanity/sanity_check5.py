#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import shlex
import statistics
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_OUT = SCRIPT_DIR / "out" / "sanity_check5"
DEFAULT_REPLICATES = 5
BASE_SEED = 12345
TOTAL_HOURS = 672
HOURS_PER_DAY = 24
ALP_HOURS = 240
COLLAGEN_HOURS = 336
ARS_HOURS = 672
ALP_DAY = ALP_HOURS // HOURS_PER_DAY
COLLAGEN_DAY = COLLAGEN_HOURS // HOURS_PER_DAY
ARS_DAY = ARS_HOURS // HOURS_PER_DAY
AA_BASAL_UG_ML = 0.0
AA_OSTEO_UG_ML = 50.0
BGP_OSTEO_M_M = 10.0

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--hours",
    str(TOTAL_HOURS),
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
    "--apoptosis",
    "--apoptosis-prob",
    "0.0010",
    "--apoptosis-osteo",
    "0.0",
    "--osteo-induction-hours",
    "72",
    "--commit-delay",
    "48",
    "--commit-k",
    "0.0125",
    "--aa-half-life",
    "8",
    "--collagen-aa-km",
    "5.0",
    "--min-div-hours",
    "38",
    "--max-div-hours",
    "52",
    "--secretion",
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
    "ha",
    "--ha-ro",
    "0.102",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--spatial-every",
    "0",
    "--no-snapshots",
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
    code: str
    label: str
    aa_ug_ml: float
    bgp_mM: float
    dex_nM: float
    mk4_uM: float
    osteogenic_commitment: bool
    out_dir: Path

    @property
    def slug(self) -> str:
        return f"{self.code.lower()}_{self.label.lower().replace('+', '_').replace('/', '_').replace(' ', '_')}"


@dataclass
class RunResult:
    index: int
    code: str
    replicate_index: int
    seed: int
    aa_ug_ml: float
    bgp_mM: float
    dex_nM: float
    mk4_uM: float
    out_dir: str
    success: bool
    returncode: int
    command: str
    error: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the 8-condition sanity suite with 5 replicate simulations per condition.",
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
        help="deterministic base seed for replicate runs",
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
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="number of parallel condition/replicate runs",
    )
    parser.add_argument(
        "--snapshots",
        dest="snapshots",
        action="store_true",
        help="force raw grid snapshots (.npy) from main.py",
    )
    parser.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="disable raw grid snapshots (.npy) from main.py",
    )
    parser.set_defaults(snapshots=False)
    args = parser.parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be >= 1")
    return args


def build_condition_specs(out_root: Path) -> list[ConditionSpec]:
    definitions = [
        ("C1", "Basal", AA_BASAL_UG_ML, 0.0, 0.0, 0.0, False),
        ("C2", "AA+beta-GP", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 0.0, 0.0, True),
        ("C3", "AA+beta-GP+D_L_K_L", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 10.0, 0.1, True),
        ("C4", "AA+beta-GP+D_L_K_H", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 10.0, 10.0, True),
        ("C5", "AA+beta-GP+D_H_K_L", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 100.0, 0.1, True),
        ("C6", "AA+beta-GP+D_H_K_H", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 100.0, 10.0, True),
        ("C7", "AA+beta-GP+D_H_K_0", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 100.0, 0.0, True),
        ("C8", "AA+beta-GP+D_0_K_H", AA_OSTEO_UG_ML, BGP_OSTEO_M_M, 0.0, 10.0, True),
    ]
    specs: list[ConditionSpec] = []
    for index, (code, label, aa_ug_ml, bgp_mM, dex_nM, mk4_uM, osteogenic) in enumerate(
        definitions, start=1
    ):
        specs.append(
            ConditionSpec(
                index=index,
                code=code,
                label=label,
                aa_ug_ml=float(aa_ug_ml),
                bgp_mM=float(bgp_mM),
                dex_nM=float(dex_nM),
                mk4_uM=float(mk4_uM),
                osteogenic_commitment=bool(osteogenic),
                out_dir=out_root / code.lower(),
            )
        )
    return specs


def validate_condition_specs(specs: list[ConditionSpec]) -> None:
    for spec in specs:
        if spec.code == "C1":
            if abs(spec.aa_ug_ml - AA_BASAL_UG_ML) > 1e-12:
                raise ValueError(f"C1 must keep aa_ug_ml={AA_BASAL_UG_ML}")
            if spec.bgp_mM != 0.0:
                raise ValueError("C1 must keep bgp_mM=0.0")
            if spec.osteogenic_commitment:
                raise ValueError("C1 must keep osteogenic_commitment=False")
            continue
        if abs(spec.aa_ug_ml - AA_OSTEO_UG_ML) > 1e-12:
            raise ValueError(f"{spec.code}: aa_ug_ml must stay at {AA_OSTEO_UG_ML}")
        if abs(spec.bgp_mM - BGP_OSTEO_M_M) > 1e-12:
            raise ValueError(f"{spec.code}: bgp_mM must stay at {BGP_OSTEO_M_M}")
        if not spec.osteogenic_commitment:
            raise ValueError(f"{spec.code}: osteogenic_commitment must be True")


def commitment_flag(spec: ConditionSpec) -> list[str]:
    return (
        ["--osteogenic-commitment"]
        if spec.osteogenic_commitment
        else ["--no-osteogenic-commitment"]
    )


def build_command(
    spec: ConditionSpec,
    python_exe: str,
    runtime: bool,
    snapshots: bool,
    seed: int,
    out_dir: Path,
) -> list[str]:
    cmd = [python_exe, *BASE_ARGS]
    cmd += ["--seed", str(int(seed))]
    cmd += ["--aa0", f"{spec.aa_ug_ml:.10g}"]
    cmd += ["--dex-conc", f"{spec.dex_nM:.10g}"]
    cmd += ["--mk4-conc", f"{spec.mk4_uM:.10g}"]
    cmd += ["--out", str(out_dir)]
    cmd += commitment_flag(spec)
    cmd += ["--snapshots"] if snapshots else ["--no-snapshots"]
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
    snapshots: bool,
) -> RunResult:
    run_out_dir = spec.out_dir / f"r{replicate_index:02d}"
    run_out_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = run_out_dir / "timeseries.csv"
    cmd = build_command(spec, python_exe, runtime, snapshots, seed, run_out_dir)
    command_str = shlex.join(cmd)

    if timeseries_path.exists() and not overwrite:
        return RunResult(
            index=spec.index,
            code=spec.code,
            replicate_index=replicate_index,
            seed=seed,
            aa_ug_ml=spec.aa_ug_ml,
            bgp_mM=spec.bgp_mM,
            dex_nM=spec.dex_nM,
            mk4_uM=spec.mk4_uM,
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
            code=spec.code,
            replicate_index=replicate_index,
            seed=seed,
            aa_ug_ml=spec.aa_ug_ml,
            bgp_mM=spec.bgp_mM,
            dex_nM=spec.dex_nM,
            mk4_uM=spec.mk4_uM,
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
            code=spec.code,
            replicate_index=replicate_index,
            seed=seed,
            aa_ug_ml=spec.aa_ug_ml,
            bgp_mM=spec.bgp_mM,
            dex_nM=spec.dex_nM,
            mk4_uM=spec.mk4_uM,
            out_dir=str(run_out_dir),
            success=False,
            returncode=-1,
            command=command_str,
            error=str(exc),
        )

    return RunResult(
        index=spec.index,
        code=spec.code,
        replicate_index=replicate_index,
        seed=seed,
        aa_ug_ml=spec.aa_ug_ml,
        bgp_mM=spec.bgp_mM,
        dex_nM=spec.dex_nM,
        mk4_uM=spec.mk4_uM,
        out_dir=str(run_out_dir),
        success=(completed.returncode == 0),
        returncode=int(completed.returncode),
        command=command_str,
        error=None if completed.returncode == 0 else f"exit {completed.returncode}",
    )


def extract_endpoint_row(spec: ConditionSpec, result: RunResult) -> dict:
    run_out_dir = Path(result.out_dir)
    timeseries_path = run_out_dir / "timeseries.csv"
    if not timeseries_path.exists():
        raise FileNotFoundError(f"missing timeseries.csv in {run_out_dir}")

    rows_by_day: dict[int, dict[str, str]] = {}
    with timeseries_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        if not reader.fieldnames or "day" not in reader.fieldnames:
            raise ValueError(f"'day' column not found in {timeseries_path}")
        for row in reader:
            rows_by_day[int(float(row["day"]))] = row

    missing_days = [
        day for day in (ALP_DAY, COLLAGEN_DAY, ARS_DAY) if day not in rows_by_day
    ]
    if missing_days:
        raise ValueError(f"missing day rows {missing_days} in {timeseries_path}")

    alp_row = rows_by_day[ALP_DAY]
    collagen_row = rows_by_day[COLLAGEN_DAY]
    ars_row = rows_by_day[ARS_DAY]

    spatial_dir = run_out_dir / "spatial_cells"
    collagen_ha_dir = run_out_dir / "collagen_ha"
    spatial_exports = (
        len(list(spatial_dir.glob("*.png"))) if spatial_dir.exists() else 0
    )
    collagen_ha_exports = (
        len(list(collagen_ha_dir.glob("*.pdf"))) if collagen_ha_dir.exists() else 0
    )

    return {
        "condition_index": spec.index,
        "condition_code": spec.code,
        "condition_label": spec.label,
        "replicate_index": result.replicate_index,
        "seed": result.seed,
        "aa_ug_ml": spec.aa_ug_ml,
        "bgp_mM": spec.bgp_mM,
        "dex_nM": spec.dex_nM,
        "mk4_uM": spec.mk4_uM,
        "osteogenic_commitment": spec.osteogenic_commitment,
        "run_out_dir": str(run_out_dir),
        f"alp_{ALP_HOURS}h": float(alp_row["alp_total"]),
        f"collagen_{COLLAGEN_HOURS}h": float(collagen_row["collagen_total"]),
        f"ars_{ARS_HOURS}h": float(ars_row["ha_total"]),
        f"n_total_{ARS_HOURS}h": float(ars_row["n_total"]),
        f"n_preob_{ARS_HOURS}h": float(ars_row["n_preob"]),
        f"n_osteoblast_{ARS_HOURS}h": float(ars_row["n_osteoblast"]),
        "spatial_cells_dir": str(spatial_dir),
        "collagen_ha_dir": str(collagen_ha_dir),
        "spatial_cells_exports": spatial_exports,
        "collagen_ha_exports": collagen_ha_exports,
    }


def write_manifest(out_root: Path, specs: list[ConditionSpec]) -> None:
    rows = [
        {
            "index": spec.index,
            "code": spec.code,
            "label": spec.label,
            "slug": spec.slug,
            "aa_ug_ml": spec.aa_ug_ml,
            "bgp_mM": spec.bgp_mM,
            "dex_nM": spec.dex_nM,
            "mk4_uM": spec.mk4_uM,
            "osteogenic_commitment": spec.osteogenic_commitment,
            "out_dir": str(spec.out_dir),
            "notes": "beta-GP is recorded for manifest completeness; the simulator does not expose a separate beta-GP CLI argument.",
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
            fieldnames=[
                "index",
                "code",
                "label",
                "slug",
                "aa_ug_ml",
                "bgp_mM",
                "dex_nM",
                "mk4_uM",
                "osteogenic_commitment",
                "out_dir",
                "notes",
            ],
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
                "code",
                "replicate_index",
                "seed",
                "aa_ug_ml",
                "bgp_mM",
                "dex_nM",
                "mk4_uM",
                "out_dir",
                "success",
                "returncode",
                "command",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_endpoint_tables(out_root: Path, endpoint_rows: list[dict]) -> None:
    endpoint_rows = sorted(
        endpoint_rows,
        key=lambda row: (int(row["condition_index"]), int(row["replicate_index"])),
    )

    with (out_root / "replicate_endpoints.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(endpoint_rows[0].keys()))
        writer.writeheader()
        writer.writerows(endpoint_rows)

    grouped: dict[tuple, list[dict]] = {}
    for row in endpoint_rows:
        key = (
            row["condition_index"],
            row["condition_code"],
            row["condition_label"],
            row["aa_ug_ml"],
            row["bgp_mM"],
            row["dex_nM"],
            row["mk4_uM"],
            row["osteogenic_commitment"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict] = []
    for key, rows in sorted(grouped.items(), key=lambda item: int(item[0][0])):

        def mean(values: list[float]) -> float:
            return statistics.fmean(values)

        def sd(values: list[float]) -> float:
            return statistics.stdev(values) if len(values) > 1 else 0.0

        alp_vals = [float(row[f"alp_{ALP_HOURS}h"]) for row in rows]
        collagen_vals = [float(row[f"collagen_{COLLAGEN_HOURS}h"]) for row in rows]
        ars_vals = [float(row[f"ars_{ARS_HOURS}h"]) for row in rows]
        total_vals = [float(row[f"n_total_{ARS_HOURS}h"]) for row in rows]
        spatial_vals = [float(row["spatial_cells_exports"]) for row in rows]
        collagen_ha_vals = [float(row["collagen_ha_exports"]) for row in rows]

        summary_rows.append(
            {
                "condition_index": key[0],
                "condition_code": key[1],
                "condition_label": key[2],
                "aa_ug_ml": key[3],
                "bgp_mM": key[4],
                "dex_nM": key[5],
                "mk4_uM": key[6],
                "osteogenic_commitment": key[7],
                f"alp_{ALP_HOURS}h_mean": mean(alp_vals),
                f"alp_{ALP_HOURS}h_sd": sd(alp_vals),
                f"collagen_{COLLAGEN_HOURS}h_mean": mean(collagen_vals),
                f"collagen_{COLLAGEN_HOURS}h_sd": sd(collagen_vals),
                f"ars_{ARS_HOURS}h_mean": mean(ars_vals),
                f"ars_{ARS_HOURS}h_sd": sd(ars_vals),
                f"n_total_{ARS_HOURS}h_mean": mean(total_vals),
                f"n_total_{ARS_HOURS}h_sd": sd(total_vals),
                "spatial_cells_exports_mean": mean(spatial_vals),
                "collagen_ha_exports_mean": mean(collagen_ha_vals),
                "n_replicates": len(rows),
            }
        )

    with (out_root / "condition_endpoint_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as fh:
        writer = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    specs = build_condition_specs(args.out)
    validate_condition_specs(specs)
    write_manifest(args.out, specs)

    print(f"[sanity_check5] prepared {len(specs)} attached conditions")
    print(f"[sanity_check5] replicates per condition: {args.replicates}")
    print(f"[sanity_check5] base seed: {int(args.base_seed)}")
    print(f"[sanity_check5] jobs: {args.jobs}")
    print(f"[sanity_check5] outputs: {args.out}")

    tasks: list[tuple[ConditionSpec, int, int]] = []
    for spec in specs:
        for rep_idx in range(1, args.replicates + 1):
            rep_seed = int(args.base_seed + (spec.index - 1) * 1000 + (rep_idx - 1))
            tasks.append((spec, rep_idx, rep_seed))

    results: list[RunResult] = []
    for spec in specs:
        print(
            f"[sanity_check5] {spec.code}: AA={spec.aa_ug_ml:.3g} ug/mL, "
            f"beta-GP={spec.bgp_mM:.3g} mM, Dex={spec.dex_nM:.3g} nM, "
            f"MK-4={spec.mk4_uM:.3g} uM"
        )
    if args.jobs == 1:
        for spec, rep_idx, rep_seed in tasks:
            result = run_condition(
                spec,
                rep_idx,
                rep_seed,
                python_exe=args.python,
                overwrite=args.overwrite,
                dry_run=args.dry_run,
                runtime=args.runtime,
                snapshots=args.snapshots,
            )
            results.append(result)
            if not result.success:
                write_run_summary(args.out, results)
                raise SystemExit(
                    f"{spec.code} replicate {rep_idx:02d} failed: {result.error or 'unknown error'}"
                )
    else:
        print(f"[sanity_check5] launching {len(tasks)} runs with --jobs={args.jobs}")
        with ThreadPoolExecutor(max_workers=int(args.jobs)) as ex:
            fut_to_task = {
                ex.submit(
                    run_condition,
                    spec,
                    rep_idx,
                    rep_seed,
                    python_exe=args.python,
                    overwrite=args.overwrite,
                    dry_run=args.dry_run,
                    runtime=args.runtime,
                    snapshots=args.snapshots,
                ): (spec, rep_idx)
                for spec, rep_idx, rep_seed in tasks
            }
            for fut in as_completed(fut_to_task):
                spec, rep_idx = fut_to_task[fut]
                result = fut.result()
                results.append(result)
                if not result.success:
                    write_run_summary(args.out, results)
                    raise SystemExit(
                        f"{spec.code} replicate {rep_idx:02d} failed: {result.error or 'unknown error'}"
                    )

    write_run_summary(args.out, results)

    if args.dry_run:
        print("[sanity_check5] dry run complete; no endpoint tables generated")
        return

    endpoint_rows = [
        extract_endpoint_row(spec, result)
        for spec in specs
        for result in results
        if result.code == spec.code
    ]
    write_endpoint_tables(args.out, endpoint_rows)

    print(
        f"[sanity_check5] completed {len(specs)} conditions x {args.replicates} replicates"
    )
    print(f"[sanity_check5] manifest: {args.out / 'condition_manifest.csv'}")
    print(f"[sanity_check5] run summary: {args.out / 'run_summary.csv'}")
    print(
        f"[sanity_check5] replicate endpoints: {args.out / 'replicate_endpoints.csv'}"
    )
    print(
        f"[sanity_check5] condition summary: {args.out / 'condition_endpoint_summary.csv'}"
    )


if __name__ == "__main__":
    main()
