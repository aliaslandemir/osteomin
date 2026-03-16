#!/usr/bin/env python3
from __future__ import annotations

import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
BASE_OUT = SCRIPT_DIR / "out" / "sanity_check1"

TOGGLES = [
    {
        "name": "random_move",
        "desc": "Disable preOB motility.",
        "negative": ["--no-random-move"],
    },
    {
        "name": "osteogenic_commitment",
        "desc": "Disable preOB to OB commitment.",
        "negative": ["--no-osteogenic-commitment"],
    },
    {
        "name": "apoptosis",
        "desc": "Disable preOB apoptosis.",
        "negative": ["--no-apoptosis"],
    },
    {
        "name": "proliferation",
        "desc": "Disable proliferation.",
        "negative": ["--no-proliferation"],
    },
    {
        "name": "secretion",
        "desc": "Disable field deposition.",
        "negative": ["--no-secretion"],
    },
    {
        "name": "commit_delay",
        "desc": "Remove the post-induction commitment delay.",
        "negative": ["--commit-delay", "0"],
    },
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
    "--ha-ro",
    "0.106",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--ha-collagen-maturation-tau",
    "48",
    "--ha-collagen-km",
    "1.0",
    "--spatial-every",
    "24",
    "--alp-plot-hours",
    "240",
    "--collagen-plot-hours",
    "336",
    "--mineral-plot-hours",
    "672",
    "--csv",
    "--no-xlsx",
    "--no-unique-id",
    "--no-snapshots",
    "--no-hourly-exports",
    "--runtime",
]

MONOTONE_COLUMNS = [
    "alp_total",
    "collagen_total",
    "ha_total",
    "alp_preob_total",
    "alp_ob_total",
    "collagen_preob_total",
    "collagen_ob_total",
    "mineral_preob_total",
    "mineral_ob_total",
    "nodule_count_ever",
]

FINAL_COLUMNS = [
    "n_preob",
    "n_osteoblast",
    "n_total",
    "alp_total",
    "collagen_total",
    "ha_total",
    "committed_today",
    "dead_preob_today",
    "dead_ob_today",
    "nodule_count_ever",
    "msd_effective",
    "avg_move_count",
    "avg_division_interval",
    "avg_live_age_hours",
]


@dataclass
class RunResult:
    toggle: str
    state: str
    out_dir: Path
    cmd: list[str]
    seed: str
    success: bool
    error: str | None


@dataclass
class AnalysisResult:
    final_metrics: dict[str, float]
    monotone_ok: bool
    nonnegative_ok: bool
    counts_ok: bool
    notes: list[str]


def run_command(args: list[str]) -> tuple[bool, str | None]:
    try:
        subprocess.run(args, check=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, f"exit {exc.returncode}"
    except OSError as exc:
        return False, str(exc)


def build_command(toggle_args: Iterable[str], out_dir: Path) -> tuple[list[str], str]:
    cmd = [sys.executable, *BASE_ARGS]
    seed = "12345"
    cmd += list(toggle_args)
    cmd += ["--out", str(out_dir)]
    return cmd, seed


def load_timeseries(out_dir: Path) -> pd.DataFrame:
    path = out_dir / "timeseries.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _is_monotone_non_decreasing(series: pd.Series) -> bool:
    values = pd.to_numeric(series, errors="coerce").fillna(0.0)
    diffs = values.diff().fillna(0.0)
    return bool((diffs >= -1e-9).all())


def analyze_timeseries(df: pd.DataFrame) -> AnalysisResult:
    final_row = df.iloc[-1]
    final_metrics: dict[str, float] = {}
    for col in FINAL_COLUMNS:
        if col in df.columns:
            final_metrics[col] = float(final_row[col])

    monotone_ok = True
    notes: list[str] = []
    for col in MONOTONE_COLUMNS:
        if col in df.columns and not _is_monotone_non_decreasing(df[col]):
            monotone_ok = False
            notes.append(f"{col} is not monotone")

    nonnegative_cols = [c for c in FINAL_COLUMNS if c in df.columns]
    nonnegative_ok = True
    for col in nonnegative_cols:
        if (pd.to_numeric(df[col], errors="coerce").fillna(0.0) < -1e-9).any():
            nonnegative_ok = False
            notes.append(f"{col} contains negative values")

    counts_ok = True
    if all(col in df.columns for col in ("n_preob", "n_osteoblast", "n_total")):
        lhs = df["n_preob"].fillna(0).astype(float) + df["n_osteoblast"].fillna(0).astype(float)
        rhs = df["n_total"].fillna(0).astype(float)
        if not ((lhs - rhs).abs() <= 1e-9).all():
            counts_ok = False
            notes.append("n_preob + n_osteoblast != n_total")

    return AnalysisResult(
        final_metrics=final_metrics,
        monotone_ok=monotone_ok,
        nonnegative_ok=nonnegative_ok,
        counts_ok=counts_ok,
        notes=notes,
    )


def compare_to_control(control: AnalysisResult, trial: AnalysisResult) -> dict[str, float]:
    delta: dict[str, float] = {}
    for key, control_val in control.final_metrics.items():
        if key in trial.final_metrics:
            delta[key] = float(trial.final_metrics[key] - control_val)
    return delta


def _fmt(value: float | None) -> str:
    if value is None:
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 1:
        return f"{value:.3f}"
    return f"{value:.4f}"


def write_report(
    results: list[RunResult],
    analyses: dict[str, AnalysisResult],
    path: Path,
) -> None:
    now = datetime.now().isoformat(timespec="seconds")
    base_cmd = shlex.join([sys.executable, *BASE_ARGS])
    lines: list[str] = [
        "# Sanity Check 1 Report",
        "",
        f"Generated: {now}",
        "",
        "## Command template",
        "",
        f"`{base_cmd} --out <dir>`",
        "",
        "## Run status",
        "",
        "| Scenario | Seed | Status | Output |",
        "| --- | ---: | --- | --- |",
    ]

    for result in results:
        status = "success" if result.success else f"failed ({result.error})"
        lines.append(
            f"| `{result.toggle}:{result.state}` | {result.seed} | {status} | `{result.out_dir.as_posix()}` |"
        )

    control_key = "positive_control:positive"
    control = analyses.get(control_key)
    if control is not None:
        lines.extend(
            [
                "",
                "## Positive control final state",
                "",
                "| Metric | Value |",
                "| --- | ---: |",
            ]
        )
        for key in FINAL_COLUMNS:
            if key in control.final_metrics:
                lines.append(f"| `{key}` | {_fmt(control.final_metrics[key])} |")

    lines.extend(
        [
            "",
            "## Structural checks",
            "",
            "| Scenario | Monotone cumulative fields | Non-negative tracked fields | Count identity | Notes |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for key, analysis in analyses.items():
        notes = "; ".join(analysis.notes) if analysis.notes else "ok"
        lines.append(
            f"| `{key}` | {'yes' if analysis.monotone_ok else 'no'} | {'yes' if analysis.nonnegative_ok else 'no'} | {'yes' if analysis.counts_ok else 'no'} | {notes} |"
        )

    if control is not None:
        lines.extend(
            [
                "",
                "## Toggle contrasts vs positive control",
                "",
                "| Scenario | d_n_total | d_ALP | d_Collagen | d_HAp | d_Nodules | d_MSD |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for key, analysis in analyses.items():
            if key == control_key:
                continue
            delta = compare_to_control(control, analysis)
            lines.append(
                "| `{}` | {} | {} | {} | {} | {} | {} |".format(
                    key,
                    _fmt(delta.get("n_total")),
                    _fmt(delta.get("alp_total")),
                    _fmt(delta.get("collagen_total")),
                    _fmt(delta.get("ha_total")),
                    _fmt(delta.get("nodule_count_ever")),
                    _fmt(delta.get("msd_effective")),
                )
            )

    lines.extend(
        [
            "",
            "## Toggle definitions",
            "",
            "| Toggle | Meaning of negative run |",
            "| --- | --- |",
        ]
    )
    for toggle in TOGGLES:
        lines.append(f"| `{toggle['name']}` | {toggle['desc']} |")

    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- This sanity suite exercises the current direct field-law model, not the legacy default `datafit` workflow.",
            "- Runs export `timeseries.csv` and use that file for consistency checks and control-vs-perturbation comparisons.",
            "- HAp is still stored internally under `ha_total` in the timeseries schema.",
            "- In the minimal HAp model, the mineral increment is scaled by a matured-collagen state that relaxes toward total collagen with timescale `--ha-collagen-maturation-tau`.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    analyses: dict[str, AnalysisResult] = {}

    positive_dir = BASE_OUT / "positive_control"
    cmd, seed = build_command([], positive_dir)
    success, error = run_command(cmd)
    results.append(
        RunResult(
            toggle="positive_control",
            state="positive",
            out_dir=positive_dir,
            cmd=cmd,
            seed=seed,
            success=success,
            error=error,
        )
    )

    if success:
        analyses["positive_control:positive"] = analyze_timeseries(load_timeseries(positive_dir))

    for toggle in TOGGLES:
        out_dir = BASE_OUT / f"{toggle['name']}_negative"
        cmd, seed = build_command(toggle["negative"], out_dir)
        success, error = run_command(cmd)
        results.append(
            RunResult(
                toggle=toggle["name"],
                state="negative",
                out_dir=out_dir,
                cmd=cmd,
                seed=seed,
                success=success,
                error=error,
            )
        )
        if success:
            analyses[f"{toggle['name']}:negative"] = analyze_timeseries(load_timeseries(out_dir))

    report_path = BASE_OUT / "report.md"
    write_report(results, analyses, report_path)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
