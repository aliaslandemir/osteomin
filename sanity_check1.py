#!/usr/bin/env python3
"""
Run sanity checks that toggle each boolean behavior on/off while keeping the
base ABM/field scenario fixed.

Each toggle produces its own `out/sanity_check/<toggle>_{on,off}` folder so
you can inspect individual runs side-by-side. A markdown report summarizes
every run once the batch completes.
"""

from __future__ import annotations

import random
import shlex
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

BASE_OUT = Path("out") / "sanity_check1"

TOGGLES = [
    {
        "name": "random_move",
        "desc": "Active cells explore neighboring lattice sites before division.",
        "on": ["--random-move"],
        "off": ["--no-random-move"],
    },
    {
        "name": "osteogenic_commitment",
        "desc": "ACTIVE cells differentiate into osteoblasts when enabled.",
        "on": ["--osteogenic-commitment"],
        "off": ["--no-osteogenic-commitment"],
    },
    {
        "name": "apoptosis",
        "desc": "Per-cell death events are sampled each day.",
        "on": ["--apoptosis"],
        "off": ["--no-apoptosis"],
    },
    {
        "name": "proliferation",
        "desc": "Cells attempt to divide once their interval elapses.",
        "on": ["--proliferation"],
        "off": ["--no-proliferation"],
    },
    {
        "name": "secretion",
        "desc": "Osteoblasts deposit collagen/ALP that fuels the field manager.",
        "on": ["--secretion"],
        "off": ["--no-secretion"],
    },
    {
        "name": "nucleation",
        "desc": "Pi+Ca form ACP, which can mature into HA.",
        "on": ["--nucleation"],
        "off": ["--no-nucleation"],
    },
    {
        "name": "commit_delay",
        "desc": "Delay (days) before commit probability turns on.",
        "on": ["--commit-delay", "3"],
        "off": ["--commit-delay", "0"],
    },
]

BASE_ARGS = [
    "main.py",
    "--days",
    "29",
    "--cells",
    "700",
    "--w",
    "100",
    "--h",
    "100",
    "--boundary-mode",
    "periodic",
    "--media-interval",
    "2",
    "--aa",
    "1.0",
    "--bgp",
    "1.0",
    "--mk4",
    "1.0",
    "--dexa",
    "1.0",
    "--move-prob",
    "1.0",
    "--max-moves",
    "1",
    "--proliferation-chance",
    "1.0",
    "--ob-div-prob",
    "0.0",
    "--apoptosis-prob",
    "0.01",
    "--apoptosis-osteo",
    "0.0005",
    "--commit-growth",
    "0.02",
    "--spatial-every",
    "1",
    "--snapshots",
    "--xlsx",
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


def run_command(args: Sequence[str]) -> tuple[bool, str | None]:
    """Run the provided command and return (success, error message)."""
    try:
        subprocess.run(args, check=True)
        return True, None
    except subprocess.CalledProcessError as exc:
        return False, f"exit {exc.returncode}"
    except OSError as exc:
        return False, str(exc)


def build_command(toggle_args: Sequence[str], out_dir: Path) -> tuple[list[str], str]:
    """Return the command and random seed for a single toggle-run."""
    cmd = [sys.executable, *BASE_ARGS]
    seed = str(random.randint(1, 999_999))
    cmd += ["--seed", seed]
    cmd += toggle_args
    cmd += ["--out", str(out_dir)]
    return cmd, seed


def write_report(results: Sequence[RunResult], path: Path) -> None:
    """Write a markdown report summarizing all toggle runs."""
    now = datetime.now().isoformat(timespec="seconds")
    lines: list[str] = [
        "# Sanity Check 1 Report",
        "",
        f"Generated: {now}",
        "",
        "## Toggle definitions",
        "",
        "| Toggle | Description |",
        "| --- | --- |",
    ]
    for toggle in TOGGLES:
        lines.append(f"| `{toggle['name']}` | {toggle['desc']} |")

    lines.extend(
        [
            "",
            "## Run matrix",
            "",
            "| Toggle | State | Seed | Output directory | Status | Command |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for result in results:
        status = "✅ success" if result.success else f"❌ {result.error or 'failed'}"
        cmd_str = shlex.join(result.cmd)
        rel_dir = result.out_dir.as_posix()
        lines.append(
            f"| `{result.toggle}` | {result.state} | {result.seed} | `{rel_dir}` | {status} | `{cmd_str}` |"
        )

    lines.extend(
        [
            "",
            "## Interpretation notes",
            "",
            "- Each directory under `out/sanity_check1/` contains the model outputs for the toggle + state combination listed above.",
            "- Compare `_on` vs `_off` directories in Excel/field outputs (`timeseries.xlsx`, `collagen_ha/`, `spatial_cells/`) to understand how the behavior impacts biology.",
            "- The `commit_delay` toggle keeps all other booleans fixed while changing the day active cells are allowed to commit.",
        ]
    )
    lines.extend(
        [
            "",
            "## Next steps",
            "",
            "1. Open relevant `timeseries.xlsx` sheets for each pair to compare deaths/commits/mineral totals.",
            "2. Use the `_snapshots` folders to verify that collagen/HA overlays obey the expected toggle.",
            "3. Adjust `--nucleation`/`--no-nucleation` runs for more than one mineralization scenario if needed.",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    BASE_OUT.mkdir(parents=True, exist_ok=True)
    results: list[RunResult] = []
    for toggle in TOGGLES:
        for state in ("on", "off"):
            out_dir = BASE_OUT / f"{toggle['name']}_{state}"
            print(f"\nRunning {toggle['name']}={state} -> {out_dir}")
            cmd, seed = build_command(toggle[state], out_dir)
            success, error = run_command(cmd)
            results.append(
                RunResult(
                    toggle=toggle["name"],
                    state=state,
                    out_dir=out_dir,
                    cmd=cmd,
                    seed=seed,
                    success=success,
                    error=error,
                )
            )
            if not success:
                print(f"Run failed for {toggle['name']}={state}: {error}")
    report_path = BASE_OUT / "report.md"
    write_report(results, report_path)


if __name__ == "__main__":
    main()
