#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_ORDER = [
    "sanity_check1.py",
    "sanity_check2.py",
    "sanity_check3.py",
    "sanity_check4.py",
    "sanity_check5.py",
    "sanity_check6.py",
]


@dataclass
class StepResult:
    script: str
    returncode: int
    elapsed_s: float

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run sanity_check1..6 sequentially using the current OsteoMin defaults.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch sanity scripts",
    )
    p.add_argument(
        "--runtime",
        action="store_true",
        help="forward --runtime to each sanity script if supported",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="forward --overwrite to each sanity script if supported",
    )
    p.add_argument(
        "--continue-on-error",
        action="store_true",
        help="keep running subsequent sanity scripts after a failure",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing",
    )
    return p.parse_args()


def script_cmd(args: argparse.Namespace, script_name: str) -> list[str]:
    cmd = [args.python, str(SCRIPT_DIR / script_name)]
    if args.runtime:
        cmd.append("--runtime")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.dry_run:
        cmd.append("--dry-run")
    return cmd


def run_step(cmd: list[str], cwd: Path) -> StepResult:
    script = Path(cmd[1]).name
    t0 = time.perf_counter()
    proc = subprocess.run(cmd, cwd=cwd)
    elapsed = time.perf_counter() - t0
    return StepResult(script=script, returncode=proc.returncode, elapsed_s=elapsed)


def main() -> int:
    args = parse_args()
    results: list[StepResult] = []

    print("Running sanity suite in order:")
    for idx, name in enumerate(DEFAULT_ORDER, start=1):
        print(f"  {idx}. {name}")
    print("")

    for idx, script_name in enumerate(DEFAULT_ORDER, start=1):
        cmd = script_cmd(args, script_name)
        print(f"[{idx}/{len(DEFAULT_ORDER)}] {script_name}")
        print(f"  cmd: {shlex.join(cmd)}")

        if args.dry_run:
            results.append(StepResult(script=script_name, returncode=0, elapsed_s=0.0))
            continue

        step = run_step(cmd, cwd=SCRIPT_DIR.parent)
        results.append(step)
        status = "OK" if step.ok else "FAILED"
        print(f"  status: {status} ({step.elapsed_s:.1f}s)")
        print("")

        if not step.ok and not args.continue_on_error:
            print("Stopping on first failure. Use --continue-on-error to run all steps.")
            break

    failures = [r for r in results if not r.ok]
    print("Summary:")
    for r in results:
        status = "OK" if r.ok else f"FAILED(rc={r.returncode})"
        print(f"  - {r.script}: {status}, {r.elapsed_s:.1f}s")

    if failures:
        print(f"\nSuite completed with {len(failures)} failure(s).")
        return 1
    print("\nSuite completed successfully.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
