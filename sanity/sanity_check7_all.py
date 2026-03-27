#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
import re


SCRIPT_DIR = Path(__file__).resolve().parent
SELF_NAME = Path(__file__).name


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
        description="Run all sanity_check*.py scripts sequentially and continue on errors.",
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
        "--stop-on-error",
        action="store_true",
        help="stop at first failing script (default is to continue)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="print commands without executing",
    )
    p.add_argument(
        "--scripts",
        nargs="+",
        default=None,
        help="optional explicit script list (e.g. sanity_check1.py sanity_check8.py)",
    )
    return p.parse_args()


def _script_sort_key(name: str) -> tuple[int, str]:
    m = re.fullmatch(r"sanity_check(\d+)(.*)\.py", name)
    if not m:
        return (10**9, name)
    return (int(m.group(1)), m.group(2))


def default_order() -> list[str]:
    scripts = [
        p.name
        for p in SCRIPT_DIR.glob("sanity_check*.py")
        if p.name != SELF_NAME
    ]
    return sorted(scripts, key=_script_sort_key)


def script_supports_flag(python_exe: str, script_name: str, flag: str) -> bool:
    cmd = [python_exe, str(SCRIPT_DIR / script_name), "--help"]
    try:
        proc = subprocess.run(
            cmd,
            cwd=SCRIPT_DIR.parent,
            text=True,
            capture_output=True,
            timeout=20,
            check=False,
        )
    except Exception:
        return False
    help_text = (proc.stdout or "") + "\n" + (proc.stderr or "")
    return flag in help_text


def script_cmd(
    args: argparse.Namespace, script_name: str, supports: dict[str, bool]
) -> list[str]:
    cmd = [args.python, str(SCRIPT_DIR / script_name)]
    if args.runtime and supports.get("runtime", False):
        cmd.append("--runtime")
    if args.overwrite and supports.get("overwrite", False):
        cmd.append("--overwrite")
    if args.dry_run and supports.get("dry_run", False):
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
    requested = args.scripts if args.scripts else default_order()
    run_order = [name for name in requested if name != SELF_NAME]
    if not run_order:
        print("No sanity_check scripts found.")
        return 1

    for name in run_order:
        if not (SCRIPT_DIR / name).exists():
            print(f"Missing script: {name}")
            return 1

    results: list[StepResult] = []

    print("Running sanity suite in order:")
    for idx, name in enumerate(run_order, start=1):
        print(f"  {idx}. {name}")
    print("")

    for idx, script_name in enumerate(run_order, start=1):
        supports = {
            "runtime": script_supports_flag(args.python, script_name, "--runtime"),
            "overwrite": script_supports_flag(args.python, script_name, "--overwrite"),
            "dry_run": script_supports_flag(args.python, script_name, "--dry-run"),
        }
        cmd = script_cmd(args, script_name, supports)
        print(f"[{idx}/{len(run_order)}] {script_name}")
        print(f"  cmd: {shlex.join(cmd)}")

        if args.dry_run:
            results.append(StepResult(script=script_name, returncode=0, elapsed_s=0.0))
            continue

        step = run_step(cmd, cwd=SCRIPT_DIR.parent)
        results.append(step)
        status = "OK" if step.ok else "FAILED"
        print(f"  status: {status} ({step.elapsed_s:.1f}s)")
        print("")

        if not step.ok and args.stop_on_error:
            print("Stopping on first failure. Omit --stop-on-error to continue.")
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
