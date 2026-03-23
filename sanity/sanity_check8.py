#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
RUNNER = ROOT / "nonspatialmodel" / "run_nonspatial_ablation.py"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sanity check 8: reduced nonspatial OsteoMin ablation run."
    )
    p.add_argument("--out", type=Path, default=ROOT / "nonspatialmodel" / "ablation_out")
    p.add_argument("--dex", type=float, default=10.0)
    p.add_argument("--mk4", type=float, default=1.0)
    p.add_argument("--hours", type=int, default=28 * 24)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cmd = [
        sys.executable,
        str(RUNNER),
        "--out",
        str(args.out),
        "--dex",
        str(args.dex),
        "--mk4",
        str(args.mk4),
        "--hours",
        str(args.hours),
    ]
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
