#!/usr/bin/env python3

"""

python3 sanity/sanity_check10.py --out sanity/out/sanity_check10_pub --jobs 8 --replicates 10

"""


from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import subprocess
import sys
import warnings
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Iterable

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "osteomin_mplconfig").resolve()))

import matplotlib
import numpy as np
import pandas as pd
from matplotlib.patches import Ellipse, Rectangle

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import chi2, qmc, t as student_t

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sanity_check2 import generate_layout, make_condition_specs


DEFAULT_OUT = SCRIPT_DIR / "out" / "sanity_check10"
DEFAULT_SOBOL_PROBLEM = SCRIPT_DIR / "sobol_problem_pca.json"
DEFAULT_DAYS = (10, 14, 28)
TOTAL_HOURS = 672
GRID_W = 33
GRID_H = 33

TYPE_BASELINE = "0_baseline"
TYPE_SPATIAL = "1_spatial_pattern"
TYPE_CELLS = "2_cell_number"
TYPE_DYNAMICS = "3_cell_dynamics"
TYPE_AA = "4_ascorbic_acid"
TYPE_BOUNDARY = "5_boundary_scale"
TYPE_SAMPLING = "6_sampling_time"
FAMILY_LABELS = {
    TYPE_BASELINE: "Baseline",
    TYPE_SPATIAL: "Spatial pattern",
    TYPE_CELLS: "Cell number",
    TYPE_DYNAMICS: "Cell dynamics",
    TYPE_AA: "Ascorbic acid",
    TYPE_BOUNDARY: "Boundary/scale",
    TYPE_SAMPLING: "Sampling time",
}

FEATURES = ("ALP", "collagen", "mineral")
LATENT_FEATURE_CANDIDATES = ("OB_fraction", "clustering", "msd_effective")
ENDPOINT_SPECS = (
    {"day": 10, "feature": "ALP", "label": "ALP day 10"},
    {"day": 14, "feature": "collagen", "label": "Collagen day 14"},
    {"day": 28, "feature": "mineral", "label": "Mineral day 28"},
)

PARAM_TO_FLAG = {
    "osteo_induction_hours": "--osteo-induction-hours",
    "commit_delay": "--commit-delay",
    "commit_k": "--commit-k",
    "move_prob": "--move-prob",
    "min_div_hours": "--min-div-hours",
    "max_div_hours": "--max-div-hours",
    "apoptosis_prob": "--apoptosis-prob",
}
INT_PARAMS = {
    "init_cells",
    "grid_spacing",
    "osteo_induction_hours",
    "commit_delay",
    "min_div_hours",
    "div_gap_hours",
    "max_div_hours",
    "w",
    "h",
    "media_interval",
    "sampling_offset_days",
}
BASELINE_VALUES = {
    "init_cells": 72,
    "seed_mode": "random",
    "grid_spacing": 3,
    "w": 33,
    "h": 33,
    "boundary_mode": "periodic",
    "media_interval": 48,
    "aa0": 50.0,
    "aa_half_life": 8.0,
    "collagen_aa_km": 5.0,
    "sampling_offset_days": 0,
    "osteo_induction_hours": 72,
    "commit_delay": 48,
    "commit_k": 0.0125,
    "move_prob": 0.27,
    "min_div_hours": 38,
    "div_gap_hours": 14,
    "max_div_hours": 52,
    "apoptosis_prob": 0.0010,
}

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--hours",
    str(TOTAL_HOURS),
    "--w",
    str(GRID_W),
    "--h",
    str(GRID_H),
    "--boundary-mode",
    "periodic",
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
    "--max-moves",
    "1",
    "--proliferation",
    "--ob-div-prob",
    "0.0",
    "--apoptosis",
    "--apoptosis-osteo",
    "0.0",
    "--osteogenic-commitment",
    "--alp-rb",
    "0.0949",
    "--alp-ro",
    "0.810",
    "--collagen-rb",
    "0.00336",
    "--collagen-ro",
    "0.0130",
    "--ha-ro",
    "0.102",
    "--ha-model",
    "ha",
    "--ha-collagen-maturation-tau",
    "48",
    "--ha-collagen-km",
    "1.0",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--spatial-every",
    "0",
    "--alp-plot-hours",
    "240",
    "--collagen-plot-hours",
    "336",
    "--mineral-plot-hours",
    "672",
    "--csv",
    "--no-xlsx",
    "--no-hourly-exports",
    "--no-snapshots",
    "--no-alp-snapshots",
    "--no-unique-id",
]

COLORS = {
    TYPE_BASELINE: "#111827",
    TYPE_SPATIAL: "#f28e2b",
    TYPE_CELLS: "#1f77b4",
    TYPE_DYNAMICS: "#7f3c8d",
    TYPE_AA: "#2ca02c",
    TYPE_BOUNDARY: "#8c564b",
    TYPE_SAMPLING: "#e15759",
    "ink": "#0f172a",
    "grid": "#d8dee9",
    "bg": "#ffffff",
    "muted": "#64748b",
    "baseline": "#111827",
    "nonsig": "#94a3b8",
}
MARKERS = {
    TYPE_BASELINE: "*",
    TYPE_SPATIAL: "o",
    TYPE_CELLS: "^",
    TYPE_DYNAMICS: "s",
    TYPE_AA: "D",
    TYPE_BOUNDARY: "P",
    TYPE_SAMPLING: "X",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 1.0,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 13.0,
        "axes.labelsize": 10.5,
        "xtick.labelsize": 9.0,
        "ytick.labelsize": 9.0,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "legend.fontsize": 8.5,
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class ScalingParams:
    center: np.ndarray
    scale: np.ndarray
    columns: list[str]


@dataclass(frozen=True)
class PcaModel:
    mean: np.ndarray
    components: np.ndarray
    explained_variance_ratio: np.ndarray
    columns: list[str]


@dataclass(frozen=True)
class DayMetrics:
    day: int
    resolution_index: float
    collapsed_fraction: float
    collapsed_pairs: int
    total_pairs: int
    latent_distance: float


def safe_to_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    return value


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity check 10: composite degeneracy screen built from the prior "
            "sanity initialization and kinetic ranges for day 10/14/28 readouts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python executable for main.py")
    parser.add_argument("--input", type=Path, default=None, help="Optional prebuilt long-format CSV/XLSX table")
    parser.add_argument("--sheet", type=str, default="", help="Optional sheet for XLSX input")
    parser.add_argument("--analyze-only", action="store_true", help="Skip simulation and analyze an existing table")
    parser.add_argument("--sobol-problem", type=Path, default=DEFAULT_SOBOL_PROBLEM, help="Sobol problem JSON")
    parser.add_argument("--conditions", type=int, default=301, help="Total number of conditions including the baseline condition")
    parser.add_argument("--replicates", type=int, default=10, help="Replicates per condition")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel simulation workers")
    parser.add_argument("--sobol-seed", type=int, default=24680, help="Sobol scramble seed")
    parser.add_argument(
        "--seed-policy",
        choices=("entropy",),
        default="entropy",
        help="Replicate seeds drawn from OS entropy (non-deterministic)",
    )
    parser.add_argument("--layout-seed", type=int, default=12345, help="Base seed for sanity2 initialization layouts")
    parser.add_argument("--cells-range", type=int, nargs=2, default=[24, 220], help="Range for initial cells")
    parser.add_argument("--grid-spacing-range", type=int, nargs=2, default=[1, 12], help="Grid spacing range")
    parser.add_argument("--aa0-range", type=float, nargs=2, default=[25.0, 90.0], help="Range for initial AA concentration (ug/mL)")
    parser.add_argument("--aa-half-life-range", type=float, nargs=2, default=[4.0, 12.0], help="Range for AA half-life (h)")
    parser.add_argument("--collagen-aa-km-range", type=float, nargs=2, default=[2.5, 8.0], help="Range for collagen-AA saturation KM")
    parser.add_argument("--media-interval-range", type=int, nargs=2, default=[24, 72], help="Range for media change interval (h)")
    parser.add_argument("--grid-size-range", type=int, nargs=2, default=[25, 49], help="Range for boundary/scale grid size (w=h)")
    parser.add_argument("--sampling-offset-days-range", type=int, nargs=2, default=[-2, 2], help="Range for endpoint sampling offset in days")
    parser.add_argument("--days", type=int, nargs="+", default=list(DEFAULT_DAYS), help="Day panels and extracted rows")
    parser.add_argument("--scaling", choices=("global", "per-day"), default="global")
    parser.add_argument("--pca-mode", choices=("global", "per-day"), default="global")
    parser.add_argument("--equivalence-threshold", type=float, default=1.0)
    parser.add_argument("--threshold-sweep", type=float, nargs="*", default=None, help="Optional list of equivalence thresholds to rerun analysis with")
    parser.add_argument("--family-col", type=str, default="")
    parser.add_argument("--spatial-col", type=str, default="spatial_class")
    parser.add_argument("--observables-only", action="store_true", help="Use only ALP/collagen/mineral for PCA and collapse metrics")
    parser.add_argument("--figures", type=str, nargs="+", default=["all"], choices=("all", "composite", "collapse", "trajectory", "distance"), help="Select figure families to generate")
    parser.add_argument("--no-pca", action="store_true", help="Skip PCA outputs and PCA composite figure")
    parser.add_argument("--export-summary-json", action="store_true", help="Export analysis configuration summary JSON")
    parser.add_argument("--save-tag", type=str, default="", help="Optional suffix appended to output directory")
    parser.add_argument("--dpi", type=int, default=600)
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing per-run outputs")
    parser.add_argument("--dry-run", action="store_true", help="Print run commands only")
    parser.add_argument("--runtime", action="store_true", help="Pass --runtime through to main.py")
    args = parser.parse_args()
    if args.conditions < 1:
        raise SystemExit("--conditions must be >= 1")
    if args.replicates < 1:
        raise SystemExit("--replicates must be >= 1")
    if args.jobs < 1:
        raise SystemExit("--jobs must be >= 1")
    if args.days:
        args.days = sorted({int(day) for day in args.days})
    if args.save_tag:
        args.out = args.out / str(args.save_tag)
    return args


def _normalize_figure_selection(figures: Iterable[str] | None) -> set[str]:
    selected = {str(f).strip().lower() for f in (figures or ["all"])}
    if "all" in selected or not selected:
        return {"composite", "collapse", "trajectory", "distance"}
    return {f for f in selected if f in {"composite", "collapse", "trajectory", "distance"}}


def load_input(path: Path, sheet: str = "") -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Input not found: {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".xls", ".xlsx"}:
        kwargs = {"sheet_name": sheet} if sheet else {}
        return pd.read_excel(path, **kwargs)
    raise SystemExit("Input must be CSV or XLSX")


def resolve_column(df: pd.DataFrame, candidates: Iterable[str], *, required: bool = True) -> str | None:
    lowered = {str(col).lower(): str(col) for col in df.columns}
    for candidate in candidates:
        key = str(candidate).lower()
        if key in lowered:
            return lowered[key]
    if required:
        raise SystemExit(f"Missing required column from candidates: {tuple(candidates)}")
    return None


def read_sobol_problem(path: Path) -> list[dict[str, Any]]:
    fallback = [
        {
            "name": "osteo_induction_hours",
            "flag": "--osteo-induction-hours",
            "bounds": [48.0, 96.0],
            "baseline": 72.0,
            "kind": "int",
            "label": "Osteogenic induction time (h)",
        },
        {
            "name": "commit_delay",
            "flag": "--commit-delay",
            "bounds": [24.0, 72.0],
            "baseline": 48.0,
            "kind": "int",
            "label": "Commitment delay (h)",
        },
        {
            "name": "commit_k",
            "flag": "--commit-k",
            "bounds": [0.0075, 0.0200],
            "baseline": 0.0125,
            "kind": "float",
            "label": "Commitment rate k (h^-1)",
        },
        {
            "name": "move_prob",
            "flag": "--move-prob",
            "bounds": [0.15, 0.40],
            "baseline": 0.27,
            "kind": "float",
            "label": "Random-move probability",
        },
        {
            "name": "min_div_hours",
            "flag": "--min-div-hours",
            "bounds": [30.0, 46.0],
            "baseline": 38.0,
            "kind": "int",
            "label": "Minimum division interval (h)",
        },
        {
            "name": "div_gap_hours",
            "flag": None,
            "bounds": [8.0, 24.0],
            "baseline": 14.0,
            "kind": "int",
            "label": "Division interval gap (h)",
        },
        {
            "name": "apoptosis_prob",
            "flag": "--apoptosis-prob",
            "bounds": [0.0005, 0.0015],
            "baseline": 0.0010,
            "kind": "float",
            "label": "PreOB apoptosis probability",
        },
    ]
    if not path.exists():
        return fallback
    payload = json.loads(path.read_text(encoding="utf-8"))
    params = payload.get("parameters", fallback)
    normalized: list[dict[str, Any]] = []
    for param in params:
        name = str(param["name"])
        bounds = [float(param["bounds"][0]), float(param["bounds"][1])]
        normalized.append(
            {
                "name": name,
                "flag": param.get("flag"),
                "bounds": bounds,
                "baseline": float(param.get("baseline", BASELINE_VALUES.get(name, np.mean(bounds)))),
                "kind": str(param.get("kind", "float")),
                "label": str(param.get("label", name)),
                "source": str(param.get("source", path.name)),
            }
        )
    return normalized


def sobol_samples(n: int, dim: int, seed: int) -> np.ndarray:
    if n <= 0:
        return np.empty((0, dim), dtype=float)
    try:
        sampler = qmc.Sobol(d=dim, scramble=True, seed=seed)
        m = max(1, int(math.ceil(math.log2(max(n, 2)))))
        samples = sampler.random_base2(m)
        if len(samples) < n:
            samples = np.vstack([samples, sampler.random(n - len(samples))])
        return samples[:n]
    except Exception:
        rng = np.random.default_rng(seed)
        return rng.random((n, dim))


def _blend_sample(value: float, lo: float, hi: float, baseline: float, *, focus_prob: float = 0.68, local_width: float = 0.25) -> float:
    if hi <= lo:
        return lo
    span = hi - lo
    local_lo = max(lo, baseline - local_width * span)
    local_hi = min(hi, baseline + local_width * span)
    if value < focus_prob:
        uu = value / max(focus_prob, 1e-9)
        return local_lo + uu * (local_hi - local_lo)
    uu = (value - focus_prob) / max(1.0 - focus_prob, 1e-9)
    return lo + uu * span


def _family_counts(total_conditions: int) -> dict[str, int]:
    budget = max(0, int(total_conditions) - 1)
    families = [TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING]
    if budget == 0:
        return {family: 0 for family in families}
    base = budget // len(families)
    rem = budget % len(families)
    counts = {family: base for family in families}
    for family in families[:rem]:
        counts[family] += 1
    return counts


def build_dynamic_profiles(args: argparse.Namespace, param_specs: list[dict[str, Any]], *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["dynamic_profile", *[spec["name"] for spec in param_specs], "max_div_hours"])
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    while len(rows) < n_profiles and attempts < 24:
        samples = sobol_samples(max(n_profiles * 4, 64), len(param_specs), args.sobol_seed + 101 + attempts)
        for sample in samples:
            rec: dict[str, Any] = {}
            for idx, spec in enumerate(param_specs):
                value = _blend_sample(
                    float(sample[idx]),
                    float(spec["bounds"][0]),
                    float(spec["bounds"][1]),
                    float(spec["baseline"]),
                )
                if spec.get("kind") == "int":
                    rec[spec["name"]] = int(round(value))
                else:
                    rec[spec["name"]] = float(value)
            rec["min_div_hours"] = max(1, int(rec["min_div_hours"]))
            rec["div_gap_hours"] = max(1, int(rec["div_gap_hours"]))
            rec["max_div_hours"] = int(rec["min_div_hours"] + rec["div_gap_hours"])
            key = tuple(
                int(rec[name]) if name in INT_PARAMS else round(float(rec[name]), 8)
                for name in [spec["name"] for spec in param_specs] + ["max_div_hours"]
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(rec)
            if len(rows) >= n_profiles:
                break
        attempts += 1
    if len(rows) < n_profiles:
        rng = np.random.default_rng(args.sobol_seed + 999)
        while len(rows) < n_profiles:
            rec = {}
            for spec in param_specs:
                lo, hi = map(float, spec["bounds"])
                value = float(rng.uniform(lo, hi))
                if spec.get("kind") == "int":
                    rec[spec["name"]] = int(round(value))
                else:
                    rec[spec["name"]] = value
            rec["min_div_hours"] = max(1, int(rec["min_div_hours"]))
            rec["div_gap_hours"] = max(1, int(rec["div_gap_hours"]))
            rec["max_div_hours"] = int(rec["min_div_hours"] + rec["div_gap_hours"])
            rows.append(rec)
    df = pd.DataFrame(rows[:n_profiles])
    df.insert(0, "dynamic_profile", [f"P{i + 1:03d}" for i in range(len(df))])
    return df


def _feasible_grid_spacings(cells: int, lo: int, hi: int) -> list[int]:
    feasible: list[int] = []
    for spacing in range(lo, hi + 1):
        n_sites = math.ceil(GRID_W / spacing) * math.ceil(GRID_H / spacing)
        if n_sites >= cells:
            feasible.append(spacing)
    return feasible


def build_cell_profiles(args: argparse.Namespace, *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["cell_profile", "init_cells", "seed_mode", "grid_spacing"])
    lo, hi = sorted(int(v) for v in args.cells_range)
    g_lo, g_hi = sorted(int(v) for v in args.grid_spacing_range)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    attempts = 0
    while len(rows) < n_profiles and attempts < 24:
        samples = sobol_samples(max(n_profiles * 4, 64), 3, args.sobol_seed + 211 + attempts)
        for u, v, w in samples:
            if u < 0.70:
                local_lo = max(lo, 48)
                local_hi = min(hi, 96)
                uu = u / 0.70
                cells = int(round(local_lo + uu * (local_hi - local_lo)))
            else:
                uu = (u - 0.70) / 0.30
                cells = int(round(lo + uu * (hi - lo)))
            cells = int(np.clip(cells, lo, hi))
            seed_mode = "random" if v < 0.68 else "grid"
            grid_spacing: int | None = None
            if seed_mode == "grid":
                feasible = _feasible_grid_spacings(cells, g_lo, g_hi)
                if not feasible:
                    seed_mode = "random"
                else:
                    index = min(len(feasible) - 1, int(math.floor(float(w) * len(feasible))))
                    grid_spacing = int(feasible[index])
            key = (seed_mode, int(cells), int(grid_spacing or 0))
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "cell_profile": f"N{len(rows) + 1:03d}",
                    "init_cells": int(cells),
                    "seed_mode": seed_mode,
                    "grid_spacing": float(grid_spacing) if grid_spacing is not None else np.nan,
                }
            )
            if len(rows) >= n_profiles:
                break
        attempts += 1
    df = pd.DataFrame(rows[:n_profiles])
    return df


def build_spatial_profiles(args: argparse.Namespace, *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["init_index", "init_slug", "init_title", "layout_seed", "seed_mode"])
    specs = make_condition_specs(args.layout_seed)
    rows: list[dict[str, Any]] = []
    n_specs = len(specs)
    n_repeats = max(1, int(math.ceil(n_profiles / max(1, n_specs))))
    for repeat in range(n_repeats):
        for spec in specs:
            if len(rows) >= n_profiles:
                break
            rows.append(
                {
                    "init_index": int(spec.index),
                    "init_slug": str(spec.slug),
                    "init_title": str(spec.title),
                    "layout_seed": int(spec.layout_seed + repeat * 100_000),
                    "seed_mode": "coords",
                }
            )
        if len(rows) >= n_profiles:
            break
    return pd.DataFrame(rows)


def build_aa_profiles(args: argparse.Namespace, *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["aa_profile", "aa0", "aa_half_life", "collagen_aa_km", "media_interval"])
    aa0_lo, aa0_hi = sorted(float(v) for v in args.aa0_range)
    half_lo, half_hi = sorted(float(v) for v in args.aa_half_life_range)
    km_lo, km_hi = sorted(float(v) for v in args.collagen_aa_km_range)
    med_lo, med_hi = sorted(int(v) for v in args.media_interval_range)
    media_ladder = [v for v in [24, 36, 48, 60, 72] if med_lo <= v <= med_hi]
    if not media_ladder:
        media_ladder = [max(med_lo, min(48, med_hi))]
    rows: list[dict[str, Any]] = []
    seen: set[tuple[float, float, float, int]] = set()
    attempts = 0
    while len(rows) < n_profiles and attempts < 24:
        samples = sobol_samples(max(n_profiles * 4, 64), 4, args.sobol_seed + 401 + attempts)
        for u0, u1, u2, u3 in samples:
            aa0 = float(np.clip(aa0_lo + float(u0) * (aa0_hi - aa0_lo), aa0_lo, aa0_hi))
            aa_half = float(np.clip(half_lo + float(u1) * (half_hi - half_lo), half_lo, half_hi))
            km = float(np.clip(km_lo + float(u2) * (km_hi - km_lo), km_lo, km_hi))
            idx = min(len(media_ladder) - 1, int(math.floor(float(u3) * len(media_ladder))))
            media_interval = int(media_ladder[idx])
            key = (round(aa0, 4), round(aa_half, 4), round(km, 4), media_interval)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "aa_profile": f"A{len(rows) + 1:03d}",
                    "aa0": aa0,
                    "aa_half_life": aa_half,
                    "collagen_aa_km": km,
                    "media_interval": media_interval,
                }
            )
            if len(rows) >= n_profiles:
                break
        attempts += 1
    return pd.DataFrame(rows[:n_profiles])


def build_boundary_profiles(args: argparse.Namespace, *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["boundary_profile", "boundary_mode", "w", "h", "init_cells"])
    s_lo, s_hi = sorted(int(v) for v in args.grid_size_range)
    sizes = [s for s in range(s_lo, s_hi + 1) if s % 2 == 1]
    if not sizes:
        sizes = [BASELINE_VALUES["w"]]
    base_density = BASELINE_VALUES["init_cells"] / float(BASELINE_VALUES["w"] * BASELINE_VALUES["h"])
    cell_lo, cell_hi = sorted(int(v) for v in args.cells_range)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[str, int, int, int]] = set()
    attempts = 0
    while len(rows) < n_profiles and attempts < 24:
        samples = sobol_samples(max(n_profiles * 6, 96), 3, args.sobol_seed + 503 + attempts)
        for u0, u1, u2 in samples:
            mode = "periodic" if float(u0) < 0.5 else "clamp"
            idx_w = min(len(sizes) - 1, int(math.floor(float(u1) * len(sizes))))
            idx_h = min(len(sizes) - 1, int(math.floor(float(u2) * len(sizes))))
            w = int(sizes[idx_w])
            h = int(sizes[idx_h])
            cells = int(round(base_density * w * h))
            cells = int(np.clip(cells, cell_lo, cell_hi))
            key = (mode, w, h, cells)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                {
                    "boundary_profile": f"B{len(rows) + 1:03d}",
                    "boundary_mode": mode,
                    "w": w,
                    "h": h,
                    "init_cells": cells,
                }
            )
            if len(rows) >= n_profiles:
                break
        attempts += 1
    return pd.DataFrame(rows[:n_profiles])


def build_sampling_profiles(args: argparse.Namespace, *, n_profiles: int) -> pd.DataFrame:
    if n_profiles <= 0:
        return pd.DataFrame(columns=["sampling_profile", "sampling_offset_days"])
    off_lo, off_hi = sorted(int(v) for v in args.sampling_offset_days_range)
    allowed = [d for d in range(off_lo, off_hi + 1) if d != 0]
    if not allowed:
        allowed = [0]
    rows: list[dict[str, Any]] = []
    for idx in range(n_profiles):
        offset = int(allowed[idx % len(allowed)])
        rows.append({"sampling_profile": f"T{idx + 1:03d}", "sampling_offset_days": offset})
    return pd.DataFrame(rows)


def build_composite_conditions(args: argparse.Namespace) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    param_specs = read_sobol_problem(args.sobol_problem)
    counts = _family_counts(args.conditions)

    base_row = {
        "condition_type": TYPE_BASELINE,
        "perturbation_family": TYPE_BASELINE,
        "init_index": 1,
        "init_slug": "baseline_random",
        "init_title": "Baseline random",
        "layout_seed": int(args.layout_seed),
        "seed_mode": "random",
        "grid_spacing": float(BASELINE_VALUES["grid_spacing"]),
        "init_cells": int(BASELINE_VALUES["init_cells"]),
        "w": int(BASELINE_VALUES["w"]),
        "h": int(BASELINE_VALUES["h"]),
        "boundary_mode": str(BASELINE_VALUES["boundary_mode"]),
        "media_interval": int(BASELINE_VALUES["media_interval"]),
        "aa0": float(BASELINE_VALUES["aa0"]),
        "aa_half_life": float(BASELINE_VALUES["aa_half_life"]),
        "collagen_aa_km": float(BASELINE_VALUES["collagen_aa_km"]),
        "sampling_offset_days": int(BASELINE_VALUES["sampling_offset_days"]),
        "osteo_induction_hours": int(BASELINE_VALUES["osteo_induction_hours"]),
        "commit_delay": int(BASELINE_VALUES["commit_delay"]),
        "commit_k": float(BASELINE_VALUES["commit_k"]),
        "move_prob": float(BASELINE_VALUES["move_prob"]),
        "min_div_hours": int(BASELINE_VALUES["min_div_hours"]),
        "div_gap_hours": int(BASELINE_VALUES["div_gap_hours"]),
        "max_div_hours": int(BASELINE_VALUES["max_div_hours"]),
        "apoptosis_prob": float(BASELINE_VALUES["apoptosis_prob"]),
        "dynamic_profile": "P000",
        "cell_profile": "N000",
        "aa_profile": "A000",
        "boundary_profile": "B000",
        "sampling_profile": "T000",
    }

    rows = [base_row]

    spatial = build_spatial_profiles(args, n_profiles=counts[TYPE_SPATIAL])
    for rec in spatial.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_SPATIAL
        row["perturbation_family"] = TYPE_SPATIAL
        row["grid_spacing"] = np.nan
        row["init_cells"] = int(BASELINE_VALUES["init_cells"])
        rows.append(row)

    cells = build_cell_profiles(args, n_profiles=counts[TYPE_CELLS])
    for rec in cells.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_CELLS
        row["perturbation_family"] = TYPE_CELLS
        row["init_slug"] = f"{row['seed_mode']}_seed"
        row["init_title"] = f"Cell count profile {rec['cell_profile']}"
        rows.append(row)

    dynamics = build_dynamic_profiles(args, param_specs, n_profiles=counts[TYPE_DYNAMICS])
    for rec in dynamics.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_DYNAMICS
        row["perturbation_family"] = TYPE_DYNAMICS
        row["init_slug"] = "baseline_random"
        row["init_title"] = f"Dynamic profile {rec['dynamic_profile']}"
        rows.append(row)

    aa_profiles = build_aa_profiles(args, n_profiles=counts[TYPE_AA])
    for rec in aa_profiles.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_AA
        row["perturbation_family"] = TYPE_AA
        row["init_slug"] = "baseline_random"
        row["init_title"] = f"Ascorbic acid profile {rec['aa_profile']}"
        rows.append(row)

    boundary_profiles = build_boundary_profiles(args, n_profiles=counts[TYPE_BOUNDARY])
    for rec in boundary_profiles.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_BOUNDARY
        row["perturbation_family"] = TYPE_BOUNDARY
        row["init_slug"] = "boundary_scale"
        row["init_title"] = f"Boundary/scale profile {rec['boundary_profile']}"
        row["seed_mode"] = "random"
        row["grid_spacing"] = np.nan
        rows.append(row)

    sampling_profiles = build_sampling_profiles(args, n_profiles=counts[TYPE_SAMPLING])
    for rec in sampling_profiles.to_dict(orient="records"):
        row = dict(base_row)
        row.update(rec)
        row["condition_type"] = TYPE_SAMPLING
        row["perturbation_family"] = TYPE_SAMPLING
        row["init_slug"] = "sampling_time"
        row["init_title"] = f"Sampling offset profile {rec['sampling_profile']}"
        rows.append(row)

    df = pd.DataFrame(rows[: args.conditions]).copy()
    df.insert(0, "condition_id", [f"C{i:03d}" for i in range(len(df))])
    df["spatial_class"] = df.apply(spatial_class_from_seeding, axis=1)
    return df, param_specs


def write_seed_coords_csv(row: pd.Series, seed_dir: Path, layout_specs: dict[str, Any]) -> str:
    if str(row.get("seed_mode", "random")) != "coords":
        return ""
    init_slug = str(row["init_slug"])
    spec = layout_specs.get(init_slug)
    if spec is None:
        raise KeyError(f"Missing layout spec for {init_slug}")
    coords = generate_layout(spec, GRID_W, GRID_H, int(row["init_cells"]), int(row["layout_seed"]))
    seed_dir.mkdir(parents=True, exist_ok=True)
    out_path = seed_dir / f"{row['condition_id']}_seed_coords.csv"
    pd.DataFrame(coords, columns=["x", "y"]).to_csv(out_path, index=False)
    return str(out_path)


def build_command(args: argparse.Namespace, row: pd.Series, seed: int, run_dir: Path) -> list[str]:
    cmd = [args.python, *BASE_ARGS, "--seed", str(int(seed)), "--out", str(run_dir)]
    cmd.extend(["--w", str(int(row["w"]))])
    cmd.extend(["--h", str(int(row["h"]))])
    cmd.extend(["--boundary-mode", str(row["boundary_mode"])])
    cmd.extend(["--media-interval", str(int(row["media_interval"]))])
    cmd.extend(["--aa0", f"{float(row['aa0']):.10g}"])
    cmd.extend(["--aa-half-life", f"{float(row['aa_half_life']):.10g}"])
    cmd.extend(["--collagen-aa-km", f"{float(row['collagen_aa_km']):.10g}"])
    cmd.extend(["--cells", str(int(row["init_cells"]))])
    seed_coords = str(row.get("seed_coords_csv", "") or "").strip()
    if seed_coords:
        cmd.extend(["--seed-coords", seed_coords])
    else:
        seed_mode = str(row.get("seed_mode", "random")).strip().lower()
        if seed_mode not in {"random", "grid"}:
            seed_mode = "random"
        cmd.extend(["--seed-mode", seed_mode])
        if seed_mode == "grid" and "grid_spacing" in row.index and pd.notna(row["grid_spacing"]):
            cmd.extend(["--grid-spacing", str(max(1, int(round(float(row["grid_spacing"])))) )])
    for name in [
        "osteo_induction_hours",
        "commit_delay",
        "commit_k",
        "move_prob",
        "min_div_hours",
        "max_div_hours",
        "apoptosis_prob",
    ]:
        flag = PARAM_TO_FLAG[name]
        value = row[name]
        if name in INT_PARAMS:
            cmd.extend([flag, str(int(round(float(value))))])
        else:
            cmd.extend([flag, f"{float(value):.10g}"])
    if args.runtime:
        cmd.append("--runtime")
    return cmd


def extract_day_rows(timeseries_path: Path, days: Iterable[int], *, sampling_offset_days: int = 0) -> pd.DataFrame:
    df = pd.read_csv(timeseries_path)
    if "day" not in df.columns:
        raise ValueError(f"Missing 'day' column in {timeseries_path}")
    wanted = {int(day) for day in days}
    df["day"] = pd.to_numeric(df["day"], errors="coerce").round().astype("Int64")
    available_days = sorted(int(v) for v in df["day"].dropna().astype(int).unique().tolist())
    if not available_days:
        raise ValueError(f"No valid day rows in {timeseries_path}")
    min_day = int(available_days[0])
    max_day = int(available_days[-1])
    rows: list[pd.Series] = []
    missing: list[int] = []
    for target_day in sorted(wanted):
        sampled_day = int(target_day + int(sampling_offset_days))
        # Sampling-time perturbations can request days outside simulated bounds
        # (e.g., day 28 + offset +1 => day 29 when simulation ends at day 28).
        # Clamp to the available timeseries window so we can reuse completed runs.
        if sampled_day < min_day:
            sampled_day = min_day
        elif sampled_day > max_day:
            sampled_day = max_day
        hit = df.loc[df["day"].astype("Int64") == sampled_day]
        if hit.empty:
            missing.append(target_day)
            continue
        row = hit.iloc[0].copy()
        row["sampled_day"] = sampled_day
        row["day"] = int(target_day)
        rows.append(row)
    if missing:
        raise ValueError(f"Missing day rows {missing} in {timeseries_path}")
    return pd.DataFrame(rows)


def classify_perturbation_family(row: pd.Series) -> str:
    value = str(row.get("perturbation_family", row.get("condition_type", "unknown")))
    if value and value != "nan":
        return value
    return "unknown"


def spatial_class_from_seeding(row: pd.Series) -> str:
    condition_type = str(row.get("condition_type", ""))
    if condition_type == TYPE_SPATIAL:
        return str(row.get("init_slug", "structured"))
    seed_mode = str(row.get("seed_mode", "random")).lower()
    if seed_mode == "grid":
        return "grid_seed"
    if seed_mode == "coords":
        return str(row.get("init_slug", "coords"))
    return "random_seed"


def run_one_condition_replicate(
    args: argparse.Namespace,
    row: pd.Series,
    replicate_index: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    condition_id = str(row["condition_id"])
    seed = int(secrets.randbelow(2_147_483_646) + 1)
    run_dir = args.out / "runs" / condition_id / f"r{replicate_index:02d}"
    timeseries_path = run_dir / "timeseries.csv"
    cmd = build_command(args, row, seed, run_dir)
    command_str = shlex.join(cmd)

    manifest_row = {
        "condition_id": condition_id,
        "condition_type": str(row["condition_type"]),
        "replicate": int(replicate_index),
        "seed": seed,
        "run_dir": str(run_dir),
        "success": False,
        "returncode": np.nan,
        "command": command_str,
        "error": "",
    }

    if args.dry_run:
        print(command_str)
        manifest_row["success"] = True
        manifest_row["returncode"] = 0
        return manifest_row, []

    if timeseries_path.exists() and not args.overwrite:
        day_df = extract_day_rows(timeseries_path, args.days, sampling_offset_days=int(row.get("sampling_offset_days", 0)))
        manifest_row["success"] = True
        manifest_row["returncode"] = 0
        return manifest_row, _rows_from_day_df(row, day_df, replicate_index, seed, run_dir)

    run_dir.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    manifest_row["returncode"] = int(completed.returncode)
    manifest_row["success"] = completed.returncode == 0
    if completed.returncode != 0:
        manifest_row["error"] = f"exit {completed.returncode}"
        return manifest_row, []
    if not timeseries_path.exists():
        manifest_row["success"] = False
        manifest_row["error"] = f"missing expected output: {timeseries_path}"
        return manifest_row, []
    day_df = extract_day_rows(timeseries_path, args.days, sampling_offset_days=int(row.get("sampling_offset_days", 0)))
    return manifest_row, _rows_from_day_df(row, day_df, replicate_index, seed, run_dir)


def _rows_from_day_df(
    row: pd.Series,
    day_df: pd.DataFrame,
    replicate_index: int,
    seed: int,
    run_dir: Path,
) -> list[dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    for _, series in day_df.iterrows():
        n_total = float(series.get("n_total", 0.0))
        n_osteoblast = float(series.get("n_osteoblast", 0.0))
        nodule_count = float(series.get("nodule_count_current", series.get("nodule_count_ever", 0.0)))
        out_rows.append(
            {
                "condition_id": str(row["condition_id"]),
                "condition_type": str(row["condition_type"]),
                "perturbation_family": str(row["perturbation_family"]),
                "spatial_class": str(row["spatial_class"]),
                "replicate": int(replicate_index),
                "seed": int(seed),
                "run_dir": str(run_dir),
                "day": int(series["day"]),
                "sampled_day": int(series.get("sampled_day", series["day"])),
                "ALP": float(series.get("alp_total", np.nan)),
                "collagen": float(series.get("collagen_total", np.nan)),
                "mineral": float(series.get("ha_total", np.nan)),
                "n_total": n_total,
                "n_preob": float(series.get("n_preob", np.nan)),
                "n_osteoblast": n_osteoblast,
                "OB_fraction": float(n_osteoblast / n_total) if n_total > 0 else 0.0,
                "nodule_count_current": nodule_count,
                "clustering": float(nodule_count / n_total) if n_total > 0 else 0.0,
                "msd_effective": float(series.get("msd_effective", np.nan)),
                "init_cells": int(row["init_cells"]),
                "seed_mode": str(row["seed_mode"]),
                "grid_spacing": float(row["grid_spacing"]) if pd.notna(row["grid_spacing"]) else np.nan,
                "init_slug": str(row["init_slug"]),
                "init_title": str(row["init_title"]),
                "layout_seed": int(row["layout_seed"]),
                "dynamic_profile": str(row["dynamic_profile"]),
                "cell_profile": str(row["cell_profile"]),
                "aa_profile": str(row["aa_profile"]),
                "boundary_profile": str(row["boundary_profile"]),
                "sampling_profile": str(row["sampling_profile"]),
                "w": int(row["w"]),
                "h": int(row["h"]),
                "boundary_mode": str(row["boundary_mode"]),
                "aa0": float(row["aa0"]),
                "aa_half_life": float(row["aa_half_life"]),
                "collagen_aa_km": float(row["collagen_aa_km"]),
                "media_interval": int(row["media_interval"]),
                "sampling_offset_days": int(row["sampling_offset_days"]),
                "osteo_induction_hours": int(row["osteo_induction_hours"]),
                "commit_delay": int(row["commit_delay"]),
                "commit_k": float(row["commit_k"]),
                "move_prob": float(row["move_prob"]),
                "min_div_hours": int(row["min_div_hours"]),
                "max_div_hours": int(row["max_div_hours"]),
                "apoptosis_prob": float(row["apoptosis_prob"]),
            }
        )
    return out_rows


def simulate_conditions(args: argparse.Namespace) -> pd.DataFrame:
    args.out.mkdir(parents=True, exist_ok=True)
    conditions_df, param_specs = build_composite_conditions(args)
    layout_specs = {spec.slug: spec for spec in make_condition_specs(args.layout_seed)}
    seed_dir = args.out / "seed_coords"
    conditions_df["seed_coords_csv"] = conditions_df.apply(
        lambda row: write_seed_coords_csv(row, seed_dir, layout_specs),
        axis=1,
    )

    safe_to_csv(conditions_df, args.out / "composite_conditions.csv")
    (args.out / "sobol_problem_resolved.json").write_text(
        json.dumps({"parameters": [{k: _jsonable(v) for k, v in spec.items()} for spec in param_specs]}, indent=2),
        encoding="utf-8",
    )

    manifest_rows: list[dict[str, Any]] = []
    long_rows: list[dict[str, Any]] = []
    if args.replicates < 2:
        warnings.warn("Statistical comparisons are weaker with --replicates < 2.", stacklevel=2)

    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = []
        for _, row in conditions_df.iterrows():
            for replicate_index in range(1, args.replicates + 1):
                futures.append(pool.submit(run_one_condition_replicate, args, row, replicate_index))
        for future in as_completed(futures):
            manifest_row, extracted = future.result()
            manifest_rows.append(manifest_row)
            long_rows.extend(extracted)

    manifest_df = pd.DataFrame(manifest_rows).sort_values(["condition_id", "replicate"]).reset_index(drop=True)
    safe_to_csv(manifest_df, args.out / "run_manifest.csv")

    long_df = pd.DataFrame(long_rows)
    if not long_df.empty:
        long_df = long_df.sort_values(["condition_id", "replicate", "day"]).reset_index(drop=True)
        safe_to_csv(long_df, args.out / "simulated_observables_long.csv")

    metadata = {
        "conditions": int(args.conditions),
        "replicates": int(args.replicates),
        "jobs": int(args.jobs),
        "days": [int(day) for day in args.days],
        "seed_policy": str(args.seed_policy),
        "layout_seed": int(args.layout_seed),
        "cells_range": [int(v) for v in args.cells_range],
        "aa0_range": [float(v) for v in args.aa0_range],
        "aa_half_life_range": [float(v) for v in args.aa_half_life_range],
        "collagen_aa_km_range": [float(v) for v in args.collagen_aa_km_range],
        "media_interval_range": [int(v) for v in args.media_interval_range],
        "grid_spacing_range": [int(v) for v in args.grid_spacing_range],
        "grid_size_range": [int(v) for v in args.grid_size_range],
        "sampling_offset_days_range": [int(v) for v in args.sampling_offset_days_range],
        "family_counts": _family_counts(args.conditions),
        "baseline_values": BASELINE_VALUES,
        "notes": (
            "Scientific baseline mirrors the main.py baseline command supplied by the user, "
            "while batch-only exports are minimized (no snapshots, no XLSX) to keep the 300-condition screen tractable."
        ),
    }
    (args.out / "sanity10_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return long_df


def infer_family(df: pd.DataFrame, preferred_col: str = "") -> pd.Series:
    if preferred_col and preferred_col in df.columns:
        return df[preferred_col].astype(str)
    for col in ("perturbation_family", "condition_type"):
        if col in df.columns:
            return df[col].astype(str)
    return pd.Series(["unknown"] * len(df), index=df.index, dtype="object")


def standardize(df: pd.DataFrame, columns: list[str], *, by: str | None = None) -> tuple[pd.DataFrame, dict[Any, ScalingParams]]:
    out = df.copy()
    scalers: dict[Any, ScalingParams] = {}
    if by is None:
        arr = out[columns].to_numpy(dtype=float)
        center = np.nanmean(arr, axis=0)
        scale = np.nanstd(arr, axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        out[[f"z_{col}" for col in columns]] = (arr - center) / scale
        scalers["global"] = ScalingParams(center=center, scale=scale, columns=list(columns))
        return out, scalers
    for key, group in out.groupby(by, observed=True):
        arr = group[columns].to_numpy(dtype=float)
        center = np.nanmean(arr, axis=0)
        scale = np.nanstd(arr, axis=0, ddof=0)
        scale = np.where(scale < 1e-12, 1.0, scale)
        z = (arr - center) / scale
        out.loc[group.index, [f"z_{col}" for col in columns]] = z
        scalers[key] = ScalingParams(center=center, scale=scale, columns=list(columns))
    return out, scalers


def fit_pca(df: pd.DataFrame, columns: list[str]) -> PcaModel:
    matrix = df[columns].to_numpy(dtype=float)
    mean = np.nanmean(matrix, axis=0)
    centered = np.nan_to_num(matrix - mean, copy=False)
    if centered.shape[0] < 2:
        components = np.eye(len(columns), dtype=float)[:2]
        evr = np.zeros(2, dtype=float)
        return PcaModel(mean=mean, components=components, explained_variance_ratio=evr, columns=list(columns))
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    components = vt[:2]
    denom = max(1, centered.shape[0] - 1)
    explained = (singular_values ** 2) / denom
    total = float(np.sum(explained))
    if total <= 0:
        evr = np.zeros(min(2, len(explained)), dtype=float)
    else:
        evr = explained[:2] / total
    return PcaModel(mean=mean, components=components, explained_variance_ratio=np.asarray(evr, dtype=float), columns=list(columns))


def transform_pca(model: PcaModel, df: pd.DataFrame) -> np.ndarray:
    matrix = df[model.columns].to_numpy(dtype=float)
    centered = np.nan_to_num(matrix - model.mean, copy=False)
    return centered @ model.components.T


def add_confidence_ellipse(ax: plt.Axes, x: np.ndarray, y: np.ndarray, color: str) -> None:
    if len(x) < 3:
        return
    cov = np.cov(np.vstack([x, y]))
    if not np.all(np.isfinite(cov)):
        return
    vals, vecs = np.linalg.eigh(cov)
    vals = np.maximum(vals, 1e-12)
    order = vals.argsort()[::-1]
    vals = vals[order]
    vecs = vecs[:, order]
    angle = math.degrees(math.atan2(vecs[1, 0], vecs[0, 0]))
    scale = math.sqrt(chi2.ppf(0.95, 2))
    width, height = 2.0 * scale * np.sqrt(vals)
    ellipse = Ellipse(
        xy=(float(np.mean(x)), float(np.mean(y))),
        width=float(width),
        height=float(height),
        angle=float(angle),
        facecolor="none",
        edgecolor=color,
        linewidth=1.0,
        alpha=0.45,
        zorder=2,
    )
    ax.add_patch(ellipse)


def bh_adjust(pvalues: pd.Series) -> pd.Series:
    mask = pvalues.notna()
    out = pd.Series(np.nan, index=pvalues.index, dtype=float)
    if mask.sum() == 0:
        return out
    vals = pvalues.loc[mask].astype(float)
    order = np.argsort(vals.to_numpy())
    ranked = vals.iloc[order]
    n = len(ranked)
    adjusted = np.empty(n, dtype=float)
    running = 1.0
    for idx in range(n - 1, -1, -1):
        rank = idx + 1
        value = float(ranked.iloc[idx]) * n / rank
        running = min(running, value)
        adjusted[idx] = running
    restored = pd.Series(adjusted, index=ranked.index)
    out.loc[mask] = restored.loc[vals.index].clip(upper=1.0)
    return out


def welch_pvalue(sample_a: np.ndarray, sample_b: np.ndarray) -> float:
    x = np.asarray(sample_a, dtype=float)
    y = np.asarray(sample_b, dtype=float)
    x = x[np.isfinite(x)]
    y = y[np.isfinite(y)]
    if len(x) == 0 or len(y) == 0:
        return float("nan")
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    mean_x = float(np.mean(x))
    mean_y = float(np.mean(y))
    var_x = float(np.var(x, ddof=1))
    var_y = float(np.var(y, ddof=1))
    denom = math.sqrt(max(var_x / len(x) + var_y / len(y), 0.0))
    if denom <= 1e-12:
        return 1.0 if abs(mean_x - mean_y) <= 1e-12 else 0.0
    t_stat = (mean_x - mean_y) / denom
    numerator = (var_x / len(x) + var_y / len(y)) ** 2
    denominator = 0.0
    if len(x) > 1 and var_x > 0:
        denominator += (var_x ** 2) / ((len(x) ** 2) * (len(x) - 1))
    if len(y) > 1 and var_y > 0:
        denominator += (var_y ** 2) / ((len(y) ** 2) * (len(y) - 1))
    if denominator <= 0.0:
        return float("nan")
    dof = numerator / denominator
    return float(2.0 * student_t.sf(abs(t_stat), dof))


def compute_day_metrics(day_df: pd.DataFrame, threshold: float, z_cols: list[str], latent_cols: list[str]) -> DayMetrics:
    day = int(day_df["day"].iloc[0])
    centroids = day_df.groupby("perturbation_family", observed=True)[z_cols].mean()
    labels = centroids.index.tolist()
    if len(labels) < 2:
        return DayMetrics(day=day, resolution_index=np.nan, collapsed_fraction=np.nan, collapsed_pairs=0, total_pairs=0, latent_distance=np.nan)
    latent_centroids = None
    if latent_cols:
        latent_centroids = day_df.groupby("perturbation_family", observed=True)[latent_cols].mean()
        latent_arr = latent_centroids.to_numpy(dtype=float)
        mu = np.nanmean(latent_arr, axis=0)
        sd = np.nanstd(latent_arr, axis=0, ddof=0)
        sd = np.where(sd < 1e-12, 1.0, sd)
        latent_centroids = pd.DataFrame((latent_arr - mu) / sd, index=latent_centroids.index, columns=latent_centroids.columns)
    distances: list[float] = []
    latent_distances: list[float] = []
    collapsed = 0
    total_pairs = 0
    for left, right in combinations(range(len(labels)), 2):
        total_pairs += 1
        d = float(np.linalg.norm(centroids.iloc[left].to_numpy(dtype=float) - centroids.iloc[right].to_numpy(dtype=float)))
        distances.append(d)
        if d <= threshold:
            collapsed += 1
        if latent_centroids is not None:
            latent_distances.append(
                float(
                    np.linalg.norm(
                        latent_centroids.iloc[left].to_numpy(dtype=float)
                        - latent_centroids.iloc[right].to_numpy(dtype=float)
                    )
                )
            )
    return DayMetrics(
        day=day,
        resolution_index=float(np.mean(distances)) if distances else np.nan,
        collapsed_fraction=float(collapsed / total_pairs) if total_pairs else np.nan,
        collapsed_pairs=int(collapsed),
        total_pairs=int(total_pairs),
        latent_distance=float(np.mean(latent_distances)) if latent_distances else np.nan,
    )


def _baseline_condition_id(df: pd.DataFrame) -> str:
    if "condition_id" in df.columns and (df["condition_id"].astype(str) == "C000").any():
        return "C000"
    mask = pd.Series(True, index=df.index)
    if "init_cells" in df.columns:
        mask &= pd.to_numeric(df["init_cells"], errors="coerce").fillna(BASELINE_VALUES["init_cells"]).astype(int) == BASELINE_VALUES["init_cells"]
    if "seed_mode" in df.columns:
        mask &= df["seed_mode"].astype(str).str.lower().eq("random")
    baseline_rows = df.loc[mask]
    if not baseline_rows.empty:
        return str(baseline_rows["condition_id"].iloc[0])
    return str(df["condition_id"].iloc[0])


def _endpoint_statistics(raw_full: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    baseline_id = _baseline_condition_id(raw_full)
    meta_cols = [
        col
        for col in [
            "condition_id",
            "condition_type",
            "perturbation_family",
            "spatial_class",
            "init_cells",
            "seed_mode",
            "grid_spacing",
            "init_slug",
            "dynamic_profile",
            "cell_profile",
            "aa_profile",
            "boundary_profile",
            "sampling_profile",
            "w",
            "h",
            "boundary_mode",
            "aa0",
            "aa_half_life",
            "collagen_aa_km",
            "media_interval",
            "sampling_offset_days",
            "sampled_day",
        ]
        if col in raw_full.columns
    ]
    endpoint_rows: list[dict[str, Any]] = []
    endpoint_vectors: list[dict[str, Any]] = []

    for spec in ENDPOINT_SPECS:
        day = int(spec["day"])
        feature = str(spec["feature"])
        subset = raw_full.loc[raw_full["day"].astype(int) == day].copy()
        if subset.empty or feature not in subset.columns:
            continue
        baseline_values = subset.loc[subset["condition_id"].astype(str) == baseline_id, feature].astype(float).to_numpy()
        baseline_mean = float(np.mean(baseline_values)) if len(baseline_values) else np.nan
        baseline_sd = float(np.std(baseline_values, ddof=1)) if len(baseline_values) > 1 else np.nan
        group_map = {str(cid): grp for cid, grp in subset.groupby("condition_id", observed=True)}
        raw_pvals: list[tuple[str, float]] = []
        for condition_id, group in group_map.items():
            values = group[feature].astype(float).to_numpy()
            row: dict[str, Any] = {"endpoint_day": day, "endpoint_feature": feature}
            for col in meta_cols:
                row[col] = group[col].iloc[0]
            row["mean"] = float(np.mean(values))
            row["sd"] = float(np.std(values, ddof=1)) if len(values) > 1 else np.nan
            row["baseline_mean"] = baseline_mean
            row["baseline_sd"] = baseline_sd
            denom = max(abs(baseline_mean), 1e-9)
            row["relative_change_pct"] = float(100.0 * (row["mean"] - baseline_mean) / denom) if np.isfinite(baseline_mean) else np.nan
            pvalue = np.nan if condition_id == baseline_id else welch_pvalue(values, baseline_values)
            row["pvalue"] = pvalue
            # Ensure this column always exists even when BH adjustment is skipped
            # (e.g., all p-values are NaN for an endpoint/day slice).
            row["qvalue"] = np.nan
            endpoint_rows.append(row)
            if condition_id != baseline_id and np.isfinite(pvalue):
                raw_pvals.append((condition_id, pvalue))
        if raw_pvals:
            q_lookup = bh_adjust(pd.Series({cid: pvalue for cid, pvalue in raw_pvals}))
            for row in endpoint_rows:
                if row["endpoint_day"] == day and row["endpoint_feature"] == feature and row["condition_id"] != baseline_id:
                    row["qvalue"] = float(q_lookup.get(row["condition_id"], np.nan))
                elif row["endpoint_day"] == day and row["endpoint_feature"] == feature:
                    row["qvalue"] = np.nan
        for condition_id, group in group_map.items():
            endpoint_vectors.append(
                {
                    "condition_id": condition_id,
                    "endpoint_feature": feature,
                    "value": float(group[feature].astype(float).mean()),
                }
            )

    stats_long = pd.DataFrame(endpoint_rows)
    if stats_long.empty:
        return stats_long, pd.DataFrame()
    stats_wide = stats_long.pivot(index="condition_id", columns="endpoint_feature", values=["mean", "sd", "relative_change_pct", "pvalue", "qvalue"])
    stats_wide.columns = [f"{feature}_{metric}" for metric, feature in stats_wide.columns]
    stats_wide = stats_wide.reset_index()
    meta = stats_long[meta_cols].drop_duplicates(subset=["condition_id"]).copy()
    stats_wide = meta.merge(stats_wide, on="condition_id", how="left")

    endpoint_matrix = pd.DataFrame(endpoint_vectors).pivot(index="condition_id", columns="endpoint_feature", values="value").reset_index()
    endpoint_matrix = meta.merge(endpoint_matrix, on="condition_id", how="left")
    for feature in FEATURES:
        if feature not in endpoint_matrix.columns:
            endpoint_matrix[feature] = np.nan
    arr = endpoint_matrix[list(FEATURES)].to_numpy(dtype=float)
    center = np.nanmean(arr, axis=0)
    scale = np.nanstd(arr, axis=0, ddof=0)
    scale = np.where(scale < 1e-12, 1.0, scale)
    z = (arr - center) / scale
    endpoint_matrix[[f"z_{feature}" for feature in FEATURES]] = z
    base_row = endpoint_matrix.loc[endpoint_matrix["condition_id"].astype(str) == baseline_id]
    if not base_row.empty:
        base_vec = base_row[[f"z_{feature}" for feature in FEATURES]].iloc[0].to_numpy(dtype=float)
        endpoint_matrix["endpoint_distance"] = np.linalg.norm(z - base_vec[None, :], axis=1)
    else:
        endpoint_matrix["endpoint_distance"] = np.nan

    stats_wide = stats_wide.merge(endpoint_matrix[["condition_id", *FEATURES, "endpoint_distance"]], on="condition_id", how="left")
    for feature in FEATURES:
        qcol = f"{feature}_qvalue"
        if qcol not in stats_wide.columns:
            stats_wide[qcol] = np.nan
    stats_wide["all_endpoints_nonsignificant"] = (
        stats_wide[[f"{feature}_qvalue" for feature in FEATURES]].fillna(np.inf) > 0.05
    ).all(axis=1)
    stats_wide["mineral_nonsignificant"] = stats_wide["mineral_qvalue"].fillna(np.inf) > 0.05

    candidates = stats_wide.loc[stats_wide["condition_id"].astype(str) != baseline_id].copy()
    if "mineral_relative_change_pct" in candidates.columns:
        candidates["abs_mineral_change_pct"] = candidates["mineral_relative_change_pct"].abs()
        candidates = candidates.sort_values(
            ["mineral_nonsignificant", "all_endpoints_nonsignificant", "abs_mineral_change_pct", "endpoint_distance"],
            ascending=[False, False, True, True],
        )
    return stats_wide, candidates


def make_plot(
    merged: pd.DataFrame,
    metrics: pd.DataFrame,
    loadings: pd.DataFrame,
    *,
    out_dir: Path,
    dpi: int,
) -> None:
    days = sorted(int(day) for day in merged["day"].unique())
    fig, axes = plt.subplots(1, len(days), figsize=(5.0 * len(days), 5.0), constrained_layout=False)
    if len(days) == 1:
        axes = np.asarray([axes])
    fig.subplots_adjust(left=0.06, right=0.99, top=0.88, bottom=0.14, wspace=0.18)

    legend_handles = []
    for family in [TYPE_BASELINE, TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING]:
        legend_handles.append(
            plt.Line2D(
                [0],
                [0],
                marker=MARKERS[family],
                color="none",
                markerfacecolor=COLORS[family],
                markeredgecolor=COLORS[family],
                markersize=6.5,
                linestyle="none",
                label=FAMILY_LABELS[family],
            )
        )
    legend_handles.append(
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="black",
            markerfacecolor="none",
            markeredgecolor="black",
            markersize=7.0,
            linestyle="none",
            label="Mineral q >= 0.05",
        )
    )

    for ax, day in zip(axes, days):
        day_df = merged.loc[merged["day"].astype(int) == int(day)].copy()
        metric_row = metrics.loc[metrics["day"].astype(int) == int(day)]
        metric = metric_row.iloc[0] if not metric_row.empty else None
        load_day = loadings.loc[loadings["day"].astype(int) == int(day)].copy()
        for family in [TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING, TYPE_BASELINE]:
            sub = day_df.loc[day_df["perturbation_family"].astype(str) == family]
            if sub.empty:
                continue
            x = sub["pc1"].to_numpy(dtype=float)
            y = sub["pc2"].to_numpy(dtype=float)
            add_confidence_ellipse(ax, x, y, COLORS[family])
            ax.scatter(
                x,
                y,
                s=38 if family != TYPE_BASELINE else 120,
                marker=MARKERS[family],
                c=COLORS[family],
                edgecolors=COLORS[family],
                linewidths=0.4,
                alpha=0.85,
                zorder=3 if family != TYPE_BASELINE else 4,
            )
        nonsig = day_df.loc[day_df["mineral_nonsignificant"].fillna(False).astype(bool)]
        if not nonsig.empty:
            ax.scatter(
                nonsig["pc1"].to_numpy(dtype=float),
                nonsig["pc2"].to_numpy(dtype=float),
                s=74,
                facecolors="none",
                edgecolors="black",
                linewidths=0.8,
                zorder=5,
            )
        if not load_day.empty:
            span = max(
                float(np.nanmax(np.abs(day_df["pc1"].to_numpy(dtype=float)))),
                float(np.nanmax(np.abs(day_df["pc2"].to_numpy(dtype=float)))),
                1.0,
            )
            arrow_scale = 0.38 * span
            for _, load in load_day.iterrows():
                feature = str(load["feature"])
                if feature.startswith("z_"):
                    feature = feature[2:]
                if feature not in FEATURES:
                    continue
                dx = float(load["pc1_loading"]) * arrow_scale
                dy = float(load["pc2_loading"]) * arrow_scale
                ax.annotate(
                    "",
                    xy=(dx, dy),
                    xytext=(0.0, 0.0),
                    arrowprops=dict(arrowstyle="-|>", color=COLORS["muted"], lw=1.0),
                    zorder=1,
                )
                ax.text(dx * 1.10, dy * 1.10, feature, fontsize=8.0, color=COLORS["muted"], ha="center", va="center")
        ax.axhline(0.0, color=COLORS["grid"], linewidth=0.8, zorder=0)
        ax.axvline(0.0, color=COLORS["grid"], linewidth=0.8, zorder=0)
        ax.grid(color=COLORS["grid"], linewidth=0.7, alpha=0.45)
        subtitle = ""
        if metric is not None:
            subtitle = (
                f"RI={float(metric['resolution_index']):.2f}, "
                f"collapsed={float(metric['collapsed_fraction']):.2f}"
            )
        ax.set_title(f"Day {day}\n{subtitle}".strip(), fontsize=12.5)
        evr1 = float(day_df["pc1_var"].iloc[0]) if "pc1_var" in day_df.columns else np.nan
        evr2 = float(day_df["pc2_var"].iloc[0]) if "pc2_var" in day_df.columns else np.nan
        ax.set_xlabel(f"PC1 ({100.0 * evr1:.1f}%)" if np.isfinite(evr1) else "PC1")
        ax.set_ylabel(f"PC2 ({100.0 * evr2:.1f}%)" if np.isfinite(evr2) else "PC2")
        ax.tick_params(direction="out", length=4.0, width=0.8)

    fig.legend(handles=legend_handles, loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.995))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sanity10_pca_collapse_composite.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_pca_collapse_composite.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_pca_collapse_composite.svg", bbox_inches="tight")
    plt.close(fig)


def make_endpoint_separate_plot(
    raw_full: pd.DataFrame,
    stats_wide: pd.DataFrame,
    *,
    out_dir: Path,
    dpi: int,
    threshold: float,
) -> None:
    _ = stats_wide  # kept for call compatibility.
    if raw_full.empty:
        return

    base_id = _baseline_condition_id(raw_full)
    day_order = [10, 14, 28]
    family_order = [TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING, TYPE_BASELINE]
    family_ticks = ["Spatial", "Cell #", "Dynamics", "AA", "Boundary", "Sampling", "Baseline"]

    cond_day = raw_full.groupby(["condition_id", "day"], observed=True).agg(
        perturbation_family=("perturbation_family", "first"),
        ALP=("ALP", "mean"),
        collagen=("collagen", "mean"),
        mineral=("mineral", "mean"),
    ).reset_index()
    cond_day = cond_day.loc[cond_day["day"].isin(day_order)].copy()
    if cond_day.empty:
        return

    # Per-condition endpoint shape normalization to remove multiplicative scale effects.
    endpoint_values = cond_day[list(FEATURES)].to_numpy(dtype=float)
    row_min = np.nanmin(endpoint_values, axis=1, keepdims=True)
    row_max = np.nanmax(endpoint_values, axis=1, keepdims=True)
    row_span = np.where((row_max - row_min) < 1e-12, 1.0, (row_max - row_min))
    norm_vals = (endpoint_values - row_min) / row_span
    cond_day[["mm_ALP", "mm_collagen", "mm_mineral"]] = norm_vals

    baseline_vectors = cond_day.loc[
        cond_day["condition_id"].astype(str) == str(base_id),
        ["day", "mm_ALP", "mm_collagen", "mm_mineral"],
    ].copy()
    if baseline_vectors.empty:
        return
    cond_day = cond_day.merge(baseline_vectors, on="day", how="left", suffixes=("", "_base"))
    cond_day["distance_to_baseline"] = np.sqrt(
        (cond_day["mm_ALP"] - cond_day["mm_ALP_base"]) ** 2
        + (cond_day["mm_collagen"] - cond_day["mm_collagen_base"]) ** 2
        + (cond_day["mm_mineral"] - cond_day["mm_mineral_base"]) ** 2
    )

    global_y = cond_day["distance_to_baseline"].to_numpy(dtype=float)
    ymax = float(np.nanmax(global_y)) if len(global_y) else 1.0
    ymax = max(ymax * 1.18, threshold * 1.35, 0.25)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 5.2), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.84, bottom=0.18, wspace=0.16)

    for panel_idx, (ax, day) in enumerate(zip(axes, day_order)):
        day_df = cond_day.loc[cond_day["day"].astype(int) == day].copy()
        if day_df.empty:
            ax.set_visible(False)
            continue

        # Visual equivalent zone.
        ax.axhspan(0.0, threshold, color="#ecfdf3", zorder=0)
        ax.axhline(threshold, color="#b91c1c", linestyle="--", linewidth=1.05, zorder=1)

        rng = np.random.default_rng(20_000 + day)
        collapsed_by_family: list[float] = []
        n_points = 0
        for ix, family in enumerate(family_order):
            fam = day_df.loc[day_df["perturbation_family"].astype(str) == family]
            if fam.empty:
                collapsed_by_family.append(np.nan)
                continue
            y = fam["distance_to_baseline"].to_numpy(dtype=float)
            n_points += len(y)
            x = ix + rng.normal(0.0, 0.065, size=len(y))

            # Draw compact box (IQR + whiskers) behind jitter for readable family summaries.
            q1, q2, q3 = np.quantile(y, [0.25, 0.50, 0.75])
            iqr = q3 - q1
            whisker_low = max(float(np.min(y)), float(q1 - 1.5 * iqr))
            whisker_high = min(float(np.max(y)), float(q3 + 1.5 * iqr))
            ax.add_patch(
                Rectangle(
                    (ix - 0.22, q1),
                    0.44,
                    max(q3 - q1, 1e-9),
                    facecolor=COLORS[family],
                    edgecolor=COLORS[family],
                    alpha=0.16,
                    linewidth=1.0,
                    zorder=2,
                )
            )
            ax.plot([ix, ix], [whisker_low, q1], color=COLORS[family], linewidth=1.0, zorder=2)
            ax.plot([ix, ix], [q3, whisker_high], color=COLORS[family], linewidth=1.0, zorder=2)
            ax.plot([ix - 0.14, ix + 0.14], [q2, q2], color=COLORS[family], linewidth=1.5, zorder=3)

            ax.scatter(
                x,
                y,
                s=34 if family != TYPE_BASELINE else 84,
                marker=MARKERS[family],
                facecolors=COLORS[family],
                edgecolors=COLORS[family],
                linewidths=0.45,
                alpha=0.82,
                zorder=4,
            )

            collapsed_pct = float(np.mean(y <= threshold) * 100.0)
            collapsed_by_family.append(collapsed_pct)
        while len(collapsed_by_family) < len(family_order):
            collapsed_by_family.append(np.nan)

        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.45)
        ax.set_xlim(-0.55, len(family_order) - 0.45)
        ax.set_ylim(0.0, ymax)
        ax.set_xticks(np.arange(len(family_order)))
        tick_labels: list[str] = []
        for name, pct in zip(family_ticks, collapsed_by_family):
            if np.isfinite(pct):
                tick_labels.append(f"{name}\n{pct:.0f}%")
            else:
                tick_labels.append(f"{name}\nNA")
        ax.set_xticklabels(tick_labels)
        if panel_idx == 0:
            ax.set_ylabel("Distance to baseline (min-max normalized space)")
        ax.set_title(f"Day {day}", fontsize=12.5)
        ax.text(
            0.02,
            0.03,
            f"Collapsed fraction shown under each family label (n={n_points})",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=7.7,
            color=COLORS["muted"],
            zorder=8,
        )
        if panel_idx == 0:
            ax.text(
                0.98,
                0.97,
                "Dashed line = equivalence threshold",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.7,
                color=COLORS["muted"],
                zorder=8,
            )
        ax.text(
            0.01,
            1.02,
            chr(ord("A") + panel_idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=11.5,
            fontweight="semibold",
            color=COLORS["ink"],
        )
        ax.tick_params(direction="out", length=4.0, width=0.8)

    fig.suptitle("Family-wise collapse by day", y=0.98, fontsize=13.0)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sanity10_endpoint_separate_panels.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_endpoint_separate_panels.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_endpoint_separate_panels.svg", bbox_inches="tight")
    plt.close(fig)


def _build_temporal_observable_table(raw_full: pd.DataFrame) -> pd.DataFrame:
    grouped = raw_full.groupby(["condition_id", "day"], observed=True).agg(
        perturbation_family=("perturbation_family", "first"),
        ALP=("ALP", "mean"),
        collagen=("collagen", "mean"),
        mineral=("mineral", "mean"),
    )
    return grouped.reset_index()


def make_option1_trajectory_plot(raw_full: pd.DataFrame, *, out_dir: Path, dpi: int) -> None:
    temporal = _build_temporal_observable_table(raw_full)
    if temporal.empty:
        return
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.7), constrained_layout=False)
    fig.subplots_adjust(left=0.07, right=0.995, top=0.86, bottom=0.18, wspace=0.18)
    for ax, feature, title in zip(axes, FEATURES, ("ALP", "Collagen", "Mineral")):
        for family in [TYPE_BASELINE, TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING]:
            subset = temporal.loc[temporal["perturbation_family"].astype(str) == family]
            if subset.empty:
                continue
            q = subset.groupby("day", observed=True)[feature].quantile([0.25, 0.50, 0.75]).unstack()
            days = q.index.to_numpy(dtype=float)
            ax.fill_between(days, q[0.25].to_numpy(dtype=float), q[0.75].to_numpy(dtype=float), color=COLORS[family], alpha=0.12, linewidth=0.0)
            ax.plot(days, q[0.50].to_numpy(dtype=float), color=COLORS[family], linewidth=2.0 if family != TYPE_BASELINE else 2.6, marker=MARKERS[family], markersize=4.5, label=FAMILY_LABELS[family])
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.7, alpha=0.45)
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel("Mean total (a.u.)")
        ax.tick_params(direction="out", length=4.0, width=0.8)
    axes[0].legend(loc="upper left", frameon=False, ncol=1)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sanity10_option1_trajectory_lines.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_option1_trajectory_lines.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_option1_trajectory_lines.svg", bbox_inches="tight")
    plt.close(fig)


def make_option2_distance_to_baseline(raw_full: pd.DataFrame, *, out_dir: Path, dpi: int) -> None:
    cond_day = raw_full.groupby(["condition_id", "day"], observed=True).agg(
        perturbation_family=("perturbation_family", "first"),
        ALP=("ALP", "mean"),
        collagen=("collagen", "mean"),
        mineral=("mineral", "mean"),
        OB_fraction=("OB_fraction", "mean"),
        clustering=("clustering", "mean"),
        msd_effective=("msd_effective", "mean"),
        init_cells=("init_cells", "first"),
        seed_mode=("seed_mode", "first"),
    ).reset_index()
    if cond_day.empty:
        return
    baseline_id = _baseline_condition_id(raw_full)
    z_cols = [f"z_{col}" for col in FEATURES]
    cond_day, _ = standardize(cond_day, list(FEATURES), by="day")
    base = cond_day.loc[cond_day["condition_id"].astype(str) == baseline_id, ["day", *z_cols]].copy()
    if base.empty:
        return
    cond_day = cond_day.merge(base, on="day", how="left", suffixes=("", "_base"))
    cond_day["dist_to_baseline"] = np.sqrt(
        (cond_day["z_ALP"] - cond_day["z_ALP_base"]) ** 2
        + (cond_day["z_collagen"] - cond_day["z_collagen_base"]) ** 2
        + (cond_day["z_mineral"] - cond_day["z_mineral_base"]) ** 2
    )

    family_order = [TYPE_BASELINE, TYPE_SPATIAL, TYPE_CELLS, TYPE_DYNAMICS, TYPE_AA, TYPE_BOUNDARY, TYPE_SAMPLING]
    family_ticks = ["Baseline", "Spatial", "Cell #", "Dynamics", "AA", "Boundary", "Sampling"]

    n_family = len(family_order)
    n_panels = n_family + 1  # + cross-family summary panel
    ncols = 4
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16.8, 3.4 * nrows), constrained_layout=False)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.92, bottom=0.10, wspace=0.24, hspace=0.34)
    flat_axes = np.atleast_1d(axes).ravel()

    for idx, family in enumerate(family_order):
        ax = flat_axes[idx]
        subset = cond_day.loc[cond_day["perturbation_family"].astype(str) == family].copy()
        if subset.empty:
            ax.set_visible(False)
            continue
        q = subset.groupby("day", observed=True)["dist_to_baseline"].quantile([0.25, 0.50, 0.75]).unstack()
        days = q.index.to_numpy(dtype=float)

        # Light raw trajectories to show spread without overpowering quartile trends.
        pivot = subset.pivot_table(index="day", columns="condition_id", values="dist_to_baseline", aggfunc="mean")
        for _, series in pivot.items():
            ax.plot(
                pivot.index.to_numpy(dtype=float),
                series.to_numpy(dtype=float),
                color=COLORS[family],
                linewidth=0.6,
                alpha=0.10,
                zorder=1,
            )

        ax.fill_between(
            days,
            q[0.25].to_numpy(dtype=float),
            q[0.75].to_numpy(dtype=float),
            color=COLORS[family],
            alpha=0.18,
            linewidth=0.0,
            zorder=2,
        )
        ax.plot(
            days,
            q[0.50].to_numpy(dtype=float),
            color=COLORS[family],
            linewidth=2.4 if family != TYPE_BASELINE else 2.8,
            marker=MARKERS[family],
            markersize=5.0,
            zorder=3,
        )
        ax.grid(axis="y", color=COLORS["grid"], linewidth=0.8, alpha=0.45)
        ax.set_title(f"{FAMILY_LABELS[family]} (n={subset['condition_id'].nunique()})", fontsize=11.0)
        ax.set_xlabel("Day")
        if idx % ncols == 0:
            ax.set_ylabel("Distance to baseline phenotype")
        ax.tick_params(axis="both", labelsize=9.3)

    # Final panel: cross-family summary at the last day.
    summary_idx = n_family
    ax_summary = flat_axes[summary_idx]
    last_day = int(np.nanmax(cond_day["day"].to_numpy(dtype=float)))
    last_df = cond_day.loc[cond_day["day"].astype(int) == last_day].copy()
    medians = (
        last_df.groupby("perturbation_family", observed=True)["dist_to_baseline"]
        .median()
        .reindex(family_order)
        .fillna(np.nan)
    )
    y_pos = np.arange(len(family_order))
    colors = [COLORS[fam] for fam in family_order]
    values = medians.to_numpy(dtype=float)
    ax_summary.barh(y_pos, values, color=colors, alpha=0.75, height=0.62)
    ax_summary.set_yticks(y_pos)
    ax_summary.set_yticklabels(family_ticks)
    ax_summary.invert_yaxis()
    ax_summary.grid(axis="x", color=COLORS["grid"], linewidth=0.8, alpha=0.45)
    ax_summary.set_xlabel(f"Median distance (day {last_day})")
    ax_summary.set_title("Cross-family rank at endpoint", fontsize=11.0)
    x_max = float(np.nanmax(values)) if np.any(np.isfinite(values)) else 1.0
    for y_i, val in zip(y_pos, values):
        if np.isfinite(val):
            ax_summary.text(val + 0.02 * max(x_max, 1.0), y_i, f"{val:.2f}", va="center", ha="left", fontsize=8.7, color=COLORS["ink"])
    ax_summary.tick_params(axis="both", labelsize=9.3)

    # Hide any unused panel slots.
    for j in range(n_panels, len(flat_axes)):
        flat_axes[j].set_visible(False)

    fig.suptitle("Distance-to-baseline trajectories by perturbation family", fontsize=13.0, y=0.985)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sanity10_option2_distance_to_baseline.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_option2_distance_to_baseline.pdf", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "sanity10_option2_distance_to_baseline.svg", bbox_inches="tight")
    plt.close(fig)


def run_analysis(args: argparse.Namespace, raw_full: pd.DataFrame) -> None:
    args.out.mkdir(parents=True, exist_ok=True)
    selected_figs = _normalize_figure_selection(getattr(args, "figures", ["all"]))
    use_observables_only = bool(getattr(args, "observables_only", False))
    use_pca = not bool(getattr(args, "no_pca", False))
    raw = raw_full.copy()

    day_col = resolve_column(raw, ("day",), required=True)
    cond_col = resolve_column(raw, ("condition_id", "condition", "conditionid"), required=True)
    rep_col = resolve_column(raw, ("replicate", "rep", "run"), required=False)
    if rep_col is None:
        raw["replicate"] = 1
        rep_col = "replicate"
    raw["condition_id"] = raw[cond_col].astype(str)
    raw["replicate"] = pd.to_numeric(raw[rep_col], errors="coerce").fillna(1).astype(int)
    raw["day"] = pd.to_numeric(raw[day_col], errors="coerce").round().astype(int)

    for target, names in {
        "ALP": ("ALP", "alp", "alp_total"),
        "collagen": ("collagen", "collagen_total"),
        "mineral": ("mineral", "ha", "ha_total", "ars", "hap"),
    }.items():
        col = resolve_column(raw, names, required=True)
        raw[target] = pd.to_numeric(raw[col], errors="coerce")

    if "n_osteoblast" in raw.columns and "n_total" in raw.columns:
        n_ob = pd.to_numeric(raw["n_osteoblast"], errors="coerce").fillna(0.0)
        n_total = pd.to_numeric(raw["n_total"], errors="coerce").fillna(0.0)
        raw["OB_fraction"] = np.where(n_total > 0.0, n_ob / n_total, 0.0)
    elif "OB_fraction" not in raw.columns:
        raw["OB_fraction"] = 0.0

    if "clustering" not in raw.columns:
        if "nodule_count_current" in raw.columns and "n_total" in raw.columns:
            nodule = pd.to_numeric(raw["nodule_count_current"], errors="coerce").fillna(0.0)
            n_total = pd.to_numeric(raw["n_total"], errors="coerce").fillna(0.0)
            raw["clustering"] = np.where(n_total > 0.0, nodule / n_total, 0.0)
        else:
            raw["clustering"] = 0.0
    if "msd_effective" not in raw.columns:
        raw["msd_effective"] = 0.0

    raw["perturbation_family"] = infer_family(raw, args.family_col)
    if args.spatial_col and args.spatial_col in raw.columns:
        raw["spatial_class"] = raw[args.spatial_col].astype(str)
    else:
        raw["spatial_class"] = raw.apply(assign_spatial_class, axis=1)

    raw = raw.loc[raw["day"].isin(args.days)].copy()
    if raw.empty:
        raise SystemExit(f"No rows found for requested days: {args.days}")

    stats_wide, candidates = _endpoint_statistics(raw)
    if not stats_wide.empty:
        safe_to_csv(stats_wide.sort_values("condition_id"), args.out / "endpoint_statistics_vs_baseline.csv")
        safe_to_csv(candidates, args.out / "degenerate_candidates.csv")

    meta_cols = [
        col
        for col in [
            "condition_id",
            "condition_type",
            "perturbation_family",
            "spatial_class",
            "init_cells",
            "seed_mode",
            "grid_spacing",
            "init_slug",
            "init_title",
            "layout_seed",
            "dynamic_profile",
            "cell_profile",
            "aa_profile",
            "boundary_profile",
            "sampling_profile",
            "w",
            "h",
            "boundary_mode",
            "aa0",
            "aa_half_life",
            "collagen_aa_km",
            "media_interval",
            "sampling_offset_days",
            "sampled_day",
        ]
        if col in raw.columns and col not in {"condition_id", "day"}
    ]
    cond_day = raw.groupby(["condition_id", "day"], observed=True).agg(
        {**{col: "first" for col in meta_cols}, **{col: "mean" for col in [*FEATURES, *LATENT_FEATURE_CANDIDATES]}}
    ).reset_index()
    if not stats_wide.empty:
        cond_day = cond_day.merge(
            stats_wide[["condition_id", "endpoint_distance", "mineral_nonsignificant", "all_endpoints_nonsignificant"]],
            on="condition_id",
            how="left",
        )
    else:
        cond_day["endpoint_distance"] = np.nan
        cond_day["mineral_nonsignificant"] = False
        cond_day["all_endpoints_nonsignificant"] = False

    scale_group = "day" if args.scaling == "per-day" else None
    scale_features = list(FEATURES) if use_observables_only else list(FEATURES) + list(LATENT_FEATURE_CANDIDATES)
    cond_day, _ = standardize(cond_day, scale_features, by=scale_group)

    if use_observables_only:
        pca_features = [f"z_{feature}" for feature in FEATURES]
        latent_z_cols: list[str] = []
    else:
        pca_features = [f"z_{feature}" for feature in FEATURES] + [f"z_{feature}" for feature in LATENT_FEATURE_CANDIDATES]
        latent_z_cols = [f"z_{feature}" for feature in LATENT_FEATURE_CANDIDATES]

    loadings = pd.DataFrame()
    if use_pca:
        if args.pca_mode == "global":
            model = fit_pca(cond_day, pca_features)
            coords = transform_pca(model, cond_day)
            cond_day["pc1"] = coords[:, 0]
            cond_day["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
            cond_day["pc1_var"] = float(model.explained_variance_ratio[0]) if len(model.explained_variance_ratio) >= 1 else np.nan
            cond_day["pc2_var"] = float(model.explained_variance_ratio[1]) if len(model.explained_variance_ratio) >= 2 else np.nan
            loadings = pd.DataFrame(
                {
                    "day": np.repeat(cond_day["day"].unique(), len(pca_features)),
                    "feature": np.tile(pca_features, len(cond_day["day"].unique())),
                    "pc1_loading": np.tile(model.components[0], len(cond_day["day"].unique())),
                    "pc2_loading": np.tile(model.components[1] if model.components.shape[0] > 1 else np.zeros(len(pca_features)), len(cond_day["day"].unique())),
                    "pc1_explained_var": float(model.explained_variance_ratio[0]) if len(model.explained_variance_ratio) >= 1 else np.nan,
                    "pc2_explained_var": float(model.explained_variance_ratio[1]) if len(model.explained_variance_ratio) >= 2 else np.nan,
                }
            )
        else:
            pcs: list[pd.DataFrame] = []
            load_rows: list[dict[str, Any]] = []
            for day, subset in cond_day.groupby("day", observed=True):
                model = fit_pca(subset, pca_features)
                coords = transform_pca(model, subset)
                block = subset.copy()
                block["pc1"] = coords[:, 0]
                block["pc2"] = coords[:, 1] if coords.shape[1] > 1 else 0.0
                block["pc1_var"] = float(model.explained_variance_ratio[0]) if len(model.explained_variance_ratio) >= 1 else np.nan
                block["pc2_var"] = float(model.explained_variance_ratio[1]) if len(model.explained_variance_ratio) >= 2 else np.nan
                pcs.append(block)
                for idx, feature in enumerate(pca_features):
                    load_rows.append(
                        {
                            "day": int(day),
                            "feature": feature,
                            "pc1_loading": float(model.components[0, idx]),
                            "pc2_loading": float(model.components[1, idx]) if model.components.shape[0] > 1 else 0.0,
                            "pc1_explained_var": float(model.explained_variance_ratio[0]) if len(model.explained_variance_ratio) >= 1 else np.nan,
                            "pc2_explained_var": float(model.explained_variance_ratio[1]) if len(model.explained_variance_ratio) >= 2 else np.nan,
                        }
                    )
            cond_day = pd.concat(pcs, ignore_index=True) if pcs else cond_day.copy()
            loadings = pd.DataFrame(load_rows)

    metrics = [
        compute_day_metrics(
            day_df,
            args.equivalence_threshold,
            [f"z_{feature}" for feature in FEATURES],
            latent_z_cols,
        ).__dict__
        for _, day_df in cond_day.groupby("day", observed=True)
    ]
    metrics_df = pd.DataFrame(metrics)
    if use_pca:
        safe_to_csv(cond_day.sort_values(["day", "condition_id"]), args.out / "pca_points_day10_14_28.csv")
        safe_to_csv(loadings.sort_values(["day", "feature"]), args.out / "pca_loadings.csv")
    safe_to_csv(metrics_df.sort_values("day"), args.out / "resolution_index_by_day.csv")

    collapsed_pairs: list[dict[str, Any]] = []
    for day, day_df in cond_day.groupby("day", observed=True):
        centroids = day_df.groupby("perturbation_family", observed=True)[[f"z_{feature}" for feature in FEATURES]].mean()
        for left, right in combinations(centroids.index.tolist(), 2):
            dist = float(np.linalg.norm(centroids.loc[left].to_numpy(dtype=float) - centroids.loc[right].to_numpy(dtype=float)))
            collapsed_pairs.append(
                {
                    "day": int(day),
                    "family_A": str(left),
                    "family_B": str(right),
                    "distance": dist,
                    "collapsed": bool(dist <= args.equivalence_threshold),
                }
            )
    safe_to_csv(pd.DataFrame(collapsed_pairs), args.out / "collapsed_pairs_by_day.csv")

    if "composite" in selected_figs and use_pca:
        make_plot(cond_day, metrics_df, loadings, out_dir=args.out, dpi=args.dpi)
    if "collapse" in selected_figs:
        make_endpoint_separate_plot(
            raw,
            stats_wide,
            out_dir=args.out,
            dpi=args.dpi,
            threshold=float(args.equivalence_threshold),
        )
    if "trajectory" in selected_figs:
        make_option1_trajectory_plot(raw, out_dir=args.out, dpi=args.dpi)
    if "distance" in selected_figs:
        make_option2_distance_to_baseline(raw, out_dir=args.out, dpi=args.dpi)

    if bool(getattr(args, "export_summary_json", False)):
        summary = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "input_path": str(getattr(args, "input", "")),
            "out_dir": str(args.out),
            "days": [int(d) for d in args.days],
            "scaling": str(args.scaling),
            "pca_mode": str(args.pca_mode),
            "observables_only": bool(use_observables_only),
            "no_pca": bool(not use_pca),
            "equivalence_threshold": float(args.equivalence_threshold),
            "figures": sorted(selected_figs),
            "family_col": str(getattr(args, "family_col", "")),
            "spatial_col": str(getattr(args, "spatial_col", "")),
        }
        (args.out / "analysis_summary.json").write_text(
            json.dumps(summary, indent=2, default=_jsonable) + "\n",
            encoding="utf-8",
        )

    if "composite" in selected_figs and use_pca:
        print(f"[sanity_check10] figure: {args.out / 'sanity10_pca_collapse_composite.png'}")
    print(f"[sanity_check10] metrics: {args.out / 'resolution_index_by_day.csv'}")
    print(f"[sanity_check10] collapsed pairs: {args.out / 'collapsed_pairs_by_day.csv'}")
    if use_pca:
        print(f"[sanity_check10] points: {args.out / 'pca_points_day10_14_28.csv'}")


def assign_spatial_class(row: pd.Series) -> str:
    if "spatial_class" in row.index and str(row["spatial_class"]) not in {"", "nan"}:
        return str(row["spatial_class"])
    return spatial_class_from_seeding(row)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    def _run_thresholds(base_args: argparse.Namespace, raw_df: pd.DataFrame) -> None:
        thresholds = getattr(base_args, "threshold_sweep", None)
        if thresholds:
            for thr in thresholds:
                run_args = argparse.Namespace(**vars(base_args))
                run_args.equivalence_threshold = float(thr)
                run_args.out = Path(base_args.out) / f"thr_{float(thr):.2f}"
                run_args.out.mkdir(parents=True, exist_ok=True)
                run_analysis(run_args, raw_df.copy())
        else:
            run_analysis(base_args, raw_df)

    if args.analyze_only:
        src = args.input if args.input is not None else args.out / "simulated_observables_long.csv"
        raw = load_input(src, args.sheet)
        if args.input is None:
            args.input = src
        _run_thresholds(args, raw)
        return
    sim_df = simulate_conditions(args)
    if args.dry_run:
        print("[sanity_check10] dry run complete; simulation commands printed only")
        return
    if sim_df.empty:
        print("[sanity_check10] no rows were produced")
        return
    _run_thresholds(args, sim_df)


if __name__ == "__main__":
    main()
