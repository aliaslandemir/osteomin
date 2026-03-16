#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from textwrap import fill

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.patches import Rectangle

from src.config import ABMConfig, FieldConfig
from src.model import OsteoModel
from src.time_utils import HOURS_PER_DAY
from src.viz import plot_summary


BASE_OUT = SCRIPT_DIR / "out" / "sanity_check2"
TOTAL_HOURS = 672
CELLS = 72
GRID_W = 33
GRID_H = 33
MODEL_SEED = 12345
REPLICATES = 5
ALP_HOURS = 240
COLLAGEN_HOURS = 336
MINERAL_HOURS = 672
PRIMARY_METRICS = [
    "alp_240h",
    "collagen_336h",
    "ha_672h",
    "final_live",
    "final_preob",
    "final_ob",
]

COLORS = {
    "ink": "#111827",
    "muted": "#64748b",
    "grid": "#d4dce6",
    "control": "#1d4ed8",
    "accent": "#ea580c",
    "alp": "#2563eb",
    "collagen": "#059669",
    "ha": "#b45309",
    "preob": "#1f77b4",
    "ob": "#d97706",
    "bg": "#f8fafc",
}


plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "legend.fontsize": 8.2,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


@dataclass(frozen=True)
class ConditionSpec:
    index: int
    slug: str
    title: str
    description: str
    layout_seed: int


@dataclass
class ReplicateResult:
    replicate_index: int
    out_dir: Path
    coords: list[tuple[int, int]]
    df: pd.DataFrame
    endpoint_metrics: dict[str, float]
    layout_metrics: dict[str, float]
    model_seed: int
    layout_seed: int


@dataclass
class ConditionResult:
    spec: ConditionSpec
    out_dir: Path
    replicates: list[ReplicateResult]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Structured initial-condition sanity suite for OsteoMin.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=BASE_OUT, help="output directory")
    parser.add_argument("--seed", type=int, default=MODEL_SEED, help="base model RNG seed")
    parser.add_argument("--cells", type=int, default=CELLS, help="number of seeded cells per condition")
    parser.add_argument("--hours", type=int, default=TOTAL_HOURS, help="simulation hours")
    parser.add_argument("--w", type=int, default=GRID_W, help="grid width")
    parser.add_argument("--h", type=int, default=GRID_H, help="grid height")
    parser.add_argument(
        "--replicates",
        type=int,
        default=REPLICATES,
        help="stochastic replicates per initialization mode",
    )
    parser.add_argument(
        "--runtime",
        action="store_true",
        help="print runtime progress from the underlying model",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="rerun conditions even if cached outputs already exist",
    )
    args = parser.parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    if args.cells <= 0:
        raise SystemExit("--cells must be >= 1")
    if args.hours <= 0:
        raise SystemExit("--hours must be >= 1")
    if args.w <= 0 or args.h <= 0:
        raise SystemExit("--w and --h must be >= 1")
    return args


def build_configs(width: int, height: int) -> tuple[ABMConfig, FieldConfig]:
    induction_hours = 72
    abm_cfg = ABMConfig(
        width=width,
        height=height,
        boundary_mode="periodic",
        enable_random_move=True,
        enable_commitment=True,
        enable_apoptosis=True,
        enable_proliferation=True,
        enable_secretion=True,
        move_prob=0.27,
        max_moves_per_hour=1,
        ob_division_prob=0.0,
        apoptosis_prob=0.0010,
        ob_apoptosis_prob=0.0,
        proliferation_chance=1.0,
        base_commit_prob=0.0005,
        osteo_induction_hours=induction_hours,
        commit_delay=48,
        commit_k_hourly=0.0125,
        min_div_hours=38,
        max_div_hours=52,
    )
    field_cfg = FieldConfig(
        width=width,
        height=height,
        media_interval=48,
        enable_secretion=True,
        osteo_induction_hours=induction_hours,
        dex_conc=0.0,
        mk4_conc=0.0,
        dex_ref_conc=20.0,
        mk4_ref_conc=0.6,
        alp_basal_rate=0.094,
        alp_osteo_rate=0.810,
        alp_initial=0.8,
        alp_dex_ref_conc=20.0,
        alp_mk4_ref_conc=0.6,
        alp_a_d=3.79,
        alp_b_d=1.92,
        alp_a_k=0.56,
        alp_b_k=0.471,
        collagen_basal_rate=0.00355,
        collagen_osteo_rate=0.0131,
        collagen_dex_ref_conc=60.0,
        collagen_mk4_ref_conc=8.0,
        collagen_a_d=2.42,
        collagen_b_d=0.770,
        collagen_a_k=0.915,
        collagen_b_k=0.167,
        aa_initial=50.0,
        aa_half_life_hours=8.0,
        collagen_aa_half_sat=5.0,
        ha_model="minimal",
        ha_initial=0.0,
        ha_osteo_rate=0.106,
        ha_dex_ref_conc=50.0,
        ha_mk4_ref_conc=0.7,
        ha_a_d=1.14,
        ha_b_d=0.365,
        ha_a_k=4.69,
        ha_b_k=2.21,
        ha_tau_hours=384.0,
        ha_hill_n=2.0,
        ha_t_commit_hours=induction_hours + 48,
        ha_collagen_maturation_tau_hours=48.0,
        ha_collagen_half_sat=1.0,
    )
    return abm_cfg, field_cfg


def make_condition_specs(seed: int) -> list[ConditionSpec]:
    titles = [
        ("base_random", "Condition 1: Uniform random", "Control layout spanning the full domain with unbiased random seeding."),
        ("center_cluster", "Condition 2: Central cluster", "Cells are sampled from a compact disk around the geometric center."),
        ("edge_band", "Condition 3: Edge band", "Cells begin near the periodic boundaries to test wrap-around equivalence."),
        ("right_half", "Condition 4: Right half", "Cells are restricted to the right half of the lattice."),
        ("diagonal_band", "Condition 5: Diagonal band", "Cells are seeded along a broad main-diagonal band."),
        ("anti_diagonal_band", "Condition 6: Anti-diagonal band", "Cells are seeded along a broad anti-diagonal band."),
        ("corner_quadrants", "Condition 7: Corner quadrants", "Cells are distributed across four compact corner neighborhoods."),
        ("dual_lobes_horizontal", "Condition 8: Dual horizontal lobes", "Cells are split into left and right clusters."),
        ("dual_lobes_vertical", "Condition 9: Dual vertical lobes", "Cells are split into top and bottom clusters."),
        ("quad_lobes", "Condition 10: Four lobes", "Cells are distributed across four off-center lobes."),
        ("lattice_even", "Condition 11: Even lattice", "Cells occupy an evenly spaced lattice-like subset of sites."),
        ("vertical_stripes", "Condition 12: Vertical stripes", "Cells are confined to repeated vertical stripe bands."),
        ("checkerboard_sparse", "Condition 13: Checkerboard sparse", "Cells are sampled from alternating parity sites to maximize spacing."),
        ("radial_spokes", "Condition 14: Radial spokes", "Cells are distributed along multiple radial spokes emerging from the center."),
        ("ring_periphery", "Condition 15: Peripheral ring", "Cells are initialized on an annular ring surrounding the center to emphasize boundary-facing growth fronts."),
    ]
    return [
        ConditionSpec(
            index=i + 1,
            slug=slug,
            title=title,
            description=description,
            layout_seed=seed + 1000 + i * 101,
        )
        for i, (slug, title, description) in enumerate(titles)
    ]


def _all_sites(width: int, height: int) -> list[tuple[int, int]]:
    return [(x, y) for y in range(height) for x in range(width)]


def _sample_sites(candidates: list[tuple[int, int]], n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    if len(candidates) < n:
        raise ValueError(f"Need {n} candidate sites, found only {len(candidates)}")
    idx = rng.choice(len(candidates), size=n, replace=False)
    coords = [candidates[int(i)] for i in idx]
    coords.sort(key=lambda item: (item[1], item[0]))
    return coords


def _select_masked(
    width: int,
    height: int,
    n: int,
    rng: np.random.Generator,
    predicate,
) -> list[tuple[int, int]]:
    candidates = [(x, y) for y in range(height) for x in range(width) if predicate(x, y)]
    return _sample_sites(candidates, n, rng)


def _select_ranked(
    width: int,
    height: int,
    n: int,
    rng: np.random.Generator,
    score_fn,
    *,
    descending: bool = True,
    pool_factor: float = 1.8,
) -> list[tuple[int, int]]:
    candidates = []
    for y in range(height):
        for x in range(width):
            jitter = float(rng.uniform(-1e-6, 1e-6))
            candidates.append((float(score_fn(x, y)) + jitter, x, y))
    candidates.sort(reverse=descending)
    pool_size = max(n, min(len(candidates), int(math.ceil(n * pool_factor))))
    pool = [(x, y) for _, x, y in candidates[:pool_size]]
    return _sample_sites(pool, n, rng)


def generate_layout(
    spec: ConditionSpec,
    width: int,
    height: int,
    n: int,
    layout_seed: int,
) -> list[tuple[int, int]]:
    rng = np.random.default_rng(layout_seed)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0

    if spec.slug == "base_random":
        return _sample_sites(_all_sites(width, height), n, rng)

    if spec.slug == "center_cluster":
        radius = min(width, height) * 0.21
        return _select_masked(width, height, n, rng, lambda x, y: ((x - cx) ** 2 + (y - cy) ** 2) <= radius**2)

    if spec.slug == "edge_band":
        band = max(3, min(width, height) // 8)
        return _select_masked(width, height, n, rng, lambda x, y: min(x, y, width - 1 - x, height - 1 - y) <= band)

    if spec.slug == "left_half":
        return _select_masked(width, height, n, rng, lambda x, y: x <= (width // 2) - 1)

    if spec.slug == "right_half":
        return _select_masked(width, height, n, rng, lambda x, y: x >= (width // 2))

    if spec.slug == "top_half":
        return _select_masked(width, height, n, rng, lambda x, y: y <= (height // 2) - 1)

    if spec.slug == "bottom_half":
        return _select_masked(width, height, n, rng, lambda x, y: y >= (height // 2))

    if spec.slug == "diagonal_band":
        band = max(3, min(width, height) // 8)
        return _select_masked(width, height, n, rng, lambda x, y: abs(y - x) <= band)

    if spec.slug == "anti_diagonal_band":
        band = max(3, min(width, height) // 8)
        return _select_masked(width, height, n, rng, lambda x, y: abs((x + y) - (width - 1)) <= band)

    if spec.slug == "corner_quadrants":
        radius = max(4.0, min(width, height) * 0.18)
        corner_centers = [(3.5, 3.5), (width - 4.5, 3.5), (3.5, height - 4.5), (width - 4.5, height - 4.5)]
        per_corner = [n // 4] * 4
        for i in range(n % 4):
            per_corner[i] += 1
        coords: list[tuple[int, int]] = []
        used: set[tuple[int, int]] = set()
        for (px, py), count in zip(corner_centers, per_corner):
            candidates = [
                (x, y)
                for y in range(height)
                for x in range(width)
                if ((x - px) ** 2 + (y - py) ** 2) <= radius**2 and (x, y) not in used
            ]
            picks = _sample_sites(candidates, count, rng)
            coords.extend(picks)
            used.update(picks)
        coords.sort(key=lambda item: (item[1], item[0]))
        return coords

    if spec.slug == "dual_lobes_horizontal":
        radius = max(4.0, min(width, height) * 0.16)
        centers = [(width * 0.27, cy), (width * 0.73, cy)]
        counts = [n // 2, n - (n // 2)]
        return _sample_multi_disks(width, height, centers, radius, counts, rng)

    if spec.slug == "dual_lobes_vertical":
        radius = max(4.0, min(width, height) * 0.16)
        centers = [(cx, height * 0.27), (cx, height * 0.73)]
        counts = [n // 2, n - (n // 2)]
        return _sample_multi_disks(width, height, centers, radius, counts, rng)

    if spec.slug == "quad_lobes":
        radius = max(3.5, min(width, height) * 0.14)
        centers = [
            (width * 0.28, height * 0.28),
            (width * 0.72, height * 0.28),
            (width * 0.28, height * 0.72),
            (width * 0.72, height * 0.72),
        ]
        counts = [n // 4] * 4
        for i in range(n % 4):
            counts[i] += 1
        return _sample_multi_disks(width, height, centers, radius, counts, rng)

    if spec.slug == "lattice_even":
        spacing = max(2, int(round(math.sqrt((width * height) / max(n, 1)))))
        candidates = [(x, y) for y in range(0, height, spacing) for x in range(0, width, spacing)]
        if len(candidates) < n:
            spacing = max(1, spacing - 1)
            candidates = [(x, y) for y in range(0, height, spacing) for x in range(0, width, spacing)]
        return _sample_sites(candidates, n, rng)

    if spec.slug == "ring_periphery":
        inner = min(width, height) * 0.18
        outer = min(width, height) * 0.36
        return _select_masked(
            width,
            height,
            n,
            rng,
            lambda x, y: inner**2 <= ((x - cx) ** 2 + (y - cy) ** 2) <= outer**2,
        )

    if spec.slug == "center_cross":
        band = max(2, min(width, height) // 10)
        return _select_masked(width, height, n, rng, lambda x, y: abs(x - cx) <= band or abs(y - cy) <= band)

    if spec.slug == "vertical_stripes":
        period = 6
        stripe_width = 2
        return _select_masked(width, height, n, rng, lambda x, y: (x % period) < stripe_width)

    if spec.slug == "horizontal_stripes":
        period = 6
        stripe_width = 2
        return _select_masked(width, height, n, rng, lambda x, y: (y % period) < stripe_width)

    if spec.slug == "checkerboard_sparse":
        return _select_masked(width, height, n, rng, lambda x, y: ((x + y) % 2) == 0)

    if spec.slug == "radial_spokes":
        angles = [0.0, math.pi / 4.0, math.pi / 2.0, 3.0 * math.pi / 4.0]
        band = 1.1

        def on_spoke(x: int, y: int) -> bool:
            dx = x - cx
            dy = y - cy
            if abs(dx) < 1e-9 and abs(dy) < 1e-9:
                return True
            theta = math.atan2(dy, dx)
            for angle in angles:
                delta = min(abs(theta - angle), 2.0 * math.pi - abs(theta - angle))
                if delta <= 0.12 or abs(delta - math.pi) <= 0.12:
                    proj = abs(-math.sin(angle) * dx + math.cos(angle) * dy)
                    if proj <= band:
                        return True
            return False

        return _select_masked(width, height, n, rng, on_spoke)

    raise ValueError(f"Unhandled condition slug: {spec.slug}")


def _sample_multi_disks(
    width: int,
    height: int,
    centers: list[tuple[float, float]],
    radius: float,
    counts: list[int],
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    used: set[tuple[int, int]] = set()
    for (px, py), count in zip(centers, counts):
        candidates = [
            (x, y)
            for y in range(height)
            for x in range(width)
            if ((x - px) ** 2 + (y - py) ** 2) <= radius**2 and (x, y) not in used
        ]
        picks = _sample_sites(candidates, count, rng)
        coords.extend(picks)
        used.update(picks)
    coords.sort(key=lambda item: (item[1], item[0]))
    return coords


def summarize_layout(coords: list[tuple[int, int]], width: int, height: int) -> dict[str, float]:
    arr = np.asarray(coords, dtype=float)
    cx = (width - 1) / 2.0
    cy = (height - 1) / 2.0
    dist_center = np.sqrt((arr[:, 0] - cx) ** 2 + (arr[:, 1] - cy) ** 2)
    edge_dist = np.minimum.reduce([arr[:, 0], arr[:, 1], (width - 1) - arr[:, 0], (height - 1) - arr[:, 1]])
    diffs = arr[:, None, :] - arr[None, :, :]
    dist = np.sqrt(np.sum(diffs**2, axis=2))
    np.fill_diagonal(dist, np.inf)
    nearest_neighbor = float(np.min(dist, axis=1).mean())
    return {
        "seed_centroid_x": float(arr[:, 0].mean()),
        "seed_centroid_y": float(arr[:, 1].mean()),
        "seed_std_x": float(arr[:, 0].std(ddof=0)),
        "seed_std_y": float(arr[:, 1].std(ddof=0)),
        "seed_mean_radius": float(dist_center.mean()),
        "seed_edge_fraction": float(np.mean(edge_dist <= 3.0)),
        "seed_center_fraction": float(np.mean(dist_center <= min(width, height) * 0.22)),
        "seed_left_fraction": float(np.mean(arr[:, 0] < cx)),
        "seed_top_fraction": float(np.mean(arr[:, 1] < cy)),
        "seed_mean_nearest_neighbor": nearest_neighbor,
    }


def prepare_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["dead_total_today"] = out["dead_preob_today"] + out["dead_ob_today"]
    out["dead_preob_cum"] = out["dead_preob_today"].cumsum()
    out["dead_ob_cum"] = out["dead_ob_today"].cumsum()
    out["dead_total_cum"] = out["dead_total_today"].cumsum()
    return out


def row_at_hours(df: pd.DataFrame, hours: int) -> pd.Series:
    target_day = hours / HOURS_PER_DAY
    idx = int((df["day"] - target_day).abs().idxmin())
    return df.loc[idx]


def compute_endpoint_metrics(df: pd.DataFrame) -> dict[str, float]:
    alp_row = row_at_hours(df, ALP_HOURS)
    collagen_row = row_at_hours(df, COLLAGEN_HOURS)
    mineral_row = row_at_hours(df, MINERAL_HOURS)
    final_row = df.iloc[-1]
    return {
        "alp_240h": float(alp_row["alp_total"]),
        "collagen_336h": float(collagen_row["collagen_total"]),
        "ha_672h": float(mineral_row["ha_total"]),
        "final_alp": float(final_row["alp_total"]),
        "final_collagen": float(final_row["collagen_total"]),
        "final_ha": float(final_row["ha_total"]),
        "final_preob": float(final_row["n_preob"]),
        "final_ob": float(final_row["n_osteoblast"]),
        "final_live": float(final_row["n_total"]),
        "final_nodule_count": float(final_row.get("nodule_count_ever", np.nan)),
        "final_msd": float(final_row.get("msd_effective", np.nan)),
    }


def _fmt(value: float) -> str:
    if pd.isna(value):
        return "NA"
    if abs(value) >= 100:
        return f"{value:.1f}"
    if abs(value) >= 10:
        return f"{value:.2f}"
    return f"{value:.3f}"


def _pct(delta: float, baseline: float) -> str:
    if abs(baseline) < 1e-12:
        return "NA"
    return f"{100.0 * delta / baseline:+.1f}%"


def _fmt_p(value: float) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.3g}"


def style_axes(ax: plt.Axes, grid_axis: str = "y") -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.0, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=0.7)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def run_replicate(
    spec: ConditionSpec,
    condition_dir: Path,
    replicate_index: int,
    abm_cfg: ABMConfig,
    field_cfg: FieldConfig,
    model_seed: int,
    total_hours: int,
    n_cells: int,
    width: int,
    height: int,
    show_runtime: bool,
    overwrite: bool,
) -> ReplicateResult:
    replicate_dir = condition_dir / f"replicate{replicate_index:02d}"
    replicate_dir.mkdir(parents=True, exist_ok=True)
    final_day = total_hours // HOURS_PER_DAY
    layout_seed = spec.layout_seed + replicate_index * 10007
    coords = generate_layout(spec, width, height, n_cells, layout_seed)
    layout_metrics = summarize_layout(coords, width, height)
    model_seed_rep = model_seed + spec.index * 1000 + replicate_index

    pd.DataFrame(coords, columns=["x", "y"]).to_csv(replicate_dir / "seed_coords.csv", index=False)
    (replicate_dir / "replicate.json").write_text(
        json.dumps(
            {
                "spec": asdict(spec),
                "replicate_index": replicate_index,
                "layout_seed": layout_seed,
                "model_seed": model_seed_rep,
                "total_hours": total_hours,
                "cells": n_cells,
                "layout_metrics": layout_metrics,
                "abm_cfg": asdict(abm_cfg),
                "field_cfg": asdict(field_cfg),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    timeseries_path = replicate_dir / "timeseries.csv"
    final_state_path = replicate_dir / "grids" / f"state_day_{final_day:03d}.npy"
    if timeseries_path.exists() and final_state_path.exists() and not overwrite:
        df = prepare_timeseries(pd.read_csv(timeseries_path))
        plot_summary(
            df,
            replicate_dir,
            field_cfg=field_cfg,
            total_hours=total_hours,
            alp_plot_hours=ALP_HOURS,
            collagen_plot_hours=COLLAGEN_HOURS,
            mineral_plot_hours=MINERAL_HOURS,
        )
    else:
        model = OsteoModel(
            outdir=replicate_dir,
            total_hours=total_hours,
            abm_cfg=abm_cfg,
            field_cfg=field_cfg,
            seed=model_seed_rep,
            spatial_every=24,
            spatial_days=[],
            snapshots_enabled=True,
            unique_id_tracking=False,
            hourly_exports=False,
            show_runtime=show_runtime,
            runtime_every_days=1,
            nodule_connectivity="40n",
        )
        model.abm.seed_from_coords(coords, day0=0)
        df = prepare_timeseries(model.run())
        model.save_timeseries_csv(df, "timeseries")
        plot_summary(
            df,
            replicate_dir,
            field_cfg=field_cfg,
            total_hours=total_hours,
            alp_plot_hours=ALP_HOURS,
            collagen_plot_hours=COLLAGEN_HOURS,
            mineral_plot_hours=MINERAL_HOURS,
        )

    return ReplicateResult(
        replicate_index=replicate_index,
        out_dir=replicate_dir,
        coords=coords,
        df=df,
        endpoint_metrics=compute_endpoint_metrics(df),
        layout_metrics=layout_metrics,
        model_seed=model_seed_rep,
        layout_seed=layout_seed,
    )


def run_condition(
    spec: ConditionSpec,
    out_root: Path,
    abm_cfg: ABMConfig,
    field_cfg: FieldConfig,
    model_seed: int,
    total_hours: int,
    n_cells: int,
    width: int,
    height: int,
    replicates: int,
    show_runtime: bool,
    overwrite: bool,
) -> ConditionResult:
    condition_dir = out_root / "conditions" / f"condition{spec.index:02d}_{spec.slug}"
    condition_dir.mkdir(parents=True, exist_ok=True)
    reps = [
        run_replicate(
            spec=spec,
            condition_dir=condition_dir,
            replicate_index=rep_idx,
            abm_cfg=abm_cfg,
            field_cfg=field_cfg,
            model_seed=model_seed,
            total_hours=total_hours,
            n_cells=n_cells,
            width=width,
            height=height,
            show_runtime=show_runtime,
            overwrite=overwrite,
        )
        for rep_idx in range(1, replicates + 1)
    ]
    return ConditionResult(spec=spec, out_dir=condition_dir, replicates=reps)


def build_replicate_summary(results: list[ConditionResult]) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for result in results:
        for rep in result.replicates:
            rows.append(
                {
                    "condition_index": result.spec.index,
                    "slug": result.spec.slug,
                    "title": result.spec.title,
                    "description": result.spec.description,
                    "replicate_index": rep.replicate_index,
                    "layout_seed": rep.layout_seed,
                    "model_seed": rep.model_seed,
                    "out_dir": rep.out_dir.as_posix(),
                    **rep.layout_metrics,
                    **rep.endpoint_metrics,
                }
            )
    return pd.DataFrame(rows).sort_values(["condition_index", "replicate_index"]).reset_index(drop=True)


def build_condition_summary(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["condition_index", "slug", "title", "description"]
    numeric_cols = [
        "seed_centroid_x",
        "seed_centroid_y",
        "seed_std_x",
        "seed_std_y",
        "seed_mean_radius",
        "seed_edge_fraction",
        "seed_center_fraction",
        "seed_left_fraction",
        "seed_top_fraction",
        "seed_mean_nearest_neighbor",
        "alp_240h",
        "collagen_336h",
        "ha_672h",
        "final_alp",
        "final_collagen",
        "final_ha",
        "final_preob",
        "final_ob",
        "final_live",
        "final_nodule_count",
        "final_msd",
    ]
    grouped = replicate_summary.groupby(group_cols, sort=True)
    mean_df = grouped[numeric_cols].mean().reset_index()
    std_df = grouped[numeric_cols].std(ddof=1).reset_index().fillna(0.0)
    n_df = grouped.size().reset_index(name="n_replicates")
    first_out = grouped["out_dir"].first().reset_index(name="representative_out_dir")
    summary = mean_df.merge(std_df, on=group_cols, suffixes=("_mean", "_sd"))
    summary = summary.merge(n_df, on=group_cols)
    summary = summary.merge(first_out, on=group_cols)
    return summary.sort_values("condition_index").reset_index(drop=True)


def holm_adjust(p_values: list[float]) -> list[float]:
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        candidate = (m - rank) * float(p_values[idx])
        running = max(running, candidate)
        adjusted[idx] = min(1.0, running)
    return adjusted


def metric_label(metric: str) -> str:
    labels = {
        "alp_240h": "ALP at 240 h",
        "collagen_336h": "Collagen at 336 h",
        "ha_672h": "HAp at 672 h",
        "final_live": "Final live cells",
        "final_preob": "Final preOB",
        "final_ob": "Final OB",
    }
    return labels.get(metric, metric)


def compute_statistics(replicate_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    control_idx = int(replicate_summary["condition_index"].min())

    for metric in PRIMARY_METRICS:
        groups = [
            grp[metric].astype(float).to_numpy()
            for _, grp in replicate_summary.groupby("condition_index", sort=True)
        ]
        if len(groups) >= 2:
            try:
                h_stat, p_global = stats.kruskal(*groups)
            except ValueError:
                # SciPy raises when all observations are identical; treat that as no evidence of a difference.
                h_stat, p_global = 0.0, 1.0
        else:
            h_stat, p_global = np.nan, np.nan

        pairwise_rows = []
        control_vals = replicate_summary.loc[replicate_summary["condition_index"] == control_idx, metric].astype(float).to_numpy()
        for condition_index, grp in replicate_summary.groupby("condition_index", sort=True):
            vals = grp[metric].astype(float).to_numpy()
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
            if int(condition_index) == control_idx:
                pairwise_rows.append(
                    {
                        "metric": metric,
                        "metric_label": metric_label(metric),
                        "condition_index": int(condition_index),
                        "slug": str(grp["slug"].iloc[0]),
                        "title": str(grp["title"].iloc[0]),
                        "n": int(len(vals)),
                        "mean": mean_val,
                        "sd": sd_val,
                        "delta_vs_control": 0.0,
                        "pct_vs_control": 0.0,
                        "u_stat": np.nan,
                        "p_value": np.nan,
                        "p_adj": np.nan,
                        "significant_vs_control": False,
                        "global_h": float(h_stat) if not pd.isna(h_stat) else np.nan,
                        "global_p": float(p_global) if not pd.isna(p_global) else np.nan,
                    }
                )
                continue
            try:
                u_stat, p_value = stats.mannwhitneyu(vals, control_vals, alternative="two-sided")
            except ValueError:
                u_stat, p_value = np.nan, 1.0
            pairwise_rows.append(
                {
                    "metric": metric,
                    "metric_label": metric_label(metric),
                    "condition_index": int(condition_index),
                    "slug": str(grp["slug"].iloc[0]),
                    "title": str(grp["title"].iloc[0]),
                    "n": int(len(vals)),
                    "mean": mean_val,
                    "sd": sd_val,
                    "delta_vs_control": mean_val - float(np.mean(control_vals)),
                    "pct_vs_control": 100.0 * (mean_val - float(np.mean(control_vals))) / max(float(np.mean(control_vals)), 1e-12),
                    "u_stat": float(u_stat),
                    "p_value": float(p_value),
                    "p_adj": np.nan,
                    "significant_vs_control": False,
                    "global_h": float(h_stat) if not pd.isna(h_stat) else np.nan,
                    "global_p": float(p_global) if not pd.isna(p_global) else np.nan,
                }
            )

        pvals = [float(row["p_value"]) for row in pairwise_rows if not pd.isna(row["p_value"])]
        adj = holm_adjust(pvals) if pvals else []
        adj_iter = iter(adj)
        for row in pairwise_rows:
            if pd.isna(row["p_value"]):
                row["p_adj"] = np.nan
                row["significant_vs_control"] = False
            else:
                row["p_adj"] = float(next(adj_iter))
                row["significant_vs_control"] = bool(row["p_adj"] < 0.05)
        rows.extend(pairwise_rows)

    return pd.DataFrame(rows).sort_values(["metric", "condition_index"]).reset_index(drop=True)


def significance_symbol(p_adj: float | float) -> str:
    if pd.isna(p_adj):
        return ""
    if p_adj < 0.001:
        return "***"
    if p_adj < 0.01:
        return "**"
    if p_adj < 0.05:
        return "*"
    return ""


def export_seed_layouts(results: list[ConditionResult], out_dir: Path, width: int, height: int) -> None:
    fig, axes = plt.subplots(4, 5, figsize=(16.0, 12.2), constrained_layout=True)
    for ax, result in zip(axes.flat, results):
        arr = np.asarray(result.replicates[0].coords)
        ax.scatter(arr[:, 0], arr[:, 1], s=26, c=COLORS["control"], edgecolors="white", linewidths=0.3)
        ax.add_patch(Rectangle((-0.5, -0.5), width, height, fill=False, lw=1.0, ec=COLORS["ink"]))
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect("equal")
        ax.set_xticks([0, width // 2, width - 1])
        ax.set_yticks([0, height // 2, height - 1])
        ax.set_title(f"C{result.spec.index:02d} {result.spec.slug}", loc="left", fontsize=9.8, pad=5)
        ax.text(
            0.02,
            -0.16,
            fill(result.spec.description, width=38),
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=7.2,
            color=COLORS["muted"],
        )
        style_axes(ax, grid_axis="")
    fig.suptitle("Structured initial seeding layouts", fontsize=15, y=1.01)
    save_figure(fig, out_dir, "01_seed_layouts")


def export_timecourse_overlays(results: list[ConditionResult], out_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), constrained_layout=True)
    series_meta = [
        ("alp_total", "ALP accumulation", COLORS["alp"]),
        ("collagen_total", "Collagen accumulation", COLORS["collagen"]),
        ("ha_total", "HAp accumulation", COLORS["ha"]),
        ("n_total", "Live cell count", COLORS["ink"]),
    ]
    for ax, (column, title, base_color) in zip(axes.flat, series_meta):
        for result in results:
            rep0 = result.replicates[0]
            ax.plot(
                rep0.df["day"].to_numpy(),
                rep0.df[column],
                color=base_color if result.spec.index == 1 else "#94a3b8",
                alpha=1.0 if result.spec.index == 1 else 0.45,
                lw=2.7 if result.spec.index == 1 else 1.0,
                zorder=3 if result.spec.index == 1 else 1,
            )
        highlight = {"alp_total": ALP_HOURS, "collagen_total": COLLAGEN_HOURS, "ha_total": MINERAL_HOURS}.get(column)
        if highlight is not None:
            ax.axvline(highlight / HOURS_PER_DAY, color=COLORS["accent"], lw=1.1, ls="--", alpha=0.8)
        style_axes(ax, grid_axis="both")
        ax.set_title(title)
        ax.set_xlabel("Day")
        ax.set_ylabel("Total")
    fig.suptitle("Representative trajectory overlays with control highlighted", fontsize=15, y=1.02)
    save_figure(fig, out_dir, "02_timecourse_overlays")


def export_endpoint_bars(replicate_summary: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(13.6, 12.0), constrained_layout=True)
    metric_colors = {
        "alp_240h": COLORS["alp"],
        "collagen_336h": COLORS["collagen"],
        "ha_672h": COLORS["ha"],
        "final_live": COLORS["ink"],
        "final_preob": COLORS["preob"],
        "final_ob": COLORS["ob"],
    }
    condition_indices = sorted(replicate_summary["condition_index"].unique().tolist())
    labels = [f"C{i:02d}" for i in condition_indices]

    for ax, metric in zip(axes.flat, PRIMARY_METRICS):
        color = metric_colors[metric]
        data = [
            replicate_summary.loc[replicate_summary["condition_index"] == idx, metric].astype(float).to_numpy()
            for idx in condition_indices
        ]
        bp = ax.boxplot(data, patch_artist=True, widths=0.62, showfliers=False)
        for i, box in enumerate(bp["boxes"], start=1):
            box.set(facecolor=color if i == 1 else "#cbd5e1", alpha=0.85, edgecolor=COLORS["ink"], linewidth=0.8)
        for median in bp["medians"]:
            median.set(color=COLORS["ink"], linewidth=1.2)
        for whisker in bp["whiskers"]:
            whisker.set(color=COLORS["ink"], linewidth=0.8)
        for cap in bp["caps"]:
            cap.set(color=COLORS["ink"], linewidth=0.8)

        for xpos, idx in enumerate(condition_indices, start=1):
            vals = replicate_summary.loc[replicate_summary["condition_index"] == idx, metric].astype(float).to_numpy()
            jitter_rng = np.random.default_rng(7000 + idx)
            jitter = jitter_rng.uniform(-0.12, 0.12, size=len(vals))
            ax.scatter(
                np.full(len(vals), xpos) + jitter,
                vals,
                s=28,
                color=COLORS["accent"] if idx == 1 else color,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.35,
                zorder=4,
            )

        subset = stats_df[stats_df["metric"] == metric]
        ymax = max(float(np.max(vals)) for vals in data)
        ymin = min(float(np.min(vals)) for vals in data)
        ypad = max((ymax - ymin) * 0.08, 0.5)
        for _, row in subset.iterrows():
            idx = int(row["condition_index"])
            if idx == 1:
                continue
            symbol = significance_symbol(float(row["p_adj"]))
            if not symbol:
                continue
            ax.text(idx, ymax + ypad, symbol, ha="center", va="bottom", fontsize=9.5, color=COLORS["ink"])

        global_p = subset["global_p"].dropna()
        global_text = f"Kruskal-Wallis p={global_p.iloc[0]:.3g}" if not global_p.empty else "Kruskal-Wallis p=NA"
        style_axes(ax, grid_axis="y")
        ax.set_xticks(np.arange(1, len(labels) + 1), labels, rotation=45, ha="right")
        ax.set_title(metric_label(metric))
        ax.text(0.02, 0.98, global_text, transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color=COLORS["muted"])
        ax.set_xlim(0.3, len(labels) + 0.7)
        ax.set_ylim(ymin - ypad * 0.4, ymax + ypad * 1.8)

    fig.suptitle("Endpoint distributions with statistical comparison versus control", fontsize=15, y=1.01)
    save_figure(fig, out_dir, "03_endpoint_bars")


def export_final_state_gallery(results: list[ConditionResult], out_dir: Path, total_hours: int) -> None:
    final_day = total_hours // HOURS_PER_DAY
    fig, axes = plt.subplots(4, 5, figsize=(16.0, 12.0), constrained_layout=True)
    vmax = max(
        float(np.load(result.replicates[0].out_dir / "grids" / f"ha_day_{final_day:03d}.npy").max())
        for result in results
    )
    vmax = max(vmax, 1e-9)
    for ax, result in zip(axes.flat, results):
        state = np.load(result.replicates[0].out_dir / "grids" / f"state_day_{final_day:03d}.npy")
        ha = np.load(result.replicates[0].out_dir / "grids" / f"ha_day_{final_day:03d}.npy")
        yy, xx = np.where(state == 1)
        oy, ox = np.where(state == 2)
        ax.imshow(ha, origin="upper", cmap="magma", vmin=0.0, vmax=vmax)
        if xx.size:
            ax.scatter(xx, yy, s=9, c="#67e8f9", edgecolors="none", alpha=0.95)
        if ox.size:
            ax.scatter(ox, oy, s=11, c="#fde68a", edgecolors="none", alpha=0.98)
        ax.set_title(f"C{result.spec.index:02d} {result.spec.slug}", loc="left", fontsize=9.4)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
    fig.suptitle("Representative final spatial states with HAp field background", fontsize=15, y=1.01)
    save_figure(fig, out_dir, "04_final_state_gallery")


def export_endpoint_heatmap(condition_summary: pd.DataFrame, out_dir: Path) -> None:
    metric_cols = [
        "alp_240h_mean",
        "collagen_336h_mean",
        "ha_672h_mean",
        "final_alp_mean",
        "final_collagen_mean",
        "final_ha_mean",
        "final_preob_mean",
        "final_ob_mean",
        "final_live_mean",
    ]
    baseline = condition_summary.iloc[0][metric_cols].astype(float)
    delta = condition_summary[metric_cols].astype(float).subtract(baseline, axis=1)
    rel = delta.divide(baseline.replace(0.0, np.nan), axis=1) * 100.0
    rel = rel.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    fig, ax = plt.subplots(figsize=(12.2, 6.6), constrained_layout=True)
    vmax = max(5.0, float(np.nanmax(np.abs(rel.to_numpy()))))
    cmap = LinearSegmentedColormap.from_list("balance", ["#1d4ed8", "#f8fafc", "#b91c1c"])
    im = ax.imshow(rel.to_numpy(), aspect="auto", cmap=cmap, norm=TwoSlopeNorm(vmin=-vmax, vcenter=0.0, vmax=vmax))
    ax.set_xticks(np.arange(len(metric_cols)), [c.replace("_mean", "") for c in metric_cols], rotation=30, ha="right")
    ax.set_yticks(np.arange(len(condition_summary)), [f"C{i:02d}" for i in condition_summary["condition_index"]])
    ax.set_title("Relative deviation from the control (%)")
    for row in range(rel.shape[0]):
        for col in range(rel.shape[1]):
            ax.text(col, row, f"{rel.iat[row, col]:+.1f}", ha="center", va="center", fontsize=7.0, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("Percent difference vs C01 mean")
    save_figure(fig, out_dir, "05_endpoint_heatmap")


def export_layout_vs_response(condition_summary: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.6, 5.6), constrained_layout=True)
    x = condition_summary["seed_mean_radius_mean"].to_numpy()
    y = condition_summary["final_ha_mean"].to_numpy()
    colors = condition_summary["seed_edge_fraction_mean"].to_numpy()
    sc = ax.scatter(
        x,
        y,
        c=colors,
        cmap="viridis",
        s=95,
        edgecolors="white",
        linewidths=0.7,
        alpha=0.95,
    )
    for _, row in condition_summary.iterrows():
        ax.text(row["seed_mean_radius_mean"] + 0.05, row["final_ha_mean"], f"C{int(row['condition_index']):02d}", fontsize=8.0)
    style_axes(ax, grid_axis="both")
    ax.set_xlabel("Initial mean radius from center")
    ax.set_ylabel("Final HAp mean")
    ax.set_title("Initial compactness versus final mineralization")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.032, pad=0.025)
    cbar.set_label("Initial edge fraction mean")
    save_figure(fig, out_dir, "06_layout_vs_response")


def export_significance_heatmap(stats_df: pd.DataFrame, out_dir: Path) -> None:
    pivot = stats_df.pivot(index="condition_index", columns="metric", values="p_adj")
    pivot = pivot.reindex(index=sorted(pivot.index), columns=PRIMARY_METRICS)
    arr = -np.log10(pivot.clip(lower=1e-12).to_numpy())
    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    im = ax.imshow(arr, aspect="auto", cmap="YlOrRd")
    ax.set_xticks(np.arange(len(PRIMARY_METRICS)), [metric_label(m) for m in PRIMARY_METRICS], rotation=35, ha="right")
    ax.set_yticks(np.arange(len(pivot.index)), [f"C{i:02d}" for i in pivot.index])
    ax.set_title("Significance strength versus control (-log10 adjusted p)")
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            p_adj = pivot.iat[i, j]
            text = "ctrl" if pd.isna(p_adj) else f"{p_adj:.3f}"
            ax.text(j, i, text, ha="center", va="center", fontsize=7.1, color=COLORS["ink"])
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("-log10 adjusted p")
    save_figure(fig, out_dir, "07_significance_heatmap")


def coefficient_of_variation(values: pd.Series) -> float:
    arr = values.astype(float).to_numpy()
    mean = float(np.mean(arr))
    if abs(mean) < 1e-12:
        return 0.0
    return float(np.std(arr, ddof=0) / mean)


def classify_similarity(delta_pct: float) -> str:
    mag = abs(delta_pct)
    if mag < 5.0:
        return "nearly unchanged"
    if mag < 15.0:
        return "moderately shifted"
    return "strongly shifted"


def explain_condition(row: pd.Series, base: pd.Series, stats_slice: pd.DataFrame) -> str:
    d_ha = 100.0 * (row["final_ha_mean"] - base["final_ha_mean"]) / max(base["final_ha_mean"], 1e-12)
    d_live = 100.0 * (row["final_live_mean"] - base["final_live_mean"]) / max(base["final_live_mean"], 1e-12)
    similarity = classify_similarity(d_ha)

    descriptors: list[str] = []
    if row["seed_center_fraction_mean"] > 0.6:
        descriptors.append("started as a compact central inoculum")
    elif row["seed_edge_fraction_mean"] > 0.6:
        descriptors.append("started predominantly at the periodic edges")
    elif row["seed_left_fraction_mean"] < 0.25:
        descriptors.append("was biased to the right side")
    elif row["seed_left_fraction_mean"] > 0.75:
        descriptors.append("was biased to the left side")
    elif row["seed_top_fraction_mean"] < 0.25:
        descriptors.append("was biased to the bottom side")
    elif row["seed_top_fraction_mean"] > 0.75:
        descriptors.append("was biased to the top side")
    else:
        descriptors.append("was spatially distributed without a strong center bias")

    if row["seed_mean_nearest_neighbor_mean"] < base["seed_mean_nearest_neighbor_mean"] * 0.8:
        descriptors.append("with higher initial crowding")
    elif row["seed_mean_nearest_neighbor_mean"] > base["seed_mean_nearest_neighbor_mean"] * 1.2:
        descriptors.append("with larger initial spacing")

    significant_metrics = []
    for metric in PRIMARY_METRICS:
        metric_rows = stats_slice[(stats_slice["metric"] == metric) & (stats_slice["condition_index"] == int(row["condition_index"]))]
        if not metric_rows.empty and bool(metric_rows["significant_vs_control"].iloc[0]):
            significant_metrics.append(metric_label(metric))
    stat_text = "No endpoint remained significant versus control after Holm correction." if not significant_metrics else "Significant versus control for: " + ", ".join(significant_metrics) + "."

    mechanism = (
        "periodic boundaries and random movement rapidly mix positional asymmetries"
        if row["seed_edge_fraction_mean"] > 0.45
        or abs(row["seed_left_fraction_mean"] - 0.5) > 0.35
        or abs(row["seed_top_fraction_mean"] - 0.5) > 0.35
        else "the coupled proliferation, commitment, secretion, and field integration smooth early geometric differences over weeks"
    )

    return (
        f"{row['title']} {', '.join(descriptors)}. "
        f"Mean final HAp was {similarity} relative to the control ({d_ha:+.1f}%), "
        f"while mean final live cells changed by {d_live:+.1f}%. "
        f"{stat_text} This suggests {mechanism}."
    )


def write_reports(condition_summary: pd.DataFrame, replicate_summary: pd.DataFrame, stats_df: pd.DataFrame, out_dir: Path) -> None:
    base = condition_summary.iloc[0]
    cv_alp = 100.0 * coefficient_of_variation(condition_summary["alp_240h_mean"])
    cv_col = 100.0 * coefficient_of_variation(condition_summary["collagen_336h_mean"])
    cv_ha = 100.0 * coefficient_of_variation(condition_summary["ha_672h_mean"])
    cv_live = 100.0 * coefficient_of_variation(condition_summary["final_live_mean"])

    lines = [
        "# Sanity Check 2 Report",
        "",
        "## Scope",
        "",
        f"This suite evaluates {len(condition_summary)} initialization modes with {int(replicate_summary['replicate_index'].max())} replicates each.",
        "The biological parameters are fixed; only the initialization geometry and replicate-level stochasticity vary.",
        "",
        "## Cohort-level interpretation",
        "",
        f"- ALP at 240 h coefficient of variation across condition means: {_fmt(cv_alp)}%",
        f"- Collagen at 336 h coefficient of variation across condition means: {_fmt(cv_col)}%",
        f"- HAp at 672 h coefficient of variation across condition means: {_fmt(cv_ha)}%",
        f"- Final live cell count coefficient of variation across condition means: {_fmt(cv_live)}%",
        "- Statistical testing uses a Kruskal-Wallis omnibus test per metric and Mann-Whitney U tests versus the control with Holm correction.",
        "",
        "## Condition-by-condition interpretation",
        "",
    ]

    for _, row in condition_summary.iterrows():
        if int(row["condition_index"]) == 1:
            lines.append(
                f"- {row['title']}: control condition. Mean final HAp = {_fmt(row['final_ha_mean'])} +/- {_fmt(row['final_ha_sd'])}, "
                f"mean final collagen = {_fmt(row['final_collagen_mean'])} +/- {_fmt(row['final_collagen_sd'])}, "
                f"mean final ALP = {_fmt(row['final_alp_mean'])} +/- {_fmt(row['final_alp_sd'])}, "
                f"mean final preOB/OB = {_fmt(row['final_preob_mean'])}/{_fmt(row['final_ob_mean'])}."
            )
            continue
        lines.append(f"- {explain_condition(row, base, stats_df)}")

    lines.extend(
        [
            "",
            "## Endpoint table",
            "",
            "| Condition | ALP 240 h | Collagen 336 h | HAp 672 h | Final ALP | Final Collagen | Final HAp | Final preOB | Final OB | Final live |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for _, row in condition_summary.iterrows():
        lines.append(
            f"| C{int(row['condition_index']):02d} `{row['slug']}` | {_fmt(row['alp_240h_mean'])} +/- {_fmt(row['alp_240h_sd'])} | {_fmt(row['collagen_336h_mean'])} +/- {_fmt(row['collagen_336h_sd'])} | {_fmt(row['ha_672h_mean'])} +/- {_fmt(row['ha_672h_sd'])} | {_fmt(row['final_alp_mean'])} +/- {_fmt(row['final_alp_sd'])} | {_fmt(row['final_collagen_mean'])} +/- {_fmt(row['final_collagen_sd'])} | {_fmt(row['final_ha_mean'])} +/- {_fmt(row['final_ha_sd'])} | {_fmt(row['final_preob_mean'])} +/- {_fmt(row['final_preob_sd'])} | {_fmt(row['final_ob_mean'])} +/- {_fmt(row['final_ob_sd'])} | {_fmt(row['final_live_mean'])} +/- {_fmt(row['final_live_sd'])} |"
        )

    lines.extend(
        [
            "",
            "## Statistical summary",
            "",
            "| Metric | Omnibus p | Significant vs control after Holm |",
            "| --- | ---: | --- |",
        ]
    )
    for metric in PRIMARY_METRICS:
        subset = stats_df[stats_df["metric"] == metric]
        global_p = subset["global_p"].dropna()
        sig_conditions = [f"C{int(idx):02d}" for idx in subset.loc[subset["significant_vs_control"], "condition_index"].tolist()]
        global_p_text = _fmt_p(float(global_p.iloc[0])) if not global_p.empty else "NA"
        lines.append(
            f"| {metric_label(metric)} | {global_p_text} | {', '.join(sig_conditions) if sig_conditions else 'none'} |"
        )

    lines.extend(
        [
            "",
            "## Pairwise versus control",
            "",
            "| Metric | Condition | Mean delta | Percent delta | Adjusted p |",
            "| --- | --- | ---: | ---: | ---: |",
        ]
    )
    for _, row in stats_df[stats_df["condition_index"] != 1].iterrows():
        p_adj_text = _fmt_p(float(row["p_adj"])) if not pd.isna(row["p_adj"]) else "NA"
        lines.append(
            f"| {row['metric_label']} | C{int(row['condition_index']):02d} `{row['slug']}` | {_fmt(float(row['delta_vs_control']))} | {float(row['pct_vs_control']):+.1f}% | {p_adj_text} |"
        )

    (out_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")


def write_statistics_tables(stats_df: pd.DataFrame, out_dir: Path) -> None:
    stats_df.to_csv(out_dir / "statistics.csv", index=False)
    lines = [
        "# Sanity Check 2 Statistics",
        "",
        "| Metric | Condition | Mean | SD | Delta vs control | Percent vs control | U | p | p_adj | Significant |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for _, row in stats_df.iterrows():
        u_text = _fmt(float(row["u_stat"])) if not pd.isna(row["u_stat"]) else "NA"
        p_text = _fmt_p(float(row["p_value"])) if not pd.isna(row["p_value"]) else "NA"
        p_adj_text = _fmt_p(float(row["p_adj"])) if not pd.isna(row["p_adj"]) else "NA"
        lines.append(
            f"| {row['metric_label']} | C{int(row['condition_index']):02d} `{row['slug']}` | {_fmt(float(row['mean']))} | {_fmt(float(row['sd']))} | {_fmt(float(row['delta_vs_control']))} | {float(row['pct_vs_control']):+.1f}% | {u_text} | {p_text} | {p_adj_text} | {'yes' if bool(row['significant_vs_control']) else 'no'} |"
        )
    (out_dir / "statistics.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    abm_cfg, field_cfg = build_configs(args.w, args.h)
    specs = make_condition_specs(args.seed)
    results = []

    for spec in specs:
        print(f"[run] {spec.title}")
        result = run_condition(
            spec=spec,
            out_root=args.out,
            abm_cfg=abm_cfg,
            field_cfg=field_cfg,
            model_seed=args.seed,
            total_hours=args.hours,
            n_cells=args.cells,
            width=args.w,
            height=args.h,
            replicates=args.replicates,
            show_runtime=args.runtime,
            overwrite=args.overwrite,
        )
        results.append(result)

    replicate_summary = build_replicate_summary(results)
    replicate_summary.to_csv(args.out / "replicate_summary.csv", index=False)

    (args.out / "manifest.json").write_text(
        json.dumps(
            {
                "out_dir": args.out.as_posix(),
                "hours": args.hours,
                "cells": args.cells,
                "grid": {"w": args.w, "h": args.h},
                "model_seed": args.seed,
                "replicates": args.replicates,
                "alp_hours": ALP_HOURS,
                "collagen_hours": COLLAGEN_HOURS,
                "mineral_hours": MINERAL_HOURS,
                "conditions": [asdict(spec) for spec in specs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Wrote {args.out / 'replicate_summary.csv'}")
    print("Analysis and figure extraction now live in figure_extract_2.py")


if __name__ == "__main__":
    main()
