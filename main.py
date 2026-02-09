"""
main.py — entry point for the OsteoMin ABM/field simulator.
Hourly inputs are supported; defaults mirror README.md.


python main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode grid \
  --grid-spacing 3 \
  --seed $(shuf -i 1-999999 -n 1) \
  --w 33 --h 33 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 --apoptosis-osteo 0.000042 \
  --osteogenic-commitment \
  --commit-base 0.0011 \
  --commit-growth 0.00011 \
  --commit-delay 72 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion --enable-nodules \
  --spatial-every 1 --out main_out \
  --nodule-window 5 \
  --datafit datafitting/datafit.npz \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports
  

python main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed $(shuf -i 1-999999 -n 1) \
  --w 33 --h 33 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 --apoptosis-osteo 0.000042 \
  --osteogenic-commitment \
  --commit-base 0.0011 \
  --commit-growth 0.00011 \
  --commit-delay 72 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion --enable-nodules \
  --spatial-every 1 --out main_out \
  --nodule-window 5 \
  --datafit datafitting/endpointresults/fit_outputs/datafit_endpoint.npz \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --datafit-all



python main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed 12345 \
  --w 33 --h 33 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 1.0 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 --apoptosis-osteo 0.000042 \
  --osteogenic-commitment \
  --commit-base 0.0011 \
  --commit-growth 0.00011 \
  --commit-delay 72 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion --enable-nodules \
  --spatial-every 1 --out main_out \
  --nodule-window 5 \
  --datafit datafitting/endpointresults/fit_outputs/datafit_endpoint.npz \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --datafit-all
  

  

"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from src.model import OsteoModel
from src.config import FieldConfig, ABMConfig
from src.time_utils import HOURS_PER_DAY
from src.viz import plot_summary, export_overlay_collagen_ha_from_npy


def _global_field_max(grids_dir: Path, field_prefix: str, days: Iterable[int]) -> float:
    """Compute the maximum value across stored grids for `field_prefix`."""
    max_val = 0.0
    for day in days:
        path = grids_dir / f"{field_prefix}_day_{day:03d}.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        if arr.size == 0:
            continue
        max_val = max(max_val, float(np.nanmax(arr)))
    return max(max_val, 1e-9)


def _grid_seed_coords(w: int, h: int, spacing: int) -> list[tuple[int, int]]:
    if spacing <= 0:
        raise ValueError("grid spacing must be >= 1")
    coords = []
    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            coords.append((x, y))
    return coords


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OsteoMin: MC3T3-E1 ABM/field bone remodeling simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--hours",
        type=int,
        default=None,
        help="total hours to simulate (24 hours per day is assumed)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="legacy alias; multiplied by 24 if --hours is omitted",
    )
    p.add_argument(
        "--cells", type=int, default=72, help="initial number of seeded cells"
    )
    p.add_argument(
        "--seed-mode",
        choices=["random", "grid"],
        default="random",
        help="initial seeding strategy: random uniform or grid-spaced then sampled",
    )
    p.add_argument(
        "--grid-spacing",
        type=int,
        default=3,
        help="grid spacing (n) for grid seeding; one candidate site per n×n block",
    )
    p.add_argument(
        "--media-interval",
        type=int,
        default=48,
        help="hours between media changes (legacy 2 days = 48 hours)",
    )
    p.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )

    p.add_argument("--w", type=int, default=33, help="grid width")
    p.add_argument("--h", type=int, default=33, help="grid height")
    p.add_argument(
        "--boundary-mode",
        choices=["clamp", "periodic"],
        default="periodic",
        help="treat edges as hard boundaries (clamp) or wrap around (periodic)",
    )

    p.add_argument(
        "--random-move",
        "--cell-movement",
        dest="random_move",
        action="store_true",
        help="enable random_move motility",
    )
    p.add_argument(
        "--no-random-move",
        "--no-cell-movement",
        dest="random_move",
        action="store_false",
        help="disable random_move motility",
    )
    p.set_defaults(random_move=True)

    p.add_argument(
        "--move-prob",
        type=float,
        default=1.0,
        help="probability a preOB cell attempts a move per hour",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=1,
        help="max random-walk steps per preOB cell per hour",
    )
    p.add_argument(
        "--min-div-hours",
        type=int,
        default=ABMConfig.min_div_hours,
        help="minimum hours between divisions for a cell",
    )
    p.add_argument(
        "--max-div-hours",
        type=int,
        default=ABMConfig.max_div_hours,
        help="maximum hours between divisions for a cell",
    )

    p.add_argument(
        "--ob-div-prob",
        type=float,
        default=0.0,
        help="rare division probability for osteoblasts",
    )

    p.add_argument(
        "--osteogenic-commitment",
        dest="commitment",
        action="store_true",
        help="enable preOB → OB differentiation",
    )
    p.add_argument(
        "--no-osteogenic-commitment",
        dest="commitment",
        action="store_false",
        help="disable preOB → OB differentiation",
    )
    p.set_defaults(commitment=True)

    p.add_argument(
        "--commit-growth",
        type=float,
        default=0.01 / (24 * 24),
        help="per-hour increase in osteogenic commitment probability (legacy 0.01/day slope)",
    )
    p.add_argument(
        "--commit-base",
        type=float,
        default=ABMConfig.base_commit_prob,
        help="base per-hour osteogenic commitment probability",
    )

    p.add_argument(
        "--commit-delay",
        type=int,
        default=72,
        help="hours before preOB cells can start committing to osteoblasts (legacy 3 days)",
    )

    p.add_argument(
        "--apoptosis", dest="apoptosis", action="store_true", help="enable apoptosis"
    )
    p.add_argument(
        "--no-apoptosis",
        dest="apoptosis",
        action="store_false",
        help="disable apoptosis",
    )
    p.set_defaults(apoptosis=True)

    p.add_argument(
        "--apoptosis-prob",
        type=float,
        default=0.01 / 24,
        help="per-cell apoptosis probability each hour (legacy 0.01/day)",
    )

    p.add_argument(
        "--apoptosis-osteo",
        dest="ob_apoptosis_prob",
        type=float,
        default=0.001 / 24,
        help="per-osteoblast apoptosis probability each hour (legacy 0.001/day)",
    )

    p.add_argument(
        "--proliferation",
        dest="proliferation",
        action="store_true",
        help="enable proliferation",
    )
    p.add_argument(
        "--no-proliferation",
        dest="proliferation",
        action="store_false",
        help="disable proliferation",
    )
    p.set_defaults(proliferation=True)

    p.add_argument(
        "--proliferation-chance",
        type=float,
        default=1.0,
        help="per-cell chance to attempt division per hour once the interval passes",
    )

    p.add_argument(
        "--secretion",
        dest="secretion",
        action="store_true",
        help="enable secretion (OB → collagen/ALP)",
    )
    p.add_argument(
        "--no-secretion",
        dest="secretion",
        action="store_false",
        help="disable secretion",
    )
    p.set_defaults(secretion=True)

    p.add_argument(
        "--enable-nodules",
        dest="nodules",
        action="store_true",
        help="detect nodules via window scan",
    )
    p.add_argument(
        "--no-nodules",
        dest="nodules",
        action="store_false",
        help="disable nodule scanning/overlays",
    )
    p.set_defaults(nodules=True)
    p.add_argument(
        "--nodule-window",
        type=int,
        default=4,
        help="window size (in lattice cells) for nodule detection",
    )
    p.add_argument(
        "--spatial-every",
        type=int,
        default=1,
        help="export spatial PNG every N days (0 = off)",
    )
    p.add_argument(
        "--spatial-days",
        type=str,
        default="",
        help="explicit export days, comma-separated e.g. '0,5,10,20'",
    )
    p.add_argument("--out", type=str, default="out", help="output directory")
    p.add_argument(
        "--xlsx", dest="xlsx", action="store_true", help="write Excel timeseries"
    )
    p.add_argument(
        "--no-xlsx",
        dest="xlsx",
        action="store_false",
        help="skip Excel timeseries export",
    )
    p.set_defaults(xlsx=True)
    p.add_argument(
        "--uniqueID",
        "--unique-id",
        dest="unique_id",
        action="store_true",
        help="track per-cell history via stable IDs and export it",
    )
    p.add_argument(
        "--no-unique-id",
        dest="unique_id",
        action="store_false",
        help="skip unique cell history export",
    )
    p.set_defaults(unique_id=True)
    p.add_argument("--csv", action="store_true", help="write CSV timeseries")
    p.add_argument(
        "--snapshots",
        dest="snapshots",
        action="store_true",
        help="also save raw .npy grids per day",
    )
    p.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="disable raw .npy grid dumps",
    )
    p.set_defaults(snapshots=True)
    p.add_argument(
        "--hourly-exports",
        dest="hourly_exports",
        action="store_true",
        help="write hourly spatial/collagen snapshots",
    )
    p.add_argument(
        "--no-hourly-exports",
        dest="hourly_exports",
        action="store_false",
        help="skip hourly snapshot exports",
    )
    p.set_defaults(hourly_exports=False)
    p.add_argument(
        "--datafit",
        type=str,
        default="datafitting/datafit.npz",
        help="path to datafit.npz bundle with hourly basal/osteo ALP, collagen, and mineral values",
    )
    p.add_argument(
        "--datafit-condition",
        type=str,
        default="C2",
        help="condition to use from datafit_endpoint.npz (C1..C8). Default C2.",
    )
    p.add_argument(
        "--datafit-all",
        action="store_true",
        help="run all conditions present in datafit_endpoint.npz (C1..C8) and write subfolders",
    )
    p.add_argument(
        "--alp-plot-hours",
        type=int,
        default=240,
        help="limit ALP plot to this many hours (0 = no limit)",
    )
    p.add_argument(
        "--collagen-plot-hours",
        type=int,
        default=336,
        help="limit collagen plot to this many hours (0 = no limit)",
    )
    p.add_argument(
        "--mineral-plot-hours",
        type=int,
        default=672,
        help="limit mineral (ACP/HA) plot to this many hours (0 = no limit)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    args.move_prob = max(0.0, min(1.0, args.move_prob))
    args.ob_div_prob = max(0.0, min(1.0, args.ob_div_prob))
    args.proliferation_chance = max(0.0, min(1.0, args.proliferation_chance))
    args.apoptosis_prob = max(0.0, min(1.0, args.apoptosis_prob))
    args.ob_apoptosis_prob = max(0.0, min(1.0, args.ob_apoptosis_prob))
    args.commit_delay = max(0, args.commit_delay)
    abm_cfg = ABMConfig(
        width=args.w,
        height=args.h,
        boundary_mode=args.boundary_mode,
        enable_random_move=args.random_move,
        enable_commitment=args.commitment,
        enable_apoptosis=args.apoptosis,
        enable_proliferation=args.proliferation,
        enable_secretion=args.secretion,
        move_prob=args.move_prob,
        max_moves_per_hour=args.max_moves,
        ob_division_prob=args.ob_div_prob,
        apoptosis_prob=args.apoptosis_prob,
        ob_apoptosis_prob=args.ob_apoptosis_prob,
        proliferation_chance=args.proliferation_chance,
        base_commit_prob=args.commit_base,
        commit_prob_growth=args.commit_growth,
        commit_delay=args.commit_delay,
        min_div_hours=max(1, int(args.min_div_hours)),
        max_div_hours=max(int(args.min_div_hours), int(args.max_div_hours)),
    )
    field_cfg = FieldConfig(
        width=args.w,
        height=args.h,
        media_interval=args.media_interval,
        enable_nodules=args.nodules,
        nodule_window=max(1, int(args.nodule_window)),
    )
    spatial_days = [int(x) for x in args.spatial_days.split(",") if x.strip()]

    if args.hours is not None:
        total_hours = max(0, int(args.hours))
    else:
        total_hours = max(0, int(args.days) * HOURS_PER_DAY)
    if total_hours <= 0:
        raise SystemExit("specify a positive --hours or --days value")

    datafit_path = Path(args.datafit) if args.datafit else None

    def _run_condition(cond: str, cond_outdir: Path) -> None:
        model = OsteoModel(
            outdir=cond_outdir,
            total_hours=total_hours,
            abm_cfg=abm_cfg,
            field_cfg=field_cfg,
            seed=args.seed,
            spatial_every=args.spatial_every,
            spatial_days=spatial_days,
            snapshots_enabled=args.snapshots,
            unique_id_tracking=args.unique_id,
            hourly_exports=args.hourly_exports,
            datafit_path=datafit_path,
            datafit_condition=cond,
        )

        if args.seed_mode == "random":
            model.seed_random(args.cells, abm_cfg.width, abm_cfg.height)
        else:
            sites = _grid_seed_coords(abm_cfg.width, abm_cfg.height, args.grid_spacing)
            if not sites:
                raise SystemExit(
                    "grid spacing produces no seed sites; adjust --grid-spacing"
                )
            n = min(int(args.cells), len(sites))
            idx = model.rng.choice(len(sites), size=n, replace=False)
            coords = [sites[i] for i in idx]
            model.abm.seed_from_coords(coords, day0=0)
        df = model.run()

        df["dead_total_today"] = df["dead_preob_today"] + df["dead_ob_today"]
        df["dead_preob_cum"] = df["dead_preob_today"].cumsum()
        df["dead_ob_cum"] = df["dead_ob_today"].cumsum()
        df["dead_total_cum"] = df["dead_total_today"].cumsum()

        if args.xlsx:
            model.save_timeseries_excel(df, "timeseries")
        if args.csv:
            model.save_timeseries_csv(df, "timeseries")
        if args.unique_id:
            model.save_cell_history_excel("cell_history")

        plot_summary(
            df,
            cond_outdir,
            alp_plot_hours=args.alp_plot_hours,
            collagen_plot_hours=args.collagen_plot_hours,
            mineral_plot_hours=args.mineral_plot_hours,
        )

        total_days = model.days
        overlay_days = sorted({d for d in model.spatial_days if 0 <= d < total_days})
        if args.spatial_every > 0:
            overlay_days = sorted(
                {
                    *overlay_days,
                    *{day for day in range(0, total_days, args.spatial_every)},
                }
            )
        if not args.snapshots:
            print("snapshots disabled; skipping collagen/HA overlays")
        else:
            all_days = range(total_days)
            collagen_vmax = _global_field_max(model.grids_dir, "collagen", all_days)
            ha_vmax = _global_field_max(model.grids_dir, "ha", all_days)
            for day in overlay_days:
                export_overlay_collagen_ha_from_npy(
                    model.grids_dir,
                    cond_outdir,
                    day,
                    global_vmax_col=collagen_vmax,
                    global_vmax_ha=ha_vmax,
                )

        print(f"[{cond}]")
        print(df.tail(5))

    if args.datafit_all and datafit_path and Path(datafit_path).exists():
        data = np.load(datafit_path)
        conds = (
            data["conditions"].tolist()
            if "conditions" in data.files
            else ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
        )
        for cond in conds:
            cond_out = outdir / str(cond)
            cond_out.mkdir(parents=True, exist_ok=True)
            _run_condition(str(cond), cond_out)
    else:
        _run_condition(args.datafit_condition, outdir)


if __name__ == "__main__":
    main()
