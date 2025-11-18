"""
main.py — Entry point for OsteoMin ABM–field bone remodeling simulator.

Run with:
    python main.py --days 30 --cells 400 --w 33 --h 33 --xlsx --snapshots
"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from src.model import OsteoModel
from src.config import FieldConfig, ABMConfig
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


# ----------------------------------------------------------------------
# Argument parser
# ----------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OsteoMin: MC3T3-E1 ABM/field bone remodeling simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # --- core simulation ---
    p.add_argument("--days", type=int, default=30, help="number of days to simulate")
    p.add_argument(
        "--cells", type=int, default=72, help="initial number of seeded cells"
    )
    p.add_argument(
        "--media-interval", type=int, default=2, help="days between media changes"
    )
    p.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducibility"
    )

    # --- lattice size ---
    p.add_argument("--w", type=int, default=33, help="grid width")
    p.add_argument("--h", type=int, default=33, help="grid height")
    p.add_argument(
        "--boundary-mode",
        choices=["clamp", "periodic"],
        default="periodic",
        help="treat edges as hard boundaries (clamp) or wrap around (periodic)",
    )

    # --- supplements (scalars) ---
    p.add_argument("--aa", type=float, default=1.0, help="ascorbic acid scaling")
    p.add_argument(
        "--bgp", type=float, default=1.0, help="beta-glycerophosphate scaling"
    )
    p.add_argument("--mk4", type=float, default=0.0, help="MK-4 dose (0–1); boosts collagen/ECM build-up")
    p.add_argument(
        "--dex",
        type=float,
        default=None,
        help="dexamethasone dose for commitment boost (0–1); absent = no Dex effect",
    )
    p.add_argument(
        "--dex-k",
        type=float,
        default=1.0,
        help="strength of dexamethasone effect on commitment (F_dex = 1 + k * D)",
    )

    # --- cell-level switches ---
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
        help="probability a preOB cell attempts a move per day",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=1,
        help="max random-walk steps per preOB cell per day",
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
        default=0.01,
        help="per-day increase in osteogenic commitment probability",
    )

    p.add_argument(
        "--commit-delay",
        type=int,
        default=3,
        help="days before preOB cells can start committing to osteoblasts",
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
        default=0.01,
        help="per-cell apoptosis probability each day",
    )

    p.add_argument(
        "--apoptosis-osteo",
        dest="ob_apoptosis_prob",
        type=float,
        default=0.001,
        help="per-osteoblast apoptosis probability each day",
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
        help="per-cell chance to attempt division once the interval passes",
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

    # --- mineralization parameters (new) ---
    p.add_argument(
        "--s-thresh",
        type=float,
        default=0.01,
        help="supersaturation threshold for ACP nucleation",
    )
    p.add_argument(
        "--k-nuc", type=float, default=0.025, help="ACP nucleation rate per day"
    )
    p.add_argument(
        "--k-ha", type=float, default=0.06, help="ACP → HA maturation rate per day"
    )
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
        "--nucleation",
        dest="nucleation",
        action="store_true",
        help="enable Pi→ACP nucleation and maturation to HA",
    )
    p.add_argument(
        "--no-nucleation",
        dest="nucleation",
        action="store_false",
        help="disable Pi→ACP nucleation and maturation to HA",
    )
    p.set_defaults(nucleation=True)

    # --- export control ---
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

    return p.parse_args()


# ----------------------------------------------------------------------
# Main entry point
# ----------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # clamp probabilities for safety
    args.move_prob = max(0.0, min(1.0, args.move_prob))
    args.ob_div_prob = max(0.0, min(1.0, args.ob_div_prob))
    args.proliferation_chance = max(0.0, min(1.0, args.proliferation_chance))
    args.apoptosis_prob = max(0.0, min(1.0, args.apoptosis_prob))
    args.ob_apoptosis_prob = max(0.0, min(1.0, args.ob_apoptosis_prob))
    args.commit_delay = max(0, args.commit_delay)
    dex_dose = args.dex if args.dex is not None else 0.0
    dex_dose = max(0.0, min(1.0, dex_dose))
    dex_k = max(0.0, args.dex_k)

    # configs
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
        max_moves_per_day=args.max_moves,
        ob_division_prob=args.ob_div_prob,
        apoptosis_prob=args.apoptosis_prob,
        ob_apoptosis_prob=args.ob_apoptosis_prob,
        proliferation_chance=args.proliferation_chance,
        commit_prob_growth=args.commit_growth,
        commit_delay=args.commit_delay,
        dex_dose=dex_dose,
        dex_k=dex_k,
    )
    field_cfg = FieldConfig(
        width=args.w,
        height=args.h,
        media_interval=args.media_interval,
        enable_nucleation=args.nucleation,
        enable_nodules=args.nodules,
        s_thresh=args.s_thresh,
        k_nuc=args.k_nuc,
        k_ha=args.k_ha,
        nodule_window=max(1, int(args.nodule_window)),
    )

    # supplements
    supplements = dict(aa=args.aa, bgp=args.bgp, mk4=max(0.0, min(1.0, args.mk4)))

    # explicit spatial export days
    spatial_days = [int(x) for x in args.spatial_days.split(",") if x.strip()]

    # build model
    model = OsteoModel(
        outdir=outdir,
        days=args.days,
        abm_cfg=abm_cfg,
        field_cfg=field_cfg,
        supplements=supplements,
        seed=args.seed,
        spatial_every=args.spatial_every,
        spatial_days=spatial_days,
        snapshots_enabled=args.snapshots,
        unique_id_tracking=args.unique_id,
    )

    # inject mineralization params into the field manager
    # seed and run
    model.seed_random(args.cells, abm_cfg.width, abm_cfg.height)
    df = model.run()

    # enrich DF with cumulative series for plots
    df["dead_total_today"] = df["dead_preob_today"] + df["dead_ob_today"]
    df["dead_preob_cum"] = df["dead_preob_today"].cumsum()
    df["dead_ob_cum"] = df["dead_ob_today"].cumsum()
    df["dead_total_cum"] = df["dead_total_today"].cumsum()

    # save timeseries
    if args.xlsx:
        model.save_timeseries_excel(df, "timeseries")
    if args.csv:
        model.save_timeseries_csv(df, "timeseries")
    if args.unique_id:
        model.save_cell_history_excel("cell_history")

    # plots
    plot_summary(df, outdir)

    # overlays driven by spatial export policy (frequency + explicit days)
    overlay_days = sorted({d for d in model.spatial_days if 0 <= d < args.days})
    if args.spatial_every > 0:
        overlay_days = sorted(
            {*overlay_days, *{day for day in range(0, args.days, args.spatial_every)}}
        )
    if not args.snapshots:
        print("snapshots disabled; skipping collagen/HA overlays")
    else:
        all_days = range(args.days)
        collagen_vmax = _global_field_max(model.grids_dir, "collagen", all_days)
        ha_vmax = _global_field_max(model.grids_dir, "ha", all_days)
        for day in overlay_days:
            export_overlay_collagen_ha_from_npy(
                model.grids_dir,
                outdir,
                day,
                global_vmax_col=collagen_vmax,
                global_vmax_ha=ha_vmax,
            )

    # log tail to console
    print(df.tail(5))


if __name__ == "__main__":
    main()
