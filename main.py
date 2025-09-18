from __future__ import annotations

import argparse
from pathlib import Path

from src.model import OsteoModel
from src.config import PDEConfig, ABMConfig
from src.viz import plot_summary


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Osteogenic ABM/PDE runner")

    # Core sim settings
    p.add_argument("--days", type=int, default=21, help="number of days to simulate")
    p.add_argument("--cells", type=int, default=300, help="initial number of cells")
    p.add_argument("--media-interval", type=int, default=2, help="days between media changes")
    p.add_argument("--seed", type=int, default=None, help="RNG seed (int)")

    # Supplements (unitless scalars)
    p.add_argument("--aa", type=float, default=1.0, help="ascorbic acid scaling")
    p.add_argument("--bgp", type=float, default=1.0, help="beta-glycerophosphate scaling")
    p.add_argument("--dexa", type=float, default=1.0, help="dexamethasone scaling")
    p.add_argument("--mk4", type=float, default=1.0, help="menaquinone-4 scaling")

    # Switches
    p.add_argument("--cell-movement", dest="cell_movement", action="store_true", help="enable cell migration")
    p.add_argument("--no-cell-movement", dest="cell_movement", action="store_false", help="disable cell migration")
    p.set_defaults(cell_movement=True)


    p.add_argument("--move-prob", type=float, default=0.8, help="daily probability an ACTIVE cell attempts a move")
    p.add_argument("--max-moves", type=int,   default=1,   help="max random-walk steps per ACTIVE cell per day")


    p.add_argument("--osteogenic-commitment", dest="commitment", action="store_true", help="enable osteogenic commitment")
    p.add_argument("--no-osteogenic-commitment", dest="commitment", action="store_false", help="disable osteogenic commitment")
    p.set_defaults(commitment=True)

    p.add_argument("--apoptosis", dest="apoptosis", action="store_true", help="enable apoptosis")
    p.add_argument("--no-apoptosis", dest="apoptosis", action="store_false", help="disable apoptosis")
    p.set_defaults(apoptosis=True)

    p.add_argument("--proliferation", dest="proliferation", action="store_true", help="enable proliferation")
    p.add_argument("--no-proliferation", dest="proliferation", action="store_false", help="disable proliferation")
    p.set_defaults(proliferation=True)

    p.add_argument("--secretion", dest="secretion", action="store_true", help="enable secretion (ABM→PDE)")
    p.add_argument("--no-secretion", dest="secretion", action="store_false", help="disable secretion (ABM→PDE)")
    p.set_defaults(secretion=True)

    # Spatial export control
    p.add_argument("--spatial-every", type=int, default=0,
                   help="export spatial PNG every N days (0 = off)")
    p.add_argument("--spatial-days", type=str, default="",
                   help="comma-separated explicit days to export, e.g. '0,5,10,20'")

    # Output
    p.add_argument("--out", type=str, default="out", help="output directory")
    p.add_argument("--xlsx", action="store_true", help="write Excel timeseries")
    p.add_argument("--snapshots", action="store_true", help="export simple state/density/deaths panels for last day")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Parse explicit spatial days (dedupe + sort later in model)
    explicit_spatial_days = []
    if args.spatial_days.strip():
        explicit_spatial_days = [int(s) for s in args.spatial_days.split(",") if s.strip()]

    # Configs
    pde_cfg = PDEConfig(width=100, height=100, media_interval=int(args.media_interval))
    abm_cfg = ABMConfig(
        width=100,
        height=100,
        enable_migration=bool(args.cell_movement),
        enable_commitment=bool(args.commitment),
        enable_apoptosis=bool(args.apoptosis),
        enable_proliferation=bool(args.proliferation),
        enable_secretion=bool(args.secretion),
        move_prob=float(args.move_prob),
        max_moves_per_day=int(args.max_moves)
    )

    supplements = {"aa": args.aa, "bgp": args.bgp, "dexa": args.dexa, "mk4": args.mk4}

    model = OsteoModel(
        days=args.days,
        outdir=outdir,
        abm_cfg=abm_cfg,
        pde_cfg=pde_cfg,
        supplements=supplements,
        seed=args.seed,
        spatial_every=args.spatial_every,
        spatial_days=explicit_spatial_days,
    )

    # Seed & run
    model.seed_random(args.cells, abm_cfg.width, abm_cfg.height)
    df = model.run()

    # Enrich DF with cumulative series for plots
    df["dead_total_today"] = df["dead_active_today"] + df["dead_ob_today"]
    df["dead_active_cum"] = df["dead_active_today"].cumsum()
    df["dead_ob_cum"]     = df["dead_ob_today"].cumsum()
    df["dead_total_cum"]  = df["dead_total_today"].cumsum()

    if args.xlsx:
        model.save_timeseries_excel(df, "timeseries")

    # Plots
    plot_summary(df, outdir)

    # Log tail
    print(df.tail(5))


if __name__ == "__main__":
    main()
