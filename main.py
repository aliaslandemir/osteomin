from __future__ import annotations

import argparse
from pathlib import Path


from src.model import OsteoModel
from src.config import PDEConfig, ABMConfig
from src.viz import plot_summary, export_single_panels


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

    # Output
    p.add_argument("--out", type=str, default="out", help="output directory")
    p.add_argument("--xlsx", action="store_true", help="write Excel timeseries")
    p.add_argument("--snapshots", action="store_true", help="export per-panel figures")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    # Configs
    pde_cfg = PDEConfig(width=100, height=100)
    abm_cfg = ABMConfig(
        width=100,
        height=100,
        enable_migration=bool(args.cell_movement),
        enable_commitment=bool(args.commitment),
        enable_apoptosis=bool(args.apoptosis),
        enable_proliferation=bool(args.proliferation),
        enable_secretion=bool(args.secretion),
    )

    supplements = {"aa": args.aa, "bgp": args.bgp, "dexa": args.dexa, "mk4": args.mk4}

    model = OsteoModel(
        pde_cfg=pde_cfg,
        abm_cfg=abm_cfg,
        days=args.days,
        media_interval=args.media_interval,
        rng_seed=args.seed,
        supplements=supplements,
        outdir=outdir,
    )

    # Seed & run
    model.seed_random(args.cells)
    df = model.run()


    if args.xlsx:
        model.save_timeseries_excel(df, "timeseries")

    # Plots
    plot_summary(df, outdir)
    if args.snapshots:
        export_single_panels(df, outdir)

    # Log tail
    print(df.tail(5))


if __name__ == "__main__":
    main()
