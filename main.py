"""
Entry point for MC3T3-E1 osteogenesis simulator (2D coverslip).
"""
from __future__ import annotations
import argparse

from src.model import OsteoModel
from src.config import PDEConfig, ABMConfig
from src.viz import plot_summary, export_single_panels


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=120)
    p.add_argument("--h", type=int, default=120)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--cells", type=int, default=300)
    p.add_argument("--aa", type=float, default=1.0)
    p.add_argument("--bgp", type=float, default=1.0)
    p.add_argument("--dexa", type=float, default=1.0)
    p.add_argument("--mk4", type=float, default=1.0)
    p.add_argument("--media-interval", type=int, default=2)
    p.add_argument("--snapdir", type=str, default="snapshots")
    p.add_argument("--out", type=str, default="results.xlsx", help="Output Excel file path")
    p.add_argument("--seed", type=int, default=42, help="Random seed (handled inside model)")
    p.add_argument("--csv", action="store_true", help="Also export CSV")
    p.add_argument("--snapshots", action="store_true", help="Save snapshot arrays and per-cell CSV")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    pde_cfg = PDEConfig(width=args.w, height=args.h)
    abm_cfg = ABMConfig(width=args.w, height=args.h)

    model = OsteoModel(
        pde_cfg=pde_cfg,
        abm_cfg=abm_cfg,
        total_days=args.days,
        snapshot_dir=args.snapdir,
        aa_level=args.aa,
        bgp_level=args.bgp,
        dexa_level=args.dexa,
        mk4_level=args.mk4,
        media_interval_days=args.media_interval,
        rng_seed=args.seed,              # model handles RNG
        enable_snapshots=args.snapshots,
        save_csv=args.csv,
    )
    model.seed_cells(args.cells)
    model.run()
    model.save(args.out)

    # Time-series summary
    plot_summary(args.out, output="summary.png")

    # Optional per-day panels
    if args.snapshots:
        export_single_panels(args.snapdir, days=[7, 14, args.days], out_dir="panels_pub")
    else:
        print("[INFO] Snapshots disabled; skipping per-day panels.")

    print("[DONE] Simulation complete.")
