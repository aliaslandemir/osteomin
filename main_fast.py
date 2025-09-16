"""
CLI entry-point – MC3T3-E1 osteogenesis (2D coverslip; AA, β-GP, Dexa, MK4).
Media refresh defaults to every 2 days.
"""
from __future__ import annotations
from pathlib import Path
import argparse
import numpy as np

from src.model import OsteoModel
from src.config import PDEConfig, ABMConfig
from src.viz import plot_summary, export_single_panels

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--w", type=int, default=120)
    p.add_argument("--h", type=int, default=120)
    p.add_argument("--days", type=int, default=21)
    p.add_argument("--cells", type=int, default=300)
    p.add_argument("--snapdir", type=str, default="snapshots")
    p.add_argument("--out", type=str, default="results.parquet")
    p.add_argument("--seed", type=int, default=42)

    # media/factors (0..1)
    p.add_argument("--aa", type=float, default=1.0, help="Ascorbic acid level")
    p.add_argument("--bgp", type=float, default=1.0, help="β-GP reservoir level")
    p.add_argument("--dexa", type=float, default=1.0, help="Dexamethasone level")
    p.add_argument("--mk4", type=float, default=1.0, help="Vitamin K2 (MK4) level")
    p.add_argument("--media-interval", type=int, default=2, help="Days between medium changes (default 2)")
    return p.parse_args()

if __name__ == "__main__":
    a = parse_args()
    np.random.seed(a.seed)

    pde_cfg = PDEConfig(width=a.w, height=a.h)
    abm_cfg = ABMConfig(width=a.w, height=a.h)

    model = OsteoModel(
        pde_cfg=pde_cfg, abm_cfg=abm_cfg,
        total_days=a.days, snapshot_dir=a.snapdir,
        aa_level=a.aa, bgp_level=a.bgp, dexa_level=a.dexa, mk4_level=a.mk4,
        media_interval_days=a.media_interval,
        rng_seed=a.seed,
    )
    model.seed_cells(a.cells)
    model.run()
    model.save(a.out)

    # Figures
    plot_summary(a.out, output="summary.png")
    export_single_panels(a.snapdir, days=[7, 14, a.days], out_dir="panels_pub")
    print("[DONE] Simulation + plots finished.")
