from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, List
import numpy as np
import polars as pl

from .abm_manager import ABMManager
from .config import ABMConfig, PDEConfig
from .pde_manager import PDEManager
from .agent import CellState
from .io_utils import cell_grids, save_timeseries_excel


__all__ = ["OsteoModel"]


class OsteoModel:
    def __init__(
        self,
        pde_cfg: PDEConfig,
        abm_cfg: ABMConfig,
        days: int,
        media_interval: int,
        rng_seed: Optional[int] = None,
        supplements: Optional[Dict[str, float]] = None,
        outdir: str | Path = "out",
    ) -> None:
        self.days = int(days)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(rng_seed)

        # PDE & ABM
        self.pde = PDEManager(pde_cfg.width, pde_cfg.height, media_interval, supplements)
        self.abm = ABMManager(cfg=abm_cfg, rng=self.rng)
        self.abm.pde = self.pde  # wire secretion hooks

        # Per-day metrics (Polars table)
        self._rows: List[Dict[str, float | int]] = []

    # ----- init seeding -----
    def seed_random(self, n_cells: int) -> None:
        n = max(0, int(n_cells))
        w, h = self.abm.width, self.abm.height
        if n == 0:
            return
        # sample without replacement
        all_sites = [(x, y) for y in range(h) for x in range(w)]
        if n > len(all_sites):
            n = len(all_sites)
        idx = self.rng.choice(len(all_sites), size=n, replace=False)
        coords = [all_sites[i] for i in idx]
        self.abm.seed_from_coords(coords)

    # ----- simulation -----
    def run(self) -> pl.DataFrame:
        for day in range(self.days):
            # day step (ABM first, then PDE accumulation)
            self.abm.step_day(day)

            # --- SAVE TRUE SPATIAL STATE + DEATHS GRIDS ---
            grids_dir = self.outdir / "grids"
            dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
            grids_dir.mkdir(parents=True, exist_ok=True)
            np.save(grids_dir / f"state_day_{day:03d}.npy", state.astype(np.uint8))
            # ---------------------------------------------------

            # deaths overlay (positions from today's apoptosis, even though cells are purged)
            if self.abm.deaths_today_xy:
                death = np.zeros((self.abm.height, self.abm.width), dtype=np.uint8)
                for (x, y) in self.abm.deaths_today_xy:
                    if 0 <= x < self.abm.width and 0 <= y < self.abm.height:
                        death[y, x] = 1
                np.save(grids_dir / f"deaths_day_{day:03d}.npy", death)

            self.pde.step_day()

            # metrics
            n_active = sum(1 for c in self.abm.cells if c.state == CellState.ACTIVE)
            n_ob = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)
            n_tot = len(self.abm.cells)

            scalars = self.pde.snapshot_scalars()
            self._rows.append({
                "day": day,
                "n_active": n_active,
                "n_osteoblast": n_ob,
                "n_total": n_tot,
                "committed_today": self.abm.committed_today,
                "died_today": self.abm.died_today,
                "cum_committed": self.abm.cum_committed,
                "cum_died": self.abm.cum_died,
                "collagen_added_today": self.abm.last_collagen_added,
                "alp_added_today": self.abm.last_alp_added,
                "free_ca": scalars["free_ca"],
                "free_pi": scalars["free_pi"],
                "collagen_total": scalars["collagen_total"],
                "alp_total": scalars["alp_total"],
            })

        df = pl.DataFrame(self._rows)
        return df




    def save_timeseries_excel(self, df: pl.DataFrame, basename: str = "timeseries") -> Path:
        xlsx_path = self.outdir / f"{basename}.xlsx"
        save_timeseries_excel(df, xlsx_path)
        return xlsx_path
