from __future__ import annotations
from pathlib import Path
from typing import Dict, List
import numpy as np
import pandas as pd

from .abm_manager import ABMManager
from .config import ABMConfig, PDEConfig
from .pde_manager import PDEManager
from .agent import CellState
from .io_utils import cell_grids, save_timeseries_excel, coords_to_mask
from .viz import export_spatial_for_day

__all__ = ["OsteoModel"]


class OsteoModel:
    """Couples ABM↔PDE, tracks events, handles I/O and spatial exports."""

    def __init__(
        self,
        days: int,
        outdir: Path,
        abm_cfg: ABMConfig,
        pde_cfg: PDEConfig,
        supplements: Dict[str, float],
        seed: int | None = None,
        spatial_every: int = 0,
        spatial_days: list[int] | None = None,
    ) -> None:
        self.days = int(days)
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.grids_dir = self.outdir / "grids"
        self.grids_dir.mkdir(parents=True, exist_ok=True)

        self.rng = np.random.default_rng(seed)
        self.abm = ABMManager(abm_cfg, self.rng)
        self.pde = PDEManager(
            width=pde_cfg.width,
            height=pde_cfg.height,
            media_interval=pde_cfg.media_interval,
            supplements=supplements,
        )
        self.abm.attach_pde(self.pde)

        self._rows: List[Dict] = []

        # Spatial export policy
        self.spatial_every = max(0, int(spatial_every))
        base_days = set((spatial_days or []))
        base_days.add(self.days - 1)  # always export last day
        self.spatial_days = {d for d in base_days if 0 <= d < self.days}

    # ---- seeding ----
    def seed_random(self, n: int, w: int, h: int) -> None:
        sites = [(x, y) for y in range(h) for x in range(w)]
        n = min(int(n), len(sites))
        idx = self.rng.choice(len(sites), size=n, replace=False)
        coords = [sites[i] for i in idx]
        self.abm.seed_from_coords(coords, day0=0)

    # ---- export policy ----
    def _should_export_spatial(self, day: int) -> bool:
        if day in self.spatial_days:
            return True
        if self.spatial_every > 0 and (day % self.spatial_every) == 0:
            return True
        return False

    # ---- run ----
    def run(self) -> pd.DataFrame:
        # ---------- DAY 0: baseline snapshot only ----------
        deaths_mask = np.zeros((self.abm.height, self.abm.width), dtype=np.uint8)
        deaths_active_mask = deaths_mask.copy()
        deaths_ob_mask = deaths_mask.copy()

        dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
        np.save(self.grids_dir / f"density_day_000.npy", dens.astype(np.uint16))
        np.save(self.grids_dir / f"state_day_000.npy",   state.astype(np.uint8))
        np.save(self.grids_dir / f"deaths_day_000.npy",  deaths_mask)
        np.save(self.grids_dir / f"deaths_active_day_000.npy", deaths_active_mask)
        np.save(self.grids_dir / f"deaths_ob_day_000.npy",     deaths_ob_mask)

        # day-0 timeseries (no events; PDE baseline)
        self._append_timeseries_row_baseline(day=0)

        # cumulative counters (start at zero)
        dead_active_cum = 0
        dead_ob_cum = 0
        dead_total_cum = 0

        if self._should_export_spatial(0):
            export_spatial_for_day(
                self.outdir, 0, state, deaths_active_mask, deaths_ob_mask,
                dead_total_cum=dead_total_cum
            )

        # ---------- DAYS 1..N-1: run dynamics ----------
        for day in range(1, self.days):
            # ABM behaviors (commitment/apoptosis/prolif/migration/secretion)
            self.abm.step_day(day)

            # event masks (today)
            deaths_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_cells)
            deaths_active_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_active_coords)
            deaths_ob_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_ob_coords)

            # save grids after today's changes
            dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
            np.save(self.grids_dir / f"density_day_{day:03d}.npy", dens.astype(np.uint16))
            np.save(self.grids_dir / f"state_day_{day:03d}.npy",   state.astype(np.uint8))
            np.save(self.grids_dir / f"deaths_day_{day:03d}.npy",  deaths_mask)
            np.save(self.grids_dir / f"deaths_active_day_{day:03d}.npy", deaths_active_mask.astype(np.uint8))
            np.save(self.grids_dir / f"deaths_ob_day_{day:03d}.npy",     deaths_ob_mask.astype(np.uint8))

            # PDE step (media refresh etc.)
            self.pde.step_day()

            # timeseries row with daily events
            self._append_timeseries_row(day)

            # update cumulative deaths (inclusive of today)
            dead_active_cum += int(deaths_active_mask.sum())
            dead_ob_cum     += int(deaths_ob_mask.sum())
            dead_total_cum   = dead_active_cum + dead_ob_cum

            # spatial export (include cumulative total in title)
            if self._should_export_spatial(day):
                export_spatial_for_day(
                    self.outdir, day, state, deaths_active_mask, deaths_ob_mask,
                    dead_total_cum=dead_total_cum
                )

        return pd.DataFrame(self._rows).sort_values("day", ignore_index=True)

    def _append_timeseries_row(self, day: int) -> None:
        n_active = sum(1 for c in self.abm.cells if c.state == CellState.ACTIVE)
        n_ob     = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)

        dead_active_today = len(self.abm.dead_active_coords)
        dead_ob_today     = len(self.abm.dead_ob_coords)
        committed_today   = int(self.abm.committed_today)

        scalars = self.pde.snapshot_scalars()
        self._rows.append({
            "day": int(day),
            "n_active": int(n_active),
            "n_osteoblast": int(n_ob),
            "n_total": int(n_active + n_ob),
            "dead_active_today": int(dead_active_today),
            "dead_ob_today": int(dead_ob_today),
            "committed_today": int(committed_today),
            "free_ca": float(scalars["free_ca"]),
            "free_pi": float(scalars["free_pi"]),
            "collagen_total": float(scalars["collagen_total"]),
            "alp_total": float(scalars["alp_total"]),
        })

    def _append_timeseries_row_baseline(self, day: int) -> None:
        """Day-0 row: all seeded cells, no events, PDE at baseline."""
        n_active = sum(1 for c in self.abm.cells if c.state == CellState.ACTIVE)
        n_ob     = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)
        self._rows.append({
            "day": int(day),
            "n_active": int(n_active),
            "n_osteoblast": int(n_ob),
            "n_total": int(n_active + n_ob),
            "dead_active_today": 0,
            "dead_ob_today": 0,
            "committed_today": 0,
            "free_ca": 1.0,
            "free_pi": 1.0,
            "collagen_total": float(self.pde.total_collagen),  # 0.0 at baseline
            "alp_total": float(self.pde.total_alp),            # 0.0 at baseline
        })

    def save_timeseries_excel(self, df: pd.DataFrame, basename: str = "timeseries") -> Path:
        out = self.outdir / f"{basename}.xlsx"
        save_timeseries_excel(df, out)
        return out
