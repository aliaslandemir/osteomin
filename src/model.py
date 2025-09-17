from __future__ import annotations
from dataclasses import asdict
from pathlib import Path
from typing import List
import numpy as np
import polars as pl
from tqdm.auto import trange

from .config import PDEConfig, ABMConfig
from .agent import Cell, CellState
from .abm_manager import ABMManager
from .pde_manager import PDEManager
from .io_utils import save_snapshot, save_timeseries_excel, save_timeseries_csv

__all__ = ["OsteoModel"]

class OsteoModel:
    """MC3T3-E1 on 2D coverslips with AA, β-GP, Dexa, MK4 (consumed daily, replenished on media change)."""

    def __init__(
        self, *, pde_cfg: PDEConfig, abm_cfg: ABMConfig,
        total_days: int = 21, snapshot_dir: str = "snapshots",
        aa_level: float = 1.0, bgp_level: float = 1.0,
        dexa_level: float = 1.0, mk4_level: float = 1.0,
        media_interval_days: int = 2, rng_seed: int | None = 42,
        enable_snapshots: bool = False, save_csv: bool = False
    ) -> None:
        self.pde = PDEManager(**asdict(pde_cfg))
        self.rng = np.random.default_rng(rng_seed)
        self.abm = ABMManager(**asdict(abm_cfg), rng=self.rng); self.abm.pde = self.pde

        self.total_days = total_days
        self.day = 0
        self.media_interval = media_interval_days
        self.snapdir = Path(snapshot_dir); self.snapdir.mkdir(exist_ok=True)
        self.enable_snapshots = enable_snapshots
        self.save_csv = save_csv

        self.aa_level = aa_level
        self.bgp_level = bgp_level
        self.dexa_level = dexa_level
        self.mk4_level = mk4_level

        # apply initial β-GP treatment level
        self.pde.bgp[:] *= self.bgp_level

        if rng_seed is not None:
            self.rng = np.random.default_rng(rng_seed)
        else:
            self.rng = np.random.default_rng()        # all randomness comes from self.rng (no global seeding)

        # histories
        self.hist_collagen: List[float] = []
        self.hist_acp: List[float] = []
        self.hist_ha: List[float] = []
        self.hist_ca: List[float] = []
        self.hist_po4: List[float] = []
        self.hist_alp: List[float] = []
        self.hist_supersat: List[float] = []
        self.hist_active: List[int] = []
        self.hist_apopt: List[int] = []           # deaths TODAY
        self.hist_ob: List[int] = []
        self.hist_ob_cum: List[int] = []          # cumulative commitments
        self.hist_dead_cum: List[int] = []        # cumulative deaths
        self.hist_nodule_index: List[float] = []
        self.hist_nodule_area: List[float] = []
        self.hist_bgp_avg: List[float] = []
        self.hist_aa_level: List[float] = []
        self.hist_bgp_level: List[float] = []
        self.hist_dexa_level: List[float] = []
        self.hist_mk4_level: List[float] = []

    def seed_cells(self, n: int = 200) -> None:
        for _ in range(n):
            x = int(self.rng.integers(self.pde.width  // 10, 9 * self.pde.width  // 10))
            y = int(self.rng.integers(self.pde.height // 10, 9 * self.pde.height // 10))
            self.abm.add_cell(Cell(x=x, y=y, state=CellState.ACTIVE))

    @staticmethod
    def _nodule_mask_from_ha(ha: np.ndarray) -> np.ndarray:
        if not np.any(ha > 0):
            return np.zeros_like(ha, dtype=float)
        nz = ha[ha > 0]; H, W = ha.shape
        for q, local_frac in [(99.9,0.6),(99.5,0.55),(99.0,0.5),(97.5,0.45),(95.0,0.40)]:
            tau = float(np.quantile(nz, q/100.0))
            sum9 = (ha + np.roll(ha,1,0)+np.roll(ha,-1,0)+np.roll(ha,1,1)+np.roll(ha,-1,1)+
                    np.roll(np.roll(ha,1,0),1,1)+np.roll(np.roll(ha,1,0),-1,1)+
                    np.roll(np.roll(ha,-1,0),1,1)+np.roll(np.roll(ha,-1,0),-1,1))
            local_mean = sum9/9.0
            mask = (ha>tau) & (local_mean>local_frac*tau)
            if mask.sum() >= max(5,int(0.0005*H*W)): return mask.astype(float)
        x = np.log1p(ha)
        hist, edges = np.histogram(x, bins=256)
        w0, w1 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
        mu = np.cumsum(hist * (edges[:-1]+edges[1:])/2); mu_t = mu[-1]
        sigma_b2 = (mu_t*w0 - mu)**2 / (w0*(w1+1e-12)+1e-12)
        t_idx = np.nanargmax(sigma_b2); thr = (edges[t_idx]+edges[t_idx+1])/2
        return (x>thr).astype(float)

    def _consume_supplements(self) -> None:
        area = self.pde.width * self.pde.height
        aa_drop = self.pde.aa_consume_per_collagen * (self.abm.last_collagen_added/(area+1e-12))
        self.aa_level = np.clip(self.aa_level - aa_drop,0,1)
        dx_drop = self.pde.dexa_consume_per_alp * (self.abm.last_alp_added/(area+1e-12))
        self.dexa_level = np.clip(self.dexa_level - dx_drop,0,1)
        mk_drop = self.pde.mk4_consume_per_dHA * (self.pde.last_dHA_sum/(area+1e-12))
        self.mk4_level = np.clip(self.mk4_level - mk_drop,0,1)

    def run(self) -> None:
        for _ in trange(self.total_days, desc="sim-day"):
            self.day += 1
            # push today’s levels to ABM/PDE
            self.abm.aa = self.aa_level
            self.abm.dexa = self.dexa_level
            self.pde.aa_level = self.aa_level
            self.pde.dexa_level = self.dexa_level
            self.pde.mk4_level = self.mk4_level

            # advance
            self.abm.step_all(self.day)
            self.pde.step()
            self._consume_supplements()

            # media change
            if self.day % self.media_interval == 0:
                # Reset freely available ions and BGP to baseline
                self.pde.media_refresh()
                # Keep the treatment level for β-GP across refreshes
                self.pde.bgp[:] = self.pde.bgp_bc_value * float(self.bgp_level)
                # Washout a fraction of soluble ALP on media change
                self.pde.alp *= 0.30
                # Replenish supplements (scalar levels)
                self.aa_level = self.dexa_level = self.mk4_level = 1.0

            self._record()
            if self.enable_snapshots and (self.day % 7 == 0 or self.day == self.total_days):
                save_snapshot(self.day, self.snapdir, self.pde, self.abm.agents,
                              deaths_xy=self.abm.deaths_today_xy)

    def _record(self) -> None:
        p = self.pde
        self.hist_collagen.append(p.collagen.mean())
        self.hist_acp.append(p.acp.mean())
        self.hist_ha.append(p.ha.mean())
        self.hist_ca.append(p.ca.mean())
        self.hist_po4.append(p.po4.mean())
        self.hist_alp.append(p.alp.mean())
        self.hist_bgp_avg.append(p.bgp.mean())
        self.hist_supersat.append(p.supersaturation().mean())

        # current live-state counts
        cur_active = sum(a.state==CellState.ACTIVE for a in self.abm.agents)
        cur_ob     = sum(a.state==CellState.OSTEOBLAST for a in self.abm.agents)
        # deaths TODAY from event counter
        cur_apopt  = self.abm.died_today

        self.hist_active.append(cur_active)
        self.hist_apopt.append(cur_apopt)   # daily deaths
        self.hist_ob.append(cur_ob)

        # cumulative events from ABM counters
        cum_ob   = self.abm.cum_committed
        cum_dead = self.abm.cum_died
        self.hist_ob_cum.append(cum_ob)
        self.hist_dead_cum.append(cum_dead)

        # nodule metrics
        nod = self._nodule_mask_from_ha(p.ha)
        self.hist_nodule_index.append((nod*p.ha).sum()/(p.ha.sum()+1e-16))
        self.hist_nodule_area.append(nod.mean())

        # supplement/baseline levels
        bgp_frac = p.bgp.mean()/(p.bgp_bc_value+1e-12)
        self.hist_aa_level.append(self.aa_level)
        self.hist_bgp_level.append(np.clip(bgp_frac,0,1))
        self.hist_dexa_level.append(self.dexa_level)
        self.hist_mk4_level.append(self.mk4_level)

    def save(self, outfile: str | Path = "results.xlsx") -> None:
        n = self.day
        df = pl.DataFrame({
            "Day": range(1,n+1),
            "AA_Level": self.hist_aa_level[:n], "BGP_Level": self.hist_bgp_level[:n],
            "Dexa_Level": self.hist_dexa_level[:n], "MK4_Level": self.hist_mk4_level[:n],
            "Collagen_Avg": self.hist_collagen[:n], "ALP_Avg": self.hist_alp[:n],
            "ACP_Avg": self.hist_acp[:n], "HA_Avg": self.hist_ha[:n],
            "Ca_Avg": self.hist_ca[:n], "PO4_Avg": self.hist_po4[:n],
            "BGP_Avg": self.hist_bgp_avg[:n], "Supersat_Avg": self.hist_supersat[:n],
            "Nodule_Index": self.hist_nodule_index[:n], "Nodule_Area": self.hist_nodule_area[:n],
            "Cells_Active": self.hist_active[:n],
            "Cells_Apoptotic": self.hist_apopt[:n],            # daily deaths
            "Cells_Osteoblast": self.hist_ob[:n],
            "Cells_Osteoblast_Cum": self.hist_ob_cum[:n],
            "Cells_Apoptotic_Cum": self.hist_dead_cum[:n],
        })
        save_timeseries_excel(df, outfile)
        if self.save_csv:
            save_timeseries_csv(df, Path(outfile).with_suffix(".csv"))
