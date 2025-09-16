from __future__ import annotations
from typing import List, Optional
import numpy as np
from .agent import Cell, CellState

__all__ = ["ABMManager"]

class ABMManager:
    """
    Agent-based layer:
      - ACTIVE cells proliferate early (<= prolif_until_day) and secrete collagen (scaled by AA).
      - DEXA biases commitment to OSTEOBLAST from day 0 (no ramp).
      - OSTEOBLASTS produce more collagen and ALP (scaled by AA and Dexa).
      - Apoptosis is a small per-day probability.

    Tracks per-day secretion to drive supplement consumption:
      - last_collagen_added
      - last_alp_added
    """
    def __init__(self, *, width:int, height:int, max_cell_density:int,
                 apoptosis_prob:float, enable_migration:bool, p_move:float,
                 min_div_days:int, max_div_days:int, prolif_until_day:int) -> None:
        self.width, self.height = width, height
        self.max_cell_density = max_cell_density
        self.apoptosis_prob = apoptosis_prob
        self.enable_migration = enable_migration
        self.p_move = p_move
        self.min_div_days = min_div_days
        self.max_div_days = max_div_days
        self.prolif_until_day = prolif_until_day

        self.agents: List[Cell] = []
        self.pde: "PDEManager | None" = None

        # run-time factor levels (set by model each day)
        self.aa = 1.0
        self.dexa = 1.0

        # bookkeeping (reset each day)
        self.last_collagen_added: float = 0.0
        self.last_alp_added: float = 0.0

    def add_cell(self, c: Cell) -> None:
        self.agents.append(c)

    def _occupancy_grid(self) -> np.ndarray:
        occ = np.zeros((self.height, self.width), dtype=np.int16)
        for a in self.agents:
            occ[a.y, a.x] += 1
        return occ

    def step_all(self, day: int) -> None:
        self.last_collagen_added = 0.0
        self.last_alp_added = 0.0

        occ = self._occupancy_grid()
        daughters: List[Cell] = []
        for c in self.agents:
            d = self._step_cell(c, day, occ)
            if d is not None:
                daughters.append(d)
        self.agents.extend(daughters)

    def _local_supersat(self, x:int, y:int) -> float:
        if self.pde is None: return 0.0
        return float((self.pde.ca[y, x] * self.pde.po4[y, x]) * (1.0 + 1.5 * np.tanh(self.pde.collagen[y, x])))

    def _step_cell(self, c: Cell, day: int, occ: np.ndarray) -> Optional[Cell]:
        # apoptosis
        if c.state in (CellState.ACTIVE, CellState.OSTEOBLAST) and np.random.random() < self.apoptosis_prob:
            c.state = CellState.APOPTOTIC
            if c.day_died is None:
                c.day_died = day
            return None
        if c.state == CellState.APOPTOTIC:
            return None

        # random walk
        if self.enable_migration and np.random.random() < self.p_move:
            dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][np.random.randint(0,4)]
            c.x = (c.x + dx) % self.width
            c.y = (c.y + dy) % self.height
            c.trajectory.append((c.x, c.y))
            c.move_count += 1

        # osteogenic switch (day-0 onward, Dexa dependent + supersaturation + density)
        if c.state == CellState.ACTIVE:
            p_switch = 0.12 * (0.2 + 0.8 * self.dexa)  # base; Dexa scales [~0.024..0.12]
            if occ[c.y, c.x] > self.max_cell_density:
                p_switch *= 1.2
            S = self._local_supersat(c.x, c.y)
            p_switch *= (1.0 + 0.6 * (S / (0.5 + S)))
            if np.random.random() < p_switch:
                c.state = CellState.OSTEOBLAST
                if not c.ever_ob:
                    c.ever_ob = True
                    c.day_committed = day

        # early proliferation window
        if c.state == CellState.ACTIVE and day <= self.prolif_until_day:
            age = day - c.day_created
            if age >= c.division_interval and occ[c.y, c.x] < self.max_cell_density:
                return Cell(x=c.x, y=c.y, day_created=day,
                            state=CellState.ACTIVE, division_interval=c.division_interval)

        # secretion (per-cell, local) – consumption bookkeeping
        if self.pde is not None:
            if c.state == CellState.ACTIVE:
                dc = 0.010 * self.aa
                self.pde.collagen[c.y, c.x] += dc
                self.last_collagen_added += dc
            elif c.state == CellState.OSTEOBLAST:
                dc = 0.020 * self.aa
                da = 0.030 * self.dexa
                self.pde.collagen[c.y, c.x] += dc
                self.pde.alp[c.y, c.x]      += da
                self.last_collagen_added += dc
                self.last_alp_added += da
        return None
