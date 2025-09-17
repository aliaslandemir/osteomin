from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from .agent import Cell, CellState

__all__ = ["ABMManager"]

class ABMManager:
    """
    Agent-based layer (one node = one cell slot):
      - ACTIVE cells can divide (osteoblasts NEVER divide).
      - DEXA biases commitment to OSTEOBLAST from day 0 (no ramp).
      - OSTEOBLASTS secrete more collagen and ALP (scaled by AA and Dexa).
      - Apoptosis removes the cell immediately (records day_died), so density is live-only.

    Tracks per-day secretion for supplement consumption:
      - last_collagen_added, last_alp_added

    Tracks commitment/death events so cumulative metrics remain correct
    even though dead cells are removed:
      - cum_committed, cum_died (cumulative totals)
      - committed_today, died_today (per-day events)
      - deaths_today_xy: List[(x,y)] (positions of cells that died today)
    """
    def __init__(self, *, width:int, height:int, max_cell_density:int,
                 apoptosis_prob:float, enable_migration:bool, p_move:float,
                 min_div_days:int, max_div_days:int) -> None:
        self.width, self.height = width, height
        self.max_cell_density = max_cell_density
        self.apoptosis_prob = apoptosis_prob
        self.enable_migration = enable_migration
        self.p_move = p_move
        self.min_div_days = min_div_days
        self.max_div_days = max_div_days

        self.agents: List[Cell] = []
        self.pde: "PDEManager | None" = None  # set by model

        # run-time factor levels (set by model each day)
        self.aa = 1.0
        self.dexa = 1.0

        # secretion bookkeeping (reset each day)
        self.last_collagen_added: float = 0.0
        self.last_alp_added: float = 0.0

        # event counters
        self.cum_committed: int = 0
        self.cum_died: int = 0
        self.committed_today: int = 0
        self.died_today: int = 0
        self.deaths_today_xy: List[Tuple[int,int]] = []

        self._rng = np.random.default_rng(42)

    # -------------------------- helpers --------------------------
    def _sample_cycle_days(self) -> int:
        # inclusive range [min, max]
        return int(self._rng.integers(self.min_div_days, self.max_div_days + 1))

    def add_cell(self, c: Cell) -> None:
        # Ensure ACTIVE cells have a valid division interval from config
        if c.state == CellState.ACTIVE and (c.division_interval is None or c.division_interval < 1):
            c.division_interval = self._sample_cycle_days()
        self.agents.append(c)

    def _occupancy_grid(self) -> np.ndarray:
        occ = np.zeros((self.height, self.width), dtype=np.int16)
        for a in self.agents:
            # only live cells occupy space
            if a.state in (CellState.ACTIVE, CellState.OSTEOBLAST):
                occ[a.y, a.x] += 1
        return occ

    def _local_supersat(self, x:int, y:int) -> float:
        if self.pde is None:
            return 0.0
        # collagen-templated supersaturation (bounded via tanh)
        return float((self.pde.ca[y, x] * self.pde.po4[y, x]) *
                     (1.0 + 1.5 * np.tanh(self.pde.collagen[y, x])))

    # --------------------------- stepper -------------------------
    def step_all(self, day: int) -> None:
        # reset per-day logs/counters
        self.last_collagen_added = 0.0
        self.last_alp_added = 0.0
        self.committed_today = 0
        self.died_today = 0
        self.deaths_today_xy = []

        occ = self._occupancy_grid()
        daughters: List[Cell] = []
        survivors: List[Cell] = []

        for c in self.agents:
            d = self._step_cell(c, day, occ)
            if isinstance(d, Cell):
                daughters.append(d)
            # keep live cells only
            if c.state != CellState.APOPTOTIC:
                survivors.append(c)

        self.agents = survivors + daughters

    def _step_cell(self, c: Cell, day: int, occ: np.ndarray) -> Optional[Cell]:
        # 1) apoptosis → remove immediately (but record event + position)
        if c.state in (CellState.ACTIVE, CellState.OSTEOBLAST) and np.random.random() < self.apoptosis_prob:
            c.state = CellState.APOPTOTIC
            if c.day_died is None:
                c.day_died = day
            self.cum_died += 1
            self.died_today += 1
            self.deaths_today_xy.append((c.x, c.y))
            # live-only occupancy: free the spot
            if occ[c.y, c.x] > 0:
                occ[c.y, c.x] -= 1
            return None
        if c.state == CellState.APOPTOTIC:
            return None

        # 2) random walk with occupancy update (guard against overcrowding)
        if self.enable_migration and np.random.random() < self.p_move:
            oldx, oldy = c.x, c.y
            dx, dy = [(-1,0),(1,0),(0,-1),(0,1)][np.random.randint(0,4)]
            nx = (c.x + dx) % self.width
            ny = (c.y + dy) % self.height
            if occ[ny, nx] < self.max_cell_density:
                if occ[oldy, oldx] > 0:
                    occ[oldy, oldx] -= 1
                occ[ny, nx] += 1
                c.x, c.y = nx, ny
                c.trajectory.append((c.x, c.y))
                c.move_count += 1

        # 3) commitment (Dexa + local S + crowding)
        if c.state == CellState.ACTIVE:
            p_switch = 0.26 * (0.2 + 0.8 * self.dexa)  # Dexa scales switching
            if occ[c.y, c.x] > self.max_cell_density:
                p_switch *= 3.0
            S = self._local_supersat(c.x, c.y)
            p_switch *= (1.0 + 0.8 * (S / (0.5 + S)))
            if np.random.random() < p_switch:
                c.state = CellState.OSTEOBLAST
                if not c.ever_ob:
                    c.ever_ob = True
                    c.day_committed = day
                    self.cum_committed += 1
                    self.committed_today += 1

        # 4) proliferation — ONLY ACTIVE; reset parent's cycle clock; update occ
        if c.state == CellState.ACTIVE:
            age = day - c.day_created
            if c.division_interval is None:
                c.division_interval = self._sample_cycle_days()
            if age >= c.division_interval and occ[c.y, c.x] < self.max_cell_density:
                # Parent resets and resamples
                c.day_created = day
                c.division_interval = self._sample_cycle_days()

                # Daughter inherits position; independent cycle
                d = Cell(x=c.x, y=c.y, day_created=day, state=CellState.ACTIVE,
                         division_interval=self._sample_cycle_days())
                # Update live occupancy
                occ[c.y, c.x] += 1
                return d

        # 5) secretion (per-cell, local) – consumption bookkeeping
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
                self.last_alp_added      += da
        return None
