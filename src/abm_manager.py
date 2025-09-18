from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np

from .agent import Cell, CellState
from .config import ABMConfig

__all__ = ["ABMManager"]


class ABMManager:
    """
    On-lattice ABM (one cell per node by default).
    Behavior is governed by ABMConfig switches:
      - enable_migration
      - enable_commitment
      - enable_apoptosis
      - enable_proliferation
      - enable_secretion
    """

    def __init__(self, cfg: ABMConfig, rng: Optional[np.random.Generator] = None) -> None:
        self.cfg = cfg
        self.width = cfg.width
        self.height = cfg.height
        self.max_cell_density = cfg.max_cell_density

        self.rng = np.random.default_rng() if rng is None else rng

        # External handle (set by model): used only if secretion is enabled.
        self.pde = None  # type: ignore

        # Agent store
        self.cells: List[Cell] = []

        # Daily event tracking
        self.committed_today: int = 0
        self.died_today: int = 0
        self.deaths_today_xy: List[Tuple[int, int]] = []

        # Cumulative counters
        self.cum_committed: int = 0
        self.cum_died: int = 0

        # Per-day secretion book-keeping
        self.last_collagen_added: float = 0.0
        self.last_alp_added: float = 0.0

    # ---------- Seeding ----------
    def seed_from_coords(self, coords: List[Tuple[int, int]], day0: int = 0) -> None:
        for (x, y) in coords:
            if 0 <= x < self.width and 0 <= y < self.height:
                self.cells.append(self._spawn_cell(x, y, day0))

    def _spawn_cell(self, x: int, y: int, day: int) -> Cell:
        c = Cell(x=x, y=y, state=CellState.ACTIVE, age_days=0, last_div_day=day)
        if self.cfg.min_div_days <= self.cfg.max_div_days:
            c.division_interval = int(self.rng.integers(self.cfg.min_div_days,
                                                        self.cfg.max_div_days + 1))
        else:
            c.division_interval = None
        return c

    # ---------- Day step ----------
    def step_day(self, day: int) -> None:
        cfg = self.cfg
        rng = self.rng

        # reset daily trackers
        self.committed_today = 0
        self.died_today = 0
        self.deaths_today_xy = []
        self.last_collagen_added = 0.0
        self.last_alp_added = 0.0

        # ---- Day 0: pure baseline (no state changes) ----
        if day == 0:
            # If you want migration on day 0, uncomment the two lines below.
            # if cfg.enable_migration and cfg.p_move > 0.0:
            #     self._do_movement(rng)
            return
        # -------------------------------------------------

        # 1) Movement (if enabled)
        if cfg.enable_migration and cfg.p_move > 0.0:
            self._do_movement(rng)

        # 2) Commitment (ACTIVE → OSTEOBLAST)
        if cfg.enable_commitment and cfg.base_commit_prob > 0.0:
            self._do_commitment(day, rng)

        # 3) Apoptosis
        if cfg.enable_apoptosis and cfg.apoptosis_prob > 0.0:
            self._do_apoptosis(day, rng)

        # Remove apoptotic cells immediately (keeps density “live-only”)
        if self.died_today:
            self._purge_apoptotic()

        # 4) Proliferation (ACTIVE only; osteoblasts never divide)
        if cfg.enable_proliferation:
            self._do_proliferation(day, rng)

        # 5) Secretion (osteoblasts contribute per day)
        if cfg.enable_secretion:
            self._do_secretion()

        # 6) Aging
        for c in self.cells:
            c.age_days += 1

    # ---------- Subsystems ----------
    def _do_movement(self, rng: np.random.Generator) -> None:
        w, h = self.width, self.height
        p = self.cfg.p_move
        # 4-neighborhood moves
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        occ = {(c.x, c.y): c for c in self.cells}  # occupancy

        for c in self.cells:
            if c.state != CellState.ACTIVE:
                continue
            if rng.random() < p:
                dx, dy = deltas[int(rng.integers(0, len(deltas)))]
                nx, ny = c.x + dx, c.y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in occ:
                    # update occupancy
                    occ.pop((c.x, c.y), None)
                    c.record_move(nx, ny)
                    occ[(nx, ny)] = c

    def _do_commitment(self, day: int, rng: np.random.Generator) -> None:
        p = self.cfg.base_commit_prob
        for c in self.cells:
            if c.state == CellState.ACTIVE and rng.random() < p:
                c.mark_committed(day)
                self.committed_today += 1
                self.cum_committed += 1

    def _do_apoptosis(self, day: int, rng: np.random.Generator) -> None:
        p = self.cfg.apoptosis_prob
        for c in self.cells:
            if c.state == CellState.APOPTOTIC:
                continue
            if rng.random() < p:
                c.mark_apoptotic(day)
                self.died_today += 1
                self.cum_died += 1
                self.deaths_today_xy.append((c.x, c.y))

    def _purge_apoptotic(self) -> None:
        self.cells = [c for c in self.cells if c.state != CellState.APOPTOTIC]

    def _do_proliferation(self, day: int, rng: np.random.Generator) -> None:
        # One cell per node (default). Spawn into empty von Neumann neighbor if division due.
        if self.cfg.min_div_days > self.cfg.max_div_days:
            return  # disabled via invalid window

        w, h = self.width, self.height
        deltas = [(1, 0), (-1, 0), (0, 1), (0, -1)]
        occ = {(c.x, c.y): True for c in self.cells}

        new_cells: List[Cell] = []
        for c in self.cells:
            if c.state != CellState.ACTIVE:
                continue
            if c.division_interval is None or c.last_div_day is None:
                continue
            if (day - c.last_div_day) < c.division_interval:
                continue

            # try up to 4 neighbors for an empty slot
            rng.shuffle(deltas)
            for dx, dy in deltas:
                nx, ny = c.x + dx, c.y + dy
                if 0 <= nx < w and 0 <= ny < h and (nx, ny) not in occ:
                    child = self._spawn_cell(nx, ny, day)
                    new_cells.append(child)
                    occ[(nx, ny)] = True
                    c.last_div_day = day
                    # re-sample next division interval for parent and set for child
                    break
            # If all neighbors are full, skip division (contact inhibition).

        if new_cells:
            self.cells.extend(new_cells)

    def _do_secretion(self) -> None:
        # Only if a PDE object is connected AND declares collagen/alp sinks
        if self.pde is None:
            return
        n_ob = sum(1 for c in self.cells if c.state == CellState.OSTEOBLAST)
        if n_ob == 0:
            return

        collagen = float(n_ob) * self.cfg.collagen_per_ob
        alp = float(n_ob) * self.cfg.alp_per_ob

        if hasattr(self.pde, "add_collagen"):
            self.pde.add_collagen(collagen)
            self.last_collagen_added = collagen
        if hasattr(self.pde, "add_alp"):
            self.pde.add_alp(alp)
            self.last_alp_added = alp
