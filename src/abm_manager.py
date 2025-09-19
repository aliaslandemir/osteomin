from __future__ import annotations
from typing import List, Tuple
import numpy as np

from .agent import Cell, CellState
from .config import ABMConfig

__all__ = ["ABMManager"]

class ABMManager:
    """On-lattice ABM with event tracking lists per day."""
    def __init__(self, cfg: ABMConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng
        self.width, self.height = int(cfg.width), int(cfg.height)

        # state
        self.cells: List[Cell] = []
        self.occupancy = np.full((self.height, self.width), -1, dtype=np.int32)

        # daily event logs
        self.dead_cells: List[Tuple[int,int]] = []          # (row, col)
        self.dead_active_coords: List[Tuple[int,int]] = []  # died while ACTIVE
        self.dead_ob_coords: List[Tuple[int,int]] = []      # died while OB
        self.committed_today: int = 0

        self.pde = None

    # ----- seeding -----
    def seed_from_coords(self, coords: List[Tuple[int,int]], day0: int = 0) -> None:
        for (x, y) in coords:
            if 0 <= x < self.width and 0 <= y < self.height and self.occupancy[y, x] == -1:
                c = self._spawn_cell(x, y, day0)
                self.cells.append(c)
                self.occupancy[y, x] = len(self.cells) - 1

    def _spawn_cell(self, x: int, y: int, day: int) -> Cell:
        c = Cell(x=x, y=y, state=CellState.ACTIVE, last_div_day=day)
        if self.cfg.min_div_days <= self.cfg.max_div_days:
            c.division_interval = int(self.rng.integers(self.cfg.min_div_days,
                                                        self.cfg.max_div_days + 1))
        return c

    def attach_pde(self, pde) -> None:
        self.pde = pde

    # ----- step -----
    def step_day(self, day: int) -> None:
        # reset dailies
        self.dead_cells = []
        self.dead_active_coords = []
        self.dead_ob_coords = []
        self.committed_today = 0

        # migration
        if self.cfg.enable_migration:
            self._migration_step()

        # commitment
        if self.cfg.enable_commitment:
            self._commitment_step(day)

        # apoptosis
        if self.cfg.enable_apoptosis:
            self._apoptosis_step(day)

        # proliferation
        if self.cfg.enable_proliferation:
            self._proliferation_step(day)

        # secretion
        if self.cfg.enable_secretion:
            self._secretion_step()

        # PDE tick last
        if self.pde is not None:
            self.pde.step_day()

    # ---- internals ----
    def _neighbors4(self, x: int, y: int):
        # wrap-free, clamp to grid (ignore out-of-bounds)
        nbrs = []
        if x > 0: nbrs.append((x-1, y))
        if x < self.width-1: nbrs.append((x+1, y))
        if y > 0: nbrs.append((x, y-1))
        if y < self.height-1: nbrs.append((x, y+1))
        return nbrs

    def _migration_step(self) -> None:
        for idx, c in enumerate(self.cells):
            if c.state != CellState.ACTIVE:
                continue
            if self.rng.random() >= self.cfg.move_prob:
                continue

            steps = 0
            while steps < self.cfg.max_moves_per_day:
                nbrs = self._neighbors4(c.x, c.y)
                self.rng.shuffle(nbrs)

                moved = False
                for (nx, ny) in nbrs:
                    if self.occupancy[ny, nx] == -1:
                        # vacate old and occupy new
                        self.occupancy[c.y, c.x] = -1
                        self.occupancy[ny, nx] = idx
                        c.record_move(nx, ny)
                        moved = True
                        break

                if not moved:
                    break  # no free neighbor; stop trying
                steps += 1

    def _commitment_step(self, day: int) -> None:
        p = float(self.cfg.base_commit_prob)
        if p <= 0.0: return
        for c in self.cells:
            if c.state == CellState.ACTIVE and self.rng.random() < p:
                c.mark_committed(day)
                self.committed_today += 1

    def _apoptosis_step(self, day: int) -> None:
        died_indices = []
        for idx, c in enumerate(self.cells):
            if self.rng.random() < self.cfg.apoptosis_prob:
                # classify by state at time of death
                if c.state == CellState.ACTIVE:
                    self.dead_active_coords.append((c.y, c.x))
                elif c.state == CellState.OSTEOBLAST:
                    self.dead_ob_coords.append((c.y, c.x))
                self.dead_cells.append((c.y, c.x))
                c.mark_apoptotic(day)
                died_indices.append(idx)

        if not died_indices:
            return

        # remove dead cells and rebuild occupancy
        keep = [i for i in range(len(self.cells)) if i not in died_indices]
        self.cells = [self.cells[i] for i in keep]
        self.occupancy[:] = -1
        for i, c in enumerate(self.cells):
            self.occupancy[c.y, c.x] = i

    def _proliferation_step(self, day: int) -> None:
        newborns: List[Cell] = []
        for idx, c in enumerate(self.cells):
            if c.state == CellState.APOPTOTIC:
                continue
            # NEW: Osteoblasts normally do not divide; allow rare division via cfg.ob_division_prob
            if c.state == CellState.OSTEOBLAST and self.rng.random() >= self.cfg.ob_division_prob:
                continue
            if (day - c.last_div_day) < c.division_interval:
                continue

            # attempt to divide into a free neighbor
            nbrs = self._neighbors4(c.x, c.y)
            self.rng.shuffle(nbrs)
            placed = False
            for (nx, ny) in nbrs:
                if self.occupancy[ny, nx] == -1:
                    child = self._spawn_cell(nx, ny, day)
                    newborns.append(child)
                    self.occupancy[ny, nx] = len(self.cells) + len(newborns) - 1
                    c.last_div_day = day
                    placed = True
                    break

            if not placed:
                # delay division if no space
                c.last_div_day += 1

        for child in newborns:
            self.cells.append(child)
        # rebuild occupancy to be safe
        self.occupancy[:] = -1
        for i, c in enumerate(self.cells):
            self.occupancy[c.y, c.x] = i

    def _secretion_step(self) -> None:
        if self.pde is None: return
        n_ob = sum(1 for c in self.cells if c.state == CellState.OSTEOBLAST)
        if n_ob <= 0: return
        self.pde.add_collagen(n_ob * self.cfg.collagen_per_ob)
        self.pde.add_alp(n_ob * self.cfg.alp_per_ob)
