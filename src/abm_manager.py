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
        mode = getattr(cfg, "boundary_mode", "clamp")
        mode = mode.lower() if isinstance(mode, str) else "clamp"
        if mode not in ("clamp", "periodic"):
            raise ValueError(f"unsupported boundary_mode '{cfg.boundary_mode}'")
        self.boundary_mode = mode

        # state
        self.cells: List[Cell] = []
        self.occupancy = np.full((self.height, self.width), -1, dtype=np.int32)
        # random motility parameters
        # stay probability
        self._stay_prob = 0.5  # change this freely (e.g., 0.6)

        # 8-neighborhood offsets
        self._direction_offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),  # axial
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),  # diagonal
        ]

        # base movement weights
        base_axial = 1.0
        base_diag = 0.5

        direction_weights = [base_axial] * 4 + [base_diag] * 4
        total = sum(direction_weights)

        # normalized movement probabilities (sum to 1.0)
        self._direction_probs = [w / total for w in direction_weights]

        # daily event logs
        self.dead_cells: List[Tuple[int, int]] = []  # (row, col)
        self.dead_preob_coords: List[Tuple[int, int]] = []  # died while preOB
        self.dead_ob_coords: List[Tuple[int, int]] = []  # died while OB
        self.committed_today: int = 0

        self.field = None

    # ----- seeding -----
    def seed_from_coords(self, coords: List[Tuple[int, int]], day0: int = 0) -> None:
        for x, y in coords:
            if (
                0 <= x < self.width
                and 0 <= y < self.height
                and self.occupancy[y, x] == -1
            ):
                c = self._spawn_cell(x, y, day0)
                self.cells.append(c)
                self.occupancy[y, x] = len(self.cells) - 1

    def _spawn_cell(self, x: int, y: int, day: int) -> Cell:
        c = Cell(x=x, y=y, state=CellState.PRE_OB, last_div_day=day)
        if self.cfg.min_div_days <= self.cfg.max_div_days:
            c.division_interval = int(
                self.rng.integers(self.cfg.min_div_days, self.cfg.max_div_days + 1)
            )
        return c

    def attach_field(self, field) -> None:
        self.field = field

    # ----- step -----
    def step_day(self, day: int) -> None:
        # reset daily trackers
        self.dead_cells = []
        self.dead_preob_coords = []
        self.dead_ob_coords = []
        self.committed_today = 0

        # 1) random_move – small motility before division/commitment
        if self.cfg.enable_random_move:
            self._random_move_step()

        # 2) proliferation – cell cycle events and daughter placement
        if self.cfg.enable_proliferation:
            self._proliferation_step(day)

        # 3) commitment – differentiate some preOB to OB
        if self.cfg.enable_commitment and day >= self.cfg.commit_delay:
            commit_prob = min(
                1.0,
                float(self.cfg.base_commit_prob + day * self.cfg.commit_prob_growth),
            )
            for c in self.cells:
                if (
                    c.state == CellState.PRE_OB
                    and (day - c.last_div_day) >= 1
                    and self.rng.random() < commit_prob
                ):
                    c.mark_committed(day)
                    self.committed_today += 1

        # 4) secretion – OBs deposit collagen & ALP before any deaths
        if self.cfg.enable_secretion:
            self._secretion_step()

        # 5) apoptosis – remove cells after they have acted biologically
        if self.cfg.enable_apoptosis:
            self._apoptosis_step(day)

        # 6) chemistry update – send OB mask, then field tick / media refresh
        if self.field is not None:
            # build boolean mask of alive osteoblast positions (H×W)
            ob_mask = np.zeros((self.height, self.width), dtype=bool)
            for c in self.cells:
                if (
                    c.state == CellState.OSTEOBLAST
                ):  # only alive cells remain in self.cells
                    ob_mask[c.y, c.x] = True
            # provide mask to field (enables localized secretion & nucleation)
            if hasattr(self.field, "set_ob_mask"):
                self.field.set_ob_mask(ob_mask)
            self.field.step_day()

    # ---- internals ----
    def _neighbors4(self, x: int, y: int):
        # returns 4-neighbors; periodic mode wraps, otherwise ignore out-of-bounds
        offsets = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.boundary_mode == "periodic":
            return [
                ((x + dx) % self.width, (y + dy) % self.height) for dx, dy in offsets
            ]

        nbrs = []
        if x > 0:
            nbrs.append((x - 1, y))
        if x < self.width - 1:
            nbrs.append((x + 1, y))
        if y > 0:
            nbrs.append((x, y - 1))
        if y < self.height - 1:
            nbrs.append((x, y + 1))
        return nbrs

    def _resolve_offset(
        self, x: int, y: int, dx: int, dy: int
    ) -> Tuple[int, int] | None:
        if self.boundary_mode == "periodic":
            return ((x + dx) % self.width, (y + dy) % self.height)
        nx = x + dx
        ny = y + dy
        if 0 <= nx < self.width and 0 <= ny < self.height:
            return nx, ny
        return None

    def _movement_target(self, x: int, y: int) -> Tuple[int, int] | None:
        # decide stay vs move
        if self.rng.random() < self._stay_prob:
            return None

        # decide direction, probs must sum to 1
        idx = self.rng.choice(len(self._direction_offsets), p=self._direction_probs)
        dx, dy = self._direction_offsets[idx]
        return self._resolve_offset(x, y, dx, dy)

    def _random_move_step(self) -> None:
        for idx, c in enumerate(self.cells):
            if c.state != CellState.PRE_OB:
                continue
            if self.rng.random() >= self.cfg.move_prob:
                continue

            steps = 0
            while steps < self.cfg.max_moves_per_day:
                target = self._movement_target(c.x, c.y)
                steps += 1
                if target is None:
                    continue
                nx, ny = target
                if self.occupancy[ny, nx] == -1:
                    self.occupancy[c.y, c.x] = -1
                    self.occupancy[ny, nx] = idx
                    c.record_move(nx, ny)
                    break

    def _apoptosis_step(self, day: int) -> None:
        died_indices = []
        for idx, c in enumerate(self.cells):
            prob = (
                self.cfg.ob_apoptosis_prob
                if c.state == CellState.OSTEOBLAST
                else self.cfg.apoptosis_prob
            )
            if self.rng.random() < prob:
                # classify by state at time of death
                if c.state == CellState.PRE_OB:
                    self.dead_preob_coords.append((c.y, c.x))
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
            if (
                c.state == CellState.OSTEOBLAST
                and self.rng.random() >= self.cfg.ob_division_prob
            ):
                continue
            if (day - c.last_div_day) < c.division_interval:
                continue
            if self.rng.random() >= self.cfg.proliferation_chance:
                continue

            placed = False
            for _ in range(len(self._direction_offsets)):
                target = self._movement_target(c.x, c.y)
                if target is None:
                    continue
                nx, ny = target
                if self.occupancy[ny, nx] == -1:
                    child = self._spawn_cell(nx, ny, day)
                    newborns.append(child)
                    self.occupancy[ny, nx] = len(self.cells) + len(newborns) - 1
                    c.last_div_day = day
                    placed = True
                    break

            if not placed:
                c.last_div_day += 1

        for child in newborns:
            self.cells.append(child)
        self.occupancy[:] = -1
        for i, c in enumerate(self.cells):
            self.occupancy[c.y, c.x] = i

    def _secretion_step(self) -> None:
        if self.field is None:
            return
        n_ob = sum(1 for c in self.cells if c.state == CellState.OSTEOBLAST)
        self.field.request_collagen_secretion(n_ob)
        if n_ob <= 0:
            return
        self.field.add_alp(n_ob * self.cfg.alp_per_ob, ob_count=n_ob)
