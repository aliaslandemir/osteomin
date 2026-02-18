from __future__ import annotations
from typing import Any, Dict, List, Tuple

import numpy as np

from .agent import Cell, CellState
from .config import ABMConfig
from .time_utils import HOURS_PER_DAY

__all__ = ["ABMManager"]


class ABMManager:
    """On-lattice ABM with hourly stepping and event tracking lists."""

    def __init__(
        self, cfg: ABMConfig, rng: np.random.Generator, track_cells: bool = False
    ):
        self.cfg = cfg
        self.rng = rng
        self.width, self.height = int(cfg.width), int(cfg.height)
        mode = getattr(cfg, "boundary_mode", "clamp")
        mode = mode.lower() if isinstance(mode, str) else "clamp"
        if mode not in ("clamp", "periodic"):
            raise ValueError(f"unsupported boundary_mode '{cfg.boundary_mode}'")
        self.boundary_mode = mode

        self.cells: List[Cell] = []
        self.occupancy = np.full((self.height, self.width), -1, dtype=np.int32)
        self.track_cells = bool(track_cells)
        self._next_cell_id = 0
        self._death_rows: List[Dict[str, Any]] = []
        self._move_prob = 0.67  # ~20 µm/h on a 30 µm lattice, matching MC3T3-E1 controls

        self._direction_offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (1, -1),
            (-1, 1),
            (1, 1),
        ]

        base_axial = 1.0
        base_diag = 1.0 / np.sqrt(2)

        direction_weights = [base_axial] * 4 + [base_diag] * 4
        total = sum(direction_weights)

        self._direction_probs = [w / total for w in direction_weights]

        self.dead_cells: List[Tuple[int, int]] = []  # (row, col)
        self.dead_preob_coords: List[Tuple[int, int]] = []  # died while preOB
        self.dead_ob_coords: List[Tuple[int, int]] = []  # died while OB
        self.committed_today: int = 0
        self._move_prob_hourly = min(1.0, max(0.0, float(self.cfg.move_prob)))
        self._max_moves_per_step = max(0, int(self.cfg.max_moves_per_hour))
        self._apoptosis_prob_preob = min(1.0, max(0.0, float(self.cfg.apoptosis_prob)))
        self._apoptosis_prob_ob = min(1.0, max(0.0, float(self.cfg.ob_apoptosis_prob)))
        self._proliferation_chance_hourly = min(
            1.0, max(0.0, float(self.cfg.proliferation_chance))
        )
        self._commit_growth_per_hour = max(0.0, float(self.cfg.commit_prob_growth))
        self._commit_delay_hours = max(0, int(self.cfg.commit_delay))
        self._min_div_interval = max(1, int(self.cfg.min_div_hours))
        self._max_div_interval = max(self._min_div_interval, int(self.cfg.max_div_hours))

        self.field = None

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
        uid = self._next_cell_id
        self._next_cell_id += 1
        c = Cell(
            x=x,
            y=y,
            unique_id=uid,
            state=CellState.PRE_OB,
            last_div_day=day,
            birth_day=day,
        )
        if self._min_div_interval <= self._max_div_interval:
            c.division_interval = int(
                self.rng.integers(self._min_div_interval, self._max_div_interval + 1)
            )
        return c

    def attach_field(self, field) -> None:
        self.field = field

    def step_hour(self, hour: int) -> None:
        self.dead_cells = []
        self.dead_preob_coords = []
        self.dead_ob_coords = []
        self.committed_today = 0

        if self.cfg.enable_random_move:
            self._random_move_step()

        if self.cfg.enable_proliferation:
            self._proliferation_step(hour)

        if self.cfg.enable_commitment and hour >= self._commit_delay_hours:
            base_prob = max(
                0.0,
                float(self.cfg.base_commit_prob + hour * self._commit_growth_per_hour),
            )
            commit_prob = min(1.0, base_prob)
            for c in self.cells:
                if (
                    c.state == CellState.PRE_OB
                    and (hour - c.last_div_day) >= HOURS_PER_DAY
                    and self.rng.random() < commit_prob
                ):
                    c.mark_committed(hour)
                    self.committed_today += 1

        if self.cfg.enable_apoptosis:
            self._apoptosis_step(hour)

        if self.cfg.enable_secretion:
            self._secretion_step()

        if self.field is not None:
            n_preob = sum(1 for c in self.cells if c.state == CellState.PRE_OB)
            n_ob = sum(1 for c in self.cells if c.state == CellState.OSTEOBLAST)
            ob_mask = np.zeros((self.height, self.width), dtype=bool)
            preob_mask = np.zeros((self.height, self.width), dtype=bool)
            for c in self.cells:
                if c.state == CellState.OSTEOBLAST:
                    ob_mask[c.y, c.x] = True
                elif c.state == CellState.PRE_OB:
                    preob_mask[c.y, c.x] = True
            if hasattr(self.field, "set_ob_mask"):
                self.field.set_ob_mask(ob_mask)
            if hasattr(self.field, "set_preob_mask"):
                self.field.set_preob_mask(preob_mask)
            self.field.step_hour(hour, n_preob, n_ob)

    def _neighbors4(self, x: int, y: int):
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
        if self.rng.random() >= self._move_prob:
            return None

        idx = self.rng.choice(len(self._direction_offsets), p=self._direction_probs)
        dx, dy = self._direction_offsets[idx]
        return self._resolve_offset(x, y, dx, dy)

    def _random_move_step(self) -> None:
        for idx, c in enumerate(self.cells):
            if c.state != CellState.PRE_OB:
                continue
            if self.rng.random() >= self._move_prob_hourly:
                continue

            steps = 0
            while steps < self._max_moves_per_step:
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

    def _apoptosis_step(self, hour: int) -> None:
        prob_pre = min(1.0, self._apoptosis_prob_preob)
        prob_ob = min(1.0, self._apoptosis_prob_ob)

        died_indices = []
        for idx, c in enumerate(self.cells):
            prob = prob_ob if c.state == CellState.OSTEOBLAST else prob_pre
            if self.rng.random() < prob:
                state_before = c.state
                if state_before == CellState.PRE_OB:
                    self.dead_preob_coords.append((c.y, c.x))
                elif state_before == CellState.OSTEOBLAST:
                    self.dead_ob_coords.append((c.y, c.x))
                self.dead_cells.append((c.y, c.x))
                c.mark_apoptotic(hour)
                if self.track_cells:
                    self._record_death_row(c, hour, state_before)
                died_indices.append(idx)

        if not died_indices:
            return

        keep = [i for i in range(len(self.cells)) if i not in died_indices]
        self.cells = [self.cells[i] for i in keep]
        self.occupancy[:] = -1
        for i, c in enumerate(self.cells):
            self.occupancy[c.y, c.x] = i

    def _proliferation_step(self, hour: int) -> None:
        newborns: List[Cell] = []
        for idx, c in enumerate(self.cells):
            if c.state == CellState.APOPTOTIC:
                continue
            if (
                c.state == CellState.OSTEOBLAST
                and self.rng.random() >= self.cfg.ob_division_prob
            ):
                continue
            if (hour - c.last_div_day) < c.division_interval:
                continue
            if self.rng.random() >= self._proliferation_chance_hourly:
                continue

            placed = False
            for _ in range(len(self._direction_offsets)):
                target = self._movement_target(c.x, c.y)
                if target is None:
                    continue
                nx, ny = target
                if self.occupancy[ny, nx] == -1:
                    child = self._spawn_cell(nx, ny, hour)
                    newborns.append(child)
                    self.occupancy[ny, nx] = len(self.cells) + len(newborns) - 1
                    c.last_div_day = hour
                    placed = True
                    break

            if not placed:
                c.last_div_day += 1

        for child in newborns:
            self.cells.append(child)
        self.occupancy[:] = -1
        for i, c in enumerate(self.cells):
            self.occupancy[c.y, c.x] = i

    def _record_death_row(self, cell: Cell, hour: int, state_before: CellState) -> None:
        if not self.track_cells:
            return
        day_index = int((hour - 1) // HOURS_PER_DAY + 1) if hour > 0 else 0
        self._death_rows.append(
            {
                "cell_id": int(cell.unique_id),
                "day": day_index,
                "hour": int(hour),
                "row": int(cell.y),
                "col": int(cell.x),
                "state": int(state_before),
                "state_name": state_before.name,
                "alive": False,
                "ever_ob": bool(cell.ever_ob),
                "birth_day": int(cell.birth_day),
                "day_committed": (
                    int(cell.day_committed) if cell.day_committed is not None else None
                ),
                "day_died": int(cell.day_died) if cell.day_died is not None else None,
                "last_div_day": int(cell.last_div_day),
                "move_count": int(cell.move_count),
            }
        )

    def drain_death_rows(self) -> List[Dict[str, Any]]:
        if not self.track_cells:
            return []
        rows = self._death_rows
        self._death_rows = []
        return rows

    def _secretion_step(self) -> None:
        if self.field is None:
            return
        if getattr(self.field, "datafit_schedule", None) is not None:
            return
        n_ob = sum(1 for c in self.cells if c.state == CellState.OSTEOBLAST)
        self.field.request_collagen_secretion(n_ob)
        if n_ob <= 0:
            return
        self.field.add_alp(n_ob * self.cfg.alp_per_ob, ob_count=n_ob)
