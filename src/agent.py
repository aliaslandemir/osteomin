from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional


__all__ = ["CellState", "Cell"]


class CellState(IntEnum):
    ACTIVE = 1        # proliferative, motile
    APOPTOTIC = 2     # flagged for removal (removed same day by ABM)
    OSTEOBLAST = 3    # committed; non-proliferative, may secrete


@dataclass(slots=True)
class Cell:
    x: int
    y: int
    state: CellState = CellState.ACTIVE

    # Time bookkeeping (in days, aligned to simulation day)
    age_days: int = 0
    last_div_day: Optional[int] = None
    division_interval: Optional[int] = None  # sampled by ABM on spawn

    # Tracking (for diagnostics/plots)
    trajectory: List[Tuple[int, int]] = field(default_factory=list, init=False)
    move_count: int = 0

    # Lifetime flags (for cumulative stats)
    ever_ob: bool = False
    day_committed: Optional[int] = None
    day_died: Optional[int] = None

    def __post_init__(self) -> None:
        # ABMManager appends subsequent positions on movement.
        self.trajectory.append((self.x, self.y))

    # Lightweight helpers (logic lives in ABMManager)
    def mark_committed(self, day: int) -> None:
        if self.state != CellState.OSTEOBLAST:
            self.state = CellState.OSTEOBLAST
            self.ever_ob = True
            self.day_committed = day

    def mark_apoptotic(self, day: int) -> None:
        if self.state != CellState.APOPTOTIC:
            self.state = CellState.APOPTOTIC
            self.day_died = day

    def record_move(self, new_x: int, new_y: int) -> None:
        self.x, self.y = new_x, new_y
        self.move_count += 1
        self.trajectory.append((self.x, self.y))
