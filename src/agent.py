from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple, Optional

__all__ = ["CellState", "Cell"]

class CellState(IntEnum):
    EMPTY = 0
    ACTIVE = 1
    APOPTOTIC = 2
    OSTEOBLAST = 3

@dataclass(slots=True)
class Cell:
    x: int
    y: int
    state: CellState = CellState.ACTIVE
    last_div_day: int = 0
    division_interval: int = 3
    move_count: int = 0
    trajectory: List[Tuple[int,int]] = field(default_factory=list)
    ever_ob: bool = False
    day_committed: Optional[int] = None
    day_died: Optional[int] = None

    def __post_init__(self) -> None:
        self.trajectory.append((self.x, self.y))

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
