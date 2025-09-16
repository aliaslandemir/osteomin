from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import List, Tuple
import numpy as np

__all__ = ["CellState", "Cell"]

class CellState(IntEnum):
    ACTIVE = 1
    APOPTOTIC = 2
    OSTEOBLAST = 3

@dataclass(slots=True)
class Cell:
    x: int
    y: int
    day_created: int = 0
    state: CellState = CellState.ACTIVE
    division_interval: int | None = None
    trajectory: List[Tuple[int, int]] = field(default_factory=list, init=False)
    move_count: int = 0

    # lifetime flags (for cumulative stats)
    ever_ob: bool = False
    day_committed: int | None = None
    day_died: int | None = None

    def __post_init__(self) -> None:
        if self.division_interval is None:
            self.division_interval = int(np.random.choice([2, 3, 4]))
        self.trajectory.append((self.x, self.y))
