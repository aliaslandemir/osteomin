from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd
from .agent import Cell, CellState

__all__ = [
    "cell_grids",
    "coords_to_mask",
    "save_cell_history_excel",
    "save_timeseries_excel",
    "save_timeseries_csv",
]

def cell_grids(h: int, w: int, cells: List[Cell]):
    density = np.zeros((h, w), dtype=np.uint16)
    state   = np.zeros((h, w), dtype=np.uint8)
    for c in cells:
        density[c.y, c.x] = 1
        state[c.y, c.x] = int(c.state)
    return density, state

def coords_to_mask(h: int, w: int, coords: List[Tuple[int,int]]) -> np.ndarray:
    m = np.zeros((h, w), dtype=np.uint8)
    for (row, col) in coords:
        if 0 <= row < h and 0 <= col < w:
            m[row, col] = 1
    return m

def save_timeseries_excel(df: pd.DataFrame, outfile: str | Path) -> None:
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outp, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="timeseries")

def save_timeseries_csv(df: pd.DataFrame, outfile: str | Path) -> None:
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(outp, index=False)


def save_cell_history_excel(df: pd.DataFrame, outfile: str | Path) -> None:
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(outp, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="cells")
