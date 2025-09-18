from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import polars as pl
from .agent import Cell

if TYPE_CHECKING:
    from .pde_manager import PDEManager  # only for type hints


__all__ = ["cell_grids", "save_timeseries_excel"]


def cell_grids(h: int, w: int, agents: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (density_grid, state_grid) from agent positions.
    Assumes apoptotic cells are removed by ABM at end-of-day.
    """
    dens = np.zeros((h, w), dtype=np.uint16)
    state = np.zeros((h, w), dtype=np.uint8)
    for c in agents:
        if 0 <= c.y < h and 0 <= c.x < w:
            dens[c.y, c.x] = min(dens[c.y, c.x] + 1, np.iinfo(dens.dtype).max)
            state[c.y, c.x] = int(c.state)
    return dens, state





def save_timeseries_excel(df: pl.DataFrame, outfile: str | Path) -> None:
    """
    """
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.write_excel(outp)
        print(f"[I/O] time-series → {outp}")
    except Exception as e:
        alt = outp.with_suffix(".xlsx")
        df.write_xlsx(alt)
        print(f"[I/O] Excel write failed ({e.__class__.__name__}: {e}); wrote XLSX → {alt}")
