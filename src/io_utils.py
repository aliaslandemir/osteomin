from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import polars as pl
from .agent import Cell

if TYPE_CHECKING:
    from .pde_manager import PDEManager

__all__ = ["save_snapshot", "save_timeseries", "cell_grids"]

def cell_grids(h: int, w: int, agents: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    dens = np.zeros((h, w), dtype=np.uint16)
    state = np.zeros((h, w), dtype=np.uint8)
    for c in agents:
        dens[c.y, c.x] += 1
        state[c.y, c.x] = int(c.state)
    return dens, state

def save_timeseries(df: pl.DataFrame, outfile: str | Path) -> None:
    """
    Save daily time-series both as Parquet and CSV (same basename).
    """
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    # Parquet
    df.write_parquet(str(outp))
    # CSV
    csv_path = outp.with_suffix(".csv")
    df.write_csv(str(csv_path))
    print(f"[I/O] time-series → {outp} and {csv_path.name}")

def save_snapshot(day: int, folder: str | Path, pde: "PDEManager", agents: List[Cell]) -> None:
    path = Path(folder); path.mkdir(parents=True, exist_ok=True)
    np.save(path / f"acp_{day}.npy", pde.acp)
    np.save(path / f"ha_{day}.npy",  pde.ha)
    np.save(path / f"ca_{day}.npy",  pde.ca)
    np.save(path / f"po4_{day}.npy", pde.po4)
    np.save(path / f"bgp_{day}.npy", pde.bgp)
    np.save(path / f"col_{day}.npy", pde.collagen)
    np.save(path / f"alp_{day}.npy", pde.alp)

    dens, state = cell_grids(pde.height, pde.width, agents)
    np.save(path / f"density_{day}.npy",   dens)
    np.save(path / f"cellstate_{day}.npy", state)

    df_cells = pl.DataFrame({
        "x": [c.x for c in agents],
        "y": [c.y for c in agents],
        "state": [int(c.state) for c in agents],
        "moves": [c.move_count for c in agents],
        "ever_ob": [c.ever_ob for c in agents],
        "day_committed": [(-1 if c.day_committed is None else c.day_committed) for c in agents],
        "day_died": [(-1 if c.day_died is None else c.day_died) for c in agents],
        "day_created": [c.day_created for c in agents],
    })
    df_cells.write_parquet(path / f"cells_{day}.parquet")
    print(f"[I/O] snapshot day {day} saved.")
