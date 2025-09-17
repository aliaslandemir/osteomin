from __future__ import annotations
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING
import numpy as np
import polars as pl
from .agent import Cell

if TYPE_CHECKING:
    from .pde_manager import PDEManager

__all__ = ["save_snapshot", "save_timeseries_excel", "save_timeseries_csv", "cell_grids"]

def cell_grids(h: int, w: int, agents: List[Cell]) -> Tuple[np.ndarray, np.ndarray]:
    """Return density and state grids from agent positions (live cells only if ABM removes apoptotic)."""
    dens = np.zeros((h, w), dtype=np.uint16)
    state = np.zeros((h, w), dtype=np.uint8)
    for c in agents:
        dens[c.y, c.x] += 1
        state[c.y, c.x] = int(c.state)
    return dens, state

def save_timeseries_excel(df: pl.DataFrame, outfile: str | Path) -> None:
    """Write daily metrics to Excel (.xlsx)."""
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.write_excel(outp)
    print(f"[I/O] time-series → {outp}")

def save_timeseries_csv(df: pl.DataFrame, outfile: str | Path) -> None:
    """Optionally write the same metrics to CSV."""
    outp = Path(outfile)
    outp.parent.mkdir(parents=True, exist_ok=True)
    df.write_csv(outp)
    print(f"[I/O] time-series → {outp}")

def save_snapshot(day: int, folder: str | Path, pde: "PDEManager", agents: List[Cell],
                  deaths_xy: List[Tuple[int, int]] | None = None) -> None:
    """
    Save raw field arrays and per-cell CSV for a given day.
    Also writes 'deaths_{day}.csv' if deaths_xy is provided and non-empty.
    """
    path = Path(folder); path.mkdir(parents=True, exist_ok=True)

    # Field arrays
    for name, arr in {
        "acp": pde.acp, "ha": pde.ha, "ca": pde.ca, "po4": pde.po4,
        "bgp": pde.bgp, "col": pde.collagen, "alp": pde.alp
    }.items():
        np.save(path / f"{name}_{day}.npy", arr)

    # Density + state maps from current agents
    dens, state = cell_grids(pde.height, pde.width, agents)
    np.save(path / f"density_{day}.npy", dens)
    np.save(path / f"cellstate_{day}.npy", state)

    # Per-cell CSV
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
    df_cells.write_csv(path / f"cells_{day}.csv")

    # Today's death positions (optional overlay for viz)
    if deaths_xy and len(deaths_xy) > 0:
        df_deaths = pl.DataFrame({"x": [x for x, _ in deaths_xy],
                                  "y": [y for _, y in deaths_xy]})
        df_deaths.write_csv(path / f"deaths_{day}.csv")

    print(f"[I/O] snapshot day {day} saved → {path}")