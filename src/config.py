from __future__ import annotations
from dataclasses import dataclass


__all__ = ["PDEConfig", "ABMConfig"]


# Keep as-is if your PDE module already consumes this.
@dataclass(frozen=True, slots=True)
class PDEConfig:
    width: int = 100
    height: int = 100
    # … add only what your existing PDEManager expects (no extras here).


@dataclass(frozen=True, slots=True)
class ABMConfig:
    # Lattice
    width: int = 100
    height: int = 100
    max_cell_density: int = 1  # cells per node (usually 1 in on-lattice ABM)

    # Global switches (mapped from CLI)
    enable_migration: bool = True
    enable_commitment: bool = True          # osteogenic commitment
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True

    # Movement
    p_move: float = 0.20  # probability per day per cell (if migration enabled)

    # Proliferation
    min_div_days: int = 2
    max_div_days: int = 3

    # Apoptosis
    apoptosis_prob: float = 0.005  # probability per day per cell

    # Commitment (example scalar that your ABM uses; do not invent new biology here)
    base_commit_prob: float = 0.3  # baseline daily chance for ACTIVE → OSTEOBLAST

    # Secretion scaling (used only if enable_secretion=True)
    collagen_per_ob: float = 1.0
    alp_per_ob: float = 1.0
