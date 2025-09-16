from __future__ import annotations
from dataclasses import dataclass

__all__ = ["PDEConfig", "ABMConfig"]

@dataclass(frozen=True, slots=True)
class PDEConfig:
    # grid & numerics (dimensionless; 1 day = 1 time unit)
    width: int = 120
    height: int = 120
    dx: float = 1.0
    pde_substeps: int = 1          # no diffusion → no need for subcycling

    # α-MEM + 10% FBS → Ca source (scaled), α-MEM (no AA) baseline PO4 small
    ca_bc_value: float = 1.0       # freely available Ca²⁺ (reservoir)
    po4_bc_value: float = 0.05     # small free phosphate in medium
    bgp_init: float = 1.0          # β-GP substrate initial (replenished at media change)

    # diffusion OFF (keep for clarity)
    D_ca: float = 0.0
    D_po4: float = 0.0
    D_bgp: float = 0.0
    D_collagen: float = 0.0

    # collagen dynamics
    collagen_prod_cap: float = 1.0  # soft cap for ECM accumulation
    collagen_decay: float = 0.001

    # nucleation & maturation kinetics
    k_acp_nucl: float = 0.010
    Km_supersat: float = 0.30
    n_supersat: float = 2.0

    k_mature: float = 0.015         # ACP→HA (scaled by MK4 level)
    Km_mature: float = 0.02
    n_mature: float = 2.5

    # ALP turnover
    alp_decay: float = 0.01         # per-day decay to bound ALP

    # consumption couplings (dimensionless scalings)
    aa_consume_per_collagen: float = 0.20   # AA decreases with collagen deposition
    dexa_consume_per_alp: float = 0.10      # Dexa decreases with ALP secretion
    mk4_consume_per_dHA: float = 0.05       # MK4 decreases with new HA formation

@dataclass(frozen=True, slots=True)
class ABMConfig:
    width: int = 120
    height: int = 120

    # population
    max_cell_density: int = 3
    apoptosis_prob: float = 0.003

    # migration
    enable_migration: bool = True
    p_move: float = 0.02

    # cell cycle (MC3T3-E1 ~24–36h early); keep 2–4 d to be conservative
    min_div_days: int = 2
    max_div_days: int = 4
    prolif_until_day: int = 7
