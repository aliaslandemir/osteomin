from __future__ import annotations
from dataclasses import dataclass

__all__ = ["PDEConfig", "ABMConfig"]

@dataclass(frozen=True, slots=True)
class PDEConfig:
    """
    Dimensionless but mechanistic defaults for 2-D coverslips.
    One node ≈ 10 µm. 100×100 grid → ~1 mm² field.

    Baseline chemistry:
      - Keep Ca relatively high and PO4 modest so ALP + βGP raises PO4
        and drives supersaturation (S>1) → ACP → HA.
    """
    # --- Grid & numerics ---
    width: int = 100
    height: int = 100
    dx: float = 10.0          # microns per node (informational)
    pde_substeps: int = 1     # no diffusion → no subcycling

    # --- Medium baselines (α-MEM no AA + 10% FBS) ---
    ca_bc_value: float = 0.60     # Ca2+ reservoir (scaled)
    po4_bc_value: float = 0.20    # modest phosphate baseline
    bgp_init: float = 1.00        # β-glycerophosphate pool (reset on media change)

    # --- Diffusion OFF (explicit) ---
    D_ca: float = 0.0
    D_po4: float = 0.0
    D_bgp: float = 0.0
    D_collagen: float = 0.0

    # --- ECM & ALP turnover ---
    collagen_prod_cap: float = 4.0   # allow matrix build-up before soft cap
    collagen_decay: float = 0.001    # per-day fractional loss
    alp_decay: float = 0.01          # per-day fractional loss

    # --- Nucleation & maturation kinetics (dimensionless) ---
    # ACP nucleation ~ Hill in supersaturation excess
    k_acp_nucl: float = 0.010
    Km_supersat: float = 0.30
    n_supersat: float = 2.0

    # ACP → HA maturation (MK-4 scaled in PDE step)
    k_mature: float = 0.015
    Km_mature: float = 0.02
    n_mature: float = 2.5

    # --- Supplement consumption coupling (bookkeeping) ---
    aa_consume_per_collagen: float = 0.20
    dexa_consume_per_alp: float = 0.10
    mk4_consume_per_dHA: float = 0.05


@dataclass(frozen=True, slots=True)
class ABMConfig:
    """
    One node = one cell.
    Division only allowed for ACTIVE cells (osteoblasts do NOT divide).
    """
    width: int = 100
    height: int = 100

    # --- Population & crowding ---
    max_cell_density: int = 2    # live cells per node
    apoptosis_prob: float = 0.01

    # --- Migration (daily step probability) ---
    enable_migration: bool = True
    p_move: float = 0.20         # ~20% chance to hop 1 px per day

    # --- Cell cycle (ACTIVE only) ---
    # MC3T3-E1 ~24–36 h early; we keep 2–3 d to be conservative in daily-step model
    min_div_days: int = 2
    max_div_days: int = 3
