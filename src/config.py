from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "PDEConfig"]

@dataclass(slots=True)
class ABMConfig:
    width: int = 100
    height: int = 100
    enable_migration: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True
    base_commit_prob: float = 0.10
    apoptosis_prob: float = 0.01
    min_div_days: int = 2
    max_div_days: int = 3
    collagen_per_ob: float = 1.5
    alp_per_ob: float = 1.2
    move_prob: float = 0.8
    max_moves_per_day: int = 1
    ob_division_prob: float = 0.0

    # Mineralization thresholds/kinetics
    C_THRESH: float = 2.0         # collagen threshold (a.u.)
    S_THRESH: float = 0.15        # Ca*Pi supersaturation threshold
    K_NUC: float = 0.05           # ACP nucleation rate per day
    K_HA: float = 0.08            # ACP->HA conversion rate per day
    NUC_RADIUS: int = 0           # neighborhood (Moore) around OBs for nucleation
    CA_PER_NUC: float = 0.02      # Ca consumed per nucleation step
    PI_PER_NUC: float = 0.02      # Pi consumed per nucleation step


@dataclass(slots=True)
class PDEConfig:
    width: int = 100
    height: int = 100
    media_interval: int = 2
