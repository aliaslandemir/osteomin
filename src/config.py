from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "PDEConfig"]


@dataclass(slots=True)
class ABMConfig:
    width: int = 100
    height: int = 100
    boundary_mode: str = "periodic"
    enable_random_move: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True
    base_commit_prob: float = 0.1
    commit_prob_growth: float = 0.02
    commit_delay: int = 0
    apoptosis_prob: float = 0.01
    ob_apoptosis_prob: float = 0.0005
    min_div_days: int = 2
    max_div_days: int = 4
    collagen_per_ob: float = 1.5
    alp_per_ob: float = 1.2
    move_prob: float = 1.0
    max_moves_per_day: int = 1
    ob_division_prob: float = 0.0
    proliferation_chance: float = 1.0

    # Mineralization thresholds/kinetics (legacy; PDEManager now owns them)
    S_THRESH: float = 0.15  # Ca*Pi supersaturation threshold
    K_NUC: float = 0.05  # ACP nucleation rate per day
    K_HA: float = 0.08  # ACP->HA conversion rate per day


@dataclass(slots=True)
class PDEConfig:
    width: int = 100
    height: int = 100
    media_interval: int = 2
