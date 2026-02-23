from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "FieldConfig"]


@dataclass(slots=True)
class ABMConfig:
    width: int = 33
    height: int = 33
    boundary_mode: str = "periodic"
    enable_random_move: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True

    base_commit_prob: float = 0.0005
    commit_prob_growth: float = 0.00003
    commit_delay: int = 72
    apoptosis_prob: float = 0.01 / 24
    ob_apoptosis_prob: float = 0.001 / 24
    min_div_hours: int = 38
    max_div_hours: int = 52
    alp_per_ob: float = 1.2
    move_prob: float = 0.27
    max_moves_per_hour: int = 1
    ob_division_prob: float = 0.0
    proliferation_chance: float = 1.0


@dataclass(slots=True)
class FieldConfig:
    width: int = 33
    height: int = 33
    media_interval: int = 48
    ha_model: str = "minimal"
    ha_tau_hours: float = 384.0
    ha_hill_n: float = 4.0
    ha_t_commit_hours: float = 72.0
