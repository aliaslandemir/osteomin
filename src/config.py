from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "FieldConfig"]


@dataclass(slots=True)
class ABMConfig:
    # Timebase: parameters below are per hour unless explicitly stated otherwise.
    width: int = 33
    height: int = 33
    boundary_mode: str = "periodic"
    enable_random_move: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True

    base_commit_prob: float = (
        0.0005  # per hour (~0.012/day), low baseline for early pre-OB commitment
    )
    commit_prob_growth: float = (
        0.00003  # per hour (~0.7/day slope) toward higher commitment over ~10 days
    )

    commit_delay: int = 72  # hours (legacy 3 days)
    apoptosis_prob: float = 0.01 / 24  # per hour
    ob_apoptosis_prob: float = 0.001 / 24  # per hour
    min_div_hours: int = 38  # hours between divisions
    max_div_hours: int = 52  # hours between divisions
    alp_per_ob: float = 1.2
    move_prob: float = 0.27  # per hour; Refaaq2020
    max_moves_per_hour: int = 1  # max moves per hour
    ob_division_prob: float = 0.0  # per hour
    proliferation_chance: float = 1.0  # per hour

    # Mineralization thresholds/kinetics (legacy; FieldManager now owns them)


@dataclass(slots=True)
class FieldConfig:
    width: int = 33
    height: int = 33
    media_interval: int = 48  # hours between media refresh (legacy default 2 days)
