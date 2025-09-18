
from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "PDEConfig"]

@dataclass(slots=True)
class ABMConfig:
    width: int = 100
    height: int = 100

    # toggles
    enable_migration: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True

    # parameters
    base_commit_prob: float = 0.30  # per-day chance ACTIVE -> OSTEOBLAST
    apoptosis_prob: float = 0.007    # per-day death chance
    min_div_days: int = 2
    max_div_days: int = 3

    # secretion (per OB per day, arbitrary units)
    collagen_per_ob: float = 1.0
    alp_per_ob: float = 1.0

    # migration controls
    move_prob: float = 0.8         # chance an ACTIVE cell attempts a move on a day
    max_moves_per_day: int = 2     # number of steps a cell may take per day

@dataclass(slots=True)
class PDEConfig:
    width: int = 100
    height: int = 100
    media_interval: int = 2  # days between media refreshes
