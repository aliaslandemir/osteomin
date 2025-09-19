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
    base_commit_prob: float = 0.5
    apoptosis_prob: float = 0.01
    min_div_days: int = 2
    max_div_days: int = 3
    collagen_per_ob: float = 1.0
    alp_per_ob: float = 1.0
    move_prob: float = 0.8
    max_moves_per_day: int = 1
    ob_division_prob: float = 0.0  # probability an OSTEOBLAST divides on a given check

@dataclass(slots=True)
class PDEConfig:
    width: int = 100
    height: int = 100
    media_interval: int = 2
