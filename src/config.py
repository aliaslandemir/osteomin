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
    osteo_induction_hours: int = 72
    commit_delay: int = 48
    commit_k_day: float = 0.06
    commit_k_growth_per_hour: float = 0.0
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
    enable_secretion: bool = True
    osteo_induction_hours: int = 72
    dex_conc: float = 0.0
    mk4_conc: float = 0.0
    alp_basal_rate: float = 0.0
    alp_osteo_rate: float = 0.0
    alp_kappa_d: float = 0.0
    alp_kappa_k: float = 0.0
    collagen_basal_rate: float = 0.0
    collagen_osteo_rate: float = 0.0
    collagen_kappa_d: float = 0.0
    collagen_kappa_k: float = 0.0
    aa_initial: float = 1.0
    aa_half_life_hours: float = 6.0
    collagen_aa_half_sat: float = 1.0
    ha_model: str = "minimal"
    ha_osteo_rate: float = 0.0
    ha_kappa_d: float = 0.0
    ha_kappa_k: float = 0.0
    ha_tau_hours: float = 384.0
    ha_hill_n: float = 4.0
    ha_t_commit_hours: float = 120.0
    ha_collagen_maturation_tau_hours: float = 48.0
    ha_collagen_half_sat: float = 1.0
