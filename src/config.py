from __future__ import annotations
from dataclasses import dataclass

__all__ = ["ABMConfig", "FieldConfig"]


@dataclass(slots=True)
class ABMConfig:
    width: int = 33
    height: int = 33
    boundary_mode: str = "periodic"
    update_order: str = "random"
    enable_random_move: bool = True
    enable_commitment: bool = True
    enable_apoptosis: bool = True
    enable_proliferation: bool = True
    enable_secretion: bool = True

    base_commit_prob: float = 0.0005
    osteo_induction_hours: int = 72
    commit_delay: int = 48
    commit_k_hourly: float = 0.0125
    apoptosis_prob: float = 1e-3
    ob_apoptosis_prob: float = 0.0
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
    dex_ref_conc: float = 22.7585
    mk4_ref_conc: float = 0.78289
    alp_basal_rate: float = 0.0949
    alp_osteo_rate: float = 0.810
    alp_initial: float = 0.0
    alp_dex_ref_conc: float = 22.7585
    alp_mk4_ref_conc: float = 0.78289
    alp_a_d: float = 3.72321
    alp_b_d: float = 1.75
    alp_a_k: float = 1.09529
    alp_b_k: float = 1.05833
    alp_alpha_d: float = 1.0
    alp_alpha_k: float = 1.0
    alp_alpha_dk: float = 0.0
    collagen_basal_rate: float = 0.00336
    collagen_osteo_rate: float = 0.0130
    collagen_dex_ref_conc: float = 17.2714
    collagen_mk4_ref_conc: float = 0.347951
    collagen_a_d: float = 2.64583
    collagen_b_d: float = 0.813749
    collagen_a_k: float = 4.66313
    collagen_b_k: float = 2.14517
    collagen_alpha_d: float = 1.0
    collagen_alpha_k: float = 1.0
    collagen_alpha_dk: float = 0.0
    aa_initial: float = 1.0
    aa_half_life_hours: float = 6.0
    collagen_aa_half_sat: float = 1.0
    ha_model: str = "ha"
    ha_initial: float = 2.34
    ha_osteo_rate: float = 0.102
    ha_dex_ref_conc: float = 67.7871
    ha_mk4_ref_conc: float = 0.669239
    ha_a_d: float = 1.1527
    ha_b_d: float = 0.345121
    ha_a_k: float = 4.66541
    ha_b_k: float = 2.15
    ha_alpha_d: float = 1.0
    ha_alpha_k: float = 1.0
    ha_alpha_dk: float = 0.0
    ha_tau_hours: float = 384.0
    ha_hill_n: float = 2.0
    ha_t_commit_hours: float = 120.0
    ha_collagen_maturation_tau_hours: float = 48.0
    ha_collagen_half_sat: float = 1.0
