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
    base_commit_prob: float = 0.0001
    commit_prob_growth: float = 0.01
    commit_delay: int = 0
    apoptosis_prob: float = 0.01
    ob_apoptosis_prob: float = 0.001
    min_div_days: int = 2
    max_div_days: int = 4
    dex_dose: float = 0.0  # CLI-provided dexamethasone dose (0–1); multiplier applied in commitment
    dex_k: float = 1.0  # slope for dex effect (F_dex = 1 + k * D)
    collagen_per_ob: float = 1.5
    alp_per_ob: float = 1.2
    move_prob: float = 1.0
    max_moves_per_day: int = 1
    ob_division_prob: float = 0.0
    proliferation_chance: float = 1.0

    # Mineralization thresholds/kinetics (legacy; FieldManager now owns them)
    S_THRESH: float = 0.01  # Ca*Pi supersaturation threshold
    K_NUC: float = 0.01  # ACP nucleation rate per day
    K_HA: float = 0.04  # ACP->HA conversion rate per day


@dataclass(slots=True)
class FieldConfig:
    width: int = 33
    height: int = 33
    media_interval: int = 2
    enable_nucleation: bool = True
    enable_nodules: bool = True
    s_thresh: float = 0.01
    k_nuc: float = 0.01
    k_ha: float = 0.04
    nodule_window: int = 4
    ca0: float = 1.0
    pi0: float = 0.0
    alp_to_pi_rate: float = 0.01
    ca_per_acp: float = 3.0
    pi_per_acp: float = 2.0
    acp_to_ha_threshold: float = 0.05
    collagen_nucleation_threshold: float = 1e-6
    acp_per_ha_reaction: float = 3.0
    ca_per_reaction: float = 1.0
    ha_per_reaction: float = 2.0
    h_plus_per_ha: float = 1.0
    kw: float = 1.0
    h_base: float = 1.0
    oh_base: float = 1.0
    aa_usage_per_collagen: float = 0.05
    bgp_usage_per_pi: float = 3e-4
    base_collagen_rate: float = 0.001
    k_aa: float = 0.2
