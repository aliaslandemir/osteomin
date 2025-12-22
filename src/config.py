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
    base_commit_prob: float = 0.0001 / 24  # per hour
    commit_prob_growth: float = 0.01 / (24 * 24)  # per hour increment (matches legacy 0.01/day slope)
    commit_delay: int = 72  # hours (3 days legacy)
    apoptosis_prob: float = 0.01 / 24  # per hour
    ob_apoptosis_prob: float = 0.001 / 24  # per hour
    min_div_days: int = 48  # hours (legacy name)
    max_div_days: int = 96  # hours (legacy name)
    dex_dose: float = 0.0  # CLI-provided dexamethasone dose (0–1); multiplier applied in commitment
    dex_k: float = 1.0  # slope for dex effect (F_dex = 1 + k * D)
    collagen_per_ob: float = 1.5
    alp_per_ob: float = 1.2
    move_prob: float = 1.0  # per hour
    max_moves_per_day: int = 1  # interpreted hourly
    ob_division_prob: float = 0.0  # per hour
    proliferation_chance: float = 1.0  # per hour

    # Mineralization thresholds/kinetics (legacy; FieldManager now owns them)
    S_THRESH: float = 0.01  # Ca*Pi supersaturation threshold
    K_NUC: float = 0.025 / 24  # ACP nucleation rate per hour
    K_HA: float = 0.06 / 24  # ACP->HA conversion rate per hour


@dataclass(slots=True)
class FieldConfig:
    width: int = 33
    height: int = 33
    media_interval: int = 48  # hours between media refresh (legacy default 2 days)
    enable_nucleation: bool = True
    enable_nodules: bool = True
    s_thresh: float = 0.01
    k_nuc: float = 0.025 / 24  # per hour
    k_ha: float = 0.06 / 24  # per hour
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
