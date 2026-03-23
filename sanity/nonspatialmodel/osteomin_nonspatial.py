#!/usr/bin/env python3
from __future__ import annotations

"""Reduced nonspatial OsteoMin model for ablation analysis.

This module implements a well-mixed (population-level) version of the OsteoMin
logic used in the spatial model, keeping the same biological mechanisms:

- preOB proliferation and apoptosis
- delayed commitment from preOB to OB
- cumulative ALP production
- cumulative collagen production with ascorbate gating
- collagen maturation with first-order relaxation (tau_col)
- cumulative mineral/HAp from mature collagen and OB commitment age
- Dex and MK-4 multiplicative modulation factors

The model uses an expected-value hourly update (deterministic), over 28 days.
"""

import math
from dataclasses import dataclass, field, replace
from typing import Dict, Iterable, List

import pandas as pd

HOURS_PER_DAY = 24
SIM_DAYS = 28
SIM_HOURS = SIM_DAYS * HOURS_PER_DAY


@dataclass(slots=True)
class DrugResponseParams:
    """Dose-response coefficients for one endpoint channel.

    Manuscript mapping: multiplicative modulation factors for Dex and MK-4.
    """

    dex_ref: float
    dex_a: float
    dex_b: float
    mk4_ref: float
    mk4_a: float
    mk4_b: float


@dataclass(slots=True)
class NonSpatialParams:
    """Parameters for reduced nonspatial OsteoMin simulation."""

    total_hours: int = SIM_HOURS

    # Initial populations
    n_preob0: float = 72.0
    n_ob0: float = 0.0

    # Cell kinetics (hourly expected-value hazards)
    prolif_rate_preob: float = 1.0 / 45.0  # ~45 h effective cycle
    prolif_rate_ob: float = 0.0
    apoptosis_preob: float = 1e-3
    apoptosis_ob: float = 0.0

    # Commitment: starts after induction + delay, hazard k_commit per hour
    osteo_induction_hours: int = 72
    commit_delay_hours: int = 48
    k_commit_hourly: float = 0.0125

    # Drug concentrations
    dex_conc_nm: float = 0.0
    mk4_conc_um: float = 0.0

    # ALP rates (cumulative)
    alp_basal_rate: float = 0.10710412147505423
    alp_osteo_rate: float = 0.9856290672451192
    alp_initial: float = 0.8

    # Collagen rates (cumulative)
    collagen_basal_rate: float = 0.0035518253599441936
    collagen_osteo_rate: float = 0.0130876

    # Ascorbate gating for collagen
    aa_initial: float = 50.0
    aa_half_life_hours: float = 48.0
    aa_media_interval_hours: int = 48
    collagen_aa_half_sat: float = 10.0

    # Collagen maturation and mineral formation
    tau_col_hours: float = 48.0
    ha_osteo_rate: float = 24.147854915293642
    ha_tau_hours: float = 384.0
    ha_hill_n: float = 4.0
    ha_collagen_half_sat: float = 1.0

    # Dose-response families
    alp_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=20.0,
            dex_a=3.79,
            dex_b=1.92,
            mk4_ref=0.6,
            mk4_a=0.56,
            mk4_b=0.471,
        )
    )
    collagen_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=60.0,
            dex_a=2.42,
            dex_b=0.770,
            mk4_ref=8.0,
            mk4_a=0.915,
            mk4_b=0.167,
        )
    )
    ha_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=50.0,
            dex_a=1.14,
            dex_b=0.365,
            mk4_ref=0.7,
            mk4_a=4.69,
            mk4_b=2.21,
        )
    )


@dataclass(slots=True)
class AblationResult:
    """Container for one ablation timeseries and endpoint summary."""

    name: str
    timeseries: pd.DataFrame
    endpoints: Dict[str, float]


def _log_offset(conc: float, ref: float) -> float:
    if conc <= 0.0:
        return 0.0
    return float(math.log10(conc / max(1e-12, ref)))


def _drug_modulation(conc_dex: float, conc_mk4: float, resp: DrugResponseParams) -> float:
    """Dex×MK-4 multiplicative modulation."""
    i_d = 1.0 if conc_dex > 0.0 else 0.0
    i_k = 1.0 if conc_mk4 > 0.0 else 0.0
    x_d = _log_offset(conc_dex, resp.dex_ref)
    x_k = _log_offset(conc_mk4, resp.mk4_ref)
    m_d = max(0.0, 1.0 + i_d * (resp.dex_a - resp.dex_b * x_d * x_d))
    m_k = max(0.0, 1.0 + i_k * (resp.mk4_a - resp.mk4_b * x_k * x_k))
    return float(m_d * m_k)


def _aa_concentration(cfg: NonSpatialParams, hour: int) -> float:
    if cfg.aa_initial <= 0.0 or hour < cfg.osteo_induction_hours:
        return 0.0
    k = math.log(2.0) / max(1e-9, cfg.aa_half_life_hours)
    if cfg.aa_media_interval_hours <= 0:
        since = max(0.0, float(hour - cfg.osteo_induction_hours))
    else:
        since = float((hour - cfg.osteo_induction_hours) % cfg.aa_media_interval_hours)
    return float(cfg.aa_initial * math.exp(-k * since))


def _commit_activation_hour(cfg: NonSpatialParams) -> int:
    return int(cfg.osteo_induction_hours + cfg.commit_delay_hours)


def _extract_at_day(df: pd.DataFrame, day: int) -> Dict[str, float]:
    row = df.loc[df["day"] == day]
    if row.empty:
        raise ValueError(f"No row found for day {day}")
    rec = row.iloc[0]
    return {
        "n_preob": float(rec["n_preob"]),
        "n_ob": float(rec["n_ob"]),
        "alp": float(rec["alp_total"]),
        "collagen": float(rec["collagen_total"]),
        "collagen_mature": float(rec["collagen_mature_total"]),
        "hap": float(rec["ha_total"]),
    }


def extract_endpoints(df: pd.DataFrame) -> Dict[str, float]:
    d10 = _extract_at_day(df, 10)
    d14 = _extract_at_day(df, 14)
    d28 = _extract_at_day(df, 28)
    return {
        "alp_day10": d10["alp"],
        "collagen_day14": d14["collagen"],
        "hap_day28": d28["hap"],
        "n_preob_day28": d28["n_preob"],
        "n_ob_day28": d28["n_ob"],
        "collagen_mature_day28": d28["collagen_mature"],
    }


def simulate_baseline(cfg: NonSpatialParams) -> pd.DataFrame:
    total_hours = max(1, int(cfg.total_hours))

    n_pre = max(0.0, float(cfg.n_preob0))
    n_ob = max(0.0, float(cfg.n_ob0))
    alp = max(0.0, float(cfg.alp_initial))
    collagen = 0.0
    collagen_mat = 0.0
    hap = 0.0

    hill_n = max(1.0, float(cfg.ha_hill_n))
    ob_age_n_sum = 0.0

    m_alp = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.alp_resp)
    m_col = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.collagen_resp)
    m_ha = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.ha_resp)

    rows: List[Dict[str, float]] = []

    def append_row(hour: int) -> None:
        day = hour // HOURS_PER_DAY
        rows.append(
            {
                "hour": float(hour),
                "day": int(day),
                "n_preob": float(n_pre),
                "n_ob": float(n_ob),
                "alp_total": float(alp),
                "collagen_total": float(collagen),
                "collagen_mature_total": float(collagen_mat),
                "ha_total": float(hap),
                "alp_modulation": float(m_alp),
                "collagen_modulation": float(m_col),
                "ha_modulation": float(m_ha),
            }
        )

    append_row(0)

    commit_start = _commit_activation_hour(cfg)
    k_commit = max(0.0, float(cfg.k_commit_hourly))

    for hour in range(1, total_hours + 1):
        births_pre = max(0.0, cfg.prolif_rate_preob) * n_pre
        births_ob = max(0.0, cfg.prolif_rate_ob) * n_ob

        commit_flow = 0.0
        if hour >= commit_start and k_commit > 0.0:
            commit_flow = min(k_commit * n_pre, n_pre)

        death_pre = max(0.0, cfg.apoptosis_preob) * n_pre
        death_ob = max(0.0, cfg.apoptosis_ob) * n_ob

        n_pre_next = n_pre + births_pre + births_ob - commit_flow - death_pre
        n_ob_next = n_ob + commit_flow - death_ob

        alp += max(0.0, cfg.alp_basal_rate) * (n_pre + n_ob)
        if hour >= cfg.osteo_induction_hours:
            alp += max(0.0, cfg.alp_osteo_rate * m_alp) * n_ob

        collagen += max(0.0, cfg.collagen_basal_rate) * (n_pre + n_ob)
        if hour >= cfg.osteo_induction_hours:
            aa_t = _aa_concentration(cfg, hour)
            aa_gate = aa_t / (max(1e-9, cfg.collagen_aa_half_sat) + aa_t) if aa_t > 0.0 else 0.0
            collagen += max(0.0, cfg.collagen_osteo_rate * m_col * aa_gate) * n_ob

        if cfg.tau_col_hours <= 0.0:
            collagen_mat = collagen
        else:
            collagen_mat += (collagen - collagen_mat) / cfg.tau_col_hours
            collagen_mat = min(collagen_mat, collagen)

        ob_age_n_sum = ob_age_n_sum + n_ob
        if n_ob > 1e-12 and death_ob > 0.0:
            keep_frac = max(0.0, 1.0 - death_ob / n_ob)
            ob_age_n_sum *= keep_frac

        if n_ob > 0.0:
            base_rate = max(0.0, cfg.ha_osteo_rate * m_ha)
            if base_rate > 0.0:
                gate_col = collagen_mat / (max(1e-9, cfg.ha_collagen_half_sat) + collagen_mat)
                tau_n = max(1e-9, cfg.ha_tau_hours) ** hill_n
                t_eff_n = max(0.0, ob_age_n_sum / n_ob) if n_ob > 1e-12 else 0.0
                hill_gate = t_eff_n / (tau_n + t_eff_n)
                hap += base_rate * gate_col * hill_gate * n_ob

        n_pre = max(0.0, n_pre_next)
        n_ob = max(0.0, n_ob_next)
        append_row(hour)

    return pd.DataFrame(rows)


def run_ablation(cfg: NonSpatialParams, ablation: str) -> AblationResult:
    name = ablation.strip().lower()
    c = cfg

    if name == "full_model":
        pass
    elif name == "no_collagen_maturation":
        c = replace(c, tau_col_hours=0.0)
    elif name == "no_drug_modulation":
        unity = DrugResponseParams(dex_ref=1.0, dex_a=0.0, dex_b=0.0, mk4_ref=1.0, mk4_a=0.0, mk4_b=0.0)
        c = replace(c, alp_resp=unity, collagen_resp=unity, ha_resp=unity)
    elif name == "no_commitment_delay":
        c = replace(c, commit_delay_hours=0, k_commit_hourly=1.0)
    elif name == "no_apoptosis":
        c = replace(c, apoptosis_preob=0.0, apoptosis_ob=0.0)
    else:
        raise ValueError(f"Unknown ablation: {ablation}")

    ts = simulate_baseline(c)
    endpoints = extract_endpoints(ts)
    return AblationResult(name=ablation, timeseries=ts, endpoints=endpoints)


def run_all_ablations(cfg: NonSpatialParams, names: Iterable[str] | None = None) -> List[AblationResult]:
    if names is None:
        names = (
            "full_model",
            "no_collagen_maturation",
            "no_drug_modulation",
            "no_commitment_delay",
            "no_apoptosis",
        )
    return [run_ablation(cfg, name) for name in names]
