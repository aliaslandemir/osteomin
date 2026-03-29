#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUT = SCRIPT_DIR / "ablation_out"
PROJECT_ROOT = SCRIPT_DIR.parents[1]
MAIN_PY = PROJECT_ROOT / "main.py"
DEFAULT_SPATIAL_OUT = PROJECT_ROOT / "main_out"

# Structural ablations: all except nonspatial_model are run with main.py (spatial model).
VARIANTS: list[tuple[str, str]] = [
    ("full_model", "Spatial model (reference)"),
    ("no_osteogenic_commitment", "No osteogenic commitment"),
    ("no_commit_delay", "No commitment delay"),
    ("no_apoptosis", "No apoptosis"),
    ("no_collagen_maturation", "No collagen maturation delay"),
    ("nonspatial_model", "Nonspatial model"),
]

SPATIAL_VARIANTS = [v for v, _ in VARIANTS if v != "nonspatial_model"]
DOSE_VARIANTS = ["full_model", "no_osteogenic_commitment"]
DOSE_CONDITIONS = ["C2", "C3"]
DOSE_REFERENCE_CONDITION = "C2"

CONDITIONS: List[Tuple[str, float, float]] = [
    ("C1", 0.0, 0.0),
    ("C2", 0.0, 0.0),   # AA+beta-GP medium, no Dex/MK-4
    ("C3", 10.0, 0.1),  # AA+beta-GP + low Dex/MK-4
    ("C4", 10.0, 1.0),
    ("C5", 30.0, 3.0),
    ("C6", 100.0, 10.0),
    ("C7", 10.0, 0.1),
    ("C8", 10.0, 3.0),
]

METRIC_COLS = [
    "alp_day10",
    "collagen_day14",
    "hap_day28",
    "cell_count_day28",
]

# Embedded nonspatial model primitives so sanity_check8 can run without osteomin_nonspatial.py.
HOURS_PER_DAY = 24
SIM_DAYS = 28
SIM_HOURS = SIM_DAYS * HOURS_PER_DAY


@dataclass(slots=True)
class DrugResponseParams:
    dex_ref: float
    dex_a: float
    dex_b: float
    mk4_ref: float
    mk4_a: float
    mk4_b: float


@dataclass(slots=True)
class NonSpatialParams:
    total_hours: int = SIM_HOURS

    n_preob0: float = 72.0
    n_ob0: float = 0.0

    prolif_rate_preob: float = 1.0 / 45.0
    prolif_rate_ob: float = 0.0
    apoptosis_preob: float = 1e-3
    apoptosis_ob: float = 0.0

    osteo_induction_hours: int = 72
    commit_delay_hours: int = 48
    k_commit_hourly: float = 0.0125

    dex_conc_nm: float = 0.0
    mk4_conc_um: float = 0.0

    alp_basal_rate: float = 0.0949
    alp_osteo_rate: float = 0.810
    alp_initial: float = 0.8

    collagen_basal_rate: float = 0.00336
    collagen_osteo_rate: float = 0.0130

    aa_initial: float = 50.0
    aa_half_life_hours: float = 8.0
    aa_media_interval_hours: int = 48
    collagen_aa_half_sat: float = 5.0

    tau_col_hours: float = 48.0
    ha_osteo_rate: float = 0.102
    ha_tau_hours: float = 384.0
    ha_hill_n: float = 2.0
    ha_collagen_half_sat: float = 1.0
    carry_capacity: float | None = None

    alp_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=22.7,
            dex_a=3.72,
            dex_b=1.75,
            mk4_ref=0.784,
            mk4_a=1.10,
            mk4_b=1.06,
        )
    )
    collagen_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=6.22,
            dex_a=2.69,
            dex_b=2.04,
            mk4_ref=0.349,
            mk4_a=4.66,
            mk4_b=2.15,
        )
    )
    ha_resp: DrugResponseParams = field(
        default_factory=lambda: DrugResponseParams(
            dex_ref=46.6,
            dex_a=1.22,
            dex_b=0.679,
            mk4_ref=0.667,
            mk4_a=4.66,
            mk4_b=2.15,
        )
    )


@dataclass(slots=True)
class AblationResult:
    name: str
    timeseries: pd.DataFrame
    endpoints: Dict[str, float]


@dataclass(slots=True)
class TaskSpec:
    run_id: int
    seed: int
    variant: str
    condition: str
    dex_nM: float
    mk4_uM: float
    mode: str  # spatial | nonspatial
    out_dir: Path


@dataclass(slots=True)
class TaskResult:
    run_id: int
    seed: int
    variant: str
    condition: str
    dex_nM: float
    mk4_uM: float
    mode: str
    out_dir: str
    success: bool
    returncode: int
    command: str
    error: str | None
    skipped: bool
    alp_day10: float | None
    collagen_day14: float | None
    hap_day28: float | None
    cell_count_day28: float | None


def _log_offset(conc: float, ref: float) -> float:
    if conc <= 0.0:
        return 0.0
    return float(math.log10(conc / max(1e-12, ref)))


def _drug_modulation(conc_dex: float, conc_mk4: float, resp: DrugResponseParams) -> float:
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
    cap = float(cfg.carry_capacity) if (cfg.carry_capacity is not None and float(cfg.carry_capacity) > 0.0) else None

    n_pre = max(0.0, float(cfg.n_preob0))
    n_ob = max(0.0, float(cfg.n_ob0))
    if cap is not None and (n_pre + n_ob) > cap:
        sf = cap / max(1e-12, (n_pre + n_ob))
        n_pre *= sf
        n_ob *= sf
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
            }
        )

    append_row(0)

    commit_start = _commit_activation_hour(cfg)
    k_commit = max(0.0, float(cfg.k_commit_hourly))

    for hour in range(1, total_hours + 1):
        crowd = 1.0
        if cap is not None:
            crowd = max(0.0, 1.0 - (n_pre + n_ob) / cap)

        births_pre = max(0.0, cfg.prolif_rate_preob) * n_pre * crowd
        births_ob = max(0.0, cfg.prolif_rate_ob) * n_ob * crowd

        commit_flow = 0.0
        if hour >= commit_start and k_commit > 0.0:
            commit_flow = min(k_commit * n_pre, n_pre)

        death_pre = max(0.0, cfg.apoptosis_preob) * n_pre
        death_ob = max(0.0, cfg.apoptosis_ob) * n_ob

        n_pre_next = n_pre + births_pre + births_ob - commit_flow - death_pre
        n_ob_next = n_ob + commit_flow - death_ob
        if cap is not None and (n_pre_next + n_ob_next) > cap:
            sf = cap / max(1e-12, (n_pre_next + n_ob_next))
            n_pre_next *= sf
            n_ob_next *= sf

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


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 1.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _lognormal_factor(rng: np.random.Generator, cv: float) -> float:
    if cv <= 0.0:
        return 1.0
    sigma = math.sqrt(math.log1p(cv * cv))
    mu = -0.5 * sigma * sigma
    return float(rng.lognormal(mean=mu, sigma=sigma))


def _jitter_nonspatial_params(cfg: NonSpatialParams, seed: int, cv: float) -> NonSpatialParams:
    if cv <= 0.0:
        return cfg
    rng = np.random.default_rng(int(seed))

    def j(v: float, lo: float = 0.0, hi: float | None = None) -> float:
        out = float(v) * _lognormal_factor(rng, cv)
        out = max(lo, out)
        if hi is not None:
            out = min(hi, out)
        return out

    return replace(
        cfg,
        prolif_rate_preob=j(cfg.prolif_rate_preob, lo=1e-9, hi=2.0),
        apoptosis_preob=j(cfg.apoptosis_preob, lo=0.0, hi=0.02),
        k_commit_hourly=j(cfg.k_commit_hourly, lo=0.0, hi=0.2),
        alp_basal_rate=j(cfg.alp_basal_rate, lo=1e-12),
        alp_osteo_rate=j(cfg.alp_osteo_rate, lo=1e-12),
        collagen_basal_rate=j(cfg.collagen_basal_rate, lo=1e-12),
        collagen_osteo_rate=j(cfg.collagen_osteo_rate, lo=1e-12),
        tau_col_hours=j(cfg.tau_col_hours, lo=1e-6),
        ha_osteo_rate=j(cfg.ha_osteo_rate, lo=1e-12),
        ha_tau_hours=j(cfg.ha_tau_hours, lo=1e-6),
        aa_initial=j(cfg.aa_initial, lo=1e-9),
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run structural ablation study: spatial variants + nonspatial comparator.")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--spatial-out", type=Path, default=DEFAULT_SPATIAL_OUT)
    p.add_argument("--python", type=str, default=None, help="python interpreter for spatial main.py")
    p.add_argument("--replicates", "--n-runs", dest="replicates", type=int, default=5)
    p.add_argument("--base-seed", type=int, default=12345)
    p.add_argument("--param-cv", type=float, default=0.10, help="replicate-level CV for nonspatial parameter uncertainty")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--endpoint-condition", type=str, default="C2", choices=[f"C{i}" for i in range(1, 9)])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--runtime", action="store_true")
    args = p.parse_args()
    if args.replicates <= 0:
        raise SystemExit("--replicates must be >= 1")
    if float(args.param_cv) < 0.0:
        raise SystemExit("--param-cv must be >= 0")
    if args.jobs <= 0:
        raise SystemExit("--jobs must be >= 1")
    if args.python is None:
        args.python = str((Path(__file__).resolve().parents[2] / ".venv" / "bin" / "python3"))
        if not Path(args.python).exists():
            args.python = "python3"
    return args


def _read_spatial_reference(spatial_out: Path) -> tuple[dict, pd.DataFrame]:
    run_json = spatial_out / "run.json"
    ts_xlsx = spatial_out / "timeseries.xlsx"
    if not run_json.exists():
        raise FileNotFoundError(f"Missing spatial run metadata: {run_json}")
    if not ts_xlsx.exists():
        raise FileNotFoundError(f"Missing spatial timeseries: {ts_xlsx}")
    run_cfg = json.loads(run_json.read_text(encoding="utf-8"))
    ts_df = pd.read_excel(ts_xlsx)
    return run_cfg, ts_df


def _build_nonspatial_base_from_spatial(run_cfg: dict, ts_df: pd.DataFrame) -> NonSpatialParams:
    abm = run_cfg.get("abm_cfg", {}) if isinstance(run_cfg, dict) else {}
    field = run_cfg.get("field_cfg", {}) if isinstance(run_cfg, dict) else {}
    total_hours = int(run_cfg.get("total_hours", 672)) if isinstance(run_cfg, dict) else 672
    model_size = run_cfg.get("model_size", {}) if isinstance(run_cfg, dict) else {}

    if (ts_df["day"] == 0).sum() == 0:
        raise ValueError("Spatial timeseries missing day 0")
    n_preob0 = float(ts_df.loc[ts_df["day"] == 0, "n_preob"].iloc[0])
    n_ob0 = float(ts_df.loc[ts_df["day"] == 0, "n_osteoblast"].iloc[0])

    min_div = float(abm.get("min_div_hours", 38.0))
    max_div = float(abm.get("max_div_hours", 52.0))
    mean_div = max(1.0, 0.5 * (min_div + max_div))
    prolif_pre = float(abm.get("proliferation_chance", 1.0)) / mean_div
    prolif_ob = float(abm.get("ob_division_prob", 0.0)) / mean_div

    return NonSpatialParams(
        total_hours=total_hours,
        n_preob0=n_preob0,
        n_ob0=n_ob0,
        prolif_rate_preob=prolif_pre,
        prolif_rate_ob=prolif_ob,
        osteo_induction_hours=int(abm.get("osteo_induction_hours", 72)),
        commit_delay_hours=int(abm.get("commit_delay", 48)),
        k_commit_hourly=float(abm.get("commit_k_hourly", 0.0125)),
        apoptosis_preob=float(abm.get("apoptosis_prob", 0.0010)),
        apoptosis_ob=float(abm.get("ob_apoptosis_prob", 0.0)),
        dex_conc_nm=float(field.get("dex_conc", 0.0)),
        mk4_conc_um=float(field.get("mk4_conc", 0.0)),
        alp_basal_rate=float(field.get("alp_basal_rate", 0.0949)),
        alp_osteo_rate=float(field.get("alp_osteo_rate", 0.810)),
        alp_initial=float(field.get("alp_initial", 0.8)),
        collagen_basal_rate=float(field.get("collagen_basal_rate", 0.00336)),
        collagen_osteo_rate=float(field.get("collagen_osteo_rate", 0.0130)),
        aa_initial=float(field.get("aa_initial", 50.0)),
        aa_half_life_hours=float(field.get("aa_half_life_hours", 8.0)),
        aa_media_interval_hours=int(field.get("media_interval", 48)),
        collagen_aa_half_sat=float(field.get("collagen_aa_half_sat", 5.0)),
        tau_col_hours=float(field.get("ha_collagen_maturation_tau_hours", 48.0)),
        ha_osteo_rate=float(field.get("ha_osteo_rate", 0.102)),
        ha_tau_hours=float(field.get("ha_tau_hours", 384.0)),
        ha_hill_n=float(field.get("ha_hill_n", 2.0)),
        ha_collagen_half_sat=float(field.get("ha_collagen_half_sat", 1.0)),
        carry_capacity=float(model_size.get("grid_sites", 0.0)) if float(model_size.get("grid_sites", 0.0)) > 0.0 else float(abm.get("width", 33)) * float(abm.get("height", 33)),
        alp_resp=DrugResponseParams(
            dex_ref=float(field.get("alp_dex_ref_conc", 22.7)),
            dex_a=float(field.get("alp_a_d", 3.72)),
            dex_b=float(field.get("alp_b_d", 1.75)),
            mk4_ref=float(field.get("alp_mk4_ref_conc", 0.784)),
            mk4_a=float(field.get("alp_a_k", 1.10)),
            mk4_b=float(field.get("alp_b_k", 1.06)),
        ),
        collagen_resp=DrugResponseParams(
            dex_ref=float(field.get("collagen_dex_ref_conc", 6.22)),
            dex_a=float(field.get("collagen_a_d", 2.69)),
            dex_b=float(field.get("collagen_b_d", 2.04)),
            mk4_ref=float(field.get("collagen_mk4_ref_conc", 0.349)),
            mk4_a=float(field.get("collagen_a_k", 4.66)),
            mk4_b=float(field.get("collagen_b_k", 2.15)),
        ),
        ha_resp=DrugResponseParams(
            dex_ref=float(field.get("ha_dex_ref_conc", 46.6)),
            dex_a=float(field.get("ha_a_d", 1.22)),
            dex_b=float(field.get("ha_b_d", 0.679)),
            mk4_ref=float(field.get("ha_mk4_ref_conc", 0.667)),
            mk4_a=float(field.get("ha_a_k", 4.66)),
            mk4_b=float(field.get("ha_b_k", 2.15)),
        ),
    )


def _bool_flag(v: bool, on_flag: str, off_flag: str) -> list[str]:
    return [on_flag if bool(v) else off_flag]


def _build_spatial_base_args(run_cfg: dict, ts_df: pd.DataFrame) -> list[str]:
    abm = run_cfg.get("abm_cfg", {}) if isinstance(run_cfg, dict) else {}
    field = run_cfg.get("field_cfg", {}) if isinstance(run_cfg, dict) else {}

    if (ts_df["day"] == 0).sum() == 0:
        raise ValueError("Spatial timeseries missing day 0")
    n_cells = int(round(float(ts_df.loc[ts_df["day"] == 0, "n_total"].iloc[0])))

    args = [
        str(MAIN_PY),
        "--hours", str(int(run_cfg.get("total_hours", 672))),
        "--cells", str(max(1, n_cells)),
        "--w", str(int(abm.get("width", 33))),
        "--h", str(int(abm.get("height", 33))),
        "--boundary-mode", str(abm.get("boundary_mode", "periodic")),
        "--seed-mode", "random",
        "--nodules", str(run_cfg.get("nodule_connectivity", "40n")),
        "--media-interval", str(int(field.get("media_interval", 48))),
        "--aa0", f"{float(field.get('aa_initial', 50.0)):.10g}",
        "--aa-half-life", f"{float(field.get('aa_half_life_hours', 8.0)):.10g}",
        "--collagen-aa-km", f"{float(field.get('collagen_aa_half_sat', 5.0)):.10g}",
        "--move-prob", f"{float(abm.get('move_prob', 0.27)):.10g}",
        "--max-moves", str(int(abm.get("max_moves_per_hour", 1))),
        "--ob-div-prob", f"{float(abm.get('ob_division_prob', 0.0)):.10g}",
        "--min-div-hours", str(int(abm.get("min_div_hours", 38))),
        "--max-div-hours", str(int(abm.get("max_div_hours", 52))),
        "--apoptosis-prob", f"{float(abm.get('apoptosis_prob', 0.001)):.10g}",
        "--apoptosis-osteo", f"{float(abm.get('ob_apoptosis_prob', 0.0)):.10g}",
        "--osteo-induction-hours", str(int(abm.get("osteo_induction_hours", 72))),
        "--commit-delay", str(int(abm.get("commit_delay", 48))),
        "--commit-k", f"{float(abm.get('commit_k_hourly', 0.0125)):.10g}",
        "--alp-rb", f"{float(field.get('alp_basal_rate', 0.0949)):.10g}",
        "--alp-ro", f"{float(field.get('alp_osteo_rate', 0.810)):.10g}",
        "--alp-p0", f"{float(field.get('alp_initial', 0.8)):.10g}",
        "--collagen-rb", f"{float(field.get('collagen_basal_rate', 0.00336)):.10g}",
        "--collagen-ro", f"{float(field.get('collagen_osteo_rate', 0.0130)):.10g}",
        "--ha-ro", f"{float(field.get('ha_osteo_rate', 0.102)):.10g}",
        "--ha-h0", f"{float(field.get('ha_initial', 2.34)):.10g}",
        "--alp-dex-ref", f"{float(field.get('alp_dex_ref_conc', 22.7)):.10g}",
        "--collagen-dex-ref", f"{float(field.get('collagen_dex_ref_conc', 6.22)):.10g}",
        "--ha-dex-ref", f"{float(field.get('ha_dex_ref_conc', 46.6)):.10g}",
        "--alp-a-d", f"{float(field.get('alp_a_d', 3.72)):.10g}",
        "--alp-b-d", f"{float(field.get('alp_b_d', 1.75)):.10g}",
        "--collagen-a-d", f"{float(field.get('collagen_a_d', 2.69)):.10g}",
        "--collagen-b-d", f"{float(field.get('collagen_b_d', 2.04)):.10g}",
        "--ha-a-d", f"{float(field.get('ha_a_d', 1.22)):.10g}",
        "--ha-b-d", f"{float(field.get('ha_b_d', 0.679)):.10g}",
        "--alp-mk4-ref", f"{float(field.get('alp_mk4_ref_conc', 0.784)):.10g}",
        "--collagen-mk4-ref", f"{float(field.get('collagen_mk4_ref_conc', 0.349)):.10g}",
        "--ha-mk4-ref", f"{float(field.get('ha_mk4_ref_conc', 0.667)):.10g}",
        "--alp-a-k", f"{float(field.get('alp_a_k', 1.10)):.10g}",
        "--alp-b-k", f"{float(field.get('alp_b_k', 1.06)):.10g}",
        "--collagen-a-k", f"{float(field.get('collagen_a_k', 4.66)):.10g}",
        "--collagen-b-k", f"{float(field.get('collagen_b_k', 2.15)):.10g}",
        "--ha-a-k", f"{float(field.get('ha_a_k', 4.66)):.10g}",
        "--ha-b-k", f"{float(field.get('ha_b_k', 2.15)):.10g}",
        "--ha-model", str(field.get("ha_model", "minimal")),
        "--ha-collagen-maturation-tau", f"{float(field.get('ha_collagen_maturation_tau_hours', 48.0)):.10g}",
        "--ha-collagen-km", f"{float(field.get('ha_collagen_half_sat', 1.0)):.10g}",
        "--ha-tau", f"{float(field.get('ha_tau_hours', 384.0)):.10g}",
        "--ha-hill-n", f"{float(field.get('ha_hill_n', 2.0)):.10g}",
        # Lightweight exports for ablation throughput.
        "--spatial-every", "0",
        "--csv",
        "--no-xlsx",
        "--no-unique-id",
        "--no-snapshots",
        "--no-hourly-exports",
    ]
    args += _bool_flag(bool(abm.get("enable_secretion", True)), "--secretion", "--no-secretion")
    args += _bool_flag(bool(abm.get("enable_random_move", True)), "--random-move", "--no-random-move")
    args += _bool_flag(bool(abm.get("enable_proliferation", True)), "--proliferation", "--no-proliferation")
    args += _bool_flag(bool(abm.get("enable_apoptosis", True)), "--apoptosis", "--no-apoptosis")
    args += _bool_flag(bool(abm.get("enable_commitment", True)), "--osteogenic-commitment", "--no-osteogenic-commitment")
    return args


def _variant_flags(variant: str) -> list[str]:
    if variant == "full_model":
        return []
    if variant == "no_osteogenic_commitment":
        return ["--no-osteogenic-commitment"]
    if variant == "no_commit_delay":
        return ["--commit-delay", "0"]
    if variant == "no_apoptosis":
        return ["--no-apoptosis"]
    if variant == "no_drug_modulation":
        return [
            "--alp-a-d", "0", "--alp-b-d", "0",
            "--collagen-a-d", "0", "--collagen-b-d", "0",
            "--ha-a-d", "0", "--ha-b-d", "0",
            "--alp-a-k", "0", "--alp-b-k", "0",
            "--collagen-a-k", "0", "--collagen-b-k", "0",
            "--ha-a-k", "0", "--ha-b-k", "0",
        ]
    if variant == "no_collagen_maturation":
        return ["--ha-collagen-maturation-tau", "1e-6"]
    raise ValueError(f"Unknown spatial ablation variant: {variant}")


def _extract_spatial_endpoints(ts_csv: Path) -> dict[str, float]:
    df = pd.read_csv(ts_csv)
    need_days = {10, 14, 28}
    have = set(int(x) for x in df["day"].tolist())
    if not need_days.issubset(have):
        raise ValueError(f"timeseries missing required days {need_days} in {ts_csv}")
    d10 = df.loc[df["day"] == 10].iloc[0]
    d14 = df.loc[df["day"] == 14].iloc[0]
    d28 = df.loc[df["day"] == 28].iloc[0]
    return {
        "alp_day10": float(d10["alp_total"]),
        "collagen_day14": float(d14["collagen_total"]),
        "hap_day28": float(d28["ha_total"]),
        "cell_count_day28": float(d28["n_total"]),
    }


def _prune_spatial_artifacts(out_dir: Path) -> None:
    # Keep only the tiny files needed for aggregation/debug.
    keep = {"timeseries.csv", "run.json"}
    if not out_dir.exists():
        return
    for p in out_dir.rglob("*"):
        if p.is_dir():
            continue
        rel = p.relative_to(out_dir)
        if rel.as_posix() in keep:
            continue
        if p.name in keep:
            continue
        try:
            p.unlink()
        except OSError:
            pass
    for name in ("figures_pdf", "collagen_ha", "grids", "spatial_cells", "alp_snapshots"):
        d = out_dir / name
        if d.exists() and d.is_dir():
            shutil.rmtree(d, ignore_errors=True)


def _run_nonspatial_task(
    task: TaskSpec,
    cfg_base: NonSpatialParams,
    param_cv: float,
    overwrite: bool,
) -> TaskResult:
    ts_path = task.out_dir / "timeseries.csv"
    cmd_txt = "<embedded nonspatial simulation>"

    if ts_path.exists() and not overwrite:
        ep = _extract_spatial_endpoints(ts_path)
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=True,
            returncode=0,
            command=cmd_txt,
            error=None,
            skipped=True,
            alp_day10=ep["alp_day10"],
            collagen_day14=ep["collagen_day14"],
            hap_day28=ep["hap_day28"],
            cell_count_day28=ep["cell_count_day28"],
        )

    cfg = replace(cfg_base, dex_conc_nm=float(task.dex_nM), mk4_conc_um=float(task.mk4_uM))
    cfg = _jitter_nonspatial_params(cfg, seed=int(task.seed), cv=float(param_cv))
    ts = simulate_baseline(cfg)
    ep = extract_endpoints(ts)
    task.out_dir.mkdir(parents=True, exist_ok=True)
    ts2 = ts.copy()
    ts2["n_total"] = ts2["n_preob"] + ts2["n_ob"]
    ts2.to_csv(ts_path, index=False)

    return TaskResult(
        run_id=task.run_id,
        seed=task.seed,
        variant=task.variant,
        condition=task.condition,
        dex_nM=task.dex_nM,
        mk4_uM=task.mk4_uM,
        mode=task.mode,
        out_dir=str(task.out_dir),
        success=True,
        returncode=0,
        command=cmd_txt,
        error=None,
        skipped=False,
        alp_day10=float(ep["alp_day10"]),
        collagen_day14=float(ep["collagen_day14"]),
        hap_day28=float(ep["hap_day28"]),
        cell_count_day28=float(ep["n_preob_day28"] + ep["n_ob_day28"]),
    )


def _run_spatial_task(
    task: TaskSpec,
    python_exe: str,
    spatial_base_args: list[str],
    overwrite: bool,
    dry_run: bool,
) -> TaskResult:
    ts_path = task.out_dir / "timeseries.csv"
    flags = _variant_flags(task.variant)
    cmd = [
        python_exe,
        *spatial_base_args,
        *flags,
        "--seed", str(int(task.seed)),
        "--dex-conc", f"{float(task.dex_nM):.10g}",
        "--mk4-conc", f"{float(task.mk4_uM):.10g}",
        "--out", str(task.out_dir),
    ]
    cmd_txt = " ".join(cmd)

    if ts_path.exists() and not overwrite:
        ep = _extract_spatial_endpoints(ts_path)
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=True,
            returncode=0,
            command=cmd_txt,
            error=None,
            skipped=True,
            alp_day10=ep["alp_day10"],
            collagen_day14=ep["collagen_day14"],
            hap_day28=ep["hap_day28"],
            cell_count_day28=ep["cell_count_day28"],
        )

    if dry_run:
        print(cmd_txt)
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=True,
            returncode=0,
            command=cmd_txt,
            error=None,
            skipped=False,
            alp_day10=None,
            collagen_day14=None,
            hap_day28=None,
            cell_count_day28=None,
        )

    task.out_dir.mkdir(parents=True, exist_ok=True)
    try:
        completed = subprocess.run(cmd, cwd=PROJECT_ROOT, check=False)
    except OSError as exc:
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=False,
            returncode=-1,
            command=cmd_txt,
            error=str(exc),
            skipped=False,
            alp_day10=None,
            collagen_day14=None,
            hap_day28=None,
            cell_count_day28=None,
        )

    if completed.returncode != 0:
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=False,
            returncode=int(completed.returncode),
            command=cmd_txt,
            error=f"exit {completed.returncode}",
            skipped=False,
            alp_day10=None,
            collagen_day14=None,
            hap_day28=None,
            cell_count_day28=None,
        )

    if not ts_path.exists():
        return TaskResult(
            run_id=task.run_id,
            seed=task.seed,
            variant=task.variant,
            condition=task.condition,
            dex_nM=task.dex_nM,
            mk4_uM=task.mk4_uM,
            mode=task.mode,
            out_dir=str(task.out_dir),
            success=False,
            returncode=2,
            command=cmd_txt,
            error=f"missing expected output: {ts_path}",
            skipped=False,
            alp_day10=None,
            collagen_day14=None,
            hap_day28=None,
            cell_count_day28=None,
        )

    ep = _extract_spatial_endpoints(ts_path)
    _prune_spatial_artifacts(task.out_dir)
    return TaskResult(
        run_id=task.run_id,
        seed=task.seed,
        variant=task.variant,
        condition=task.condition,
        dex_nM=task.dex_nM,
        mk4_uM=task.mk4_uM,
        mode=task.mode,
        out_dir=str(task.out_dir),
        success=True,
        returncode=0,
        command=cmd_txt,
        error=None,
        skipped=False,
        alp_day10=ep["alp_day10"],
        collagen_day14=ep["collagen_day14"],
        hap_day28=ep["hap_day28"],
        cell_count_day28=ep["cell_count_day28"],
    )


def _task_key(task: TaskSpec) -> tuple[int, str, str]:
    return (int(task.run_id), str(task.variant), str(task.condition))


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    run_cfg, ts_df = _read_spatial_reference(args.spatial_out.resolve())
    spatial_base_args = _build_spatial_base_args(run_cfg, ts_df)
    nonspatial_base = _build_nonspatial_base_from_spatial(run_cfg, ts_df)

    label_map = dict(VARIANTS)
    cond_map = {c: (d, k) for c, d, k in CONDITIONS}
    cond_order = list(DOSE_CONDITIONS)
    dose_ref_cond = str(DOSE_REFERENCE_CONDITION)
    if dose_ref_cond not in cond_map:
        raise SystemExit(f"unknown dose reference condition: {dose_ref_cond}")
    ref_cond = str(args.endpoint_condition)
    if ref_cond not in cond_map:
        raise SystemExit(f"unknown endpoint condition: {ref_cond}")

    rep_seeds = [int(args.base_seed + i) for i in range(int(args.replicates))]

    tasks: list[TaskSpec] = []
    for run_id, seed in enumerate(rep_seeds):
        for variant, _ in VARIANTS:
            dex_ref, mk4_ref = cond_map[ref_cond]
            mode = "nonspatial" if variant == "nonspatial_model" else "spatial"
            tasks.append(
                TaskSpec(
                    run_id=run_id,
                    seed=int(seed),
                    variant=variant,
                    condition=ref_cond,
                    dex_nM=float(dex_ref),
                    mk4_uM=float(mk4_ref),
                    mode=mode,
                    out_dir=out / "runs" / f"r{run_id + 1:02d}" / variant / ref_cond,
                )
            )

        for variant in DOSE_VARIANTS:
            for cond in DOSE_CONDITIONS:
                dex, mk4 = cond_map[str(cond)]
                tasks.append(
                    TaskSpec(
                        run_id=run_id,
                        seed=int(seed),
                        variant=variant,
                        condition=str(cond),
                        dex_nM=float(dex),
                        mk4_uM=float(mk4),
                        mode="spatial",
                        out_dir=out / "runs" / f"r{run_id + 1:02d}" / variant / str(cond),
                    )
                )

    uniq: dict[tuple[int, str, str], TaskSpec] = {}
    for t in tasks:
        uniq[_task_key(t)] = t
    task_list = sorted(uniq.values(), key=lambda t: (t.run_id, t.variant, t.condition))

    print(f"[ablation_study_run] spatial reference: {args.spatial_out.resolve()}")
    print(f"[ablation_study_run] endpoint reference condition: {ref_cond}")
    print(f"[ablation_study_run] replicates per ablation case: {args.replicates}")
    print(f"[ablation_study_run] jobs: {args.jobs}")
    print(f"[ablation_study_run] nonspatial param CV: {float(args.param_cv):.4f}")
    print(f"[ablation_study_run] total tasks: {len(task_list)}")

    if args.dry_run:
        for t in task_list:
            if t.mode == "spatial":
                _run_spatial_task(t, args.python, spatial_base_args, args.overwrite, dry_run=True)
            else:
                print(
                    f"<embedded nonspatial simulation> run_id={t.run_id} "
                    f"variant={t.variant} cond={t.condition} out={t.out_dir}"
                )
        return

    t0 = time.perf_counter()
    results: list[TaskResult] = []
    total = len(task_list)

    def _exec_task(t: TaskSpec) -> TaskResult:
        if t.mode == "spatial":
            return _run_spatial_task(t, args.python, spatial_base_args, args.overwrite, dry_run=False)
        return _run_nonspatial_task(t, nonspatial_base, float(args.param_cv), overwrite=args.overwrite)

    if int(args.jobs) == 1:
        for i, t in enumerate(task_list, start=1):
            r = _exec_task(t)
            results.append(r)
            if args.runtime:
                elapsed = time.perf_counter() - t0
                avg = elapsed / float(i)
                eta = max(0.0, avg * (total - i))
                status = "ok" if r.success else "failed"
                print(
                    f"[ablation_study_run] run {i}/{total}: {t.variant} {t.condition} "
                    f"({status}, elapsed {elapsed:.1f}s, eta {eta:.1f}s)"
                )
            else:
                status = "ok" if r.success else "failed"
                print(f"[ablation_study_run] run {i}/{total}: {t.variant} {t.condition} ({status})")
    else:
        with ThreadPoolExecutor(max_workers=int(args.jobs)) as pool:
            futs = {pool.submit(_exec_task, t): t for t in task_list}
            done = 0
            for fut in as_completed(futs):
                t = futs[fut]
                r = fut.result()
                results.append(r)
                done += 1
                if args.runtime:
                    elapsed = time.perf_counter() - t0
                    avg = elapsed / float(done)
                    eta = max(0.0, avg * (total - done))
                    status = "ok" if r.success else "failed"
                    print(
                        f"[ablation_study_run] run {done}/{total}: {t.variant} {t.condition} "
                        f"({status}, elapsed {elapsed:.1f}s, eta {eta:.1f}s)"
                    )
                else:
                    status = "ok" if r.success else "failed"
                    print(f"[ablation_study_run] run {done}/{total}: {t.variant} {t.condition} ({status})")

    results.sort(key=lambda r: (r.run_id, r.variant, r.condition))
    summary_df = pd.DataFrame([asdict(r) for r in results])
    summary_df.to_csv(out / "run_summary.csv", index=False)

    failed = [r for r in results if not r.success]
    if failed:
        raise SystemExit(f"{len(failed)} task(s) failed. See {out / 'run_summary.csv'}")

    result_map = {(r.run_id, r.variant, r.condition): r for r in results}

    raw_rows: list[dict[str, float | int | str]] = []
    rel_rows: list[dict[str, float | int | str]] = []
    dose_rows: list[dict[str, float | int | str]] = []

    for run_id in range(int(args.replicates)):
        full_key = (run_id, "full_model", ref_cond)
        full_row = result_map[full_key]

        full_metrics = {
            "alp_day10": float(full_row.alp_day10),
            "collagen_day14": float(full_row.collagen_day14),
            "hap_day28": float(full_row.hap_day28),
            "cell_count_day28": float(full_row.cell_count_day28),
        }

        endpoint_vals: dict[str, dict[str, float]] = {}
        for variant, label in VARIANTS:
            r = result_map[(run_id, variant, ref_cond)]
            vals = {
                "alp_day10": float(r.alp_day10),
                "collagen_day14": float(r.collagen_day14),
                "hap_day28": float(r.hap_day28),
                "cell_count_day28": float(r.cell_count_day28),
            }
            endpoint_vals[variant] = vals
            raw_rows.append(
                {
                    "run_id": run_id,
                    "seed": int(rep_seeds[run_id]),
                    "variant": variant,
                    "label": label,
                    "mode": r.mode,
                    "condition": ref_cond,
                    **vals,
                }
            )
            rel = {"run_id": run_id, "variant": variant, "label": label}
            for c in METRIC_COLS:
                rel[c] = (vals[c] - full_metrics[c]) / max(1e-12, full_metrics[c])
            rel_rows.append(rel)

        # Dose robustness for selected spatial variants.
        for variant in DOSE_VARIANTS:
            c_ref = result_map[(run_id, variant, dose_ref_cond)]
            denom = max(1e-12, float(c_ref.hap_day28))
            for cond in cond_order:
                rv = result_map[(run_id, variant, cond)]
                rf = result_map[(run_id, "full_model", cond)]
                hap_raw = float(rv.hap_day28)
                full_raw = float(rf.hap_day28)
                dose_rows.append(
                    {
                        "run_id": run_id,
                        "variant": variant,
                        "condition": cond,
                        "dose_reference_condition": dose_ref_cond,
                        "hap_fold_vs_ref": hap_raw / denom,
                        "hap_fold_vs_c2": hap_raw / denom if dose_ref_cond == "C2" else float("nan"),
                        "hap_raw": hap_raw,
                        "ratio_to_full": hap_raw / max(1e-12, full_raw),
                    }
                )

    raw_df = pd.DataFrame(raw_rows).sort_values(["run_id", "variant"], kind="stable").reset_index(drop=True)
    rel_df = pd.DataFrame(rel_rows).sort_values(["run_id", "variant"], kind="stable").reset_index(drop=True)
    dose_df = (
        pd.DataFrame(dose_rows)
        .sort_values(["run_id", "variant", "condition"], kind="stable")
        .reset_index(drop=True)
    )

    rel_summary_rows = []
    for (variant, label), g in rel_df.groupby(["variant", "label"], sort=False):
        row: dict[str, float | str] = {"variant": variant, "label": label}
        for c in METRIC_COLS:
            v = g[c].to_numpy(dtype=float)
            row[c] = float(np.median(v))
            row[f"{c}_lo"] = float(np.percentile(v, 2.5))
            row[f"{c}_hi"] = float(np.percentile(v, 97.5))
            row[f"{c}_pneg"] = float(np.mean(v < 0.0))
        rel_summary_rows.append(row)
    rel_summary_df = pd.DataFrame(rel_summary_rows)

    raw_df.to_csv(out / "ablation_raw_metrics_runs.csv", index=False)
    rel_df.to_csv(out / "ablation_runs_relative.csv", index=False)
    rel_summary_df.to_csv(out / "ablation_relative_summary.csv", index=False)
    dose_df.to_csv(out / "ablation_dose_runs.csv", index=False)

    metadata = {
        "replicates": int(args.replicates),
        "n_runs": int(args.replicates),
        "jobs": int(args.jobs),
        "base_seed": int(args.base_seed),
        "replicate_seeds": rep_seeds,
        "endpoint_condition": ref_cond,
        "spatial_out": str(args.spatial_out.resolve()),
        "spatial_base_args": spatial_base_args,
        "variants": [{"id": v, "label": l} for v, l in VARIANTS],
        "dose_variants": DOSE_VARIANTS,
        "dose_conditions": cond_order,
        "dose_reference_condition": dose_ref_cond,
        "conditions": [{"id": c, "dex": d, "mk4": k} for c, d, k in CONDITIONS],
        "nonspatial": {
            "carry_capacity": float(nonspatial_base.carry_capacity) if nonspatial_base.carry_capacity is not None else None,
            "prolif_rate_preob": float(nonspatial_base.prolif_rate_preob),
            "apoptosis_preob": float(nonspatial_base.apoptosis_preob),
            "param_cv": float(args.param_cv),
            "calibration": {"method": "none"},
        },
        "notes": (
            "Spatial model used for all ablations except nonspatial_model; "
            "nonspatial uses matched baseline parameters plus replicate-level uncertainty (param_cv) "
            "without endpoint rescaling; heavy daily exports disabled for throughput."
        ),
    }
    (out / "ablation_study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {out / 'ablation_raw_metrics_runs.csv'}")
    print(f"Saved: {out / 'ablation_runs_relative.csv'}")
    print(f"Saved: {out / 'ablation_relative_summary.csv'}")
    print(f"Saved: {out / 'ablation_dose_runs.csv'}")
    print(f"Saved: {out / 'run_summary.csv'}")
    print(f"Saved: {out / 'ablation_study_metadata.json'}")
    if args.runtime:
        print(f"[ablation_study_run] total runtime: {time.perf_counter() - t0:.2f}s")


if __name__ == "__main__":
    main()
