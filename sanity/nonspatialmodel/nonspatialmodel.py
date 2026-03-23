#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd


HOURS_PER_DAY = 24
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent


@dataclass(slots=True)
class Cell:
    state: str
    birth_hour: int
    last_div_hour: int
    division_interval: int
    commit_hour: Optional[int] = None


@dataclass(slots=True)
class NonSpatialConfig:
    total_hours: int
    seed: Optional[int]
    cells: int
    area_sites: int
    media_interval: int
    osteo_induction_hours: int
    commit_delay: int
    commit_k: float
    commitment: bool
    proliferation: bool
    proliferation_chance: float
    ob_div_prob: float
    min_div_hours: int
    max_div_hours: int
    apoptosis: bool
    apoptosis_prob: float
    ob_apoptosis_prob: float
    secretion: bool
    dex_conc: float
    mk4_conc: float
    alp_basal_rate: float
    alp_osteo_rate: float
    alp_initial: float
    collagen_basal_rate: float
    collagen_osteo_rate: float
    aa_initial: float
    aa_half_life_hours: float
    collagen_aa_half_sat: float
    ha_osteo_rate: float
    ha_tau_hours: float
    ha_hill_n: float
    ha_collagen_maturation_tau_hours: float
    ha_collagen_half_sat: float
    alp_dex_ref_conc: float
    collagen_dex_ref_conc: float
    ha_dex_ref_conc: float
    alp_a_d: float
    alp_b_d: float
    collagen_a_d: float
    collagen_b_d: float
    ha_a_d: float
    ha_b_d: float
    alp_mk4_ref_conc: float
    collagen_mk4_ref_conc: float
    ha_mk4_ref_conc: float
    alp_a_k: float
    alp_b_k: float
    collagen_a_k: float
    collagen_b_k: float
    ha_a_k: float
    ha_b_k: float


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OsteoMin non-spatial (well-mixed) model using main.py defaults",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--hours", type=int, default=None)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--cells", type=int, default=72)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--w", type=int, default=33, help="used only to compute area = w*h")
    p.add_argument("--h", type=int, default=33, help="used only to compute area = w*h")
    p.add_argument("--out", type=str, default=str(SCRIPT_DIR / "out"))
    p.add_argument(
        "--from-run-json",
        type=str,
        default=None,
        help="optional main model run.json to clone parameters from",
    )
    p.add_argument("--csv", action="store_true", help="also write CSV timeseries")
    p.add_argument("--xlsx", dest="xlsx", action="store_true")
    p.add_argument("--no-xlsx", dest="xlsx", action="store_false")
    p.set_defaults(xlsx=True)

    p.add_argument("--media-interval", type=int, default=48)
    p.add_argument("--dex-conc", "--dex", type=float, default=0.0)
    p.add_argument("--mk4-conc", "--mk4", type=float, default=0.0)

    p.add_argument("--alp-rb", type=float, default=0.10710412147505423)
    p.add_argument("--alp-ro", type=float, default=0.9856290672451192)
    p.add_argument("--alp-p0", type=float, default=0.8)

    p.add_argument("--collagen-rb", type=float, default=0.0035518253599441936)
    p.add_argument("--collagen-ro", type=float, default=0.0130876)
    p.add_argument("--aa0", type=float, default=50.0)
    p.add_argument("--aa-half-life", type=float, default=48.0)
    p.add_argument("--collagen-aa-km", type=float, default=10.0)

    p.add_argument("--ha-ro", type=float, default=24.147854915293642)
    p.add_argument("--ha-tau", type=float, default=384.0)
    p.add_argument("--ha-hill-n", type=float, default=4.0)
    p.add_argument("--ha-collagen-maturation-tau", type=float, default=48.0)
    p.add_argument("--ha-collagen-km", type=float, default=1.0)

    p.add_argument("--alp-dex-ref", type=float, default=20.0)
    p.add_argument("--collagen-dex-ref", type=float, default=60.0)
    p.add_argument("--ha-dex-ref", "--ars-dex-ref", dest="ha_dex_ref", type=float, default=50.0)

    p.add_argument("--alp-a-d", type=float, default=None)
    p.add_argument("--alp-b-d", type=float, default=None)
    p.add_argument("--collagen-a-d", type=float, default=None)
    p.add_argument("--collagen-b-d", type=float, default=None)
    p.add_argument("--ha-a-d", type=float, default=None)
    p.add_argument("--ha-b-d", type=float, default=None)
    p.add_argument("--dex-a", type=float, default=None)
    p.add_argument("--dex-b", type=float, default=None)

    p.add_argument("--alp-mk4-ref", type=float, default=0.6)
    p.add_argument("--collagen-mk4-ref", type=float, default=8.0)
    p.add_argument("--ha-mk4-ref", "--ars-mk4-ref", dest="ha_mk4_ref", type=float, default=0.7)

    p.add_argument("--alp-a-k", type=float, default=None)
    p.add_argument("--alp-b-k", type=float, default=None)
    p.add_argument("--collagen-a-k", type=float, default=None)
    p.add_argument("--collagen-b-k", type=float, default=None)
    p.add_argument("--ha-a-k", type=float, default=None)
    p.add_argument("--ha-b-k", type=float, default=None)
    p.add_argument("--drug-a-k", type=float, default=None)
    p.add_argument("--drug-b-k", type=float, default=None)

    p.add_argument("--osteogenic-commitment", dest="commitment", action="store_true")
    p.add_argument("--no-osteogenic-commitment", dest="commitment", action="store_false")
    p.set_defaults(commitment=True)
    p.add_argument("--osteo-induction-hours", type=int, default=72)
    p.add_argument("--commit-delay", type=int, default=48)
    p.add_argument("--commit-k", type=float, default=0.0125)

    p.add_argument("--proliferation", dest="proliferation", action="store_true")
    p.add_argument("--no-proliferation", dest="proliferation", action="store_false")
    p.set_defaults(proliferation=True)
    p.add_argument("--proliferation-chance", type=float, default=1.0)
    p.add_argument("--ob-div-prob", type=float, default=0.0)
    p.add_argument("--min-div-hours", type=int, default=38)
    p.add_argument("--max-div-hours", type=int, default=52)

    p.add_argument("--apoptosis", dest="apoptosis", action="store_true")
    p.add_argument("--no-apoptosis", dest="apoptosis", action="store_false")
    p.set_defaults(apoptosis=True)
    p.add_argument("--apoptosis-prob", type=float, default=1e-3)
    p.add_argument("--apoptosis-osteo", dest="ob_apoptosis_prob", type=float, default=0.0)

    p.add_argument("--secretion", dest="secretion", action="store_true")
    p.add_argument("--no-secretion", dest="secretion", action="store_false")
    p.set_defaults(secretion=True)
    return p.parse_args()


def _apply_run_json_overrides(args: argparse.Namespace) -> argparse.Namespace:
    if not args.from_run_json:
        return args
    path = Path(args.from_run_json).expanduser()
    if not path.exists():
        raise SystemExit(f"--from-run-json not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    field = payload.get("field_cfg", {}) or {}
    abm = payload.get("abm_cfg", {}) or {}
    model_size = payload.get("model_size", {}) or {}

    if payload.get("total_hours") is not None:
        args.hours = int(payload["total_hours"])
    if payload.get("seed") is not None and args.seed is None:
        args.seed = int(payload["seed"])
    if model_size.get("initial_cell_count") is not None:
        init_cells = int(model_size["initial_cell_count"])
        if init_cells > 0:
            args.cells = init_cells

    if abm.get("width") is not None:
        args.w = int(abm["width"])
    if abm.get("height") is not None:
        args.h = int(abm["height"])
    if abm.get("osteo_induction_hours") is not None:
        args.osteo_induction_hours = int(abm["osteo_induction_hours"])
    if abm.get("commit_delay") is not None:
        args.commit_delay = int(abm["commit_delay"])
    if abm.get("commit_k_hourly") is not None:
        args.commit_k = float(abm["commit_k_hourly"])
    if abm.get("apoptosis_prob") is not None:
        args.apoptosis_prob = float(abm["apoptosis_prob"])
    if abm.get("ob_apoptosis_prob") is not None:
        args.ob_apoptosis_prob = float(abm["ob_apoptosis_prob"])
    if abm.get("proliferation_chance") is not None:
        args.proliferation_chance = float(abm["proliferation_chance"])
    if abm.get("ob_division_prob") is not None:
        args.ob_div_prob = float(abm["ob_division_prob"])
    if abm.get("min_div_hours") is not None:
        args.min_div_hours = int(abm["min_div_hours"])
    if abm.get("max_div_hours") is not None:
        args.max_div_hours = int(abm["max_div_hours"])
    if abm.get("enable_commitment") is not None:
        args.commitment = bool(abm["enable_commitment"])
    if abm.get("enable_apoptosis") is not None:
        args.apoptosis = bool(abm["enable_apoptosis"])
    if abm.get("enable_proliferation") is not None:
        args.proliferation = bool(abm["enable_proliferation"])
    if abm.get("enable_secretion") is not None:
        args.secretion = bool(abm["enable_secretion"])

    if field.get("media_interval") is not None:
        args.media_interval = int(field["media_interval"])
    if field.get("dex_conc") is not None:
        args.dex_conc = float(field["dex_conc"])
    if field.get("mk4_conc") is not None:
        args.mk4_conc = float(field["mk4_conc"])
    if field.get("alp_basal_rate") is not None:
        args.alp_rb = float(field["alp_basal_rate"])
    if field.get("alp_osteo_rate") is not None:
        args.alp_ro = float(field["alp_osteo_rate"])
    if field.get("alp_initial") is not None:
        args.alp_p0 = float(field["alp_initial"])
    if field.get("collagen_basal_rate") is not None:
        args.collagen_rb = float(field["collagen_basal_rate"])
    if field.get("collagen_osteo_rate") is not None:
        args.collagen_ro = float(field["collagen_osteo_rate"])
    if field.get("aa_initial") is not None:
        args.aa0 = float(field["aa_initial"])
    if field.get("aa_half_life_hours") is not None:
        args.aa_half_life = float(field["aa_half_life_hours"])
    if field.get("collagen_aa_half_sat") is not None:
        args.collagen_aa_km = float(field["collagen_aa_half_sat"])
    if field.get("ha_osteo_rate") is not None:
        args.ha_ro = float(field["ha_osteo_rate"])
    if field.get("ha_tau_hours") is not None:
        args.ha_tau = float(field["ha_tau_hours"])
    if field.get("ha_hill_n") is not None:
        args.ha_hill_n = float(field["ha_hill_n"])
    if field.get("ha_collagen_maturation_tau_hours") is not None:
        args.ha_collagen_maturation_tau = float(field["ha_collagen_maturation_tau_hours"])
    if field.get("ha_collagen_half_sat") is not None:
        args.ha_collagen_km = float(field["ha_collagen_half_sat"])

    if field.get("alp_dex_ref_conc") is not None:
        args.alp_dex_ref = float(field["alp_dex_ref_conc"])
    if field.get("collagen_dex_ref_conc") is not None:
        args.collagen_dex_ref = float(field["collagen_dex_ref_conc"])
    if field.get("ha_dex_ref_conc") is not None:
        args.ha_dex_ref = float(field["ha_dex_ref_conc"])
    if field.get("alp_mk4_ref_conc") is not None:
        args.alp_mk4_ref = float(field["alp_mk4_ref_conc"])
    if field.get("collagen_mk4_ref_conc") is not None:
        args.collagen_mk4_ref = float(field["collagen_mk4_ref_conc"])
    if field.get("ha_mk4_ref_conc") is not None:
        args.ha_mk4_ref = float(field["ha_mk4_ref_conc"])

    for name in (
        "alp_a_d",
        "alp_b_d",
        "collagen_a_d",
        "collagen_b_d",
        "ha_a_d",
        "ha_b_d",
        "alp_a_k",
        "alp_b_k",
        "collagen_a_k",
        "collagen_b_k",
        "ha_a_k",
        "ha_b_k",
    ):
        if field.get(name) is not None:
            setattr(args, name, float(field[name]))
    return args


def _log_concentration_offset(conc: float, ref_conc: float) -> float:
    if conc <= 0.0:
        return 0.0
    return float(math.log10(conc / max(1e-12, ref_conc)))


def _resolve_config(args: argparse.Namespace) -> NonSpatialConfig:
    if args.hours is not None:
        total_hours = max(0, int(args.hours))
    else:
        total_hours = max(0, int(args.days) * HOURS_PER_DAY)
    if total_hours <= 0:
        raise SystemExit("specify a positive --hours or --days value")
    if args.dex_conc != 0.0 and not (1.0 <= float(args.dex_conc) <= 100.0):
        raise SystemExit(
            f"Dex concentration {args.dex_conc:g} nM is outside the supported non-toxic range of 1 to 100 nM"
        )
    if args.mk4_conc != 0.0 and not (0.1 <= float(args.mk4_conc) <= 10.0):
        raise SystemExit(
            f"MK-4 concentration {args.mk4_conc:g} uM is outside the supported non-toxic range of 0.1 to 10 uM"
        )

    alp_a_d = float(args.alp_a_d) if args.alp_a_d is not None else (float(args.dex_a) if args.dex_a is not None else 3.79)
    alp_b_d = float(args.alp_b_d) if args.alp_b_d is not None else (float(args.dex_b) if args.dex_b is not None else 1.92)
    collagen_a_d = float(args.collagen_a_d) if args.collagen_a_d is not None else (float(args.dex_a) if args.dex_a is not None else 2.42)
    collagen_b_d = float(args.collagen_b_d) if args.collagen_b_d is not None else (float(args.dex_b) if args.dex_b is not None else 0.770)
    ha_a_d = float(args.ha_a_d) if args.ha_a_d is not None else (float(args.dex_a) if args.dex_a is not None else 1.14)
    ha_b_d = float(args.ha_b_d) if args.ha_b_d is not None else (float(args.dex_b) if args.dex_b is not None else 0.365)

    alp_a_k = float(args.alp_a_k) if args.alp_a_k is not None else (float(args.drug_a_k) if args.drug_a_k is not None else 0.56)
    alp_b_k = float(args.alp_b_k) if args.alp_b_k is not None else (float(args.drug_b_k) if args.drug_b_k is not None else 0.471)
    collagen_a_k = float(args.collagen_a_k) if args.collagen_a_k is not None else (float(args.drug_a_k) if args.drug_a_k is not None else 0.915)
    collagen_b_k = float(args.collagen_b_k) if args.collagen_b_k is not None else (float(args.drug_b_k) if args.drug_b_k is not None else 0.167)
    ha_a_k = float(args.ha_a_k) if args.ha_a_k is not None else (float(args.drug_a_k) if args.drug_a_k is not None else 4.69)
    ha_b_k = float(args.ha_b_k) if args.ha_b_k is not None else (float(args.drug_b_k) if args.drug_b_k is not None else 2.21)

    return NonSpatialConfig(
        total_hours=total_hours,
        seed=args.seed,
        cells=max(0, int(args.cells)),
        area_sites=max(1, int(args.w) * int(args.h)),
        media_interval=max(0, int(args.media_interval)),
        osteo_induction_hours=max(0, int(args.osteo_induction_hours)),
        commit_delay=max(0, int(args.commit_delay)),
        commit_k=max(0.0, float(args.commit_k)),
        commitment=bool(args.commitment),
        proliferation=bool(args.proliferation),
        proliferation_chance=max(0.0, min(1.0, float(args.proliferation_chance))),
        ob_div_prob=max(0.0, min(1.0, float(args.ob_div_prob))),
        min_div_hours=max(1, int(args.min_div_hours)),
        max_div_hours=max(int(args.min_div_hours), int(args.max_div_hours)),
        apoptosis=bool(args.apoptosis),
        apoptosis_prob=max(0.0, min(1.0, float(args.apoptosis_prob))),
        ob_apoptosis_prob=max(0.0, min(1.0, float(args.ob_apoptosis_prob))),
        secretion=bool(args.secretion),
        dex_conc=float(args.dex_conc),
        mk4_conc=float(args.mk4_conc),
        alp_basal_rate=float(args.alp_rb),
        alp_osteo_rate=float(args.alp_ro),
        alp_initial=max(0.0, float(args.alp_p0)),
        collagen_basal_rate=float(args.collagen_rb),
        collagen_osteo_rate=float(args.collagen_ro),
        aa_initial=max(0.0, float(args.aa0)),
        aa_half_life_hours=max(1e-9, float(args.aa_half_life)),
        collagen_aa_half_sat=max(1e-9, float(args.collagen_aa_km)),
        ha_osteo_rate=float(args.ha_ro),
        ha_tau_hours=max(1e-9, float(args.ha_tau)),
        ha_hill_n=max(1.0, float(args.ha_hill_n)),
        ha_collagen_maturation_tau_hours=max(1e-9, float(args.ha_collagen_maturation_tau)),
        ha_collagen_half_sat=max(1e-9, float(args.ha_collagen_km)),
        alp_dex_ref_conc=max(1e-12, float(args.alp_dex_ref)),
        collagen_dex_ref_conc=max(1e-12, float(args.collagen_dex_ref)),
        ha_dex_ref_conc=max(1e-12, float(args.ha_dex_ref)),
        alp_a_d=alp_a_d,
        alp_b_d=alp_b_d,
        collagen_a_d=collagen_a_d,
        collagen_b_d=collagen_b_d,
        ha_a_d=ha_a_d,
        ha_b_d=ha_b_d,
        alp_mk4_ref_conc=max(1e-12, float(args.alp_mk4_ref)),
        collagen_mk4_ref_conc=max(1e-12, float(args.collagen_mk4_ref)),
        ha_mk4_ref_conc=max(1e-12, float(args.ha_mk4_ref)),
        alp_a_k=alp_a_k,
        alp_b_k=alp_b_k,
        collagen_a_k=collagen_a_k,
        collagen_b_k=collagen_b_k,
        ha_a_k=ha_a_k,
        ha_b_k=ha_b_k,
    )


def _aa_concentration(cfg: NonSpatialConfig, hour: int) -> float:
    if cfg.aa_initial <= 0.0 or hour < cfg.osteo_induction_hours:
        return 0.0
    k = math.log(2.0) / cfg.aa_half_life_hours
    if cfg.media_interval <= 0:
        since = max(0.0, float(hour) - cfg.osteo_induction_hours)
    else:
        since = float((hour - cfg.osteo_induction_hours) % cfg.media_interval)
    return float(cfg.aa_initial * math.exp(-k * since))


def run_nonspatial(cfg: NonSpatialConfig) -> pd.DataFrame:
    rng = np.random.default_rng(cfg.seed)
    cells: List[Cell] = []
    for _ in range(cfg.cells):
        div_interval = int(rng.integers(cfg.min_div_hours, cfg.max_div_hours + 1))
        cells.append(
            Cell(
                state="preob",
                birth_hour=0,
                last_div_hour=0,
                division_interval=div_interval,
                commit_hour=None,
            )
        )

    total_alp = cfg.alp_initial * cfg.area_sites
    total_collagen = 0.0
    total_ha = 0.0
    collagen_mature_total = 0.0
    alp_preob_total = 0.0
    alp_ob_total = 0.0
    collagen_preob_total = 0.0
    collagen_ob_total = 0.0
    mineral_preob_total = 0.0
    mineral_ob_total = 0.0

    rows: List[dict] = []
    day_blocks = (cfg.total_hours + HOURS_PER_DAY - 1) // HOURS_PER_DAY
    total_days = 1 + day_blocks
    commit_activation_hour = cfg.osteo_induction_hours + cfg.commit_delay
    commit_prob_hourly = min(1.0, max(0.0, 1.0 - math.exp(-cfg.commit_k)))

    def append_row(day: int, dead_preob_today: int, dead_ob_today: int, committed_today: int) -> None:
        n_preob = sum(1 for c in cells if c.state == "preob")
        n_ob = sum(1 for c in cells if c.state == "ob")
        rows.append(
            {
                "day": int(day),
                "n_preob": int(n_preob),
                "n_osteoblast": int(n_ob),
                "n_total": int(n_preob + n_ob),
                "dead_preob_today": int(dead_preob_today),
                "dead_ob_today": int(dead_ob_today),
                "committed_today": int(committed_today),
                "collagen_total": float(total_collagen),
                "alp_total": float(total_alp),
                "ha_total": float(total_ha),
                "collagen_preob_total": float(collagen_preob_total),
                "collagen_ob_total": float(collagen_ob_total),
                "alp_preob_total": float(alp_preob_total),
                "alp_ob_total": float(alp_ob_total),
                "mineral_preob_total": float(mineral_preob_total),
                "mineral_ob_total": float(mineral_ob_total),
                "msd_effective": 0.0,
                "avg_division_interval": (
                    float(np.mean([c.division_interval for c in cells])) if cells else 0.0
                ),
                "avg_move_count": 0.0,
                "avg_live_age_hours": (
                    float(np.mean([max(0, day * HOURS_PER_DAY - c.birth_hour) for c in cells])) if cells else 0.0
                ),
                "nodule_count_current": 0,
                "nodule_count_ever": 0,
                "nodule_count": 0,
            }
        )

    append_row(0, 0, 0, 0)
    dead_preob_today = 0
    dead_ob_today = 0
    committed_today = 0
    for hour in range(1, cfg.total_hours + 1):
        if (hour - 1) % HOURS_PER_DAY == 0:
            dead_preob_today = 0
            dead_ob_today = 0
            committed_today = 0

        if cfg.proliferation and cells:
            newborns: List[Cell] = []
            for c in cells:
                if c.state == "ob" and rng.random() >= cfg.ob_div_prob:
                    continue
                if (hour - c.last_div_hour) < c.division_interval:
                    continue
                if rng.random() >= cfg.proliferation_chance:
                    continue
                occupied = len(cells) + len(newborns)
                if occupied >= cfg.area_sites:
                    continue
                free_frac = max(0.0, float(cfg.area_sites - occupied) / float(cfg.area_sites))
                place_prob = 1.0 - (1.0 - free_frac) ** 8
                if rng.random() >= place_prob:
                    continue
                div_interval = int(rng.integers(cfg.min_div_hours, cfg.max_div_hours + 1))
                newborns.append(
                    Cell(
                        state="preob",
                        birth_hour=hour,
                        last_div_hour=hour,
                        division_interval=div_interval,
                        commit_hour=None,
                    )
                )
                c.last_div_hour = hour
            cells.extend(newborns)

        if cfg.commitment and cfg.commit_k > 0.0 and hour >= commit_activation_hour:
            for c in cells:
                if c.state != "preob":
                    continue
                if (hour - c.last_div_hour) < HOURS_PER_DAY:
                    continue
                if rng.random() < commit_prob_hourly:
                    c.state = "ob"
                    c.commit_hour = hour
                    committed_today += 1

        if cfg.apoptosis and cells:
            survivors: List[Cell] = []
            for c in cells:
                p = cfg.ob_apoptosis_prob if c.state == "ob" else cfg.apoptosis_prob
                if rng.random() < p:
                    if c.state == "ob":
                        dead_ob_today += 1
                    else:
                        dead_preob_today += 1
                else:
                    survivors.append(c)
            cells = survivors

        if cfg.secretion:
            n_preob = sum(1 for c in cells if c.state == "preob")
            n_ob = sum(1 for c in cells if c.state == "ob")
            n_live = n_preob + n_ob

            alp_added_basal = max(0.0, cfg.alp_basal_rate) * n_live
            total_alp += alp_added_basal
            alp_preob_total += alp_added_basal

            col_added_basal = max(0.0, cfg.collagen_basal_rate) * n_live
            total_collagen += col_added_basal
            collagen_preob_total += col_added_basal

            i_d = 1.0 if cfg.dex_conc > 0.0 else 0.0
            i_k = 1.0 if cfg.mk4_conc > 0.0 else 0.0
            alp_dex_offset = _log_concentration_offset(cfg.dex_conc, cfg.alp_dex_ref_conc)
            alp_mk4_offset = _log_concentration_offset(cfg.mk4_conc, cfg.alp_mk4_ref_conc)
            col_dex_offset = _log_concentration_offset(cfg.dex_conc, cfg.collagen_dex_ref_conc)
            col_mk4_offset = _log_concentration_offset(cfg.mk4_conc, cfg.collagen_mk4_ref_conc)
            ha_dex_offset = _log_concentration_offset(cfg.dex_conc, cfg.ha_dex_ref_conc)
            ha_mk4_offset = _log_concentration_offset(cfg.mk4_conc, cfg.ha_mk4_ref_conc)

            m_d_alp = max(0.0, 1.0 + i_d * (cfg.alp_a_d - cfg.alp_b_d * alp_dex_offset**2))
            m_k_alp = max(0.0, 1.0 + i_k * (cfg.alp_a_k - cfg.alp_b_k * alp_mk4_offset**2))
            m_d_col = max(0.0, 1.0 + i_d * (cfg.collagen_a_d - cfg.collagen_b_d * col_dex_offset**2))
            m_k_col = max(0.0, 1.0 + i_k * (cfg.collagen_a_k - cfg.collagen_b_k * col_mk4_offset**2))
            m_d_ha = max(0.0, 1.0 + i_d * (cfg.ha_a_d - cfg.ha_b_d * ha_dex_offset**2))
            m_k_ha = max(0.0, 1.0 + i_k * (cfg.ha_a_k - cfg.ha_b_k * ha_mk4_offset**2))

            if hour >= cfg.osteo_induction_hours and n_ob > 0:
                alp_added_ob = max(0.0, cfg.alp_osteo_rate * m_d_alp * m_k_alp) * n_ob
                total_alp += alp_added_ob
                alp_ob_total += alp_added_ob

                aa_t = _aa_concentration(cfg, hour)
                aa_gate = aa_t / (cfg.collagen_aa_half_sat + aa_t) if aa_t > 0.0 else 0.0
                col_added_ob = max(0.0, cfg.collagen_osteo_rate * m_d_col * m_k_col * aa_gate) * n_ob
                total_collagen += col_added_ob
                collagen_ob_total += col_added_ob

            collagen_mature_total += (total_collagen - collagen_mature_total) / cfg.ha_collagen_maturation_tau_hours
            collagen_mature_total = min(collagen_mature_total, total_collagen)

            if n_ob > 0:
                base_rate = max(0.0, cfg.ha_osteo_rate * m_d_ha * m_k_ha)
                if base_rate > 0.0:
                    tau_n = cfg.ha_tau_hours ** cfg.ha_hill_n
                    gate_c = (collagen_mature_total / cfg.area_sites)
                    gate_c = gate_c / (cfg.ha_collagen_half_sat + gate_c)
                    hill_sum = 0.0
                    for c in cells:
                        if c.state != "ob" or c.commit_hour is None:
                            continue
                        t_eff = max(0.0, float(hour - c.commit_hour))
                        t_eff_n = t_eff ** cfg.ha_hill_n
                        hill_sum += t_eff_n / (tau_n + t_eff_n) if (tau_n + t_eff_n) > 0 else 0.0
                    ha_added = base_rate * gate_c * hill_sum
                    total_ha += ha_added
                    mineral_ob_total += ha_added

        end_of_day = (hour % HOURS_PER_DAY == 0) or (hour == cfg.total_hours)
        if end_of_day:
            day_index = (hour - 1) // HOURS_PER_DAY + 1
            append_row(day_index, dead_preob_today, dead_ob_today, committed_today)

    if len(rows) < total_days:
        while len(rows) < total_days:
            append_row(len(rows), 0, 0, 0)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.from_run_json:
        frj = Path(args.from_run_json).expanduser()
        if not frj.is_absolute():
            args.from_run_json = str((PROJECT_ROOT / frj).resolve())
    args = _apply_run_json_overrides(args)
    cfg = _resolve_config(args)
    outdir = Path(args.out).expanduser()
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    df = run_nonspatial(cfg)
    if args.csv:
        df.to_csv(outdir / "timeseries.csv", index=False)
    if args.xlsx:
        df.to_excel(outdir / "timeseries.xlsx", index=False)
    if not args.csv and not args.xlsx:
        df.to_csv(outdir / "timeseries.csv", index=False)

    (outdir / "run.json").write_text(json.dumps({"model": "nonspatial", "config": asdict(cfg)}, indent=2), encoding="utf-8")
    print(f"[nonspatialmodel] rows={len(df)} out={outdir}")


if __name__ == "__main__":
    main()
