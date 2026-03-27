#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from osteomin_nonspatial import NonSpatialParams, extract_endpoints, run_ablation, simulate_baseline


VARIANTS = [
    ("full_model", "Full model"),
    ("no_osteogenic_commitment", "No osteogenic commitment"),
    ("no_commit_delay", "No commitment delay"),
    ("no_apoptosis", "No apoptosis"),
    ("no_drug_modulation", "No drug modulation"),
    ("no_collagen_maturation", "No collagen maturation delay"),
    ("minimal_nonspatial", "Minimal nonspatial model"),
]

DOSE_VARIANTS = ["full_model", "no_drug_modulation", "no_osteogenic_commitment"]

CONDITIONS: List[Tuple[str, float, float]] = [
    ("C1", 0.0, 0.0),
    ("C2", 1.0, 0.1),
    ("C3", 3.0, 0.3),
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
    "dose_peak_score",
    "decoupling_score",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run robust nonspatial ablation ensemble study.")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "ablation_out")
    p.add_argument("--n-runs", type=int, default=220)
    p.add_argument("--param-cv", type=float, default=0.22)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--endpoint-condition", type=str, default="C4", choices=[f"C{i}" for i in range(1, 9)])
    return p.parse_args()


def _safe_abs_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2 or y.size < 2:
        return 1.0
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 1.0
    return float(abs(np.corrcoef(x, y)[0, 1]))


def _sample_positive(base_val: float, cv: float, rng: np.random.Generator) -> float:
    if base_val <= 0.0:
        return base_val
    sigma = float(np.sqrt(np.log(1.0 + cv * cv)))
    mu = float(np.log(base_val) - 0.5 * sigma * sigma)
    return float(rng.lognormal(mu, sigma))


def _sample_bounded_normal(base_val: float, frac_sd: float, lo: float, hi: float, rng: np.random.Generator) -> float:
    sd = max(1e-12, abs(base_val) * frac_sd)
    v = float(rng.normal(base_val, sd))
    return float(min(hi, max(lo, v)))


def _sample_params(base: NonSpatialParams, cv: float, rng: np.random.Generator) -> NonSpatialParams:
    def _jitter_resp(resp):
        return replace(
            resp,
            dex_ref=_sample_positive(resp.dex_ref, cv, rng),
            mk4_ref=_sample_positive(resp.mk4_ref, cv, rng),
            dex_a=_sample_positive(max(resp.dex_a, 1e-9), cv, rng),
            dex_b=_sample_positive(max(resp.dex_b, 1e-9), cv, rng),
            mk4_a=_sample_positive(max(resp.mk4_a, 1e-9), cv, rng),
            mk4_b=_sample_positive(max(resp.mk4_b, 1e-9), cv, rng),
        )

    return replace(
        base,
        alp_basal_rate=_sample_positive(base.alp_basal_rate, cv, rng),
        alp_osteo_rate=_sample_positive(base.alp_osteo_rate, cv, rng),
        collagen_basal_rate=_sample_positive(base.collagen_basal_rate, cv, rng),
        collagen_osteo_rate=_sample_positive(base.collagen_osteo_rate, cv, rng),
        ha_osteo_rate=_sample_positive(base.ha_osteo_rate, cv, rng),
        prolif_rate_preob=_sample_positive(max(base.prolif_rate_preob, 1e-9), cv, rng),
        k_commit_hourly=_sample_positive(base.k_commit_hourly, cv, rng),
        ha_tau_hours=_sample_positive(base.ha_tau_hours, cv, rng),
        tau_col_hours=_sample_positive(base.tau_col_hours, cv, rng),
        aa_half_life_hours=_sample_positive(base.aa_half_life_hours, cv, rng),
        aa_initial=_sample_positive(base.aa_initial, cv, rng),
        collagen_aa_half_sat=_sample_positive(base.collagen_aa_half_sat, cv, rng),
        ha_collagen_half_sat=_sample_positive(base.ha_collagen_half_sat, cv, rng),
        ha_hill_n=_sample_bounded_normal(base.ha_hill_n, cv, 1.0, 8.0, rng),
        apoptosis_preob=max(0.0, min(0.05, _sample_positive(base.apoptosis_preob, cv, rng))),
        commit_delay_hours=int(round(_sample_bounded_normal(float(base.commit_delay_hours), cv, 0.0, 7.0 * 24.0, rng))),
        osteo_induction_hours=int(round(_sample_bounded_normal(float(base.osteo_induction_hours), cv * 0.5, 0.0, 10.0 * 24.0, rng))),
        alp_resp=_jitter_resp(base.alp_resp),
        collagen_resp=_jitter_resp(base.collagen_resp),
        ha_resp=_jitter_resp(base.ha_resp),
    )


def _run_variant(cfg: NonSpatialParams, variant: str) -> Dict[str, float]:
    v = variant.strip().lower()

    if v == "full_model":
        ts = simulate_baseline(cfg)
        ep = extract_endpoints(ts)
        return ep

    if v == "no_osteogenic_commitment":
        c = replace(cfg, k_commit_hourly=0.0)
        ts = simulate_baseline(c)
        return extract_endpoints(ts)

    if v == "no_commit_delay":
        c = replace(cfg, commit_delay_hours=0)
        ts = simulate_baseline(c)
        return extract_endpoints(ts)

    if v == "no_apoptosis":
        c = replace(cfg, apoptosis_preob=0.0, apoptosis_ob=0.0)
        ts = simulate_baseline(c)
        return extract_endpoints(ts)

    if v == "no_drug_modulation":
        return run_ablation(cfg, "no_drug_modulation").endpoints

    if v == "no_collagen_maturation":
        return run_ablation(cfg, "no_collagen_maturation").endpoints

    if v == "minimal_nonspatial":
        c = replace(
            cfg,
            prolif_rate_preob=0.0,
            prolif_rate_ob=0.0,
            apoptosis_preob=0.0,
            apoptosis_ob=0.0,
            k_commit_hourly=0.0,
            tau_col_hours=0.0,
        )
        u = replace(cfg.alp_resp, dex_a=0.0, dex_b=0.0, mk4_a=0.0, mk4_b=0.0)
        c = replace(c, alp_resp=u, collagen_resp=u, ha_resp=u)
        ts = simulate_baseline(c)
        return extract_endpoints(ts)

    raise ValueError(f"Unknown variant: {variant}")


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)

    # Default base aligned to your provided main.py settings (nonspatial focus)
    base = NonSpatialParams(
        total_hours=672,
        n_preob0=72.0,
        n_ob0=0.0,
        osteo_induction_hours=72,
        commit_delay_hours=48,
        k_commit_hourly=0.0125,
        apoptosis_preob=0.0010,
        apoptosis_ob=0.0,
        dex_conc_nm=0.0,
        mk4_conc_um=0.0,
        alp_basal_rate=0.0949,
        alp_osteo_rate=0.810,
        collagen_basal_rate=0.00336,
        collagen_osteo_rate=0.0130,
        aa_initial=50.0,
        aa_half_life_hours=8.0,
        collagen_aa_half_sat=5.0,
        ha_osteo_rate=0.102,
        ha_tau_hours=384.0,
        ha_hill_n=2.0,
    )

    rng = np.random.default_rng(int(args.seed))
    label_map = dict(VARIANTS)
    cond_order = [c[0] for c in CONDITIONS]
    cond_map = {c: (d, k) for c, d, k in CONDITIONS}
    ref_dex, ref_mk4 = cond_map[str(args.endpoint_condition)]

    rel_rows: List[Dict[str, float]] = []
    raw_rows: List[Dict[str, float]] = []
    dose_rows: List[Dict[str, float]] = []

    for run_id in range(int(args.n_runs)):
        cfg_run = _sample_params(base, float(args.param_cv), rng)

        # Endpoint effects at selected reference condition (default C4 for informative ablation contrast)
        base_vals: Dict[str, Dict[str, float]] = {}
        claim_vals: Dict[str, Dict[str, float]] = {}

        for var, _ in VARIANTS:
            cfg_ref = replace(cfg_run, dex_conc_nm=float(ref_dex), mk4_conc_um=float(ref_mk4))
            ep = _run_variant(cfg_ref, var)
            base_vals[var] = {
                "alp_day10": float(ep["alp_day10"]),
                "collagen_day14": float(ep["collagen_day14"]),
                "hap_day28": float(ep["hap_day28"]),
            }

        # Condition sweeps for claim metrics and dose figure
        cond_metrics: Dict[Tuple[str, str], Dict[str, float]] = {}
        for cname, dex, mk4 in CONDITIONS:
            cfg_cond = replace(cfg_run, dex_conc_nm=float(dex), mk4_conc_um=float(mk4))
            for var, _ in VARIANTS:
                ep = _run_variant(cfg_cond, var)
                cond_metrics[(var, cname)] = {
                    "alp_day10": float(ep["alp_day10"]),
                    "collagen_day14": float(ep["collagen_day14"]),
                    "hap_day28": float(ep["hap_day28"]),
                }

        for var, _ in VARIANTS:
            alp = np.array([cond_metrics[(var, c)]["alp_day10"] for c in cond_order], dtype=float)
            col = np.array([cond_metrics[(var, c)]["collagen_day14"] for c in cond_order], dtype=float)
            hap = np.array([cond_metrics[(var, c)]["hap_day28"] for c in cond_order], dtype=float)
            peak = float(max(0.0, np.max(hap) - max(hap[0], hap[-1])))
            dec = float(1.0 + (1.0 - 0.5 * (_safe_abs_corr(alp, hap) + _safe_abs_corr(col, hap))))
            claim_vals[var] = {
                "dose_peak_score": peak,
                "decoupling_score": dec,
            }

        full = {**base_vals["full_model"], **claim_vals["full_model"]}
        for var, _ in VARIANTS:
            vals = {**base_vals[var], **claim_vals[var]}
            cfg_ref = replace(cfg_run, dex_conc_nm=float(ref_dex), mk4_conc_um=float(ref_mk4))
            raw_rows.append(
                {
                    "run_id": run_id,
                    "variant": var,
                    "label": label_map[var],
                    "osteo_induction_hours": int(cfg_ref.osteo_induction_hours),
                    "commit_delay_hours": int(cfg_ref.commit_delay_hours),
                    "ha_tau_hours": float(cfg_ref.ha_tau_hours),
                    "alp_day10": float(vals["alp_day10"]),
                    "collagen_day14": float(vals["collagen_day14"]),
                    "hap_day28": float(vals["hap_day28"]),
                    "dose_peak_score": float(vals["dose_peak_score"]),
                    "decoupling_score": float(vals["decoupling_score"]),
                }
            )
            row = {"run_id": run_id, "variant": var, "label": label_map[var]}
            for c in METRIC_COLS:
                row[c] = (float(vals[c]) - float(full[c])) / max(1e-12, float(full[c]))
            rel_rows.append(row)

        # Dose robustness rows for selected variants
        for var in DOSE_VARIANTS:
            denom = max(1e-12, cond_metrics[(var, "C1")]["hap_day28"])
            for cname in cond_order:
                hap_raw = float(cond_metrics[(var, cname)]["hap_day28"])
                full_raw = float(cond_metrics[("full_model", cname)]["hap_day28"])
                dose_rows.append(
                    {
                        "run_id": run_id,
                        "variant": var,
                        "condition": cname,
                        "hap_fold_vs_c1": hap_raw / denom,
                        "hap_raw": hap_raw,
                        "ratio_to_full": hap_raw / max(1e-12, full_raw),
                    }
                )

    rel_df = pd.DataFrame(rel_rows)
    raw_df = pd.DataFrame(raw_rows)
    dose_df = pd.DataFrame(dose_rows)

    rel_summary_rows = []
    for (var, label), g in rel_df.groupby(["variant", "label"], sort=False):
        row = {"variant": var, "label": label}
        for c in METRIC_COLS:
            v = g[c].to_numpy(dtype=float)
            row[c] = float(np.median(v))
            row[f"{c}_lo"] = float(np.percentile(v, 2.5))
            row[f"{c}_hi"] = float(np.percentile(v, 97.5))
            row[f"{c}_pneg"] = float(np.mean(v < 0.0))
        rel_summary_rows.append(row)
    rel_summary = pd.DataFrame(rel_summary_rows)

    raw_df.to_csv(out / "ablation_raw_metrics_runs.csv", index=False)
    rel_df.to_csv(out / "ablation_runs_relative.csv", index=False)
    rel_summary.to_csv(out / "ablation_relative_summary.csv", index=False)
    dose_df.to_csv(out / "ablation_dose_runs.csv", index=False)

    metadata = {
        "n_runs": int(args.n_runs),
        "param_cv": float(args.param_cv),
        "seed": int(args.seed),
        "endpoint_condition": str(args.endpoint_condition),
        "base": asdict(base),
        "variants": [{"id": v, "label": l} for v, l in VARIANTS],
        "dose_variants": DOSE_VARIANTS,
        "conditions": [{"id": c, "dex": d, "mk4": k} for c, d, k in CONDITIONS],
    }
    (out / "ablation_study_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"Saved: {out / 'ablation_raw_metrics_runs.csv'}")
    print(f"Saved: {out / 'ablation_runs_relative.csv'}")
    print(f"Saved: {out / 'ablation_relative_summary.csv'}")
    print(f"Saved: {out / 'ablation_dose_runs.csv'}")
    print(f"Saved: {out / 'ablation_study_metadata.json'}")


if __name__ == "__main__":
    main()
