#!/usr/bin/env python3
"""
datafitmain.py

Unified driver to run ALP endpoint-based model and matrix/mineral model together.

Default values mirror the standalone example runs:
  - ALP: r_basal=0.70, r_osteo=1.16, gamma=1.0, day_start=3, day_measure=10
  - Matrix/mineral: collagen_C1=2740, collagen_C2=2880, ars_C1=0.65, ars_C2=50.87 (percent)


python3 -m datafitting.datafitmain \
  --collagen_C1 2740 --collagen_C2 2880 \
  --ars_C1 0.65 --ars_C2 50.87 \
  --ars_units percent \
  --sAA_C1 0 --sAA_C2 1 \
  --zeta_C1 1 --zeta_C2 1 \
  --Km_vals 5 10 15 30 \
  --n_vals 1 2 3 4 \
  --sobol \
  --outdir out_bas_v_osteo \
  --collagen_day_measure 14 \
  --ars_day_measure 28 \
  --osteo_day_start 3 \
  --alp_r_basal 0.70 \
  --alp_r_osteo 1.16 \
  --alp_gamma 1.0 \
  --alp_day_measure 10 \
  --alp_dt_hours 1 \
  --alp_outdir alp_linear_out




"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from datafitting.alptopi.unifiedalpmodel import run_alp_model
from datafitting.mineralmodel.unified_matrix_mineral import run_matrix_mineral_from_kwargs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Unified data-fitting runner for ALP and matrix/mineral models.")
    p.add_argument("--skip-alp", action="store_true", help="Skip ALP run.")
    p.add_argument("--skip-matrix", action="store_true", help="Skip matrix/mineral run.")
    p.add_argument("--osteo_day_start", type=float, default=None, help="Override osteogenic induction day for all models.")

    # ALP endpoints (basal vs osteo)
    alp = p.add_argument_group("ALP endpoints")
    alp.add_argument("--alp_r_basal", type=float, default=0.70, help="Endpoint ALP in basal condition at day_measure.")
    alp.add_argument("--alp_r_osteo", type=float, default=1.16, help="Endpoint ALP in osteogenic condition at day_measure (TOTAL).")
    alp.add_argument("--alp_gamma", type=float, default=1.0, help="Scaling factor for osteogenic increment.")
    alp.add_argument("--alp_day_start", type=float, default=3.0, help="Day of osteogenic induction.")
    alp.add_argument("--alp_day_measure", type=float, default=10.0, help="Measurement day for endpoints.")
    alp.add_argument("--alp_dt_hours", type=float, default=1.0, help="Time step in hours.")
    alp.add_argument("--alp_outdir", type=str, default="alp_linear_out", help="ALP output directory.")
    alp.add_argument("--alp_units", type=str, default="ALP_units", help="Units label for stored endpoints.")

    # Matrix/mineral endpoints and protocol
    mm = p.add_argument_group("Matrix / mineral endpoints + AA protocol")
    mm.add_argument("--collagen_C1", type=float, default=2740.0, help="Basal collagen endpoint (day 14).")
    mm.add_argument("--collagen_C2", type=float, default=2880.0, help="Osteogenic collagen endpoint (day 14).")
    mm.add_argument("--ars_C1", type=float, default=0.65, help="Basal ARS endpoint (day 28).")
    mm.add_argument("--ars_C2", type=float, default=50.87, help="Osteogenic ARS endpoint (day 28).")
    mm.add_argument("--ars_units", choices=["percent", "fraction"], default="percent", help="Units for ARS endpoints.")
    mm.add_argument("--AA0_ugml", type=float, default=50.0, help="AA pulse concentration (ug/mL).")
    mm.add_argument("--half_life_h", type=float, default=3.0, help="AA half-life (hours).")
    mm.add_argument("--media_interval_h", type=float, default=48.0, help="Media change interval (hours).")
    mm.add_argument("--day_start_AA", type=float, default=3.0, help="AA induction day.")
    mm.add_argument("--collagen_day_measure", type=float, default=14.0, help="Collagen measurement day.")
    mm.add_argument("--ars_day_measure", type=float, default=28.0, help="Mineral (ARS) measurement day.")

    mm.add_argument("--Km_vals", type=float, nargs="+", default=[5.0, 10.0, 15.0, 30.0], help="Km candidates.")
    mm.add_argument("--n_vals", type=float, nargs="+", default=[1.0, 2.0, 3.0, 4.0], help="Hill exponent candidates.")
    mm.add_argument("--smod_C1", type=float, default=None, help="Basal AA modulation (preferred input).")
    mm.add_argument("--smod_C2", type=float, default=None, help="Osteogenic AA modulation (preferred input).")
    mm.add_argument("--sAA_C1", type=float, default=0.0, help="Alias: basal AA modulation if smod_C1 not set.")
    mm.add_argument("--sAA_C2", type=float, default=1.0, help="Alias: osteogenic AA modulation if smod_C2 not set.")
    mm.add_argument("--zeta_C1", type=float, default=1.0, help="Basal mineral max.")
    mm.add_argument("--zeta_C2", type=float, default=1.0, help="Osteogenic mineral max.")
    mm.add_argument("--t_start", type=float, default=14.0, help="Mineral onset day.")
    mm.add_argument("--dt_h", type=float, default=1.0, help="Time step (hours).")
    mm.add_argument("--outdir", type=str, default="out_bas_v_osteo", help="Matrix/mineral output directory.")

    mm.add_argument("--sobol", action="store_true", help="Enable Sobol endpoint analysis.")
    mm.add_argument("--sobol_baseN", type=int, default=512, help="Sobol base N.")
    mm.add_argument("--smod2_vals", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0], help="Sobol s_mod candidates.")
    mm.add_argument("--sAA2_vals", type=float, nargs="+", default=None, help="Sobol s_mod aliases (if provided).")
    mm.add_argument("--half_life_vals", type=float, nargs="+", default=[1.5, 3.0, 6.0, 12.0], help="Sobol AA half-life candidates.")
    mm.add_argument("--media_interval_vals", type=float, nargs="+", default=[24.0, 48.0, 72.0, 96.0], help="Sobol media interval candidates.")
    mm.add_argument("--zeta2_vals", type=float, nargs="+", default=[0.5, 1.0, 1.5, 2.0], help="Sobol zeta candidates.")
    mm.add_argument("--t_start_vals", type=float, nargs="+", default=[12.0, 14.0, 16.0, 18.0], help="Sobol mineral onset candidates.")
    mm.add_argument("--k_vals", type=float, nargs="+", default=None, help="Sobol k candidates (optional).")

    return p.parse_args()


def main():
    args = parse_args()
    summary = {}
    base_dir = Path(__file__).resolve().parent
    combined_dir = base_dir
    if args.osteo_day_start is not None:
        args.alp_day_start = float(args.osteo_day_start)
        args.day_start_AA = float(args.osteo_day_start)

    if not args.skip_alp:
        alp_outdir = Path(args.alp_outdir)
        if not alp_outdir.is_absolute():
            alp_outdir = base_dir / alp_outdir
        summary["alp"] = run_alp_model(
            r_basal=float(args.alp_r_basal),
            r_osteo=float(args.alp_r_osteo),
            gamma=float(args.alp_gamma),
            day_start=float(args.alp_day_start),
            day_measure=float(args.alp_day_measure),
            dt_hours=float(args.alp_dt_hours),
            outdir=alp_outdir,
            units=args.alp_units,
        )
        print(f"[ALP] completed; outputs in {alp_outdir.resolve()}")
    else:
        print("[ALP] skipped.")

    if not args.skip_matrix:
        mm_outdir = Path(args.outdir)
        if not mm_outdir.is_absolute():
            mm_outdir = base_dir / mm_outdir
        summary["matrix"] = run_matrix_mineral_from_kwargs(
            collagen_C1=float(args.collagen_C1),
            collagen_C2=float(args.collagen_C2),
            ars_C1=float(args.ars_C1),
            ars_C2=float(args.ars_C2),
            ars_units=args.ars_units,
            AA0_ugml=float(args.AA0_ugml),
            half_life_h=float(args.half_life_h),
            media_interval_h=float(args.media_interval_h),
            day_start_AA=float(args.day_start_AA),
            collagen_day_measure=float(args.collagen_day_measure),
            ars_day_measure=float(args.ars_day_measure),
            Km_vals=list(map(float, args.Km_vals)),
            n_vals=list(map(float, args.n_vals)),
            smod_C1=args.smod_C1,
            smod_C2=args.smod_C2,
            sAA_C1=args.sAA_C1,
            sAA_C2=args.sAA_C2,
            zeta_C1=float(args.zeta_C1),
            zeta_C2=float(args.zeta_C2),
            t_start=float(args.t_start),
            dt_h=float(args.dt_h),
            outdir=str(mm_outdir),
            sobol=bool(args.sobol),
            sobol_baseN=int(args.sobol_baseN),
            smod2_vals=list(map(float, args.smod2_vals)),
            sAA2_vals=None if args.sAA2_vals is None else list(map(float, args.sAA2_vals)),
            half_life_vals=list(map(float, args.half_life_vals)),
            media_interval_vals=list(map(float, args.media_interval_vals)),
            zeta2_vals=list(map(float, args.zeta2_vals)),
            t_start_vals=list(map(float, args.t_start_vals)),
            k_vals=None if args.k_vals is None else list(map(float, args.k_vals)),
        )
        print(f"[Matrix/Mineral] completed; outputs in {mm_outdir.resolve()}")
    else:
        print("[Matrix/Mineral] skipped.")

    if "alp" in summary and "matrix" in summary:
        alp = summary["alp"]
        mm = summary["matrix"]

        t_alp_hours = np.asarray(alp["t_hours"], dtype=float)
        t_mm_days = np.asarray(mm["t_days"], dtype=float)
        t_mm_hours = t_mm_days * 24.0
        t_common_hours = np.union1d(t_alp_hours, t_mm_hours)
        t_common_days = t_common_hours / 24.0

        def interp_hours(src_t, src_y):
            return np.interp(t_common_hours, np.asarray(src_t, dtype=float), np.asarray(src_y, dtype=float))

        alp_total = interp_hours(t_alp_hours, alp["P_total"])
        alp_basal = interp_hours(t_alp_hours, alp["P_basal"])
        alp_osteo = interp_hours(t_alp_hours, alp["P_osteo"])

        mm_C1 = np.interp(t_common_days, t_mm_days, np.asarray(mm["C1"], dtype=float))
        mm_C2 = np.interp(t_common_days, t_mm_days, np.asarray(mm["C2"], dtype=float))
        mm_H1 = np.interp(t_common_days, t_mm_days, np.asarray(mm["H1_fraction"], dtype=float))
        mm_H2 = np.interp(t_common_days, t_mm_days, np.asarray(mm["H2_fraction"], dtype=float))

        np.savez(
            combined_dir / "datafit.npz",
            time_hours=t_common_hours,
            time_days=t_common_days,
            alp_total=alp_total,
            alp_basal=alp_basal,
            alp_osteo=alp_osteo,
            collagen_C1=mm_C1,
            collagen_C2=mm_C2,
            mineral_C1=mm_H1,
            mineral_C2=mm_H2,
            meta={"alp": str(alp["outdir"]), "matrix": str(mm["outdir"])},
        )

        df = pd.DataFrame(
            {
                "Hour": t_common_hours,
                "Day": t_common_days,
                "ALP_total": alp_total,
                "ALP_basal": alp_basal,
                "ALP_osteo": alp_osteo,
                "Collagen_C1": mm_C1,
                "Collagen_C2": mm_C2,
                "Mineral_C1": mm_H1,
                "Mineral_C2": mm_H2,
            }
        )
        df.to_excel(combined_dir / "datafit_timeseries.xlsx", index=False)

        np.savez(
            combined_dir / "datafit_hourly.npz",
            time_hours=t_common_hours,
            time_days=t_common_days,
            alp_total=alp_total,
            alp_basal=alp_basal,
            alp_osteo=alp_osteo,
            collagen_C1=mm_C1,
            collagen_C2=mm_C2,
            mineral_C1=mm_H1,
            mineral_C2=mm_H2,
        )

        print(f"[Combined] saved datafit bundle to {combined_dir}")

    return summary


if __name__ == "__main__":
    main()
