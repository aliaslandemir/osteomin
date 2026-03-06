"""



Entry point for OsteoMin.

python main.py \
  --hours 672 \
  --cells 72 \
  --seed-mode random \
  --seed 12345 \
  --w 33 --h 33 \
  --nodules 40n \
  --dex-conc 10 \
  --mk4-conc 10 \
  --alp-rb 0.001 \
  --alp-ro 0.002 \
  --alp-a-d 0.08 \
  --alp-b-d 0.08 \
  --alp-a-k 0.06 \
  --alp-b-k 0.06 \
  --collagen-rb 0.001 \
  --collagen-ro 0.002 \
  --collagen-a-d 0.08 \
  --collagen-b-d 0.08 \
  --collagen-a-k 0.06 \
  --collagen-b-k 0.06 \
  --ha-model minimal \
  --ha-ro 0.075 \
  --ha-a-d 0.08 \
  --ha-b-d 0.08 \
  --ha-a-k 0.06 \
  --ha-b-k 0.06 \
  --ha-tau 384 \
  --ha-hill-n 4 \
  --boundary-mode periodic \
  --media-interval 48 \
  --random-move --move-prob 0.27 --max-moves 1 \
  --proliferation --ob-div-prob 0.0 \
  --apoptosis --apoptosis-prob 0.00042 \
  --osteogenic-commitment \
  --osteo-induction-day 3 \
  --commit-delay 48 \
  --commit-k 0.08 \
  --commit-k-growth 0.0008 \
  --aa0 1.0 \
  --aa-half-life 6 \
  --collagen-aa-km 1.0 \
  --min-div-hours 38 --max-div-hours 52 \
  --secretion \
  --spatial-every 1 \
  --out main_out \
  --alp-plot-hours 240 \
  --collagen-plot-hours 336 \
  --mineral-plot-hours 672 \
  --no-hourly-exports \
  --runtime










"""

from __future__ import annotations
import argparse
from pathlib import Path
from typing import Iterable

import numpy as np

from src.model import OsteoModel
from src.config import FieldConfig, ABMConfig
from src.modulatoreff import print_drug_response_summary
from src.time_utils import HOURS_PER_DAY
from src.viz import plot_summary, export_overlay_collagen_ha


def _global_field_max(grids_dir: Path, field_prefix: str, days: Iterable[int]) -> float:
    """Compute the maximum value across stored grids for `field_prefix`."""
    max_val = 0.0
    for day in days:
        path = grids_dir / f"{field_prefix}_day_{day:03d}.npy"
        if not path.exists():
            continue
        arr = np.load(path)
        if arr.size == 0:
            continue
        max_val = max(max_val, float(np.nanmax(arr)))
    return max(max_val, 1e-9)


def _grid_seed_coords(w: int, h: int, spacing: int) -> list[tuple[int, int]]:
    if spacing <= 0:
        raise ValueError("grid spacing must be >= 1")
    coords = []
    for y in range(0, h, spacing):
        for x in range(0, w, spacing):
            coords.append((x, y))
    return coords


def parse_args() -> argparse.Namespace:
    default_abm = ABMConfig()
    p = argparse.ArgumentParser(
        description="OsteoMin: MC3T3-E1 ABM/field bone remodeling simulator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    p.add_argument(
        "--hours",
        type=int,
        default=672,
        help="total hours to simulate (24 hours per day is assumed)",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="legacy alias; multiplied by 24 if --hours is omitted",
    )
    p.add_argument(
        "--cells", type=int, default=72, help="initial number of seeded cells"
    )
    p.add_argument(
        "--seed-mode",
        choices=["random", "grid"],
        default="random",
        help="initial seeding strategy: random uniform or grid-spaced then sampled",
    )
    p.add_argument(
        "--grid-spacing",
        type=int,
        default=3,
        help="grid spacing (n) for grid seeding; one candidate site per n×n block",
    )
    p.add_argument(
        "--media-interval",
        type=int,
        default=48,
        help="hours between media changes (legacy 2 days = 48 hours)",
    )
    p.add_argument(
        "--dex-conc",
        "--dex",
        type=float,
        default=10.0,
        help="Dex concentration c_D used to compute log10(c_D/c_D0) with c_D0 fixed to 10 nM; c_D<=0 is treated as baseline (x_D=0)",
    )
    p.add_argument(
        "--mk4-conc",
        "--mk4",
        type=float,
        default=10.0,
        help="MK-4 concentration c_K used to compute log10(c_K/c_K0) with c_K0 fixed to 1.0 uM; c_K<=0 is treated as baseline (x_K=0)",
    )
    p.add_argument(
        "--alp-rb",
        type=float,
        default=0.001,
        help="basal ALP rate r_b^P per preOB site per hour",
    )
    p.add_argument(
        "--alp-ro",
        type=float,
        default=0.002,
        help="osteogenic ALP rate r_o^P per OB site per hour",
    )
    p.add_argument(
        "--alp-p0",
        type=float,
        default=0.0,
        help="initial ALP field baseline P0 applied at simulation start",
    )
    p.add_argument(
        "--alp-a-d",
        type=float,
        default=0.08,
        help="Dex peaked-response height a_D^P in m_D^P = 1 + I_D (a_D^P - b_D^P x_D^2)",
    )
    p.add_argument(
        "--alp-b-d",
        type=float,
        default=0.08,
        help="Dex peaked-response curvature b_D^P in m_D^P = 1 + I_D (a_D^P - b_D^P x_D^2)",
    )
    p.add_argument(
        "--alp-a-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response height a_K^P in m_K^P = 1 + I_K (a_K^P - b_K^P x_K^2)",
    )
    p.add_argument(
        "--alp-b-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response curvature b_K^P in m_K^P = 1 + I_K (a_K^P - b_K^P x_K^2)",
    )
    p.add_argument(
        "--collagen-rb",
        type=float,
        default=0.001,
        help="basal collagen rate r_b^C per preOB site per hour",
    )
    p.add_argument(
        "--collagen-ro",
        type=float,
        default=0.002,
        help="osteogenic collagen rate r_o^C per OB site per hour before AA gating",
    )
    p.add_argument(
        "--collagen-a-d",
        type=float,
        default=0.08,
        help="Dex peaked-response height a_D^C in m_D^C = 1 + I_D (a_D^C - b_D^C x_D^2)",
    )
    p.add_argument(
        "--collagen-b-d",
        type=float,
        default=0.08,
        help="Dex peaked-response curvature b_D^C in m_D^C = 1 + I_D (a_D^C - b_D^C x_D^2)",
    )
    p.add_argument(
        "--collagen-a-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response height a_K^C in m_K^C = 1 + I_K (a_K^C - b_K^C x_K^2)",
    )
    p.add_argument(
        "--collagen-b-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response curvature b_K^C in m_K^C = 1 + I_K (a_K^C - b_K^C x_K^2)",
    )
    p.add_argument(
        "--aa0",
        "--aa-initial",
        type=float,
        default=1.0,
        help="AA concentration A0 immediately after medium refresh",
    )
    p.add_argument(
        "--aa-half-life",
        "--aa-half-life-hours",
        type=float,
        default=6.0,
        help="AA half-life in hours for exponential decay between media refreshes",
    )
    p.add_argument(
        "--collagen-aa-km",
        type=float,
        default=1.0,
        help="half-saturation constant K_A^C for AA-gated collagen production",
    )
    p.add_argument(
        "--ha-model",
        type=str,
        choices=["minimal"],
        default="minimal",
        help="HAp production model for OsteoMin field dynamics",
    )
    p.add_argument(
        "--ha-ro",
        type=float,
        default=0.075,
        help="osteogenic HAp rate r_o^H per OB site per hour before Hill gating",
    )
    p.add_argument(
        "--ha-h0",
        type=float,
        default=0.0,
        help="initial mineral field baseline H0 applied at simulation start",
    )
    p.add_argument(
        "--ha-a-d",
        type=float,
        default=0.08,
        help="Dex peaked-response height a_D^H in m_D^H = 1 + I_D (a_D^H - b_D^H x_D^2)",
    )
    p.add_argument(
        "--ha-b-d",
        type=float,
        default=0.08,
        help="Dex peaked-response curvature b_D^H in m_D^H = 1 + I_D (a_D^H - b_D^H x_D^2)",
    )
    p.add_argument(
        "--ha-a-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response height a_K^H in m_K^H = 1 + I_K (a_K^H - b_K^H x_K^2)",
    )
    p.add_argument(
        "--ha-b-k",
        type=float,
        default=0.06,
        help="MK-4 peaked-response curvature b_K^H in m_K^H = 1 + I_K (a_K^H - b_K^H x_K^2)",
    )
    p.add_argument(
        "--ha-tau",
        type=float,
        default=384.0,
        help="tau (hours) for minimal HAp Hill activation",
    )
    p.add_argument(
        "--ha-hill-n",
        type=float,
        default=4.0,
        help="Hill exponent n for minimal HAp activation",
    )
    p.add_argument(
        "--ha-collagen-maturation-tau",
        type=float,
        default=48.0,
        help="tau_col in hours for first-order collagen maturation before mineralization",
    )
    p.add_argument(
        "--ha-collagen-km",
        type=float,
        default=1.0,
        help="K_col half-saturation constant for matured-collagen mineral support",
    )
    p.add_argument(
        "--seed", type=int, default=12345, help="random seed for reproducibility"
    )

    p.add_argument("--w", type=int, default=33, help="grid width")
    p.add_argument("--h", type=int, default=33, help="grid height")
    p.add_argument(
        "--boundary-mode",
        choices=["clamp", "periodic"],
        default="periodic",
        help="treat edges as hard boundaries (clamp) or wrap around (periodic)",
    )

    p.add_argument(
        "--random-move",
        "--cell-movement",
        dest="random_move",
        action="store_true",
        help="enable random_move motility",
    )
    p.add_argument(
        "--no-random-move",
        "--no-cell-movement",
        dest="random_move",
        action="store_false",
        help="disable random_move motility",
    )
    p.set_defaults(random_move=True)

    p.add_argument(
        "--move-prob",
        type=float,
        default=0.27,
        help="probability a preOB cell attempts a move per hour",
    )
    p.add_argument(
        "--max-moves",
        type=int,
        default=1,
        help="max random-walk steps per preOB cell per hour",
    )
    p.add_argument(
        "--min-div-hours",
        type=int,
        default=default_abm.min_div_hours,
        help="minimum hours between divisions for a cell",
    )
    p.add_argument(
        "--max-div-hours",
        type=int,
        default=default_abm.max_div_hours,
        help="maximum hours between divisions for a cell",
    )

    p.add_argument(
        "--ob-div-prob",
        type=float,
        default=0.0,
        help="rare division probability for osteoblasts",
    )

    p.add_argument(
        "--osteogenic-commitment",
        dest="commitment",
        action="store_true",
        help="enable preOB → OB differentiation",
    )
    p.add_argument(
        "--no-osteogenic-commitment",
        dest="commitment",
        action="store_false",
        help="disable preOB → OB differentiation",
    )
    p.set_defaults(commitment=True)

    p.add_argument(
        "--commit-growth",
        type=float,
        default=0.01 / (24 * 24),
        help="legacy per-hour increase in commitment probability (used only when --commit-k and --commit-k-growth are both 0)",
    )
    p.add_argument(
        "--commit-base",
        type=float,
        default=default_abm.base_commit_prob,
        help="legacy base per-hour commitment probability (used only when --commit-k and --commit-k-growth are both 0)",
    )

    p.add_argument(
        "--commit-delay",
        type=int,
        default=48,
        help="hours after osteogenic induction before preOB cells can start committing (default 2 days)",
    )
    p.add_argument(
        "--osteo-induction-day",
        type=float,
        default=3.0,
        help="day of osteogenic induction (baseline day 3 -> 72 hours)",
    )
    p.add_argument(
        "--commit-k",
        type=float,
        default=0.08,
        help="commitment hazard rate k in day^-1 after induction+delay gate",
    )
    p.add_argument(
        "--commit-k-growth",
        type=float,
        default=0.0008,
        help="hourly increase of commitment hazard k (day^-1 per hour) after induction+delay gate",
    )

    p.add_argument(
        "--apoptosis", dest="apoptosis", action="store_true", help="enable apoptosis"
    )
    p.add_argument(
        "--no-apoptosis",
        dest="apoptosis",
        action="store_false",
        help="disable apoptosis",
    )
    p.set_defaults(apoptosis=True)

    p.add_argument(
        "--apoptosis-prob",
        type=float,
        default=0.00042,
        help="per-cell apoptosis probability each hour (legacy 0.01/day)",
    )

    p.add_argument(
        "--apoptosis-osteo",
        dest="ob_apoptosis_prob",
        type=float,
        default=0.001 / 24,
        help="legacy argument (ignored): osteoblast apoptosis is disabled",
    )

    p.add_argument(
        "--proliferation",
        dest="proliferation",
        action="store_true",
        help="enable proliferation",
    )
    p.add_argument(
        "--no-proliferation",
        dest="proliferation",
        action="store_false",
        help="disable proliferation",
    )
    p.set_defaults(proliferation=True)

    p.add_argument(
        "--proliferation-chance",
        type=float,
        default=1.0,
        help="per-cell chance to attempt division per hour once the interval passes",
    )

    p.add_argument(
        "--secretion",
        dest="secretion",
        action="store_true",
        help="enable secretion (OB → collagen/ALP)",
    )
    p.add_argument(
        "--no-secretion",
        dest="secretion",
        action="store_false",
        help="disable secretion",
    )
    p.set_defaults(secretion=True)

    p.add_argument(
        "--nodules",
        type=str,
        choices=["4n", "8n", "24n", "40n", "60n", "84n", "112n", "144n"],
        default="40n",
        help="nodule connectivity rule: 4n (Manhattan r=1), 8n (Moore r=1), 24n (Moore r=2), 40n (Manhattan r=4), 60n (Manhattan r=5), 84n (Manhattan r=6), 112n (Manhattan r=7), 144n (Manhattan r=8)",
    )
    p.add_argument(
        "--spatial-every",
        type=int,
        default=1,
        help="export spatial PNG every N days (0 = off)",
    )
    p.add_argument(
        "--spatial-days",
        type=str,
        default="",
        help="explicit export days, comma-separated e.g. '0,5,10,20'",
    )
    p.add_argument("--out", type=str, default="main_out", help="output directory")
    p.add_argument(
        "--xlsx", dest="xlsx", action="store_true", help="write Excel timeseries"
    )
    p.add_argument(
        "--no-xlsx",
        dest="xlsx",
        action="store_false",
        help="skip Excel timeseries export",
    )
    p.set_defaults(xlsx=True)
    p.add_argument(
        "--uniqueID",
        "--unique-id",
        dest="unique_id",
        action="store_true",
        help="track per-cell history via stable IDs and export it",
    )
    p.add_argument(
        "--no-unique-id",
        dest="unique_id",
        action="store_false",
        help="skip unique cell history export",
    )
    p.set_defaults(unique_id=True)
    p.add_argument("--csv", action="store_true", help="write CSV timeseries")
    p.add_argument(
        "--snapshots",
        dest="snapshots",
        action="store_true",
        help="also save raw .npy grids per day",
    )
    p.add_argument(
        "--no-snapshots",
        dest="snapshots",
        action="store_false",
        help="disable raw .npy grid dumps",
    )
    p.set_defaults(snapshots=True)
    p.add_argument(
        "--hourly-exports",
        dest="hourly_exports",
        action="store_true",
        help="write hourly spatial/collagen snapshots",
    )
    p.add_argument(
        "--no-hourly-exports",
        dest="hourly_exports",
        action="store_false",
        help="skip hourly snapshot exports",
    )
    p.set_defaults(hourly_exports=False)
    p.add_argument(
        "--runtime",
        action="store_true",
        help="print low-overhead runtime progress during the simulation",
    )
    p.add_argument(
        "--runtime-every",
        type=int,
        default=1,
        help="print runtime progress every N simulated days",
    )
    p.add_argument(
        "--datafit",
        type=str,
        default=None,
        help="deprecated legacy option; datafit-driven dynamics are disabled",
    )
    p.add_argument(
        "--datafit-condition",
        type=str,
        default="C2",
        help="deprecated legacy option; ignored unless --datafit is supplied",
    )
    p.add_argument(
        "--datafit-all",
        action="store_true",
        help="deprecated legacy option; ignored unless --datafit is supplied",
    )
    p.add_argument(
        "--alp-plot-hours",
        type=int,
        default=240,
        help="limit ALP plot to this many hours (0 = no limit)",
    )
    p.add_argument(
        "--collagen-plot-hours",
        type=int,
        default=336,
        help="limit collagen plot to this many hours (0 = no limit)",
    )
    p.add_argument(
        "--mineral-plot-hours",
        type=int,
        default=672,
        help="limit mineral (ACP/HAp) plot to this many hours (0 = no limit)",
    )

    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    args.move_prob = max(0.0, min(1.0, args.move_prob))
    args.ob_div_prob = max(0.0, min(1.0, args.ob_div_prob))
    args.proliferation_chance = max(0.0, min(1.0, args.proliferation_chance))
    args.apoptosis_prob = max(0.0, min(1.0, args.apoptosis_prob))
    args.ob_apoptosis_prob = max(0.0, min(1.0, args.ob_apoptosis_prob))
    args.commit_delay = max(0, args.commit_delay)
    args.osteo_induction_day = max(0.0, float(args.osteo_induction_day))
    args.commit_k = max(0.0, float(args.commit_k))
    args.commit_k_growth = max(0.0, float(args.commit_k_growth))
    if args.dex_conc != 0.0 and not (1.0 <= float(args.dex_conc) <= 100.0):
        raise SystemExit(
            f"Dex concentration {args.dex_conc:g} nM is outside the supported non-toxic range of 1 to 100 nM"
        )
    if args.mk4_conc != 0.0 and not (0.1 <= float(args.mk4_conc) <= 10.0):
        raise SystemExit(
            f"MK-4 concentration {args.mk4_conc:g} uM is outside the supported non-toxic range of 0.1 to 10 uM"
        )
    if args.datafit:
        raise SystemExit(
            "legacy --datafit execution is disabled; OsteoMin now uses equation-based dynamics only"
        )
    induction_hours = int(round(args.osteo_induction_day * HOURS_PER_DAY))
    abm_cfg = ABMConfig(
        width=args.w,
        height=args.h,
        boundary_mode=args.boundary_mode,
        enable_random_move=args.random_move,
        enable_commitment=args.commitment,
        enable_apoptosis=args.apoptosis,
        enable_proliferation=args.proliferation,
        enable_secretion=args.secretion,
        move_prob=args.move_prob,
        max_moves_per_hour=args.max_moves,
        ob_division_prob=args.ob_div_prob,
        apoptosis_prob=args.apoptosis_prob,
        ob_apoptosis_prob=args.ob_apoptosis_prob,
        proliferation_chance=args.proliferation_chance,
        base_commit_prob=args.commit_base,
        commit_prob_growth=args.commit_growth,
        osteo_induction_hours=induction_hours,
        commit_delay=args.commit_delay,
        commit_k_day=args.commit_k,
        commit_k_growth_per_hour=args.commit_k_growth,
        min_div_hours=max(1, int(args.min_div_hours)),
        max_div_hours=max(int(args.min_div_hours), int(args.max_div_hours)),
    )
    field_cfg = FieldConfig(
        width=args.w,
        height=args.h,
        media_interval=args.media_interval,
        enable_secretion=args.secretion,
        osteo_induction_hours=induction_hours,
        dex_conc=args.dex_conc,
        mk4_conc=args.mk4_conc,
        dex_ref_conc=10.0,
        mk4_ref_conc=1.0,
        alp_basal_rate=args.alp_rb,
        alp_osteo_rate=args.alp_ro,
        alp_initial=args.alp_p0,
        alp_a_d=args.alp_a_d,
        alp_b_d=args.alp_b_d,
        alp_a_k=args.alp_a_k,
        alp_b_k=args.alp_b_k,
        collagen_basal_rate=args.collagen_rb,
        collagen_osteo_rate=args.collagen_ro,
        collagen_a_d=args.collagen_a_d,
        collagen_b_d=args.collagen_b_d,
        collagen_a_k=args.collagen_a_k,
        collagen_b_k=args.collagen_b_k,
        aa_initial=args.aa0,
        aa_half_life_hours=args.aa_half_life,
        collagen_aa_half_sat=args.collagen_aa_km,
        ha_model=args.ha_model,
        ha_initial=args.ha_h0,
        ha_osteo_rate=args.ha_ro,
        ha_a_d=args.ha_a_d,
        ha_b_d=args.ha_b_d,
        ha_a_k=args.ha_a_k,
        ha_b_k=args.ha_b_k,
        ha_tau_hours=args.ha_tau,
        ha_hill_n=args.ha_hill_n,
        ha_t_commit_hours=induction_hours + args.commit_delay,
        ha_collagen_maturation_tau_hours=args.ha_collagen_maturation_tau,
        ha_collagen_half_sat=args.ha_collagen_km,
    )
    print_drug_response_summary(field_cfg)
    spatial_days = [int(x) for x in args.spatial_days.split(",") if x.strip()]

    if args.hours is not None:
        total_hours = max(0, int(args.hours))
    else:
        total_hours = max(0, int(args.days) * HOURS_PER_DAY)
    if total_hours <= 0:
        raise SystemExit("specify a positive --hours or --days value")

    def _run_condition(cond_outdir: Path) -> None:
        model = OsteoModel(
            outdir=cond_outdir,
            total_hours=total_hours,
            abm_cfg=abm_cfg,
            field_cfg=field_cfg,
            seed=args.seed,
            spatial_every=args.spatial_every,
            spatial_days=spatial_days,
            snapshots_enabled=args.snapshots,
            unique_id_tracking=args.unique_id,
            hourly_exports=args.hourly_exports,
            show_runtime=args.runtime,
            runtime_every_days=args.runtime_every,
            nodule_connectivity=args.nodules,
        )

        if args.seed_mode == "random":
            model.seed_random(args.cells, abm_cfg.width, abm_cfg.height)
        else:
            sites = _grid_seed_coords(abm_cfg.width, abm_cfg.height, args.grid_spacing)
            if not sites:
                raise SystemExit(
                    "grid spacing produces no seed sites; adjust --grid-spacing"
                )
            n = min(int(args.cells), len(sites))
            idx = model.rng.choice(len(sites), size=n, replace=False)
            coords = [sites[i] for i in idx]
            model.abm.seed_from_coords(coords, day0=0)
        df = model.run()

        df["dead_total_today"] = df["dead_preob_today"] + df["dead_ob_today"]
        df["dead_preob_cum"] = df["dead_preob_today"].cumsum()
        df["dead_ob_cum"] = df["dead_ob_today"].cumsum()
        df["dead_total_cum"] = df["dead_total_today"].cumsum()

        if args.xlsx:
            model.save_timeseries_excel(df, "timeseries")
        if args.csv:
            model.save_timeseries_csv(df, "timeseries")
        if args.unique_id:
            model.save_cell_history_excel("cell_history")

        plot_summary(
            df,
            cond_outdir,
            field_cfg=field_cfg,
            modulator_df=model.field.modulator_frame(),
            total_hours=total_hours,
            alp_plot_hours=args.alp_plot_hours,
            collagen_plot_hours=args.collagen_plot_hours,
            mineral_plot_hours=args.mineral_plot_hours,
        )

        total_days = model.days
        overlay_days = sorted({d for d in model.spatial_days if 0 <= d < total_days})
        if args.spatial_every > 0:
            overlay_days = sorted(
                {
                    *overlay_days,
                    *{day for day in range(0, total_days, args.spatial_every)},
                }
            )
        if not args.snapshots:
            print("snapshots disabled; skipping collagen/HAp overlays")
        else:
            all_days = range(total_days)
            collagen_vmax = _global_field_max(model.grids_dir, "collagen", all_days)
            ha_vmax = _global_field_max(model.grids_dir, "ha", all_days)
            prev_col: np.ndarray | None = None
            prev_ha: np.ndarray | None = None
            prev_meta: tuple | None = None
            df_by_day = df.set_index("day")
            for day in overlay_days:
                col = np.load(model.grids_dir / f"collagen_day_{day:03d}.npy")
                ha = np.load(model.grids_dir / f"ha_day_{day:03d}.npy")

                n_preob = n_ob = dead_today = nodule_count = None
                nodule_centers = None
                if day in df_by_day.index:
                    row = df_by_day.loc[day]
                    n_preob = int(row["n_preob"])
                    n_ob = int(row["n_osteoblast"])
                    dead_today = int(row["dead_total_today"])
                    if "nodule_count_current" in row:
                        nodule_count = int(row["nodule_count_current"])
                    elif "nodule_count" in row:
                        nodule_count = int(row["nodule_count"])
                if day in model.nodule_centers_by_day:
                    nodule_centers = model.nodule_centers_by_day[day]
                centers_sig = ()
                if nodule_centers:
                    centers_sig = tuple(
                        (round(float(x), 3), round(float(y), 3))
                        for x, y in nodule_centers
                    )
                meta_sig = (n_preob, n_ob, dead_today, nodule_count, centers_sig)
                if (
                    prev_col is not None
                    and prev_ha is not None
                    and prev_meta is not None
                    and np.array_equal(col, prev_col)
                    and np.array_equal(ha, prev_ha)
                    and meta_sig == prev_meta
                ):
                    continue
                prev_col = col
                prev_ha = ha
                prev_meta = meta_sig

                export_overlay_collagen_ha(
                    cond_outdir / "collagen_ha" / f"collagen_ha_day_{day:03d}.pdf",
                    f"Day {day:02d}",
                    col,
                    ha,
                    n_preob=n_preob,
                    n_ob=n_ob,
                    dead_today=dead_today,
                    nodule_count=nodule_count,
                    nodule_centers=nodule_centers,
                    global_vmax_col=collagen_vmax,
                    global_vmax_ha=ha_vmax,
                )

        print("[OsteoMin]")
        print(df.tail(5))

    _run_condition(outdir)


if __name__ == "__main__":
    main()
