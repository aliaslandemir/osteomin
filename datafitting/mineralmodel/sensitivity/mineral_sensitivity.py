#!/usr/bin/env python3
"""
Multiday Sobol sensitivity analysis for the ACP → HA mineralization module.

- Uses core functions from `mineral_fit_model.py`
- First fits baseline K_nuc and K_ha to basal/osteo ARS endpoints
- Then runs Sobol sensitivity on mineral (HA proxy) at multiple days
  (default: 7, 14, 21, 28; final ARS readout is at day 28)

Outputs:
  * Sobol indices for each eval day for:
      - HA_osteo(day)                 (absolute osteogenic mineral level)
      - HA_osteo(day) - HA_basal(day) (osteogenic vs basal contrast)
  * CSV of all samples and outputs
  * Publication-style 2×2 bar plots (S1/ST with CI whiskers) over time
  * A short text report describing the setup and dominant parameters
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol

# Reuse your mineral model implementation
from mineral_fit_model import (
    load_pi_daily_osteo,
    load_collagen,
    simulate_condition,
    fit_rates,
)


# ---------------------------------------------------------
# Model wrapper for one Sobol sample (multi-day)
# ---------------------------------------------------------
def evaluate_mineral_sample_multiday(
    k_nuc_factor,
    k_ha_factor,
    collagen_factor,
    pi_factor,
    S_thresh_sample,
    n_exp_sample,
    ksp_factor,
    media_interval_cont,
    *,
    # baseline model / data
    K_nuc_base,
    K_ha_base,
    ksp_eff_base,
    ca_mM,
    pi_basal_mM,
    pi_osteo_mM,
    pi_daily_osteo_base,
    collagen_daily_base,
    days,
    dt,
    eval_days,
):
    """
    For a single Sobol parameter vector:
    - construct actual parameters by scaling baselines
    - simulate basal and osteo
    - return arrays of HA_osteo(day) and [HA_osteo(day) - HA_basal(day)]
      for all eval_days.
    """
    # Actual kinetic parameters
    K_nuc = K_nuc_base * k_nuc_factor
    K_ha = K_ha_base * k_ha_factor

    # Scalars applied to inputs / thermodynamics
    collagen_daily = collagen_daily_base * collagen_factor
    pi_daily_osteo = pi_daily_osteo_base * pi_factor
    ksp_eff = ksp_eff_base * ksp_factor

    # Supersaturation kinetics
    S_thresh = S_thresh_sample
    n_exp = n_exp_sample

    # Medium change interval as integer days
    media_interval = max(1, int(round(media_interval_cont)))

    days_int = int(days)
    pi_daily_basal = np.zeros(days_int)

    # Simulate basal
    t_b, Ca_b, Pi_b, ACP_b, HA_b, S_b = simulate_condition(
        K_nuc,
        K_ha,
        ca_mM,
        pi_basal_mM,
        pi_daily_basal,
        collagen_daily,
        ksp_eff,
        S_thresh,
        n_exp,
        days_int,
        dt,
        media_interval,
    )

    # Simulate osteo
    t_o, Ca_o, Pi_o, ACP_o, HA_o, S_o = simulate_condition(
        K_nuc,
        K_ha,
        ca_mM,
        pi_osteo_mM,
        pi_daily_osteo,
        collagen_daily,
        ksp_eff,
        S_thresh,
        n_exp,
        days_int,
        dt,
        media_interval,
    )

    # Pick values at evaluation days
    ha_osteo_vec = []
    ha_diff_vec = []

    for d in eval_days:
        idx = int(round(d / dt))
        idx = max(0, min(idx, len(HA_o) - 1))
        ha_basal = HA_b[idx]
        ha_osteo = HA_o[idx]
        ha_osteo_vec.append(ha_osteo)
        ha_diff_vec.append(ha_osteo - ha_basal)

    return np.array(ha_osteo_vec), np.array(ha_diff_vec)


# ---------------------------------------------------------
# Plotting helpers (multiday 2×2)
# ---------------------------------------------------------
def plot_sobol_multiday(
    df_by_day,
    eval_days,
    out_png,
    out_pdf,
    title,
    panel_title_prefix,
):
    """
    df_by_day: dict[day] -> DataFrame with columns
               parameter, S1, S1_conf, ST, ST_conf
    Creates a 2×2 multiday figure (up to 4 days).
    """
    days = list(eval_days)
    n_days = len(days)

    # Order parameters by ST of last day
    last_day = days[-1]
    df_last = df_by_day[last_day]
    params = df_last["parameter"].values
    order = np.argsort(df_last["ST"].values)[::-1]
    params_sorted = params[order]

    nrows, ncols = 2, 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 8), constrained_layout=True)

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
        }
    )

    x = np.arange(len(params_sorted))
    w = 0.35

    for i, day in enumerate(days):
        r = i // ncols
        c = i % ncols
        ax = axes[r, c]

        df = df_by_day[day].set_index("parameter").loc[params_sorted].reset_index()

        ax.bar(
            x - w / 2,
            df["S1"],
            w,
            yerr=df["S1_conf"],
            capsize=3,
            label="S1",
        )
        ax.bar(
            x + w / 2,
            df["ST"],
            w,
            yerr=df["ST_conf"],
            capsize=3,
            label="ST",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(params_sorted, rotation=45, ha="right")
        ax.set_ylabel("Sobol index")
        ax.set_title(f"{panel_title_prefix} at day {day}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(frameon=False)

    # Hide any unused axes if < 4 days
    for j in range(n_days, nrows * ncols):
        r = j // ncols
        c = j % ncols
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=15)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    p = argparse.ArgumentParser(
        description="Multiday Sobol sensitivity for ACP→HA mineralization (ARS)."
    )

    # Experimental endpoints (ARS-like, used to fit baseline K's)
    p.add_argument(
        "--basal_mineral",
        required=True,
        type=float,
        help="Basal endpoint mineral proxy (final HA units, from ARS).",
    )
    p.add_argument(
        "--osteo_mineral",
        required=True,
        type=float,
        help="Osteogenic endpoint mineral proxy (final HA units, from ARS).",
    )

    # Medium composition
    p.add_argument(
        "--ca_mM", type=float, default=2.0, help="Medium Ca concentration (mM)."
    )
    p.add_argument(
        "--pi_basal_mM",
        type=float,
        default=1.0,
        help="Initial Pi in basal medium (mM).",
    )
    p.add_argument(
        "--pi_osteo_mM",
        type=float,
        default=1.0,
        help="Initial Pi in osteogenic medium (mM).",
    )

    # Base medium change interval used in fitting
    p.add_argument(
        "--media_interval_base",
        type=int,
        default=2,
        help="Baseline medium change interval (days) used in K fit.",
    )

    # Pi input from ALP model
    p.add_argument(
        "--pi_daily_osteo_xlsx",
        type=str,
        default=None,
        help="Excel file with Pi_mM_daily for osteo.",
    )
    p.add_argument(
        "--pi_daily_sheet",
        type=str,
        default="Sheet1",
        help="Sheet name for Pi_mM_daily.",
    )
    p.add_argument(
        "--pi_daily_column",
        type=str,
        default="Pi_mM_daily",
        help="Column name in Pi file containing daily Pi input (mM/day).",
    )
    p.add_argument(
        "--pi_scale_osteo",
        type=float,
        default=1.0,
        help="Baseline scaling factor for osteo Pi_mM_daily.",
    )

    # Collagen input from AA → collagen model
    p.add_argument(
        "--collagen_xlsx",
        type=str,
        default=None,
        help="Excel file with collagen time course (e.g. aa_collagen_output.xlsx).",
    )
    p.add_argument(
        "--collagen_sheet", type=str, default="Sheet1", help="Sheet name for collagen."
    )
    p.add_argument(
        "--collagen_column",
        type=str,
        default="Collagen_total",
        help="Column name with collagen trajectory (e.g. Collagen_total).",
    )
    p.add_argument(
        "--collagen_scale",
        type=float,
        default=1.0,
        help="Baseline scaling factor for collagen gating.",
    )

    # Supersaturation and kinetics base parameters
    p.add_argument(
        "--ksp_eff",
        type=float,
        default=1.0,
        help="Baseline effective solubility product for S.",
    )
    p.add_argument(
        "--S_thresh",
        type=float,
        default=1.0,
        help="Baseline supersaturation threshold for nucleation.",
    )
    p.add_argument(
        "--n_exp",
        type=float,
        default=1.0,
        help="Baseline exponent for supersaturation driving term.",
    )

    # Simulation control
    p.add_argument("--days", type=int, default=28, help="Total simulation time (days).")
    p.add_argument("--dt", type=float, default=0.1, help="Time step (days).")
    p.add_argument(
        "--eval_days",
        type=float,
        nargs="+",
        default=[7, 14, 21, 28],
        help="Days at which mineral outputs are evaluated (space-separated list).",
    )

    # Sobol sampling control
    p.add_argument("--N", type=int, default=1024, help="Base Sobol sample size.")

    # Sensitivity parameter ranges (multiplicative factors / ranges)
    p.add_argument("--k_nuc_factor_min", type=float, default=0.5)
    p.add_argument("--k_nuc_factor_max", type=float, default=1.5)

    p.add_argument("--k_ha_factor_min", type=float, default=0.5)
    p.add_argument("--k_ha_factor_max", type=float, default=1.5)

    p.add_argument("--collagen_factor_min", type=float, default=0.5)
    p.add_argument("--collagen_factor_max", type=float, default=1.5)

    p.add_argument("--pi_factor_min", type=float, default=0.5)
    p.add_argument("--pi_factor_max", type=float, default=1.5)

    p.add_argument("--S_thresh_min", type=float, default=0.5)
    p.add_argument("--S_thresh_max", type=float, default=2.0)

    p.add_argument("--n_exp_min", type=float, default=0.5)
    p.add_argument("--n_exp_max", type=float, default=3.0)

    p.add_argument("--ksp_factor_min", type=float, default=0.5)
    p.add_argument("--ksp_factor_max", type=float, default=1.5)

    # Medium interval range for Sobol (different scenarios)
    p.add_argument("--media_min", type=float, default=1.0)
    p.add_argument("--media_max", type=float, default=3.0)

    # Output directory
    p.add_argument(
        "--outdir",
        type=str,
        default="mineral_sensitivity_output",
        help="Output directory.",
    )

    args = p.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    days = int(args.days)
    eval_days = sorted(set(float(d) for d in args.eval_days))

    print(f"[*] Evaluating mineral outputs at days: {eval_days}")

    # -----------------------------------------------------
    # 1) Load Pi and collagen time courses (baselines)
    # -----------------------------------------------------
    pi_daily_osteo_base = (
        load_pi_daily_osteo(
            args.pi_daily_osteo_xlsx,
            args.pi_daily_sheet,
            args.pi_daily_column,
            days,
        )
        * args.pi_scale_osteo
    )

    collagen_daily_base = (
        load_collagen(
            args.collagen_xlsx,
            args.collagen_sheet,
            args.collagen_column,
            days,
        )
        * args.collagen_scale
    )

    # -----------------------------------------------------
    # 2) Fit baseline K_nuc and K_ha to endpoints
    # -----------------------------------------------------
    print("[*] Fitting baseline K_nuc and K_ha to basal/osteo endpoints...")
    K_nuc_base, K_ha_base, fit_err = fit_rates(
        args.basal_mineral,
        args.osteo_mineral,
        args.ca_mM,
        args.pi_basal_mM,
        args.pi_osteo_mM,
        pi_daily_osteo_base,
        collagen_daily_base,
        args.ksp_eff,
        args.S_thresh,
        args.n_exp,
        days,
        args.dt,
        args.media_interval_base,
        logK_min=-4.0,
        logK_max=4.0,
        logK_points=33,
    )

    print(
        f"    Baseline fit: K_nuc = {K_nuc_base:.3e}, "
        f"K_ha = {K_ha_base:.3e}, squared error = {fit_err:.3e}"
    )

    # -----------------------------------------------------
    # 3) Sobol sampling problem definition
    # -----------------------------------------------------
    N = 2 ** int(np.ceil(np.log2(args.N)))  # enforce power-of-two N
    print(f"[*] Using Sobol sampler with N = {N}")

    problem = {
        "num_vars": 8,
        "names": [
            "k_nuc_factor",
            "k_ha_factor",
            "collagen_factor",
            "pi_factor",
            "S_thresh",
            "n_exp",
            "ksp_factor",
            "media_interval_days",
        ],
        "bounds": [
            [args.k_nuc_factor_min, args.k_nuc_factor_max],
            [args.k_ha_factor_min, args.k_ha_factor_max],
            [args.collagen_factor_min, args.collagen_factor_max],
            [args.pi_factor_min, args.pi_factor_max],
            [args.S_thresh_min, args.S_thresh_max],
            [args.n_exp_min, args.n_exp_max],
            [args.ksp_factor_min, args.ksp_factor_max],
            [args.media_min, args.media_max],
        ],
    }

    X = sobol_sampler.sample(problem, N, calc_second_order=True)
    print(f"[*] Total model evaluations: {len(X)}")

    n_samples = len(X)
    n_days = len(eval_days)

    Y_ha = np.zeros((n_samples, n_days))
    Y_diff = np.zeros((n_samples, n_days))

    # -----------------------------------------------------
    # 4) Model evaluations
    # -----------------------------------------------------
    for i, row in enumerate(X):
        (
            k_nuc_factor,
            k_ha_factor,
            collagen_factor,
            pi_factor,
            S_thresh_sample,
            n_exp_sample,
            ksp_factor,
            media_interval_cont,
        ) = row

        ha_vec, diff_vec = evaluate_mineral_sample_multiday(
            k_nuc_factor,
            k_ha_factor,
            collagen_factor,
            pi_factor,
            S_thresh_sample,
            n_exp_sample,
            ksp_factor,
            media_interval_cont,
            K_nuc_base=K_nuc_base,
            K_ha_base=K_ha_base,
            ksp_eff_base=args.ksp_eff,
            ca_mM=args.ca_mM,
            pi_basal_mM=args.pi_basal_mM,
            pi_osteo_mM=args.pi_osteo_mM,
            pi_daily_osteo_base=pi_daily_osteo_base,
            collagen_daily_base=collagen_daily_base,
            days=days,
            dt=args.dt,
            eval_days=eval_days,
        )

        Y_ha[i, :] = ha_vec
        Y_diff[i, :] = diff_vec

    # -----------------------------------------------------
    # 5) Sobol analysis for each day
    # -----------------------------------------------------
    df_ha_by_day = {}
    df_diff_by_day = {}

    for j, day in enumerate(eval_days):
        print(f"[*] Sobol analysis: HA_osteo at day {day} ...")
        Si_ha = sobol.analyze(problem, Y_ha[:, j], calc_second_order=True)

        print(f"[*] Sobol analysis: HA_osteo - HA_basal at day {day} ...")
        Si_diff = sobol.analyze(problem, Y_diff[:, j], calc_second_order=True)

        df_ha = pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": Si_ha["S1"],
                "S1_conf": Si_ha["S1_conf"],
                "ST": Si_ha["ST"],
                "ST_conf": Si_ha["ST_conf"],
            }
        )
        df_diff = pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": Si_diff["S1"],
                "S1_conf": Si_diff["S1_conf"],
                "ST": Si_diff["ST"],
                "ST_conf": Si_diff["ST_conf"],
            }
        )

        day_int = int(round(day))
        df_ha.to_csv(
            outdir / f"mineral_sobol_HA_osteo_day{day_int}.csv",
            index=False,
        )
        df_diff.to_csv(
            outdir / f"mineral_sobol_HA_contrast_day{day_int}.csv",
            index=False,
        )

        df_ha_by_day[day_int] = df_ha
        df_diff_by_day[day_int] = df_diff

    # -----------------------------------------------------
    # 6) Save samples + outputs
    # -----------------------------------------------------
    df_samples = pd.DataFrame(X, columns=problem["names"])
    for j, day in enumerate(eval_days):
        d_int = int(round(day))
        df_samples[f"HA_osteo_day{d_int}"] = Y_ha[:, j]
        df_samples[f"HA_osteo_minus_basal_day{d_int}"] = Y_diff[:, j]

    df_samples.to_csv(outdir / "mineral_sensitivity_samples.csv", index=False)

    # -----------------------------------------------------
    # 7) Multiday plots
    # -----------------------------------------------------
    eval_int_days = [int(round(d)) for d in eval_days]

    # HA_osteo
    plot_sobol_multiday(
        df_ha_by_day,
        eval_int_days,
        outdir / "mineral_sobol_bars_HA_osteo_multiday.png",
        outdir / "mineral_sobol_bars_HA_osteo_multiday.pdf",
        title="Mineral module (ACP→HA): Sobol sensitivity over time (HA_osteo)",
        panel_title_prefix="HA_osteo",
    )

    # HA_osteo - HA_basal
    plot_sobol_multiday(
        df_diff_by_day,
        eval_int_days,
        outdir / "mineral_sobol_bars_contrast_multiday.png",
        outdir / "mineral_sobol_bars_contrast_multiday.pdf",
        title="Mineral module (ACP→HA): Sobol sensitivity over time (contrast)",
        panel_title_prefix="HA_osteo - HA_basal",
    )

    # -----------------------------------------------------
    # 8) Text summary (based on last day, typically 28)
    # -----------------------------------------------------
    last_day = eval_int_days[-1]
    df_last = df_ha_by_day[last_day]
    dom_row = df_last.loc[df_last["ST"].idxmax()]

    text = f"""
Multiday Sobol global sensitivity: ACP → HA mineralization
---------------------------------------------------------

Baseline fit to endpoints:
- Basal mineral = {args.basal_mineral}
- Osteo mineral = {args.osteo_mineral}
- Fitted K_nuc = {K_nuc_base:.3e}
- Fitted K_ha  = {K_ha_base:.3e}
- Squared error = {fit_err:.3e}

Inputs and trajectories:
- Pi_mM_daily(osteo) from: {args.pi_daily_osteo_xlsx or 'None (zeros)'}
- Collagen_total from: {args.collagen_xlsx or 'None (constant ones)'}
- Baseline ksp_eff = {args.ksp_eff}, S_thresh = {args.S_thresh}, n_exp = {args.n_exp}
- Baseline medium interval for fit = {args.media_interval_base} days
- Medium interval range in Sobol: {args.media_min}–{args.media_max} days

Sobol settings:
- Base N = {args.N}, used N = {N} (power of 2)
- Total evaluations = {len(X)}
- Outputs at days {eval_int_days}:
    1) HA_osteo(day)
    2) HA_osteo(day) - HA_basal(day)

Example (day {last_day}):
- Most influential parameter for HA_osteo(day {last_day}) by total-effect index:
  {dom_row['parameter']} (ST = {dom_row['ST']:.3f})

See:
- mineral_sobol_HA_osteo_day*.csv
- mineral_sobol_HA_contrast_day*.csv
- mineral_sobol_bars_HA_osteo_multiday.pdf
- mineral_sobol_bars_contrast_multiday.pdf

for full rankings and confidence intervals across time points.
"""

    with open(outdir / "mineral_sobol_results_multiday.txt", "w") as f:
        f.write(text)

    print("\n[✓] Finished multiday mineral sensitivity analysis.")
    print(f"Results, plots, and text saved in: {outdir.resolve()}")


if __name__ == "__main__":
    main()
