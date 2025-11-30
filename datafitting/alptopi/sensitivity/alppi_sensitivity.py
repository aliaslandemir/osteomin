#!/usr/bin/env python3
"""
Multiday Sobol sensitivity analysis for the β-GP / ALP → Pi module.

- Uses the model implemented in `alp_linear_model.py`
- Sobol global sensitivity for cumulative Pi at multiple time points
  (default days 7, 14, 21, 28)
- Varies:
    * basal cellular ALP
    * osteogenic cellular ALP
    * residual serum ALP
    * β-GP concentration
    * medium change interval
    * ALP ramp start day
    * ALP ramp end / measurement day

Outputs:
    * CSVs with Sobol indices per day
    * CSV with all samples and Pi_cum outputs
    * PNG/PDF 2×2 bar plot (S1/ST with CI's) for Pi_cum at all days
    * A small text report describing the setup and dominant parameters
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sampler
from SALib.analyze import sobol

from alp_linear_model import alp_linear_profile, simulate_bgp_to_pi


# ---------------------------------------------------------
# Wrapper: one model evaluation (multi-day)
# ---------------------------------------------------------
def compute_pi_cumulative_multiday(
    basal_alp,
    osteo_alp,
    serum_alp,
    bgp_mM,
    media_interval_days,
    start_day,
    measure_day,
    *,
    days_total,
    eval_days,
):
    """
    Run ALP(t) + β-GP→Pi for one parameter set and return
    cumulative Pi at all eval_days (list of days).
    """
    days_total_int = int(round(days_total))
    media_int = max(1, int(round(media_interval_days)))
    start_day_int = max(0, int(round(start_day)))
    measure_day_int = max(start_day_int + 1, int(round(measure_day)))

    # ALP(t) profile (hourly)
    alp_hours = alp_linear_profile(
        basal_cell_alp=basal_alp,
        osteo_cell_alp=osteo_alp,
        serum_alp=serum_alp,
        start_day=start_day_int,
        measure_day=measure_day_int,
        total_days=days_total_int,
    )

    # β-GP → Pi simulation
    days, pi_daily, pi_cum, bgp_daily = simulate_bgp_to_pi(
        bgp_mM=bgp_mM,
        alp_hours=alp_hours,
        total_days=days_total_int,
        media_interval_days=media_int,
        start_day=start_day_int,
    )

    def get_pi_at_day(t):
        d = max(1, min(int(round(t)), days_total_int))
        return pi_cum[d - 1]

    return np.array([get_pi_at_day(d) for d in eval_days], dtype=float)


# ---------------------------------------------------------
# Plotting helper: multiday, 2×2 like collagen
# ---------------------------------------------------------
def plot_sobol_multiday(df_by_day, eval_days, out_png, out_pdf, title):
    """
    df_by_day: dict[day] -> DataFrame with columns:
               parameter, S1, S1_conf, ST, ST_conf
    """
    days = list(eval_days)
    n_days = len(days)

    # Use last day to order parameters (most "mature" Pi)
    last_day = days[-1]
    df_last = df_by_day[last_day]
    params = df_last["parameter"].values
    order = np.argsort(df_last["ST"].values)[::-1]
    params_sorted = params[order]

    # layout: 2×2 (up to 4 days)
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
        ax.set_title(f"Pi_cum at day {day}")
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        if i == 0:
            ax.legend(frameon=False)

    # Hide any unused panels if < 4 days
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
    parser = argparse.ArgumentParser(
        description="Multiday Sobol sensitivity for β-GP / ALP → Pi module"
    )

    # Global settings
    parser.add_argument("--N", type=int, default=1024, help="Base Sobol sample size")
    parser.add_argument("--days_total", type=int, default=28, help="Total days")
    parser.add_argument(
        "--eval_days",
        type=int,
        nargs="+",
        default=[7, 14, 21, 28],
        help="Days at which cumulative Pi is evaluated (space-separated list)",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="alppi_sensitivity_output",
        help="Output directory",
    )

    # Parameter ranges
    parser.add_argument(
        "--basal_alp_min",
        type=float,
        default=0.1,
        help="Min basal cellular ALP (mU/mL)",
    )
    parser.add_argument(
        "--basal_alp_max",
        type=float,
        default=1.0,
        help="Max basal cellular ALP (mU/mL)",
    )

    parser.add_argument(
        "--osteo_alp_min",
        type=float,
        default=0.5,
        help="Min osteogenic cellular ALP (mU/mL)",
    )
    parser.add_argument(
        "--osteo_alp_max",
        type=float,
        default=3.0,
        help="Max osteogenic cellular ALP (mU/mL)",
    )

    parser.add_argument(
        "--serum_alp_min",
        type=float,
        default=0.0,
        help="Min serum ALP baseline (mU/mL)",
    )
    parser.add_argument(
        "--serum_alp_max",
        type=float,
        default=0.2,
        help="Max serum ALP baseline (mU/mL)",
    )

    parser.add_argument(
        "--bgp_min",
        type=float,
        default=5.0,
        help="Min β-GP concentration per medium change (mM)",
    )
    parser.add_argument(
        "--bgp_max",
        type=float,
        default=15.0,
        help="Max β-GP concentration per medium change (mM)",
    )

    parser.add_argument(
        "--media_min",
        type=float,
        default=1.0,
        help="Min medium change interval (days)",
    )
    parser.add_argument(
        "--media_max",
        type=float,
        default=3.0,
        help="Max medium change interval (days)",
    )

    parser.add_argument(
        "--start_day_min",
        type=float,
        default=2.0,
        help="Min ALP ramp start day",
    )
    parser.add_argument(
        "--start_day_max",
        type=float,
        default=4.0,
        help="Max ALP ramp start day",
    )

    parser.add_argument(
        "--measure_day_min",
        type=float,
        default=7.0,
        help="Min ALP ramp end / measurement day",
    )
    parser.add_argument(
        "--measure_day_max",
        type=float,
        default=14.0,
        help="Max ALP ramp end / measurement day",
    )

    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    eval_days = sorted(set(int(d) for d in args.eval_days))
    print(f"[*] Evaluating Pi_cum at days: {eval_days}")

    # enforce power-of-two N
    N = 2 ** int(np.ceil(np.log2(args.N)))
    print(f"[*] Using Sobol sampler with N = {N}")

    problem = {
        "num_vars": 7,
        "names": [
            "basal_alp",
            "osteo_alp",
            "serum_alp",
            "bgp_mM",
            "media_interval_days",
            "start_day",
            "measure_day",
        ],
        "bounds": [
            [args.basal_alp_min, args.basal_alp_max],
            [args.osteo_alp_min, args.osteo_alp_max],
            [args.serum_alp_min, args.serum_alp_max],
            [args.bgp_min, args.bgp_max],
            [args.media_min, args.media_max],
            [args.start_day_min, args.start_day_max],
            [args.measure_day_min, args.measure_day_max],
        ],
    }

    # Sobol sampling
    X = sobol_sampler.sample(problem, N, calc_second_order=True)
    print(f"[*] Total model evaluations: {len(X)}")

    n_samples = len(X)
    n_days = len(eval_days)
    Y = np.zeros((n_samples, n_days))

    # Model runs
    for i, row in enumerate(X):
        (
            basal_alp,
            osteo_alp,
            serum_alp,
            bgp_mM,
            media_interval_days,
            start_day,
            measure_day,
        ) = row

        Y[i, :] = compute_pi_cumulative_multiday(
            basal_alp=basal_alp,
            osteo_alp=osteo_alp,
            serum_alp=serum_alp,
            bgp_mM=bgp_mM,
            media_interval_days=media_interval_days,
            start_day=start_day,
            measure_day=measure_day,
            days_total=args.days_total,
            eval_days=eval_days,
        )

    # Sobol analysis per day
    df_by_day = {}
    for j, day in enumerate(eval_days):
        print(f"[*] Sobol analysis for Pi_cum at day {day} ...")
        Si = sobol.analyze(problem, Y[:, j], calc_second_order=True)

        df = pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            }
        )
        df_by_day[day] = df
        df.to_csv(out / f"alppi_sobol_Pi_cum_day{day}.csv", index=False)

    # Save samples + all outputs
    df_samples = pd.DataFrame(X, columns=problem["names"])
    for j, day in enumerate(eval_days):
        df_samples[f"Pi_cum_day{day}"] = Y[:, j]
    df_samples.to_csv(out / "alppi_sensitivity_samples.csv", index=False)

    # Multiday plot
    plot_sobol_multiday(
        df_by_day,
        eval_days,
        out / "alppi_sobol_bars_multiday.png",
        out / "alppi_sobol_bars_multiday.pdf",
        "β-GP / ALP → Pi module: Sobol sensitivity over time",
    )

    # Small auto-generated report
    last_day = eval_days[-1]
    df_last = df_by_day[last_day]
    dom_row = df_last.loc[df_last["ST"].idxmax()]

    text = f"""
Multiday Sobol global sensitivity: β-GP / ALP → Pi module
--------------------------------------------------------

Parameter ranges:
- basal_alp: {args.basal_alp_min}–{args.basal_alp_max} mU/mL
- osteo_alp: {args.osteo_alp_min}–{args.osteo_alp_max} mU/mL
- serum_alp: {args.serum_alp_min}–{args.serum_alp_max} mU/mL
- bgp_mM: {args.bgp_min}–{args.bgp_max} mM
- media_interval_days: {args.media_min}–{args.media_max} days
- start_day: {args.start_day_min}–{args.start_day_max}
- measure_day: {args.measure_day_min}–{args.measure_day_max}

Sobol settings:
- Base N = {args.N}, used N = {N} (power of 2)
- Total evaluations = {len(X)}
- Pi_cum evaluated at days {eval_days}

Example summary (day {last_day}):
- Dominant parameter (by total-effect ST): {dom_row['parameter']} (ST = {dom_row['ST']:.3f})

Across days {eval_days[0]}→{last_day}, the relative importance of ALP
levels, β-GP availability, and medium-change schedule can be read from
alppi_sobol_bars_multiday.pdf. Parameters with ST ≈ 0 have negligible
influence on cumulative Pi in the tested range.
"""

    with open(out / "alppi_sobol_results_multiday.txt", "w") as f:
        f.write(text)

    print("\n[✓] Finished.")
    print(f"Results, multiday plots, and text saved in: {out.resolve()}")


if __name__ == "__main__":
    main()
