#!/usr/bin/env python3
"""
β-GP → Pi model with linear ALP(t)

Inputs (per mL well):
- basal_alp:  cellular ALP activity in basal medium (mU/mL)
- osteo_alp:  cellular ALP activity in osteogenic medium (mU/mL)
- serum_alp:  ALP activity contributed by serum/FBS (mU/mL, constant baseline)
- bgp_mM:     β-glycerophosphate concentration in medium (mM)
- start_day:  day when osteogenic supplements (AA + β-GP + Dex) start (default 3)
- measure_day: day where osteo_alp was measured (e.g. 10)
- media:      medium change interval in days (default 2)
- days:       total culture duration (default 28)

ALP(t) model (per hour):
- For t < start_day: ALP_total = serum_alp + basal_alp
- From start_day → measure_day: cellular ALP rises linearly
  from basal_alp to osteo_alp; serum_alp is added as constant baseline.
- After measure_day: ALP_total = serum_alp + osteo_alp (plateau).

β-GP → Pi conversion:
- 1 mU = 1 nmol/min = 60 µM/h = 0.06 mM/h (1 mL well).
- Pi production rate (mM/h) = ALP_total(t) * 0.06.
- Pi production is capped by available β-GP (1:1 stoichiometry).
- Medium changes reset β-GP to bgp_mM but Pi is cumulative.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------
# ALP(t): linear cellular rise + constant serum ALP
# ---------------------------------------------------------------------


def alp_linear_profile(
    basal_cell_alp: float,
    osteo_cell_alp: float,
    serum_alp: float,
    start_day: int,
    measure_day: int,
    total_days: int,
) -> np.ndarray:
    """
    Construct hourly ALP(t) in mU/mL.

    Cellular ALP:
      - basal_cell_alp before start_day
      - linear rise to osteo_cell_alp between start_day and measure_day
      - plateau at osteo_cell_alp after measure_day

    Total ALP = serum_alp (constant) + cellular ALP(t).
    """
    total_hours = total_days * 24
    alp_hours = np.zeros(total_hours + 1)

    for t in range(total_hours + 1):
        day = t / 24.0

        if day < start_day:
            alp_cell = basal_cell_alp
        elif day <= measure_day:
            frac = (day - start_day) / (measure_day - start_day)
            frac = max(0.0, min(1.0, frac))
            alp_cell = basal_cell_alp + frac * (osteo_cell_alp - basal_cell_alp)
        else:
            alp_cell = osteo_cell_alp

        alp_hours[t] = serum_alp + alp_cell

    return alp_hours


# ---------------------------------------------------------------------
# β-GP → Pi simulation
# ---------------------------------------------------------------------


def simulate_bgp_to_pi(
    bgp_mM: float,
    alp_hours: np.ndarray,
    total_days: int,
    media_interval_days: int,
    start_day: int,
) -> tuple:
    """
    Simulate β-GP → Pi, driven by ALP(t).

    Parameters
    ----------
    bgp_mM : float
        Initial β-GP concentration per medium change (mM).
    alp_hours : np.ndarray
        Hourly ALP(t) in mU/mL.
    total_days : int
    media_interval_days : int
        Medium change interval (days).
    start_day : int
        Day supplements (β-GP) start.

    Returns
    -------
    days : np.ndarray
    pi_daily_mM : np.ndarray
    pi_cum_mM : np.ndarray
    bgp_daily_mM : np.ndarray
    """
    total_hours = total_days * 24

    bgp_mM_hours = np.zeros(total_hours + 1)
    pi_mM_hours = np.zeros(total_hours + 1)

    start_h = start_day * 24
    bgp_now = 0.0
    pi_now = 0.0

    if start_h <= total_hours:
        bgp_now = bgp_mM
        bgp_mM_hours[start_h] = bgp_now
        pi_mM_hours[start_h] = pi_now

    for t in range(start_h + 1, total_hours + 1):
        # Medium change: reset β-GP, Pi stays
        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            bgp_now = bgp_mM

        alp_t = alp_hours[t]  # mU/mL

        # Pi production rate: 1 mU = 0.06 mM/h (1 mL well)
        ideal_dPi_mM = alp_t * 0.06

        # Cap by available β-GP
        dPi_mM = min(ideal_dPi_mM, bgp_now)

        bgp_now -= dPi_mM
        pi_now += dPi_mM

        bgp_mM_hours[t] = bgp_now
        pi_mM_hours[t] = pi_now

    # Aggregate daily
    days = np.arange(1, total_days + 1)
    pi_daily = np.zeros(total_days)
    pi_cum = np.zeros(total_days)
    bgp_daily = np.zeros(total_days)

    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        pi_daily[d] = max(0.0, pi_mM_hours[e] - pi_mM_hours[s])
        pi_cum[d] = pi_mM_hours[e]
        bgp_daily[d] = np.mean(bgp_mM_hours[s:e])

    return days, pi_daily, pi_cum, bgp_daily


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------


def plot_results(
    days: np.ndarray,
    alp_hours: np.ndarray,
    pi_daily: np.ndarray,
    pi_cum: np.ndarray,
    bgp_daily: np.ndarray,
    outdir: Path,
) -> None:
    plt.style.use("seaborn-v0_8-white")

    alp_daily = np.array(
        [np.mean(alp_hours[d * 24 : (d + 1) * 24]) for d in range(len(days))]
    )

    fig, axes = plt.subplots(3, 1, figsize=(8, 12), sharex=True)

    # ALP
    ax0 = axes[0]
    ax0.plot(days, alp_daily, linewidth=2.3)
    ax0.set_ylabel("ALP (mU/mL)", fontsize=12)
    ax0.set_title("Total ALP activity (serum + cellular, linear rise)", fontsize=14)
    ax0.grid(True, alpha=0.3)

    # Pi
    ax1 = axes[1]
    ax1.plot(days, pi_cum, linewidth=2.3, label="Cumulative Pi (mM)")
    ax1.plot(days, pi_daily, linewidth=1.8, label="Daily Pi (mM/day)")
    ax1.set_ylabel("Pi (mM)", fontsize=12)
    ax1.set_title("Pi production from β-GP via ALP", fontsize=14)
    ax1.legend(frameon=False)
    ax1.grid(True, alpha=0.3)

    # β-GP
    ax2 = axes[2]
    ax2.plot(days, bgp_daily, linewidth=2.3)
    ax2.set_xlabel("Day", fontsize=12)
    ax2.set_ylabel("β-GP (mM, daily mean)", fontsize=12)
    ax2.set_title("β-GP availability in medium", fontsize=14)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    pdf_path = outdir / "alp_linear_bgp_pi_plots.pdf"
    png_path = outdir / "alp_linear_bgp_pi_plots.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close()

    print(f"[✓] Plots saved:\n    {pdf_path}\n    {png_path}")


# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="Linear ALP(t) β-GP → Pi model")

    parser.add_argument(
        "--basal_alp", type=float, required=True, help="Basal cellular ALP (mU/mL)"
    )
    parser.add_argument(
        "--osteo_alp", type=float, required=True, help="Osteogenic cellular ALP (mU/mL)"
    )
    parser.add_argument(
        "--serum_alp",
        type=float,
        default=0.0,
        help="Serum/FBS ALP baseline (mU/mL, constant)",
    )
    parser.add_argument(
        "--bgp_mM",
        type=float,
        default=10.0,
        help="β-GP concentration per medium change (mM)",
    )
    parser.add_argument(
        "--start_day",
        type=int,
        default=3,
        help="Day osteogenic supplements (β-GP) start",
    )
    parser.add_argument(
        "--measure_day",
        type=int,
        default=10,
        help="Day osteogenic ALP endpoint measured",
    )
    parser.add_argument(
        "--media", type=int, default=2, help="Medium change interval (days)"
    )
    parser.add_argument("--days", type=int, default=28, help="Total simulation days")
    parser.add_argument(
        "--outfile",
        type=str,
        default="alp_linear_output.xlsx",
        help="Excel output filename",
    )

    args = parser.parse_args()

    outdir = Path("output_alp_linear")
    outdir.mkdir(exist_ok=True)

    # ALP(t)
    alp_hours = alp_linear_profile(
        basal_cell_alp=args.basal_alp,
        osteo_cell_alp=args.osteo_alp,
        serum_alp=args.serum_alp,
        start_day=args.start_day,
        measure_day=args.measure_day,
        total_days=args.days,
    )

    # β-GP → Pi
    days, pi_daily, pi_cum, bgp_daily = simulate_bgp_to_pi(
        bgp_mM=args.bgp_mM,
        alp_hours=alp_hours,
        total_days=args.days,
        media_interval_days=args.media,
        start_day=args.start_day,
    )

    # Excel output
    df = pd.DataFrame(
        {
            "Day": days,
            "ALP_daily_mU_per_mL": [
                np.mean(alp_hours[d * 24 : (d + 1) * 24]) for d in range(args.days)
            ],
            "BGP_mM_daily": bgp_daily,
            "Pi_mM_daily": pi_daily,
            "Pi_mM_cumulative": pi_cum,
        }
    )

    xlsx_path = outdir / args.outfile
    df.to_excel(xlsx_path, index=False)
    print(f"[✓] Excel saved: {xlsx_path}")

    # Plots
    plot_results(days, alp_hours, pi_daily, pi_cum, bgp_daily, outdir)

    print("\n[✓] Linear ALP → Pi simulation complete.\n")


if __name__ == "__main__":
    main()
