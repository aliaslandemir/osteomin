#!/usr/bin/env python3
"""
Michaelis–Menten AA → Collagen model (standalone)
-------------------------------------------------

- AA half-life ~3 h (α-MEM + FBS, 37°C)
- Supplements added at day 3
- Medium changed every N days (default 2)
- Collagen measured at day 14
- Km is fixed (literature)
- Rmax is fitted automatically

Biological basis:
- Roach 1985: rapid AA decay in medium (t1/2 = 1–3 h)
- Michels 2013: AA instability in culture media
- Franceschi 1994: AA-dependent collagen synthesis saturation
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt


# ======================================================================
# AA DECAY SIMULATION
# ======================================================================


def simulate_aa(
    AA0, half_life_h, media_interval_days, total_days, supplement_start_day=3
):

    k = np.log(2) / half_life_h
    total_hours = total_days * 24
    AA_hours = np.zeros(total_hours + 1)

    # begin dosing at day 3
    start_h = supplement_start_day * 24
    if start_h <= total_hours:
        AA_hours[start_h] = AA0

    for t in range(start_h + 1, total_hours + 1):
        AA_hours[t] = AA_hours[t - 1] * np.exp(-k)

        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            AA_hours[t] = AA0

    # daily metrics
    AA_eff_12h = np.zeros(total_days)
    AA_24h = np.zeros(total_days)

    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        mid = s + 12
        AA_24h[d] = np.mean(AA_hours[s:e])
        AA_eff_12h[d] = np.mean(AA_hours[s:mid])

    return AA_hours, AA_24h, AA_eff_12h


# ======================================================================
# MM COLLAGEN MODEL
# ======================================================================


def mm_rate(R0, Rmax, Km, AA_norm):
    return R0 + (Rmax * AA_norm) / (Km + AA_norm)


def simulate_collagen_mm(days, R0, Rmax, Km, AA_norm):
    daily = np.zeros(days)
    total = np.zeros(days)

    for d in range(days):
        rate = mm_rate(R0, Rmax, Km, AA_norm[d])
        daily[d] = rate
        total[d] = rate if d == 0 else total[d - 1] + rate

    return daily, total


# ======================================================================
# FITTING Rmax
# ======================================================================


def fit_Rmax(Km, R0, basal_IF, osteo_IF, measure_day=14):
    """
    Fit Rmax from:
    osteo_IF = measure_day * [ R0 + Rmax/(Km+1) ]
    """
    target_rate = osteo_IF / measure_day
    Rmax = (Km + 1) * (target_rate - R0)
    return max(Rmax, 0.0)


# ======================================================================
# PLOTTING
# ======================================================================


def plot_results(days, AA_eff_12h, collagen_daily, collagen_total, outdir):
    plt.style.use("seaborn-v0_8-white")

    x = np.arange(1, days + 1)
    fig, ax = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    ax[0].plot(x, AA_eff_12h, linewidth=2.2)
    ax[0].set_ylabel("Effective AA (µg/mL)\n(first 12 h)", fontsize=12)
    ax[0].set_title("AA Exposure (12 h window)", fontsize=14)
    ax[0].grid(True, alpha=0.3)

    ax[1].plot(x, collagen_total, linewidth=2.5, label="Total collagen")
    ax[1].plot(x, collagen_daily, linewidth=1.8, label="Daily collagen")
    ax[1].set_xlabel("Day", fontsize=12)
    ax[1].set_ylabel("Collagen units", fontsize=12)
    ax[1].set_title("Michaelis–Menten Collagen Production", fontsize=14)
    ax[1].legend(frameon=False)
    ax[1].grid(True, alpha=0.3)

    plt.tight_layout()
    outdir.mkdir(exist_ok=True)
    plt.savefig(outdir / "mm_plots.pdf", dpi=300)
    plt.savefig(outdir / "mm_plots.png", dpi=300)
    plt.close(fig)


# ======================================================================
# MAIN
# ======================================================================


def main():
    parser = argparse.ArgumentParser(description="MM AA→Collagen Model")

    parser.add_argument("--conc", type=float, required=True)
    parser.add_argument("--halflife", type=float, default=3.0)
    parser.add_argument("--media", type=int, default=2)
    parser.add_argument("--days", type=int, default=28)

    parser.add_argument("--basal_if", type=float, required=True)
    parser.add_argument("--osteo_if", type=float, required=True)

    parser.add_argument(
        "--Km", type=float, default=0.5, help="Michaelis constant (normalized AA units)"
    )

    parser.add_argument("--outfile", type=str, default="mm_output.xlsx")

    args = parser.parse_args()

    outdir = Path("output_mm")
    outdir.mkdir(exist_ok=True)

    # -----------------------------------------------------
    # AA simulation
    # -----------------------------------------------------
    AA_hours, AA_24h, AA_eff_12h = simulate_aa(
        args.conc, args.halflife, args.media, args.days, supplement_start_day=3
    )

    AA_norm = (
        AA_eff_12h / np.max(AA_eff_12h)
        if np.max(AA_eff_12h) > 0
        else np.zeros_like(AA_eff_12h)
    )

    # -----------------------------------------------------
    # Fit R0 and Rmax
    # -----------------------------------------------------
    MEASURE_DAY = 14
    R0 = args.basal_if / MEASURE_DAY
    Rmax = fit_Rmax(args.Km, R0, args.basal_if, args.osteo_if, measure_day=MEASURE_DAY)

    print(f"\nFitted parameters:")
    print(f"  R0   = {R0:.4f}")
    print(f"  Rmax = {Rmax:.4f}  (for Km={args.Km})\n")

    # -----------------------------------------------------
    # Simulate collagen
    # -----------------------------------------------------
    collagen_daily, collagen_total = simulate_collagen_mm(
        args.days, R0, Rmax, args.Km, AA_norm
    )

    # -----------------------------------------------------
    # Save Excel
    # -----------------------------------------------------
    df = pd.DataFrame(
        {
            "Day": np.arange(1, args.days + 1),
            "AA_eff_12h": AA_eff_12h,
            "AA_norm": AA_norm,
            "Collagen_daily": collagen_daily,
            "Collagen_total": collagen_total,
        }
    )

    df.to_excel(outdir / args.outfile, index=False)
    print(f"[✓] Excel saved: {outdir/args.outfile}")

    # -----------------------------------------------------
    # Plots
    # -----------------------------------------------------
    plot_results(args.days, AA_eff_12h, collagen_daily, collagen_total, outdir)

    print("\n[✓] MM simulation complete.\n")


if __name__ == "__main__":
    main()
