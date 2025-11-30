#!/usr/bin/env python3
"""
AA → Collagen accumulation model (OsteoMin module)

- AA in α-MEM + FBS at 37 °C with half-life ~3 h
- Osteogenic supplements added at DAY 3
- Medium changed every N days (default: 2)
- Collagen IF measured at DAY 14 (fixed)

Biological assumptions based on:
- Roach et al., Biochim Biophys Acta 1985: fast AA decay in medium;
- Michels et al., Crit Rev Food Sci Nutr 2013: AA instability in culture media;
- Franceschi et al., J Bone Miner Res 1994: AA-dependent collagen synthesis in MC3T3-E1.

Effective AA for collagen is taken as the mean concentration
over the first 12 h (0.5 day) of each culture day. AA older
than 12 h is assumed not to contribute appreciably to new
collagen hydroxylation/synthesis.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# =====================================================================
# AA DECAY SIMULATION
# =====================================================================


def simulate_aa(
    AA0, half_life_h, media_interval_days, total_days, supplement_start_day=3
):
    """
    Simulate AA concentration over time with exponential decay and
    medium replenishment.

    Returns
    -------
    AA_hours : np.ndarray
        Hourly AA concentrations (length = total_days*24 + 1).
    AA_daily_24h : np.ndarray
        Mean AA over each 24 h day.
    AA_eff_12h : np.ndarray
        Mean AA over the first 12 h of each day (effective window).
    """
    k = np.log(2) / half_life_h
    total_hours = total_days * 24

    AA_hours = np.zeros(total_hours + 1)

    # Start AA dosing at supplement_start_day
    start_h = supplement_start_day * 24
    if start_h <= total_hours:
        AA_hours[start_h] = AA0

    for t in range(start_h + 1, total_hours + 1):
        # Exponential decay each hour
        AA_hours[t] = AA_hours[t - 1] * np.exp(-k * 1.0)

        # Medium change every media_interval_days after first dose
        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            AA_hours[t] = AA0

    # Daily averages
    AA_daily_24h = np.zeros(total_days)
    AA_eff_12h = np.zeros(total_days)

    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        mid = s + 12  # first 12 h

        AA_daily_24h[d] = np.mean(AA_hours[s:e])
        AA_eff_12h[d] = np.mean(AA_hours[s:mid])

    return AA_hours, AA_daily_24h, AA_eff_12h


# =====================================================================
# COLLAGEN MODELS
# =====================================================================


def linear_rate(R0, R_AA, AA_norm):
    """Linear response to normalized AA."""
    return R0 + R_AA * AA_norm


def mm_rate(R0, Rmax, Km, AA_norm):
    """Michaelis-Menten response to normalized AA."""
    return R0 + (Rmax * AA_norm) / (Km + AA_norm)


def simulate_collagen(days, model_type, R0, R_AA, AA_norm, Rmax=None, Km=1.0):
    """
    Simulate daily and cumulative collagen production.

    Parameters
    ----------
    AA_norm : np.ndarray
        Normalized effective AA (0–1) for each day, from 12 h window.
    """
    daily = np.zeros(days)
    total = np.zeros(days)

    for d in range(days):
        if model_type == "linear":
            rate = linear_rate(R0, R_AA, AA_norm[d])
        elif model_type == "mm":
            if Rmax is None:
                raise ValueError("Rmax must be provided for MM model.")
            rate = mm_rate(R0, Rmax, Km, AA_norm[d])
        else:
            raise ValueError("Unknown model type")

        daily[d] = rate
        total[d] = rate if d == 0 else total[d - 1] + rate

    return daily, total


# =====================================================================
# PLOTTING
# =====================================================================


def plot_results(
    model, days, AA_daily_24h, AA_eff_12h, collagen_daily, collagen_total, outdir
):
    """
    Create a two-panel scientific-style figure:
    - Top: effective AA (first 12 h) vs day
    - Bottom: collagen daily & cumulative vs day
    """
    plt.style.use("seaborn-v0_8-white")

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    # --- Effective AA (12 h window) ---
    ax = axes[0]
    x = np.arange(1, days + 1)

    ax.plot(x, AA_eff_12h, linewidth=2.3)
    ax.set_ylabel("Effective AA (µg/mL)\n(first 12 h)", fontsize=12)
    ax.set_title("Ascorbic Acid Exposure (α-MEM, 37 °C)", fontsize=14)
    ax.grid(True, alpha=0.3)

    # Show supplement onset at day 3
    ax.axvline(3, linestyle="--", linewidth=1.2, color="black", alpha=0.4)
    ax.text(
        3.05, ax.get_ylim()[1] * 0.9, "Supplements on", fontsize=9, va="top", ha="left"
    )

    # --- Collagen accumulation ---
    ax2 = axes[1]
    ax2.plot(x, collagen_total, label="Total collagen", linewidth=2.4)
    ax2.plot(x, collagen_daily, label="Daily collagen", linewidth=1.8)

    ax2.set_xlabel("Culture day", fontsize=12)
    ax2.set_ylabel("Collagen units", fontsize=12)
    ax2.set_title("Collagen Production Driven by AA Pulses", fontsize=14)
    ax2.legend(frameon=False)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    pdf_path = outdir / f"{model}_plots.pdf"
    png_path = outdir / f"{model}_plots.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)

    print(f"[✓] Saved figures:\n    {pdf_path}\n    {png_path}")
    plt.close(fig)


# =====================================================================
# MARKDOWN REPORT
# =====================================================================


def write_markdown_report(
    outdir,
    args,
    R0,
    R_AA,
    AA_daily_24h,
    AA_eff_12h,
    AA_norm,
    collagen_daily,
    collagen_total,
):
    """
    Write a markdown summary of inputs, fitted parameters, and
    key outputs.
    """
    md_path = outdir / "results.md"
    with open(md_path, "w") as f:
        f.write("# AA → Collagen Model Summary\n\n")

        f.write("## Input Parameters\n")
        f.write(f"- **Model type:** {args.model}\n")
        f.write(f"- **Initial AA concentration:** {args.conc} µg/mL\n")
        f.write(f"- **AA half-life:** {args.halflife} h\n")
        f.write(f"- **Medium interval:** {args.media} days\n")
        f.write(f"- **Basal IF:** {args.basal_if}\n")
        f.write(f"- **Osteo IF:** {args.osteo_if}\n")
        f.write(f"- **Measurement day:** 14 (fixed)\n")
        f.write(f"- **Supplements added at day:** 3\n")
        f.write(f"- **Total simulation days:** {args.days}\n\n")

        f.write("## Fitted Parameters\n")
        f.write(f"- **R0 (basal collagen/day):** {R0:.4f}\n")
        f.write(f"- **R_AA (extra collagen/day):** {R_AA:.4f}\n")
        if args.model == "mm":
            f.write(f"- **Rmax:** {args.Rmax}\n")
            f.write(f"- **Km:** {args.Km}\n")
        f.write("\n")

        f.write("## AA Exposure (Diagnostic)\n")
        f.write(f"- **Max 24 h mean AA:** {np.max(AA_daily_24h):.4f} µg/mL\n")
        f.write(
            f"- **Max effective AA (first 12 h):** " f"{np.max(AA_eff_12h):.4f} µg/mL\n"
        )
        f.write(f"- **Max AA effect (normalized):** {np.max(AA_norm):.4f}\n\n")

        f.write("## Collagen Outputs\n")
        f.write(
            f"- **Total collagen at day {args.days}:** " f"{collagen_total[-1]:.4f}\n"
        )
        f.write(f"- **Daily collagen on last day:** " f"{collagen_daily[-1]:.4f}\n\n")

        f.write("## Files Generated\n")
        f.write("- Excel sheet: `aa_collagen_output.xlsx`\n")
        f.write("- Plots: `*_plots.pdf` and `*_plots.png`\n")

    print(f"[✓] Markdown report saved: {md_path}")


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="AA→Collagen model (OsteoMin)")

    parser.add_argument(
        "--conc", type=float, required=True, help="Initial AA concentration (µg/mL)"
    )
    parser.add_argument(
        "--halflife", type=float, default=3.0, help="AA half-life (hours), default 3 h"
    )
    parser.add_argument(
        "--media", type=int, default=2, help="Medium change interval (days)"
    )
    parser.add_argument(
        "--days", type=int, default=28, help="Total culture duration (days)"
    )

    parser.add_argument(
        "--basal_if", type=float, required=True, help="Basal collagen IF/DAPI at day 14"
    )
    parser.add_argument(
        "--osteo_if",
        type=float,
        required=True,
        help="Osteogenic collagen IF/DAPI at day 14",
    )

    parser.add_argument(
        "--model",
        choices=["linear", "mm"],
        default="linear",
        help="Collagen response model",
    )
    parser.add_argument(
        "--Km", type=float, default=0.5, help="Km for MM model (in normalized AA units)"
    )
    parser.add_argument(
        "--Rmax", type=float, default=1.0, help="Max AA-driven collagen rate (MM model)"
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default="aa_collagen_output.xlsx",
        help="Name of XLSX output file",
    )

    args = parser.parse_args()

    # Output directory
    outdir = Path(f"output_{args.model}")
    outdir.mkdir(exist_ok=True)

    # ------------------------------------------------------------
    # AA SIMULATION
    # ------------------------------------------------------------
    AA_hours, AA_daily_24h, AA_eff_12h = simulate_aa(
        AA0=args.conc,
        half_life_h=args.halflife,
        media_interval_days=args.media,
        total_days=args.days,
        supplement_start_day=3,
    )

    # Normalize effective AA (avoid divide-by-zero)
    if np.max(AA_eff_12h) > 0:
        AA_norm = AA_eff_12h / np.max(AA_eff_12h)
    else:
        AA_norm = np.zeros_like(AA_eff_12h)

    # ------------------------------------------------------------
    # FIT R0 and R_AA from IF at day 14
    # ------------------------------------------------------------
    MEASURE_DAY = 14
    R0 = args.basal_if / MEASURE_DAY
    R_AA = (args.osteo_if - args.basal_if) / MEASURE_DAY

    print("\nFitted parameters:")
    print(f"  R0   = {R0:.4f}  (basal collagen/day)")
    print(f"  R_AA = {R_AA:.4f}  (AA-driven collagen/day)")

    # ------------------------------------------------------------
    # COLLAGEN SIMULATION
    # ------------------------------------------------------------
    collagen_daily, collagen_total = simulate_collagen(
        days=args.days,
        model_type=args.model,
        R0=R0,
        R_AA=R_AA,
        AA_norm=AA_norm,
        Rmax=args.Rmax,
        Km=args.Km,
    )

    # ------------------------------------------------------------
    # SAVE XLSX
    # ------------------------------------------------------------
    df = pd.DataFrame(
        {
            "Day": np.arange(1, args.days + 1),
            "AA_daily_24h": AA_daily_24h,
            "AA_eff_12h": AA_eff_12h,
            "AA_norm": AA_norm,
            "Collagen_daily": collagen_daily,
            "Collagen_total": collagen_total,
        }
    )

    xlsx_path = outdir / args.outfile
    df.to_excel(xlsx_path, index=False)
    print(f"[✓] Excel saved: {xlsx_path}")

    # ------------------------------------------------------------
    # PLOTS
    # ------------------------------------------------------------
    plot_results(
        model=args.model,
        days=args.days,
        AA_daily_24h=AA_daily_24h,
        AA_eff_12h=AA_eff_12h,
        collagen_daily=collagen_daily,
        collagen_total=collagen_total,
        outdir=outdir,
    )

    # ------------------------------------------------------------
    # MARKDOWN REPORT
    # ------------------------------------------------------------
    write_markdown_report(
        outdir=outdir,
        args=args,
        R0=R0,
        R_AA=R_AA,
        AA_daily_24h=AA_daily_24h,
        AA_eff_12h=AA_eff_12h,
        AA_norm=AA_norm,
        collagen_daily=collagen_daily,
        collagen_total=collagen_total,
    )

    print("\n[✓] Simulation complete.\n")


if __name__ == "__main__":
    main()
