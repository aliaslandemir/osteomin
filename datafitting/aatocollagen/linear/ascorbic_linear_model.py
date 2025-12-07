#!/usr/bin/env python3
"""
AA → Collagen accumulation model (OsteoMin, linear version)

- AA in α-MEM + FBS at 37 °C with half-life ~3 h
- Osteogenic supplements added at SUPP_DAY (default: 3)
- Medium changed every N days (default: 2)
- Collagen IF measured at --measure_day (e.g. 14)

This script:
- simulates AA decay + pulses,
- converts that into a daily collagen rate (linear in normalized AA),
- accumulates collagen,
- fits R0 and R_AA to basal + osteo IF at measure_day,
- saves:
    * XLSX table,
    * time-course figure,
    * fit-comparison figure,
    * markdown summary.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUPP_DAY = 3  # day osteogenic medium starts (AA + βGP)


# =====================================================================
# AA DECAY SIMULATION
# =====================================================================


def simulate_aa(
    AA0: float,
    half_life_h: float,
    media_interval_days: int,
    total_days: int,
    supplement_start_day: int = SUPP_DAY,
):
    """
    Simulate AA concentration with exponential decay and medium changes.

    Returns
    -------
    AA_daily_24h : (days,) mean AA over each 24 h day
    AA_eff_12h   : (days,) mean AA over first 12 h (effective window)
    AA_norm      : (days,) normalized AA_eff_12h (0–1)
    """
    k = np.log(2.0) / float(half_life_h)
    total_hours = total_days * 24

    AA_hours = np.zeros(total_hours + 1, dtype=float)

    start_h = supplement_start_day * 24
    if start_h <= total_hours:
        AA_hours[start_h] = AA0

    for t in range(start_h + 1, total_hours + 1):
        AA_hours[t] = AA_hours[t - 1] * np.exp(-k)
        # medium change
        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            AA_hours[t] = AA0

    AA_daily_24h = np.zeros(total_days, dtype=float)
    AA_eff_12h = np.zeros(total_days, dtype=float)

    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        mid = s + 12
        AA_daily_24h[d] = AA_hours[s:e].mean()
        AA_eff_12h[d] = AA_hours[s:mid].mean()

    max_eff = AA_eff_12h.max()
    if max_eff > 0:
        AA_norm = AA_eff_12h / max_eff
    else:
        AA_norm = np.zeros_like(AA_eff_12h)

    return AA_daily_24h, AA_eff_12h, AA_norm


# =====================================================================
# COLLAGEN MODEL (LINEAR)
# =====================================================================


def simulate_collagen(
    days: int,
    R0: float,
    R_AA: float,
    AA_norm: np.ndarray,
):
    """
    Linear model:
        daily[d] = R0 + R_AA * AA_norm[d]
        total[d] = sum_{i<=d} daily[i]
    """
    daily = np.zeros(days, dtype=float)
    total = np.zeros(days, dtype=float)

    for d in range(days):
        rate = R0 + R_AA * AA_norm[d]
        daily[d] = rate
        total[d] = rate if d == 0 else total[d - 1] + rate

    return daily, total


# =====================================================================
# FITTING
# =====================================================================


def fit_parameters(
    basal_if: float,
    osteo_if: float,
    AA_norm: np.ndarray,
    measure_day: int,
):
    """
    Fit R0, R_AA given:
      basal_if  = collagen at measure_day in basal
      osteo_if  = collagen at measure_day in osteo
      AA_norm   = normalized AA_eff_12h (0–1)
      measure_day = integer day (>0)

    Assumptions:
      basal:  collagen_bas(d)  = R0 * d
      osteo:  collagen_ost(d)  = R0 * d + R_AA * sum(AA_norm[1..d])
    """
    D = measure_day
    R0 = basal_if / D

    S = np.sum(AA_norm[:D])  # sum over days 1..D (index 0..D-1)
    if S == 0:
        R_AA = 0.0
    else:
        R_AA = (osteo_if - basal_if) / S

    return R0, R_AA


# =====================================================================
# PLOTTING
# =====================================================================


def plot_timecourse(
    days: int,
    measure_day: int,
    AA_eff_12h: np.ndarray,
    collagen_daily: np.ndarray,
    collagen_total: np.ndarray,
    osteo_if: float,
    outdir: Path,
):
    """
    2-panel figure:
      top: AA_eff_12h vs day
      bottom: collagen_total vs day,
              + linear line to experimental endpoint,
              + markers for model & experimental endpoints.
    """
    plt.style.use("seaborn-v0_8-white")
    x = np.arange(1, days + 1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 9), sharex=True)

    # --- AA panel ---
    ax = axes[0]
    ax.plot(x, AA_eff_12h, linewidth=2.0)
    ax.set_ylabel("Effective AA (µg/mL)\n(first 12 h)")
    ax.set_title("Ascorbic acid exposure (α-MEM, 37 °C)")
    ax.grid(True, alpha=0.3)

    ax.axvline(SUPP_DAY, linestyle="--", linewidth=1.2, color="black", alpha=0.4)
    ax.text(
        SUPP_DAY + 0.05,
        ax.get_ylim()[1] * 0.9,
        "Supplements on",
        fontsize=9,
        va="top",
        ha="left",
    )

    # --- Collagen panel ---
    ax2 = axes[1]
    ax2.plot(x, collagen_total, label="Model (osteo, total)", linewidth=2.2)

    # linear phenom. line from 0 to experimental endpoint
    D = measure_day
    if D <= days:
        x_lin = np.array([0, D], dtype=float)
        y_lin = np.array([0.0, osteo_if], dtype=float)
        ax2.plot(x_lin, y_lin, "--", linewidth=1.5, label="Linear (exp endpoint)")

        # markers at endpoint
        ax2.plot(D, collagen_total[D - 1], "o", ms=7, label="Model at measure day")
        ax2.plot(D, osteo_if, "x", ms=8, mew=2, label="Experimental endpoint")

    ax2.set_xlabel("Culture day")
    ax2.set_ylabel("Collagen units")
    ax2.set_title("Collagen production driven by AA pulses")
    ax2.grid(True, alpha=0.3)
    ax2.legend(frameon=False, fontsize=9)

    plt.tight_layout()
    pdf_path = outdir / "timecourse.pdf"
    png_path = outdir / "timecourse.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)


def plot_fit_comparison(
    days: int,
    measure_day: int,
    basal_if: float,
    osteo_if: float,
    R0: float,
    R_AA: float,
    AA_norm: np.ndarray,
    collagen_total: np.ndarray,
    outdir: Path,
):
    """
    2-panel figure:
      left : model osteo collagen vs time + endpoint marker
      right: basal & osteo, experimental vs model, with connecting lines
             and a quantitative text box (ratios, error, R0, R_AA).
    """
    plt.style.use("seaborn-v0_8-white")
    x = np.arange(1, days + 1)
    D = min(measure_day, days)

    model_basal = R0 * D
    model_osteo = collagen_total[D - 1]

    ratio_exp = osteo_if / basal_if if basal_if != 0 else np.nan
    ratio_model = model_osteo / model_basal if model_basal != 0 else np.nan
    rel_err_osteo = (
        (model_osteo - osteo_if) / osteo_if * 100.0 if osteo_if != 0 else np.nan
    )

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # --- left: timecourse again, zoom on fit ---
    ax = axes[0]
    ax.plot(x, collagen_total, linewidth=2.2, label="Model (osteo, total)")
    ax.axvline(D, linestyle="--", linewidth=1.0, color="gray", alpha=0.7)
    ax.plot(D, model_osteo, "o", ms=7, label=f"Model day {D}")
    ax.plot(D, osteo_if, "x", ms=8, mew=2, label="Exp day {D}")

    ax.set_xlabel("Culture day")
    ax.set_ylabel("Collagen units")
    ax.set_title("Modelled osteo collagen vs time")
    ax.grid(True, alpha=0.3)
    ax.legend(frameon=False, fontsize=8)

    # --- right: basal vs osteo, exp vs model ---
    ax2 = axes[1]
    conditions = ["Basal", "Osteo"]
    xpos = np.arange(len(conditions))

    exp_vals = [basal_if, osteo_if]
    model_vals = [model_basal, model_osteo]

    # lines + markers
    ax2.plot(xpos, exp_vals, "o-", label="Experimental")
    ax2.plot(xpos, model_vals, "s--", label="Model")

    # numeric labels
    for i, (e, m) in enumerate(zip(exp_vals, model_vals)):
        ax2.text(xpos[i] - 0.05, e, f"{e:.2f}", ha="right", va="bottom", fontsize=8)
        ax2.text(xpos[i] + 0.05, m, f"{m:.2f}", ha="left", va="bottom", fontsize=8)

    ax2.set_xticks(xpos)
    ax2.set_xticklabels(conditions)
    ax2.set_ylabel(f"Collagen (IF/DAPI) at day {D}")
    ax2.set_title("Experimental vs model at measurement day")
    ax2.grid(True, axis="y", alpha=0.3)
    ax2.legend(frameon=False, fontsize=8)

    # quantitative text box – same info as printed
    text = (
        f"R0 (basal/day)   = {R0:.4f}\n"
        f"R_AA (AA/day)    = {R_AA:.4f}\n"
        f"Exp ratio O/B    = {ratio_exp:.3f}\n"
        f"Model ratio O/B  = {ratio_model:.3f}\n"
        f"Osteo error      = {rel_err_osteo:.1f}%"
    )
    ax2.text(
        0.5,
        0.05,
        text,
        transform=ax2.transAxes,
        ha="center",
        va="bottom",
        fontsize=8,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.9),
    )

    plt.tight_layout()
    pdf_path = outdir / "fit_comparison.pdf"
    png_path = outdir / "fit_comparison.png"
    plt.savefig(pdf_path, dpi=300)
    plt.savefig(png_path, dpi=300)
    plt.close(fig)


# =====================================================================
# MARKDOWN REPORT
# =====================================================================


def write_markdown(
    outdir: Path,
    args: argparse.Namespace,
    measure_day: int,
    R0: float,
    R_AA: float,
    AA_daily_24h: np.ndarray,
    AA_eff_12h: np.ndarray,
    AA_norm: np.ndarray,
    collagen_daily: np.ndarray,
    collagen_total: np.ndarray,
):
    D = measure_day
    model_basal = R0 * D
    model_osteo = collagen_total[D - 1]

    ratio_exp = args.osteo_if / args.basal_if if args.basal_if != 0 else np.nan
    ratio_model = model_osteo / model_basal if model_basal != 0 else np.nan
    rel_err_osteo = (
        (model_osteo - args.osteo_if) / args.osteo_if * 100.0
        if args.osteo_if != 0
        else np.nan
    )

    md_lines = []

    md_lines.append("# AA → Collagen Model Summary\n")

    md_lines.append("## Input Parameters")
    md_lines.append(f"- Model type: linear")
    md_lines.append(f"- Initial AA concentration: {args.conc} µg/mL")
    md_lines.append(f"- AA half-life: {args.halflife} h")
    md_lines.append(f"- Medium interval: {args.media} days")
    md_lines.append(f"- Basal IF (day {D}): {args.basal_if}")
    md_lines.append(f"- Osteo IF (day {D}): {args.osteo_if}")
    md_lines.append(f"- Measurement day: {D}")
    md_lines.append(f"- Supplements added at day: {SUPP_DAY}")
    md_lines.append(f"- Total simulation days: {args.days}\n")

    md_lines.append("## Fitted Parameters")
    md_lines.append(f"- R0 (basal collagen/day): {R0:.4f}")
    md_lines.append(f"- R_AA (AA-driven collagen/day): {R_AA:.4f}\n")

    md_lines.append("## AA Exposure (Diagnostic)")
    md_lines.append(f"- Max 24 h mean AA: {AA_daily_24h.max():.4f} µg/mL")
    md_lines.append(f"- Max effective AA (first 12 h): {AA_eff_12h.max():.4f} µg/mL")
    md_lines.append(f"- Max AA effect (normalized): {AA_norm.max():.4f}\n")

    md_lines.append("## Collagen Outputs")
    md_lines.append(f"- Total collagen at day {args.days}: {collagen_total[-1]:.4f}")
    md_lines.append(f"- Daily collagen on last day: {collagen_daily[-1]:.4f}\n")

    md_lines.append("## Fit Summary at Measurement Day")
    md_lines.append(f"- Basal (exp vs model): {args.basal_if:.3f} vs {model_basal:.3f}")
    md_lines.append(f"- Osteo (exp vs model): {args.osteo_if:.3f} vs {model_osteo:.3f}")
    md_lines.append(f"- Exp ratio (Osteo/Basal): {ratio_exp:.3f}")
    md_lines.append(f"- Model ratio (Osteo/Basal): {ratio_model:.3f}")
    md_lines.append(f"- Osteo relative error (model vs exp): {rel_err_osteo:.1f}%\n")

    md_lines.append("## Files Generated")
    md_lines.append("- Excel: `aa_collagen_output.xlsx`")
    md_lines.append("- Time-course: `timecourse.pdf/png`")
    md_lines.append("- Fit comparison: `fit_comparison.pdf/png`")

    md_path = outdir / "results.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")


# =====================================================================
# MAIN
# =====================================================================


def main():
    parser = argparse.ArgumentParser(description="AA→Collagen linear model")

    parser.add_argument(
        "--conc", type=float, required=True, help="Initial AA concentration (µg/mL)"
    )
    parser.add_argument(
        "--halflife", type=float, default=3.0, help="AA half-life (hours), default 3"
    )
    parser.add_argument(
        "--media", type=int, default=2, help="Medium change interval (days)"
    )
    parser.add_argument(
        "--days", type=int, default=28, help="Total culture duration (days)"
    )
    parser.add_argument(
        "--measure_day",
        type=int,
        default=14,
        help="Day of collagen IF/DAPI measurement",
    )

    parser.add_argument(
        "--basal_if",
        type=float,
        required=True,
        help="Basal collagen IF/DAPI at measure_day",
    )
    parser.add_argument(
        "--osteo_if",
        type=float,
        required=True,
        help="Osteogenic collagen IF/DAPI at measure_day",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default="aa_collagen_output.xlsx",
        help="Name of XLSX output file",
    )

    args = parser.parse_args()
    D = args.measure_day

    outdir = Path("output_linear")
    outdir.mkdir(exist_ok=True)

    # Echo inputs
    print("\nInput values:")
    print(f"  AA conc         : {args.conc} µg/mL")
    print(f"  AA half-life    : {args.halflife} h")
    print(f"  Medium interval : {args.media} days")
    print(f"  Days simulated  : {args.days}")
    print(f"  Measure day     : {D}")
    print(f"  Basal IF (d{D}) : {args.basal_if}")
    print(f"  Osteo IF (d{D}) : {args.osteo_if}\n")

    # AA simulation
    AA_daily_24h, AA_eff_12h, AA_norm = simulate_aa(
        AA0=args.conc,
        half_life_h=args.halflife,
        media_interval_days=args.media,
        total_days=args.days,
        supplement_start_day=SUPP_DAY,
    )

    # Fit parameters
    R0, R_AA = fit_parameters(
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        AA_norm=AA_norm,
        measure_day=D,
    )

    print("Fitted parameters:")
    print(f"  R0   = {R0:.4f}  (basal collagen/day)")
    print(f"  R_AA = {R_AA:.4f}  (AA-driven collagen/day)\n")

    # Collagen simulation
    collagen_daily, collagen_total = simulate_collagen(
        days=args.days,
        R0=R0,
        R_AA=R_AA,
        AA_norm=AA_norm,
    )

    # Fit summary console
    model_basal = R0 * D
    model_osteo = collagen_total[D - 1]
    ratio_exp = args.osteo_if / args.basal_if if args.basal_if != 0 else np.nan
    ratio_model = model_osteo / model_basal if model_basal != 0 else np.nan
    rel_err_osteo = (
        (model_osteo - args.osteo_if) / args.osteo_if * 100.0
        if args.osteo_if != 0
        else np.nan
    )

    print(f"Fit summary at day {D}")
    print(f"  Basal  exp/model : {args.basal_if:.3f} / {model_basal:.3f}")
    print(f"  Osteo  exp/model : {args.osteo_if:.3f} / {model_osteo:.3f}")
    print(f"  Exp ratio O/B    : {ratio_exp:.3f}")
    print(f"  Model ratio O/B  : {ratio_model:.3f}")
    print(f"  Osteo error      : {rel_err_osteo:.1f}%\n")

    # Save XLSX
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

    # Figures
    plot_timecourse(
        days=args.days,
        measure_day=D,
        AA_eff_12h=AA_eff_12h,
        collagen_daily=collagen_daily,
        collagen_total=collagen_total,
        osteo_if=args.osteo_if,
        outdir=outdir,
    )
    plot_fit_comparison(
        days=args.days,
        measure_day=D,
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        R0=R0,
        R_AA=R_AA,
        AA_norm=AA_norm,
        collagen_total=collagen_total,
        outdir=outdir,
    )
    print("[✓] Figures saved")

    # Markdown
    write_markdown(
        outdir=outdir,
        args=args,
        measure_day=D,
        R0=R0,
        R_AA=R_AA,
        AA_daily_24h=AA_daily_24h,
        AA_eff_12h=AA_eff_12h,
        AA_norm=AA_norm,
        collagen_daily=collagen_daily,
        collagen_total=collagen_total,
    )
    print("[✓] Markdown report saved\n[✓] Simulation complete.\n")


if __name__ == "__main__":
    main()


# To Run

# python ascorbic_linear_model.py   --conc 50   --basal_if 2.5   --osteo_if 3.6   --halflife 3   --media 2   --days 14   --measure_day 14
