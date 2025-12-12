#!/usr/bin/env python3
"""
AA → Collagen accumulation model (OsteoMin, linear version; minimal plotting)
UPDATED: uses a 24 h effective exposure window (daily mean) instead of 12 h.

What this script does
- Simulates AA decay + medium-change pulses starting at SUPP_DAY (default 3)
- Uses 24 h mean AA as the daily exposure signal, normalized to 0–1:
      AA_norm(d) = AA_daily_24h(d) / max(AA_daily_24h)
- Converts AA exposure into an additive collagen production rate:
      rate(d) = R0 + R_AA * AA_norm(d)
- Accumulates collagen over days
- Fits R0 and R_AA using ONLY your endpoints at --measure_day:
      BASAL (basal) endpoint -> R0
      AA (osteo) endpoint -> R_AA
- Saves:
    * XLSX table
    * ONE figure: BASAL vs AA collagen time-course + endpoint markers (PDF/PNG)
    * markdown summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUPP_DAY = 3  # day osteogenic medium starts (AA + βGP)


# =====================================================================
# AA DECAY SIMULATION (24 h window)
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
    AA_daily_24h : (days,) mean AA over each 24 h day  (this is the exposure driver)
    AA_norm      : (days,) normalized AA_daily_24h (0–1)
    """
    k = np.log(2.0) / float(half_life_h)
    total_hours = total_days * 24

    AA_hours = np.zeros(total_hours + 1, dtype=float)

    start_h = supplement_start_day * 24
    if start_h <= total_hours:
        AA_hours[start_h] = AA0

    for t in range(start_h + 1, total_hours + 1):
        AA_hours[t] = AA_hours[t - 1] * np.exp(-k)
        # medium change -> reset to AA0
        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            AA_hours[t] = AA0

    AA_daily_24h = np.zeros(total_days, dtype=float)
    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        AA_daily_24h[d] = AA_hours[s:e].mean()

    max_daily = float(AA_daily_24h.max())
    AA_norm = AA_daily_24h / max_daily if max_daily > 0 else np.zeros_like(AA_daily_24h)

    return AA_daily_24h, AA_norm


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
    Returns arrays length = days, representing days 1..days.
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
    Fit R0, R_AA given endpoints at day D = measure_day.

    Basal (BASAL):  collagen_BASAL(D) = R0 * D  ->  R0 = basal_if / D

    Osteo (AA):  collagen_AA(D) = R0 * D + R_AA * sum_{d=1..D} AA_norm[d]
                 (implemented with indices 0..D-1)

    So:
      R_AA = (osteo_if - basal_if) / sum(AA_norm[:D])
    """
    D = int(measure_day)
    if D <= 0:
        raise ValueError("measure_day must be >= 1")

    R0 = float(basal_if) / float(D)

    S = float(np.sum(AA_norm[:D]))
    R_AA = 0.0 if S == 0 else (float(osteo_if) - float(basal_if)) / S

    return R0, R_AA


# =====================================================================
# PLOTS
# =====================================================================
def plot_basal_vs_aa_minimal(
    days: int,
    measure_day: int,
    R0: float,
    collagen_total_osteo: np.ndarray,
    basal_if: float,
    osteo_if: float,
    outdir: Path,
):
    """
    Single figure:
    - BASAL (basal) model prediction over time: R0*d
    - AA (osteo) model prediction over time: collagen_total_osteo[d]
    - Shows ONLY the measured endpoints at measure_day as 'x' markers
    """
    D = int(measure_day)
    if D > days:
        raise ValueError("measure_day cannot exceed --days in this plotting setup.")

    # x in days 0..days
    x = np.arange(0, days + 1, dtype=float)

    # Model curves (start at 0)
    y_basal = R0 * x
    y_aa = np.zeros_like(x)
    y_aa[0] = 0.0
    y_aa[1:] = collagen_total_osteo  # for days 1..days

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    ax.plot(x, y_basal, lw=2.2, label="BASAL (Basal model)")
    ax.plot(x, y_aa, lw=2.2, label="AA (Osteo model)")

    ax.plot(
        [D],
        [basal_if],
        marker="x",
        ms=8,
        mew=2,
        linestyle="None",
        label="BASAL measured",
    )
    ax.plot(
        [D],
        [osteo_if],
        marker="x",
        ms=8,
        mew=2,
        linestyle="None",
        label="AA measured",
    )

    ax.axvline(D, lw=1.2, ls="--", alpha=0.6)
    ax.text(D + 0.2, ax.get_ylim()[1] * 0.95, f"day {D}", va="top", fontsize=9)

    ax.set_xlabel("Day")
    ax.set_ylabel("Collagen (IF/DAPI units)")
    ax.set_title(
        "AA effect on collagen: model-predicted time-course\n(constrained by one endpoint per condition)"
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=1)
    ax.margins(x=0.02)

    fig.tight_layout()

    pdf_path = outdir / "basal_vs_aa_minimal.pdf"
    png_path = outdir / "basal_vs_aa_minimal.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_aa_exposure_daily(
    days: int,
    supplement_start_day: int,
    measure_day: int,
    AA_daily_24h: np.ndarray,
    AA_norm: np.ndarray,
    outdir: Path,
):
    """
    Single figure with two stacked axes:
      (top) AA_daily_24h (µg/mL)
      (bottom) AA_norm_24h (0–1) used by the collagen model
    Marks: supplement start + measurement day.
    """
    D = int(measure_day)
    x = np.arange(1, days + 1, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(5.4, 4.2), sharex=True, gridspec_kw={"hspace": 0.15}
    )

    ax1.plot(x, AA_daily_24h, lw=2.2)
    ax1.set_ylabel("AA daily mean (µg/mL)")
    ax1.set_title("AA exposure to cells (24 h mean)")

    ax2.plot(x, AA_norm, lw=2.2)
    ax2.set_ylabel("AA_norm (0–1)")
    ax2.set_xlabel("Day")

    # markers
    for ax in (ax1, ax2):
        ax.axvline(supplement_start_day, lw=1.2, ls="--", alpha=0.6)
        ax.axvline(D, lw=1.2, ls="--", alpha=0.6)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(x=0.02)

    ax1.text(
        supplement_start_day + 0.2,
        ax1.get_ylim()[1] * 0.95,
        f"start day {supplement_start_day}",
        va="top",
        fontsize=9,
    )
    ax1.text(
        D + 0.2, ax1.get_ylim()[1] * 0.80, f"measured day {D}", va="top", fontsize=9
    )

    fig.tight_layout()

    pdf_path = outdir / "aa_exposure_daily.pdf"
    png_path = outdir / "aa_exposure_daily.png"
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
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
    AA_norm: np.ndarray,
    collagen_daily: np.ndarray,
    collagen_total: np.ndarray,
):
    D = int(measure_day)
    model_basal = R0 * D
    model_osteo = collagen_total[D - 1]

    ratio_exp = args.osteo_if / args.basal_if if args.basal_if != 0 else np.nan
    ratio_model = model_osteo / model_basal if model_basal != 0 else np.nan

    md = []
    md.append("# AA → Collagen (linear, 24 h exposure) — minimal report\n")

    md.append("## Inputs")
    md.append(f"- AA0: {args.conc} µg/mL")
    md.append(f"- AA half-life: {args.halflife} h")
    md.append(f"- Medium interval: {args.media} days")
    md.append(f"- Supplements start day: {SUPP_DAY}")
    md.append(f"- Simulated days: {args.days}")
    md.append(f"- Measurement day: {D}")
    md.append(f"- BASAL (basal) endpoint at day {D}: {args.basal_if}")
    md.append(f"- AA (osteo) endpoint at day {D}: {args.osteo_if}\n")

    md.append("## Fitted parameters (from endpoints)")
    md.append(f"- R0: {R0:.6g} (basal collagen/day)")
    md.append(f"- R_AA: {R_AA:.6g} (AA-driven collagen/day)\n")

    md.append("## Diagnostics (24 h)")
    md.append(f"- max AA_daily_24h: {AA_daily_24h.max():.6g} µg/mL")
    md.append(f"- sum AA_norm[1..D]: {np.sum(AA_norm[:D]):.6g}\n")

    md.append("## Endpoint consistency")
    md.append(f"- Model basal at day {D}: {model_basal:.6g}")
    md.append(f"- Model osteo at day {D}: {model_osteo:.6g}")
    md.append(f"- Exp ratio (AA/BASAL): {ratio_exp:.6g}")
    md.append(f"- Model ratio (AA/BASAL): {ratio_model:.6g}\n")

    md.append("## Files")
    md.append("- `aa_collagen_output.xlsx`")
    md.append("- `basal_vs_aa_minimal.pdf` / `basal_vs_aa_minimal.png`")
    md.append("- `aa_exposure_daily.pdf` / `aa_exposure_daily.png`")

    (outdir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AA→Collagen linear model (24 h exposure; minimal plotting)"
    )

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
        help="BASAL/Basal collagen IF/DAPI at measure_day",
    )
    parser.add_argument(
        "--osteo_if",
        type=float,
        required=True,
        help="AA/Osteo collagen IF/DAPI at measure_day",
    )

    parser.add_argument(
        "--outfile",
        type=str,
        default="aa_collagen_output.xlsx",
        help="Name of XLSX output file",
    )

    args = parser.parse_args()
    D = int(args.measure_day)
    if D < 1:
        raise ValueError("--measure_day must be >= 1")
    if D > args.days:
        raise ValueError("--measure_day must be <= --days")

    outdir = Path("output_linear")
    outdir.mkdir(exist_ok=True)

    # AA simulation (24 h mean exposure)
    AA_daily_24h, AA_norm = simulate_aa(
        AA0=args.conc,
        half_life_h=args.halflife,
        media_interval_days=args.media,
        total_days=args.days,
        supplement_start_day=SUPP_DAY,
    )

    # NEW: AA exposure plot
    plot_aa_exposure_daily(
        days=args.days,
        supplement_start_day=SUPP_DAY,
        measure_day=D,
        AA_daily_24h=AA_daily_24h,
        AA_norm=AA_norm,
        outdir=outdir,
    )

    # Fit parameters from endpoints
    R0, R_AA = fit_parameters(
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        AA_norm=AA_norm,
        measure_day=D,
    )

    # Collagen simulation
    collagen_daily, collagen_total = simulate_collagen(
        days=args.days,
        R0=R0,
        R_AA=R_AA,
        AA_norm=AA_norm,
    )

    # Save table
    df = pd.DataFrame(
        {
            "Day": np.arange(1, args.days + 1),
            "AA_daily_24h": AA_daily_24h,
            "AA_norm_24h": AA_norm,
            "Collagen_daily_osteo": collagen_daily,
            "Collagen_total_osteo": collagen_total,
            "Collagen_total_basal": R0 * np.arange(1, args.days + 1),
        }
    )
    xlsx_path = outdir / args.outfile
    df.to_excel(xlsx_path, index=False)

    # Plot: BASAL vs AA (osteo)
    plot_basal_vs_aa_minimal(
        days=args.days,
        measure_day=D,
        R0=R0,
        collagen_total_osteo=collagen_total,
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        outdir=outdir,
    )

    # Markdown
    write_markdown(
        outdir=outdir,
        args=args,
        measure_day=D,
        R0=R0,
        R_AA=R_AA,
        AA_daily_24h=AA_daily_24h,
        AA_norm=AA_norm,
        collagen_daily=collagen_daily,
        collagen_total=collagen_total,
    )

    print(f"[✓] Saved: {xlsx_path}")
    print(f"[✓] Saved: {outdir / 'basal_vs_aa_minimal.pdf'}")
    print(f"[✓] Saved: {outdir / 'basal_vs_aa_minimal.png'}")
    print(f"[✓] Saved: {outdir / 'aa_exposure_daily.pdf'}")
    print(f"[✓] Saved: {outdir / 'aa_exposure_daily.png'}")
    print(f"[✓] Saved: {outdir / 'results.md'}")


if __name__ == "__main__":
    main()


"""
Example:
python ascorbic_linear_model.py \
  --conc 50 \
  --halflife 3 \
  --media 2 \
  --days 14 \
  --measure_day 14 \
  --basal_if 1.36 \
  --osteo_if 1.58
"""
