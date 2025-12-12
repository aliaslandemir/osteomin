#!/usr/bin/env python3
"""
AA → Collagen accumulation model (OsteoMin, Michaelis–Menten version; nicer plots)
UPDATED: uses a 24 h effective exposure window (daily mean).
ADDED (biological): also computes a 3 h “active-window” mean exposure per day (matches your ~3 h AA half-life idea).
UNITS: AA concentration and Km are in µg/mL ONLY.

What this script does
- Simulates AA decay + medium-change pulses starting at SUPP_DAY (default 3)
- Uses 24 h mean AA as the daily exposure signal (µg/mL)
- Also computes 3 h mean AA per day (first 3 hours of each day) as an “active-window” exposure proxy
- Converts AA exposure into an additive collagen production rate via Michaelis–Menten:
      sat(d)  = AA_daily_24h(d) / (Km + AA_daily_24h(d))
      rate(d) = R0 + R_max * sat(d)
- Accumulates collagen over days
- Fits R0 and R_max using ONLY your endpoints at --measure_day (Km must be provided):
      BASAL (basal) endpoint -> R0
      AA (osteo) endpoint -> R_max
- Saves:
    * XLSX table
    * Figure 1: BASAL vs AA collagen totals + endpoint markers + AA-driven rate component (PDF/PNG)
    * Figure 2: AA exposure (24 h mean + 3 h mean) + sat(d) over time (PDF/PNG)
    * Figure 3: Michaelis–Menten curve: sat(AA) and AA-driven rate vs AA,
                with daily AA points (24 h + 3 h) overlaid and a biologically motivated x-range (PDF/PNG)
    * markdown summary
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

SUPP_DAY = 3  # day osteogenic medium starts (AA + βGP)
ACTIVE_WINDOW_H = 3  # biological “active lifetime” proxy window for exposure plots


# =====================================================================
# AA DECAY SIMULATION (24 h + active-window means)
# =====================================================================
def simulate_aa(
    AA0_ug_ml: float,
    half_life_h: float,
    media_interval_days: int,
    total_days: int,
    supplement_start_day: int = SUPP_DAY,
    active_window_h: int = ACTIVE_WINDOW_H,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns
    -------
    AA_daily_24h : (days,) mean AA over each 24 h day [µg/mL]
    AA_daily_aw  : (days,) mean AA over first `active_window_h` hours of each day [µg/mL]
    AA_hours     : (total_hours+1,) hourly AA [µg/mL] (for debugging/traceability)
    """
    if total_days < 1:
        raise ValueError("total_days must be >= 1")
    if half_life_h <= 0:
        raise ValueError("half_life_h must be > 0")
    if media_interval_days < 1:
        raise ValueError("media_interval_days must be >= 1")
    if AA0_ug_ml < 0:
        raise ValueError("AA0 must be >= 0")
    if active_window_h < 1 or active_window_h > 24:
        raise ValueError("active_window_h must be in [1, 24]")

    k = np.log(2.0) / float(half_life_h)
    total_hours = total_days * 24

    AA_hours = np.zeros(total_hours + 1, dtype=float)

    start_h = supplement_start_day * 24
    if start_h <= total_hours:
        AA_hours[start_h] = AA0_ug_ml

    # decay + medium changes
    for t in range(start_h + 1, total_hours + 1):
        AA_hours[t] = AA_hours[t - 1] * np.exp(-k)
        # medium change -> reset to AA0
        if (t - start_h) > 0 and (t - start_h) % (media_interval_days * 24) == 0:
            AA_hours[t] = AA0_ug_ml

    AA_daily_24h = np.zeros(total_days, dtype=float)
    AA_daily_aw = np.zeros(total_days, dtype=float)

    for d in range(total_days):
        s = d * 24
        e = (d + 1) * 24
        AA_daily_24h[d] = AA_hours[s:e].mean()

        aw_e = s + active_window_h
        AA_daily_aw[d] = AA_hours[s:aw_e].mean()

    return AA_daily_24h, AA_daily_aw, AA_hours


# =====================================================================
# MICHAELIS–MENTEN TERMS
# =====================================================================
def mm_sat(AA_ug_ml: np.ndarray, Km_ug_ml: float) -> np.ndarray:
    """sat = AA/(Km+AA) in [0,1] for AA,Km>=0."""
    AA_ug_ml = np.asarray(AA_ug_ml, dtype=float)
    Km_ug_ml = float(Km_ug_ml)
    if Km_ug_ml <= 0:
        raise ValueError("Km must be > 0 (µg/mL)")

    denom = Km_ug_ml + AA_ug_ml
    out = np.zeros_like(AA_ug_ml)
    m = denom > 0
    out[m] = AA_ug_ml[m] / denom[m]
    return np.clip(out, 0.0, 1.0)


def simulate_collagen_mm(
    days: int,
    R0: float,
    R_max: float,
    AA_daily_24h_ug_ml: np.ndarray,
    Km_ug_ml: float,
):
    """
    daily[d] = R0 + R_max * sat_24h[d]
    total[d] = cumulative sum
    """
    sat_24h = mm_sat(AA_daily_24h_ug_ml, Km_ug_ml)

    daily = np.zeros(days, dtype=float)
    total = np.zeros(days, dtype=float)

    for d in range(days):
        rate = R0 + R_max * sat_24h[d]
        daily[d] = rate
        total[d] = rate if d == 0 else total[d - 1] + rate

    aa_driven = R_max * sat_24h
    return sat_24h, aa_driven, daily, total


# =====================================================================
# FITTING (R0 and R_max; Km fixed)
# =====================================================================
def fit_parameters_mm(
    basal_if: float,
    osteo_if: float,
    AA_daily_24h_ug_ml: np.ndarray,
    Km_ug_ml: float,
    measure_day: int,
):
    """
    BASAL endpoint -> R0 via R0 = basal_if / D
    AA endpoint -> R_max via:
        osteo_if = basal_if + R_max * sum_{d<=D} sat_24h[d]
        => R_max = (osteo_if - basal_if) / sum(sat_24h[:D])
    """
    D = int(measure_day)
    if D <= 0:
        raise ValueError("measure_day must be >= 1")
    if basal_if < 0 or osteo_if < 0:
        raise ValueError("Endpoints should be >= 0 in IF/DAPI units.")

    R0 = float(basal_if) / float(D)

    sat_24h = mm_sat(AA_daily_24h_ug_ml, Km_ug_ml)
    S = float(np.sum(sat_24h[:D]))
    R_max = 0.0 if S == 0 else (float(osteo_if) - float(basal_if)) / S

    return R0, R_max


# =====================================================================
# PLOTS (constrained_layout; no tight_layout warnings)
# =====================================================================
def _style_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_basal_vs_aa_better(
    days: int,
    measure_day: int,
    R0: float,
    collagen_total_osteo: np.ndarray,
    aa_driven: np.ndarray,
    basal_if: float,
    osteo_if: float,
    outdir: Path,
):
    D = int(measure_day)
    if D > days:
        raise ValueError("measure_day cannot exceed --days")

    x0 = np.arange(0, days + 1, dtype=float)
    x = np.arange(1, days + 1, dtype=float)

    y_basal = R0 * x0
    y_aa = np.zeros_like(x0)
    y_aa[0] = 0.0
    y_aa[1:] = collagen_total_osteo

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.4), sharex=True, constrained_layout=True
    )

    ax1.plot(x0, y_basal, lw=2.4, label="BASAL (model)")
    ax1.plot(x0, y_aa, lw=2.4, label="AA (osteo model)")
    ax1.scatter([D], [basal_if], marker="x", s=80, linewidths=2.2, label="BASAL measured")
    ax1.scatter([D], [osteo_if], marker="x", s=80, linewidths=2.2, label="AA measured")
    ax1.axvline(D, lw=1.2, ls="--", alpha=0.6)

    ax1.set_ylabel("Collagen (IF/DAPI units)")
    ax1.set_title("Collagen totals over time (MM AA-driven rate; fit by one endpoint/condition)")
    ax1.legend(frameon=False, fontsize=9, ncol=2)
    _style_axes(ax1)

    ax2.plot(x, aa_driven, lw=2.4)
    ax2.axvline(D, lw=1.2, ls="--", alpha=0.6)
    ax2.set_xlabel("Day")
    ax2.set_ylabel("AA-driven rate\n(R_max·AA/(Km+AA))")
    _style_axes(ax2)

    fig.savefig(outdir / "basal_vs_aa_better.pdf", dpi=300)
    fig.savefig(outdir / "basal_vs_aa_better.png", dpi=300)
    plt.close(fig)


def plot_aa_exposure_and_sat(
    days: int,
    supplement_start_day: int,
    measure_day: int,
    AA_daily_24h_ug_ml: np.ndarray,
    AA_daily_aw_ug_ml: np.ndarray,
    sat_24h: np.ndarray,
    outdir: Path,
    active_window_h: int = ACTIVE_WINDOW_H,
):
    """
    Figure 2:
      Panel 1: AA 24h mean + AA active-window mean (first `active_window_h` hours)
      Panel 2: sat_24h = AA_24h/(Km+AA_24h) used by the model
    """
    D = int(measure_day)
    x = np.arange(1, days + 1, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.0), sharex=True, constrained_layout=True
    )

    ax1.plot(x, AA_daily_24h_ug_ml, lw=2.4, label="AA daily mean (24 h)")
    ax1.plot(x, AA_daily_aw_ug_ml, lw=2.0, ls="--", label=f"AA mean (first {active_window_h} h)")
    ax1.axvline(supplement_start_day, lw=1.2, ls="--", alpha=0.6)
    ax1.axvline(D, lw=1.2, ls="--", alpha=0.6)
    ax1.set_ylabel("AA (µg/mL)")
    ax1.set_title("AA exposure over time (24 h vs active-window mean) + saturation term")
    ax1.legend(frameon=False, fontsize=9, ncol=2)
    _style_axes(ax1)

    ax2.plot(x, sat_24h, lw=2.4)
    ax2.axvline(supplement_start_day, lw=1.2, ls="--", alpha=0.6)
    ax2.axvline(D, lw=1.2, ls="--", alpha=0.6)
    ax2.set_ylabel("sat = AA/(Km+AA)\n(using 24 h mean)")
    ax2.set_xlabel("Day")
    _style_axes(ax2)

    fig.savefig(outdir / "aa_exposure_daily.pdf", dpi=300)
    fig.savefig(outdir / "aa_exposure_daily.png", dpi=300)
    plt.close(fig)


def plot_mm_curve(
    AA_daily_24h_ug_ml: np.ndarray,
    AA_daily_aw_ug_ml: np.ndarray,
    AA0_ug_ml: float,
    Km_ug_ml: float,
    R_max: float,
    outdir: Path,
    active_window_h: int = ACTIVE_WINDOW_H,
):
    """
    Figure 3: the actual Michaelis–Menten curve (this is where you SEE the MM shape).

    Biologically motivated x-range:
    - Ensure we show up to ~95% saturation: AA95 ≈ 19·Km
    - Ensure we cover what cells actually see: max of (AA0, max daily 24h mean, max daily active-window mean)
    - Small padding only (5%)
    """
    AA_daily_24h_ug_ml = np.asarray(AA_daily_24h_ug_ml, dtype=float)
    AA_daily_aw_ug_ml = np.asarray(AA_daily_aw_ug_ml, dtype=float)

    pts24 = AA_daily_24h_ug_ml[AA_daily_24h_ug_ml > 0]
    ptsaw = AA_daily_aw_ug_ml[AA_daily_aw_ug_ml > 0]

    AA95 = 19.0 * float(Km_ug_ml)  # ~95% saturation point
    AAmax_obs = float(max(AA0_ug_ml, AA_daily_24h_ug_ml.max(), AA_daily_aw_ug_ml.max()))
    xmax = max(AA95, AAmax_obs) * 1.05
    x = np.linspace(0.0, xmax, 600)

    sat_curve = mm_sat(x, Km_ug_ml)
    rate_curve = R_max * sat_curve

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(7.2, 5.6), sharex=True, constrained_layout=True
    )

    # --- saturation curve ---
    ax1.plot(x, sat_curve, lw=2.6)
    if pts24.size:
        ax1.scatter(pts24, mm_sat(pts24, Km_ug_ml), s=22, alpha=0.9, label="Daily AA points (24 h mean)")
    if ptsaw.size:
        ax1.scatter(
            ptsaw, mm_sat(ptsaw, Km_ug_ml),
            s=22, alpha=0.9, marker="D",
            label=f"Daily AA points (first {active_window_h} h mean)"
        )

    # reference lines
    ax1.axvline(Km_ug_ml, lw=1.2, ls="--", alpha=0.7)
    ax1.axvline(AA0_ug_ml, lw=1.2, ls="--", alpha=0.4)
    ax1.axvline(AA95, lw=1.0, ls=":", alpha=0.6)

    for y, lab in [(0.5, "50%"), (0.9, "90%"), (0.95, "95%")]:
        ax1.axhline(y, lw=1.0, ls="--", alpha=0.35)

    ax1.set_ylabel("sat = AA/(Km+AA)")
    ax1.set_title("Michaelis–Menten saturation curve (daily exposure points overlaid)")
    ax1.legend(frameon=False, fontsize=9, loc="lower right")
    _style_axes(ax1)

    # --- AA-driven rate curve ---
    ax2.plot(x, rate_curve, lw=2.6)
    if pts24.size:
        ax2.scatter(pts24, R_max * mm_sat(pts24, Km_ug_ml), s=22, alpha=0.9)
    if ptsaw.size:
        ax2.scatter(ptsaw, R_max * mm_sat(ptsaw, Km_ug_ml), s=22, alpha=0.9, marker="D")

    ax2.axvline(Km_ug_ml, lw=1.2, ls="--", alpha=0.7)
    ax2.axvline(AA0_ug_ml, lw=1.2, ls="--", alpha=0.4)
    ax2.axvline(AA95, lw=1.0, ls=":", alpha=0.6)

    ax2.set_xlabel("AA (µg/mL)")
    ax2.set_ylabel("AA-driven rate\n(R_max·sat)")
    _style_axes(ax2)

    fig.savefig(outdir / "mm_curve.pdf", dpi=300)
    fig.savefig(outdir / "mm_curve.png", dpi=300)
    plt.close(fig)


# =====================================================================
# MARKDOWN REPORT
# =====================================================================
def write_markdown(
    outdir: Path,
    args: argparse.Namespace,
    measure_day: int,
    R0: float,
    R_max: float,
    AA_daily_24h_ug_ml: np.ndarray,
    AA_daily_aw_ug_ml: np.ndarray,
    sat_24h: np.ndarray,
    aa_driven: np.ndarray,
    collagen_total: np.ndarray,
    active_window_h: int = ACTIVE_WINDOW_H,
):
    D = int(measure_day)
    model_basal = R0 * D
    model_osteo = collagen_total[D - 1]

    ratio_exp = args.osteo_if / args.basal_if if args.basal_if != 0 else np.nan
    ratio_model = model_osteo / model_basal if model_basal != 0 else np.nan

    md = []
    md.append("# AA → Collagen (Michaelis–Menten, 24 h exposure) — minimal report\n")

    md.append("## Inputs")
    md.append(f"- AA0: {args.conc} µg/mL")
    md.append(f"- Km: {args.km} µg/mL")
    md.append(f"- AA half-life: {args.halflife} h")
    md.append(f"- Medium interval: {args.media} days")
    md.append(f"- Supplements start day: {SUPP_DAY}")
    md.append(f"- Simulated days: {args.days}")
    md.append(f"- Measurement day: {D}")
    md.append(f"- Active-window mean shown: first {active_window_h} h per day")
    md.append(f"- BASAL endpoint at day {D}: {args.basal_if}")
    md.append(f"- AA (osteo) endpoint at day {D}: {args.osteo_if}\n")

    md.append("## Fitted parameters (from endpoints, fixed Km)")
    md.append(f"- R0: {R0:.6g} (basal collagen/day)")
    md.append(f"- R_max: {R_max:.6g} (max AA-driven collagen/day)\n")

    md.append("## Diagnostics")
    md.append(f"- max AA_daily_24h: {AA_daily_24h_ug_ml.max():.6g} µg/mL")
    md.append(f"- max AA_daily_{active_window_h}h: {AA_daily_aw_ug_ml.max():.6g} µg/mL")
    md.append(f"- sat_24h range: [{sat_24h.min():.6g}, {sat_24h.max():.6g}]")
    md.append(f"- sum sat_24h[1..D]: {np.sum(sat_24h[:D]):.6g}")
    md.append(f"- max AA-driven rate: {aa_driven.max():.6g}\n")

    md.append("## Endpoint consistency")
    md.append(f"- Model basal at day {D}: {model_basal:.6g}")
    md.append(f"- Model osteo at day {D}: {model_osteo:.6g}")
    md.append(f"- Exp ratio (AA/BASAL): {ratio_exp:.6g}")
    md.append(f"- Model ratio (AA/BASAL): {ratio_model:.6g}\n")

    md.append("## Files")
    md.append("- `aa_collagen_output.xlsx`")
    md.append("- `basal_vs_aa_better.pdf` / `basal_vs_aa_better.png`")
    md.append("- `aa_exposure_daily.pdf` / `aa_exposure_daily.png`")
    md.append("- `mm_curve.pdf` / `mm_curve.png`")

    (outdir / "results.md").write_text("\n".join(md) + "\n", encoding="utf-8")


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(
        description="AA→Collagen Michaelis–Menten model (24 h exposure; nicer plots; µg/mL only)"
    )

    parser.add_argument("--conc", type=float, required=True, help="Initial AA concentration (µg/mL)")
    parser.add_argument("--km", type=float, required=True, help="Km (µg/mL)")
    parser.add_argument("--halflife", type=float, default=3.0, help="AA half-life (hours), default 3")
    parser.add_argument("--media", type=int, default=2, help="Medium change interval (days)")
    parser.add_argument("--days", type=int, default=28, help="Total culture duration (days)")
    parser.add_argument("--measure_day", type=int, default=14, help="Day of collagen IF/DAPI measurement")

    parser.add_argument("--basal_if", type=float, required=True, help="BASAL collagen IF/DAPI at measure_day")
    parser.add_argument("--osteo_if", type=float, required=True, help="AA (osteo) collagen IF/DAPI at measure_day")

    parser.add_argument("--outfile", type=str, default="aa_collagen_output.xlsx", help="Name of XLSX output file")
    args = parser.parse_args()

    D = int(args.measure_day)
    if D < 1:
        raise ValueError("--measure_day must be >= 1")
    if D > args.days:
        raise ValueError("--measure_day must be <= --days")
    if args.km <= 0:
        raise ValueError("--km must be > 0 (µg/mL)")

    outdir = Path("output_mm")
    outdir.mkdir(exist_ok=True)

    # AA exposure
    AA_daily_24h, AA_daily_aw, _AA_hours = simulate_aa(
        AA0_ug_ml=args.conc,
        half_life_h=args.halflife,
        media_interval_days=args.media,
        total_days=args.days,
        supplement_start_day=SUPP_DAY,
        active_window_h=ACTIVE_WINDOW_H,
    )

    # Fit parameters from endpoints (Km fixed)
    R0, R_max = fit_parameters_mm(
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        AA_daily_24h_ug_ml=AA_daily_24h,
        Km_ug_ml=args.km,
        measure_day=D,
    )

    # Simulate collagen under osteo (AA) using MM
    sat_24h, aa_driven, collagen_daily, collagen_total = simulate_collagen_mm(
        days=args.days,
        R0=R0,
        R_max=R_max,
        AA_daily_24h_ug_ml=AA_daily_24h,
        Km_ug_ml=args.km,
    )

    # Save table
    df = pd.DataFrame(
        {
            "Day": np.arange(1, args.days + 1),
            "AA_daily_24h_ug_ml": AA_daily_24h,
            f"AA_daily_{ACTIVE_WINDOW_H}h_ug_ml": AA_daily_aw,
            "AA_sat_MM_24h": sat_24h,
            "AA_driven_rate": aa_driven,
            "Collagen_daily_osteo": collagen_daily,
            "Collagen_total_osteo": collagen_total,
            "Collagen_total_basal": R0 * np.arange(1, args.days + 1),
        }
    )
    xlsx_path = outdir / args.outfile
    df.to_excel(xlsx_path, index=False)

    # Plots
    plot_basal_vs_aa_better(
        days=args.days,
        measure_day=D,
        R0=R0,
        collagen_total_osteo=collagen_total,
        aa_driven=aa_driven,
        basal_if=args.basal_if,
        osteo_if=args.osteo_if,
        outdir=outdir,
    )

    plot_aa_exposure_and_sat(
        days=args.days,
        supplement_start_day=SUPP_DAY,
        measure_day=D,
        AA_daily_24h_ug_ml=AA_daily_24h,
        AA_daily_aw_ug_ml=AA_daily_aw,
        sat_24h=sat_24h,
        outdir=outdir,
        active_window_h=ACTIVE_WINDOW_H,
    )

    plot_mm_curve(
        AA_daily_24h_ug_ml=AA_daily_24h,
        AA_daily_aw_ug_ml=AA_daily_aw,
        AA0_ug_ml=args.conc,
        Km_ug_ml=args.km,
        R_max=R_max,
        outdir=outdir,
        active_window_h=ACTIVE_WINDOW_H,
    )

    # Markdown
    write_markdown(
        outdir=outdir,
        args=args,
        measure_day=D,
        R0=R0,
        R_max=R_max,
        AA_daily_24h_ug_ml=AA_daily_24h,
        AA_daily_aw_ug_ml=AA_daily_aw,
        sat_24h=sat_24h,
        aa_driven=aa_driven,
        collagen_total=collagen_total,
        active_window_h=ACTIVE_WINDOW_H,
    )

    print(f"[✓] Saved: {xlsx_path}")
    print(f"[✓] Saved: {outdir / 'basal_vs_aa_better.pdf'}")
    print(f"[✓] Saved: {outdir / 'basal_vs_aa_better.png'}")
    print(f"[✓] Saved: {outdir / 'aa_exposure_daily.pdf'}")
    print(f"[✓] Saved: {outdir / 'aa_exposure_daily.png'}")
    print(f"[✓] Saved: {outdir / 'mm_curve.pdf'}")
    print(f"[✓] Saved: {outdir / 'mm_curve.png'}")
    print(f"[✓] Saved: {outdir / 'results.md'}")


if __name__ == "__main__":
    main()


"""
Example (ALL in µg/mL):
python ascorbic_mm_model.py \
  --conc 50 \
  --km 5.0 \
  --halflife 3 \
  --media 2 \
  --days 14 \
  --measure_day 14 \
  --basal_if 1.36 \
  --osteo_if 1.58
"""
