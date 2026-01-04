#!/usr/bin/env python3
"""
unifiedalpmodel.py

Piecewise-linear ALP accumulation model (hourly) using ENDPOINT inputs.

Interpretation of inputs
------------------------
This script expects the user to provide endpoint ALP measurements at the
measurement day (default day 10):

  --r_basal  : P_day_measure^basal   (measured ALP in basal condition)
  --r_osteo  : P_day_measure^osteo   (measured ALP in osteogenic condition; TOTAL)

These are NOT hourly rates.

Model (cumulative, piecewise-linear)
------------------------------------
Let P(t) be cumulative ALP activity up to day_measure. Induction begins at day_start.

Basal contribution is active from day 0:
  dP_basal/dt = r_b   (units/hour)

Osteogenic contribution is active only after day_start:
  dP_osteo/dt = gamma * r_o   (units/hour)

Total:
  P_total(t) = P_basal(t) + P_osteo(t)

Endpoint constraints (with day_measure = D, day_start = S):
  P_basal(D) = P_day_measure^basal
  P_total(D) = P_day_measure^osteo

Therefore:
  r_b = P_day_measure^basal / (D*24)
  r_o = (P_day_measure^osteo - P_day_measure^basal) / ((D-S)*24*gamma)

Outputs
-------
- alp_params.npz            : fast runtime input for OsteoMin (includes computed rates)
- alp_params.json           : readable parameters + metadata
- alp_hourly.csv            : hourly trajectory (total + components)
- alp_daily_sampled.csv     : values sampled at integer days
- alp_model_components_plot.(pdf|png) : publication-style plot with endpoint markers

Example
-------
python unifiedalpmodel.py \
  --r_basal 0.70 \
  --r_osteo 1.16 \
  --gamma 1.0 \
  --day_start 3 --day_measure 10 \
  --dt_hours 1 \
  --outdir alp_linear_out
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Plot style (aligned with mineral model)
# -----------------------------
def _set_style():
    """Lightweight PLOS/Nature-friendly defaults with Okabe–Ito colors."""
    okabe_ito = [
        "#0072B2",
        "#D55E00",
        "#009E73",
        "#CC79A7",
        "#56B4E9",
        "#E69F00",
        "#F0E442",
        "#000000",
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 10.0,
            "axes.titlesize": 11.0,
            "axes.labelsize": 10.5,
            "legend.fontsize": 9.0,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "lines.linewidth": 2.0,
            "axes.prop_cycle": plt.cycler(color=okabe_ito),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def derive_hourly_rates_from_endpoints(
    P_day_measure_basal: float,
    P_day_measure_osteo_total: float,
    gamma: float,
    day_start: float,
    day_measure: float,
) -> tuple[float, float]:
    """Derive (r_b, r_o) in units/hour from endpoint constraints."""
    if day_measure <= 0:
        raise ValueError("day_measure must be > 0.")
    if not (0 <= day_start < day_measure):
        raise ValueError("day_start must satisfy 0 <= day_start < day_measure.")
    if gamma <= 0:
        raise ValueError("gamma must be > 0.")
    if P_day_measure_basal < 0 or P_day_measure_osteo_total < 0:
        raise ValueError("Endpoint ALP values must be >= 0.")
    if P_day_measure_osteo_total < P_day_measure_basal:
        raise ValueError(
            "P_day_measure^osteo must be >= P_day_measure^basal for this additive model."
        )

    hours_total = day_measure * 24.0
    hours_osteo = (day_measure - day_start) * 24.0

    r_b = P_day_measure_basal / hours_total
    r_o = (P_day_measure_osteo_total - P_day_measure_basal) / (hours_osteo * gamma)
    return float(r_b), float(r_o)


def alp_piecewise_linear_components(
    r_basal_hour: float,
    r_osteo_hour: float,
    gamma: float,
    day_start: float,
    day_measure: float,
    dt_hours: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_hours, P_total, P_basal_component, P_osteo_component
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0.")
    if day_measure <= 0:
        raise ValueError("day_measure must be > 0.")
    if not (0 <= day_start <= day_measure):
        raise ValueError("day_start must be within [0, day_measure].")
    if r_basal_hour < 0 or r_osteo_hour < 0:
        raise ValueError("Hourly rates must be >= 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")

    total_steps = int(round((day_measure * 24.0) / dt_hours))
    t_hours = np.arange(0, total_steps + 1) * dt_hours
    t_days = t_hours / 24.0

    P_basal = np.zeros_like(t_hours, dtype=float)
    P_osteo = np.zeros_like(t_hours, dtype=float)

    for i in range(1, len(t_hours)):
        td_prev = t_days[i - 1]

        # basal always contributes
        P_basal[i] = P_basal[i - 1] + dt_hours * r_basal_hour

        # osteogenic only after induction
        if td_prev > day_start:
            P_osteo[i] = P_osteo[i - 1] + dt_hours * (gamma * r_osteo_hour)
        else:
            P_osteo[i] = P_osteo[i - 1]

    P_total = P_basal + P_osteo
    return t_hours, P_total, P_basal, P_osteo


def sample_at_integer_days(t_hours: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample y at exact integer days (0..floor(day_measure))."""
    dt = float(np.median(np.diff(t_hours))) if len(t_hours) > 1 else 1.0
    t_days = t_hours / 24.0
    max_day = int(np.floor(t_days[-1] + 1e-9))
    days = np.arange(0, max_day + 1, dtype=int)
    sampled = np.zeros_like(days, dtype=float)

    for i, d in enumerate(days):
        idx = int(round((d * 24.0) / dt))
        idx = min(idx, len(y) - 1)
        sampled[i] = float(y[idx])

    return days, sampled


def _index_for_day(t_hours: np.ndarray, day: float) -> int:
    dt = float(np.median(np.diff(t_hours))) if len(t_hours) > 1 else 1.0
    idx = int(round((day * 24.0) / dt))
    return int(np.clip(idx, 0, len(t_hours) - 1))


def save_params(outdir: Path, params: dict) -> tuple[Path, Path]:
    """Save parameters in both NPZ (fast runtime) and JSON (readable)."""
    npz_path = outdir / "alp_params.npz"
    json_path = outdir / "alp_params.json"

    np.savez(
        npz_path,
        # Endpoints
        P_day_measure_basal=float(params["P_day_measure_basal"]),
        P_day_measure_osteo=float(params["P_day_measure_osteo"]),
        # Derived hourly rates (what OsteoMin should use)
        r_basal_hour=float(params["r_basal_hour"]),
        r_osteo_hour=float(params["r_osteo_hour"]),
        gamma=float(params["gamma"]),
        day_start=float(params["day_start"]),
        day_measure=float(params["day_measure"]),
        dt_hours=float(params["dt_hours"]),
        units=str(params.get("units", "ALP_units")),
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2, sort_keys=True)

    return npz_path, json_path


def save_components_plot(
    t_hours: np.ndarray,
    P_total: np.ndarray,
    P_basal: np.ndarray,
    P_osteo: np.ndarray,
    day_start: float,
    day_measure: float,
    outdir: Path,
    title: str,
) -> tuple[Path, Path]:
    """
    Publication-style plot:
    - total + components
    - subtle vertical lines at day_start and day_measure
    - triangle marker at induction on total curve
    - filled circle marker at measurement endpoint on total curve
    - open circle marker at basal endpoint (on basal curve) at day_measure
    """
    _set_style()

    t_days = t_hours / 24.0
    idx_start = _index_for_day(t_hours, day_start)
    idx_end = _index_for_day(t_hours, day_measure)

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)

    ax.plot(t_days, P_total, linewidth=2.6, label="Total ALP")
    ax.plot(t_days, P_basal, linewidth=1.8, linestyle="--", label="Basal contribution")
    ax.plot(t_days, P_osteo, linewidth=1.8, linestyle="--", label="Osteogenic contribution")

    # Reference lines
    ax.axvline(day_start, linestyle=":", linewidth=1.2, alpha=0.5)
    ax.axvline(day_measure, linestyle=":", linewidth=1.2, alpha=0.5)

    # Markers
    ax.scatter([day_start], [P_total[idx_start]], s=80, marker="^", zorder=6, clip_on=False, label="Induction")
    ax.scatter([day_measure], [P_total[idx_end]], s=90, marker="o", zorder=7, clip_on=False, label="Osteo endpoint")
    ax.scatter(
        [day_measure],
        [P_basal[idx_end]],
        s=75,
        marker="o",
        facecolors="none",
        linewidths=1.6,
        zorder=7,
        clip_on=False,
    )

    # Layout
    xpad = max(0.2, 0.03 * day_measure)
    y_min = float(min(P_total.min(), P_basal.min(), P_osteo.min()))
    y_max = float(max(P_total.max(), P_basal.max(), P_osteo.max()))
    ypad = 0.08 * (y_max - y_min + 1e-12)

    ax.set_xlim(0, day_measure + xpad)
    ax.set_ylim(y_min - ypad, y_max + ypad)

    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel("Cumulative ALP activity (measured units)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(frameon=False, loc="upper left")

    pdf_path = outdir / "alp_model_components_plot.pdf"
    png_path = outdir / "alp_model_components_plot.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.15)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return pdf_path, png_path


def _barplot_horizontal(names, S1, ST, title, outdir: Path, fname: str):
    _set_style()
    names = list(names)
    S1 = np.asarray(S1, dtype=float)
    ST = np.asarray(ST, dtype=float)

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.2, 0.42 * len(names) + 1.0), constrained_layout=True)
    ax.barh(y + 0.18, S1[::-1], height=0.34, label="S1")
    ax.barh(y - 0.18, ST[::-1], height=0.34, label="ST")
    ax.set_yticks(y, names[::-1])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Sensitivity index")
    ax.set_title(title)
    ax.grid(False)
    ax.legend(frameon=False, loc="lower right")
    fig.savefig(outdir / f"{fname}.png", bbox_inches="tight", pad_inches=0.12, dpi=100)
    fig.savefig(outdir / f"{fname}.pdf", bbox_inches="tight", pad_inches=0.12, dpi=300)
    plt.close(fig)


def sobol_alp_rates(
    *,
    outdir: Path,
    r_basal_endpoint: float,
    r_osteo_endpoint: float,
    gamma: float,
    day_start: float,
    day_measure: float,
    baseN: int,
    rel_span: float,
):
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    rel_span = float(rel_span)
    if rel_span <= 0:
        raise ValueError("rel_span must be > 0.")

    rb_lo = max(0.0, r_basal_endpoint * (1.0 - rel_span))
    rb_hi = r_basal_endpoint * (1.0 + rel_span)
    ro_lo = max(0.0, r_osteo_endpoint * (1.0 - rel_span))
    ro_hi = r_osteo_endpoint * (1.0 + rel_span)
    g_lo = max(1e-8, gamma * (1.0 - rel_span))
    g_hi = gamma * (1.0 + rel_span)
    ds_lo = 0.0
    ds_hi = max(0.5, day_measure - 1e-6)

    names = ["P_basal_end", "P_osteo_end", "gamma", "day_start"]
    bounds = [[rb_lo, rb_hi], [ro_lo, ro_hi], [g_lo, g_hi], [ds_lo, ds_hi]]
    problem = {"num_vars": len(names), "names": names, "bounds": bounds}
    X = saltelli.sample(problem, N=int(baseN), calc_second_order=True)

    r_b = np.zeros(X.shape[0], dtype=float)
    r_o = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        rb_i, ro_i = derive_hourly_rates_from_endpoints(
            P_day_measure_basal=float(X[i, 0]),
            P_day_measure_osteo_total=float(X[i, 1]),
            gamma=float(X[i, 2]),
            day_start=float(X[i, 3]),
            day_measure=float(day_measure),
        )
        r_b[i] = rb_i
        r_o[i] = ro_i

    Si_rb = sobol.analyze(problem, r_b, calc_second_order=True, print_to_console=False)
    Si_ro = sobol.analyze(problem, r_o, calc_second_order=True, print_to_console=False)

    _barplot_horizontal(names, Si_rb["S1"], Si_rb["ST"], "Sobol: ALP basal rate", outdir, "sobol_alp_r_basal")
    _barplot_horizontal(names, Si_ro["S1"], Si_ro["ST"], "Sobol: ALP osteogenic rate", outdir, "sobol_alp_r_osteo")

    return {
        "names": names,
        "r_basal": {"S1": Si_rb["S1"].tolist(), "ST": Si_rb["ST"].tolist(), "S2": Si_rb["S2"].tolist()},
        "r_osteo": {"S1": Si_ro["S1"].tolist(), "ST": Si_ro["ST"].tolist(), "S2": Si_ro["S2"].tolist()},
    }


def run_alp_model(
    *,
    r_basal: float,
    r_osteo: float,
    gamma: float = 1.0,
    day_start: float = 3.0,
    day_measure: float = 10.0,
    dt_hours: float = 1.0,
    outdir: Path | str = "alp_linear_output",
    units: str = "ALP_units",
    sobol: bool = False,
    sobol_baseN: int = 512,
    sobol_rel_span: float = 0.3,
) -> dict:
    """Run ALP model with keyword args matching the CLI; returns output paths."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    P_day_measure_basal = float(r_basal)
    P_day_measure_osteo = float(r_osteo)

    r_b_hour, r_o_hour = derive_hourly_rates_from_endpoints(
        P_day_measure_basal=P_day_measure_basal,
        P_day_measure_osteo_total=P_day_measure_osteo,
        gamma=float(gamma),
        day_start=float(day_start),
        day_measure=float(day_measure),
    )

    t_hours, P_total, P_basal, P_osteo = alp_piecewise_linear_components(
        r_basal_hour=r_b_hour,
        r_osteo_hour=r_o_hour,
        gamma=float(gamma),
        day_start=float(day_start),
        day_measure=float(day_measure),
        dt_hours=float(dt_hours),
    )

    pd.DataFrame(
        {
            "hour": t_hours,
            "day": t_hours / 24.0,
            "ALP_total": P_total,
            "ALP_basal": P_basal,
            "ALP_osteo": P_osteo,
        }
    ).to_csv(outdir / "alp_hourly.csv", index=False)

    days_i, total_sampled = sample_at_integer_days(t_hours, P_total)
    _, basal_sampled = sample_at_integer_days(t_hours, P_basal)
    _, osteo_sampled = sample_at_integer_days(t_hours, P_osteo)

    pd.DataFrame(
        {
            "Day": days_i,
            "ALP_total_sampled": total_sampled,
            "ALP_basal_sampled": basal_sampled,
            "ALP_osteo_sampled": osteo_sampled,
        }
    ).to_csv(outdir / "alp_daily_sampled.csv", index=False)

    params = {
        "P_day_measure_basal": P_day_measure_basal,
        "P_day_measure_osteo": P_day_measure_osteo,
        "r_basal_hour": r_b_hour,
        "r_osteo_hour": r_o_hour,
        "gamma": float(gamma),
        "day_start": float(day_start),
        "day_measure": float(day_measure),
        "dt_hours": float(dt_hours),
        "units": units,
        "note": (
            "Inputs r_basal and r_osteo are endpoint ALP values at day_measure (basal vs osteo TOTAL). "
            "Hourly rates are derived to satisfy P_basal(day_measure)=P_basal_endpoint and "
            "P_total(day_measure)=P_osteo_endpoint."
        ),
    }
    npz_path, json_path = save_params(outdir, params)

    title = "ALP accumulation: basal + osteogenic contributions"
    pdf_path, png_path = save_components_plot(
        t_hours=t_hours,
        P_total=P_total,
        P_basal=P_basal,
        P_osteo=P_osteo,
        day_start=float(day_start),
        day_measure=float(day_measure),
        outdir=outdir,
        title=title,
    )

    if sobol:
        try:
            import SALib  # noqa: F401
        except Exception:
            raise RuntimeError("SALib not installed. Install: pip install SALib")

        sobol_out = sobol_alp_rates(
            outdir=outdir,
            r_basal_endpoint=P_day_measure_basal,
            r_osteo_endpoint=P_day_measure_osteo,
            gamma=float(gamma),
            day_start=float(day_start),
            day_measure=float(day_measure),
            baseN=int(sobol_baseN),
            rel_span=float(sobol_rel_span),
        )
        (outdir / "sobol_alp_rates.json").write_text(json.dumps(sobol_out, indent=2), encoding="utf-8")

    return {
        "outdir": outdir,
        "params_npz": npz_path,
        "params_json": json_path,
        "hourly_csv": outdir / "alp_hourly.csv",
        "daily_csv": outdir / "alp_daily_sampled.csv",
        "plot_pdf": pdf_path,
        "plot_png": png_path,
        "r_basal_hour": r_b_hour,
        "r_osteo_hour": r_o_hour,
        "t_hours": t_hours,
        "P_total": P_total,
        "P_basal": P_basal,
        "P_osteo": P_osteo,
    }


def main():
    p = argparse.ArgumentParser(
        description="Piecewise-linear ALP model (hourly) using endpoint inputs (basal vs osteo at day_measure)."
    )

    # NOTE: names kept as you used them, but they are ENDPOINTS now.
    p.add_argument("--r_basal", type=float, required=True, help="Endpoint ALP in basal condition at day_measure.")
    p.add_argument("--r_osteo", type=float, required=True, help="Endpoint ALP in osteogenic condition at day_measure (TOTAL).")
    p.add_argument("--gamma", type=float, default=1.0, help="Scaling factor for osteogenic increment (dimensionless).")

    p.add_argument("--day_start", type=float, default=3.0, help="Day of osteogenic induction (default 3).")
    p.add_argument("--day_measure", type=float, default=10.0, help="Measurement day for endpoints (default 10).")
    p.add_argument("--dt_hours", type=float, default=1.0, help="Time step in hours (default 1).")
    p.add_argument("--outdir", type=str, default="alp_linear_output", help="Output directory.")
    p.add_argument("--units", type=str, default="ALP_units", help="Units label stored with params (for endpoints).")
    p.add_argument("--sobol", action="store_true", help="Enable Sobol sensitivity for derived ALP rates.")
    p.add_argument("--sobol_baseN", type=int, default=512, help="Sobol base N.")
    p.add_argument("--sobol_rel_span", type=float, default=0.3, help="Relative span for Sobol parameter bounds.")

    args = p.parse_args()

    run_alp_model(
        r_basal=float(args.r_basal),
        r_osteo=float(args.r_osteo),
        gamma=float(args.gamma),
        day_start=float(args.day_start),
        day_measure=float(args.day_measure),
        dt_hours=float(args.dt_hours),
        outdir=Path(args.outdir),
        units=args.units,
        sobol=bool(args.sobol),
        sobol_baseN=int(args.sobol_baseN),
        sobol_rel_span=float(args.sobol_rel_span),
    )


if __name__ == "__main__":
    main()
