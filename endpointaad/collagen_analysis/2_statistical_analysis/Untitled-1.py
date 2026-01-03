#!/usr/bin/env python3
"""
unifiedalpmodel.py

Piecewise-linear ALP accumulation model (hourly) + saving parameters for reuse.

Model (hourly production)
-------------------------
Let P(t) be cumulative ALP activity (assay units). Simulate from day 0 to day_measure
(default day 10) with dt_hours (default 1 h).

Basal phase (0 .. day_start):
    dP/dt = r_basal

Osteogenic phase (day_start .. day_measure):
    dP/dt = r_basal + gamma * r_osteo

Where:
- r_basal : basal ALP production rate (units/hour)
- r_osteo : additional osteogenic production rate from b-GP (units/hour)
- gamma   : dimensionless scaling of osteogenic contribution (MK4, Dex, etc.)
            gamma = 1 is the reference osteogenic condition.

Outputs
-------
- alp_params.npz            : fast runtime input for OsteoMin
- alp_params.json           : human-readable parameters + metadata
- alp_hourly.csv            : total + basal + osteo contributions (hourly)
- alp_daily_sampled.csv     : sampled at integer days
- alp_model_components_plot.(pdf|png)

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


def alp_piecewise_linear_components(
    r_basal: float,
    r_osteo: float,
    gamma: float,
    day_start: float = 3.0,
    day_measure: float = 10.0,
    dt_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Returns:
      t_hours, P_total, P_basal_component, P_osteo_component

    Components are cumulative contributions integrated from 0.
    """
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0.")
    if day_measure <= 0:
        raise ValueError("day_measure must be > 0.")
    if not (0 <= day_start <= day_measure):
        raise ValueError("day_start must be within [0, day_measure].")
    if r_basal < 0 or r_osteo < 0:
        raise ValueError("Rates r_basal and r_osteo must be >= 0.")
    if gamma < 0:
        raise ValueError("gamma must be >= 0.")

    total_steps = int(round((day_measure * 24.0) / dt_hours))
    t_hours = np.arange(0, total_steps + 1) * dt_hours
    t_days = t_hours / 24.0

    P_basal = np.zeros_like(t_hours, dtype=float)
    P_osteo = np.zeros_like(t_hours, dtype=float)

    # Euler integration (exact here because rates are piecewise constant)
    for i in range(1, len(t_hours)):
        td_prev = t_days[i - 1]

        # basal always contributes up to measurement day in THIS window
        P_basal[i] = P_basal[i - 1] + dt_hours * r_basal

        # osteogenic contribution only after induction
        if td_prev > day_start:
            P_osteo[i] = P_osteo[i - 1] + dt_hours * (gamma * r_osteo)
        else:
            P_osteo[i] = P_osteo[i - 1]

    P_total = P_basal + P_osteo
    return t_hours, P_total, P_basal, P_osteo


def sample_at_integer_days(
    t_hours: np.ndarray, y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
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


def save_params(outdir: Path, params: dict) -> tuple[Path, Path]:
    """Save parameters in both NPZ (fast runtime) and JSON (readable)."""
    npz_path = outdir / "alp_params.npz"
    json_path = outdir / "alp_params.json"

    # NPZ for fast loading in OsteoMin
    np.savez(
        npz_path,
        r_basal=float(params["r_basal"]),
        r_osteo=float(params["r_osteo"]),
        gamma=float(params["gamma"]),
        day_start=float(params["day_start"]),
        day_measure=float(params["day_measure"]),
        dt_hours=float(params["dt_hours"]),
        units=str(params.get("units", "ALP_units_per_hour")),
    )

    # JSON for provenance / readability
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
    """Plot total ALP plus basal/osteo cumulative components, with endpoint marker."""
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )

    t_days = t_hours / 24.0

    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    # Lines: total + components (no explicit colors set)
    ax.plot(t_days, P_total, linewidth=2.4, label="Total ALP (basal + osteo)")
    ax.plot(t_days, P_basal, linewidth=1.8, linestyle="--", label="Basal contribution")
    ax.plot(
        t_days, P_osteo, linewidth=1.8, linestyle="--", label="Osteogenic contribution"
    )

    # Induction and measurement markers
    ax.axvline(day_start, linestyle=":", linewidth=1.4, alpha=0.8)
    ax.axvline(day_measure, linestyle=":", linewidth=1.4, alpha=0.8)

    # Endpoint marker (exact last index)
    ax.scatter([day_measure], [P_total[-1]], s=80, zorder=6, clip_on=False)

    # Light annotation
    ax.text(
        day_start + 0.05,
        np.max(P_total) * 0.05,
        "Induction",
        rotation=90,
        va="bottom",
        alpha=0.75,
    )
    ax.text(
        day_measure + 0.05,
        np.max(P_total) * 0.05,
        "Measurement",
        rotation=90,
        va="bottom",
        alpha=0.75,
    )

    # Padding
    xpad = max(0.2, 0.03 * day_measure)
    y_min = float(np.min(P_total))
    y_max = float(np.max(P_total))
    ypad = 0.08 * (y_max - y_min + 1e-12)

    ax.set_xlim(0, day_measure + xpad)
    ax.set_ylim(y_min - ypad, y_max + ypad)

    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel("Cumulative ALP activity (measured units)")
    ax.set_title(title)
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, loc="upper left")

    pdf_path = outdir / "alp_model_components_plot.pdf"
    png_path = outdir / "alp_model_components_plot.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return pdf_path, png_path


def main():
    p = argparse.ArgumentParser(
        description="Piecewise-linear ALP accumulation model (hourly) + reusable parameter export."
    )
    p.add_argument(
        "--r_basal",
        type=float,
        required=True,
        help="Basal ALP production rate (units per hour).",
    )
    p.add_argument(
        "--r_osteo",
        type=float,
        required=True,
        help="Additional osteogenic ALP production rate (units per hour).",
    )
    p.add_argument(
        "--gamma",
        type=float,
        default=1.0,
        help="Scaling factor for osteogenic contribution (dimensionless).",
    )

    p.add_argument(
        "--day_start",
        type=float,
        default=3.0,
        help="Day of osteogenic supplementation (default 3).",
    )
    p.add_argument(
        "--day_measure",
        type=float,
        default=10.0,
        help="Day of ALP measurement (default 10).",
    )
    p.add_argument(
        "--dt_hours", type=float, default=1.0, help="Time step in hours (default 1)."
    )
    p.add_argument(
        "--outdir", type=str, default="alp_linear_output", help="Output directory."
    )

    p.add_argument(
        "--units",
        type=str,
        default="ALP_units_per_hour",
        help="Units label stored with params.",
    )
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Simulate from day 0 to day_measure (inclusive)
    t_hours, P_total, P_basal, P_osteo = alp_piecewise_linear_components(
        r_basal=args.r_basal,
        r_osteo=args.r_osteo,
        gamma=args.gamma,
        day_start=args.day_start,
        day_measure=args.day_measure,
        dt_hours=args.dt_hours,
    )

    # Save trajectories
    df_hourly = pd.DataFrame(
        {
            "hour": t_hours,
            "day": t_hours / 24.0,
            "ALP_total": P_total,
            "ALP_basal": P_basal,
            "ALP_osteo": P_osteo,
        }
    )
    df_hourly.to_csv(outdir / "alp_hourly.csv", index=False)

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

    # Save parameters for later (fast runtime + readable)
    params = {
        "r_basal": float(args.r_basal),
        "r_osteo": float(args.r_osteo),
        "gamma": float(args.gamma),
        "day_start": float(args.day_start),
        "day_measure": float(args.day_measure),
        "dt_hours": float(args.dt_hours),
        "units": args.units,
        "note": "Piecewise-linear cumulative ALP; basal + gamma*osteo after induction.",
    }
    save_params(outdir, params)

    # Plot (components + endpoint)
    title = "ALP accumulation (piecewise-linear): basal + osteogenic contributions"
    save_components_plot(
        t_hours=t_hours,
        P_total=P_total,
        P_basal=P_basal,
        P_osteo=P_osteo,
        day_start=args.day_start,
        day_measure=args.day_measure,
        outdir=outdir,
        title=title,
    )


if __name__ == "__main__":
    main()
