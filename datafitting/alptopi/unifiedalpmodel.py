#!/usr/bin/env python3
"""
unifiedalpmodel.py

Endpoint-anchored ALP accumulation model (1 h resolution) + variance-based sensitivity.

Plotting note
-------------
Do NOT plot daily means if you want the endpoint marker to sit on the curve.
Daily means average over an entire day, while the model anchors the instantaneous
endpoint at day_measure. Therefore, plots are generated from the hourly trajectory.

Naming (clear)
--------------
alp_baseline    : ALP at osteogenic onset (day_start), in assay units
alp_endpoint    : Measured ALP at day_measure (default day 10)
tau_alp_hours   : Timescale controlling ALP rise kinetics (hours)
day_start       : Osteogenic supplementation start day (default 3)
day_measure     : ALP measurement day (default 10)

Model (endpoint-anchored saturating rise)
-----------------------------------------
For t <= day_start:  ALP(t) = alp_baseline
For day_start < t <= day_measure:
    ALP(t) = alp_baseline + (alp_endpoint - alp_baseline) * g(t)/g(day_measure)
where g(t) = 1 - exp(-(t - day_start)/tau_alp)

Outputs
-------
- alp_hourly.csv            : hourly trajectory (includes exact endpoint time)
- alp_daily_sampled.csv     : ALP sampled at integer days (exact, not averaged)
- alp_model_publication_plot.(pdf|png)
- alp_sensitivity.(pdf|png) and alp_sensitivity.csv (optional)

Example
-------
python unifiedalpmodel.py \
  --alp_baseline 0.70 --alp_endpoint 1.16 \
  --day_start 3 --day_measure 10 \
  --tau_alp_hours 48 \
  --taus_for_plot 12 24 48 96 \
  --outdir unifiedalp_out
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def alp_endpoint_anchored_profile(
    alp_baseline: float,
    alp_endpoint: float,
    tau_alp_hours: float,
    day_start: float = 3.0,
    day_measure: float = 10.0,
    dt_hours: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute ALP activity from day 0 to day_measure (inclusive) at dt_hours resolution."""
    if tau_alp_hours <= 0:
        raise ValueError("tau_alp_hours must be > 0.")
    if day_measure <= day_start:
        raise ValueError("day_measure must be > day_start.")
    if dt_hours <= 0:
        raise ValueError("dt_hours must be > 0.")

    total_steps = int(round((day_measure * 24.0) / dt_hours))
    t_hours = np.arange(0, total_steps + 1) * dt_hours
    t_days = t_hours / 24.0

    alp = np.full_like(t_hours, fill_value=alp_baseline, dtype=float)

    denom = 1.0 - np.exp(-((day_measure - day_start) * 24.0) / tau_alp_hours)
    if denom <= 0:
        denom = np.finfo(float).eps

    for i, td in enumerate(t_days):
        if td <= day_start:
            alp[i] = alp_baseline
        else:
            num = 1.0 - np.exp(-((td - day_start) * 24.0) / tau_alp_hours)
            alp[i] = alp_baseline + (alp_endpoint - alp_baseline) * (num / denom)

    # enforce exact endpoint (robust to rounding)
    alp[-1] = alp_endpoint
    return t_hours, alp


def sample_at_integer_days(t_hours: np.ndarray, alp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sample ALP at exact integer days (0..day_measure)."""
    dt = float(np.median(np.diff(t_hours))) if len(t_hours) > 1 else 1.0
    t_days = t_hours / 24.0
    max_day = int(np.floor(t_days[-1] + 1e-9))
    days = np.arange(0, max_day + 1, dtype=int)
    sampled = np.zeros_like(days, dtype=float)

    for i, d in enumerate(days):
        idx = int(round((d * 24.0) / dt))
        idx = min(idx, len(alp) - 1)
        sampled[i] = float(alp[idx])

    return days, sampled


def alp_auc_between_days(t_hours: np.ndarray, alp: np.ndarray, day_start: float, day_measure: float) -> float:
    """Area under ALP curve between (day_start, day_measure], in day-units."""
    t_days = t_hours / 24.0
    mask = (t_days > day_start) & (t_days <= day_measure)
    return float(np.trapz(alp[mask], t_days[mask]))


def save_alp_plot(
    curves: dict[float, tuple[np.ndarray, np.ndarray]],
    alp_baseline: float,
    alp_endpoint: float,
    day_start: float,
    day_measure: float,
    outdir: Path,
) -> tuple[Path, Path]:
    """
    Publication-style plot using hourly trajectories (no endpoint averaging artifacts),
    with padding to prevent clipping of titles and endpoint markers.
    """
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

    fig, ax = plt.subplots(figsize=(7.0, 4.4), constrained_layout=True)

    for tau_h, (t_hours, alp) in curves.items():
        ax.plot(t_hours / 24.0, alp, linewidth=2.0, label=rf"$\tau_\mathrm{{ALP}}={tau_h:g}$ h")

    # Visual anchors
    ax.axvline(day_start, linestyle="--", linewidth=1.2, alpha=0.6)
    ax.axvline(day_measure, linestyle="--", linewidth=1.2, alpha=0.6)

    # Exact anchor markers at start and endpoint (avoid clipping)
    ax.scatter([day_start, day_measure], [alp_baseline, alp_endpoint], s=70, zorder=6, clip_on=False)

    # Padding so the endpoint marker is not cut off
    xpad = max(0.15, 0.02 * day_measure)  # in days
    y_min = min(alp_baseline, min(v[1].min() for v in curves.values()))
    y_max = max(alp_endpoint, max(v[1].max() for v in curves.values()))
    ypad = 0.06 * (y_max - y_min + 1e-12)

    ax.set_xlim(0, day_measure + xpad)
    ax.set_ylim(y_min - ypad, y_max + ypad)

    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel("ALP activity (measured units)")
    ax.set_title("Endpoint-anchored ALP accumulation (up to measurement day)")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(frameon=False, title="Timescale", loc="upper left")

    pdf_path = outdir / "alp_model_publication_plot.pdf"
    png_path = outdir / "alp_model_publication_plot.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return pdf_path, png_path


def sobol_sensitivity(metric_func, bounds: dict[str, tuple[float, float]], N: int, rng: np.random.Generator):
    """Sobol-style variance-based sensitivity (Saltelli first-order, Jansen total-order)."""
    names = list(bounds.keys())
    k = len(names)
    A = rng.random((N, k))
    B = rng.random((N, k))

    def scale(U):
        X = np.empty_like(U)
        for j, name in enumerate(names):
            lo, hi = bounds[name]
            X[:, j] = lo + (hi - lo) * U[:, j]
        return X

    XA, XB = scale(A), scale(B)

    def evaluate(X):
        Y = np.zeros(X.shape[0])
        for i in range(X.shape[0]):
            params = {names[j]: float(X[i, j]) for j in range(k)}
            Y[i] = float(metric_func(params))
        return Y

    YA, YB = evaluate(XA), evaluate(XB)
    VY = np.var(np.concatenate([YA, YB]), ddof=1)
    if VY <= 0:
        return names, np.zeros(k), np.zeros(k)

    S1 = np.zeros(k)
    ST = np.zeros(k)

    for j in range(k):
        XAB = XA.copy()
        XAB[:, j] = XB[:, j]
        YAB = evaluate(XAB)
        S1[j] = np.mean(YB * (YAB - YA)) / VY
        ST[j] = 0.5 * np.mean((YA - YAB) ** 2) / VY

    return names, np.clip(S1, 0, 1), np.clip(ST, 0, 1)


def save_sensitivity_plot(names, S1, ST, outdir: Path):
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

    x = np.arange(len(names))
    width = 0.38
    fig, ax = plt.subplots(figsize=(7.2, 4.6), constrained_layout=True)

    ax.bar(x - width / 2, S1, width, label=r"First-order $S_1$")
    ax.bar(x + width / 2, ST, width, label=r"Total-order $S_T$")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Sensitivity index")
    ax.set_title("Variance-based sensitivity for ALP trajectory summary (AUC)")
    ax.grid(True, axis="y", linestyle="--", alpha=0.35)
    ax.legend(frameon=False)

    pdf_path = outdir / "alp_sensitivity.pdf"
    png_path = outdir / "alp_sensitivity.png"
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.2)
    fig.savefig(png_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)
    return pdf_path, png_path


def main():
    p = argparse.ArgumentParser(description="Endpoint-anchored ALP accumulation model (1h) + sensitivity.")
    p.add_argument("--alp_baseline", type=float, required=True, help="ALP at osteogenic onset (day_start).")
    p.add_argument("--alp_endpoint", type=float, required=True, help="Measured ALP at day_measure.")
    p.add_argument("--tau_alp_hours", type=float, default=48.0, help="ALP rise timescale (hours).")
    p.add_argument("--day_start", type=float, default=3.0, help="Day of osteogenic supplementation.")
    p.add_argument("--day_measure", type=float, default=10.0, help="Day of ALP measurement.")
    p.add_argument("--dt_hours", type=float, default=1.0, help="Time step in hours (default 1).")
    p.add_argument("--taus_for_plot", type=float, nargs="*", default=None,
                   help="Optional tau values (hours) to overlay in plot.")
    p.add_argument("--outdir", type=str, default="unifiedalp_output", help="Output directory.")

    p.add_argument("--do_sensitivity", action="store_true", help="Run sensitivity analysis.")
    p.add_argument("--sens_N", type=int, default=3000, help="Base sample size for sensitivity (>=1000 recommended).")
    p.add_argument("--sens_seed", type=int, default=7, help="Random seed for sensitivity.")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # Main trajectory (hourly)
    t_hours, alp = alp_endpoint_anchored_profile(
        alp_baseline=args.alp_baseline,
        alp_endpoint=args.alp_endpoint,
        tau_alp_hours=args.tau_alp_hours,
        day_start=args.day_start,
        day_measure=args.day_measure,
        dt_hours=args.dt_hours,
    )

    pd.DataFrame({"hour": t_hours, "day": t_hours / 24.0, "ALP": alp}).to_csv(outdir / "alp_hourly.csv", index=False)
    days_i, alp_sampled = sample_at_integer_days(t_hours, alp)
    pd.DataFrame({"Day": days_i, "ALP_sampled": alp_sampled}).to_csv(outdir / "alp_daily_sampled.csv", index=False)

    # Overlay plot (hourly curves)
    taus = args.taus_for_plot if args.taus_for_plot else [args.tau_alp_hours]
    curves = {}
    for tau in taus:
        t_i, alp_i = alp_endpoint_anchored_profile(
            alp_baseline=args.alp_baseline,
            alp_endpoint=args.alp_endpoint,
            tau_alp_hours=float(tau),
            day_start=args.day_start,
            day_measure=args.day_measure,
            dt_hours=args.dt_hours,
        )
        curves[float(tau)] = (t_i, alp_i)

    save_alp_plot(curves, args.alp_baseline, args.alp_endpoint, args.day_start, args.day_measure, outdir)

    # Sensitivity (AUC metric)
    if args.do_sensitivity:
        rng = np.random.default_rng(args.sens_seed)
        bounds = {
            "alp_baseline": (0.5 * args.alp_baseline, 1.5 * args.alp_baseline),
            "alp_endpoint": (0.75 * args.alp_endpoint, 1.25 * args.alp_endpoint),
            "tau_alp_hours": (6.0, 144.0),
        }

        def metric(params):
            t_m, alp_m = alp_endpoint_anchored_profile(
                alp_baseline=params["alp_baseline"],
                alp_endpoint=params["alp_endpoint"],
                tau_alp_hours=params["tau_alp_hours"],
                day_start=args.day_start,
                day_measure=args.day_measure,
                dt_hours=args.dt_hours,
            )
            return alp_auc_between_days(t_m, alp_m, args.day_start, args.day_measure)

        names, S1, ST = sobol_sensitivity(metric, bounds, int(args.sens_N), rng)
        pd.DataFrame({"parameter": names, "S1": S1, "ST": ST}).to_csv(outdir / "alp_sensitivity.csv", index=False)
        save_sensitivity_plot(names, S1, ST, outdir)


if __name__ == "__main__":
    main()
