#!/usr/bin/env python3
"""
unifiedcollagenmodel.py  (FULL + JOURNAL-STYLE CLEAN PLOTS)

Figure conventions (Advanced Science / PLOS CB friendly):
- Basal (blue) and Osteo (orange) curves + minimal legend
- Endpoint dots (circles) at day 14 for BOTH curves (match line color)
- Protocol markers ABOVE axis (not on curves, not in legend):
    start day 3  = red square
    end   day 14 = black star
- Vertical dashed lines at day 3 and day 14
- No arrows, no callout text

Model:
  dC/dt = r0 + beta * alpha * A/(Km + A)

Calibration (reference, beta=1 for aa_endpoint):
  r0    = basal_endpoint / T
  alpha = (aa_endpoint - basal_endpoint) / S_ref
  S_ref = sum dt * A_ref/(Km_ref + A_ref)
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from SALib.sample import sobol as sobol_sample
from SALib.analyze import sobol as sobol_analyze


# ---------------------------
# Fixed timeline
# ---------------------------
DAY_START = 3.0
DAY_MEASURE = 14.0
DT_H = 1.0

# Okabe–Ito (colorblind-safe, common in journals)
BASAL_COLOR = "#0072B2"   # blue
OSTEO_COLOR = "#D55E00"   # vermillion/orange


# ---------------------------
# Helpers
# ---------------------------
def aa_profile_hourly(
    AA0_ugml: float,
    half_life_h: float,
    media_interval_h: float
) -> tuple[np.ndarray, np.ndarray]:
    """AA(t) in µg/mL from day 0→14 (inclusive), AA=0 before day 3, pulses + exponential decay after."""
    if AA0_ugml < 0:
        raise ValueError("AA0_ugml must be >= 0")
    if half_life_h <= 0:
        raise ValueError("half_life_h must be > 0")
    if media_interval_h <= 0:
        raise ValueError("media_interval_h must be > 0")

    k = np.log(2.0) / float(half_life_h)

    total_steps = int(round((DAY_MEASURE * 24.0) / DT_H))
    t_hours = np.arange(0, total_steps + 1) * DT_H

    A = np.zeros_like(t_hours, dtype=float)
    start_h = int(round((DAY_START * 24.0) / DT_H))
    interval_steps = max(1, int(round(media_interval_h / DT_H)))

    if start_h <= total_steps:
        idx = np.arange(start_h, total_steps + 1)
        rel = idx - start_h
        since_change_steps = rel % interval_steps
        since_change_h = since_change_steps * DT_H
        A[idx] = AA0_ugml * np.exp(-k * since_change_h)
        A[start_h] = AA0_ugml

    return t_hours, A


def mm(A: np.ndarray, Km: float) -> np.ndarray:
    Km = float(Km)
    if Km <= 0:
        raise ValueError("Km must be > 0")
    A = np.asarray(A, dtype=float)
    return np.clip(A / (Km + A), 0.0, 1.0)


def endpoint_S(A_hours: np.ndarray, Km: float) -> float:
    """S = sum dt * A/(Km+A) using Euler sat[i] for step i->i+1."""
    sat = mm(A_hours, Km)
    return float(np.sum(sat[:-1]) * DT_H)


def simulate_C_hourly(
    t_hours: np.ndarray,
    A_hours: np.ndarray,
    r0: float,
    alpha: float,
    beta: float,
    Km: float
) -> np.ndarray:
    """Explicit Euler with dt=1h."""
    sat = mm(A_hours, Km)
    C = np.zeros_like(t_hours, dtype=float)
    for i in range(len(t_hours) - 1):
        C[i + 1] = C[i] + (r0 + beta * alpha * sat[i]) * DT_H
    return C


# ---------------------------
# Plot styling (journal-ish)
# ---------------------------
def set_journal_style():
    plt.rcParams.update(
        {
            # Typography
            "font.size": 10.5,
            "axes.labelsize": 11,
            "axes.titlesize": 12,
            "legend.fontsize": 10,
            "font.family": "sans-serif",
            # Lines
            "lines.linewidth": 2.2,
            "axes.linewidth": 1.2,
            # Ticks
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.size": 4,
            "ytick.major.size": 4,
            "xtick.major.width": 1.0,
            "ytick.major.width": 1.0,
            "xtick.minor.size": 2.5,
            "ytick.minor.size": 2.5,
            # Spines
            "axes.spines.top": False,
            "axes.spines.right": False,
            # Output
            "figure.dpi": 150,
            "savefig.dpi": 300,
        }
    )


def format_axes(ax):
    ax.grid(True, linestyle="--", alpha=0.18)
    ax.set_xlim(0, DAY_MEASURE + 0.4)
    ax.minorticks_on()


def protocol_lines(ax):
    ax.axvline(DAY_START, linestyle="--", linewidth=1.2, alpha=0.55)
    ax.axvline(DAY_MEASURE, linestyle="--", linewidth=1.2, alpha=0.55)


def protocol_markers_above_axis(ax):
    """
    Put protocol markers ABOVE the plotting area.
    Uses transform: x=data coords, y=axes fraction.
    """
    trans = ax.get_xaxis_transform()
    y_above = 0.02
    ax.plot(
        DAY_START, y_above,
        marker="s", markersize=9, color="red",
        linestyle="None", transform=trans, clip_on=False, zorder=10
    )
    ax.plot(
        DAY_MEASURE, y_above,
        marker="*", markersize=14, color="black",
        linestyle="None", transform=trans, clip_on=False, zorder=10
    )


def endpoint_dots(ax, y_end_basal: float, y_end_osteo: float):
    """Circle dots on the actual day-14 endpoints (match line colors)."""
    ax.plot(
        DAY_MEASURE, y_end_basal,
        marker="o", markersize=6.0, color=BASAL_COLOR,
        linestyle="None", zorder=8
    )
    ax.plot(
        DAY_MEASURE, y_end_osteo,
        marker="o", markersize=6.0, color=OSTEO_COLOR,
        linestyle="None", zorder=8
    )


def save_fig(fig, outdir: Path, stem: str):
    fig.savefig(outdir / f"{stem}.png", bbox_inches="tight", pad_inches=0.18)
    fig.savefig(outdir / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.18)


# ---------------------------
# Plotting
# ---------------------------
def plot_aa(t_hours: np.ndarray, A: np.ndarray, outdir: Path):
    set_journal_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.3), constrained_layout=True)

    ax.plot(t_hours / 24.0, A, label="AA(t)")
    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel(r"AA ($\mu$g/mL)")
    ax.set_title("AA exposure profile")

    format_axes(ax)
    ax.set_ylim(bottom=0)
    protocol_lines(ax)
    protocol_markers_above_axis(ax)

    ax.legend(frameon=False, loc="upper right")
    save_fig(fig, outdir, "aa_profile")
    plt.close(fig)


def plot_basal_vs_osteo(t_hours: np.ndarray, C_basal: np.ndarray, C_osteo: np.ndarray, outdir: Path):
    set_journal_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)

    t_days = t_hours / 24.0
    ax.plot(t_days, C_basal, color=BASAL_COLOR, label="Basal")
    ax.plot(t_days, C_osteo, color=OSTEO_COLOR, label="Osteo")

    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel("Collagen signal (a.u.)")
    ax.set_title("Collagen accumulation")

    format_axes(ax)
    protocol_lines(ax)

    endpoint_dots(ax, float(C_basal[-1]), float(C_osteo[-1]))
    protocol_markers_above_axis(ax)

    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, outdir, "collagen_basal_vs_osteo")
    plt.close(fig)


def plot_delta(t_hours: np.ndarray, delta: np.ndarray, outdir: Path):
    set_journal_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)

    t_days = t_hours / 24.0
    ax.plot(t_days, delta, label=r"$\Delta C(t)$")

    ax.set_xlabel("Day (post-seeding)")
    ax.set_ylabel(r"$\Delta$Collagen (osteo − basal)")
    ax.set_title("AA-driven collagen increment")

    format_axes(ax)
    protocol_lines(ax)

    # endpoint dot
    ax.plot(
        DAY_MEASURE, float(delta[-1]),
        marker="o", markersize=6.5, color="black",
        linestyle="None", zorder=8
    )
    protocol_markers_above_axis(ax)

    ax.legend(frameon=False, loc="upper left")
    save_fig(fig, outdir, "collagen_delta_AA")
    plt.close(fig)


def plot_sobol_bar(Si: dict, names: list[str], outdir: Path):
    set_journal_style()
    S1 = np.clip(np.array(Si["S1"], dtype=float), 0, 1)
    ST = np.clip(np.array(Si["ST"], dtype=float), 0, 1)

    x = np.arange(len(names))
    w = 0.38
    fig, ax = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)

    ax.bar(x - w / 2, S1, w, label=r"First-order $S_1$")
    ax.bar(x + w / 2, ST, w, label=r"Total-order $S_T$")

    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Sensitivity index")
    ax.set_title(f"Sobol sensitivity of endpoint (day {DAY_MEASURE:g})")

    ax.grid(True, axis="y", linestyle="--", alpha=0.18)
    ax.legend(frameon=False)

    save_fig(fig, outdir, "sobol_barplot")
    plt.close(fig)


# ---------------------------
# Main
# ---------------------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--basal_endpoint", type=float, required=True)
    ap.add_argument("--aa_endpoint", type=float, required=True)  # used to fit alpha at beta=1
    ap.add_argument("--AA0_ugml", type=float, required=True)
    ap.add_argument("--beta", type=float, required=True)

    ap.add_argument("--Km_ref", type=float, required=True)
    ap.add_argument("--media_interval_ref", type=float, default=48.0)
    ap.add_argument("--half_life_ref", type=float, default=3.0)

    ap.add_argument("--Km_plot", type=float, default=None)
    ap.add_argument("--media_interval_plot", type=float, default=None)
    ap.add_argument("--half_life_plot", type=float, default=None)

    ap.add_argument("--do_sensitivity", action="store_true")
    ap.add_argument("--sens_N", type=int, default=4096, help="Prefer 2^n (2048/4096/8192/...)")
    ap.add_argument("--sens_seed", type=int, default=7)
    ap.add_argument("--Km_bounds", type=float, nargs=2, default=[5.0, 50.0])
    ap.add_argument("--media_bounds", type=float, nargs=2, default=[24.0, 72.0])
    ap.add_argument("--half_life_bounds", type=float, nargs=2, default=[1.5, 6.0])

    ap.add_argument("--outdir", type=str, default="collagen_out")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # timebase
    T_hours = DAY_MEASURE * 24.0
    t_hours = np.arange(0, int(round(T_hours / DT_H)) + 1) * DT_H

    # calibration
    r0 = float(args.basal_endpoint) / T_hours
    _, A_ref = aa_profile_hourly(args.AA0_ugml, args.half_life_ref, args.media_interval_ref)
    S_ref = endpoint_S(A_ref, args.Km_ref)
    alpha = (float(args.aa_endpoint) - float(args.basal_endpoint)) / S_ref if S_ref > 0 else 0.0

    # plot scenario overrides
    Km_plot = float(args.Km_plot) if args.Km_plot is not None else float(args.Km_ref)
    mi_plot = float(args.media_interval_plot) if args.media_interval_plot is not None else float(args.media_interval_ref)
    hl_plot = float(args.half_life_plot) if args.half_life_plot is not None else float(args.half_life_ref)

    # simulate
    t_AA, A_plot = aa_profile_hourly(args.AA0_ugml, hl_plot, mi_plot)
    C_basal = r0 * t_hours
    C_osteo = simulate_C_hourly(t_hours, A_plot, r0=r0, alpha=alpha, beta=args.beta, Km=Km_plot)
    delta = C_osteo - C_basal

    # plots
    plot_aa(t_AA, A_plot, outdir)
    plot_basal_vs_osteo(t_hours, C_basal, C_osteo, outdir)
    plot_delta(t_hours, delta, outdir)

    # save CSV
    pd.DataFrame(
        {
            "hour": t_hours,
            "day": t_hours / 24.0,
            "AA_ugml": A_plot,
            "C_basal": C_basal,
            "C_osteo": C_osteo,
            "delta_AA": delta,
        }
    ).to_csv(outdir / "collagen_timeseries.csv", index=False)

    # analysis summary (numbers you cite in text)
    summary = [
        f"day_start={DAY_START}, day_measure={DAY_MEASURE}, dt_h={DT_H}",
        f"basal_endpoint(day14)={args.basal_endpoint}",
        f"aa_endpoint(day14, used to fit alpha at beta=1)={args.aa_endpoint}",
        f"AA0_ugml={args.AA0_ugml}, beta={args.beta}",
        "",
        "Calibration (reference):",
        f"Km_ref={args.Km_ref}, media_interval_ref={args.media_interval_ref}, half_life_ref={args.half_life_ref}",
        f"r0_per_h={r0:.6g}",
        f"alpha_per_h={alpha:.6g}",
        f"S_ref={S_ref:.6g}",
        "",
        "Plot scenario:",
        f"Km_plot={Km_plot}, media_interval_plot={mi_plot}, half_life_plot={hl_plot}",
        f"Predicted C_basal(day14)={C_basal[-1]:.6g}",
        f"Predicted C_osteo(day14)={C_osteo[-1]:.6g}",
        f"Predicted Δ(day14)={delta[-1]:.6g}",
    ]
    (outdir / "fit_summary.txt").write_text("\n".join(summary) + "\n", encoding="utf-8")

    # sensitivity (analytic endpoint)
    if args.do_sensitivity:
        problem = {
            "num_vars": 3,
            "names": ["Km_ugml", "media_interval_h", "half_life_h"],
            "bounds": [
                [float(args.Km_bounds[0]), float(args.Km_bounds[1])],
                [float(args.media_bounds[0]), float(args.media_bounds[1])],
                [float(args.half_life_bounds[0]), float(args.half_life_bounds[1])],
            ],
        }

        np.random.seed(int(args.sens_seed))
        X = sobol_sample.sample(problem, int(args.sens_N), calc_second_order=False)

        Y = np.zeros(X.shape[0], dtype=float)
        for i in range(X.shape[0]):
            Km, mi, hl = float(X[i, 0]), float(X[i, 1]), float(X[i, 2])
            _, A_h = aa_profile_hourly(args.AA0_ugml, hl, mi)
            S = endpoint_S(A_h, Km)
            Y[i] = r0 * T_hours + args.beta * alpha * S

        Si = sobol_analyze.analyze(problem, Y, calc_second_order=False, print_to_console=False)

        pd.DataFrame(
            {
                "parameter": problem["names"],
                "S1": Si["S1"],
                "S1_conf": Si["S1_conf"],
                "ST": Si["ST"],
                "ST_conf": Si["ST_conf"],
            }
        ).to_csv(outdir / "sobol_indices.csv", index=False)

        plot_sobol_bar(Si, problem["names"], outdir)

    print(f"[✓] Wrote outputs to: {outdir.resolve()}")


if __name__ == "__main__":
    main()



"""

python unifiedcollagenmodel.py   --basal_endpoint 2740 --aa_endpoint 2880   --AA0_ugml 50 --beta 1.0   --Km_ref 15 --media_interval_ref 48 --half_life_ref 3   --do_sensitivity --sens_N 4096   --Km_bounds 5 50 --media_bounds 24 72 --half_life_bounds 1.5 6   --outdir collagen_out

"""