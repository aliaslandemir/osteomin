#!/usr/bin/env python3
"""
unified_matrix_mineral.py — FINAL CLEAN

Collagen model (alpha removed):
  dC/dt = r0 + s_mod * A(t)/(Km + A(t))

- Km is free (effective AA sensitivity).
- s_mod is a dimensionless modulator capturing DEX/MK4 effects (etc).
- Endpoint visibility fixed (no clipping at day 14).
- Adds ★ marker at day 3 for osteogenic induction (AA start).
- Km sweep made visible without inset: 2-panel figure:
    (top) time courses 0–14 d
    (bottom) endpoint vs Km (+ observed line)

Outputs:
- collagen_timecourses.(png/pdf)
- collagen_km_sweep.(png/pdf)
- mineral_timecourses.(png/pdf)
- mineral_n_sweep.(png/pdf)
- candidate_selection_explainer.(png/pdf)
- sobol_endpoint_collagen.(png/pdf)  (if --sobol)
- sobol_endpoint_mineral.(png/pdf)   (if --sobol)
- matrix_mineral_timeseries.csv
- matrix_mineral_outputs.npz



python unified_matrix_mineral.py \
  --collagen_C1 2740 --collagen_C2 2880 \
  --ars_C1 0.65 --ars_C2 50.87 \
  --ars_units percent \
  --sAA_C1 0 --sAA_C2 1 \
  --zeta_C1 1 --zeta_C2 1 \
  --Km_vals 5 10 15 30 \
  --n_vals 1 2 3 4 \
  --sobol \
  --outdir out_bas_v_osteo



"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DAY_START_AA = 3.0
DAY_COLLAGEN = 14.0
DAY_MINERAL = 28.0


# -----------------------------
# Plot style (clean / Nature-ish)
# -----------------------------
def _set_style():
    """PLOS Computational Biology–friendly defaults.

    Notes:
      - Sans-serif fonts (prefers Arial/Helvetica; falls back to DejaVu Sans).
      - Embedded fonts in PDF (fonttype 42).
      - High-resolution exports for line art (600 dpi PNG/TIFF).
      - Colorblind-safe default color cycle (Okabe–Ito).
    """
    okabe_ito = [
        "#0072B2",  # blue
        "#D55E00",  # vermillion
        "#009E73",  # bluish green
        "#CC79A7",  # reddish purple
        "#56B4E9",  # sky blue
        "#E69F00",  # orange
        "#F0E442",  # yellow
        "#000000",  # black
    ]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9.0,
            "legend.fontsize": 8.0,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "axes.linewidth": 0.8,
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 3.2,
            "ytick.major.size": 3.2,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "axes.prop_cycle": plt.cycler(color=okabe_ito),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_axes(ax, grid: bool = False):
    ax.tick_params(which="both", top=False, right=False)
    if grid:
        ax.grid(True, alpha=0.18, linewidth=0.6)
    else:
        ax.grid(False)


def _save(fig, outdir: Path, name: str):
    """Export as PNG + PDF + TIFF (LZW), with tight bounding boxes."""
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", pad_inches=0.12)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12)
    # TIFF is frequently requested by journals; keep it lossless + high dpi.
    fig.savefig(
        outdir / f"{name}.tif",
        bbox_inches="tight",
        pad_inches=0.12,
        dpi=600,
        pil_kwargs={"compression": "tiff_lzw"},
    )




# -----------------------------
# AA + response
# -----------------------------
def mm_saturation(A: np.ndarray, Km: float) -> np.ndarray:
    Km = float(Km)
    if Km <= 0:
        raise ValueError("Km must be > 0")
    A = np.asarray(A, dtype=float)
    return np.clip(A / (Km + A), 0.0, 1.0)


def aa_profile(
    t_days: np.ndarray,
    AA0_ugml: float,
    half_life_h: float,
    media_interval_h: float,
    day_start: float,
) -> np.ndarray:
    """Pulsed AA with exponential decay between medium changes."""
    if AA0_ugml < 0:
        raise ValueError("AA0_ugml must be >= 0")
    if half_life_h <= 0 or media_interval_h <= 0:
        raise ValueError("half_life_h and media_interval_h must be > 0")

    k = np.log(2.0) / float(half_life_h)  # 1/h
    A = np.zeros_like(t_days, dtype=float)

    t_h = t_days * 24.0
    start_h = float(day_start) * 24.0
    interval_h = float(media_interval_h)

    idx = np.where(t_h >= start_h)[0]
    if idx.size == 0:
        return A

    rel_h = t_h[idx] - start_h
    since_change = np.mod(rel_h, interval_h)
    A[idx] = AA0_ugml * np.exp(-k * since_change)

    if np.isclose(t_h[idx[0]], start_h):
        A[idx[0]] = AA0_ugml
    return A


# -----------------------------
# Collagen model (alpha removed)
# -----------------------------
def simulate_collagen_hold_after_14(
    t_days: np.ndarray,
    A: np.ndarray,
    r0: float,
    s_mod: float,
    Km: float,
) -> np.ndarray:
    """Euler integrate to day 14; hold constant afterward."""
    dt = float(np.median(np.diff(t_days)))
    sat = mm_saturation(A, Km)

    C = np.zeros_like(t_days, dtype=float)
    for i in range(len(t_days) - 1):
        t = float(t_days[i])
        if t < DAY_COLLAGEN - 1e-12:
            C[i + 1] = C[i] + (float(r0) + float(s_mod) * sat[i]) * dt
        else:
            C[i + 1] = C[i]
    return C


# -----------------------------
# Mineral model
# -----------------------------
def mineral_hill(t_days: np.ndarray, t_start: float, k: float, n: float, zeta: float) -> np.ndarray:
    t = np.asarray(t_days, dtype=float)
    H = np.zeros_like(t, dtype=float)
    if k <= 0 or zeta <= 0:
        return H
    tau = 1.0 / float(k)
    m = t >= float(t_start)
    x = t[m] - float(t_start)
    num = np.power(x, float(n))
    den = np.power(tau, float(n)) + num
    H[m] = float(zeta) * (num / den)
    return np.clip(H, 0.0, float(zeta))


def solve_k_from_endpoint(H_end: float, T_eff: float, n: float, zeta: float) -> float:
    H_end = float(H_end)
    zeta = float(zeta)
    if H_end <= 0 or T_eff <= 0 or zeta <= 0:
        return 0.0
    if H_end >= zeta:
        H_end = zeta * (1.0 - 1e-8)
    ratio = (zeta / H_end) - 1.0
    if ratio <= 0:
        return 0.0
    tau = float(T_eff) * (ratio ** (1.0 / float(n)))
    return 0.0 if tau <= 0 else 1.0 / tau


def ars_to_fraction(x: float, units: str) -> tuple[float, float, str]:
    x = float(x)
    if units == "percent":
        return x / 100.0, 100.0, "ARS-positive area (% of field)"
    if units == "fraction":
        return x, 1.0, "ARS-positive area (fraction)"
    raise ValueError("ars_units must be 'percent' or 'fraction'")


# -----------------------------
# Sobol helpers
# -----------------------------
def _quantize_to_4(u: np.ndarray, vals4: list[float]) -> np.ndarray:
    vals = np.asarray(vals4, dtype=float)
    u = np.clip(u, 0.0, 1.0)
    idx = np.minimum((u * 4.0).astype(int), 3)
    return vals[idx]


def _barplot_horizontal(names, S1, ST, title, outdir: Path, fname: str):
    _set_style()
    names = list(names)
    S1 = np.asarray(S1, dtype=float)
    ST = np.asarray(ST, dtype=float)

    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(6.6, 0.42 * len(names) + 1.0), constrained_layout=True)
    ax.barh(y + 0.18, S1[::-1], height=0.34, label="First-order S1")
    ax.barh(y - 0.18, ST[::-1], height=0.34, label="Total-order ST")
    ax.set_yticks(y, names[::-1])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Sensitivity index")
    ax.set_title(title)
    _format_axes(ax, grid=False)
    ax.legend(frameon=False, loc="lower right")
    _save(fig, outdir, fname)
    plt.close(fig)


def sobol_endpoint_collagen(
    *,
    outdir: Path,
    t_days: np.ndarray,
    AA0_ugml: float,
    day_start_AA: float,
    r0: float,
    Km_vals: list[float],
    s_mod2_vals: list[float],
    half_life_vals: list[float],
    media_interval_vals: list[float],
    baseN: int,
):
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    names = ["Km", "s_mod2", "half_life_h", "media_interval_h"]
    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names)}
    X = saltelli.sample(problem, N=int(baseN), calc_second_order=True)

    Km_s = _quantize_to_4(X[:, 0], Km_vals)
    s2_s = _quantize_to_4(X[:, 1], s_mod2_vals)
    hl_s = _quantize_to_4(X[:, 2], half_life_vals)
    mi_s = _quantize_to_4(X[:, 3], media_interval_vals)

    Y = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        A_i = aa_profile(t_days, AA0_ugml, float(hl_s[i]), float(mi_s[i]), day_start_AA)
        C2 = simulate_collagen_hold_after_14(t_days, A_i, r0=float(r0), s_mod=float(s2_s[i]), Km=float(Km_s[i]))
        Y[i] = float(np.interp(DAY_COLLAGEN, t_days, C2))

    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    _barplot_horizontal(names, Si["S1"], Si["ST"], "Sobol sensitivity: Collagen endpoint (day 14)", outdir, "sobol_endpoint_collagen")
    return {"names": names, "S1": Si["S1"].tolist(), "ST": Si["ST"].tolist(), "S2": Si["S2"].tolist()}


def sobol_endpoint_mineral(
    *,
    outdir: Path,
    t_days: np.ndarray,
    n_vals: list[float],
    k_vals: list[float],
    zeta2_vals: list[float],
    t_start_vals: list[float],
    baseN: int,
):
    from SALib.sample import saltelli
    from SALib.analyze import sobol

    names = ["n", "k_per_day", "zeta2", "t_start_day"]
    problem = {"num_vars": len(names), "names": names, "bounds": [[0.0, 1.0]] * len(names)}
    X = saltelli.sample(problem, N=int(baseN), calc_second_order=True)

    n_s = _quantize_to_4(X[:, 0], n_vals)
    k_s = _quantize_to_4(X[:, 1], k_vals)
    z_s = _quantize_to_4(X[:, 2], zeta2_vals)
    t0_s = _quantize_to_4(X[:, 3], t_start_vals)

    Y = np.zeros(X.shape[0], dtype=float)
    for i in range(X.shape[0]):
        H2 = mineral_hill(t_days, t_start=float(t0_s[i]), k=float(k_s[i]), n=float(n_s[i]), zeta=float(z_s[i]))
        Y[i] = float(np.interp(DAY_MINERAL, t_days, H2))

    Si = sobol.analyze(problem, Y, calc_second_order=True, print_to_console=False)
    _barplot_horizontal(names, Si["S1"], Si["ST"], "Sobol sensitivity: Mineral endpoint (day 28)", outdir, "sobol_endpoint_mineral")
    return {"names": names, "S1": Si["S1"].tolist(), "ST": Si["ST"].tolist(), "S2": Si["S2"].tolist()}


# -----------------------------
# Plots (fixed visibility + day3 star)
# -----------------------------
def plot_collagen_timecourses_0_14(
    outdir: Path,
    t_days: np.ndarray,
    A: np.ndarray,
    r0: float,
    collagen_C1_obs: float,
    collagen_C2_obs: float,
    s_mod_C1: float,
    s_mod_C2: float,
    Km_star: float,
    day_induction: float,
):
    _set_style()
    m14 = t_days <= DAY_COLLAGEN + 1e-12
    t = t_days[m14]

    C1 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C1, Km=Km_star)
    C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C2, Km=Km_star)

    C1_end = float(np.interp(DAY_COLLAGEN, t_days, C1))
    C2_end = float(np.interp(DAY_COLLAGEN, t_days, C2))
    C2_ind = float(np.interp(day_induction, t_days, C2))

    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    ax.plot(t, C1[m14], label="C1 basal")
    ax.plot(t, C2[m14], label="C2 osteogenic")

    # Reference lines
    ax.axvline(day_induction, ls="--", lw=1.0, alpha=0.8)
    ax.axvline(DAY_COLLAGEN, ls=":", lw=1.0, alpha=0.9)

    # ★ induction marker on osteogenic curve (day 3)
    ax.scatter([day_induction], [C2_ind], marker="*", s=180, edgecolor="black", linewidth=0.8, zorder=8, label="Induction (day 3)")

    # Endpoints: use x-jitter + extend xlim so nothing clips
    xj1, xj2 = DAY_COLLAGEN - 0.08, DAY_COLLAGEN + 0.08

    # observed endpoints (circles)
    ax.scatter([xj1], [collagen_C1_obs], s=150, marker="o", edgecolor="black", linewidth=1.1, zorder=9)
    ax.scatter([xj2], [collagen_C2_obs], s=150, marker="o", edgecolor="black", linewidth=1.1, zorder=9)

    # model endpoints (diamonds)
    ax.scatter([xj1], [C1_end], s=120, marker="D", edgecolor="black", linewidth=1.0, zorder=8)
    ax.scatter([xj2], [C2_end], s=120, marker="D", edgecolor="black", linewidth=1.0, zorder=8)

    # labels (small)
    ax.annotate(f"obs {collagen_C1_obs:.0f}", (xj1, collagen_C1_obs), xytext=(-6, -14),
                textcoords="offset points", fontsize=8, ha="right", va="top")
    ax.annotate(f"obs {collagen_C2_obs:.0f}", (xj2, collagen_C2_obs), xytext=(6, -14),
                textcoords="offset points", fontsize=8, ha="left", va="top")
    ax.annotate(f"mdl {C1_end:.0f}", (xj1, C1_end), xytext=(-6, 10),
                textcoords="offset points", fontsize=8, ha="right", va="bottom")
    ax.annotate(f"mdl {C2_end:.0f}", (xj2, C2_end), xytext=(6, 10),
                textcoords="offset points", fontsize=8, ha="left", va="bottom")

    ax.set_xlim(0.0, DAY_COLLAGEN + 0.6)  # prevents clipping
    ax.set_xlabel("Day")
    ax.set_ylabel("Collagen proxy (a.u.)")
    ax.set_title(f"Collagen (0–14 d; Km*={Km_star:g})")
    _format_axes(ax, grid=False)

    ax.legend(frameon=False, loc="upper left")
    ax.text(0.02, 0.02, "○ observed (day 14)\n◇ model (day 14)\n★ induction (day 3)",
            transform=ax.transAxes, fontsize=8, va="bottom", ha="left")

    _save(fig, outdir, "collagen_timecourses")
    plt.close(fig)


def plot_km_sweep_collagen(
    outdir: Path,
    t_days: np.ndarray,
    A: np.ndarray,
    r0: float,
    collagen_C2_obs: float,
    Km_vals: list[float],
    s_mod_C2: float,
    day_induction: float,
):
    """
    Two-panel Km sweep (no inset):
      Top: C2(t) 0–14 days for each Km
      Bottom: C2(day14) vs Km (points) + observed line
    """
    _set_style()
    m14 = t_days <= DAY_COLLAGEN + 1e-12
    t = t_days[m14]

    trajs = []
    ends = []
    inds = []
    for Km in Km_vals:
        C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C2, Km=float(Km))
        trajs.append(C2[m14])
        ends.append(float(np.interp(DAY_COLLAGEN, t_days, C2)))
        inds.append(float(np.interp(day_induction, t_days, C2)))

    ends = np.asarray(ends, dtype=float)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(3.7, 4.4),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.1]}
    )

    # Top panel: time courses
    for Km, y in zip(Km_vals, trajs):
        ax1.plot(t, y, label=f"Km={float(Km):g}")

    ax1.axvline(day_induction, ls="--", lw=1.0, alpha=0.8)
    ax1.axvline(DAY_COLLAGEN, ls=":", lw=1.0, alpha=0.9)
    ax1.scatter([DAY_COLLAGEN + 0.10], [collagen_C2_obs], s=140, marker="o",
                edgecolor="black", linewidth=1.1, zorder=10, label="Observed (day 14)")

    # show induction star for the chosen Km curve? (use Km closest to median just for marker location)
    # but keep it simple: mark induction on observed panel line using one star at mean induction height
    ax1.scatter([day_induction], [np.mean(inds)], marker="*", s=160,
                edgecolor="black", linewidth=0.8, zorder=10)

    ax1.set_xlim(0.0, DAY_COLLAGEN + 0.6)  # prevent clipping
    ax1.set_ylabel("Collagen proxy (a.u.)")
    ax1.set_title("Collagen: Km sweep (C2; no α)")
    _format_axes(ax1, grid=False)
    ax1.legend(frameon=False, ncol=2, loc="upper left")

    # Bottom panel: endpoint vs Km
    ax2.plot(Km_vals, ends, marker="o", markersize=6, markeredgecolor="black", linewidth=2.0)
    ax2.axhline(collagen_C2_obs, ls=":", lw=1.2)
    ax2.set_xlabel("Km candidate")
    ax2.set_ylabel("C2(day 14)")
    ax2.set_title("Endpoint vs Km (line = observed)")
    _format_axes(ax2, grid=False)

    _save(fig, outdir, "collagen_km_sweep")
    plt.close(fig)


def plot_mineral_timecourses(
    outdir: Path,
    t_days: np.ndarray,
    H1: np.ndarray,
    H2: np.ndarray,
    plot_scale: float,
    ylabel: str,
    t_start: float,
    n_star: float,
    day_induction: float,
):
    _set_style()
    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)
    ax.plot(t_days, H1 * plot_scale, label="C1 basal")
    ax.plot(t_days, H2 * plot_scale, label="C2 osteogenic")

    ax.axvline(day_induction, ls="--", lw=1.0, alpha=0.8)
    ax.axvline(float(t_start), ls=":", lw=1.0, alpha=0.9)
    ax.axvline(DAY_MINERAL, ls=":", lw=1.0, alpha=0.9)

    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Mineral (n*={n_star:g}; endpoint-derived k)")
    _format_axes(ax, grid=False)
    ax.legend(frameon=False, loc="upper left")

    _save(fig, outdir, "mineral_timecourses")
    plt.close(fig)


def plot_n_sweep_mineral(
    outdir: Path,
    t_days: np.ndarray,
    ars_C1: float,
    ars_C2: float,
    ars_units: str,
    n_vals: list[float],
    t_start: float,
    zeta_C2: float,
    day_induction: float,
):
    _set_style()
    _, plot_scale, ylabel = ars_to_fraction(float(ars_C1), ars_units)
    H2_end, _, _ = ars_to_fraction(float(ars_C2), ars_units)

    fig, ax = plt.subplots(figsize=(3.6, 2.7), constrained_layout=True)

    for n in n_vals:
        T_eff = DAY_MINERAL - float(t_start)
        k2 = solve_k_from_endpoint(H2_end, T_eff, float(n), float(zeta_C2))
        H2 = mineral_hill(t_days, t_start=float(t_start), k=float(k2), n=float(n), zeta=float(zeta_C2))
        ax.plot(t_days, H2 * plot_scale, label=f"n={float(n):g}")

    ax.axvline(day_induction, ls="--", lw=1.0, alpha=0.8)
    ax.axvline(float(t_start), ls=":", lw=1.0, alpha=0.9)
    ax.axvline(DAY_MINERAL, ls=":", lw=1.0, alpha=0.9)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title("Mineral: n sweep (C2; endpoint matched at day 28)")
    _format_axes(ax, grid=False)
    ax.legend(frameon=False, ncol=2, loc="upper left")

    _save(fig, outdir, "mineral_n_sweep")
    plt.close(fig)


def plot_candidate_selection_explainer(
    *,
    outdir: Path,
    t_days: np.ndarray,
    A: np.ndarray,
    r0: float,
    collagen_C2_obs: float,
    Km_vals: list[float],
    s_mod_C2: float,
    Km_star: float,
    ars_C2: float,
    ars_units: str,
    n_vals: list[float],
    t_start: float,
    zeta_C2: float,
    n_star: float,
):
    _set_style()

    # Collagen endpoint vs Km
    C2_end = []
    for Km in Km_vals:
        C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C2, Km=float(Km))
        C2_end.append(float(np.interp(DAY_COLLAGEN, t_days, C2)))
    C2_end = np.asarray(C2_end, dtype=float)
    err = np.abs(C2_end - float(collagen_C2_obs))

    # Mineral setup
    H2_end, plot_scale, ylabel = ars_to_fraction(float(ars_C2), ars_units)

    k_req = []
    H2_trajs = []
    for n in n_vals:
        T_eff = DAY_MINERAL - float(t_start)
        k2 = solve_k_from_endpoint(H2_end, T_eff, float(n), float(zeta_C2))
        k_req.append(k2)
        H2_trajs.append(mineral_hill(t_days, t_start=float(t_start), k=float(k2), n=float(n), zeta=float(zeta_C2)))
    k_req = np.asarray(k_req, dtype=float)

    fig, axes = plt.subplots(2, 2, figsize=(6.8, 4.8), constrained_layout=True)

    # (A) collagen misfit
    ax = axes[0, 0]
    x = np.arange(len(Km_vals))
    ax.bar(x, err)
    ax.set_xticks(x, [f"{v:g}" for v in Km_vals])
    if Km_star in Km_vals:
        idx = Km_vals.index(Km_star)
        ax.scatter([idx], [err[idx]], s=90, marker="*", zorder=6)
    ax.set_xlabel("Km candidate")
    ax.set_ylabel(r"$|C_2(14)-C_{obs}|$")
    ax.set_title("Collagen: endpoint misfit")
    _format_axes(ax, grid=False)

    # (B) mineral trajectories
    ax = axes[0, 1]
    for n, k2, H2 in zip(n_vals, k_req, H2_trajs):
        ax.plot(t_days, H2 * plot_scale, label=f"n={n:g} (k={k2:.3g}/d)")
    ax.scatter([DAY_MINERAL], [H2_end * plot_scale], s=18, zorder=5)
    ax.axvline(float(t_start), ls=":", lw=1.0, alpha=0.9)
    ax.axvline(DAY_MINERAL, ls=":", lw=1.0, alpha=0.9)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title("Mineral: n changes dynamics, endpoint fixed")
    _format_axes(ax, grid=False)
    ax.legend(frameon=False, fontsize=7, loc="upper left")

    # (C) collagen endpoint vs Km
    ax = axes[1, 0]
    ax.plot(Km_vals, C2_end, marker="o", markersize=6, markeredgecolor="black", linewidth=2.0)
    ax.axhline(collagen_C2_obs, ls=":", lw=1.2)
    ax.set_xlabel("Km candidate")
    ax.set_ylabel("C2(day 14) model")
    ax.set_title("Collagen: endpoint vs Km")
    _format_axes(ax, grid=False)

    # (D) k vs n
    ax = axes[1, 1]
    x = np.arange(len(n_vals))
    ax.bar(x, k_req)
    ax.set_xticks(x, [f"{v:g}" for v in n_vals])
    if n_star in n_vals:
        nidx = n_vals.index(n_star)
        ax.scatter([nidx], [k_req[nidx]], s=90, marker="*", zorder=6)
    ax.set_xlabel("n candidate")
    ax.set_ylabel("Implied k (1/day)")
    ax.set_title("k required to match H2(day 28)")
    _format_axes(ax, grid=False)

    fig.suptitle(f"Candidate comparison (Km*={Km_star:g}, n*={n_star:g})", y=1.01)
    _save(fig, outdir, "candidate_selection_explainer")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()

    # endpoints
    ap.add_argument("--collagen_C1", type=float, required=True)
    ap.add_argument("--collagen_C2", type=float, required=True)
    ap.add_argument("--ars_C1", type=float, required=True)
    ap.add_argument("--ars_C2", type=float, required=True)
    ap.add_argument("--ars_units", choices=["percent", "fraction"], default="percent")

    # AA protocol
    ap.add_argument("--AA0_ugml", type=float, default=50.0)
    ap.add_argument("--half_life_h", type=float, default=3.0)
    ap.add_argument("--media_interval_h", type=float, default=48.0)
    ap.add_argument("--day_start_AA", type=float, default=DAY_START_AA)

    # candidates
    ap.add_argument("--Km_vals", type=float, nargs=4, default=[5.0, 10.0, 15.0, 30.0])
    ap.add_argument("--n_vals", type=float, nargs=4, default=[1.0, 2.0, 3.0, 4.0])

    # modulators (preferred)
    ap.add_argument("--smod_C1", type=float, default=None)
    ap.add_argument("--smod_C2", type=float, default=None)

    # aliases (backward compatible)
    ap.add_argument("--sAA_C1", type=float, default=None)
    ap.add_argument("--sAA_C2", type=float, default=None)

    # mineral
    ap.add_argument("--zeta_C1", type=float, default=1.0)
    ap.add_argument("--zeta_C2", type=float, default=1.0)
    ap.add_argument("--t_start", type=float, default=14.0)

    # time/out
    ap.add_argument("--dt_h", type=float, default=1.0)
    ap.add_argument("--outdir", type=str, default="out_bas_v_osteo")

    # Sobol
    ap.add_argument("--sobol", action="store_true")
    ap.add_argument("--sobol_baseN", type=int, default=512)

    ap.add_argument("--smod2_vals", type=float, nargs=4, default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--sAA2_vals", type=float, nargs=4, default=None)
    ap.add_argument("--half_life_vals", type=float, nargs=4, default=[1.5, 3.0, 6.0, 12.0])
    ap.add_argument("--media_interval_vals", type=float, nargs=4, default=[24.0, 48.0, 72.0, 96.0])

    ap.add_argument("--zeta2_vals", type=float, nargs=4, default=[0.5, 1.0, 1.5, 2.0])
    ap.add_argument("--t_start_vals", type=float, nargs=4, default=[12.0, 14.0, 16.0, 18.0])
    ap.add_argument("--k_vals", type=float, nargs=4, default=None)

    args = ap.parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # resolve s_mod values
    if args.smod_C1 is None:
        args.smod_C1 = 0.0 if args.sAA_C1 is None else float(args.sAA_C1)
    if args.smod_C2 is None:
        args.smod_C2 = 1.0 if args.sAA_C2 is None else float(args.sAA_C2)

    smod2_vals = args.smod2_vals if args.sAA2_vals is None else args.sAA2_vals
    smod2_vals = list(map(float, smod2_vals))

    dt_days = float(args.dt_h) / 24.0
    t_days = np.arange(0.0, DAY_MINERAL + 0.5 * dt_days, dt_days)

    # AA
    A = aa_profile(t_days, args.AA0_ugml, args.half_life_h, args.media_interval_h, args.day_start_AA)

    Km_vals = list(map(float, args.Km_vals))
    n_vals = list(map(float, args.n_vals))

    # Integral of sat(A,Km) over 0–14 (for r0 solving)
    m14 = t_days <= DAY_COLLAGEN + 1e-12
    t14 = t_days[m14]
    dt = float(np.median(np.diff(t14)))

    sat_int = []
    for Km in Km_vals:
        sat = mm_saturation(A[m14], float(Km))
        sat_int.append(float(np.sum(sat[:-1]) * dt))
    sat_int = np.asarray(sat_int, dtype=float)

    # Solve r0(Km) from C1 endpoint:
    # C1(14) = r0*14 + s_mod_C1 * Int(Km)
    r0_per_Km = (float(args.collagen_C1) - float(args.smod_C1) * sat_int) / float(DAY_COLLAGEN)

    # Predict C2 endpoint per Km:
    # C2(14) = r0(Km)*14 + s_mod_C2 * Int(Km)
    C2_pred = r0_per_Km * float(DAY_COLLAGEN) + float(args.smod_C2) * sat_int
    err = np.abs(C2_pred - float(args.collagen_C2))

    # Select Km* by C2 endpoint misfit
    Km_index = int(np.argmin(err))
    Km_star = float(Km_vals[Km_index])
    r0 = float(r0_per_Km[Km_index])

    # Mineral endpoints conversion
    H1_end, plot_scale, ylabel = ars_to_fraction(args.ars_C1, args.ars_units)
    H2_end, _, _ = ars_to_fraction(args.ars_C2, args.ars_units)

    # keep n* as middle candidate
    n_index = 2 if len(n_vals) >= 3 else 0
    n_star = float(n_vals[n_index])

    T_eff = DAY_MINERAL - float(args.t_start)
    k1_star = solve_k_from_endpoint(H1_end, T_eff, n_star, float(args.zeta_C1))
    k2_star = solve_k_from_endpoint(H2_end, T_eff, n_star, float(args.zeta_C2))

    # simulate exported run
    C1 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=float(args.smod_C1), Km=Km_star)
    C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=float(args.smod_C2), Km=Km_star)

    H1 = mineral_hill(t_days, t_start=float(args.t_start), k=float(k1_star), n=n_star, zeta=float(args.zeta_C1))
    H2 = mineral_hill(t_days, t_start=float(args.t_start), k=float(k2_star), n=n_star, zeta=float(args.zeta_C2))

    # save timeseries + npz
    df = pd.DataFrame(
        {
            "day": t_days,
            "AA_ugml": A,
            "C_C1": C1,
            "C_C2": C2,
            "H_C1_fraction": H1,
            "H_C2_fraction": H2,
            "H_C1_plot": H1 * plot_scale,
            "H_C2_plot": H2 * plot_scale,
        }
    )
    df.to_csv(outdir / "matrix_mineral_timeseries.csv", index=False)

    np.savez(
        outdir / "matrix_mineral_outputs.npz",
        t_days=t_days,
        AA_ugml=A,
        C_C1=C1,
        C_C2=C2,
        H_C1_fraction=H1,
        H_C2_fraction=H2,
        ars_units=np.array([args.ars_units]),
        plot_scale=np.array([float(plot_scale)]),
        Km_vals=np.array(Km_vals, dtype=float),
        n_vals=np.array(n_vals, dtype=float),
        Km_selected=np.array([Km_star]),
        n_selected=np.array([n_star]),
        r0_per_day=np.array([r0]),
        s_mod_C1=np.array([float(args.smod_C1)]),
        s_mod_C2=np.array([float(args.smod_C2)]),
        k_C1=np.array([float(k1_star)]),
        k_C2=np.array([float(k2_star)]),
        collagen_C2_pred_all=np.array(C2_pred, dtype=float),
        collagen_C2_abs_misfit_all=np.array(err, dtype=float),
    )

    # plots
    plot_collagen_timecourses_0_14(
        outdir=outdir,
        t_days=t_days,
        A=A,
        r0=r0,
        collagen_C1_obs=float(args.collagen_C1),
        collagen_C2_obs=float(args.collagen_C2),
        s_mod_C1=float(args.smod_C1),
        s_mod_C2=float(args.smod_C2),
        Km_star=Km_star,
        day_induction=float(args.day_start_AA),
    )

    plot_km_sweep_collagen(
        outdir=outdir,
        t_days=t_days,
        A=A,
        r0=r0,
        collagen_C2_obs=float(args.collagen_C2),
        Km_vals=Km_vals,
        s_mod_C2=float(args.smod_C2),
        day_induction=float(args.day_start_AA),
    )

    plot_mineral_timecourses(
        outdir=outdir,
        t_days=t_days,
        H1=H1,
        H2=H2,
        plot_scale=float(plot_scale),
        ylabel=ylabel,
        t_start=float(args.t_start),
        n_star=n_star,
        day_induction=float(args.day_start_AA),
    )

    plot_n_sweep_mineral(
        outdir=outdir,
        t_days=t_days,
        ars_C1=float(args.ars_C1),
        ars_C2=float(args.ars_C2),
        ars_units=str(args.ars_units),
        n_vals=n_vals,
        t_start=float(args.t_start),
        zeta_C2=float(args.zeta_C2),
        day_induction=float(args.day_start_AA),
    )

    plot_candidate_selection_explainer(
        outdir=outdir,
        t_days=t_days,
        A=A,
        r0=r0,
        collagen_C2_obs=float(args.collagen_C2),
        Km_vals=Km_vals,
        s_mod_C2=float(args.smod_C2),
        Km_star=Km_star,
        ars_C2=float(args.ars_C2),
        ars_units=str(args.ars_units),
        n_vals=n_vals,
        t_start=float(args.t_start),
        zeta_C2=float(args.zeta_C2),
        n_star=n_star,
    )

    # Sobol
    sobol_out = {
        "selection": {
            "Km_star": Km_star,
            "Km_index": int(Km_index),
            "n_star": n_star,
            "n_index": int(n_index),
            "Km_vals": Km_vals,
            "n_vals": n_vals,
            "s_mod_C1": float(args.smod_C1),
            "s_mod_C2": float(args.smod_C2),
        }
    }

    if args.sobol:
        try:
            import SALib  # noqa: F401
        except Exception:
            raise RuntimeError("SALib not installed. Install: pip install SALib")

        if args.k_vals is None:
            kref = max(float(k2_star), 1e-12)
            k_vals = [0.5 * kref, 0.8 * kref, 1.2 * kref, 2.0 * kref]
        else:
            k_vals = list(map(float, args.k_vals))

        coll = sobol_endpoint_collagen(
            outdir=outdir,
            t_days=t_days,
            AA0_ugml=float(args.AA0_ugml),
            day_start_AA=float(args.day_start_AA),
            r0=float(r0),
            Km_vals=Km_vals,
            s_mod2_vals=smod2_vals,
            half_life_vals=list(map(float, args.half_life_vals)),
            media_interval_vals=list(map(float, args.media_interval_vals)),
            baseN=int(args.sobol_baseN),
        )

        mineral = sobol_endpoint_mineral(
            outdir=outdir,
            t_days=t_days,
            n_vals=n_vals,
            k_vals=k_vals,
            zeta2_vals=list(map(float, args.zeta2_vals)),
            t_start_vals=list(map(float, args.t_start_vals)),
            baseN=int(args.sobol_baseN),
        )

        sobol_out["sobol"] = {
            "collagen_C2_day14": coll,
            "mineral_H2_day28_fraction": mineral,
            "notes": {"endpoint_interpretation": "Sobol indices quantify variance contribution to endpoint outputs.", "ars_units": args.ars_units},
        }

        (outdir / "sobol_indices_endpoints.json").write_text(json.dumps(sobol_out, indent=2), encoding="utf-8")

    print(f"[✓] wrote outputs to: {outdir.resolve()}")
    print(f"[i] selected Km*={Km_star:g} (index {Km_index}), r0={r0:.6g}/day")
    print(f"[i] s_mod_C1={float(args.smod_C1):g}, s_mod_C2={float(args.smod_C2):g}")
    print(f"[i] n*={n_star:g} (index {n_index}); ARS units={args.ars_units}")


if __name__ == "__main__":
    main()
