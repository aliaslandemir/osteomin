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
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

DAY_START_AA = 3.0
DAY_COLLAGEN = 14.0
DAY_MINERAL = 28.0

COLOR_BASAL = "#1F77B4"
COLOR_OSTEO = "#D62728"
COLOR_ACCENT = "#6C757D"
COLOR_SHADE = "#C7CBD1"


# -----------------------------
# Plot style (clean / Nature-ish)
# -----------------------------
def _set_style():
    """Clean, journal-style defaults with clear contrasts."""
    cycle = [COLOR_BASAL, COLOR_OSTEO, "#2CA02C", "#9467BD", "#8C564B", "#17BECF", "#7F7F7F"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 9.0,
            "axes.titlesize": 10.0,
            "axes.labelsize": 9.5,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.9,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "xtick.major.size": 3.6,
            "ytick.major.size": 3.6,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 2.2,
            "axes.prop_cycle": plt.cycler(color=cycle),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _format_axes(ax, grid: bool = False):
    ax.tick_params(which="both", top=False, right=False)
    if grid:
        ax.grid(True, alpha=0.15, linewidth=0.6)
    else:
        ax.grid(False)


def _save(fig, outdir: Path, name: str):
    """Export as PNG + PDF (100/300 dpi), with tight bounding boxes."""
    outdir.mkdir(parents=True, exist_ok=True)

    fig.savefig(outdir / f"{name}.png", bbox_inches="tight", pad_inches=0.12, dpi=100)
    fig.savefig(outdir / f"{name}.pdf", bbox_inches="tight", pad_inches=0.12, dpi=300)




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
    """Euler integrate to collagen measurement day; hold constant afterward."""
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

    delta = C2 - C1
    eps = 1e-6
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.divide(C2, C1, out=np.zeros_like(C2), where=C1 > eps)
    rel_pct = (ratio - 1.0) * 100.0
    rel_pct = np.nan_to_num(rel_pct, nan=0.0, posinf=0.0, neginf=0.0)

    df = pd.DataFrame(
        {
            "day": t,
            "collagen_basal": C1[m14],
            "collagen_osteo": C2[m14],
            "delta": delta[m14],
            "delta_pct": rel_pct[m14],
        }
    )
    df.to_excel(outdir / "collagen_timecourse_values.xlsx", index=False)
    np.savez(
        outdir / "collagen_timecourse_values.npz",
        day=t,
        collagen_basal=C1[m14],
        collagen_osteo=C2[m14],
        delta=delta[m14],
        delta_pct=rel_pct[m14],
    )

    fig, ax = plt.subplots(figsize=(4.4, 3.2), constrained_layout=True)
    ax.plot(t, delta[m14], color=COLOR_OSTEO, linewidth=1.7)
    ax.fill_between(t, 0, delta[m14], where=delta[m14] >= 0, interpolate=True, color=COLOR_OSTEO, alpha=0.22)
    ax.axhline(0.0, color="black", lw=0.9, linestyle="--")
    ax.axvline(day_induction, color=COLOR_ACCENT, ls="--", lw=0.8, label=f"Induction (day {int(day_induction)})")
    ax.axvline(DAY_COLLAGEN, color=COLOR_ACCENT, ls=":", lw=0.8, label=f"Measurement (day {int(DAY_COLLAGEN)})")
    ax.set_xlabel("Day")
    ax.set_ylabel("Δ collagen (Osteogenic − Basal)")
    ax.set_title("Paired Δ collagen (main figure)")
    ypad = 0.08 * (delta[m14].max() - delta[m14].min() + 1e-9)
    ax.set_ylim(delta[m14].min() - ypad, delta[m14].max() + ypad)
    _format_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", fontsize=8)

    ax2 = ax.twinx()
    ax2.plot(t, rel_pct[m14], color=COLOR_BASAL, linestyle=":", linewidth=1.1, label="Relative Δ (%)")
    ax2.set_ylabel("Relative Δ (%)", color=COLOR_BASAL)
    ax2.tick_params(axis="y", colors=COLOR_BASAL)
    ax2.set_ylim(0, max(5.0, rel_pct[m14].max() * 1.2))
    ax2.legend(frameon=False, loc="lower right", fontsize=7)

    _save(fig, outdir, "collagen_timecourses")
    plt.close(fig)


def plot_km_sweep_collagen(
    outdir: Path,
    t_days: np.ndarray,
    A: np.ndarray,
    r0: float,
    collagen_C2_obs: float,
    Km_vals: list[float],
    s_mod_C1: float,
    s_mod_C2_candidates: list[float],
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
    c1_ends = []
    inds = []
    for Km, smod2 in zip(Km_vals, s_mod_C2_candidates):
        C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=float(smod2), Km=float(Km))
        C1 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C1, Km=float(Km))
        trajs.append(C2[m14])
        ends.append(float(np.interp(DAY_COLLAGEN, t_days, C2)))
        c1_ends.append(float(np.interp(DAY_COLLAGEN, t_days, C1)))
        inds.append(float(np.interp(day_induction, t_days, C2)))

    ends = np.asarray(ends, dtype=float)
    c1_ends = np.asarray(c1_ends, dtype=float)
    delta = ends - c1_ends

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(4.8, 4.8),
        constrained_layout=True,
        gridspec_kw={"height_ratios": [2.2, 1.1]}
    )

    # Top panel: time courses (color-mapped by Km)
    cmap = plt.cm.viridis
    km_norm = plt.Normalize(min(Km_vals), max(Km_vals))
    for Km, y in zip(Km_vals, trajs):
        ax1.plot(t, y, color=cmap(km_norm(float(Km))), alpha=0.95)
        ax1.scatter([DAY_COLLAGEN], [y[-1]], s=18, color=cmap(km_norm(float(Km))), zorder=4)

    ax1.axvline(day_induction, ls="--", lw=0.9, alpha=0.7, color=COLOR_ACCENT)
    ax1.axvline(DAY_COLLAGEN, ls=":", lw=0.9, alpha=0.8, color=COLOR_ACCENT)
    ax1.scatter([DAY_COLLAGEN + 0.10], [collagen_C2_obs], s=70, marker="o",
                facecolor="white", edgecolor="black", linewidth=1.0, zorder=10, label="Observed")

    # show induction star for the chosen Km curve? (use Km closest to median just for marker location)
    # but keep it simple: mark induction on observed panel line using one star at mean induction height
    ax1.scatter([day_induction], [np.mean(inds)], marker="*", s=90,
                edgecolor="black", linewidth=0.8, zorder=10)

    ax1.set_xlim(0.0, DAY_COLLAGEN + 0.6)  # prevent clipping
    ax1.set_ylim(bottom=0.0)
    ax1.set_ylabel("Collagen proxy (a.u.)")
    ax1.set_title("Km sweep: osteogenic collagen")
    _format_axes(ax1, grid=True)
    ax1.legend(frameon=False, ncol=1, loc="upper left")

    sm = plt.cm.ScalarMappable(norm=km_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax1, pad=0.02, fraction=0.05)
    cbar.set_label("Km")

    # Bottom panel: endpoint vs Km with delta
    ax2.plot(Km_vals, ends, marker="o", markersize=4.5, markeredgecolor="black", linewidth=2.0, color=COLOR_OSTEO, label="C2")
    ax2.axhline(collagen_C2_obs, ls=":", lw=1.0, color=COLOR_ACCENT, alpha=0.8, label="Observed")
    ax2.set_xlabel("Km candidate")
    ax2.set_ylabel("C2(day 14)")
    ax2.set_title("Endpoint match")
    ypad = 0.05 * (ends.max() - ends.min() + 1e-9)
    ax2.set_ylim(ends.min() - ypad, ends.max() + ypad)
    _format_axes(ax2, grid=True)

    ax2b = ax2.twinx()
    ax2b.plot(Km_vals, delta, linestyle="--", color=COLOR_SHADE, marker="s", markersize=4.5, label="Δ (C2-C1)")
    ax2b.set_ylabel("Δ (collagen)")
    ax2b.set_ylim(0, delta.max() * 1.4 if delta.max() > 0 else 1.0)
    ax2b.spines["right"].set_visible(True)

    lines, labels = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", fontsize=8)

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
    fig, ax = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)
    ax.plot(t_days, H1 * plot_scale, label="Basal", color=COLOR_BASAL)
    ax.plot(t_days, H2 * plot_scale, label="Osteogenic", color=COLOR_OSTEO)
    ax.fill_between(t_days, H1 * plot_scale, H2 * plot_scale, color=COLOR_SHADE, alpha=0.22, zorder=1, label="Δ (osteo-basal)")

    ax.axvline(day_induction, ls="--", lw=0.9, alpha=0.7, color=COLOR_ACCENT)
    ax.axvline(float(t_start), ls=":", lw=0.9, alpha=0.8, color=COLOR_ACCENT)
    ax.axvline(DAY_MINERAL, ls=":", lw=0.9, alpha=0.8, color=COLOR_ACCENT)

    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title("Mineral deposition")
    ax.set_ylim(bottom=0.0)
    _format_axes(ax, grid=True)
    ax.legend(frameon=False, loc="upper left", ncol=2, handlelength=1.6)

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

    fig, ax = plt.subplots(figsize=(4.6, 3.1), constrained_layout=True)

    cmap = plt.cm.plasma
    n_norm = plt.Normalize(min(n_vals), max(n_vals))
    for n in n_vals:
        T_eff = DAY_MINERAL - float(t_start)
        k2 = solve_k_from_endpoint(H2_end, T_eff, float(n), float(zeta_C2))
        H2 = mineral_hill(t_days, t_start=float(t_start), k=float(k2), n=float(n), zeta=float(zeta_C2))
        ax.plot(t_days, H2 * plot_scale, color=cmap(n_norm(float(n))), alpha=0.95)
        ax.scatter([DAY_MINERAL], [H2[-1] * plot_scale], s=18, color=cmap(n_norm(float(n))), zorder=4)

    ax.axvline(day_induction, ls="--", lw=0.9, alpha=0.7, color=COLOR_ACCENT)
    ax.axvline(float(t_start), ls=":", lw=0.9, alpha=0.8, color=COLOR_ACCENT)
    ax.axvline(DAY_MINERAL, ls=":", lw=0.9, alpha=0.8, color=COLOR_ACCENT)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title("n sweep (endpoint matched)")
    ax.set_ylim(bottom=0.0)
    _format_axes(ax, grid=True)
    sm = plt.cm.ScalarMappable(norm=n_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.02, fraction=0.05)
    cbar.set_label("n")

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

    fig, axes = plt.subplots(2, 2, figsize=(7.8, 5.2), constrained_layout=True)

    # (A) collagen misfit
    ax = axes[0, 0]
    x = np.arange(len(Km_vals))
    ax.bar(x, err, color=COLOR_OSTEO, alpha=0.85)
    ax.set_xticks(x, [f"{v:g}" for v in Km_vals])
    if Km_star in Km_vals:
        idx = Km_vals.index(Km_star)
        ax.scatter([idx], [err[idx]], s=90, marker="*", zorder=6)
    ax.set_xlabel("Km candidate")
    ax.set_ylabel("Abs error")
    ax.set_title("Collagen misfit")
    _format_axes(ax, grid=True)

    # (B) mineral trajectories
    ax = axes[0, 1]
    cmap = plt.cm.plasma
    n_norm = plt.Normalize(min(n_vals), max(n_vals))
    for n, k2, H2 in zip(n_vals, k_req, H2_trajs):
        ax.plot(t_days, H2 * plot_scale, color=cmap(n_norm(float(n))), alpha=0.95)
    ax.scatter([DAY_MINERAL], [H2_end * plot_scale], s=22, zorder=5, facecolor="white", edgecolor="black", linewidth=0.9)
    ax.axvline(float(t_start), ls=":", lw=1.0, alpha=0.9)
    ax.axvline(DAY_MINERAL, ls=":", lw=1.0, alpha=0.9)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title("Mineral dynamics")
    ax.set_ylim(bottom=0.0)
    _format_axes(ax, grid=True)
    sm = plt.cm.ScalarMappable(norm=n_norm, cmap=cmap)
    cbar = fig.colorbar(sm, ax=ax, pad=0.01, fraction=0.05)
    cbar.set_label("n")

    # (C) collagen endpoint vs Km
    ax = axes[1, 0]
    ax.plot(Km_vals, C2_end, marker="o", markersize=5.0, markeredgecolor="black", linewidth=2.0, color=COLOR_OSTEO)
    ax.axhline(collagen_C2_obs, ls=":", lw=1.0, color=COLOR_ACCENT, alpha=0.8)
    ax.set_xlabel("Km candidate")
    ax.set_ylabel("C2(day 14) model")
    ax.set_title("Collagen endpoint vs Km")
    _format_axes(ax, grid=True)

    # (D) k vs n
    ax = axes[1, 1]
    x = np.arange(len(n_vals))
    ax.bar(x, k_req, color=COLOR_OSTEO, alpha=0.85)
    ax.set_xticks(x, [f"{v:g}" for v in n_vals])
    if n_star in n_vals:
        nidx = n_vals.index(n_star)
        ax.scatter([nidx], [k_req[nidx]], s=90, marker="*", zorder=6)
    ax.set_xlabel("n candidate")
    ax.set_ylabel("k (1/day)")
    ax.set_title("k to match endpoint")
    _format_axes(ax, grid=True)

    fig.suptitle("Candidate comparison", y=1.01)
    _save(fig, outdir, "candidate_selection_explainer")
    plt.close(fig)


# -----------------------------
# Main
# -----------------------------
def _build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--collagen_day_measure", type=float, default=DAY_COLLAGEN)
    ap.add_argument("--ars_day_measure", type=float, default=DAY_MINERAL)

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

    return ap


def run_matrix_mineral(args: argparse.Namespace):
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    global DAY_COLLAGEN, DAY_MINERAL
    DAY_COLLAGEN = float(args.collagen_day_measure)
    DAY_MINERAL = float(args.ars_day_measure)
    if DAY_COLLAGEN <= 0 or DAY_MINERAL <= 0:
        raise ValueError("collagen_day_measure and ars_day_measure must be > 0.")
    if DAY_MINERAL <= DAY_COLLAGEN:
        raise ValueError("ars_day_measure must be > collagen_day_measure.")
    if not (0 <= float(args.day_start_AA) < DAY_COLLAGEN):
        raise ValueError("day_start_AA must satisfy 0 <= day_start_AA < collagen_day_measure.")

    # resolve s_mod values
    if args.smod_C1 is None:
        args.smod_C1 = 0.0 if args.sAA_C1 is None else float(args.sAA_C1)
    if args.smod_C2 is None:
        args.smod_C2 = None
    elif args.sAA_C2 is not None:
        args.smod_C2 = float(args.sAA_C2)

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

    int_day = float(DAY_COLLAGEN)
    # Solve r0(Km) from C1 endpoint:
    r0_per_Km = (float(args.collagen_C1) - float(args.smod_C1) * sat_int) / int_day

    if args.smod_C2 is None:
        smod_C2_by_Km = np.zeros_like(sat_int)
        mask = sat_int > 0
        smod_C2_by_Km[mask] = (
            float(args.collagen_C2) - r0_per_Km[mask] * int_day
        ) / sat_int[mask]
    else:
        smod_C2_by_Km = np.full_like(sat_int, float(args.smod_C2))

    C2_pred = r0_per_Km * int_day + smod_C2_by_Km * sat_int
    err = np.abs(C2_pred - float(args.collagen_C2))

    # Select Km* by C2 endpoint misfit
    Km_index = int(np.argmin(err))
    Km_star = float(Km_vals[Km_index])
    r0 = float(r0_per_Km[Km_index])
    s_mod_C2_selected = float(smod_C2_by_Km[Km_index])
    args.smod_C2 = s_mod_C2_selected

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
    C2 = simulate_collagen_hold_after_14(t_days, A, r0=r0, s_mod=s_mod_C2_selected, Km=Km_star)

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
        s_mod_C2=np.array([s_mod_C2_selected]),
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
        s_mod_C2=s_mod_C2_selected,
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
        s_mod_C1=float(args.smod_C1),
        s_mod_C2_candidates=list(map(float, smod_C2_by_Km)),
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
    return {
        "outdir": outdir,
        "Km_star": Km_star,
        "Km_index": int(Km_index),
        "n_star": n_star,
        "n_index": int(n_index),
        "k1_star": float(k1_star),
        "k2_star": float(k2_star),
        "r0": float(r0),
        "t_days": t_days,
        "AA_ugml": A,
        "C1": C1,
        "C2": C2,
        "H1_fraction": H1,
        "H2_fraction": H2,
        "plot_scale": float(plot_scale),
        "ylabel": ylabel,
    }


def run_matrix_mineral_from_kwargs(**kwargs):
    """Programmatic entry point that mirrors CLI defaults."""
    defaults = {
        "ars_units": "percent",
        "AA0_ugml": 50.0,
        "half_life_h": 3.0,
        "media_interval_h": 48.0,
        "day_start_AA": DAY_START_AA,
        "collagen_day_measure": DAY_COLLAGEN,
        "ars_day_measure": DAY_MINERAL,
        "Km_vals": [5.0, 10.0, 15.0, 30.0],
        "n_vals": [1.0, 2.0, 3.0, 4.0],
        "smod_C1": None,
        "smod_C2": None,
        "sAA_C1": None,
        "sAA_C2": None,
        "zeta_C1": 1.0,
        "zeta_C2": 1.0,
        "t_start": 14.0,
        "dt_h": 1.0,
        "outdir": "out_bas_v_osteo",
        "sobol": False,
        "sobol_baseN": 512,
        "smod2_vals": [0.0, 0.5, 1.0, 2.0],
        "sAA2_vals": None,
        "half_life_vals": [1.5, 3.0, 6.0, 12.0],
        "media_interval_vals": [24.0, 48.0, 72.0, 96.0],
        "zeta2_vals": [0.5, 1.0, 1.5, 2.0],
        "t_start_vals": [12.0, 14.0, 16.0, 18.0],
        "k_vals": None,
    }
    merged = {**defaults, **kwargs}
    required = ["collagen_C1", "collagen_C2", "ars_C1", "ars_C2"]
    missing = [k for k in required if k not in merged]
    if missing:
        raise ValueError(f"Missing required endpoints: {missing}")

    args_ns = SimpleNamespace(**merged)
    return run_matrix_mineral(args_ns)


def main():
    ap = _build_parser()
    args = ap.parse_args()
    run_matrix_mineral(args)


if __name__ == "__main__":
    main()
