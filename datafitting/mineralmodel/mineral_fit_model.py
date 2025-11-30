#!/usr/bin/env python3
"""
ACP → HA mineralization model

This script implements a minimal, phenomenological kinetic model for
osteoblastic matrix mineralization in vitro. It is designed to be coupled to:

    (1) An AA → collagen model that provides a daily collagen time course
        (e.g. column "Collagen_total" in aa_collagen_output.xlsx, column F).
    (2) An ALP → Pi model that provides a daily Pi input time course in
        osteogenic medium (e.g. column "Pi_mM_daily" in alp_output.xlsx /
        alp_linear_output.xlsx, column D).

Conceptual chain (per *effective* osteoblast):

    AA  →  collagen matrix
    ALP ( + β-GP ) → extracellular Pi
    Ca + Pi (supersaturation, on collagen) → ACP nuclei → HA-like mineral

Mathematically, the model is "lumped" and well-mixed. All quantities are
interpreted as *per effective cell* or, equivalently, per average cell in a
culture well. Absolute scaling to experimental ARS readouts is done by
fitting two rate constants (ACP nucleation and ACP→HA conversion) so that
simulated HA at day N matches basal and osteogenic endpoints.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------
# Utility: load osteogenic Pi_mM_daily (basal = zero vector)
# ----------------------------------------------------------------------


def load_pi_daily_osteo(path, sheet, col, days):
    """
    Load osteogenic Pi input (mM/day) from an Excel file.
    """
    if path is None:
        return np.zeros(days)

    df = pd.read_excel(path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. Columns: {list(df.columns)}")

    vals = df[col].to_numpy(dtype=float)
    if len(vals) == 0:
        return np.zeros(days)

    if len(vals) < days:
        vals = np.concatenate([vals, np.full(days - len(vals), vals[-1])])

    return vals[:days]


# ----------------------------------------------------------------------
# Utility: load collagen time course
# ----------------------------------------------------------------------


def load_collagen(path, sheet, col, days):
    """
    Load collagen time course from an Excel file.
    """
    if path is None:
        return np.ones(days)

    df = pd.read_excel(path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found in {path}. Columns: {list(df.columns)}")

    vals = df[col].to_numpy(dtype=float)
    if len(vals) == 0:
        return np.zeros(days)

    if len(vals) < days:
        vals = np.concatenate([vals, np.full(days - len(vals), vals[-1])])

    vals = vals[:days]
    vals = np.clip(vals, a_min=0.0, a_max=None)

    return vals


# ----------------------------------------------------------------------
# Core ACP → HA simulator
# ----------------------------------------------------------------------


def simulate_condition(
    K_nuc,
    K_ha,
    Ca0,
    Pi0,
    Pi_daily,
    collagen_daily,
    ksp_eff,
    S_thresh,
    n_exp,
    days,
    dt,
    media_interval,
):
    """
    Simulate ACP nucleation and HA growth for one condition (basal or osteo).
    """
    steps = max(int(days / dt), days)
    t = np.linspace(0.0, days, steps + 1)

    Ca = np.zeros_like(t)
    Pi = np.zeros_like(t)
    ACP = np.zeros_like(t)
    HA = np.zeros_like(t)
    S = np.zeros_like(t)

    Ca[0] = Ca0
    Pi[0] = Pi0
    ACP[0] = 0.0
    HA[0] = 0.0

    next_media_change = media_interval if media_interval > 0 else None

    for i in range(1, len(t)):
        ti = t[i - 1]
        Ca_i = Ca[i - 1]
        Pi_i = Pi[i - 1]
        ACP_i = ACP[i - 1]
        HA_i = HA[i - 1]

        # 1. Pi input from ALP (mM/day → mM per dt)
        day_index = min(int(np.floor(ti)), days - 1)
        Pi_input = Pi_daily[day_index] * dt
        Pi_i += Pi_input

        # 2. Medium change: reset dissolved ions, keep solid phase
        if next_media_change is not None and ti >= next_media_change - 1e-9:
            Ca_i = Ca0
            Pi_i = Pi0
            next_media_change += media_interval

        # 3. Supersaturation
        S_i = (Ca_i**3) * (Pi_i**2) / max(ksp_eff, 1e-12)

        # 4. Collagen-gated ACP nucleation
        collagen_factor = collagen_daily[day_index]
        drive = max(S_i - S_thresh, 0.0)
        if drive > 0.0 and collagen_factor > 0.0:
            r_acp = K_nuc * collagen_factor * (drive**n_exp)
        else:
            r_acp = 0.0

        # 5. ACP → HA conversion
        r_ha = K_ha * ACP_i

        # 6. Stoichiometry-limited updates (ensure non-negative Ca/P)
        dACP = (r_acp - r_ha) * dt
        dHA = r_ha * dt
        dCa = -3.0 * r_acp * dt
        dPi = -2.0 * r_acp * dt

        if Ca_i + dCa < 0.0 or Pi_i + dPi < 0.0:
            s_ca = Ca_i / (-dCa) if dCa < 0 else 1.0
            s_pi = Pi_i / (-dPi) if dPi < 0 else 1.0
            s = min(1.0, s_ca, s_pi)
            dACP *= s
            dHA *= s
            dCa *= s
            dPi *= s

        Ca_i += dCa
        Pi_i += dPi
        ACP_i += dACP
        HA_i += dHA

        Ca_i = max(Ca_i, 0.0)
        Pi_i = max(Pi_i, 0.0)
        ACP_i = max(ACP_i, 0.0)
        HA_i = max(HA_i, 0.0)

        Ca[i] = Ca_i
        Pi[i] = Pi_i
        ACP[i] = ACP_i
        HA[i] = HA_i
        S[i] = (Ca_i**3) * (Pi_i**2) / max(ksp_eff, 1e-12)

    return t, Ca, Pi, ACP, HA, S


# ----------------------------------------------------------------------
# Grid search (fit K_nuc, K_ha)
# ----------------------------------------------------------------------


def fit_rates(
    basal_target,
    osteo_target,
    Ca0,
    Pi0_basal,
    Pi0_osteo,
    Pi_daily_osteo,
    collagen_daily,
    ksp_eff,
    S_thresh,
    n_exp,
    days,
    dt,
    media_interval,
    logK_min,
    logK_max,
    logK_points,
):
    """
    Fit K_nuc and K_ha by minimizing squared error between final HA values
    and experimental basal / osteo endpoints.

    logK_min, logK_max, logK_points control the search grid in log10-space.
    """
    zeros = np.zeros(days)

    best_err = np.inf
    best_Kn = None
    best_Kh = None

    logK = np.linspace(logK_min, logK_max, logK_points)

    for lg1 in logK:
        for lg2 in logK:
            K1 = 10.0**lg1
            K2 = 10.0**lg2

            _, _, _, _, HA_basal, _ = simulate_condition(
                K1,
                K2,
                Ca0,
                Pi0_basal,
                zeros,
                collagen_daily,
                ksp_eff,
                S_thresh,
                n_exp,
                days,
                dt,
                media_interval,
            )

            _, _, _, _, HA_osteo, _ = simulate_condition(
                K1,
                K2,
                Ca0,
                Pi0_osteo,
                Pi_daily_osteo,
                collagen_daily,
                ksp_eff,
                S_thresh,
                n_exp,
                days,
                dt,
                media_interval,
            )

            err = (HA_basal[-1] - basal_target) ** 2 + (HA_osteo[-1] - osteo_target) ** 2

            if err < best_err:
                best_err = err
                best_Kn = K1
                best_Kh = K2

    return best_Kn, best_Kh, best_err


# ----------------------------------------------------------------------
# Publication-grade plotting
# ----------------------------------------------------------------------


def publication_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "legend.fontsize": 10,
            "figure.dpi": 150,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linestyle": ":",
        }
    )


def plot_publication(days, basal, osteo, basal_target, osteo_target, outdir):
    """
    Three aligned panels with shared x-axis:

        (1) HA vs time with endpoint markers
        (2) Ca and Pi vs time
        (3) Supersaturation S vs time
    """
    publication_style()

    fig, axes = plt.subplots(
        3,
        1,
        figsize=(7.2, 9.0),
        sharex=True,
        constrained_layout=True,
        gridspec_kw={"hspace": 0.3},
    )

    t = np.asarray(days)

    # ---- (1) HA panel ----
    ax = axes[0]
    basal_line = ax.plot(t, basal["HA"], lw=2.0, label="Basal (model)")
    osteo_line = ax.plot(t, osteo["HA"], lw=2.0, label="Osteo (model)")

    basal_color = basal_line[0].get_color()
    osteo_color = osteo_line[0].get_color()

    ax.scatter(
        [t[-1]],
        [basal_target],
        s=60,
        facecolors=basal_color,
        edgecolors="black",
        linewidths=1.0,
        label="Basal endpoint",
        zorder=5,
    )
    ax.scatter(
        [t[-1]],
        [osteo_target],
        s=60,
        facecolors=osteo_color,
        edgecolors="black",
        linewidths=1.0,
        label="Osteo endpoint",
        zorder=5,
    )

    ax.set_ylabel("Mineral (HA proxy, a.u.)")
    ax.set_title("Matrix mineralization (ACP → HA, collagen-gated)")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    # ---- (2) Ca & Pi panel ----
    ax = axes[1]
    ax.plot(t, basal["Ca"], lw=1.7, ls="--", label="Basal Ca")
    ax.plot(t, basal["Pi"], lw=1.7, ls="-", label="Basal Pi")
    ax.plot(t, osteo["Ca"], lw=1.7, ls="--", label="Osteo Ca")
    ax.plot(t, osteo["Pi"], lw=1.7, ls="-", label="Osteo Pi")

    ax.set_ylabel("Concentration (mM)")
    ax.set_title("Dissolved Ca and Pi (medium vs consumption)")
    ax.legend(frameon=False, loc="upper left", ncol=2)

    # ---- (3) Supersaturation panel ----
    ax = axes[2]
    ax.plot(t, basal["S"], lw=1.8, label="Basal S")
    ax.plot(t, osteo["S"], lw=1.8, label="Osteo S")

    ax.set_ylabel("Supersaturation index S")
    ax.set_xlabel("Culture time (days)")
    ax.set_title("Ca–Pi supersaturation driving ACP nucleation")
    ax.legend(frameon=False, loc="upper left")

    fig.suptitle(
        "Coupled AA–collagen, ALP–Pi, and ACP–HA mineralization model",
        fontsize=13,
    )

    fig.savefig(outdir / "mineral_fit_plots.png", dpi=300)
    fig.savefig(outdir / "mineral_fit_plots.pdf", dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------------
# Excel writer (with metadata)
# ----------------------------------------------------------------------


def save_excel(days, basal, osteo, pi_daily_basal, pi_daily_osteo, collagen_daily, basename):
    """
    Save model outputs, Pi input, and collagen trajectory into an Excel file.
    """
    days_vec = np.asarray(days)
    coll_vec = np.asarray(collagen_daily)

    day_index = np.minimum(days_vec.astype(int), len(pi_daily_basal) - 1)
    pi_in_basal = pi_daily_basal[day_index]
    pi_in_osteo = pi_daily_osteo[day_index]

    coll_for_grid = coll_vec[np.minimum(days_vec.astype(int), len(coll_vec) - 1)]

    df_basal = pd.DataFrame(
        {
            "Day": days_vec,
            "Ca_mM": basal["Ca"],
            "Pi_mM": basal["Pi"],
            "ACP_units": basal["ACP"],
            "HA_units": basal["HA"],
            "Supersaturation_S": basal["S"],
            "Pi_input_mM_per_day": pi_in_basal,
            "Collagen_total": coll_for_grid,
        }
    )

    df_osteo = pd.DataFrame(
        {
            "Day": days_vec,
            "Ca_mM": osteo["Ca"],
            "Pi_mM": osteo["Pi"],
            "ACP_units": osteo["ACP"],
            "HA_units": osteo["HA"],
            "Supersaturation_S": osteo["S"],
            "Pi_input_mM_per_day": pi_in_osteo,
            "Collagen_total": coll_for_grid,
        }
    )

    with pd.ExcelWriter(basename) as writer:
        df_basal.to_excel(excel_writer=writer, sheet_name="Basal", index=False)
        df_osteo.to_excel(excel_writer=writer, sheet_name="Osteo", index=False)

        metadata = pd.DataFrame(
            {
                "Description": [
                    "Minimal ACP → HA model (collagen-gated nucleation)",
                    "Supersaturation-driven nucleation: r_acp ∝ collagen * max(S - S_thresh, 0)^n",
                    "ACP → HA first-order conversion: r_ha = K_ha * ACP",
                    "Medium changes reset Ca and Pi only; ACP/HA remain in the solid phase.",
                    "Osteogenic Pi input provided by ALP → Pi model (Pi_mM_daily).",
                    "Collagen input provided by AA → collagen model (Collagen_total).",
                    "All variables are interpreted as per effective osteoblast (well-averaged).",
                ]
            }
        )
        metadata.to_excel(excel_writer=writer, sheet_name="Metadata", index=False)


# ----------------------------------------------------------------------
# Command-line interface
# ----------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description="ACP→HA mineralization model (collagen-gated).")

    # Experimental endpoints
    p.add_argument("--basal_mineral", required=True, type=float, help="Basal endpoint mineral proxy (final HA units).")
    p.add_argument("--osteo_mineral", required=True, type=float, help="Osteogenic endpoint mineral proxy (final HA units).")

    # Medium composition
    p.add_argument("--ca_mM", type=float, default=2.0, help="Medium Ca concentration (mM).")
    p.add_argument("--pi_basal_mM", type=float, default=1.0, help="Initial Pi in basal medium (mM).")
    p.add_argument("--pi_osteo_mM", type=float, default=1.0, help="Initial Pi in osteogenic medium (mM).")

    p.add_argument("--media_interval", type=int, default=2, help="Medium change interval (days).")

    # Pi input from ALP model
    p.add_argument("--pi_daily_osteo_xlsx", type=str, default=None, help="Excel file with Pi_mM_daily for osteo.")
    p.add_argument("--pi_daily_sheet", type=str, default="Sheet1", help="Sheet name for Pi_mM_daily.")
    p.add_argument(
        "--pi_daily_column",
        type=str,
        default="Pi_mM_daily",
        help="Column name in Pi file containing daily Pi input (mM/day).",
    )
    p.add_argument(
        "--pi_scale_osteo",
        type=float,
        default=1.0,
        help="Global scaling factor for osteo Pi_mM_daily (e.g. 2.0 doubles Pi input).",
    )

    # Collagen input from AA → collagen model
    p.add_argument(
        "--collagen_xlsx",
        type=str,
        default=None,
        help="Excel file with collagen time course (e.g. aa_collagen_output.xlsx).",
    )
    p.add_argument("--collagen_sheet", type=str, default="Sheet1", help="Sheet name for collagen.")
    p.add_argument(
        "--collagen_column",
        type=str,
        default="Collagen_total",
        help="Column name with collagen trajectory (e.g. Collagen_total).",
    )
    p.add_argument(
        "--collagen_scale",
        type=float,
        default=1.0,
        help="Global scaling factor for collagen gating (e.g. 2.0 doubles nucleation strength).",
    )

    # Supersaturation and kinetics parameters
    p.add_argument("--ksp_eff", type=float, default=1.0, help="Effective solubility product for S.")
    p.add_argument("--S_thresh", type=float, default=1.0, help="Supersaturation threshold for nucleation.")
    p.add_argument("--n_exp", type=float, default=1.0, help="Exponent for supersaturation driving term.")

    # Simulation control
    p.add_argument("--days", type=int, default=28, help="Total simulation time (days).")
    p.add_argument("--dt", type=float, default=0.1, help="Time step (days).")

    # Fit grid control
    p.add_argument(
        "--logK_min",
        type=float,
        default=-4.0,
        help="Minimum log10(K) explored for K_nuc and K_ha.",
    )
    p.add_argument(
        "--logK_max",
        type=float,
        default=4.0,
        help="Maximum log10(K) explored for K_nuc and K_ha.",
    )
    p.add_argument(
        "--logK_points",
        type=int,
        default=33,
        help="Number of grid points per parameter in log10-space.",
    )

    # Output directory
    p.add_argument("--outdir", type=str, default="mineral_fit_output", help="Output directory.")

    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    days = args.days

    # Pi input and collagen trajectory (with user scalings)
    pi_daily_osteo = load_pi_daily_osteo(
        args.pi_daily_osteo_xlsx, args.pi_daily_sheet, args.pi_daily_column, days
    ) * args.pi_scale_osteo
    pi_daily_basal = np.zeros(days)

    collagen_daily = load_collagen(
        args.collagen_xlsx, args.collagen_sheet, args.collagen_column, days
    ) * args.collagen_scale

    # Fit K_nuc and K_ha
    K_nuc, K_ha, err = fit_rates(
        args.basal_mineral,
        args.osteo_mineral,
        args.ca_mM,
        args.pi_basal_mM,
        args.pi_osteo_mM,
        pi_daily_osteo,
        collagen_daily,
        args.ksp_eff,
        args.S_thresh,
        args.n_exp,
        args.days,
        args.dt,
        args.media_interval,
        args.logK_min,
        args.logK_max,
        args.logK_points,
    )

    print(f"Best-fit K_nuc = {K_nuc:.3e}, K_ha = {K_ha:.3e} (squared error = {err:.3e})")

    # Simulate basal and osteo with best-fit parameters
    days_b, Ca_b, Pi_b, ACP_b, HA_b, S_b = simulate_condition(
        K_nuc,
        K_ha,
        args.ca_mM,
        args.pi_basal_mM,
        pi_daily_basal,
        collagen_daily,
        args.ksp_eff,
        args.S_thresh,
        args.n_exp,
        args.days,
        args.dt,
        args.media_interval,
    )

    days_o, Ca_o, Pi_o, ACP_o, HA_o, S_o = simulate_condition(
        K_nuc,
        K_ha,
        args.ca_mM,
        args.pi_osteo_mM,
        pi_daily_osteo,
        collagen_daily,
        args.ksp_eff,
        args.S_thresh,
        args.n_exp,
        args.days,
        args.dt,
        args.media_interval,
    )

    basal = {"Ca": Ca_b, "Pi": Pi_b, "ACP": ACP_b, "HA": HA_b, "S": S_b}
    osteo = {"Ca": Ca_o, "Pi": Pi_o, "ACP": ACP_o, "HA": HA_o, "S": S_o}

    save_excel(
        days_b,
        basal,
        osteo,
        pi_daily_basal,
        pi_daily_osteo,
        collagen_daily,
        outdir / "mineral_fit_output.xlsx",
    )
    plot_publication(days_b, basal, osteo, args.basal_mineral, args.osteo_mineral, outdir)



    print("\n[✓] Completed. Publication-ready outputs generated.\n")


if __name__ == "__main__":
    main()
