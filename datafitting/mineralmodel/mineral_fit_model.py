#!/usr/bin/env python3
"""


This model is *lumped, low-dimensional, and phenomenological*. It is not a full
physicochemical speciation model; instead it provides a mechanistic yet
tractable description of:

  - Ca–Pi supersaturation
  - ACP nucleation
  - ACP→HA conversion
  - Ion consumption according to simplified stoichiometry
  - Medium changes
  - ALP-derived phosphate input (only osteogenic condition)

Outputs:
  - Daily Ca, Pi, ACP, HA, supersaturation S
  - Side-by-side basal vs osteogenic simulations
  - Publication-grade figure (PDF + PNG)
  - Excel file with metadata
  - Plaintext README in the output folder

Designed for MC3T3-E1 endpoint fitting:
  - Basal ARS area ~ 6
  - Osteogenic ARS area ~ 60
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------------------------------------------
# Utility: Load osteogenic Pi_mM_daily (basal = zero vector)
# ------------------------------------------------------------------------------


def load_pi_daily_osteo(path, sheet, col, days):
    """Load osteogenic Pi_mM_daily; if file missing, returns zeros."""
    if path is None:
        return np.zeros(days)

    df = pd.read_excel(path, sheet_name=sheet)
    if col not in df.columns:
        raise ValueError(f"Column '{col}' not found. Columns: {list(df.columns)}")

    vals = df[col].to_numpy(float)
    if len(vals) == 0:
        return np.zeros(days)

    # Pad if needed
    if len(vals) < days:
        vals = np.concatenate([vals, np.full(days - len(vals), vals[-1])])

    return vals[:days]


# ------------------------------------------------------------------------------
# Core ACP → HA simulator
# ------------------------------------------------------------------------------


def simulate_condition(
    K_nuc,
    K_ha,
    Ca0,
    Pi0,
    Pi_daily,
    ksp_eff,
    S_thresh,
    n_exp,
    days,
    dt,
    media_interval,
):

    steps = max(int(days / dt), days)

    # initial
    ca = Ca0
    pi = Pi0 + Pi_daily[0]
    acp = 0.0
    ha = 0.0
    S = 0.0

    t_next_day = 1
    cur_day = 0

    T = []
    Ca = []
    Pi = []
    ACP = []
    HA = []
    SS = []

    for step in range(steps):
        t = (step + 1) * dt
        new_day = int(np.floor(t + 1e-9))

        # medium change
        if new_day != cur_day:
            for d in range(cur_day + 1, new_day + 1):
                if d < days and d % media_interval == 0:
                    ca = Ca0
                    pi = Pi0 + Pi_daily[min(d, len(Pi_daily) - 1)]
            cur_day = new_day

        # supersaturation
        S = (ca**3 * pi**2) / ksp_eff if ksp_eff > 0 else 0.0
        drive = max(0, S - S_thresh)

        r_acp = K_nuc * drive**n_exp if drive > 0 else 0.0
        r_ha = K_ha * acp

        # update phases
        acp += (r_acp - 3 * r_ha) * dt
        ha += (2 * r_ha) * dt

        # ions
        ca += (-3 * r_acp - r_ha) * dt
        pi += (-2 * r_acp) * dt

        # no-negative
        acp = max(acp, 0)
        ha = max(ha, 0)
        ca = max(ca, 0)
        pi = max(pi, 0)

        if t >= t_next_day - 1e-9 and t_next_day <= days:
            T.append(t_next_day)
            Ca.append(ca)
            Pi.append(pi)
            ACP.append(acp)
            HA.append(ha)
            SS.append(S)
            t_next_day += 1

    return (
        np.array(T),
        np.array(Ca),
        np.array(Pi),
        np.array(ACP),
        np.array(HA),
        np.array(SS),
    )


# ------------------------------------------------------------------------------
# Grid search (fit K_nuc, K_ha)
# ------------------------------------------------------------------------------


def fit_rates(
    basal_target,
    osteo_target,
    Ca0,
    Pi0_basal,
    Pi0_osteo,
    Pi_daily_osteo,
    ksp_eff,
    S_thresh,
    n_exp,
    days,
    dt,
    media_interval,
):
    zeros = np.zeros(days)

    best_err = np.inf
    best_Kn = None
    best_Kh = None

    logK = np.linspace(-4, 1, 25)

    for lg1 in logK:
        for lg2 in logK:
            K1 = 10**lg1
            K2 = 10**lg2

            _, _, _, _, HA_basal, _ = simulate_condition(
                K1,
                K2,
                Ca0,
                Pi0_basal,
                zeros,
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
                ksp_eff,
                S_thresh,
                n_exp,
                days,
                dt,
                media_interval,
            )

            err = (HA_basal[-1] - basal_target) ** 2 + (
                HA_osteo[-1] - osteo_target
            ) ** 2

            if err < best_err:
                best_err = err
                best_Kn = K1
                best_Kh = K2

    return best_Kn, best_Kh, best_err


# ------------------------------------------------------------------------------
# Publication-grade plotting
# ------------------------------------------------------------------------------


def publication_style():
    plt.rcParams.update(
        {
            "font.size": 11,
            "font.family": "sans-serif",
            "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
            "axes.labelsize": 13,
            "axes.titlesize": 14,
            "legend.fontsize": 11,
            "lines.linewidth": 2.2,
            "figure.dpi": 150,
        }
    )


def plot_publication(days, basal, osteo, basal_target, osteo_target, outdir):

    publication_style()

    fig, axes = plt.subplots(3, 1, figsize=(7.2, 10), sharex=True)

    # ---------------- HA panel ----------------
    ax = axes[0]
    ax.plot(days, basal["HA"], label="Basal (model)", color="#1f77b4")
    ax.plot(days, osteo["HA"], label="Osteo (model)", color="#d62728")

    ax.scatter(
        days[-1],
        basal_target,
        edgecolor="black",
        facecolor="#1f77b4",
        s=70,
        label="Basal endpoint",
    )
    ax.scatter(
        days[-1],
        osteo_target,
        edgecolor="black",
        facecolor="#d62728",
        s=70,
        label="Osteo endpoint",
    )

    ax.set_ylabel("Mineral (HA proxy)")
    ax.set_title("Mineralization dynamics (ACP → HA)")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    # ---------------- Ca & Pi panel ----------------
    ax = axes[1]
    ax.plot(days, basal["Ca"], "--", color="#1f77b4", label="Basal Ca")
    ax.plot(days, basal["Pi"], "-", color="#1f77b4", alpha=0.7, label="Basal Pi")
    ax.plot(days, osteo["Ca"], "--", color="#d62728", label="Osteo Ca")
    ax.plot(days, osteo["Pi"], "-", color="#d62728", alpha=0.7, label="Osteo Pi")

    ax.set_ylabel("Concentration (mM)")
    ax.set_title("Extracellular Ca and Pi")
    ax.legend(ncol=2, frameon=False)
    ax.grid(alpha=0.25)

    # ---------------- Supersaturation panel ----------------
    ax = axes[2]
    ax.plot(days, basal["S"], label="Basal", color="#1f77b4")
    ax.plot(days, osteo["S"], label="Osteo", color="#d62728")

    ax.set_ylabel("Supersaturation (S)")
    ax.set_xlabel("Day")
    ax.set_title("Supersaturation index")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)

    fig.tight_layout()

    fig.savefig(outdir / "mineral_fit_plots.png", dpi=300)
    fig.savefig(outdir / "mineral_fit_plots.pdf", dpi=300)
    plt.close(fig)


# ------------------------------------------------------------------------------
# Excel writer (with metadata)
# ------------------------------------------------------------------------------


def save_excel(days, basal, osteo, basename):

    df_basal = pd.DataFrame(
        {
            "Day": days,
            "Ca_mM": basal["Ca"],
            "Pi_mM": basal["Pi"],
            "ACP_units": basal["ACP"],
            "HA_units": basal["HA"],
            "Supersaturation_S": basal["S"],
        }
    )

    df_osteo = pd.DataFrame(
        {
            "Day": days,
            "Ca_mM": osteo["Ca"],
            "Pi_mM": osteo["Pi"],
            "ACP_units": osteo["ACP"],
            "HA_units": osteo["HA"],
            "Supersaturation_S": osteo["S"],
        }
    )

    with pd.ExcelWriter(basename) as writer:
        df_basal.to_excel(writer, "Basal", index=False)
        df_osteo.to_excel(writer, "Osteo", index=False)

        metadata = pd.DataFrame(
            {
                "Description": [
                    "Minimal ACP → HA model",
                    "Supersaturation-driven nucleation",
                    "Endpoint-calibrated using ARS day-28 values",
                    "Osteogenic Pi from ALP → Pi model",
                ]
            }
        )
        metadata.to_excel(writer, "Metadata", index=False)


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------


def main():

    p = argparse.ArgumentParser()

    p.add_argument("--basal_mineral", required=True, type=float)
    p.add_argument("--osteo_mineral", required=True, type=float)

    p.add_argument("--ca_mM", type=float, default=2.0)
    p.add_argument("--pi_basal_mM", type=float, default=1.0)
    p.add_argument("--pi_osteo_mM", type=float, default=1.0)

    p.add_argument("--media_interval", type=int, default=2)

    p.add_argument("--pi_daily_osteo_xlsx", type=str, default=None)
    p.add_argument("--pi_daily_sheet", type=str, default="Sheet1")
    p.add_argument("--pi_daily_column", type=str, default="Pi_mM_daily")

    p.add_argument("--ksp_eff", type=float, default=1.0)
    p.add_argument("--S_thresh", type=float, default=1.0)
    p.add_argument("--n_exp", type=float, default=1.0)

    p.add_argument("--days", type=int, default=28)
    p.add_argument("--dt", type=float, default=0.25)

    p.add_argument("--knuc", type=float, default=None)
    p.add_argument("--kha", type=float, default=None)

    args = p.parse_args()
    days = args.days

    outdir = Path("output_mineral_fit")
    outdir.mkdir(exist_ok=True)

    # Load Pi daily only for osteo
    pi_daily_osteo = load_pi_daily_osteo(
        args.pi_daily_osteo_xlsx,
        args.pi_daily_sheet,
        args.pi_daily_column,
        args.days,
    )
    pi_daily_basal = np.zeros(days)

    # ---------------- parameter estimation ----------------
    if args.knuc is None or args.kha is None:
        print("[i] Fitting K_NUC and K_HA ...")
        K1, K2, err = fit_rates(
            args.basal_mineral,
            args.osteo_mineral,
            args.ca_mM,
            args.pi_basal_mM,
            args.pi_osteo_mM,
            pi_daily_osteo,
            args.ksp_eff,
            args.S_thresh,
            args.n_exp,
            args.days,
            args.dt,
            args.media_interval,
        )
        K_nuc, K_ha = K1, K2
    else:
        K_nuc, K_ha = args.knuc, args.kha

    # ---------------- final sims ----------------
    days_b, Ca_b, Pi_b, ACP_b, HA_b, S_b = simulate_condition(
        K_nuc,
        K_ha,
        args.ca_mM,
        args.pi_basal_mM,
        pi_daily_basal,
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
        args.ksp_eff,
        args.S_thresh,
        args.n_exp,
        args.days,
        args.dt,
        args.media_interval,
    )

    basal = {"Ca": Ca_b, "Pi": Pi_b, "ACP": ACP_b, "HA": HA_b, "S": S_b}
    osteo = {"Ca": Ca_o, "Pi": Pi_o, "ACP": ACP_o, "HA": HA_o, "S": S_o}

    # ---------------- Save ----------------
    save_excel(days_b, basal, osteo, outdir / "mineral_fit_output.xlsx")
    plot_publication(
        days_b, basal, osteo, args.basal_mineral, args.osteo_mineral, outdir
    )

    # Save a short readme
    with open(outdir / "README.txt", "w") as f:
        f.write(
            "ACP→HA mineralization model\n"
            "Publication-friendly output\n"
            "Basal vs osteogenic with ALP-derived Pi\n"
        )

    print("\n[✓] Completed. Publication-ready outputs generated.\n")


if __name__ == "__main__":
    main()
