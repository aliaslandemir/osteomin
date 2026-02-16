#!/usr/bin/env python3
"""Fit collagen control/AA data and report hourly basal and osteogenic rates."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np


T_DAYS = np.array([1, 2, 4, 6, 8, 10, 12], dtype=float)
C_CONTROL = np.array([0.15, 0.20, 0.40, 0.60, 0.70, 0.90, 0.90], dtype=float)
C_AA = np.array([0.20, 0.25, 1.20, 2.10, 3.80, 4.30, 4.60], dtype=float)


def k_from_half_life_hours(tau_h: float) -> float:
    return math.log(2.0) / max(tau_h, 1e-12)


def aa_conc(t_h: float, t_osteo_h: float, a0: float, k_a: float, dt_med_h: float) -> float:
    if t_h < t_osteo_h:
        return 0.0
    n = math.floor((t_h - t_osteo_h) / dt_med_h)
    t_n = t_osteo_h + n * dt_med_h
    return a0 * math.exp(-k_a * (t_h - t_n))


def gate_mm(a: float, k_a: float) -> float:
    return a / (k_a + a)


def simulate(
    t_days: np.ndarray,
    c0: float,
    r_b_h: float,
    r_o_h: float,
    k_a_mm: float,
    a0: float,
    tau_a_h: float,
    dt_med_h: float,
    t_osteo_h: float,
    dt_h: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t_end_h = float(np.max(t_days) * 24.0)
    n_steps = int(np.ceil(t_end_h / dt_h)) + 1

    k_decay = k_from_half_life_hours(tau_a_h)

    c_con = c0
    c_ost = c0

    con_pred = []
    ost_pred = []
    aa_gate_pred = []

    target_hours = 24.0 * np.asarray(t_days, dtype=float)
    idx = 0

    for i in range(n_steps):
        t_h = i * dt_h

        c_con += r_b_h * dt_h

        a_t = aa_conc(t_h, t_osteo_h, a0, k_decay, dt_med_h)
        g_a = gate_mm(a_t, k_a_mm)
        u = 1.0 if t_h >= t_osteo_h else 0.0
        c_ost += (r_b_h + u * r_o_h * g_a) * dt_h

        while idx < len(target_hours) and t_h + 1e-9 >= target_hours[idx]:
            con_pred.append(c_con)
            ost_pred.append(c_ost)
            aa_gate_pred.append(g_a)
            idx += 1

        if idx >= len(target_hours):
            break

    return np.asarray(con_pred), np.asarray(ost_pred), np.asarray(aa_gate_pred)


def fit_rates(
    t_days: np.ndarray,
    c_control: np.ndarray,
    c_aa: np.ndarray,
    k_a_mm: float,
    a0: float,
    tau_a_h: float,
    dt_med_h: float,
    t_osteo_h: float,
    dt_h: float,
) -> dict:
    try:
        from scipy.optimize import least_squares

        def residuals(x: np.ndarray) -> np.ndarray:
            c0, r_b_h, r_o_h = x
            con_hat, aa_hat, _ = simulate(
                t_days, c0, r_b_h, r_o_h, k_a_mm, a0, tau_a_h, dt_med_h, t_osteo_h, dt_h
            )
            return np.concatenate([con_hat - c_control, aa_hat - c_aa])

        x0 = np.array([0.0, 0.01, 0.02], dtype=float)
        lb = np.array([0.0, 0.0, 0.0], dtype=float)
        ub = np.array([np.inf, np.inf, np.inf], dtype=float)
        res = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
        c0, r_b_h, r_o_h = map(float, res.x)
        sse = float(np.sum(res.fun**2))
        method = "scipy.least_squares"
    except Exception:
        c0_grid = np.linspace(0.0, max(c_control.min(), 0.4), 81)
        rb_grid = np.linspace(0.0, 0.04, 161)
        ro_grid = np.linspace(0.0, 0.08, 161)
        best = None
        for c0 in c0_grid:
            for r_b_h in rb_grid:
                for r_o_h in ro_grid:
                    con_hat, aa_hat, _ = simulate(
                        t_days, c0, r_b_h, r_o_h, k_a_mm, a0, tau_a_h, dt_med_h, t_osteo_h, dt_h
                    )
                    sse = float(np.sum((con_hat - c_control) ** 2) + np.sum((aa_hat - c_aa) ** 2))
                    if best is None or sse < best[0]:
                        best = (sse, float(c0), float(r_b_h), float(r_o_h))
        sse, c0, r_b_h, r_o_h = best
        method = "grid"

    con_hat, aa_hat, gate_hat = simulate(
        t_days, c0, r_b_h, r_o_h, k_a_mm, a0, tau_a_h, dt_med_h, t_osteo_h, dt_h
    )
    return {
        "c0": c0,
        "r_b_h": r_b_h,
        "r_o_h": r_o_h,
        "sse": sse,
        "method": method,
        "con_hat": con_hat,
        "aa_hat": aa_hat,
        "gate_hat": gate_hat,
    }


def save_outputs(
    out_dir: Path,
    fit: dict,
    t_days: np.ndarray,
    k_a_mm: float,
    a0: float,
    tau_a_h: float,
    dt_med_h: float,
    t_osteo_h: float,
    dt_h: float,
    t_max_day: float,
    dt_day: float,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t_dense = np.arange(0.0, t_max_day + 1e-12, dt_day)
    con_dense, aa_dense, gate_dense = simulate(
        t_dense,
        fit["c0"],
        fit["r_b_h"],
        fit["r_o_h"],
        k_a_mm,
        a0,
        tau_a_h,
        dt_med_h,
        t_osteo_h,
        dt_h,
    )
    t_h_dense = 24.0 * t_dense

    csv_path = out_dir / "collagen_hourly_fit.csv"
    mat = np.column_stack([
        t_dense,
        t_h_dense,
        con_dense,
        aa_dense,
        gate_dense,
        np.full_like(t_dense, fit["r_b_h"]),
        np.full_like(t_dense, fit["r_o_h"]),
    ])
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header="day,hour,collagen_control_pred,collagen_osteo_pred,aa_gate_pred,r_basal_per_h,r_osteo_per_h",
        comments="",
    )

    png_path = out_dir / "collagen_hourly_fit.png"
    try:
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), gridspec_kw={"width_ratios": [3, 2]})

        ax = axes[0]
        ax.scatter(T_DAYS, C_CONTROL, label="Control data", color="#1f77b4")
        ax.scatter(T_DAYS, C_AA, label="AA data", color="#d62728")
        ax.plot(t_dense, con_dense, color="#1f77b4", lw=2)
        ax.plot(t_dense, aa_dense, color="#d62728", lw=2)
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("Collagen signal (assay units)")
        ax.set_title("Collagen fit")
        ax.grid(alpha=0.25)
        ax.legend()

        ax2 = axes[1]
        ax2.plot(t_dense, gate_dense, color="#2ca02c", lw=2)
        ax2.set_xlabel("Time (day)")
        ax2.set_ylabel("AA gate A/(K_A+A)")
        ax2.set_ylim(0, 1.02)
        ax2.set_title("AA gating")
        ax2.grid(alpha=0.25)

        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception:
        png_path = Path("")

    return csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--A0", type=float, default=50.0, help="AA refreshed concentration (same units as K_A)")
    ap.add_argument("--K_A", type=float, default=10.0, help="AA half-saturation constant for collagen gate")
    ap.add_argument("--tau_A_h", type=float, default=48.0, help="AA half-life (hours)")
    ap.add_argument("--dt_med_h", type=float, default=24.0, help="Media refresh period (hours)")
    ap.add_argument("--t_osteo_h", type=float, default=0.0, help="Osteogenic start time (hours)")
    ap.add_argument("--dt_h", type=float, default=0.25, help="Integrator step (hours)")
    ap.add_argument("--t_max_day", type=float, default=12.0)
    ap.add_argument("--dt_day", type=float, default=0.1)
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "collagen")
    args = ap.parse_args()

    fit = fit_rates(
        T_DAYS,
        C_CONTROL,
        C_AA,
        args.K_A,
        args.A0,
        args.tau_A_h,
        args.dt_med_h,
        args.t_osteo_h,
        args.dt_h,
    )

    csv_path, png_path = save_outputs(
        args.out_dir,
        fit,
        T_DAYS,
        args.K_A,
        args.A0,
        args.tau_A_h,
        args.dt_med_h,
        args.t_osteo_h,
        args.dt_h,
        args.t_max_day,
        args.dt_day,
    )

    print("=== Collagen hourly fit (AA-gated) ===")
    print(f"fit_method      : {fit['method']}")
    print(f"C0              : {fit['c0']:.6g} assay units")
    print(f"r_basal_C       : {fit['r_b_h']:.6g} assay units/hour")
    print(f"r_osteo_C       : {fit['r_o_h']:.6g} assay units/hour")
    print(f"K_A             : {args.K_A:.6g}")
    print(f"A0              : {args.A0:.6g}")
    print(f"tau_A_h         : {args.tau_A_h:.6g}")
    print(f"dt_med_h        : {args.dt_med_h:.6g}")
    final_day = T_DAYS[-1]
    final_idx = -1
    print(f"SSE             : {fit['sse']:.6g}")
    print(
        f"final_day       : {final_day:.0f} d | "
        f"CON obs={C_CONTROL[final_idx]:.6g}, pred={fit['con_hat'][final_idx]:.6g} | "
        f"AA obs={C_AA[final_idx]:.6g}, pred={fit['aa_hat'][final_idx]:.6g}"
    )
    print(f"saved_csv       : {csv_path}")
    if str(png_path):
        print(f"saved_plot      : {png_path}")
    else:
        print("saved_plot      : matplotlib unavailable (plot not generated)")


if __name__ == "__main__":
    main()
