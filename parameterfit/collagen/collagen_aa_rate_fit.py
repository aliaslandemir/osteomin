#!/usr/bin/env python3
"""Collagen rate fit."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


T_DAYS = np.array([1, 2, 4, 6, 8, 10, 12], dtype=float)

C_CONTROL = np.array([0.15, 0.20, 0.40, 0.60, 0.70, 0.85, 0.80], dtype=float)


C_AA = np.array([0.15, 0.25, 1.40, 2.10, 3.80, 4.10, 4.40], dtype=float)

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


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"SSE": sse, "RMSE": rmse, "MAE": mae, "R2": r2}


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

        # Rates are fit without AA gate; gating is applied in OsteoMin runtime.
        g_a = 1.0
        u = 1.0 if t_h >= t_osteo_h else 0.0
        c_ost += (r_b_h + u * r_o_h) * dt_h

        while idx < len(target_hours) and t_h + 1e-9 >= target_hours[idx]:
            con_pred.append(c_con)
            ost_pred.append(c_ost)
            aa_gate_pred.append(g_a)
            idx += 1

        if idx >= len(target_hours):
            break

    return np.asarray(con_pred), np.asarray(ost_pred), np.asarray(aa_gate_pred)


def confidence_halfwidth_sim(
    t_days: np.ndarray,
    params: np.ndarray,
    pcov: np.ndarray,
    k_a_mm: float,
    a0: float,
    tau_a_h: float,
    dt_med_h: float,
    t_osteo_h: float,
    dt_h: float,
    n_obs: int,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence half-widths for simulated control and AA mean curves."""
    c0, r_b_h, r_o_h = [float(v) for v in params]
    p = 3
    dof = max(1, n_obs - p)
    try:
        from scipy.stats import t as student_t

        crit = float(student_t.ppf(1.0 - alpha / 2.0, dof))
    except Exception:
        crit = 1.96

    theta = np.array([c0, r_b_h, r_o_h], dtype=float)
    jac_con = np.zeros((len(t_days), p), dtype=float)
    jac_aa = np.zeros((len(t_days), p), dtype=float)

    for j in range(p):
        step = 1e-6 * max(1.0, abs(theta[j]))
        theta_p = theta.copy()
        theta_m = theta.copy()
        theta_p[j] += step
        theta_m[j] = max(0.0, theta_m[j] - step)
        con_p, aa_p, _ = simulate(
            t_days,
            float(theta_p[0]),
            float(theta_p[1]),
            float(theta_p[2]),
            k_a_mm,
            a0,
            tau_a_h,
            dt_med_h,
            t_osteo_h,
            dt_h,
        )
        con_m, aa_m, _ = simulate(
            t_days,
            float(theta_m[0]),
            float(theta_m[1]),
            float(theta_m[2]),
            k_a_mm,
            a0,
            tau_a_h,
            dt_med_h,
            t_osteo_h,
            dt_h,
        )
        denom = max(theta_p[j] - theta_m[j], 1e-12)
        jac_con[:, j] = (con_p - con_m) / denom
        jac_aa[:, j] = (aa_p - aa_m) / denom

    var_con = np.einsum("ij,jk,ik->i", jac_con, pcov, jac_con)
    var_aa = np.einsum("ij,jk,ik->i", jac_aa, pcov, jac_aa)
    hw_con = crit * np.sqrt(np.maximum(var_con, 0.0))
    hw_aa = crit * np.sqrt(np.maximum(var_aa, 0.0))
    return hw_con, hw_aa


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

        t_days_arr = np.asarray(t_days, dtype=float)
        y_obs = np.concatenate([c_control, c_aa])

        def stacked_model(theta: np.ndarray) -> np.ndarray:
            c0, r_b_h, r_o_h = [float(v) for v in theta]
            con_hat, aa_hat, _ = simulate(
                t_days_arr,
                c0,
                r_b_h,
                r_o_h,
                k_a_mm,
                a0,
                tau_a_h,
                dt_med_h,
                t_osteo_h,
                dt_h,
            )
            return np.concatenate([con_hat, aa_hat])

        def residuals(theta: np.ndarray) -> np.ndarray:
            return stacked_model(theta) - y_obs

        x0 = np.array([0.0, 0.01, 0.02], dtype=float)
        res = least_squares(
            residuals,
            x0=x0,
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        c0, r_b_h, r_o_h = map(float, res.x)
        y_hat = stacked_model(res.x)
        sse = float(np.sum((y_hat - y_obs) ** 2))
        method = "scipy.least_squares"

        dof = max(1, y_obs.size - res.x.size)
        s_sq = sse / dof
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
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
        pcov = np.zeros((3, 3), dtype=float)

    con_hat, aa_hat, gate_hat = simulate(
        t_days, c0, r_b_h, r_o_h, k_a_mm, a0, tau_a_h, dt_med_h, t_osteo_h, dt_h
    )
    y_all = np.concatenate([c_control, c_aa])
    yhat_all = np.concatenate([con_hat, aa_hat])
    return {
        "c0": c0,
        "r_b_h": r_b_h,
        "r_o_h": r_o_h,
        "sse": sse,
        "method": method,
        "con_hat": con_hat,
        "aa_hat": aa_hat,
        "gate_hat": gate_hat,
        "pcov": pcov,
        "metrics_overall": _metrics(y_all, yhat_all),
        "metrics_basal": _metrics(c_control, con_hat),
        "metrics_osteo": _metrics(c_aa, aa_hat),
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
) -> tuple[Path, Path, Path, Path]:
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
    csv_path = out_dir / "collagen_fit_predictions.csv"
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
        header="day,hour,collagen_basal_pred,collagen_osteo_pred,aa_gate_pred,r_basal_per_h,r_osteo_per_h",
        comments="",
    )

    js_path = out_dir / "collagen_fit_metrics.json"
    payload = {
        "fit_method": fit["method"],
        "r_basal_per_h": fit["r_b_h"],
        "r_osteo_per_h": fit["r_o_h"],
        "note": "Ungated rate fit: AA-gate parameters are not used.",
        "metrics_overall": fit["metrics_overall"],
        "metrics_basal": fit["metrics_basal"],
        "metrics_osteo": fit["metrics_osteo"],
        "data": {
            "t_days": T_DAYS.tolist(),
            "collagen_basal": C_CONTROL.tolist(),
            "collagen_osteo": C_AA.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    c_basal = "#1f78b4"
    c_osteo = "#d95f02"

    t_h_data = 24.0 * T_DAYS
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(t_h_dense, con_dense, lw=2.4, color=c_basal)
    ax.plot(t_h_dense, aa_dense, lw=2.4, color=c_osteo)
    ax.scatter(t_h_data, C_CONTROL, s=54, color=c_basal, edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)
    ax.scatter(t_h_data, C_AA, s=54, color=c_osteo, edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)

    x_max_h = float(t_max_day) * 24.0
    x_pad_h = max(4.0, 0.02 * x_max_h)
    y_all = np.concatenate([C_CONTROL, C_AA, con_dense, aa_dense])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.06, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_xlim(-x_pad_h, x_max_h + x_pad_h)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Collagen signal (assay units)")
    ax.set_title("Collagen Activity Fit", fontsize=11, pad=8)
    ax.xaxis.set_major_locator(MultipleLocator(24.0 * 2.0))
    ax.yaxis.set_major_locator(MultipleLocator(0.5))
    ax.yaxis.set_minor_locator(MultipleLocator(0.25))
    ax.tick_params(labelsize=9, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    met = fit["metrics_overall"]
    txt = "\n".join([
        f"r_b (AU h^-1)  {fit['r_b_h']:.3g}",
        f"r_o (AU h^-1)  {fit['r_o_h']:.3g}",
        f"RMSE (AU)      {met['RMSE']:.3g}",
        f"R2             {met['R2']:.3g}",
    ])
    ax.text(
        0.02, 0.70, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.5, family="monospace",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#bcbcbc", alpha=0.96),
    )

    legend_handles = [
        Line2D([0], [0], color=c_basal, lw=2.4, label="Basal fit"),
        Line2D([0], [0], color=c_osteo, lw=2.4, label="Osteogenic fit"),
        Line2D([0], [0], color=c_basal, lw=2.0, marker="o", markersize=7, markerfacecolor=c_basal, markeredgecolor="white", markeredgewidth=0.8, label="Basal data"),
        Line2D([0], [0], color=c_osteo, lw=2.0, marker="o", markersize=7, markerfacecolor=c_osteo, markeredgecolor="white", markeredgewidth=0.8, label="Osteogenic data"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9, handlelength=2.0, borderaxespad=0.6)

    fig.tight_layout()
    png_path = out_dir / "collagen_fit_style.png"
    pdf_path = out_dir / "collagen_fit_style.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, js_path, png_path, pdf_path


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

    csv_path, js_path, png_path, pdf_path = save_outputs(
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

    print("=== Collagen hourly fit (ungated rates) ===")
    print(f"fit_method      : {fit['method']}")
    print(f"r_basal_C       : {fit['r_b_h']:.6g} assay units/hour")
    print(f"r_osteo_C       : {fit['r_o_h']:.6g} assay units/hour")
    final_day = T_DAYS[-1]
    final_idx = -1
    met = fit["metrics_overall"]
    print(f"SSE             : {met['SSE']:.6g}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")
    print(
        f"final_day       : {final_day:.0f} d | "
        f"CON obs={C_CONTROL[final_idx]:.6g}, pred={fit['con_hat'][final_idx]:.6g} | "
        f"AA obs={C_AA[final_idx]:.6g}, pred={fit['aa_hat'][final_idx]:.6g}"
    )
    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
