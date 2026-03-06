#!/usr/bin/env python3
"""


ARS fit.

Luttrell2019MC3T3Transcriptomics

"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np



T_DAYS = np.array([4, 7, 10, 14, 21, 28], dtype=float)
ARS = np.array([
    2.0,
    5.0,
    5.0,
    12.0,
    13.0,
    30.0,
], dtype=float)

TAU_H = 384.0
N_H = 4.0


def g_hill(t_h: np.ndarray, tau_h: float = TAU_H, n_h: float = N_H) -> np.ndarray:
    t_h = np.asarray(t_h, dtype=float)
    t_pow = np.power(t_h, n_h)
    return t_pow / (np.power(tau_h, n_h) + t_pow)


def model(t_h: np.ndarray, a0: float, a1: float) -> np.ndarray:
    return a0 + a1 * g_hill(t_h)


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


def confidence_halfwidth(t_h: np.ndarray, pcov: np.ndarray, n_obs: int, alpha: float = 0.05) -> np.ndarray:
    p = 2
    dof = max(1, n_obs - p)
    try:
        from scipy.stats import t as student_t
        crit = float(student_t.ppf(1.0 - alpha / 2.0, dof))
    except Exception:
        crit = 1.96

    g = g_hill(t_h)
    grad = np.column_stack([np.ones_like(t_h, dtype=float), g])
    var = np.einsum("ij,jk,ik->i", grad, pcov, grad)
    return crit * np.sqrt(np.maximum(var, 0.0))


def fit_ars(t_days: np.ndarray, ars: np.ndarray) -> dict:
    t_h = 24.0 * np.asarray(t_days, dtype=float)

    try:
        from scipy.optimize import least_squares

        x0 = np.array([max(0.0, float(np.min(ars))), max(0.0, float(np.max(ars) - np.min(ars)))], dtype=float)
        y_obs = np.asarray(ars, dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(t_h, float(theta[0]), float(theta[1])) - y_obs

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        a0, a1 = map(float, res.x)
        y_hat = model(t_h, a0, a1)
        sse = float(np.sum((y_hat - ars) ** 2))
        method = "scipy.least_squares"

        dof = max(1, y_obs.size - res.x.size)
        s_sq = sse / dof
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
        ci_obs = confidence_halfwidth(t_h, pcov, n_obs=len(t_h), alpha=0.05)

    except Exception:
        g = g_hill(t_h)
        A = np.column_stack([np.ones_like(g), g])
        x, *_ = np.linalg.lstsq(A, ars, rcond=None)
        a0, a1 = map(float, x)
        a0 = max(0.0, a0)
        a1 = max(0.0, a1)
        y_hat = model(t_h, a0, a1)
        sse = float(np.sum((y_hat - ars) ** 2))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((2, 2), dtype=float)
        ci_obs = np.zeros_like(t_h)

    ars_hat = model(t_h, a0, a1)
    return {
        "a0": a0,
        "a1": a1,
        "sse": sse,
        "method": method,
        "t_h": t_h,
        "ars_hat": ars_hat,
        "pcov": pcov,
        "ci_obs": ci_obs,
        "metrics_overall": _metrics(ars, ars_hat),
    }


def save_outputs(out_dir: Path, fit: dict, t_max_day: float, dt_day: float) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t_day_dense = np.arange(0.0, t_max_day + 1e-12, dt_day)
    t_h_dense = 24.0 * t_day_dense
    ars_dense = model(t_h_dense, fit["a0"], fit["a1"])

    csv_path = out_dir / "ars_fit_predictions.csv"
    mat = np.column_stack(
        [
            t_day_dense,
            t_h_dense,
            g_hill(t_h_dense),
            ars_dense,
            np.full_like(t_day_dense, TAU_H),
            np.full_like(t_day_dense, N_H),
            np.full_like(t_day_dense, fit["a0"]),
            np.full_like(t_day_dense, fit["a1"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header=(
            "day,hour,gating_g,ars_pred,"
            "tau_h,n_h,H0_baseline,osteo_contribution_scale"
        ),
        comments="",
    )

    js_path = out_dir / "ars_fit_metrics.json"
    payload = {
        "fit_method": fit["method"],
        "tau_H_h": TAU_H,
        "n_H": N_H,
        "H0_baseline": fit["a0"],
        "osteo_contribution_scale": fit["a1"],
        "metrics_overall": fit["metrics_overall"],
        "data": {
            "t_days": T_DAYS.tolist(),
            "ars": ARS.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    t_h_data = 24.0 * T_DAYS
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    c_ars = "#1f78b4"
    ax.plot(t_h_dense, ars_dense, color=c_ars, lw=2.5)
    ax.scatter(t_h_data, ARS, marker="s", s=56, color=c_ars, edgecolor="white", linewidth=0.8, zorder=3, clip_on=False)

    x_max_h = float(t_max_day) * 24.0
    x_pad_h = max(4.0, 0.02 * x_max_h)
    y_all = np.concatenate([ARS, ars_dense])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.5, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_xlim(-x_pad_h, x_max_h + x_pad_h)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.xaxis.set_major_locator(MultipleLocator(24.0 * 4.0))
    ax.yaxis.set_major_locator(MultipleLocator(5.0))
    ax.yaxis.set_minor_locator(MultipleLocator(2.5))

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Alizarin Red S (absorbance units)")
    ax.set_title("ARS Mineralization Fit", fontsize=11, pad=8)
    ax.tick_params(labelsize=9, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    met = fit["metrics_overall"]
    txt = "\n".join(
        [
            f"tau_H (h)      {TAU_H:.3g}",
            f"n_H            {N_H:.3g}",
            f"H0 baseline    {fit['a0']:.3g}",
            f"osteo contrib  {fit['a1']:.3g}",
            f"RMSE           {met['RMSE']:.3g}",
            f"R2             {met['R2']:.3g}",
        ]
    )
    ax.text(
        0.02, 0.70, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.6, family="monospace",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#bcbcbc", alpha=0.96),
    )

    legend_handles = [
        Line2D([0], [0], color=c_ars, lw=2.5, label="Mineralization fit"),
        Line2D(
            [0], [0], color=c_ars, lw=2.0, marker="s", markersize=7,
            markerfacecolor=c_ars, markeredgecolor="white", markeredgewidth=0.8,
            label="Mineralization data",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9, handlelength=2.0, borderaxespad=0.6)

    fig.tight_layout()
    png_path = out_dir / "ars_fit_style.png"
    pdf_path = out_dir / "ars_fit_style.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_max_day", type=float, default=28.0)
    ap.add_argument("--dt_day", type=float, default=0.1)
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "ars")
    args = ap.parse_args()

    fit = fit_ars(T_DAYS, ARS)
    csv_path, js_path, png_path, pdf_path = save_outputs(args.out_dir, fit, args.t_max_day, args.dt_day)

    print("=== ARS mineralization gating fit ===")
    print(f"fit_method      : {fit['method']}")
    print(f"tau_H           : {TAU_H:.6g} h (fixed)")
    print(f"n_H             : {N_H:.6g} (fixed)")
    print(f"H0_baseline     : {fit['a0']:.6g} abs units")
    print(f"osteo_contrib   : {fit['a1']:.6g} abs units")
    met = fit["metrics_overall"]
    print(f"SSE             : {met['SSE']:.6g}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")
    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
