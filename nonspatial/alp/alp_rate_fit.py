#!/usr/bin/env python3
"""


Fit ALP control/osteogenic data and report hourly basal and osteogenic rates.


@article{Hong2010pro_alp,
  author    = {Hong, Dun and Chen, Hai-Xiao and Yu, Hai-Qiang and Liang, Yong and Wang, Carrie and Lian, Qing-Quan and Deng, Hai-Teng and Ge, Ren-Shan},
  title     = {Morphological and proteomic analysis of early stage of osteoblast differentiation in osteoblastic progenitor cells},
  journal   = {Experimental Cell Research},
  year      = {2010},
  volume    = {316},
  number    = {14},
  pages     = {2291--2300},
  doi       = {10.1016/j.yexcr.2010.05.011},
  pmid      = {20483354},
  pmcid     = {PMC4580249},
  publisher = {Elsevier}
}


"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


T_DAYS = np.array([3, 7, 14, 21], dtype=float)



ALP_CON = np.array(
    [
        60.0,   # day 3
        80.0,   # day 7
        95.0,   # day 14
        100.0,  # day 21
    ],
    dtype=float,
)

ALP_OS = np.array(
    [
        85.0,   # day 3
        260.0,  # day 7
        350.0,  # day 14
        510.0,  # day 21
    ],
    dtype=float,
)


def model(
    t_h: np.ndarray, p0: float, r_b: float, r_o: float
) -> tuple[np.ndarray, np.ndarray]:
    con = p0 + r_b * t_h
    os_ = p0 + (r_b + r_o) * t_h
    return con, os_


def _stacked_model(t_h_stacked: np.ndarray, p0: float, r_b: float, r_o: float) -> np.ndarray:
    n = t_h_stacked.size // 2
    t_h = t_h_stacked[:n]
    con_hat, os_hat = model(t_h, p0, r_b, r_o)
    return np.concatenate([con_hat, os_hat])


def confidence_halfwidth(
    t_h: np.ndarray, pcov: np.ndarray, n_obs: int, alpha: float = 0.05
) -> tuple[np.ndarray, np.ndarray]:
    """95% confidence half-width for fitted mean curves via parameter covariance."""
    p = 3
    dof = max(1, n_obs - p)
    try:
        from scipy.stats import t as student_t

        crit = float(student_t.ppf(1.0 - alpha / 2.0, dof))
    except Exception:
        crit = 1.96

    grad_con = np.column_stack(
        [np.ones_like(t_h, dtype=float), t_h, np.zeros_like(t_h, dtype=float)]
    )
    grad_os = np.column_stack([np.ones_like(t_h, dtype=float), t_h, t_h])
    var_con = np.einsum("ij,jk,ik->i", grad_con, pcov, grad_con)
    var_os = np.einsum("ij,jk,ik->i", grad_os, pcov, grad_os)
    hw_con = crit * np.sqrt(np.maximum(var_con, 0.0))
    hw_os = crit * np.sqrt(np.maximum(var_os, 0.0))
    return hw_con, hw_os


def fit_rates(t_days: np.ndarray, con: np.ndarray, os_: np.ndarray) -> dict:
    t_h = 24.0 * np.asarray(t_days, dtype=float)

    try:
        from scipy.optimize import curve_fit

        n = len(t_h)
        t_stacked = np.concatenate([t_h, t_h])
        y_stacked = np.concatenate([con, os_])
        x0 = np.array([max(0.0, min(con.min(), os_.min())), 0.1, 0.1], dtype=float)
        lb = np.array([0.0, 0.0, 0.0], dtype=float)
        ub = np.array([np.inf, np.inf, np.inf], dtype=float)
        popt, pcov = curve_fit(
            _stacked_model,
            t_stacked,
            y_stacked,
            p0=x0,
            bounds=(lb, ub),
            maxfev=10000,
        )
        p0, r_b, r_o = map(float, popt)
        y_hat = _stacked_model(t_stacked, p0, r_b, r_o)
        sse = float(np.sum((y_hat - y_stacked) ** 2))
        method = "scipy.curve_fit"
        ci_con_obs, ci_os_obs = confidence_halfwidth(t_h, pcov, n_obs=2 * n, alpha=0.05)
    except Exception:
        n = len(t_h)
        a = np.zeros((2 * n, 3), dtype=float)
        y = np.zeros((2 * n,), dtype=float)
        a[:n, 0] = 1.0
        a[:n, 1] = t_h
        y[:n] = con
        a[n:, 0] = 1.0
        a[n:, 1] = t_h
        a[n:, 2] = t_h
        y[n:] = os_
        x, *_ = np.linalg.lstsq(a, y, rcond=None)
        p0, r_b, r_o = map(float, x)
        p0 = max(0.0, p0)
        r_b = max(0.0, r_b)
        r_o = max(0.0, r_o)
        con_hat, os_hat = model(t_h, p0, r_b, r_o)
        sse = float(np.sum((con_hat - con) ** 2) + np.sum((os_hat - os_) ** 2))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((3, 3), dtype=float)
        ci_con_obs = np.zeros_like(t_h)
        ci_os_obs = np.zeros_like(t_h)

    con_hat, os_hat = model(t_h, p0, r_b, r_o)
    return {
        "p0": p0,
        "r_b_h": r_b,
        "r_o_h": r_o,
        "sse": sse,
        "method": method,
        "t_h": t_h,
        "con_hat": con_hat,
        "os_hat": os_hat,
        "pcov": pcov,
        "ci_con_obs": ci_con_obs,
        "ci_os_obs": ci_os_obs,
    }


def save_outputs(
    out_dir: Path, fit: dict, t_max_day: float, dt_day: float
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t_day_dense = np.arange(0.0, t_max_day + 1e-12, dt_day)
    t_h_dense = 24.0 * t_day_dense
    con_dense, os_dense = model(t_h_dense, fit["p0"], fit["r_b_h"], fit["r_o_h"])
    ci_con_dense, ci_os_dense = confidence_halfwidth(
        t_h_dense, fit["pcov"], n_obs=2 * len(fit["t_h"]), alpha=0.05
    )

    csv_path = out_dir / "alp_hourly_fit.csv"
    mat = np.column_stack(
        [
            t_day_dense,
            t_h_dense,
            con_dense,
            os_dense,
            np.maximum(con_dense - ci_con_dense, 0.0),
            con_dense + ci_con_dense,
            np.maximum(os_dense - ci_os_dense, 0.0),
            os_dense + ci_os_dense,
            np.full_like(t_day_dense, fit["r_b_h"]),
            np.full_like(t_day_dense, fit["r_o_h"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header=(
            "day,hour,alp_control_pred,alp_osteo_pred,"
            "alp_control_ci95_low,alp_control_ci95_high,"
            "alp_osteo_ci95_low,alp_osteo_ci95_high,"
            "r_basal_per_h,r_osteo_per_h"
        ),
        comments="",
    )

    png_path = out_dir / "alp_hourly_fit.png"
    try:
        import matplotlib.pyplot as plt

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(8.2, 5.1))
        ax.fill_between(
            t_day_dense,
            np.maximum(con_dense - ci_con_dense, 0.0),
            con_dense + ci_con_dense,
            color="#1f77b4",
            alpha=0.18,
            linewidth=0.0,
            label="Control 95% CI",
        )
        ax.fill_between(
            t_day_dense,
            np.maximum(os_dense - ci_os_dense, 0.0),
            os_dense + ci_os_dense,
            color="#d62728",
            alpha=0.15,
            linewidth=0.0,
            label="Osteogenic 95% CI",
        )
        ax.plot(t_day_dense, con_dense, color="#1f77b4", lw=2.6, label="Control fit")
        ax.plot(t_day_dense, os_dense, color="#d62728", lw=2.6, label="Osteogenic fit")
        ax.scatter(
            T_DAYS,
            ALP_CON,
            color="#1f77b4",
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=3,
            label="Control data",
        )
        ax.scatter(
            T_DAYS,
            ALP_OS,
            color="#d62728",
            edgecolor="white",
            linewidth=0.8,
            s=48,
            zorder=3,
            label="Osteogenic data",
        )
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("ALP signal (assay units)")
        ax.set_title("ALP Rate Fit with 95% Confidence Bands")
        ax.grid(alpha=0.3, linewidth=0.8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(float(np.min(t_day_dense)), float(np.max(t_day_dense)))
        ax.legend(loc="upper left", ncol=2, frameon=True, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(png_path, dpi=220)
        plt.close(fig)
    except Exception:
        png_path = Path("")

    return csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_max_day", type=float, default=21.0)
    ap.add_argument("--dt_day", type=float, default=0.1)
    ap.add_argument(
        "--out_dir", type=Path, default=Path(__file__).resolve().parent / "alp"
    )
    args = ap.parse_args()

    fit = fit_rates(T_DAYS, ALP_CON, ALP_OS)
    csv_path, png_path = save_outputs(args.out_dir, fit, args.t_max_day, args.dt_day)

    print("=== ALP hourly fit ===")
    print(f"fit_method      : {fit['method']}")
    print(f"P0              : {fit['p0']:.6g} assay units")
    print(f"r_basal_P       : {fit['r_b_h']:.6g} assay units/hour")
    print(f"r_osteo_P       : {fit['r_o_h']:.6g} assay units/hour")
    final_day = T_DAYS[-1]
    final_idx = -1
    print(f"SSE             : {fit['sse']:.6g}")
    print(
        f"final_day       : {final_day:.0f} d | "
        f"CON obs={ALP_CON[final_idx]:.6g}, pred={fit['con_hat'][final_idx]:.6g} | "
        f"OS obs={ALP_OS[final_idx]:.6g}, pred={fit['os_hat'][final_idx]:.6g}"
    )
    print(f"saved_csv       : {csv_path}")
    if str(png_path):
        print(f"saved_plot      : {png_path}")
    else:
        print("saved_plot      : matplotlib unavailable (plot not generated)")


if __name__ == "__main__":
    main()
