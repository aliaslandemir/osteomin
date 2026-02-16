#!/usr/bin/env python3
"""Fit ALP control/osteogenic data and report hourly basal and osteogenic rates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


T_DAYS = np.array([0, 3, 5, 7, 10, 14], dtype=float)

ALP_CON = np.array(
    [
        55.0000,  # day 0 (approx baseline)
        58.2210,  # day 3
        70.0000,  # day 5 (interpolated)
        79.2453,  # day 7
        82.0000,  # day 10 (interpolated)
        84.0970,  # day 14
    ],
    dtype=float,
)

ALP_OS = np.array(
    [
        60.0000,  # day 0 (approx baseline)
        80.8625,  # day 3
        200.0000,  # day 5 (estimated from figure trend)
        253.9084,  # day 7
        430.0000,  # day 10 (estimated)
        511.0512,  # day 14
    ],
    dtype=float,
)


def model(
    t_h: np.ndarray, p0: float, r_b: float, r_o: float
) -> tuple[np.ndarray, np.ndarray]:
    con = p0 + r_b * t_h
    os_ = p0 + (r_b + r_o) * t_h
    return con, os_


def fit_rates(t_days: np.ndarray, con: np.ndarray, os_: np.ndarray) -> dict:
    t_h = 24.0 * np.asarray(t_days, dtype=float)

    try:
        from scipy.optimize import least_squares

        def residuals(x: np.ndarray) -> np.ndarray:
            p0, r_b, r_o = x
            con_hat, os_hat = model(t_h, p0, r_b, r_o)
            return np.concatenate([con_hat - con, os_hat - os_])

        x0 = np.array([max(0.0, min(con.min(), os_.min())), 0.1, 0.1], dtype=float)
        lb = np.array([0.0, 0.0, 0.0], dtype=float)
        ub = np.array([np.inf, np.inf, np.inf], dtype=float)
        res = least_squares(residuals, x0=x0, bounds=(lb, ub), method="trf")
        p0, r_b, r_o = map(float, res.x)
        sse = float(np.sum(res.fun**2))
        method = "scipy.least_squares"
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
    }


def save_outputs(
    out_dir: Path, fit: dict, t_max_day: float, dt_day: float
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    t_day_dense = np.arange(0.0, t_max_day + 1e-12, dt_day)
    t_h_dense = 24.0 * t_day_dense
    con_dense, os_dense = model(t_h_dense, fit["p0"], fit["r_b_h"], fit["r_o_h"])

    csv_path = out_dir / "alp_hourly_fit.csv"
    mat = np.column_stack(
        [
            t_day_dense,
            t_h_dense,
            con_dense,
            os_dense,
            np.full_like(t_day_dense, fit["r_b_h"]),
            np.full_like(t_day_dense, fit["r_o_h"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header="day,hour,alp_control_pred,alp_osteo_pred,r_basal_per_h,r_osteo_per_h",
        comments="",
    )

    png_path = out_dir / "alp_hourly_fit.png"
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.scatter(T_DAYS, ALP_CON, label="Control data", color="#1f77b4")
        ax.scatter(T_DAYS, ALP_OS, label="Osteogenic data", color="#d62728")
        ax.plot(t_day_dense, con_dense, color="#1f77b4", lw=2)
        ax.plot(t_day_dense, os_dense, color="#d62728", lw=2)
        ax.set_xlabel("Time (day)")
        ax.set_ylabel("ALP signal (assay units)")
        ax.set_title("ALP standardized fit (hourly basal + osteogenic rates)")
        ax.grid(alpha=0.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(png_path, dpi=180)
        plt.close(fig)
    except Exception:
        png_path = Path("")

    return csv_path, png_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t_max_day", type=float, default=14.0)
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
