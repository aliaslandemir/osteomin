#!/usr/bin/env python3
"""ALP rate fit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


T_DAYS = np.array([3, 7, 14, 21], dtype=float)

ALP_CON = np.array([60.0, 80.0, 95.0, 100.0], dtype=float)
ALP_OS  = np.array([85.0, 260.0, 350.0, 510.0], dtype=float)


def model(t_days: np.ndarray, p0: float, r_b_h: float, r_o_h: float, *, t0_day: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return predicted (control, osteogenic) ALP at given days."""
    tau_h = 24.0 * (np.asarray(t_days, dtype=float) - float(t0_day))
    tau_h = np.maximum(tau_h, 0.0)
    con = p0 + r_b_h * tau_h
    os_ = p0 + (r_b_h + r_o_h) * tau_h
    return con, os_


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    """Compute common regression metrics."""
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    mae = float(np.mean(np.abs(resid)))
    rmse = float(np.sqrt(np.mean(resid**2)))
    sst = float(np.sum((y - np.mean(y))**2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"SSE": sse, "MAE": mae, "RMSE": rmse, "R2": r2}


def fit_single_baseline(
    t_days: np.ndarray,
    con: np.ndarray,
    os_: np.ndarray,
    *,
    t0_day: float = 0.0,
    nonneg: bool = True,
) -> Dict:
    """
    Least-squares fit for p0, r_b and r_o with a single shared baseline at t0_day.
    This supports t0_day values that are not present in the sampled timepoints.
    """
    t_days = np.asarray(t_days, dtype=float)
    con = np.asarray(con, dtype=float)
    os_ = np.asarray(os_, dtype=float)

    tau_h = 24.0 * (t_days - float(t0_day))
    mask = tau_h >= 0
    tau = tau_h[mask]
    if tau.size < 1:
        raise ValueError("Need at least one timepoint at or after t0_day to fit rates.")

    # Stacked linear system:
    # con = p0 + r_b*tau
    # os  = p0 + r_b*tau + r_o*tau
    a_con = np.column_stack([np.ones_like(tau), tau, np.zeros_like(tau)])
    a_os = np.column_stack([np.ones_like(tau), tau, tau])
    A = np.vstack([a_con, a_os])
    y = np.concatenate([con[mask], os_[mask]])
    beta, _, _, _ = np.linalg.lstsq(A, y, rcond=None)

    p0 = float(beta[0])
    r_b = float(beta[1])
    r_o = float(beta[2])

    if nonneg:
        r_b = max(0.0, r_b)
        r_o = max(0.0, r_o)

    con_hat, os_hat = model(t_days, p0, r_b, r_o, t0_day=t0_day)

    y_all = np.concatenate([con, os_])
    yhat_all = np.concatenate([con_hat, os_hat])
    met_all = _metrics(y_all, yhat_all)
    met_con = _metrics(con, con_hat)
    met_os  = _metrics(os_, os_hat)

    return {
        "t0_day": float(t0_day),
        "p0": float(p0),
        "r_b_h": float(r_b),
        "r_o_h": float(r_o),
        "tau_h": tau_h,
        "con_hat": con_hat,
        "os_hat": os_hat,
        "metrics_overall": met_all,
        "metrics_con": met_con,
        "metrics_os": met_os,
    }


def save_predictions_csv(out_dir: Path, fit: Dict, *, t_max_day: float, dt_day: float) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    t_day = np.arange(0.0, float(t_max_day) + 1e-12, float(dt_day))
    tau_h = 24.0 * (t_day - fit["t0_day"])
    tau_h = np.maximum(tau_h, 0.0)
    con, os_ = model(t_day, fit["p0"], fit["r_b_h"], fit["r_o_h"], t0_day=fit["t0_day"])

    csv_path = out_dir / "alp_fit_predictions.csv"
    mat = np.column_stack(
        [
            t_day,
            24.0 * t_day,
            tau_h,
            con,
            os_,
            np.full_like(t_day, fit["p0"]),
            np.full_like(t_day, fit["r_b_h"]),
            np.full_like(t_day, fit["r_o_h"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header="day,hour,tau_h,alp_con_pred,alp_os_pred,P0_at_t0,r_basal_per_h,r_osteo_extra_per_h",
        comments="",
    )
    return csv_path


def save_metrics_json(out_dir: Path, fit: Dict) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    js_path = out_dir / "alp_fit_metrics.json"
    payload = {
        "t0_day": fit["t0_day"],
        "P0_at_t0": fit["p0"],
        "r_basal_per_h": fit["r_b_h"],
        "r_osteo_extra_per_h": fit["r_o_h"],
        "metrics_overall": fit["metrics_overall"],
        "metrics_control": fit["metrics_con"],
        "metrics_osteo": fit["metrics_os"],
        "data": {
            "t_days": T_DAYS.tolist(),
            "alp_con": ALP_CON.tolist(),
            "alp_os": ALP_OS.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))
    return js_path


def make_style_plot(out_dir: Path, fit: Dict, *, t_max_day: float, dt_day: float) -> Tuple[Path, Path]:
    """Render fitted curves and observations."""
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import MultipleLocator

    t_day = np.arange(0.0, float(t_max_day) + 1e-12, float(dt_day))
    t_hour = 24.0 * t_day
    con_dense, os_dense = model(t_day, fit["p0"], fit["r_b_h"], fit["r_o_h"], t0_day=fit["t0_day"])

    t_data_hour = 24.0 * T_DAYS

    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    c_basal = "#1f78b4"
    c_osteo = "#d95f02"

    ax.plot(t_hour, con_dense, lw=2.5, color=c_basal, label="Control fit")
    ax.plot(t_hour, os_dense, lw=2.5, color=c_osteo, label="Osteogenic fit")

    ax.scatter(
        t_data_hour, ALP_CON, s=56, color=c_basal, edgecolor="white",
        linewidth=0.8, zorder=3, clip_on=False, label="Control data"
    )
    ax.scatter(
        t_data_hour, ALP_OS, s=56, color=c_osteo, edgecolor="white",
        linewidth=0.8, zorder=3, clip_on=False, label="Osteogenic data"
    )

    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("ALP activity (assay units)")
    x_max_h = float(t_max_day) * 24.0
    x_pad_h = max(6.0, 0.02 * x_max_h)
    y_all = np.concatenate([ALP_CON, ALP_OS, con_dense, os_dense])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(8.0, 0.06 * (y_max - y_min if y_max > y_min else 1.0))

    ax.set_xlim(-x_pad_h, x_max_h + x_pad_h)
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)
    ax.xaxis.set_major_locator(MultipleLocator(24.0 * 3.0))
    y_lo, y_hi = ax.get_ylim()
    y_start = 50.0 * np.floor(y_lo / 50.0)
    y_stop = 50.0 * np.ceil(y_hi / 50.0)
    y_ticks = np.arange(y_start, y_stop + 1e-12, 50.0)

    ax.set_yticks(y_ticks)
    ax.yaxis.set_minor_locator(MultipleLocator(25.0))

    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.set_title("ALP Activity Fit", fontsize=11, pad=8)
    ax.tick_params(labelsize=9, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.xaxis.label.set_size(10.5)
    ax.yaxis.label.set_size(10.5)

    met = fit["metrics_overall"]
    txt = "\n".join(
        [
            f"P₀ (AU)        {fit['p0']:.3g}",
            f"r_b (AU h⁻¹)   {fit['r_b_h']:.3g}",
            f"r_o (AU h⁻¹)   {fit['r_o_h']:.3g}",
            f"RMSE (AU)      {met['RMSE']:.3g}",
            f"R²             {met['R2']:.3g}",
        ]
    )
    ax.text(
        0.02, 0.70, txt,
        transform=ax.transAxes,
        va="top", ha="left",
        fontsize=8.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#bcbcbc", alpha=0.96),
    )

    legend_handles = [
        Line2D([0], [0], color=c_basal, lw=2.5, label="Basal fit"),
        Line2D([0], [0], color=c_osteo, lw=2.5, label="Osteogenic fit"),
        Line2D(
            [0], [0], color=c_basal, lw=2.0, marker="o", markersize=7,
            markerfacecolor=c_basal, markeredgecolor="white", markeredgewidth=0.8,
            label="Basal data",
        ),
        Line2D(
            [0], [0], color=c_osteo, lw=2.0, marker="o", markersize=7,
            markerfacecolor=c_osteo, markeredgecolor="white", markeredgewidth=0.8,
            label="Osteogenic data",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=9,
        handlelength=2.0,
        borderaxespad=0.6,
    )

    fig.tight_layout()

    png_path = out_dir / "alp_fit_style.png"
    pdf_path = out_dir / "alp_fit_style.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t0_day", type=float, default=0.0)
    ap.add_argument("--t_max_day", type=float, default=21.0)
    ap.add_argument("--dt_day", type=float, default=0.1)
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "alp_fit")
    ap.add_argument("--allow_negative", action="store_true", help="Allow negative fitted rates (default clamps to >=0).")
    args = ap.parse_args()

    fit = fit_single_baseline(
        T_DAYS, ALP_CON, ALP_OS,
        t0_day=args.t0_day,
        nonneg=(not args.allow_negative),
    )

    out_dir: Path = args.out_dir
    csv_path = save_predictions_csv(out_dir, fit, t_max_day=args.t_max_day, dt_day=args.dt_day)
    js_path = save_metrics_json(out_dir, fit)
    png_path, pdf_path = make_style_plot(out_dir, fit, t_max_day=args.t_max_day, dt_day=args.dt_day)

    met = fit["metrics_overall"]
    print("=== ALP fit (single baseline at t0_day) ===")
    print(f"t0_day          : {fit['t0_day']:.0f} d")
    print(f"P0 (at t0)      : {fit['p0']:.6g} assay units")
    print(f"r_basal_P       : {fit['r_b_h']:.6g} assay units/hour")
    print(f"r_osteo_extra_P : {fit['r_o_h']:.6g} assay units/hour")
    print(f"SSE             : {met['SSE']:.6g}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"MAE             : {met['MAE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")

    final_idx = -1
    con_hat, os_hat = fit["con_hat"], fit["os_hat"]
    print(
        f"final_day       : {T_DAYS[final_idx]:.0f} d | "
        f"CON obs={ALP_CON[final_idx]:.6g}, pred={con_hat[final_idx]:.6g} | "
        f"OS obs={ALP_OS[final_idx]:.6g}, pred={os_hat[final_idx]:.6g}"
    )

    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
