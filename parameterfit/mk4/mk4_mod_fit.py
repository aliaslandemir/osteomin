#!/usr/bin/env python3
"""
MK-4 dose-response fit for ARS mineralization.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


MK4_DOSE_uM = np.array([0.0, 0.1, 1.0, 10.0], dtype=float)
ARS_MINERALIZATION = np.array([100.0, 420.0, 560.0, 270.0], dtype=float)

CK0_uM = 0.6


def gating_m(c_uM: np.ndarray, a_k: float, b_k: float, c_k0_uM: float = CK0_uM) -> np.ndarray:
    """m(c) = 1 + I(c>0) * (a_k - b_k * log10(c/c0)^2)."""
    c = np.asarray(c_uM, dtype=float)
    out = np.ones_like(c, dtype=float)
    mask = c > 0.0
    if np.any(mask):
        x = np.log10(c[mask] / float(c_k0_uM))
        raw = 1.0 + (a_k - b_k * x * x)
        out[mask] = np.maximum(0.0, raw)
    return out


def model(c_uM: np.ndarray, y0: float, a_k: float, b_k: float) -> np.ndarray:
    return float(y0) * gating_m(c_uM, a_k, b_k, CK0_uM)


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


def fit_mk4(c_uM: np.ndarray, y: np.ndarray) -> dict:
    c_uM = np.asarray(c_uM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_uM > 0.0
    c_fit = c_uM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_k and b_k.")

    try:
        from scipy.optimize import least_squares

        x0 = np.array([0.5, 0.5], dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(c_fit, y0, float(theta[0]), float(theta[1])) - y_fit

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        a_k, b_k = map(float, res.x)
        method = "scipy.least_squares"
        yhat_fit = model(c_fit, y0, a_k, b_k)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = max(1, y_fit.size - res.x.size)
        s_sq = sse / dof
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
    except Exception:
        x = np.log10(c_fit / CK0_uM)
        z = x * x
        r = (y_fit / y0) - 1.0
        A = np.column_stack([np.ones_like(z), -z])
        theta, *_ = np.linalg.lstsq(A, r, rcond=None)
        a_k = max(0.0, float(theta[0]))
        b_k = max(0.0, float(theta[1]))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((2, 2), dtype=float)

    yhat_all = model(c_uM, y0, a_k, b_k)
    return {
        "method": method,
        "y0": y0,
        "a_k": a_k,
        "b_k": b_k,
        "pcov": pcov,
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(y_fit, model(c_fit, y0, a_k, b_k)),
    }


def save_outputs(
    out_dir: Path,
    fit: dict,
    c_min_uM: float,
    c_max_uM: float,
    n_dense: int,
) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_doses = MK4_DOSE_uM[MK4_DOSE_uM > 0.0]
    if positive_doses.size:
        default_positive_start = max(
            float(np.min(positive_doses)), np.nextafter(0.0, 1.0)
        )
    else:
        default_positive_start = np.nextafter(0.0, 1.0)
    positive_start = c_min_uM if c_min_uM > 0.0 else default_positive_start
    positive_start = max(positive_start, np.nextafter(0.0, 1.0))

    c_dense_pos = np.logspace(np.log10(positive_start), np.log10(c_max_uM), int(n_dense))
    include_zero = c_min_uM <= 0.0
    if include_zero:
        c_dense_csv = np.concatenate(([0.0], c_dense_pos))
    else:
        c_dense_csv = c_dense_pos

    m_dense_csv = gating_m(c_dense_csv, fit["a_k"], fit["b_k"], CK0_uM)
    y_dense_csv = fit["y0"] * m_dense_csv
    log10_mk4 = np.full_like(c_dense_csv, float("nan"))
    mask_positive = c_dense_csv > 0.0
    log10_mk4[mask_positive] = np.log10(c_dense_csv[mask_positive] / CK0_uM)

    m_dense_pos = gating_m(c_dense_pos, fit["a_k"], fit["b_k"], CK0_uM)
    y_dense_pos = fit["y0"] * m_dense_pos

    csv_path = out_dir / "mk4_fit_predictions.csv"
    mat = np.column_stack(
        [
            c_dense_csv,
            log10_mk4,
            m_dense_csv,
            y_dense_csv,
            np.full_like(c_dense_csv, CK0_uM),
            np.full_like(c_dense_csv, fit["y0"]),
            np.full_like(c_dense_csv, fit["a_k"]),
            np.full_like(c_dense_csv, fit["b_k"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header=(
            "mk4_uM,log10_mk4_over_cK0,mk4_multiplier_pred,ars_pred,"
            "cK0_uM,ars_control_y0,aK,bK"
        ),
        comments="",
    )

    js_path = out_dir / "mk4_fit_metrics.json"
    payload = {
        "fit_method": fit["method"],
        "cK0_uM": CK0_uM,
        "ars_control_y0": fit["y0"],
        "aK": fit["a_k"],
        "bK": fit["b_k"],
        "metrics_overall": fit["metrics_overall"],
        "metrics_dosed": fit["metrics_dosed"],
        "data": {
            "mk4_uM": MK4_DOSE_uM.tolist(),
            "ars_mineralization_percent_control": ARS_MINERALIZATION.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c_mk4 = "#d95f02"
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(c_dense_pos, y_dense_pos, color=c_mk4, lw=2.4)
    positive_mask = MK4_DOSE_uM > 0.0
    if np.any(positive_mask):
        ax.scatter(
            MK4_DOSE_uM[positive_mask],
            ARS_MINERALIZATION[positive_mask],
            marker="o",
            s=54,
            color=c_mk4,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            clip_on=False,
        )
    control_x = 0.7 * positive_start
    ax.scatter(
        [control_x],
        [ARS_MINERALIZATION[0]],
        marker="o",
        s=54,
        color=c_mk4,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_xlim(c_dense_pos[0] * 0.9, c_dense_pos[-1] * 1.1)
    ax.set_xticks([control_x, 0.1, 1.0, 10.0])
    ax.set_xticklabels(["0", "0.1", "1", "10"])
    y_all = np.concatenate([ARS_MINERALIZATION, y_dense_pos])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.4, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("MK-4 concentration (uM, log scale)")
    ax.set_ylabel("Mineralization (% of control)")
    ax.set_title("MK-4 Dose-Response Fit", fontsize=11, pad=8)
    ax.tick_params(labelsize=9, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.axvline(CK0_uM, color="#777777", ls="--", lw=1.0, alpha=0.9)

    met = fit["metrics_dosed"]
    txt = "\n".join(
        [
            f"cK0 (uM)       {CK0_uM:.3g}",
            f"ARS control    {fit['y0']:.3g}",
            f"aK             {fit['a_k']:.3g}",
            f"bK             {fit['b_k']:.3g}",
            f"RMSE           {met['RMSE']:.3g}",
            f"R2             {met['R2']:.3g}",
        ]
    )
    ax.text(
        0.98,
        0.08,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=8.6,
        family="monospace",
        bbox=dict(boxstyle="round,pad=0.28", facecolor="white", edgecolor="#bcbcbc", alpha=0.96),
    )

    legend_handles = [
        Line2D([0], [0], color=c_mk4, lw=2.4, label="MK-4 fit"),
        Line2D(
            [0],
            [0],
            color=c_mk4,
            lw=2.0,
            marker="o",
            markersize=7,
            markerfacecolor=c_mk4,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="ARS data",
        ),
    ]
    ax.legend(handles=legend_handles, loc="upper left", frameon=False, fontsize=9, handlelength=2.0, borderaxespad=0.6)

    fig.tight_layout()
    png_path = out_dir / "mk4_fit_style.png"
    pdf_path = out_dir / "mk4_fit_style.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_min_uM", type=float, default=0.0)
    ap.add_argument("--c_max_uM", type=float, default=20.0)
    ap.add_argument("--n_dense", type=int, default=400)
    ap.add_argument("--out_dir", type=Path, default=Path(__file__).resolve().parent / "mk4_fit")
    args = ap.parse_args()

    if args.c_min_uM < 0.0:
        raise ValueError("--c_min_uM must be >= 0 for log-scale plotting and log-space predictions.")
    if args.c_max_uM <= args.c_min_uM:
        raise ValueError("--c_max_uM must be greater than --c_min_uM.")
    if args.n_dense < 20:
        raise ValueError("--n_dense must be >= 20.")

    fit = fit_mk4(MK4_DOSE_uM, ARS_MINERALIZATION)
    csv_path, js_path, png_path, pdf_path = save_outputs(
        args.out_dir,
        fit,
        args.c_min_uM,
        args.c_max_uM,
        args.n_dense,
    )

    met = fit["metrics_dosed"]
    print("=== MK-4 dose-response fit (ARS mineralization) ===")
    print(f"fit_method      : {fit['method']}")
    print(f"cK0_uM          : {CK0_uM:.6g}")
    print(f"ars_control_y0  : {fit['y0']:.6g} % of control")
    print(f"aK              : {fit['a_k']:.6g}")
    print(f"bK              : {fit['b_k']:.6g}")
    print(f"SSE             : {met['SSE']:.6g}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")

    print("dose_uM | ars_obs | ars_pred | multiplier")
    for c, y, yhat in zip(MK4_DOSE_uM, ARS_MINERALIZATION, fit["yhat_all"]):
        m = yhat / fit["y0"] if fit["y0"] > 0 else float("nan")
        print(f"{c:7.4g} | {y:7.4g} | {yhat:8.4g} | {m:9.4g}")

    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
