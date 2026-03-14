#!/usr/bin/env python3
"""
Phenomenological Dex dose-response fit for ALP using a symmetric
log-parabolic model centered at 20 nM.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np



DEX_DOSE_nM = np.array([0.0, 1.0, 10.0, 100.0], dtype=float)
ALP_DAY5_DEX = np.array([0.8, 1.2, 3.6, 3.2], dtype=float)

CD0_nM = 20.0

def dex_modulation_mP(
    c_dex_nM: np.ndarray, a_d: float, b_d: float, c_d0_nM: float = CD0_nM
) -> np.ndarray:
    """m_D^P(c_D) = 1 + I(c_D>0) * (a_D - b_D * log10(c_D/c_D0)^2)."""
    c = np.asarray(c_dex_nM, dtype=float)
    out = np.ones_like(c, dtype=float)
    mask = c > 0.0
    if np.any(mask):
        x_d = np.log10(c[mask] / float(c_d0_nM))
        raw = 1.0 + (a_d - b_d * x_d * x_d)
        out[mask] = np.maximum(0.0, raw)
    return out


def model(c_dex_nM: np.ndarray, y0: float, a_d: float, b_d: float) -> np.ndarray:
    return float(y0) * dex_modulation_mP(c_dex_nM, a_d, b_d, CD0_nM)


def _metrics(y: np.ndarray, yhat: np.ndarray) -> Dict[str, float]:
    y = np.asarray(y, dtype=float)
    yhat = np.asarray(yhat, dtype=float)
    resid = y - yhat
    sse = float(np.sum(resid**2))
    rmse = float(np.sqrt(np.mean(resid**2)))
    mae = float(np.mean(np.abs(resid)))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")
    return {"SSE": sse, "RMSE": rmse, "MAE": mae, "R2": r2}


def fit_dex(c_dex_nM: np.ndarray, y: np.ndarray) -> dict:
    c_dex_nM = np.asarray(c_dex_nM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_dex_nM > 0.0
    c_fit = c_dex_nM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_d and b_d.")

    try:
        from scipy.optimize import least_squares

        x0 = np.array([0.8, 0.6], dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(c_fit, y0, float(theta[0]), float(theta[1])) - y_fit

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        a_d, b_d = map(float, res.x)
        method = "scipy.least_squares"
        yhat_fit = model(c_fit, y0, a_d, b_d)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = max(1, y_fit.size - res.x.size)
        s_sq = sse / dof
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
    except Exception:
        x = np.log10(c_fit / CD0_nM)
        z = x * x
        r = (y_fit / y0) - 1.0
        A = np.column_stack([np.ones_like(z), -z])
        theta, *_ = np.linalg.lstsq(A, r, rcond=None)
        a_d = max(0.0, float(theta[0]))
        b_d = max(0.0, float(theta[1]))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((2, 2), dtype=float)

    yhat_all = model(c_dex_nM, y0, a_d, b_d)
    return {
        "method": method,
        "y0": y0,
        "a_d": a_d,
        "b_d": b_d,
        "pcov": pcov,
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(y_fit, model(c_fit, y0, a_d, b_d)),
    }


def save_outputs(
    out_dir: Path,
    fit: dict,
    c_min_nM: float,
    c_max_nM: float,
    n_dense: int,
) -> tuple[Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_doses = DEX_DOSE_nM[DEX_DOSE_nM > 0.0]
    if positive_doses.size:
        default_positive_start = max(float(np.min(positive_doses)), np.nextafter(0.0, 1.0))
    else:
        default_positive_start = np.nextafter(0.0, 1.0)
    positive_start = c_min_nM if c_min_nM > 0.0 else default_positive_start
    positive_start = max(positive_start, np.nextafter(0.0, 1.0))

    c_dense_pos = np.logspace(np.log10(positive_start), np.log10(c_max_nM), int(n_dense))
    include_zero = c_min_nM <= 0.0
    if include_zero:
        c_dense_csv = np.concatenate(([0.0], c_dense_pos))
    else:
        c_dense_csv = c_dense_pos

    m_dense_csv = dex_modulation_mP(c_dense_csv, fit["a_d"], fit["b_d"], CD0_nM)
    y_dense_csv = fit["y0"] * m_dense_csv
    log10_dex = np.full_like(c_dense_csv, float("nan"))
    mask_positive = c_dense_csv > 0.0
    log10_dex[mask_positive] = np.log10(c_dense_csv[mask_positive] / CD0_nM)

    m_dense_pos = dex_modulation_mP(c_dense_pos, fit["a_d"], fit["b_d"], CD0_nM)
    y_dense_pos = fit["y0"] * m_dense_pos

    csv_path = out_dir / "dex_fit_predictions.csv"
    mat = np.column_stack(
        [
            c_dense_csv,
            log10_dex,
            m_dense_csv,
            y_dense_csv,
            np.full_like(c_dense_csv, CD0_nM),
            np.full_like(c_dense_csv, fit["y0"]),
            np.full_like(c_dense_csv, fit["a_d"]),
            np.full_like(c_dense_csv, fit["b_d"]),
        ]
    )
    np.savetxt(
        csv_path,
        mat,
        delimiter=",",
        header=(
            "dex_nM,log10_dex_over_cD0,dex_multiplier_pred,alp_pred,"
            "cD0_nM,alp_control_y0,aD,bD"
        ),
        comments="",
    )

    js_path = out_dir / "dex_fit_metrics.json"
    payload = {
        "fit_method": fit["method"],
        "cD0_nM": CD0_nM,
        "alp_control_y0": fit["y0"],
        "aD": fit["a_d"],
        "bD": fit["b_d"],
        "metrics_overall": fit["metrics_overall"],
        "metrics_dosed": fit["metrics_dosed"],
        "data": {
            "dex_nM": DEX_DOSE_nM.tolist(),
            "alp_day5_dex": ALP_DAY5_DEX.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c_dex = "#d95f02"
    fig = plt.figure(figsize=(7.8, 4.8))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(c_dense_pos, y_dense_pos, color=c_dex, lw=2.4)
    positive_mask = DEX_DOSE_nM > 0.0
    if np.any(positive_mask):
        ax.scatter(
            DEX_DOSE_nM[positive_mask],
            ALP_DAY5_DEX[positive_mask],
            marker="o",
            s=54,
            color=c_dex,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
            clip_on=False,
        )
    control_x = 0.7 * positive_start
    ax.scatter(
        [control_x],
        [ALP_DAY5_DEX[0]],
        marker="o",
        s=54,
        color=c_dex,
        edgecolor="white",
        linewidth=0.8,
        zorder=3,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_xlim(c_dense_pos[0] * 0.9, c_dense_pos[-1] * 1.1)
    ax.set_xticks([control_x, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", "1", "10", "100"])
    y_all = np.concatenate([ALP_DAY5_DEX, y_dense_pos])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.4, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("Dex concentration (nM, log scale)")
    ax.set_ylabel("ALP activity (nmol/min/μg protein)")
    ax.set_title("Dex Dose-Response Fit", fontsize=11, pad=8)
    ax.tick_params(labelsize=9, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.axvline(CD0_nM, color="#777777", ls="--", lw=1.0, alpha=0.9)

    met = fit["metrics_dosed"]
    txt = "\n".join(
        [
            f"cD0 (nM)       {CD0_nM:.3g}",
            f"ALP control    {fit['y0']:.3g}",
            f"aD             {fit['a_d']:.3g}",
            f"bD             {fit['b_d']:.3g}",
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
        bbox=dict(
            boxstyle="round,pad=0.28",
            facecolor="white",
            edgecolor="#bcbcbc",
            alpha=0.96,
        ),
    )

    legend_handles = [
        Line2D([0], [0], color=c_dex, lw=2.4, label="Dex fit"),
        Line2D(
            [0],
            [0],
            color=c_dex,
            lw=2.0,
            marker="o",
            markersize=7,
            markerfacecolor=c_dex,
            markeredgecolor="white",
            markeredgewidth=0.8,
            label="ALP day-5 data",
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
    png_path = out_dir / "dex_fit_style.png"
    pdf_path = out_dir / "dex_fit_style.pdf"
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_min_nM", type=float, default=0.0)
    ap.add_argument("--c_max_nM", type=float, default=100.0)
    ap.add_argument("--n_dense", type=int, default=400)
    ap.add_argument(
        "--out_dir", type=Path, default=Path(__file__).resolve().parent / "dex_fit"
    )
    args = ap.parse_args()

    if args.c_min_nM < 0.0:
        raise ValueError(
            "--c_min_nM must be >= 0 for log-scale plotting and log-space predictions."
        )
    if args.c_max_nM <= args.c_min_nM:
        raise ValueError("--c_max_nM must be greater than --c_min_nM.")
    if args.n_dense < 20:
        raise ValueError("--n_dense must be >= 20.")

    fit = fit_dex(DEX_DOSE_nM, ALP_DAY5_DEX)
    csv_path, js_path, png_path, pdf_path = save_outputs(
        args.out_dir,
        fit,
        args.c_min_nM,
        args.c_max_nM,
        args.n_dense,
    )

    met = fit["metrics_dosed"]
    print("=== Dex dose-response fit (ALP day 5) ===")
    print(f"fit_method      : {fit['method']}")
    print(f"cD0_nM          : {CD0_nM:.6g}")
    print(f"alp_control_y0  : {fit['y0']:.6g} U/g")
    print(f"aD              : {fit['a_d']:.6g}")
    print(f"bD              : {fit['b_d']:.6g}")
    print(f"SSE             : {met['SSE']:.6g}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")

    print("dose_nM | alp_obs | alp_pred | multiplier")
    for c, y, yhat in zip(DEX_DOSE_nM, ALP_DAY5_DEX, fit["yhat_all"]):
        m = yhat / fit["y0"] if fit["y0"] > 0 else float("nan")
        print(f"{c:7.4g} | {y:7.4g} | {yhat:8.4g} | {m:9.4g}")

    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
