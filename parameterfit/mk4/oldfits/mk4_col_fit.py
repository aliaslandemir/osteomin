#!/usr/bin/env python3
"""
MK-4 dose-response fit for collagen response with shape fixed from MK-4 ARS.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


MK4_DOSE_uM = np.array([0.0, 10.0], dtype=float)
COL_RESPONSE = np.array([1.0, 1.10], dtype=float)

# Shape borrowed from mk4_ars_grid_fit metrics.
ARS_GRID_AK = 4.663134844322316
ARS_GRID_BK = 2.1451658350203324


def gating_m(
    c_uM: np.ndarray, a_k: float, b_k: float, c_k0_uM: float
) -> np.ndarray:
    """m(c) = 1 + I(c>0) * (a_k - b_k * log10(c/c0)^2)."""
    c = np.asarray(c_uM, dtype=float)
    out = np.ones_like(c, dtype=float)
    mask = c > 0.0
    if np.any(mask):
        x = np.log10(c[mask] / float(c_k0_uM))
        raw = 1.0 + (a_k - b_k * x * x)
        out[mask] = np.maximum(0.0, raw)
    return out


def model(c_uM: np.ndarray, y0: float, c_k0_uM: float) -> np.ndarray:
    return float(y0) * gating_m(c_uM, ARS_GRID_AK, ARS_GRID_BK, c_k0_uM)


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


def fit_mk4(c_uM: np.ndarray, y: np.ndarray) -> dict:
    c_uM = np.asarray(c_uM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_uM > 0.0
    c_fit = c_uM[mask]
    y_fit = y[mask]

    if c_fit.size < 1:
        raise ValueError("Need at least one non-zero dose point to fit c_k0_uM.")

    try:
        from scipy.optimize import least_squares

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(c_fit, y0, float(theta[0])) - y_fit

        res = least_squares(
            residuals,
            x0=np.array([0.35], dtype=float),
            bounds=(np.nextafter(0.0, 1.0), np.inf),
            max_nfev=10000,
        )
        c_k0_uM = float(res.x[0])
        method = "scipy.least_squares_fixed_aK_bK"
        yhat_fit = model(c_fit, y0, c_k0_uM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - res.x.size
        s_sq = sse / max(1, dof)
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
    except Exception:
        target_multiplier = float(y_fit[0] / y0)
        rhs = (1.0 + ARS_GRID_AK - target_multiplier) / ARS_GRID_BK
        rhs = max(0.0, rhs)
        log_offset = np.sqrt(rhs)
        # Choose the lower-peak branch so the peak stays inside the studied range.
        c_k0_uM = float(c_fit[0] / (10.0**log_offset))
        method = "analytic_fixed_aK_bK_lower_branch"
        pcov = np.zeros((1, 1), dtype=float)
        dof = y_fit.size - 1

    yhat_all = model(c_uM, y0, c_k0_uM)
    return {
        "method": method,
        "y0": y0,
        "a_k": ARS_GRID_AK,
        "b_k": ARS_GRID_BK,
        "c_k0_uM": c_k0_uM,
        "pcov": pcov,
        "dof": int(dof),
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(y_fit, model(c_fit, y0, c_k0_uM)),
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

    c_dense_pos = np.logspace(
        np.log10(positive_start), np.log10(c_max_uM), int(n_dense)
    )
    include_zero = c_min_uM <= 0.0
    c_dense_csv = np.concatenate(([0.0], c_dense_pos)) if include_zero else c_dense_pos

    m_dense_csv = gating_m(
        c_dense_csv, fit["a_k"], fit["b_k"], fit["c_k0_uM"]
    )
    y_dense_csv = fit["y0"] * m_dense_csv
    log10_mk4 = np.full_like(c_dense_csv, float("nan"))
    mask_positive = c_dense_csv > 0.0
    log10_mk4[mask_positive] = np.log10(c_dense_csv[mask_positive] / fit["c_k0_uM"])

    m_dense_pos = gating_m(
        c_dense_pos, fit["a_k"], fit["b_k"], fit["c_k0_uM"]
    )
    y_dense_pos = fit["y0"] * m_dense_pos

    csv_path = out_dir / "mk4_col_fit_predictions.csv"
    mat = np.column_stack(
        [
            c_dense_csv,
            log10_mk4,
            m_dense_csv,
            y_dense_csv,
            np.full_like(c_dense_csv, fit["c_k0_uM"]),
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
            "mk4_uM,log10_mk4_over_cK0,mk4_multiplier_pred,col_pred,"
            "cK0_uM,col_control_y0,aK,bK"
        ),
        comments="",
    )

    js_path = out_dir / "mk4_col_fit_metrics.json"
    payload = {
        "fit_method": fit["method"],
        "shape_source": "mk4_ars_grid_fit",
        "cK0_uM": fit["c_k0_uM"],
        "col_control_y0": fit["y0"],
        "aK": fit["a_k"],
        "bK": fit["b_k"],
        "degrees_of_freedom": fit["dof"],
        "metrics_overall": fit["metrics_overall"],
        "metrics_dosed": fit["metrics_dosed"],
        "data": {
            "mk4_uM": MK4_DOSE_uM.tolist(),
            "collagen_response": COL_RESPONSE.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c_mk4 = "#1f77b4"
    cm_to_in = 1.0 / 2.54
    fig = plt.figure(figsize=(8.5 * cm_to_in, 6.6 * cm_to_in))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(c_dense_pos, y_dense_pos, color=c_mk4, lw=1.2)
    positive_mask = MK4_DOSE_uM > 0.0
    if np.any(positive_mask):
        ax.scatter(
            MK4_DOSE_uM[positive_mask],
            COL_RESPONSE[positive_mask],
            marker="o",
            s=30,
            color=c_mk4,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            clip_on=False,
        )
    control_x = 0.7 * positive_start
    ax.scatter(
        [control_x],
        [COL_RESPONSE[0]],
        marker="o",
        s=30,
        color=c_mk4,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_xlim(c_dense_pos[0] * 0.9, c_dense_pos[-1] * 1.1)
    ax.set_xticks([control_x, 0.1, 1.0, 10.0])
    ax.set_xticklabels(["0", "0.1", "1", "10"])
    y_all = np.concatenate([COL_RESPONSE, y_dense_pos])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.4, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("MK-4 concentration (uM, log scale)")
    ax.set_ylabel("Collagen response")
    ax.set_title("MK-4 Collagen Fit", fontsize=9, pad=4)
    ax.tick_params(labelsize=8, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.axvline(fit["c_k0_uM"], color="#777777", ls="--", lw=1.0, alpha=0.9)

    met = fit["metrics_dosed"]
    txt = "\n".join(
        [
            f"cK0 = {fit['c_k0_uM']:.2f} uM",
            f"y0 = {fit['y0']:.2f}",
            f"aK = {fit['a_k']:.2f}",
            f"bK = {fit['b_k']:.2f}",
            f"DoF = {fit['dof']}",
            f"RMSE = {met['RMSE']:.2e}",
        ]
    )
    ax.text(
        0.98,
        0.08,
        txt,
        transform=ax.transAxes,
        va="bottom",
        ha="right",
        fontsize=7.2,
        bbox=dict(
            boxstyle="round,pad=0.22",
            facecolor="white",
            edgecolor="#bcbcbc",
            alpha=0.90,
        ),
    )

    legend_handles = [
        Line2D([0], [0], color=c_mk4, lw=1.2, label="MK-4 fit"),
        Line2D([0], [0], color="#777777", lw=1.0, ls="--", label="cK0"),
        Line2D(
            [0],
            [0],
            color=c_mk4,
            lw=1.2,
            marker="o",
            markersize=5.5,
            markerfacecolor=c_mk4,
            markeredgecolor="white",
            markeredgewidth=0.6,
            label="Collagen data",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        handlelength=1.8,
        borderaxespad=0.4,
    )

    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    fig.tight_layout(pad=0.4)
    png_path = out_dir / "mk4_col_fit_style.png"
    pdf_path = out_dir / "mk4_col_fit_style.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_min_uM", type=float, default=0.0)
    ap.add_argument("--c_max_uM", type=float, default=20.0)
    ap.add_argument("--n_dense", type=int, default=400)
    ap.add_argument(
        "--out_dir", type=Path, default=Path(__file__).resolve().parent / "mk4_col_fit"
    )
    args = ap.parse_args()

    if args.c_min_uM < 0.0:
        raise ValueError(
            "--c_min_uM must be >= 0 for log-scale plotting and log-space predictions."
        )
    if args.c_max_uM <= args.c_min_uM:
        raise ValueError("--c_max_uM must be greater than --c_min_uM.")
    if args.n_dense < 20:
        raise ValueError("--n_dense must be >= 20.")

    fit = fit_mk4(MK4_DOSE_uM, COL_RESPONSE)
    csv_path, js_path, png_path, pdf_path = save_outputs(
        args.out_dir,
        fit,
        args.c_min_uM,
        args.c_max_uM,
        args.n_dense,
    )

    met = fit["metrics_dosed"]
    print("=== MK-4 dose-response fit (Collagen, fixed ARS shape) ===")
    print(f"fit_method      : {fit['method']}")
    print("shape_source    : mk4_ars_grid_fit")
    print(f"cK0_uM          : {fit['c_k0_uM']:.6g}")
    print(f"col_control_y0  : {fit['y0']:.6g}")
    print(f"aK              : {fit['a_k']:.6g}")
    print(f"bK              : {fit['b_k']:.6g}")
    print(f"DoF             : {fit['dof']}")
    print(f"RMSE            : {met['RMSE']:.6g}")

    print("dose_uM | col_obs | col_pred | multiplier")
    for c, y, yhat in zip(MK4_DOSE_uM, COL_RESPONSE, fit["yhat_all"]):
        m = yhat / fit["y0"] if fit["y0"] > 0 else float("nan")
        print(f"{c:7.4g} | {y:7.4g} | {yhat:8.4g} | {m:9.4g}")

    print(f"saved_csv       : {csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
