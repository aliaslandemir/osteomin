#!/usr/bin/env python3
"""
Phenomenological Dex dose-response fit for collagen using a symmetric
log-parabolic model with cD0 estimated via autojoint optimization
(optional logarithmic grid search fallback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


# Chiang2025DexIGF1R

DEX_DOSE_nM = np.array([0.0, 1.0, 10.0], dtype=float)
COL_REL = np.array([1.0, 2.4, 3.6], dtype=float)


def dex_modulation_mP(
    c_dex_nM: np.ndarray, a_d: float, b_d: float, c_d0_nM: float
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


def model(
    c_dex_nM: np.ndarray, y0: float, a_d: float, b_d: float, c_d0_nM: float
) -> np.ndarray:
    return float(y0) * dex_modulation_mP(c_dex_nM, a_d, b_d, c_d0_nM)


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


def fit_ab_for_cd0(
    c_fit: np.ndarray, y_fit: np.ndarray, y0: float, c_d0_nM: float
) -> dict:
    try:
        from scipy.optimize import least_squares

        x0 = np.array([0.8, 0.6], dtype=float)

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(c_fit, y0, float(theta[0]), float(theta[1]), c_d0_nM) - y_fit

        res = least_squares(
            residuals,
            x0=x0,
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        a_d, b_d = map(float, res.x)
        method = "scipy.least_squares"
        yhat_fit = model(c_fit, y0, a_d, b_d, c_d0_nM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - res.x.size
        s_sq = sse / max(1, dof)
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
    except Exception:
        x = np.log10(c_fit / c_d0_nM)
        z = x * x
        r = (y_fit / y0) - 1.0
        A = np.column_stack([np.ones_like(z), -z])
        theta, *_ = np.linalg.lstsq(A, r, rcond=None)
        a_d = max(0.0, float(theta[0]))
        b_d = max(0.0, float(theta[1]))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((2, 2), dtype=float)
        yhat_fit = model(c_fit, y0, a_d, b_d, c_d0_nM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - 2

    return {
        "method": method,
        "a_d": a_d,
        "b_d": b_d,
        "pcov": pcov,
        "yhat_fit": yhat_fit,
        "rmse": float(np.sqrt(np.mean((y_fit - yhat_fit) ** 2))),
        "dof": dof,
    }


def fit_dex_grid(
    c_dex_nM: np.ndarray,
    y: np.ndarray,
    c_d0_min_nM: float,
    c_d0_max_nM: float,
    n_grid: int,
) -> dict:
    c_dex_nM = np.asarray(c_dex_nM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_dex_nM > 0.0
    c_fit = c_dex_nM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_d and b_d.")
    if c_d0_min_nM <= 0.0:
        raise ValueError("c_d0_min_nM must be > 0 for logarithmic grid search.")
    if c_d0_max_nM <= c_d0_min_nM:
        raise ValueError("c_d0_max_nM must be greater than c_d0_min_nM.")
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3.")

    c_d0_grid_nM = np.logspace(np.log10(c_d0_min_nM), np.log10(c_d0_max_nM), n_grid)
    grid_rows = []
    best = None

    for c_d0_nM in c_d0_grid_nM:
        row = fit_ab_for_cd0(c_fit, y_fit, y0, float(c_d0_nM))
        row["c_d0_nM"] = float(c_d0_nM)
        grid_rows.append(row)
        if best is None or row["rmse"] < best["rmse"]:
            best = row

    assert best is not None
    yhat_all = model(c_dex_nM, y0, best["a_d"], best["b_d"], best["c_d0_nM"])
    return {
        "fit_method": f"{best['method']} + log_grid_search",
        "grid_method": "numpy.logspace",
        "y0": y0,
        "c_d0_nM": best["c_d0_nM"],
        "a_d": best["a_d"],
        "b_d": best["b_d"],
        "pcov": best["pcov"],
        "dof": int(best["dof"]),
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(
            y_fit, model(c_fit, y0, best["a_d"], best["b_d"], best["c_d0_nM"])
        ),
        "grid_profile": {
            "c_d0_nM": c_d0_grid_nM,
            "rmse": np.array([row["rmse"] for row in grid_rows], dtype=float),
            "a_d": np.array([row["a_d"] for row in grid_rows], dtype=float),
            "b_d": np.array([row["b_d"] for row in grid_rows], dtype=float),
        },
    }


def fit_dex_autojoint(
    c_dex_nM: np.ndarray,
    y: np.ndarray,
    c_d0_min_nM: float,
    c_d0_max_nM: float,
) -> dict:
    c_dex_nM = np.asarray(c_dex_nM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_dex_nM > 0.0
    c_fit = c_dex_nM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_d and b_d.")
    if c_d0_min_nM <= 0.0:
        raise ValueError("c_d0_min_nM must be > 0.")
    if c_d0_max_nM <= c_d0_min_nM:
        raise ValueError("c_d0_max_nM must be greater than c_d0_min_nM.")

    lo_log = float(np.log10(c_d0_min_nM))
    hi_log = float(np.log10(c_d0_max_nM))
    x0 = np.array([0.8, 0.6, 0.5 * (lo_log + hi_log)], dtype=float)

    try:
        from scipy.optimize import least_squares

        def residuals(theta: np.ndarray) -> np.ndarray:
            a_d, b_d, log10_c_d0 = map(float, theta)
            c_d0 = 10.0 ** log10_c_d0
            return model(c_fit, y0, a_d, b_d, c_d0) - y_fit

        res = least_squares(
            residuals,
            x0=x0,
            bounds=([0.0, 0.0, lo_log], [np.inf, np.inf, hi_log]),
            max_nfev=20000,
        )
        a_d, b_d, log10_c_d0 = map(float, res.x)
        c_d0_nM = float(10.0 ** log10_c_d0)
        method = "scipy.least_squares_autojoint"
        yhat_fit = model(c_fit, y0, a_d, b_d, c_d0_nM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - res.x.size
        s_sq = sse / max(1, dof)
        jt_j = res.jac.T @ res.jac
        pcov = np.linalg.pinv(jt_j) * s_sq
    except Exception:
        out = fit_dex_grid(
            c_dex_nM=c_dex_nM,
            y=y,
            c_d0_min_nM=c_d0_min_nM,
            c_d0_max_nM=c_d0_max_nM,
            n_grid=200,
        )
        out["fit_method"] = "log_grid_search_fallback"
        return out

    yhat_all = model(c_dex_nM, y0, a_d, b_d, c_d0_nM)
    return {
        "fit_method": method,
        "grid_method": "autojoint",
        "y0": y0,
        "c_d0_nM": c_d0_nM,
        "a_d": a_d,
        "b_d": b_d,
        "pcov": pcov,
        "dof": int(dof),
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(y_fit, model(c_fit, y0, a_d, b_d, c_d0_nM)),
        "grid_profile": {
            "c_d0_nM": np.array([c_d0_nM], dtype=float),
            "rmse": np.array(
                [float(np.sqrt(np.mean((y_fit - model(c_fit, y0, a_d, b_d, c_d0_nM)) ** 2)))],
                dtype=float,
            ),
            "a_d": np.array([a_d], dtype=float),
            "b_d": np.array([b_d], dtype=float),
        },
    }


def save_outputs(
    out_dir: Path,
    fit: dict,
    c_min_nM: float,
    c_max_nM: float,
    n_dense: int,
) -> tuple[Path, Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_doses = DEX_DOSE_nM[DEX_DOSE_nM > 0.0]
    if positive_doses.size:
        default_positive_start = max(
            float(np.min(positive_doses)), np.nextafter(0.0, 1.0)
        )
    else:
        default_positive_start = np.nextafter(0.0, 1.0)
    positive_start = c_min_nM if c_min_nM > 0.0 else default_positive_start
    positive_start = max(positive_start, np.nextafter(0.0, 1.0))

    c_dense_pos = np.logspace(
        np.log10(positive_start), np.log10(c_max_nM), int(n_dense)
    )
    include_zero = c_min_nM <= 0.0
    if include_zero:
        c_dense_csv = np.concatenate(([0.0], c_dense_pos))
    else:
        c_dense_csv = c_dense_pos

    m_dense_csv = dex_modulation_mP(c_dense_csv, fit["a_d"], fit["b_d"], fit["c_d0_nM"])
    y_dense_csv = fit["y0"] * m_dense_csv
    log10_dex = np.full_like(c_dense_csv, float("nan"))
    mask_positive = c_dense_csv > 0.0
    log10_dex[mask_positive] = np.log10(c_dense_csv[mask_positive] / fit["c_d0_nM"])

    m_dense_pos = dex_modulation_mP(c_dense_pos, fit["a_d"], fit["b_d"], fit["c_d0_nM"])
    y_dense_pos = fit["y0"] * m_dense_pos

    csv_path = out_dir / "dex_col_autojoint_fit_predictions.csv"
    mat = np.column_stack(
        [
            c_dense_csv,
            log10_dex,
            m_dense_csv,
            y_dense_csv,
            np.full_like(c_dense_csv, fit["c_d0_nM"]),
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
            "dex_nM,log10_dex_over_cD0,dex_multiplier_pred,col_pred,"
            "cD0_nM,col_control_y0,aD,bD"
        ),
        comments="",
    )

    grid_csv_path = out_dir / "dex_col_autojoint_profile.csv"
    grid_mat = np.column_stack(
        [
            fit["grid_profile"]["c_d0_nM"],
            fit["grid_profile"]["rmse"],
            fit["grid_profile"]["a_d"],
            fit["grid_profile"]["b_d"],
        ]
    )
    np.savetxt(
        grid_csv_path,
        grid_mat,
        delimiter=",",
        header="cD0_nM,RMSE,aD,bD",
        comments="",
    )

    js_path = out_dir / "dex_col_autojoint_fit_metrics.json"
    payload = {
        "fit_method": fit["fit_method"],
        "grid_method": fit["grid_method"],
        "cD0_nM": fit["c_d0_nM"],
        "col_control_y0": fit["y0"],
        "aD": fit["a_d"],
        "bD": fit["b_d"],
        "degrees_of_freedom": fit["dof"],
        "metrics_overall": fit["metrics_overall"],
        "metrics_dosed": fit["metrics_dosed"],
        "grid_profile": {
            "cD0_nM": fit["grid_profile"]["c_d0_nM"].tolist(),
            "RMSE": fit["grid_profile"]["rmse"].tolist(),
            "aD": fit["grid_profile"]["a_d"].tolist(),
            "bD": fit["grid_profile"]["b_d"].tolist(),
        },
        "data": {
            "dex_nM": DEX_DOSE_nM.tolist(),
            "COL_REL": COL_REL.tolist(),
        },
    }
    js_path.write_text(json.dumps(payload, indent=2))

    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    c_dex = "#1f77b4"
    cm_to_in = 1.0 / 2.54
    fig = plt.figure(figsize=(8.5 * cm_to_in, 6.6 * cm_to_in))
    ax = fig.add_subplot(1, 1, 1)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    ax.plot(c_dense_pos, y_dense_pos, color=c_dex, lw=1.2)
    positive_mask = DEX_DOSE_nM > 0.0
    if np.any(positive_mask):
        ax.scatter(
            DEX_DOSE_nM[positive_mask],
            COL_REL[positive_mask],
            marker="o",
            s=30,
            color=c_dex,
            edgecolor="white",
            linewidth=0.6,
            zorder=3,
            clip_on=False,
        )
    control_x = 0.7 * positive_start
    ax.scatter(
        [control_x],
        [COL_REL[0]],
        marker="o",
        s=30,
        color=c_dex,
        edgecolor="white",
        linewidth=0.6,
        zorder=3,
        clip_on=False,
    )

    ax.set_xscale("log")
    ax.set_xlim(c_dense_pos[0] * 0.9, c_dense_pos[-1] * 1.1)
    ax.set_xticks([control_x, 1.0, 10.0, 100.0])
    ax.set_xticklabels(["0", "1", "10", "100"])
    y_all = np.concatenate([COL_REL, y_dense_pos])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.4, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("Dex concentration (nM, log scale)")
    ax.set_ylabel("Collagen relative signal")
    ax.set_title("Dex Collagen Fit Autojoint", fontsize=9, pad=4)
    ax.tick_params(labelsize=8, top=False, labeltop=False)
    ax.xaxis.set_ticks_position("bottom")
    ax.grid(True, which="major", color="#b0b0b0", alpha=0.28, linewidth=0.75)
    ax.grid(True, which="minor", color="#d0d0d0", alpha=0.18, linewidth=0.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    ax.axvline(fit["c_d0_nM"], color="#777777", ls="--", lw=1.0, alpha=0.9)

    met = fit["metrics_dosed"]
    txt = "\n".join(
        [
            f"cD0 = {fit['c_d0_nM']:.2f} nM",
            f"y0 = {fit['y0']:.2f}",
            f"aD = {fit['a_d']:.2f}",
            f"bD = {fit['b_d']:.2f}",
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
        Line2D([0], [0], color=c_dex, lw=1.2, label="Dex fit"),
        Line2D([0], [0], color="#777777", lw=1.0, ls="--", label="Best cD0"),
        Line2D(
            [0],
            [0],
            color=c_dex,
            lw=1.2,
            marker="o",
            markersize=5.5,
            markerfacecolor=c_dex,
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
    png_path = out_dir / "dex_col_autojoint_fit_style.png"
    pdf_path = out_dir / "dex_col_autojoint_fit_style.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, grid_csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_min_nM", type=float, default=0.0)
    ap.add_argument("--c_max_nM", type=float, default=200.0)
    ap.add_argument("--n_dense", type=int, default=400)
    ap.add_argument("--c_d0_min_nM", type=float, default=1.0)
    ap.add_argument("--c_d0_max_nM", type=float, default=100.0)
    ap.add_argument("--n_grid", type=int, default=200)
    ap.add_argument(
        "--fit-mode",
        choices=["autojoint", "grid"],
        default="autojoint",
        help="fit cD0 jointly with aD/bD (autojoint) or use legacy grid search",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "dex_col_autojoint_fit",
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

    if args.fit_mode == "grid":
        fit = fit_dex_grid(
            DEX_DOSE_nM,
            COL_REL,
            c_d0_min_nM=args.c_d0_min_nM,
            c_d0_max_nM=args.c_d0_max_nM,
            n_grid=args.n_grid,
        )
    else:
        fit = fit_dex_autojoint(
            DEX_DOSE_nM,
            COL_REL,
            c_d0_min_nM=args.c_d0_min_nM,
            c_d0_max_nM=args.c_d0_max_nM,
        )
    csv_path, grid_csv_path, js_path, png_path, pdf_path = save_outputs(
        args.out_dir,
        fit,
        args.c_min_nM,
        args.c_max_nM,
        args.n_dense,
    )

    met = fit["metrics_dosed"]
    print("=== Dex dose-response autojoint fit (Collagen) ===")
    print(f"fit_method      : {fit['fit_method']}")
    print(f"grid_method     : {fit['grid_method']}")
    print(f"cD0_nM          : {fit['c_d0_nM']:.6g}")
    print(f"col_control_y0  : {fit['y0']:.6g}")
    print(f"aD              : {fit['a_d']:.6g}")
    print(f"bD              : {fit['b_d']:.6g}")
    print(f"DoF             : {fit['dof']}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")

    print("dose_nM | col_obs | col_pred | multiplier")
    for c, y, yhat in zip(DEX_DOSE_nM, COL_REL, fit["yhat_all"]):
        m = yhat / fit["y0"] if fit["y0"] > 0 else float("nan")
        print(f"{c:7.4g} | {y:7.4g} | {yhat:8.4g} | {m:9.4g}")

    print(f"saved_csv       : {csv_path}")
    print(f"saved_profile   : {grid_csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
