#!/usr/bin/env python3
"""
MK-4 dose-response fit for ARS mineralization with cK0 estimated via
autojoint optimization (optional logarithmic grid search fallback).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np


MK4_DOSE_uM = np.array([0.0, 0.1, 1.0, 10.0], dtype=float)
ARS_MINERALIZATION = np.array([100.0, 420.0, 560.0, 270.0], dtype=float)


def gating_m(c_uM: np.ndarray, a_k: float, b_k: float, c_k0_uM: float) -> np.ndarray:
    c = np.asarray(c_uM, dtype=float)
    out = np.ones_like(c, dtype=float)
    mask = c > 0.0
    if np.any(mask):
        x = np.log10(c[mask] / float(c_k0_uM))
        out[mask] = 1.0 + (a_k - b_k * x * x)
    return out


def model(
    c_uM: np.ndarray, y0: float, a_k: float, b_k: float, c_k0_uM: float
) -> np.ndarray:
    return float(y0) * gating_m(c_uM, a_k, b_k, c_k0_uM)


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


def fit_ab_for_ck0(
    c_fit: np.ndarray, y_fit: np.ndarray, y0: float, c_k0_uM: float
) -> dict:
    try:
        from scipy.optimize import least_squares

        def residuals(theta: np.ndarray) -> np.ndarray:
            return model(c_fit, y0, float(theta[0]), float(theta[1]), c_k0_uM) - y_fit

        res = least_squares(
            residuals,
            x0=np.array([0.5, 0.5], dtype=float),
            bounds=(0.0, np.inf),
            max_nfev=10000,
        )
        a_k, b_k = map(float, res.x)
        method = "scipy.least_squares"
        yhat_fit = model(c_fit, y0, a_k, b_k, c_k0_uM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - res.x.size
        s_sq = sse / max(1, dof)
        pcov = np.linalg.pinv(res.jac.T @ res.jac) * s_sq
    except Exception:
        x = np.log10(c_fit / c_k0_uM)
        z = x * x
        r = (y_fit / y0) - 1.0
        theta, *_ = np.linalg.lstsq(
            np.column_stack([np.ones_like(z), -z]), r, rcond=None
        )
        a_k = max(0.0, float(theta[0]))
        b_k = max(0.0, float(theta[1]))
        method = "numpy.linalg.lstsq"
        pcov = np.zeros((2, 2), dtype=float)
        yhat_fit = model(c_fit, y0, a_k, b_k, c_k0_uM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - 2

    return {
        "method": method,
        "a_k": a_k,
        "b_k": b_k,
        "pcov": pcov,
        "rmse": float(np.sqrt(np.mean((y_fit - yhat_fit) ** 2))),
        "dof": dof,
    }


def fit_mk4_grid(
    c_uM: np.ndarray,
    y: np.ndarray,
    c_k0_min_uM: float,
    c_k0_max_uM: float,
    n_grid: int,
) -> dict:
    c_uM = np.asarray(c_uM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_uM > 0.0
    c_fit = c_uM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_k and b_k.")
    if c_k0_min_uM <= 0.0:
        raise ValueError("c_k0_min_uM must be > 0 for logarithmic grid search.")
    if c_k0_max_uM <= c_k0_min_uM:
        raise ValueError("c_k0_max_uM must be greater than c_k0_min_uM.")
    if n_grid < 3:
        raise ValueError("n_grid must be >= 3.")

    c_k0_grid_uM = np.logspace(np.log10(c_k0_min_uM), np.log10(c_k0_max_uM), n_grid)
    grid_rows = []
    best = None

    for c_k0_uM in c_k0_grid_uM:
        row = fit_ab_for_ck0(c_fit, y_fit, y0, float(c_k0_uM))
        row["c_k0_uM"] = float(c_k0_uM)
        grid_rows.append(row)
        if best is None or row["rmse"] < best["rmse"]:
            best = row

    assert best is not None
    yhat_all = model(c_uM, y0, best["a_k"], best["b_k"], best["c_k0_uM"])
    return {
        "fit_method": f"{best['method']} + log_grid_search",
        "grid_method": "numpy.logspace",
        "y0": y0,
        "c_k0_uM": best["c_k0_uM"],
        "a_k": best["a_k"],
        "b_k": best["b_k"],
        "pcov": best["pcov"],
        "dof": int(best["dof"]),
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(
            y_fit, model(c_fit, y0, best["a_k"], best["b_k"], best["c_k0_uM"])
        ),
        "grid_profile": {
            "c_k0_uM": c_k0_grid_uM,
            "rmse": np.array([row["rmse"] for row in grid_rows], dtype=float),
            "a_k": np.array([row["a_k"] for row in grid_rows], dtype=float),
            "b_k": np.array([row["b_k"] for row in grid_rows], dtype=float),
        },
    }


def fit_mk4_autojoint(
    c_uM: np.ndarray,
    y: np.ndarray,
    c_k0_min_uM: float,
    c_k0_max_uM: float,
) -> dict:
    c_uM = np.asarray(c_uM, dtype=float)
    y = np.asarray(y, dtype=float)
    y0 = float(y[0])
    mask = c_uM > 0.0
    c_fit = c_uM[mask]
    y_fit = y[mask]

    if c_fit.size < 2:
        raise ValueError("Need at least two non-zero dose points to fit a_k and b_k.")
    if c_k0_min_uM <= 0.0:
        raise ValueError("c_k0_min_uM must be > 0.")
    if c_k0_max_uM <= c_k0_min_uM:
        raise ValueError("c_k0_max_uM must be greater than c_k0_min_uM.")

    lo_log = float(np.log10(c_k0_min_uM))
    hi_log = float(np.log10(c_k0_max_uM))
    x0 = np.array([0.5, 0.5, 0.5 * (lo_log + hi_log)], dtype=float)

    try:
        from scipy.optimize import least_squares

        def residuals(theta: np.ndarray) -> np.ndarray:
            a_k, b_k, log10_c_k0 = map(float, theta)
            c_k0 = 10.0 ** log10_c_k0
            return model(c_fit, y0, a_k, b_k, c_k0) - y_fit

        res = least_squares(
            residuals,
            x0=x0,
            bounds=([0.0, 0.0, lo_log], [np.inf, np.inf, hi_log]),
            max_nfev=20000,
        )
        a_k, b_k, log10_c_k0 = map(float, res.x)
        c_k0_uM = float(10.0 ** log10_c_k0)
        method = "scipy.least_squares_autojoint"
        yhat_fit = model(c_fit, y0, a_k, b_k, c_k0_uM)
        sse = float(np.sum((y_fit - yhat_fit) ** 2))
        dof = y_fit.size - res.x.size
        s_sq = sse / max(1, dof)
        pcov = np.linalg.pinv(res.jac.T @ res.jac) * s_sq
    except Exception:
        out = fit_mk4_grid(
            c_uM=c_uM,
            y=y,
            c_k0_min_uM=c_k0_min_uM,
            c_k0_max_uM=c_k0_max_uM,
            n_grid=200,
        )
        out["fit_method"] = "log_grid_search_fallback"
        return out

    yhat_all = model(c_uM, y0, a_k, b_k, c_k0_uM)
    return {
        "fit_method": method,
        "grid_method": "autojoint",
        "y0": y0,
        "c_k0_uM": c_k0_uM,
        "a_k": a_k,
        "b_k": b_k,
        "pcov": pcov,
        "dof": int(dof),
        "yhat_all": yhat_all,
        "metrics_overall": _metrics(y, yhat_all),
        "metrics_dosed": _metrics(y_fit, model(c_fit, y0, a_k, b_k, c_k0_uM)),
        "grid_profile": {
            "c_k0_uM": np.array([c_k0_uM], dtype=float),
            "rmse": np.array(
                [float(np.sqrt(np.mean((y_fit - model(c_fit, y0, a_k, b_k, c_k0_uM)) ** 2)))],
                dtype=float,
            ),
            "a_k": np.array([a_k], dtype=float),
            "b_k": np.array([b_k], dtype=float),
        },
    }


def save_outputs(
    out_dir: Path,
    fit: dict,
    c_min_uM: float,
    c_max_uM: float,
    n_dense: int,
) -> tuple[Path, Path, Path, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    positive_doses = MK4_DOSE_uM[MK4_DOSE_uM > 0.0]
    default_positive_start = max(float(np.min(positive_doses)), np.nextafter(0.0, 1.0))
    positive_start = c_min_uM if c_min_uM > 0.0 else default_positive_start
    positive_start = max(positive_start, np.nextafter(0.0, 1.0))

    c_dense_pos = np.logspace(
        np.log10(positive_start), np.log10(c_max_uM), int(n_dense)
    )
    c_dense_csv = (
        np.concatenate(([0.0], c_dense_pos)) if c_min_uM <= 0.0 else c_dense_pos
    )

    m_dense_csv = gating_m(c_dense_csv, fit["a_k"], fit["b_k"], fit["c_k0_uM"])
    y_dense_csv = fit["y0"] * m_dense_csv
    log10_mk4 = np.full_like(c_dense_csv, float("nan"))
    mask_positive = c_dense_csv > 0.0
    log10_mk4[mask_positive] = np.log10(c_dense_csv[mask_positive] / fit["c_k0_uM"])

    m_dense_pos = gating_m(c_dense_pos, fit["a_k"], fit["b_k"], fit["c_k0_uM"])
    y_dense_pos = fit["y0"] * m_dense_pos

    csv_path = out_dir / "mk4_ars_autojoint_fit_predictions.csv"
    np.savetxt(
        csv_path,
        np.column_stack(
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
        ),
        delimiter=",",
        header="mk4_uM,log10_mk4_over_cK0,mk4_multiplier_pred,ars_pred,cK0_uM,ars_control_y0,aK,bK",
        comments="",
    )

    grid_csv_path = out_dir / "mk4_ars_autojoint_profile.csv"
    np.savetxt(
        grid_csv_path,
        np.column_stack(
            [
                fit["grid_profile"]["c_k0_uM"],
                fit["grid_profile"]["rmse"],
                fit["grid_profile"]["a_k"],
                fit["grid_profile"]["b_k"],
            ]
        ),
        delimiter=",",
        header="cK0_uM,RMSE,aK,bK",
        comments="",
    )

    js_path = out_dir / "mk4_ars_autojoint_fit_metrics.json"
    js_path.write_text(
        json.dumps(
            {
                "fit_method": fit["fit_method"],
                "grid_method": fit["grid_method"],
                "cK0_uM": fit["c_k0_uM"],
                "ars_control_y0": fit["y0"],
                "aK": fit["a_k"],
                "bK": fit["b_k"],
                "degrees_of_freedom": fit["dof"],
                "metrics_overall": fit["metrics_overall"],
                "metrics_dosed": fit["metrics_dosed"],
                "grid_profile": {
                    "cK0_uM": fit["grid_profile"]["c_k0_uM"].tolist(),
                    "RMSE": fit["grid_profile"]["rmse"].tolist(),
                    "aK": fit["grid_profile"]["a_k"].tolist(),
                    "bK": fit["grid_profile"]["b_k"].tolist(),
                },
                "data": {
                    "mk4_uM": MK4_DOSE_uM.tolist(),
                    "ars_mineralization_percent_control": ARS_MINERALIZATION.tolist(),
                },
            },
            indent=2,
        )
    )

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
            ARS_MINERALIZATION[positive_mask],
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
        [ARS_MINERALIZATION[0]],
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
    y_all = np.concatenate([ARS_MINERALIZATION, y_dense_pos])
    y_min, y_max = float(np.min(y_all)), float(np.max(y_all))
    y_pad = max(0.4, 0.08 * (y_max - y_min if y_max > y_min else 1.0))
    ax.set_ylim(max(0.0, y_min - y_pad), y_max + y_pad)

    ax.set_xlabel("MK-4 concentration (uM, log scale)")
    ax.set_ylabel("Mineralization (% of control)")
    ax.set_title("MK-4 ARS Fit Autojoint", fontsize=9, pad=4)
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

    ax.legend(
        handles=[
            Line2D([0], [0], color=c_mk4, lw=1.2, label="MK-4 fit"),
            Line2D([0], [0], color="#777777", lw=1.0, ls="--", label="Best cK0"),
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
                label="ARS data",
            ),
        ],
        loc="upper left",
        frameon=False,
        fontsize=7.2,
        handlelength=1.8,
        borderaxespad=0.4,
    )

    ax.xaxis.label.set_size(9)
    ax.yaxis.label.set_size(9)

    fig.tight_layout(pad=0.4)
    png_path = out_dir / "mk4_ars_autojoint_fit_style.png"
    pdf_path = out_dir / "mk4_ars_autojoint_fit_style.pdf"
    fig.savefig(png_path, dpi=600)
    fig.savefig(pdf_path)
    plt.close(fig)

    return csv_path, grid_csv_path, js_path, png_path, pdf_path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--c_min_uM", type=float, default=0.0)
    ap.add_argument("--c_max_uM", type=float, default=20.0)
    ap.add_argument("--n_dense", type=int, default=400)
    ap.add_argument("--c_k0_min_uM", type=float, default=0.1)
    ap.add_argument("--c_k0_max_uM", type=float, default=10.0)
    ap.add_argument("--n_grid", type=int, default=200)
    ap.add_argument(
        "--fit-mode",
        choices=["autojoint", "grid"],
        default="autojoint",
        help="fit cK0 jointly with aK/bK (autojoint) or use legacy grid search",
    )
    ap.add_argument(
        "--out_dir",
        type=Path,
        default=Path(__file__).resolve().parent / "mk4_ars_autojoint_fit",
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

    if args.fit_mode == "grid":
        fit = fit_mk4_grid(
            MK4_DOSE_uM,
            ARS_MINERALIZATION,
            args.c_k0_min_uM,
            args.c_k0_max_uM,
            args.n_grid,
        )
    else:
        fit = fit_mk4_autojoint(
            MK4_DOSE_uM,
            ARS_MINERALIZATION,
            args.c_k0_min_uM,
            args.c_k0_max_uM,
        )
    csv_path, grid_csv_path, js_path, png_path, pdf_path = save_outputs(
        args.out_dir,
        fit,
        args.c_min_uM,
        args.c_max_uM,
        args.n_dense,
    )

    met = fit["metrics_dosed"]
    print("=== MK-4 dose-response autojoint fit (ARS mineralization) ===")
    print(f"fit_method      : {fit['fit_method']}")
    print(f"grid_method     : {fit['grid_method']}")
    print(f"cK0_uM          : {fit['c_k0_uM']:.6g}")
    print(f"ars_control_y0  : {fit['y0']:.6g}")
    print(f"aK              : {fit['a_k']:.6g}")
    print(f"bK              : {fit['b_k']:.6g}")
    print(f"DoF             : {fit['dof']}")
    print(f"RMSE            : {met['RMSE']:.6g}")
    print(f"R2              : {met['R2']:.6g}")

    print("dose_uM | ars_obs | ars_pred | multiplier")
    for c, y, yhat in zip(MK4_DOSE_uM, ARS_MINERALIZATION, fit["yhat_all"]):
        m = yhat / fit["y0"] if fit["y0"] > 0 else float("nan")
        print(f"{c:7.4g} | {y:7.4g} | {yhat:8.4g} | {m:9.4g}")

    print(f"saved_csv       : {csv_path}")
    print(f"saved_profile   : {grid_csv_path}")
    print(f"saved_metrics   : {js_path}")
    print(f"saved_plot_png  : {png_path}")
    print(f"saved_plot_pdf  : {pdf_path}")


if __name__ == "__main__":
    main()
