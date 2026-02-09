"""
Weighted least squares fit of Dex (D) and MK-4 (K) effects from endpoint assays.

Design (matches your experimental logic):
- C1 = basal (no osteogenic supplements)
- C2 = osteogenic baseline (AA + β-GP; no Dex/MK-4)
- C3–C8 = osteogenic + Dex/MK-4 at known concentrations

Model per assay (piecewise by dose level; no intercept beyond osteogenic baseline):
1) Normalize each osteogenic condition to the osteogenic increment:
      denom = (Y_C2 - Y_C1)
      z_i   = (Y_i - Y_C1)/denom - 1
   so that for C3–C8:
      z_i ≈ bD_low*1[D=10] + bD_high*1[D=100] + bK_low*1[K=0.1µM] + bK_high*1[K=10µM]

2) Weighted least squares:
   - If SD and N exist: use SEM = SD/sqrt(N)
   - else: use SD
   - weights = 1 / (SEM_z^2), where SEM_z = SEM/|denom|

Prediction:
- C1: PredMean = Y_C1  (no osteogenic increment)
- C2..C8: PredMean = Y_C1 + denom*(1 + dose effects)

Outputs:
- bD_low, bD_high, bK_low, bK_high + SE + 95% CI (t-critical)
- diagnostics + predicted vs observed
- CSVs: pred_*.csv, fitdata_*.csv

Run:
  python d_k_fit.py
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt


# -----------------------------
# Concentrations (Table 1)
# Units used in fitting (standardized):
#   D in nM
#   K in nM  (converted from µM)
# -----------------------------
def condition_concentrations() -> pd.DataFrame:
    conc = pd.DataFrame(
        {
            "Condition": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            "D_nM": [0, 0, 10, 10, 100, 100, 100, 0],
            "K_uM": [0, 0, 0.1, 10, 0.1, 10, 0, 10],
        }
    )
    conc["K_nM"] = conc["K_uM"] * 1000.0
    return conc


# -----------------------------
# IO
# -----------------------------
def read_endpoint_table(path: str) -> pd.DataFrame:
    """
    Reads a text file with columns: Condition Mean SD N
    Accepts tab-separated or whitespace-separated.
    """
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    df.columns = [c.strip() for c in df.columns]
    colmap = {c: c for c in df.columns}
    for c in df.columns:
        cl = c.lower()
        if cl == "condition":
            colmap[c] = "Condition"
        elif cl == "mean":
            colmap[c] = "Mean"
        elif cl == "sd":
            colmap[c] = "SD"
        elif cl == "n":
            colmap[c] = "N"
    df = df.rename(columns=colmap)

    if "Condition" not in df.columns or "Mean" not in df.columns:
        raise ValueError(
            f"{path}: must contain at least Condition and Mean columns. Found: {df.columns.tolist()}"
        )

    if "SD" not in df.columns:
        df["SD"] = np.nan
    if "N" not in df.columns:
        df["N"] = np.nan

    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    df["SD"] = pd.to_numeric(df["SD"], errors="coerce")
    df["N"] = pd.to_numeric(df["N"], errors="coerce")

    if df["Mean"].isna().any():
        bad = df[df["Mean"].isna()]
        raise ValueError(f"{path}: unparsable Mean values:\n{bad}")

    return df


# -----------------------------
# Core WLS
# -----------------------------
def _weights_from_sd_n(sd: np.ndarray, n: np.ndarray) -> np.ndarray:
    """
    Build weights from SD and N:
      SEM = SD/sqrt(N) if both available; else SEM=SD; else SEM=1 (equal weights)
      w = 1/SEM^2
    """
    sd = sd.astype(float)
    n = n.astype(float)

    sd_ok = np.isfinite(sd) & (sd > 0)
    n_ok = np.isfinite(n) & (n > 0)

    sem = np.ones_like(sd, dtype=float)
    sem[sd_ok] = sd[sd_ok]
    sem[sd_ok & n_ok] = sd[sd_ok & n_ok] / np.sqrt(n[sd_ok & n_ok])

    # avoid inf weights
    sem = np.where(np.isfinite(sem) & (sem > 0), sem, 1.0)
    return 1.0 / (sem**2)


def _dose_indicators(m: pd.DataFrame) -> pd.DataFrame:
    """Build piecewise indicator columns for D and K dose levels."""
    d = m["D_nM"].to_numpy(float)
    k = m["K_nM"].to_numpy(float)
    out = pd.DataFrame(
        {
            "D_low": (np.isclose(d, 10.0)).astype(float),
            "D_high": (np.isclose(d, 100.0)).astype(float),
            "K_low": (np.isclose(k, 100.0)).astype(float),  # 0.1 µM = 100 nM
            "K_high": (np.isclose(k, 10000.0)).astype(float),  # 10 µM = 10000 nM
        }
    )
    return out


def wls_fit_dk_effects(df: pd.DataFrame, assay: str) -> dict:
    """
    Fit aD, aK using C1 and C2 anchoring and WLS on C3..C8.

    z_i = (Y_i - Y1)/(Y2 - Y1) - 1  ≈ bD_low*1[D=10] + bD_high*1[D=100]
                                            + bK_low*1[K=0.1µM] + bK_high*1[K=10µM]
    """
    conc = condition_concentrations()
    m = df.merge(conc, on="Condition", how="inner")

    # sanity: require all C1..C8
    missing = set(conc["Condition"]) - set(m["Condition"])
    if missing:
        raise ValueError(f"{assay}: missing conditions in data: {sorted(missing)}")

    Y1 = float(m.loc[m["Condition"] == "C1", "Mean"].iloc[0])
    Y2 = float(m.loc[m["Condition"] == "C2", "Mean"].iloc[0])
    denom = Y2 - Y1
    if np.isclose(denom, 0.0):
        raise ValueError(f"{assay}: Y_C2 - Y_C1 ≈ 0; cannot normalize.")

    # fit set C3..C8
    fit_conds = ["C3", "C4", "C5", "C6", "C7", "C8"]
    mf = m[m["Condition"].isin(fit_conds)].copy()

    mf["z"] = (mf["Mean"] - Y1) / denom - 1.0

    ind = _dose_indicators(mf)
    X = np.column_stack(
        [
            ind["D_low"].to_numpy(float),
            ind["D_high"].to_numpy(float),
            ind["K_low"].to_numpy(float),
            ind["K_high"].to_numpy(float),
        ]
    )
    y = mf["z"].to_numpy(float)  # J

    # weights: propagate SEM through z scaling by |denom|
    w_mean = _weights_from_sd_n(
        mf["SD"].to_numpy(float), mf["N"].to_numpy(float)
    )  # = 1/SEM_mean^2
    w_z = w_mean * (
        denom**2
    )  # because Var(z)=Var(mean)/denom^2 => w_z = denom^2 / Var(mean)

    # WLS normal equations
    W = np.diag(w_z)
    XtWX = X.T @ W @ X
    XtWy = X.T @ W @ y
    b = np.linalg.solve(XtWX, XtWy)  # [bD_low, bD_high, bK_low, bK_high]

    # diagnostics + SE
    yhat = X @ b
    resid = y - yhat
    dof = max(len(y) - X.shape[1], 1)
    sigma2 = float(resid.T @ W @ resid) / dof
    cov = sigma2 * np.linalg.inv(XtWX)
    se = np.sqrt(np.diag(cov))

    tcrit = float(stats.t.ppf(0.975, dof)) if dof > 0 else 1.96
    ci = np.vstack([b - tcrit * se, b + tcrit * se]).T

    # predictions (fix C1 separately)
    m2 = m.copy()
    ind2 = _dose_indicators(m2)
    modifier = (
        1.0
        + b[0] * ind2["D_low"].to_numpy(float)
        + b[1] * ind2["D_high"].to_numpy(float)
        + b[2] * ind2["K_low"].to_numpy(float)
        + b[3] * ind2["K_high"].to_numpy(float)
    )
    pred = Y1 + denom * modifier
    pred[m2["Condition"].to_numpy() == "C1"] = (
        Y1  # basal condition has no osteo increment
    )
    m2["PredMean"] = pred

    # predicted z for fit conditions
    mf["z_hat"] = yhat
    mf["resid_z"] = resid

    return {
        "assay": assay,
        "Y_C1": Y1,
        "Y_C2": Y2,
        "denom": denom,
        "bD_low": float(b[0]),
        "bD_high": float(b[1]),
        "bK_low": float(b[2]),
        "bK_high": float(b[3]),
        "bD_low_SE": float(se[0]),
        "bD_high_SE": float(se[1]),
        "bK_low_SE": float(se[2]),
        "bK_high_SE": float(se[3]),
        "bD_low_CI95_low": float(ci[0, 0]),
        "bD_low_CI95_high": float(ci[0, 1]),
        "bD_high_CI95_low": float(ci[1, 0]),
        "bD_high_CI95_high": float(ci[1, 1]),
        "bK_low_CI95_low": float(ci[2, 0]),
        "bK_low_CI95_high": float(ci[2, 1]),
        "bK_high_CI95_low": float(ci[3, 0]),
        "bK_high_CI95_high": float(ci[3, 1]),
        "dof": int(dof),
        "sigma2": sigma2,
        "fit_table": mf[
            ["Condition", "D_nM", "K_nM", "Mean", "SD", "N", "z", "z_hat", "resid_z"]
        ]
        .sort_values("Condition")
        .reset_index(drop=True),
        "pred_table": m2[["Condition", "D_nM", "K_nM", "Mean", "SD", "N", "PredMean"]]
        .sort_values("Condition")
        .reset_index(drop=True),
    }


def report(res: dict) -> None:
    print("\n" + "=" * 72)
    print(f"{res['assay']} — WLS fit (piecewise by dose level)")
    print("-" * 72)
    print(
        f"Anchors: C1={res['Y_C1']:.6g}, C2={res['Y_C2']:.6g}, denom(C2-C1)={res['denom']:.6g}"
    )
    print(
        f"bD_low (D=10nM) = {res['bD_low']:.6g} (SE {res['bD_low_SE']:.3g})  "
        f"95%CI [{res['bD_low_CI95_low']:.6g}, {res['bD_low_CI95_high']:.6g}]"
    )
    print(
        f"bD_high (D=100nM) = {res['bD_high']:.6g} (SE {res['bD_high_SE']:.3g})  "
        f"95%CI [{res['bD_high_CI95_low']:.6g}, {res['bD_high_CI95_high']:.6g}]"
    )
    print(
        f"bK_low (K=0.1µM) = {res['bK_low']:.6g} (SE {res['bK_low_SE']:.3g})  "
        f"95%CI [{res['bK_low_CI95_low']:.6g}, {res['bK_low_CI95_high']:.6g}]"
    )
    print(
        f"bK_high (K=10µM) = {res['bK_high']:.6g} (SE {res['bK_high_SE']:.3g})  "
        f"95%CI [{res['bK_high_CI95_low']:.6g}, {res['bK_high_CI95_high']:.6g}]"
    )
    print(f"DOF={res['dof']}   weighted sigma^2={res['sigma2']:.3g}")
    print("\nPredicted vs Observed:")
    print(res["pred_table"].to_string(index=False))


def run_all(
    alp_path: str, collagen_path: str, ars_path: str, out_dir: str = "fit_outputs"
) -> None:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    tables = {
        "ALP": read_endpoint_table(alp_path),
        "Collagen": read_endpoint_table(collagen_path),
        "ARS": read_endpoint_table(ars_path),
    }

    results = []
    for assay, df in tables.items():
        res = wls_fit_dk_effects(df, assay)
        results.append(res)
        report(res)

        # save per assay
        res["pred_table"].to_csv(out_path / f"pred_{assay.lower()}.csv", index=False)
        res["fit_table"].to_csv(out_path / f"fitdata_{assay.lower()}.csv", index=False)

    # summary csv
    summary = pd.DataFrame(
        [
            {
                "Assay": r["assay"],
                "bD_low": r["bD_low"],
                "bD_low_SE": r["bD_low_SE"],
                "bD_low_CI95_low": r["bD_low_CI95_low"],
                "bD_low_CI95_high": r["bD_low_CI95_high"],
                "bD_high": r["bD_high"],
                "bD_high_SE": r["bD_high_SE"],
                "bD_high_CI95_low": r["bD_high_CI95_low"],
                "bD_high_CI95_high": r["bD_high_CI95_high"],
                "bK_low": r["bK_low"],
                "bK_low_SE": r["bK_low_SE"],
                "bK_low_CI95_low": r["bK_low_CI95_low"],
                "bK_low_CI95_high": r["bK_low_CI95_high"],
                "bK_high": r["bK_high"],
                "bK_high_SE": r["bK_high_SE"],
                "bK_high_CI95_low": r["bK_high_CI95_low"],
                "bK_high_CI95_high": r["bK_high_CI95_high"],
            }
            for r in results
        ]
    )
    summary.to_csv(out_path / "dk_fit_summary.csv", index=False)
    print("\nSaved: pred_*.csv, fitdata_*.csv, dk_fit_summary.csv")

    # plots
    for r in results:
        plot_fit_diagnostics(r, out_path)


def plot_fit_diagnostics(res: dict, out_path: Path) -> None:
    """
    Professional diagnostic plots per assay:
    1) Observed vs Predicted (normalized by C2-C1)
    2) z residuals vs condition (fit set)
    3) Contribution by condition (Dex + MK-4 dose indicators)
    """
    assay = res["assay"]
    pred = res["pred_table"].copy()
    fit = res["fit_table"].copy()
    denom = float(res["denom"])
    y1 = float(res["Y_C1"])
    bD_low = float(res["bD_low"])
    bD_high = float(res["bD_high"])
    bK_low = float(res["bK_low"])
    bK_high = float(res["bK_high"])

    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.4,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    color_obs = "#2C7FB8"
    color_resid = "#D95F0E"
    color_d = "#1f77b4"
    color_k = "#ff7f0e"
    grid_kw = {"alpha": 0.12, "linestyle": "--", "linewidth": 0.6}

    def style_ax(ax: plt.Axes) -> None:
        ax.grid(**grid_kw)
        ax.set_facecolor("#FBFBFB")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    # 1) Observed vs Predicted (normalized)
    fig, ax = plt.subplots(figsize=(4.8, 4.4))
    obs_norm = (pred["Mean"].to_numpy(float) - y1) / denom
    pred_norm = (pred["PredMean"].to_numpy(float) - y1) / denom
    ax.scatter(
        obs_norm,
        pred_norm,
        s=72,
        color=color_obs,
        edgecolor="white",
        linewidth=0.7,
        alpha=0.9,
        zorder=3,
    )
    for _, row in pred.iterrows():
        xo = (float(row["Mean"]) - y1) / denom
        yp = (float(row["PredMean"]) - y1) / denom
        ax.text(xo, yp, row["Condition"], fontsize=8, ha="left", va="bottom")
    lim_min = float(min(obs_norm.min(), pred_norm.min()))
    lim_max = float(max(obs_norm.max(), pred_norm.max()))
    pad = 0.05 * (lim_max - lim_min if lim_max > lim_min else 1.0)
    ax.plot(
        [lim_min - pad, lim_max + pad],
        [lim_min - pad, lim_max + pad],
        ls="--",
        color="#555555",
        lw=1.0,
    )
    ax.set_xlim(lim_min - pad, lim_max + pad)
    ax.set_ylim(lim_min - pad, lim_max + pad)
    ax.set_xlabel("Observed (normalized by C2–C1)")
    ax.set_ylabel("Predicted (normalized)")
    ax.set_title(f"{assay}: Observed vs Predicted (normalized)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path / f"{assay.lower()}_obs_vs_pred.png", dpi=300)
    fig.savefig(out_path / f"{assay.lower()}_obs_vs_pred.pdf")
    plt.close(fig)

    # 2) Residuals in z-space
    fig, ax = plt.subplots(figsize=(4.8, 2.8))
    y = fit["resid_z"].to_numpy(float)
    x = np.arange(len(fit))
    ax.axhline(0, color="#333333", lw=0.9)
    ax.scatter(
        x, y, s=62, color=color_resid, edgecolor="white", linewidth=0.7, alpha=0.9
    )
    ax.set_xticks(x, fit["Condition"].tolist())
    ax.set_ylabel("Residual z")
    ax.set_title(f"{assay}: Fit Residuals (z)")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path / f"{assay.lower()}_resid_z.png", dpi=300)
    fig.savefig(out_path / f"{assay.lower()}_resid_z.pdf")
    plt.close(fig)

    # 3) Contribution by condition (aD*D and aK*K)
    fig, ax = plt.subplots(figsize=(5.6, 3.0))
    conds = pred["Condition"].tolist()
    d = pred["D_nM"].to_numpy(float)
    k = pred["K_nM"].to_numpy(float)
    d_low = (np.isclose(d, 10.0)).astype(float)
    d_high = (np.isclose(d, 100.0)).astype(float)
    k_low = (np.isclose(k, 100.0)).astype(float)
    k_high = (np.isclose(k, 10000.0)).astype(float)

    contrib_d = bD_low * d_low + bD_high * d_high
    contrib_k = bK_low * k_low + bK_high * k_high
    x = np.arange(len(conds))
    ax.bar(x, contrib_d, color=color_d, alpha=0.85, label="Dex dose")
    ax.bar(x, contrib_k, bottom=contrib_d, color=color_k, alpha=0.85, label="MK-4 dose")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, conds)
    ax.set_ylabel("Modifier contribution (z-space)")
    ax.set_title(f"{assay}: Dose-level contributions")
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    style_ax(ax)
    fig.tight_layout()
    fig.savefig(out_path / f"{assay.lower()}_dk_contrib.png", dpi=300)
    fig.savefig(out_path / f"{assay.lower()}_dk_contrib.pdf")
    plt.close(fig)

    # Combined 3-panel figure (Science Advances-style)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8))
    ax0, ax1, ax2 = axes

    ax0.scatter(
        obs_norm,
        pred_norm,
        s=60,
        color=color_obs,
        edgecolor="white",
        linewidth=0.6,
        alpha=0.9,
        zorder=3,
    )
    for _, row in pred.iterrows():
        xo = (float(row["Mean"]) - y1) / denom
        yp = (float(row["PredMean"]) - y1) / denom
        ax0.text(xo, yp, row["Condition"], fontsize=7.5, ha="left", va="bottom")
    ax0.plot(
        [lim_min - pad, lim_max + pad],
        [lim_min - pad, lim_max + pad],
        ls="--",
        color="#666666",
        lw=0.9,
    )
    ax0.set_xlim(lim_min - pad, lim_max + pad)
    ax0.set_ylim(lim_min - pad, lim_max + pad)
    ax0.set_xlabel("Observed (norm)")
    ax0.set_ylabel("Predicted (norm)")
    ax0.set_title("Observed vs Predicted")
    style_ax(ax0)

    x_fit = np.arange(len(fit))
    ax1.axhline(0, color="#333333", lw=0.9)
    ax1.scatter(
        x_fit, y, s=55, color=color_resid, edgecolor="white", linewidth=0.6, alpha=0.9
    )
    ax1.set_xticks(x_fit, fit["Condition"].tolist())
    ax1.set_ylabel("Residual z")
    ax1.set_title("Fit Residuals")
    style_ax(ax1)

    ax2.bar(x, contrib_d, color=color_d, alpha=0.85, label="Dex")
    ax2.bar(x, contrib_k, bottom=contrib_d, color=color_k, alpha=0.85, label="MK-4")
    ax2.axhline(0, color="#333333", lw=0.8)
    ax2.set_xticks(x, conds)
    ax2.set_ylabel("Contribution (z)")
    ax2.set_title("Dose Contributions")
    ax2.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    style_ax(ax2)

    fig.suptitle(f"{assay}: Dex/MK‑4 Dose Effects", y=1.02, fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path / f"{assay.lower()}_dk_diagnostics.png", dpi=300)
    fig.savefig(out_path / f"{assay.lower()}_dk_diagnostics.pdf")
    plt.close(fig)

    # 4) Low vs high dose effects (observed vs predicted z)
    fig, ax = plt.subplots(figsize=(6.2, 3.2))
    # observed z for each condition (all, including C3..C8)
    pred_df = pred.copy()
    pred_df["z_obs"] = (pred_df["Mean"] - y1) / denom - 1.0
    pred_df["z_pred"] = (pred_df["PredMean"] - y1) / denom - 1.0

    def _mean_z(mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(pred_df.loc[mask, "z_obs"]))

    def _mean_z_pred(mask):
        if mask.sum() == 0:
            return np.nan
        return float(np.mean(pred_df.loc[mask, "z_pred"]))

    d = pred_df["D_nM"].to_numpy(float)
    k = pred_df["K_nM"].to_numpy(float)
    mask_d_low = np.isclose(d, 10.0)
    mask_d_high = np.isclose(d, 100.0)
    mask_k_low = np.isclose(k, 100.0)
    mask_k_high = np.isclose(k, 10000.0)

    labels = ["Dex 10 nM", "Dex 100 nM", "MK-4 0.1 µM", "MK-4 10 µM"]
    obs_means = [
        _mean_z(mask_d_low),
        _mean_z(mask_d_high),
        _mean_z(mask_k_low),
        _mean_z(mask_k_high),
    ]
    pred_means = [
        _mean_z_pred(mask_d_low),
        _mean_z_pred(mask_d_high),
        _mean_z_pred(mask_k_low),
        _mean_z_pred(mask_k_high),
    ]

    x = np.arange(len(labels))
    ax.plot(x, obs_means, marker="o", color=color_obs, label="Observed mean z")
    ax.plot(x, pred_means, marker="s", color=color_resid, label="Predicted mean z")
    ax.axhline(0, color="#333333", lw=0.8)
    ax.set_xticks(x, labels, rotation=15, ha="right")
    ax.set_ylabel("Mean z (normalized)")
    ax.set_title(f"{assay}: Low/High Dose Effects (Observed vs Predicted)")
    style_ax(ax)
    ax.legend(frameon=False, ncol=2, fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(out_path / f"{assay.lower()}_dose_effects.png", dpi=300)
    fig.savefig(out_path / f"{assay.lower()}_dose_effects.pdf")
    plt.close(fig)


if __name__ == "__main__":
    # Put this script in the same folder as your txt files, or change paths here:
    alp_path = "alp_result.txt"
    collagen_path = "collagen_result.txt"
    ars_path = "ars_result.txt"

    run_all(alp_path, collagen_path, ars_path, out_dir="fit_outputs")
