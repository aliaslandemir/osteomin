import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

try:
    from scipy.optimize import least_squares
except Exception as exc:
    raise ImportError("iterative6.py requires scipy. Install with: pip install scipy") from exc


CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
CONCENTRATIONS = np.array([
    [0.000, 0, 0, 0],      # C1
    [0.284, 10, 0, 0],     # C2
    [0.284, 10, 10, 0.1],  # C3
    [0.284, 10, 10, 10],   # C4
    [0.284, 10, 100, 0.1], # C5
    [0.284, 10, 100, 10],  # C6
    [0.284, 10, 100, 0],   # C7
    [0.284, 10, 0, 10],    # C8
], dtype=float)


AA_DECAY_PER_HOUR = 0.11
MEDIA_REFRESH_HOURS = 24.0
DEX_EC50_NM = 10.0
MK_EC50_UM = 0.5
KAC_FIXED_MM = 0.1

ALP_DAY = 10.0
COLL_DAY = 14.0
ARS_DAY = 28.0

MIN_TAU_DAY = 14.0
MIN_HILL_N = 4.0


HERE = Path(__file__).resolve().parent
ALP_FILE = HERE / "alp_metadata_table.csv"
COLL_FILE = HERE / "coll_metadata_table.csv"
ARS_FILE = HERE / "ars_metadata_table.csv"
CELL_FILE = HERE / "cellnumbers.csv"
PLOTS_DIR = HERE / "plots6"


def set_style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.linestyle": "--",
        "grid.linewidth": 0.6,
        "grid.alpha": 0.3,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
        "savefig.facecolor": "white",
    })


def _normalize_columns(df):
    out = df.copy()
    out.columns = [c.strip().lower() for c in out.columns]
    return out


def _find_col(df, candidates, label):
    cols = set(df.columns)
    for c in candidates:
        if c in cols:
            return c
    raise ValueError(f"Missing {label} column. Found: {sorted(df.columns)}")


def load_assay_table(path, expected_reps, value_candidates):
    df = _normalize_columns(pd.read_csv(path))
    ccol = _find_col(df, ["condition", "cond"], "condition")
    vcol = _find_col(df, value_candidates, "value")
    df[ccol] = df[ccol].astype(str).str.upper().str.strip()
    df[vcol] = pd.to_numeric(df[vcol], errors="coerce")

    rows = []
    for cond in CONDITIONS:
        vals = df.loc[df[ccol] == cond, vcol].dropna().values.astype(float)
        if len(vals) != expected_reps:
            raise ValueError(f"{path.name}: {cond} expected {expected_reps} reps, got {len(vals)}")
        rows.append(vals)
    return np.array(rows)


def load_cell_terms(path):
    df = _normalize_columns(pd.read_csv(path))
    day_col = _find_col(df, ["day"], "day")
    pre_col = _find_col(df, ["n_preob", "preob", "pre_ob"], "n_preob")
    ob_col = _find_col(df, ["n_osteoblast", "osteoblast", "ob"], "n_osteoblast")

    day = pd.to_numeric(df[day_col], errors="coerce").values.astype(float)
    pre = pd.to_numeric(df[pre_col], errors="coerce").values.astype(float)
    ob = pd.to_numeric(df[ob_col], errors="coerce").values.astype(float)

    ok = np.isfinite(day) & np.isfinite(pre) & np.isfinite(ob)
    day, pre, ob = day[ok], pre[ok], ob[ok]
    idx = np.argsort(day)
    day, pre, ob = day[idx], pre[idx], ob[idx]
    if len(day) == 0:
        raise ValueError("cellnumbers.csv has no valid rows")

    aa_decay_factor = (1.0 - np.exp(-AA_DECAY_PER_HOUR * MEDIA_REFRESH_HOURS)) / (
        AA_DECAY_PER_HOUR * MEDIA_REFRESH_HOURS
    )

    def at_day(v, t):
        return float(np.interp(t, day, v))

    def _grid_to_t(t):
        d = day[day < t]
        if len(d) == 0 or d[-1] != t:
            d = np.concatenate([d, [t]])
        return d

    def integrate_to_t(vals, t):
        d = _grid_to_t(t)
        vv = np.interp(d, day, vals)
        return float(np.trapezoid(vv, x=d))

    def integrate_combo_to_t(pre_w, ob_w, t):
        d = _grid_to_t(t)
        pv = np.interp(d, day, pre)
        ov = np.interp(d, day, ob)
        return float(np.trapezoid(pre_w * pv + ob_w * ov, x=d))

    pre_alp, ob_alp = at_day(pre, ALP_DAY), at_day(ob, ALP_DAY)
    pre_coll, ob_coll = at_day(pre, COLL_DAY), at_day(ob, COLL_DAY)
    pre_ars, ob_ars = at_day(pre, ARS_DAY), at_day(ob, ARS_DAY)

    def condition_drive(pre_val, ob_val):
        total = pre_val + ob_val
        drive = np.full(len(CONDITIONS), total, dtype=float)
        frac_pre = pre_val / max(total, 1e-12)
        mode = "total_from_csv"
        return drive, mode, frac_pre

    alp_drive, alp_c1_mode, alp_frac = condition_drive(pre_alp, ob_alp)
    coll_drive, coll_c1_mode, coll_frac = condition_drive(pre_coll, ob_coll)
    ars_drive, ars_c1_mode, ars_frac = condition_drive(pre_ars, ob_ars)

    alp_integral = integrate_combo_to_t(1.0, 0.35, ALP_DAY)
    coll_integral = integrate_to_t(ob, COLL_DAY)
    ars_integral = integrate_to_t(ob, ARS_DAY)

    drive_endpoint = {
        "alp": alp_drive.copy(),
        "coll": coll_drive.copy(),
        "ars": ars_drive.copy(),
    }
    drive_integral = {
        "alp": np.full(len(CONDITIONS), alp_integral, dtype=float),
        "coll": np.full(len(CONDITIONS), coll_integral, dtype=float),
        "ars": np.full(len(CONDITIONS), ars_integral, dtype=float),
    }
    drive_hybrid = {
        "alp": alp_drive.copy(),
        "coll": np.full(len(CONDITIONS), coll_integral, dtype=float),
        "ars": np.full(len(CONDITIONS), ars_integral, dtype=float),
    }

    return {
        "aa_decay_factor": float(aa_decay_factor),
        "drive_endpoint": drive_endpoint,
        "drive_integral": drive_integral,
        "drive_hybrid": drive_hybrid,
        "pre_day_alp": pre_alp,
        "ob_day_alp": ob_alp,
        "pre_day_coll": pre_coll,
        "ob_day_coll": ob_coll,
        "pre_day_ars": pre_ars,
        "ob_day_ars": ob_ars,
        "c1_mode_alp": alp_c1_mode,
        "c1_mode_coll": coll_c1_mode,
        "c1_mode_ars": ars_c1_mode,
        "c1_pre_frac_alp": float(alp_frac),
        "c1_pre_frac_coll": float(coll_frac),
        "c1_pre_frac_ars": float(ars_frac),
        "integral_alp": alp_integral,
        "integral_coll": coll_integral,
        "integral_ars": ars_integral,
    }


def get_drive_set(cell_terms, mode):
    if mode == "endpoint":
        return cell_terms["drive_endpoint"]
    if mode == "integral":
        return cell_terms["drive_integral"]
    if mode == "hybrid":
        return cell_terms["drive_hybrid"]
    raise ValueError(f"Unknown drive mode: {mode}")


def condition_gates(cell_terms):
    aa = CONCENTRATIONS[:, 0]
    dex = CONCENTRATIONS[:, 2]
    mk = CONCENTRATIONS[:, 3]
    aa_eff = aa * cell_terms["aa_decay_factor"]
    g_a = aa_eff / (KAC_FIXED_MM + aa_eff + 1e-12)
    g_d = dex / (DEX_EC50_NM + dex + 1e-12)
    g_k = mk / (MK_EC50_UM + mk + 1e-12)
    return g_a, g_d, g_k


def mineral_time_gate(day=ARS_DAY, tau=MIN_TAU_DAY, n=MIN_HILL_N):
    return float((day**n) / (tau**n + day**n + 1e-12))


def _assay_drive_and_h_from_set(assay_idx, drive_set):
    if assay_idx == 0:
        return drive_set["alp"], 1.0
    if assay_idx == 1:
        return drive_set["coll"], 1.0
    return drive_set["ars"], mineral_time_gate()


def predict_conditions(theta, cell_terms, drive_set):
    rbP, roP, kDP, kKP, kDKP, rbC, roC, kDC, kKC, kDKC, rbH, roH, kDH, kKH, kDKH = theta
    g_a, g_d, g_k = condition_gates(cell_terms)
    h_ars = mineral_time_gate()

    m_p = 1.0 + kDP * g_d + kKP * g_k + kDKP * g_d * g_k
    m_c = 1.0 + kDC * g_d + kKC * g_k + kDKC * g_d * g_k
    m_h = 1.0 + kDH * g_d + kKH * g_k + kDKH * g_d * g_k

    alp = (rbP + roP * g_a * m_p) * drive_set["alp"]
    coll = (rbC + roC * g_a * m_c) * drive_set["coll"]
    ars = (rbH + roH * g_a * h_ars * m_h) * drive_set["ars"]
    return alp, coll, ars


def make_dataset(alp, coll, ars, cond_use=None):
    if cond_use is None:
        cond_use = list(range(len(CONDITIONS)))
    cond_idx = []
    y = []
    assay = []
    for i in cond_use:
        for v in alp[i]:
            cond_idx.append(i)
            y.append(float(v))
            assay.append(0)
        for v in coll[i]:
            cond_idx.append(i)
            y.append(float(v))
            assay.append(1)
        for v in ars[i]:
            cond_idx.append(i)
            y.append(float(v))
            assay.append(2)
    return np.array(cond_idx, dtype=int), np.array(y), np.array(assay, dtype=int)


def residual_full(theta, cell_terms, drive_set, cond_idx, y, assay, scales, reg=2e-4):
    alp_p, coll_p, ars_p = predict_conditions(theta, cell_terms, drive_set)
    pred = np.empty_like(y, dtype=float)
    for j in range(len(y)):
        c = cond_idx[j]
        if assay[j] == 0:
            pred[j] = alp_p[c]
        elif assay[j] == 1:
            pred[j] = coll_p[c]
        else:
            pred[j] = ars_p[c]

    r_data = (y - pred) / scales[assay]

    mod_idx = np.array([1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14], dtype=int)
    r_reg = np.sqrt(reg) * theta[mod_idx]

    _, g_d, g_k = condition_gates(cell_terms)
    m_p = 1.0 + theta[2] * g_d + theta[3] * g_k + theta[4] * g_d * g_k
    m_c = 1.0 + theta[7] * g_d + theta[8] * g_k + theta[9] * g_d * g_k
    m_h = 1.0 + theta[12] * g_d + theta[13] * g_k + theta[14] * g_d * g_k
    neg = np.concatenate([
        np.maximum(0.0, -m_p),
        np.maximum(0.0, -m_c),
        np.maximum(0.0, -m_h),
    ])
    r_pos = np.sqrt(15.0) * neg

    return np.concatenate([r_data, r_reg, r_pos])


def build_initial_theta(alp, coll, ars, cell_terms, drive_set):
    g_a, g_d, g_k = condition_gates(cell_terms)
    mats = [alp, coll, ars]
    theta = np.zeros(15, dtype=float)
    idx_map = [
        (0, 1, 2, 3, 4),
        (5, 6, 7, 8, 9),
        (10, 11, 12, 13, 14),
    ]
    ro_hi = [3.0, 6.0, 3.0]
    for a in range(3):
        mat = mats[a]
        ib, io, iD, iK, iDK = idx_map[a]
        drive, h = _assay_drive_and_h_from_set(a, drive_set)
        rb = float(np.mean(mat[0]) / max(drive[0], 1e-12))
        y2 = float(np.mean(mat[1]) / max(drive[1], 1e-12))
        y7 = float(np.mean(mat[6]) / max(drive[6], 1e-12))
        y8 = float(np.mean(mat[7]) / max(drive[7], 1e-12))
        ro = (y2 - rb) / max(g_a[1] * h, 1e-12)
        ro = float(np.clip(max(1e-6, ro), 0.0, ro_hi[a]))
        if ro <= 1e-6:
            kD = 0.0
            kK = 0.0
        else:
            kD = ((y7 - rb) / max(ro * g_a[6] * h, 1e-12) - 1.0) / max(g_d[6], 1e-12)
            kK = ((y8 - rb) / max(ro * g_a[7] * h, 1e-12) - 1.0) / max(g_k[7], 1e-12)
        theta[ib] = rb
        theta[io] = ro
        theta[iD] = float(np.clip(kD, -4.0, 4.0))
        theta[iK] = float(np.clip(kK, -4.0, 4.0))
        theta[iDK] = 0.0
    return theta


def fit_mode_least_squares(alp, coll, ars, cell_terms, mode, n_starts=5, seed=42):
    drive_set = get_drive_set(cell_terms, mode)
    cond_idx, y, assay = make_dataset(alp, coll, ars, cond_use=list(range(len(CONDITIONS))))
    scales = np.array([np.std(alp, ddof=1), np.std(coll, ddof=1), np.std(ars, ddof=1)], dtype=float) + 1e-9

    lo = np.array([
        0.0, 0.0, -6.0, -6.0, -6.0,
        0.0, 0.0, -6.0, -6.0, -6.0,
        0.0, 0.0, -6.0, -6.0, -6.0,
    ], dtype=float)
    hi = np.array([
        2.0, 3.0, 6.0, 6.0, 6.0,
        6.0, 6.0, 6.0, 6.0, 6.0,
        2.0, 3.0, 6.0, 6.0, 6.0,
    ], dtype=float)

    theta0 = build_initial_theta(alp, coll, ars, cell_terms, drive_set)
    rng = np.random.default_rng(seed)
    starts = [theta0]
    span = hi - lo
    for _ in range(max(0, n_starts - 1)):
        x = np.clip(theta0 + rng.normal(0.0, 0.06, size=theta0.shape) * span, lo, hi)
        starts.append(x)

    rows = []
    best = None
    for i, x0 in enumerate(starts, start=1):
        ls = least_squares(
            lambda v: residual_full(v, cell_terms, drive_set, cond_idx, y, assay, scales, reg=2e-4),
            x0=x0,
            bounds=(lo, hi),
            loss="soft_l1",
            f_scale=0.8,
            x_scale="jac",
            max_nfev=50000,
            verbose=0,
        )
        cost = float(2.0 * ls.cost)
        rows.append({
            "mode": mode,
            "run": i,
            "cost": cost,
            "success": bool(ls.success),
            "nfev": int(ls.nfev),
        })
        if best is None or cost < float(2.0 * best.cost):
            best = ls

    hist_df = pd.DataFrame(rows)
    return best, hist_df, theta0, drive_set, scales


def _r2(y, yp):
    sst = np.sum((y - y.mean()) ** 2)
    if sst <= 0:
        return np.nan
    return float(1.0 - np.sum((y - yp) ** 2) / sst)


def _rmse(y, yp):
    return float(np.sqrt(np.mean((y - yp) ** 2)))


def _mae(y, yp):
    return float(np.mean(np.abs(y - yp)))


def _bias(y, yp):
    return float(np.mean(yp - y))


def _condition_summary(values_matrix):
    mean = values_matrix.mean(axis=1)
    n_rep = values_matrix.shape[1]
    if n_rep > 1:
        sd = values_matrix.std(axis=1, ddof=1)
        sem = sd / np.sqrt(n_rep)
    else:
        sd = np.zeros(values_matrix.shape[0], dtype=float)
        sem = np.zeros(values_matrix.shape[0], dtype=float)
    return mean, sd, sem


def fit_parameters(alp, coll, ars, cell_terms, seed=42):
    mode_records = []
    candidates = {}
    for i, mode in enumerate(["endpoint", "integral", "hybrid"]):
        best, hist_df, theta0, drive_set, scales = fit_mode_least_squares(
            alp, coll, ars, cell_terms, mode=mode, n_starts=5, seed=seed + 37 * i
        )
        theta = best.x
        alp_p, coll_p, ars_p = predict_conditions(theta, cell_terms, drive_set)
        r2_alp = _r2(alp.mean(axis=1), alp_p)
        r2_coll = _r2(coll.mean(axis=1), coll_p)
        r2_ars = _r2(ars.mean(axis=1), ars_p)
        obj = float(np.sum(residual_full(theta, cell_terms, drive_set, *make_dataset(alp, coll, ars), scales, reg=2e-4) ** 2))
        mode_records.append({
            "mode": mode,
            "objective": obj,
            "final_cost": float(2.0 * best.cost),
            "r2_alp": r2_alp,
            "r2_collagen": r2_coll,
            "r2_ars": r2_ars,
            "r2_mean": float(np.nanmean([r2_alp, r2_coll, r2_ars])),
        })
        candidates[mode] = {
            "ls": best,
            "history": hist_df,
            "theta0": theta0,
            "drive_set": drive_set,
            "scales": scales,
        }

    mode_df = pd.DataFrame(mode_records).sort_values(["objective", "r2_mean"], ascending=[True, False]).reset_index(drop=True)
    selected_mode = str(mode_df.iloc[0]["mode"])
    sel = candidates[selected_mode]

    theta = sel["ls"].x.copy()
    names = [
        "r_basal_P", "r_osteo_P", "k_DP", "k_KP", "k_DKP",
        "r_basal_C", "r_osteo_C", "k_DC", "k_KC", "k_DKC",
        "r_basal_H", "r_osteo_H", "k_DH", "k_KH", "k_DKH",
    ]
    status = np.array(["fitted_simple_ls"] * 15, dtype=object)

    se = np.full(theta.shape[0], np.nan, dtype=float)
    try:
        j = sel["ls"].jac
        dof = max(1, len(sel["ls"].fun) - len(sel["ls"].x))
        sigma2 = float(np.sum(sel["ls"].fun ** 2) / dof)
        cov = sigma2 * np.linalg.pinv(j.T @ j)
        se[:] = np.sqrt(np.maximum(np.diag(cov), 0.0))
    except Exception:
        pass

    fit_meta = {
        "names": names,
        "status": status,
        "ls": sel["ls"],
        "history_df": sel["history"],
        "theta0": sel["theta0"],
        "drive_set": sel["drive_set"],
        "selected_mode": selected_mode,
        "mode_df": mode_df,
        "objective_selected": float(mode_df.loc[mode_df["mode"] == selected_mode, "objective"].iloc[0]),
    }
    return theta, se, fit_meta


ASSAY_STYLE = {
    "ALP": {"obs": "#2C7FB8", "band": "#BFD9EE", "model": "#D62728"},
    "Collagen": {"obs": "#2A9D8F", "band": "#BFE3DB", "model": "#D62728"},
    "ARS": {"obs": "#8C6BB1", "band": "#D9CCE8", "model": "#D62728"},
}


def metrics_table(alp, coll, ars, alp_p, coll_p, ars_p):
    alp_m, _, _ = _condition_summary(alp)
    coll_m, _, _ = _condition_summary(coll)
    ars_m, _, _ = _condition_summary(ars)

    rows = []
    for name, y, yp in [
        ("ALP", alp_m, alp_p),
        ("Collagen", coll_m, coll_p),
        ("ARS", ars_m, ars_p),
    ]:
        rows.append({
            "assay": name,
            "n_conditions": len(y),
            "R2_condition_means": _r2(y, yp),
            "RMSE_condition_means": _rmse(y, yp),
            "MAE_condition_means": _mae(y, yp),
            "Bias_model_minus_exp": _bias(y, yp),
        })
    return pd.DataFrame(rows)


def plot_scatter_with_fit(values_matrix, pred, title, ylabel, outpath):
    x = np.arange(len(CONDITIONS))
    plt.figure(figsize=(8.5, 5))
    n_reps = values_matrix.shape[1]
    jitter = np.linspace(-0.16, 0.16, n_reps) if n_reps > 1 else np.array([0.0])
    for i in range(len(CONDITIONS)):
        y = values_matrix[i]
        xj = np.full_like(y, x[i], dtype=float) + jitter[: len(y)]
        plt.scatter(xj, y, s=32, color="#1f77b4", alpha=0.9, edgecolors="white", linewidths=0.45)
    plt.plot(x, pred, color="#d62728", linewidth=1.2, alpha=0.85)
    plt.scatter(x, pred, s=56, color="#d62728", marker="D", label="Model")
    plt.xticks(x, CONDITIONS)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_condition_comparison(alp, coll, ars, alp_p, coll_p, ars_p, outpath):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2), sharex=True)
    x = np.arange(len(CONDITIONS))
    assays = [
        ("ALP", alp, alp_p, "#1f77b4", "#d62728"),
        ("Collagen", coll, coll_p, "#2ca02c", "#d62728"),
        ("ARS", ars, ars_p, "#9467bd", "#d62728"),
    ]
    for ax, (name, mat, pred, obs_c, fit_c) in zip(axes, assays):
        m, _, sem = _condition_summary(mat)
        ax.errorbar(x, m, yerr=sem, fmt="o", color=obs_c, capsize=3, label="Experiment mean ± SEM")
        ax.plot(x, pred, color=fit_c, linewidth=1.4, marker="D", markersize=4.6, label="Model")
        ax.set_title(name)
        ax.set_xticks(x)
        ax.set_xticklabels(CONDITIONS, rotation=0)
        ax.set_xlabel("Condition")
        ax.set_ylabel("Signal")
        ax.grid(True, axis="y")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.03))
    fig.suptitle("Condition-wise Experimental vs Model", y=1.10, fontsize=12)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parity_single(name, y, yp, outpath, labels=None):
    style = ASSAY_STYLE[name]
    lo = min(np.min(y), np.min(yp))
    hi = max(np.max(y), np.max(yp))
    pad = 0.08 * (hi - lo + 1e-12)

    fig, ax = plt.subplots(figsize=(5.2, 5.0))
    ax.scatter(
        y,
        yp,
        s=64,
        color=style["obs"],
        edgecolors="white",
        linewidths=0.8,
        alpha=0.95,
        zorder=3,
    )
    if labels is not None:
        span = hi - lo + 1e-12
        dx = 0.012 * span
        dy = 0.012 * span
        for i, lab in enumerate(labels):
            ax.text(y[i] + dx, yp[i] + dy, str(lab), fontsize=8.2, color="#333333", alpha=0.9, zorder=4)
    ax.plot(
        [lo - pad, hi + pad],
        [lo - pad, hi + pad],
        color="#4A4A4A",
        linewidth=1.2,
        linestyle="--",
        zorder=2,
    )
    ax.set_title(f"{name}: Parity Plot", pad=10)
    ax.set_xlabel("Experimental mean")
    ax.set_ylabel("Model prediction")
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.grid(True, axis="both", linestyle="--", linewidth=0.6, alpha=0.25)
    ax.set_facecolor("white")

    r2 = _r2(y, yp)
    rmse = _rmse(y, yp)
    ax.text(
        0.03,
        0.97,
        f"R² = {r2:.3f}\nRMSE = {rmse:.3g}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CFCFCF"),
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_parity(alp, coll, ars, alp_p, coll_p, ars_p, outdir):
    assays = [
        ("ALP", alp.mean(axis=1), alp_p),
        ("Collagen", coll.mean(axis=1), coll_p),
        ("ARS", ars.mean(axis=1), ars_p),
    ]
    for name, y, yp in assays:
        slug = name.lower().replace(" ", "_")
        plot_parity_single(name, y, yp, outdir / f"parity_{slug}_6.png", labels=CONDITIONS)


def plot_parameter_forest(names, theta, se, outpath):
    z = 1.96
    lo = theta - z * se
    hi = theta + z * se
    y = np.arange(len(names))[::-1]
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    for i in range(len(names)):
        if np.isfinite(lo[i]) and np.isfinite(hi[i]):
            ax.plot([lo[i], hi[i]], [y[i], y[i]], color="#4c78a8", linewidth=1.5)
        ax.scatter(theta[i], y[i], color="#d62728", s=28, zorder=3)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels(names)
    ax.set_xlabel("Estimate (95% approx CI)")
    ax.set_title("Parameter Estimates")
    ax.grid(True, axis="x")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_concentration_matrix(outpath):
    labels = ["AA (mM)", "beta-GP (mM)", "Dex (nM)", "MK-4 (uM)"]
    units = ["mM", "mM", "nM", "uM"]
    raw = CONCENTRATIONS.T
    norm = np.zeros_like(raw, dtype=float)
    for i in range(raw.shape[0]):
        m = np.max(raw[i])
        if m > 0:
            norm[i] = raw[i] / m

    fig, ax = plt.subplots(figsize=(10.0, 3.8))
    im = ax.imshow(norm, cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(CONDITIONS)))
    ax.set_xticklabels(CONDITIONS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels)
    ax.set_title("Concentration Design Matrix (row-normalized)")
    for i in range(raw.shape[0]):
        for j in range(raw.shape[1]):
            t = "0" if raw[i, j] == 0 else f"{raw[i, j]:g} {units[i]}"
            color = "#1a1a1a" if norm[i, j] < 0.55 else "white"
            ax.text(j, i, t, ha="center", va="center", fontsize=8.8, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Relative level")
    ax.set_xticks(np.arange(-0.5, len(CONDITIONS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_single(name, mat, pred, outpath):
    style = ASSAY_STYLE[name]
    x = np.arange(len(CONDITIONS))
    m, _, sem = _condition_summary(mat)

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.fill_between(x, m - sem, m + sem, color=style["band"], alpha=0.9, linewidth=0, zorder=1)
    ax.plot(
        x,
        m,
        marker="o",
        color=style["obs"],
        linewidth=2.0,
        markersize=5.8,
        label="Experimental mean ± SEM",
        zorder=3,
    )
    ax.plot(
        x,
        pred,
        marker="D",
        color=style["model"],
        linewidth=1.8,
        markersize=5.4,
        label="Model",
        zorder=4,
    )
    ax.set_title(f"{name}: Experimental vs Model", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(CONDITIONS)
    ax.set_xlabel("Condition")
    ax.set_ylabel("Signal")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.7, alpha=0.25)
    ax.set_facecolor("white")
    ax.text(
        0.03,
        0.96,
        f"R² = {_r2(m, pred):.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#CFCFCF"),
    )
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_summary_figure(alp, coll, ars, alp_p, coll_p, ars_p, outdir):
    panels = [
        ("ALP", alp, alp_p),
        ("Collagen", coll, coll_p),
        ("ARS", ars, ars_p),
    ]
    for name, mat, pred in panels:
        slug = name.lower().replace(" ", "_")
        plot_summary_single(name, mat, pred, outdir / f"summary_{slug}_6.png")


def plot_optimization_history(history_df, outpath):
    fig, ax = plt.subplots(figsize=(7.4, 4.6))
    if len(history_df) > 0:
        runs = sorted(history_df["run"].unique())
        for i, run_id in enumerate(runs):
            sub = history_df[history_df["run"] == run_id].sort_values("run")
            x = np.arange(1, len(sub) + 1)
            ax.plot(x, sub["cost"].values, marker="o", linewidth=1.4, alpha=0.9, label=f"Run {int(run_id)}")
    ax.set_title("Simple Least-Squares Optimization History")
    ax.set_xlabel("Run Index")
    ax.set_ylabel("Cost (2*least_squares.cost)")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.35)
    if len(history_df) > 0:
        ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300, bbox_inches="tight")
    plt.close(fig)


def fit_report(theta, se, fit_meta, alp, coll, ars, cell_terms):
    names = fit_meta["names"]
    status = fit_meta["status"]
    ls = fit_meta["ls"]
    theta0 = fit_meta["theta0"]
    selected_mode = fit_meta["selected_mode"]
    mode_df = fit_meta["mode_df"]
    objective_selected = fit_meta["objective_selected"]
    drive_set = fit_meta["drive_set"]
    alp_p, coll_p, ars_p = predict_conditions(theta, cell_terms, drive_set)
    alp_m, coll_m, ars_m = alp.mean(axis=1), coll.mean(axis=1), ars.mean(axis=1)

    lines = []
    lines.append("=== Fit Summary (iterative6) ===")
    lines.append("Method: simple global least-squares fit (no stage locking)")
    lines.append("Optimizer: scipy least_squares with multiple starts")
    lines.append(f"Selected drive mode: {selected_mode}")
    lines.append(f"K_AC fixed value: {KAC_FIXED_MM:.6g} mM")
    lines.append(f"Mineral gate h(t): tau={MIN_TAU_DAY:g} day, n={MIN_HILL_N:g}, h(ARS_DAY)={mineral_time_gate():.4f}")
    lines.append(f"Endpoint days: ALP={ALP_DAY:g}, Collagen={COLL_DAY:g}, ARS={ARS_DAY:g}")
    lines.append(f"Final local cost: {float(2 * ls.cost):.6f}")
    lines.append(f"Converged: {ls.success} ({ls.message})")
    lines.append(f"Selected objective value: {objective_selected:.6f}")
    lines.append("")
    lines.append(
        f"Endpoint cells: ALP(pre={cell_terms['pre_day_alp']:.2f}, ob={cell_terms['ob_day_alp']:.2f}), "
        f"Coll(pre={cell_terms['pre_day_coll']:.2f}, ob={cell_terms['ob_day_coll']:.2f}), "
        f"ARS(pre={cell_terms['pre_day_ars']:.2f}, ob={cell_terms['ob_day_ars']:.2f})"
    )
    lines.append(
        f"C1 drive mode: ALP={cell_terms['c1_mode_alp']} (pre frac={cell_terms['c1_pre_frac_alp']:.3f}), "
        f"Coll={cell_terms['c1_mode_coll']} (pre frac={cell_terms['c1_pre_frac_coll']:.3f}), "
        f"ARS={cell_terms['c1_mode_ars']} (pre frac={cell_terms['c1_pre_frac_ars']:.3f})"
    )
    lines.append("")
    lines.append(f"R2 ALP: {_r2(alp_m, alp_p):.4f}")
    lines.append(f"R2 Collagen: {_r2(coll_m, coll_p):.4f}")
    lines.append(f"R2 ARS: {_r2(ars_m, ars_p):.4f}")
    lines.append("")
    lines.append("Drive mode candidates:")
    for _, r in mode_df.iterrows():
        lines.append(
            f"  {r['mode']}: objective={r['objective']:.6f}, "
            f"R2(ALP/Coll/ARS)=({r['r2_alp']:.4f}/{r['r2_collagen']:.4f}/{r['r2_ars']:.4f}), "
            f"R2_mean={r['r2_mean']:.4f}, cost={r['final_cost']:.6f}"
        )
    lines.append("")
    lines.append("Initial seed (selected mode):")
    for n, v in zip(names, theta0):
        lines.append(f"{n} init = {v:.6e}")
    lines.append("")
    lines.append("Parameters (estimate +/- approx SE)")
    for n, v, s, st in zip(names, theta, se, status):
        if np.isfinite(s):
            lines.append(f"{n} = {v:.6e} +/- {s:.3e} [{st}]")
        else:
            lines.append(f"{n} = {v:.6e} [{st}]")

    return "\n".join(lines), names


def main():
    set_style()
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    alp = load_assay_table(ALP_FILE, expected_reps=3, value_candidates=["alp_activity", "value", "raw"])
    coll = load_assay_table(COLL_FILE, expected_reps=5, value_candidates=["collagen", "raw", "value"])
    ars = load_assay_table(ARS_FILE, expected_reps=9, value_candidates=["ars", "raw", "value"])
    cell_terms = load_cell_terms(CELL_FILE)

    theta, se, fit_meta = fit_parameters(alp, coll, ars, cell_terms, seed=42)
    drive_set = fit_meta["drive_set"]
    alp_p, coll_p, ars_p = predict_conditions(theta, cell_terms, drive_set)

    plot_scatter_with_fit(alp, alp_p, "ALP by Condition (dots + fit)", "ALP Activity", PLOTS_DIR / "alp_by_condition_fit6.png")
    plot_scatter_with_fit(coll, coll_p, "Collagen by Condition (dots + fit)", "Collagen", PLOTS_DIR / "collagen_by_condition_fit6.png")
    plot_scatter_with_fit(ars, ars_p, "ARS by Condition (dots + fit)", "ARS", PLOTS_DIR / "ars_by_condition_fit6.png")
    plot_concentration_matrix(PLOTS_DIR / "concentrations_matrix6.png")
    plot_condition_comparison(alp, coll, ars, alp_p, coll_p, ars_p, PLOTS_DIR / "condition_comparison6.png")
    plot_parity(alp, coll, ars, alp_p, coll_p, ars_p, PLOTS_DIR)
    plot_summary_figure(alp, coll, ars, alp_p, coll_p, ars_p, PLOTS_DIR)

    metrics = metrics_table(alp, coll, ars, alp_p, coll_p, ars_p)
    metrics.to_csv(PLOTS_DIR / "gof_metrics6.csv", index=False)

    cond_table = pd.DataFrame({
        "condition": CONDITIONS,
        "alp_exp_mean": alp.mean(axis=1),
        "alp_model": alp_p,
        "coll_exp_mean": coll.mean(axis=1),
        "coll_model": coll_p,
        "ars_exp_mean": ars.mean(axis=1),
        "ars_model": ars_p,
    })
    cond_table.to_csv(PLOTS_DIR / "condition_comparison6.csv", index=False)

    summary, names = fit_report(theta, se, fit_meta, alp, coll, ars, cell_terms)
    (PLOTS_DIR / "fit_summary6.txt").write_text(summary + "\n", encoding="utf-8")

    fit_meta["history_df"].to_csv(PLOTS_DIR / "fit_runs6.csv", index=False)
    fit_meta["mode_df"].to_csv(PLOTS_DIR / "fit_mode_candidates6.csv", index=False)
    plot_optimization_history(fit_meta["history_df"], PLOTS_DIR / "optimization_history6.png")

    out = pd.DataFrame({
        "parameter": names,
        "estimate": theta,
        "approx_se": se,
        "fit_status": fit_meta["status"],
    })
    out.to_csv(PLOTS_DIR / "fitted_parameters6.csv", index=False)
    plot_parameter_forest(names, theta, se, PLOTS_DIR / "parameter_forest6.png")

    print(summary)
    print(f"Saved outputs to: {PLOTS_DIR}")


if __name__ == "__main__":
    main()
