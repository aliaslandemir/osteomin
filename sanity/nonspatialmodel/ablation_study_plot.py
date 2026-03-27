#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from dataclasses import replace

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
from osteomin_nonspatial import (
    DrugResponseParams,
    HOURS_PER_DAY,
    NonSpatialParams,
    _aa_concentration,
    _commit_activation_hour,
    _drug_modulation,
    extract_endpoints,
)
from scipy.stats import mannwhitneyu

HEATMAP_COLS = [
    "alp_day10",
    "collagen_day14",
    "hap_day28",
    "cell_count_day28",
]

HEATMAP_LABELS = ["ALP d10", "Collagen d14", "HAp d28", "Cells d28"]
SPATIAL_REF_COLS = ["alp_day10", "collagen_day14", "hap_day28", "cell_count_day28"]
SPATIAL_VARIANT = "spatial_reference_model"
SPATIAL_LABEL = "Spatial model (reference)"
VARIANT_ORDER = [
    ("full_model", "Nonspatial model"),
    ("no_osteogenic_commitment", "No osteogenic commitment"),
    ("no_commit_delay", "No commitment delay"),
    ("no_apoptosis", "No apoptosis"),
    ("no_drug_modulation", "No drug modulation"),
    ("no_collagen_maturation", "No collagen maturation delay"),
]
LINE_COLORS = {
    "full_model": "#0072B2",
    "no_drug_modulation": "#D55E00",
    "no_osteogenic_commitment": "#009E73",
}

VARIANT_LABELS = {
    "full_model": "Nonspatial model",
    "no_drug_modulation": "No drug modulation",
    "no_osteogenic_commitment": "No osteogenic commitment",
}


def _significance_stars(p_or_q: float) -> str:
    if not np.isfinite(p_or_q):
        return ""
    if p_or_q < 0.001:
        return "***"
    if p_or_q < 0.01:
        return "**"
    if p_or_q < 0.05:
        return "*"
    return ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Plot robust nonspatial ablation study figures from run outputs.")
    p.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "ablation_out")
    p.add_argument("--spatial-out", type=Path, default=Path(__file__).resolve().parents[2] / "main_out")
    p.add_argument("--band-type", choices=("prediction", "confidence"), default="prediction")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _bootstrap_ci_for_median(samples: np.ndarray, rng: np.random.Generator, n_boot: int = 700) -> tuple[np.ndarray, np.ndarray]:
    n_runs, n_cond = samples.shape
    boot = np.empty((n_boot, n_cond), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_runs, size=n_runs)
        boot[b, :] = np.median(samples[idx, :], axis=0)
    lo = np.percentile(boot, 2.5, axis=0)
    hi = np.percentile(boot, 97.5, axis=0)
    return lo, hi


def _bh_adjust(pvals: list[float]) -> list[float]:
    m = len(pvals)
    if m == 0:
        return []
    order = np.argsort(np.asarray(pvals, dtype=float))
    ranked = np.asarray(pvals, dtype=float)[order]
    q = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = min(prev, ranked[i] * m / rank)
        q[i] = val
        prev = val
    out = np.empty(m, dtype=float)
    out[order] = q
    return out.tolist()


def _load_spatial_reference(spatial_out: Path) -> dict[str, float]:
    ts_path = spatial_out / "timeseries.xlsx"
    run_json = spatial_out / "run.json"
    if not ts_path.exists():
        raise FileNotFoundError(f"Missing spatial timeseries: {ts_path}")
    if not run_json.exists():
        raise FileNotFoundError(f"Missing spatial run metadata: {run_json}")
    df = pd.read_excel(ts_path)
    need_days = {10: "alp_total", 14: "collagen_total", 28: "ha_total"}
    for day in need_days:
        if (df["day"] == day).sum() == 0:
            raise ValueError(f"Spatial timeseries missing day {day} in {ts_path}")
    run_cfg = json.loads(run_json.read_text(encoding="utf-8"))
    # Force 0/0 for cross-model comparability with this spatial run setup.
    dex = 0.0
    mk4 = 0.0
    grid_sites = int(run_cfg.get("model_size", {}).get("grid_sites", 0))
    if grid_sites <= 0:
        w = int(run_cfg.get("abm_cfg", {}).get("width", 0))
        h = int(run_cfg.get("abm_cfg", {}).get("height", 0))
        grid_sites = max(1, w * h)
    abm = run_cfg.get("abm_cfg", {}) if isinstance(run_cfg, dict) else {}
    field = run_cfg.get("field_cfg", {}) if isinstance(run_cfg, dict) else {}
    return {
        "alp_day10": float(df.loc[df["day"] == 10, "alp_total"].iloc[0]),
        "collagen_day14": float(df.loc[df["day"] == 14, "collagen_total"].iloc[0]),
        "hap_day28": float(df.loc[df["day"] == 28, "ha_total"].iloc[0]),
        "cell_count_day28": float(df.loc[df["day"] == 28, "n_total"].iloc[0]),
        "n_preob0": float(df.loc[df["day"] == 0, "n_preob"].iloc[0]),
        "n_ob0": float(df.loc[df["day"] == 0, "n_osteoblast"].iloc[0]),
        "dex_conc_nm": dex,
        "mk4_conc_um": mk4,
        "grid_sites": float(grid_sites),
        "osteo_induction_hours": float(abm.get("osteo_induction_hours", np.nan)),
        "commit_delay_hours": float(abm.get("commit_delay", np.nan)),
        "ha_tau_hours": float(field.get("ha_tau_hours", np.nan)),
        "run_cfg": run_cfg,
    }


def _make_base_nonspatial_cfg(spatial_vals: dict[str, float]) -> NonSpatialParams:
    run_cfg = spatial_vals.get("run_cfg", {})
    abm = run_cfg.get("abm_cfg", {}) if isinstance(run_cfg, dict) else {}
    field = run_cfg.get("field_cfg", {}) if isinstance(run_cfg, dict) else {}
    total_hours = int(run_cfg.get("total_hours", 672)) if isinstance(run_cfg, dict) else 672

    min_div = float(abm.get("min_div_hours", 38.0))
    max_div = float(abm.get("max_div_hours", 52.0))
    mean_div = max(1.0, 0.5 * (min_div + max_div))
    prolif_pre = float(abm.get("proliferation_chance", 1.0)) / mean_div
    prolif_ob = float(abm.get("ob_division_prob", 0.0)) / mean_div

    return NonSpatialParams(
        total_hours=total_hours,
        n_preob0=float(spatial_vals.get("n_preob0", 72.0)),
        n_ob0=float(spatial_vals.get("n_ob0", 0.0)),
        prolif_rate_preob=prolif_pre,
        prolif_rate_ob=prolif_ob,
        osteo_induction_hours=int(abm.get("osteo_induction_hours", 72)),
        commit_delay_hours=int(abm.get("commit_delay", 48)),
        k_commit_hourly=float(abm.get("commit_k_hourly", 0.0125)),
        apoptosis_preob=float(abm.get("apoptosis_prob", 0.0010)),
        apoptosis_ob=float(abm.get("ob_apoptosis_prob", 0.0)),
        dex_conc_nm=float(spatial_vals.get("dex_conc_nm", 0.0)),
        mk4_conc_um=float(spatial_vals.get("mk4_conc_um", 0.0)),
        alp_basal_rate=float(field.get("alp_basal_rate", 0.0949)),
        alp_osteo_rate=float(field.get("alp_osteo_rate", 0.810)),
        alp_initial=float(field.get("alp_initial", 0.8)),
        collagen_basal_rate=float(field.get("collagen_basal_rate", 0.00336)),
        collagen_osteo_rate=float(field.get("collagen_osteo_rate", 0.0130)),
        aa_initial=float(field.get("aa_initial", 50.0)),
        aa_half_life_hours=float(field.get("aa_half_life_hours", 8.0)),
        collagen_aa_half_sat=float(field.get("collagen_aa_half_sat", 5.0)),
        ha_osteo_rate=float(field.get("ha_osteo_rate", 0.102)),
        ha_tau_hours=float(field.get("ha_tau_hours", 384.0)),
        ha_hill_n=float(field.get("ha_hill_n", 2.0)),
        ha_collagen_half_sat=float(field.get("ha_collagen_half_sat", 1.0)),
    )


def _simulate_baseline_capacity(cfg: NonSpatialParams, grid_cap: float) -> pd.DataFrame:
    total_hours = max(1, int(cfg.total_hours))
    cap = max(1.0, float(grid_cap))

    n_pre = max(0.0, float(cfg.n_preob0))
    n_ob = max(0.0, float(cfg.n_ob0))
    if n_pre + n_ob > cap:
        sf = cap / max(1e-12, n_pre + n_ob)
        n_pre *= sf
        n_ob *= sf

    alp = max(0.0, float(cfg.alp_initial))
    collagen = 0.0
    collagen_mat = 0.0
    hap = 0.0
    hill_n = max(1.0, float(cfg.ha_hill_n))
    ob_age_n_sum = 0.0

    m_alp = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.alp_resp)
    m_col = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.collagen_resp)
    m_ha = _drug_modulation(cfg.dex_conc_nm, cfg.mk4_conc_um, cfg.ha_resp)

    rows: list[dict[str, float]] = []

    def append_row(hour: int) -> None:
        day = hour // HOURS_PER_DAY
        rows.append(
            {
                "hour": float(hour),
                "day": int(day),
                "n_preob": float(n_pre),
                "n_ob": float(n_ob),
                "alp_total": float(alp),
                "collagen_total": float(collagen),
                "collagen_mature_total": float(collagen_mat),
                "ha_total": float(hap),
            }
        )

    append_row(0)
    commit_start = _commit_activation_hour(cfg)
    k_commit = max(0.0, float(cfg.k_commit_hourly))

    for hour in range(1, total_hours + 1):
        n_tot = n_pre + n_ob
        crowd = max(0.0, 1.0 - n_tot / cap)

        births_pre = max(0.0, cfg.prolif_rate_preob) * n_pre * crowd
        births_ob = max(0.0, cfg.prolif_rate_ob) * n_ob * crowd

        commit_flow = 0.0
        if hour >= commit_start and k_commit > 0.0:
            commit_flow = min(k_commit * n_pre, n_pre)

        death_pre = max(0.0, cfg.apoptosis_preob) * n_pre
        death_ob = max(0.0, cfg.apoptosis_ob) * n_ob

        n_pre_next = n_pre + births_pre + births_ob - commit_flow - death_pre
        n_ob_next = n_ob + commit_flow - death_ob
        n_pre_next = max(0.0, n_pre_next)
        n_ob_next = max(0.0, n_ob_next)
        tot_next = n_pre_next + n_ob_next
        if tot_next > cap:
            sf = cap / max(1e-12, tot_next)
            n_pre_next *= sf
            n_ob_next *= sf

        alp += max(0.0, cfg.alp_basal_rate) * (n_pre + n_ob)
        if hour >= cfg.osteo_induction_hours:
            alp += max(0.0, cfg.alp_osteo_rate * m_alp) * n_ob

        collagen += max(0.0, cfg.collagen_basal_rate) * (n_pre + n_ob)
        if hour >= cfg.osteo_induction_hours:
            aa_t = _aa_concentration(cfg, hour)
            aa_gate = aa_t / (max(1e-9, cfg.collagen_aa_half_sat) + aa_t) if aa_t > 0.0 else 0.0
            collagen += max(0.0, cfg.collagen_osteo_rate * m_col * aa_gate) * n_ob

        if cfg.tau_col_hours <= 0.0:
            collagen_mat = collagen
        else:
            collagen_mat += (collagen - collagen_mat) / cfg.tau_col_hours
            collagen_mat = min(collagen_mat, collagen)

        ob_age_n_sum = ob_age_n_sum + n_ob
        if n_ob > 1e-12 and death_ob > 0.0:
            keep_frac = max(0.0, 1.0 - death_ob / n_ob)
            ob_age_n_sum *= keep_frac

        if n_ob > 0.0:
            base_rate = max(0.0, cfg.ha_osteo_rate * m_ha)
            if base_rate > 0.0:
                gate_col = collagen_mat / (max(1e-9, cfg.ha_collagen_half_sat) + collagen_mat)
                tau_n = max(1e-9, cfg.ha_tau_hours) ** hill_n
                t_eff_n = max(0.0, ob_age_n_sum / max(1e-12, n_ob))
                hill_gate = t_eff_n / (tau_n + t_eff_n)
                hap += base_rate * gate_col * hill_gate * n_ob

        n_pre = n_pre_next
        n_ob = n_ob_next
        append_row(hour)

    return pd.DataFrame(rows)


def _run_variant(cfg: NonSpatialParams, variant: str, grid_cap: float) -> dict[str, float]:
    c = cfg
    if variant == "full_model":
        pass
    elif variant == "no_osteogenic_commitment":
        c = replace(c, k_commit_hourly=0.0)
    elif variant == "no_commit_delay":
        c = replace(c, commit_delay_hours=0)
    elif variant == "no_apoptosis":
        c = replace(c, apoptosis_preob=0.0, apoptosis_ob=0.0)
    elif variant == "no_drug_modulation":
        unity = DrugResponseParams(dex_ref=1.0, dex_a=0.0, dex_b=0.0, mk4_ref=1.0, mk4_a=0.0, mk4_b=0.0)
        c = replace(c, alp_resp=unity, collagen_resp=unity, ha_resp=unity)
    elif variant == "no_collagen_maturation":
        c = replace(c, tau_col_hours=0.0)
    else:
        raise ValueError(f"Unknown variant: {variant}")
    ts = _simulate_baseline_capacity(c, grid_cap=grid_cap)
    ep = extract_endpoints(ts)
    return {
        "osteo_induction_hours": int(c.osteo_induction_hours),
        "commit_delay_hours": int(c.commit_delay_hours),
        "ha_tau_hours": float(c.ha_tau_hours),
        "alp_day10": float(ep["alp_day10"]),
        "collagen_day14": float(ep["collagen_day14"]),
        "hap_day28": float(ep["hap_day28"]),
        "cell_count_day28": min(float(ep["n_preob_day28"]) + float(ep["n_ob_day28"]), float(grid_cap)),
        "cell_count_day28_raw": float(ep["n_preob_day28"]) + float(ep["n_ob_day28"]),
    }


def _build_comparison_table(spatial_vals: dict[str, float]) -> pd.DataFrame:
    base = _make_base_nonspatial_cfg(spatial_vals)
    grid_cap = max(1.0, float(spatial_vals.get("grid_sites", 1.0)))
    nonsp_rows = []
    for variant, label in VARIANT_ORDER:
        ep = _run_variant(base, variant, grid_cap=grid_cap)
        nonsp_rows.append(
            {
                "variant": variant,
                "label": label,
                "osteo_induction_hours": float(ep["osteo_induction_hours"]),
                "commit_delay_hours": float(ep["commit_delay_hours"]),
                "ha_tau_hours": float(ep["ha_tau_hours"]),
                "alp_day10": float(ep["alp_day10"]),
                "collagen_day14": float(ep["collagen_day14"]),
                "hap_day28": float(ep["hap_day28"]),
                "cell_count_day28": float(ep["cell_count_day28"]),
                "cell_count_day28_raw": float(ep["cell_count_day28_raw"]),
            }
        )

    spatial_row = {"variant": SPATIAL_VARIANT, "label": SPATIAL_LABEL}
    for c in HEATMAP_COLS:
        spatial_row[c] = float(spatial_vals.get(c, float("nan")))
    spatial_row["osteo_induction_hours"] = float(spatial_vals.get("osteo_induction_hours", np.nan))
    spatial_row["commit_delay_hours"] = float(spatial_vals.get("commit_delay_hours", np.nan))
    spatial_row["ha_tau_hours"] = float(spatial_vals.get("ha_tau_hours", np.nan))
    cmp = pd.concat([pd.DataFrame([spatial_row]), pd.DataFrame(nonsp_rows)], ignore_index=True)

    ref_by_metric = {c: SPATIAL_VARIANT for c in SPATIAL_REF_COLS}

    # Record per-metric reference and fold-changes on a compact, symmetric scale.
    stats_rows: list[dict[str, float | str]] = []
    for c in HEATMAP_COLS:
        ref_variant = ref_by_metric[c]
        ref_row = cmp[cmp["variant"] == ref_variant]
        if ref_row.empty:
            raise ValueError(f"Reference variant {ref_variant} missing for metric {c}")
        ref_val = float(ref_row[c].iloc[0])
        for i in cmp.index:
            val = float(cmp.at[i, c]) if pd.notna(cmp.at[i, c]) else float("nan")
            if not np.isfinite(ref_val) or abs(ref_val) < 1e-12 or not np.isfinite(val):
                ratio = float("nan")
                log2_fc = float("nan")
            else:
                ratio = val / ref_val
                log2_fc = float(np.log2(max(ratio, 1e-12)))
            cmp.at[i, f"{c}_ratio"] = ratio
            cmp.at[i, f"{c}_log2fc"] = log2_fc
            stats_rows.append(
                {
                    "variant": str(cmp.at[i, "variant"]),
                    "label": str(cmp.at[i, "label"]),
                    "metric": c,
                    "reference_variant": ref_variant,
                    "value": val,
                    "reference_value": ref_val,
                    "ratio": ratio,
                    "log2_fold_change": log2_fc,
                }
            )
    cmp.attrs["heatmap_stats"] = pd.DataFrame(stats_rows)
    return cmp


def _compute_significance(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty or "variant" not in df_raw.columns:
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    test_metrics = [c for c in HEATMAP_COLS if c in df_raw.columns]
    full = df_raw[df_raw["variant"] == "full_model"]
    if full.empty:
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    rows: list[dict[str, float | str | bool]] = []
    pvals: list[float] = []
    keys: list[tuple[str, str]] = []
    allowed_variants = {v for v, _ in VARIANT_ORDER}
    for variant in df_raw["variant"].drop_duplicates().tolist():
        if variant not in allowed_variants:
            continue
        if variant == "full_model":
            continue
        a = df_raw[df_raw["variant"] == variant]
        b = full
        for metric in test_metrics:
            x = a[metric].to_numpy(dtype=float)
            y = b[metric].to_numpy(dtype=float)
            if x.size < 2 or y.size < 2:
                continue
            mw = mannwhitneyu(x, y, alternative="two-sided")
            n1 = max(1, x.size)
            n2 = max(1, y.size)
            rbc = float((2.0 * mw.statistic) / (n1 * n2) - 1.0)
            p = float(mw.pvalue)
            rows.append({"variant": variant, "metric": metric, "p": p, "rbc": rbc})
            pvals.append(p)
            keys.append((variant, metric))

    if not rows:
        return pd.DataFrame(columns=["variant", "metric", "p", "q", "rbc", "significant", "stars", "metric_label"])

    qvals = _bh_adjust(pvals)
    metric_label_map = dict(zip(HEATMAP_COLS, HEATMAP_LABELS))
    out_rows = []
    for i, r in enumerate(rows):
        q = float(qvals[i])
        out_rows.append(
            {
                "variant": r["variant"],
                "metric": r["metric"],
                "p": float(r["p"]),
                "q": q,
                "rbc": float(r["rbc"]),
                "significant": bool(q < 0.05),
                "stars": _significance_stars(q),
                "metric_label": metric_label_map.get(str(r["metric"]), str(r["metric"])),
            }
        )
    return pd.DataFrame(out_rows)


def plot_heatmap(df_cmp: pd.DataFrame, out: Path, sig_df: pd.DataFrame | None = None) -> None:
    def fold_label(ratio: float) -> str:
        if not np.isfinite(ratio):
            return "N/A"
        if abs(ratio - 1.0) < 0.005:
            return "1.00x"
        return f"{ratio:.2f}x"

    cmap = LinearSegmentedColormap.from_list("plos_cb_clean", ["#C44E52", "#F7F7F7", "#4C72B0"], N=256)
    row_order = df_cmp["variant"].tolist()
    row_labels = df_cmp["label"].tolist()
    mat = np.full((len(row_order), len(HEATMAP_COLS)), np.nan, dtype=float)
    annot = [["" for _ in HEATMAP_COLS] for _ in row_order]

    for i, var in enumerate(row_order):
        for j, c in enumerate(HEATMAP_COLS):
            log2fc = df_cmp.loc[df_cmp["variant"] == var, f"{c}_log2fc"]
            ratio = df_cmp.loc[df_cmp["variant"] == var, f"{c}_ratio"]
            if log2fc.empty or ratio.empty:
                continue
            mat[i, j] = float(log2fc.iloc[0]) if pd.notna(log2fc.iloc[0]) else float("nan")
            annot[i][j] = fold_label(float(ratio.iloc[0]))

    star_map: dict[tuple[str, str], str] = {}
    if isinstance(sig_df, pd.DataFrame) and not sig_df.empty:
        for _, r in sig_df.iterrows():
            v = str(r.get("variant", ""))
            m = str(r.get("metric", ""))
            s = str(r.get("stars", "")) if pd.notna(r.get("stars", "")) else ""
            if v and m and s:
                star_map[(v, m)] = s

    finite_abs = np.abs(mat[np.isfinite(mat)])
    vmax = max(1.0, float(np.percentile(finite_abs, 92))) if finite_abs.size else 1.0
    vmax = min(vmax, 6.0)
    cmap = cmap.copy()
    cmap.set_bad("#EEF2F7")
    plt.rcParams.update({"font.size": 9.5, "xtick.labelsize": 9, "ytick.labelsize": 9.5})
    fig, ax = plt.subplots(figsize=(10.8, 6.1))
    fig.subplots_adjust(left=0.18, right=0.965, top=0.84, bottom=0.08)
    ax.set_facecolor("#F8FAFC")

    im = ax.imshow(np.ma.masked_invalid(mat), cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

    tick_labels = HEATMAP_LABELS
    ax.set_xticks(np.arange(len(tick_labels)), tick_labels)
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    ax.tick_params(axis="y", pad=2)
    ax.xaxis.tick_top()
    ax.tick_params(top=True, bottom=False, labelsize=9)

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="#FFFFFF", linestyle="-", linewidth=1.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            v = mat[i, j]
            txt = annot[i][j]
            if not txt:
                continue
            sig_txt = star_map.get((row_order[i], HEATMAP_COLS[j]), "")
            if sig_txt:
                txt = f"{txt}\n{sig_txt}"
            color = "white" if np.isfinite(v) and abs(v) > 0.45 * vmax else "black"
            ax.text(j, i, txt, ha="center", va="center", color=color, fontsize=8.3, fontweight="semibold")

    ax.axvline(2.5, color="#B8C1CC", lw=1.0)
    cbar = fig.colorbar(im, ax=ax, fraction=0.040, pad=0.015)
    cbar.set_label("log2 fold-change vs spatial reference", fontsize=9)
    cbar.ax.tick_params(labelsize=8.5)

    fig.suptitle("Ablation Comparison vs Spatial Reference", y=0.995, fontsize=13.5, fontweight="semibold")

    fig.savefig(out / "ablation_heatmap_relative_change.png", dpi=300)
    fig.savefig(out / "ablation_heatmap_relative_change.pdf")
    plt.close(fig)
    stats_df = df_cmp.attrs.get("heatmap_stats", pd.DataFrame())
    if isinstance(stats_df, pd.DataFrame):
        stats_df.to_csv(out / "ablation_heatmap_stats.csv", index=False)


def plot_dose_robustness(df_dose: pd.DataFrame, out: Path, band_type: str, seed: int) -> pd.DataFrame:
    cond_order = [f"C{i}" for i in range(1, 9)]
    x = np.arange(len(cond_order))
    variants = ["full_model", "no_drug_modulation", "no_osteogenic_commitment"]
    rng = np.random.default_rng(seed)

    plt.rcParams.update({"font.size": 10, "axes.labelsize": 11, "legend.fontsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9})
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), constrained_layout=True)
    ax_top, ax_bot, ax_dyn, ax_peak = axes.ravel()

    rows = []
    raw_arrays = {}
    fold_arrays = {}

    for v in variants:
        d = df_dose[df_dose["variant"] == v]
        arr_fold = d.pivot(index="run_id", columns="condition", values="hap_fold_vs_c1").loc[:, cond_order].to_numpy(dtype=float)
        arr_raw = d.pivot(index="run_id", columns="condition", values="hap_raw").loc[:, cond_order].to_numpy(dtype=float)
        raw_arrays[v] = arr_raw
        fold_arrays[v] = arr_fold

        med = np.median(arr_fold, axis=0)
        if band_type == "confidence":
            lo, hi = _bootstrap_ci_for_median(arr_fold, rng)
        else:
            lo = np.percentile(arr_fold, 2.5, axis=0)
            hi = np.percentile(arr_fold, 97.5, axis=0)

        ax_top.fill_between(x, lo, hi, color=LINE_COLORS[v], alpha=0.20, linewidth=0.0)
        ax_top.plot(x, med, lw=2.5, marker="o", markersize=5, color=LINE_COLORS[v], label=VARIANT_LABELS[v])

        for i, c in enumerate(cond_order):
            rows.append({"variant": v, "condition": c, "median": float(med[i]), "lower": float(lo[i]), "upper": float(hi[i]), "panel": "fold_vs_c1", "band_type": band_type})

    full_raw = raw_arrays["full_model"]
    ratio_arrays = {}
    for v in ["no_drug_modulation", "no_osteogenic_commitment"]:
        ratio = raw_arrays[v] / np.maximum(1e-12, full_raw)
        ratio_arrays[v] = ratio
        med = np.median(ratio, axis=0)
        if band_type == "confidence":
            lo, hi = _bootstrap_ci_for_median(ratio, rng)
        else:
            lo = np.percentile(ratio, 2.5, axis=0)
            hi = np.percentile(ratio, 97.5, axis=0)

        ax_bot.fill_between(x, lo, hi, color=LINE_COLORS[v], alpha=0.20, linewidth=0.0)
        ax_bot.plot(x, med, lw=2.5, marker="o", markersize=5, color=LINE_COLORS[v], label=f"{VARIANT_LABELS[v]} / Full")

        for i, c in enumerate(cond_order):
            rows.append({"variant": v, "condition": c, "median": float(med[i]), "lower": float(lo[i]), "upper": float(hi[i]), "panel": "ratio_to_full", "band_type": band_type})

    ylabel = "HAp d28 fold vs C1 (median ± 95% CI)" if band_type == "confidence" else "HAp d28 fold vs C1 (median ± 95% band)"
    ax_top.set_ylabel(ylabel)
    ax_top.set_title("Dose-response curves")
    ax_top.legend(loc="upper right", frameon=False)
    ax_top.grid(False)

    ax_bot.set_xlabel("Condition")
    ax_bot.set_ylabel("Ablation / Full (median ± 95%)")
    ax_bot.axhline(1.0, color="#6B7280", lw=1.2)
    ax_bot.legend(loc="upper right", frameon=False)
    ax_bot.grid(False)
    ax_bot.set_xticks(x, cond_order)

    # Panel 3: Dynamic range distribution across runs
    dyn_vals = []
    dyn_labels = []
    for v in variants:
        arr = fold_arrays[v]
        dyn = np.max(arr, axis=1) - np.min(arr, axis=1)
        dyn_vals.append(dyn)
        dyn_labels.append(VARIANT_LABELS[v])
    vp = ax_dyn.violinplot(dyn_vals, showmeans=False, showmedians=True, widths=0.8)
    for i, b in enumerate(vp["bodies"]):
        b.set_facecolor(list(LINE_COLORS.values())[i])
        b.set_alpha(0.28)
        b.set_edgecolor(list(LINE_COLORS.values())[i])
    vp["cmedians"].set_color("#111827")
    vp["cmedians"].set_linewidth(1.5)
    ax_dyn.set_xticks(np.arange(1, len(dyn_labels) + 1), dyn_labels, rotation=12, ha="right")
    ax_dyn.set_ylabel("Dose dynamic range")
    ax_dyn.set_title("Across-run dynamic range")
    ax_dyn.grid(False)

    # Panel 4: Peak-condition probability heatmap
    peak_prob = np.zeros((len(variants), len(cond_order)), dtype=float)
    for i, v in enumerate(variants):
        arr = fold_arrays[v]
        idx = np.argmax(arr, axis=1)
        for j in range(len(cond_order)):
            peak_prob[i, j] = np.mean(idx == j)
    im = ax_peak.imshow(peak_prob, vmin=0.0, vmax=1.0, cmap="Blues", aspect="auto")
    ax_peak.set_xticks(np.arange(len(cond_order)), cond_order)
    ax_peak.set_yticks(np.arange(len(variants)), [VARIANT_LABELS[v] for v in variants])
    ax_peak.set_title("Peak-condition probability")
    for i in range(len(variants)):
        for j in range(len(cond_order)):
            txt = f"{peak_prob[i, j]*100:.0f}%"
            color = "white" if peak_prob[i, j] > 0.55 else "black"
            ax_peak.text(j, i, txt, ha="center", va="center", fontsize=8, color=color)
    cbar = fig.colorbar(im, ax=ax_peak, fraction=0.046, pad=0.04)
    cbar.set_label("Run fraction")

    # Panel labels
    for ax, letter in zip([ax_top, ax_bot, ax_dyn, ax_peak], ["A", "B", "C", "D"]):
        ax.text(-0.10, 1.06, letter, transform=ax.transAxes, fontsize=12, fontweight="bold", va="top")

    fig.savefig(out / "dose_response_robustness.png", dpi=300)
    fig.savefig(out / "dose_response_robustness.pdf")
    plt.close(fig)

    bands = pd.DataFrame(rows)
    bands.to_csv(out / "dose_response_robustness_bands.csv", index=False)
    return bands


def plot_conclusion_matrix(df_cmp: pd.DataFrame, out: Path) -> None:
    def status(ratio: float) -> float:
        if not np.isfinite(ratio):
            return float("nan")
        if 0.80 <= ratio <= 1.20:
            return 2.0
        if 0.50 <= ratio <= 1.50:
            return 1.0
        return 0.0

    rows = []
    for _, r in df_cmp.iterrows():
        rows.append(
            {
                "label": r["label"],
                "ALP maintained": status(float(r["alp_day10_ratio"])),
                "Collagen maintained": status(float(r["collagen_day14_ratio"])),
                "HAp maintained": status(float(r["hap_day28_ratio"])),
                "Cell count maintained": status(float(r["cell_count_day28_ratio"])),
            }
        )
    mat_df = pd.DataFrame(rows)
    cols = ["ALP maintained", "Collagen maintained", "HAp maintained", "Cell count maintained"]
    mat = mat_df[cols].to_numpy(dtype=float)

    cmap = ListedColormap(["#D55E00", "#F0E442", "#009E73", "#D1D5DB"])
    mat_show = np.where(np.isnan(mat), 3.0, mat)
    fig, ax = plt.subplots(figsize=(9.8, 5.8), constrained_layout=True)
    ax.imshow(mat_show, cmap=cmap, vmin=-0.5, vmax=3.5, aspect="equal")
    ax.set_xticks(np.arange(len(cols)), cols, rotation=20, ha="right")
    ax.set_yticks(np.arange(len(mat_df)), mat_df["label"])
    ax.set_title("Conclusion matrix")

    txt_map = {0: "No", 1: "Partial", 2: "Yes", 3: "N/A"}
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            key = 3 if np.isnan(mat[i, j]) else int(mat[i, j])
            ax.text(j, i, txt_map[key], ha="center", va="center", fontsize=9)

    ax.set_xticks(np.arange(-0.5, mat.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, mat.shape[0], 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)

    fig.savefig(out / "conclusion_matrix.png", dpi=300)
    fig.savefig(out / "conclusion_matrix.pdf")
    mat_df.to_csv(out / "conclusion_matrix_table.csv", index=False)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out = args.out.resolve()
    spatial_out = args.spatial_out.resolve()

    dose_csv = out / "ablation_dose_runs.csv"
    if not dose_csv.exists():
        raise SystemExit("Missing ablation_dose_runs.csv. Run ablation_study_run.py first.")
    raw_csv = out / "ablation_raw_metrics_runs.csv"

    df_dose = pd.read_csv(dose_csv)
    df_raw = pd.read_csv(raw_csv) if raw_csv.exists() else pd.DataFrame()
    spatial_vals = _load_spatial_reference(spatial_out)
    df_cmp = _build_comparison_table(spatial_vals)
    df_cmp.to_csv(out / "ablation_comparison_table.csv", index=False)
    sig_df = _compute_significance(df_raw)
    sig_df.to_csv(out / "ablation_significance_table.csv", index=False)

    plot_heatmap(df_cmp, out, sig_df=sig_df)
    plot_dose_robustness(df_dose, out, band_type=args.band_type, seed=int(args.seed))
    plot_conclusion_matrix(df_cmp, out)

    print(f"Saved: {out / 'ablation_heatmap_relative_change.png'}")
    print(f"Saved: {out / 'dose_response_robustness.png'}")
    print(f"Saved: {out / 'conclusion_matrix.png'}")
    print(f"Saved: {out / 'dose_response_robustness_bands.csv'}")
    print(f"Saved: {out / 'ablation_heatmap_stats.csv'}")
    print(f"Saved: {out / 'ablation_comparison_table.csv'}")
    print(f"Saved: {out / 'ablation_significance_table.csv'}")


if __name__ == "__main__":
    main()
