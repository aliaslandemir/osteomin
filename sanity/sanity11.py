#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Dict, List, Tuple

os.environ.setdefault("MPLCONFIGDIR", str((Path("/tmp") / "osteomin_mplconfig").resolve()))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

DEFAULT_SIM_ROOT = SCRIPT_DIR / "out" / "sanity_check5"
DEFAULT_EXP_METRICS = (
    PROJECT_ROOT
    / "endpointaad"
    / "ars_analysis"
    / "ars_output_spatial_thresh"
    / "ars_spatial_thresh_metrics.csv"
)
DEFAULT_OUT = SCRIPT_DIR / "out" / "sanity11"

COND_ORDER = [f"C{i}" for i in range(1, 9)]
COND_LABEL = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo + D_LK_L",
    "C4": "Osteo + D_LK_H",
    "C5": "Osteo + D_HK_L",
    "C6": "Osteo + D_HK_H",
    "C7": "Osteo + D_HK_0",
    "C8": "Osteo + D_0K_H",
}

BAR_SIM = "#4C78A8"
BAR_EXP = "#E45756"
POINT_SIM = "#2B4D6E"
POINT_EXP = "#8D2F34"
INK = "#1A1A1A"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sanity 11: compare day-28 simulation spatial heterogeneity "
            "against experimental ARS spatial-threshold outputs."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--sim-root", type=Path, default=DEFAULT_SIM_ROOT, help="Root containing c1..c8/r*/grids outputs")
    parser.add_argument("--exp-metrics", type=Path, default=DEFAULT_EXP_METRICS, help="Experimental ARS metrics CSV")
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT, help="Output directory")
    parser.add_argument("--day", type=int, default=28, help="Simulation day to load (ha_day_XXX.npy)")
    parser.add_argument("--window", type=int, default=8, help="Local-variance window in grid pixels")
    parser.add_argument("--ha-thresh", type=float, default=0.0, help="Threshold for binary mineral mask from HAp grid")
    parser.add_argument("--eps", type=float, default=1e-8, help="Small positive value for robust division")
    parser.add_argument("--alpha", type=float, default=0.05, help="BH threshold for pairwise tests vs C2")
    parser.add_argument("--ref-cond", type=str, default="C2", choices=COND_ORDER, help="Reference condition for relative-effect comparison")
    return parser.parse_args()


def compute_cv(signal: np.ndarray, eps: float) -> float:
    mu = float(np.mean(signal))
    sigma = float(np.std(signal, ddof=0))
    return float(sigma / max(mu, eps))


def compute_h_spatial(signal: np.ndarray, window: int) -> float:
    local_mean = uniform_filter(signal, size=window, mode="reflect")
    local_mean_sq = uniform_filter(signal * signal, size=window, mode="reflect")
    local_var = np.clip(local_mean_sq - local_mean * local_mean, 0.0, None)
    return float(np.mean(local_var))


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def pairwise_vs_ref_bh(
    cond_to_vals: Dict[str, np.ndarray],
    ref: str,
    targets: List[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    ref_vals = np.asarray(cond_to_vals.get(ref, np.array([])), dtype=float)
    ref_vals = ref_vals[np.isfinite(ref_vals)]

    out: Dict[str, Dict[str, float]] = {}
    names: List[str] = []
    p_raws: List[float] = []

    for c in targets:
        vals = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]
        if ref_vals.size == 0 or vals.size == 0:
            out[c] = {"p_raw": np.nan, "p_adj": np.nan, "sig": False}
            continue
        _, p_raw = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
        out[c] = {"p_raw": float(p_raw)}
        names.append(c)
        p_raws.append(float(p_raw))

    for c, p_adj in zip(names, benjamini_hochberg(p_raws)):
        out[c]["p_adj"] = float(p_adj)
        out[c]["sig"] = bool(p_adj < alpha)
    return out


def _mean_sd(vals: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean, np.nan
    sd = float(np.std(vals, ddof=1))
    return mean, sd


def load_simulation_metrics(sim_root: Path, day: int, window: int, ha_thresh: float, eps: float) -> Tuple[pd.DataFrame, list[str]]:
    day_tag = f"{day:03d}"
    rows = []
    missing = []
    for cond in COND_ORDER:
        cond_dir = sim_root / cond.lower()
        rep_dirs = sorted([p for p in cond_dir.glob("r*") if p.is_dir()])
        if not rep_dirs:
            missing.append(f"{cond}: no replicate directories under {cond_dir}")
            continue

        found = 0
        for rep_dir in rep_dirs:
            p = rep_dir / "grids" / f"ha_day_{day_tag}.npy"
            if not p.exists():
                continue
            ha = np.load(p).astype(np.float32)
            signal = (ha > ha_thresh).astype(np.float32)
            rows.append(
                {
                    "cond": cond,
                    "replicate": rep_dir.name,
                    "grid_path": str(p),
                    "ha_threshold": float(ha_thresh),
                    "window_px": int(window),
                    "cv_intensity": compute_cv(signal, eps=eps),
                    "h_spatial": compute_h_spatial(signal, window=window),
                    "fraction_positive": float(np.mean(signal)),
                }
            )
            found += 1
        if found == 0:
            missing.append(f"{cond}: no files matched */grids/ha_day_{day_tag}.npy")
    return pd.DataFrame(rows), missing


def summarize_by_condition(df: pd.DataFrame, n_col: str) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(
            columns=["cond", n_col, "cv_mean", "cv_sd", "h_spatial_mean", "h_spatial_sd", "fraction_positive_mean", "fraction_positive_sd"]
        )
    out = (
        df.groupby("cond", as_index=False)
        .agg(
            **{
                n_col: ("cond", "count"),
                "cv_mean": ("cv_intensity", "mean"),
                "cv_sd": ("cv_intensity", "std"),
                "h_spatial_mean": ("h_spatial", "mean"),
                "h_spatial_sd": ("h_spatial", "std"),
                "fraction_positive_mean": ("fraction_positive", "mean"),
                "fraction_positive_sd": ("fraction_positive", "std"),
            }
        )
        .sort_values("cond")
    )
    return out


def load_experimental_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    need = {"cond", "cv_intensity", "h_spatial"}
    missing = sorted(need - set(df.columns))
    if missing:
        raise SystemExit(f"experimental metrics file is missing required columns: {', '.join(missing)}")
    out = df.loc[df["cond"].isin(COND_ORDER), ["cond", "cv_intensity", "h_spatial"]].copy()
    out["source"] = "experiment"
    return out


def write_stats_txt(
    out_path: Path,
    metric_name: str,
    cond_to_vals: Dict[str, np.ndarray],
    alpha: float,
) -> None:
    conds = [c for c in COND_ORDER if c in cond_to_vals and np.isfinite(cond_to_vals[c]).any()]
    groups = []
    for c in conds:
        vals = np.asarray(cond_to_vals[c], dtype=float)
        vals = vals[np.isfinite(vals)]
        if vals.size > 0:
            groups.append(vals)
    if len(groups) >= 2:
        try:
            h_kw, p_kw = stats.kruskal(*groups)
        except ValueError:
            h_kw, p_kw = 0.0, 1.0
    else:
        h_kw, p_kw = np.nan, np.nan

    fam = pairwise_vs_ref_bh(cond_to_vals, ref="C2", targets=[c for c in COND_ORDER if c not in ("C2",)], alpha=alpha)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{metric_name}\n")
        f.write("Test: Mann-Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini-Hochberg across C2-vs-target comparisons\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Kruskal-Wallis across all conditions: H={h_kw:.4g}, p={p_kw:.4g}\n\n")
        f.write("Pairwise comparisons vs C2\n")
        for c in [x for x in COND_ORDER if x != "C2"]:
            rec = fam.get(c, {})
            f.write(
                f"  {c} vs C2: p_raw={rec.get('p_raw', np.nan):.4g}, "
                f"p_adj={rec.get('p_adj', np.nan):.4g}\n"
            )


def plot_metric_compare(
    sim_df: pd.DataFrame,
    exp_df: pd.DataFrame,
    value_col: str,
    ylabel: str,
    title: str,
    out_png: Path,
    out_pdf: Path,
) -> None:
    sim_map = {c: sim_df.loc[sim_df["cond"] == c, value_col].dropna().to_numpy(dtype=float) for c in COND_ORDER}
    exp_map = {c: exp_df.loc[exp_df["cond"] == c, value_col].dropna().to_numpy(dtype=float) for c in COND_ORDER}

    fig, ax = plt.subplots(figsize=(9.2, 5.6))
    x = np.arange(len(COND_ORDER), dtype=float)
    w = 0.34
    rng = np.random.default_rng(42)

    for i, c in enumerate(COND_ORDER):
        s = sim_map[c]
        e = exp_map[c]
        sm, ss = _mean_sd(s)
        em, es = _mean_sd(e)
        if np.isfinite(sm):
            ax.bar(x[i] - w / 2, sm, width=w, color=BAR_SIM, alpha=0.60, edgecolor=INK, linewidth=1.0, zorder=2)
            if np.isfinite(ss):
                ax.errorbar(x[i] - w / 2, sm, yerr=ss, color=POINT_SIM, capsize=3, lw=1.2, zorder=4)
            xs = x[i] - w / 2 + rng.normal(0.0, 0.022, size=s.size)
            ax.scatter(xs, s, s=24, color=POINT_SIM, edgecolor="white", linewidth=0.4, alpha=0.95, zorder=5)
        if np.isfinite(em):
            ax.bar(x[i] + w / 2, em, width=w, color=BAR_EXP, alpha=0.45, edgecolor=INK, linewidth=1.0, zorder=1)
            if np.isfinite(es):
                ax.errorbar(x[i] + w / 2, em, yerr=es, color=POINT_EXP, capsize=3, lw=1.2, zorder=4)
            xe = x[i] + w / 2 + rng.normal(0.0, 0.022, size=e.size)
            ax.scatter(xe, e, s=24, color=POINT_EXP, edgecolor="white", linewidth=0.4, alpha=0.95, zorder=5)

    ax.set_xticks(x)
    ax.set_xticklabels(COND_ORDER)
    ax.set_xlabel("Condition")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="y", color="#d0d7de", linestyle="-", linewidth=0.7, alpha=0.4)
    ax.legend(
        [
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BAR_SIM, markeredgecolor=INK, markersize=10, alpha=0.65),
            plt.Line2D([0], [0], marker="s", color="none", markerfacecolor=BAR_EXP, markeredgecolor=INK, markersize=10, alpha=0.55),
        ],
        ["Simulation (day 28)", "Experimental (day 28)"],
        frameon=False,
        loc="best",
    )

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def relative_effect(values: pd.Series, ref_value: float) -> pd.Series:
    vals = values.astype(float)
    denom = vals.abs() + abs(float(ref_value))
    out = pd.Series(np.zeros(len(vals), dtype=float), index=values.index)
    valid = denom > 0.0
    out.loc[valid] = (vals.loc[valid] - float(ref_value)) / denom.loc[valid]
    return out


def build_relative_effect_table(compare: pd.DataFrame, ref_cond: str) -> pd.DataFrame:
    df = compare.copy()
    for metric in ("h_spatial", "cv"):
        sim_col = f"sim_{metric}_mean"
        exp_col = f"exp_{metric}_mean"
        sim_ref = df.loc[df["cond"] == ref_cond, sim_col]
        exp_ref = df.loc[df["cond"] == ref_cond, exp_col]
        sim_ref_val = float(sim_ref.iloc[0]) if not sim_ref.empty and np.isfinite(sim_ref.iloc[0]) else np.nan
        exp_ref_val = float(exp_ref.iloc[0]) if not exp_ref.empty and np.isfinite(exp_ref.iloc[0]) else np.nan
        if np.isfinite(sim_ref_val):
            df[f"sim_rel_{metric}"] = relative_effect(df[sim_col], sim_ref_val)
        else:
            df[f"sim_rel_{metric}"] = np.nan
        if np.isfinite(exp_ref_val):
            df[f"exp_rel_{metric}"] = relative_effect(df[exp_col], exp_ref_val)
        else:
            df[f"exp_rel_{metric}"] = np.nan
    return df


def agreement_metrics(rel_df: pd.DataFrame, metric: str, ref_cond: str) -> dict[str, float]:
    s_col = f"sim_rel_{metric}"
    e_col = f"exp_rel_{metric}"
    sub = rel_df.loc[rel_df["cond"] != ref_cond, ["cond", s_col, e_col]].dropna()
    if sub.empty:
        return {"n": 0, "kendall_tau": np.nan, "sign_agreement": np.nan, "mae": np.nan}
    sim = sub[s_col].to_numpy(dtype=float)
    exp = sub[e_col].to_numpy(dtype=float)
    tau = float(stats.kendalltau(sim, exp).statistic) if len(sub) > 1 else np.nan
    sign_agreement = float(np.mean(np.sign(sim) == np.sign(exp)))
    mae = float(np.mean(np.abs(sim - exp)))
    return {"n": int(len(sub)), "kendall_tau": tau, "sign_agreement": sign_agreement, "mae": mae}


def plot_relative_effect_scatter(rel_df: pd.DataFrame, ref_cond: str, out_png: Path, out_pdf: Path) -> None:
    panels = [
        ("h_spatial", "Spatial Heterogeneity", "A"),
        ("cv", "Mask Heterogeneity (CV)", "B"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 5.2), sharex=True, sharey=True)
    if len(panels) == 1:
        axes = [axes]

    for ax, (metric, title, panel_letter) in zip(axes, panels):
        s_col = f"sim_rel_{metric}"
        e_col = f"exp_rel_{metric}"
        sub = rel_df[["cond", s_col, e_col]].dropna()
        vals_min = -1.0
        vals_max = 1.0
        ax.plot([vals_min, vals_max], [vals_min, vals_max], color="#c7c7c7", lw=0.9, zorder=1)
        ax.axhline(0.0, color="#c7c7c7", lw=0.9, zorder=1)
        ax.axvline(0.0, color="#c7c7c7", lw=0.9, zorder=1)

        for _, row in sub.iterrows():
            cond = row["cond"]
            x = float(row[e_col])
            y = float(row[s_col])
            if cond == "C2":
                ax.scatter(x, y, s=48, marker="s", color="#222222", edgecolor="white", linewidth=0.5, zorder=4)
            else:
                ax.scatter(x, y, s=52, marker="^", color="#8D2F34", edgecolor="white", linewidth=0.5, zorder=4)
            ax.text(x + 0.018, y + 0.018, cond, fontsize=9.2, color=INK, zorder=5)

        m = agreement_metrics(rel_df, metric, ref_cond)
        ax.text(
            0.98,
            0.06,
            f"\u03c4 = {m['kendall_tau']:.2f}\nMAE = {m['mae']:.2f}\nsign agreement = {100.0 * m['sign_agreement']:.0f}%",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.0,
            color=INK,
        )
        ax.set_title(f"{panel_letter}   {title}", loc="left", fontsize=12.5, fontweight="semibold")
        ax.set_xlim(vals_min, vals_max)
        ax.set_ylim(vals_min, vals_max)
        ax.set_xticks(np.linspace(-1.0, 1.0, 5))
        ax.set_yticks(np.linspace(-1.0, 1.0, 5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    axes[0].set_ylabel(f"Simulated relative effect (vs {ref_cond})")
    for ax in axes:
        ax.set_xlabel(f"Experimental relative effect (vs {ref_cond})")

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=320, bbox_inches="tight")
    fig.savefig(out_pdf, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if args.window < 2:
        raise SystemExit("--window must be >= 2")
    if args.eps <= 0:
        raise SystemExit("--eps must be > 0")

    args.out.mkdir(parents=True, exist_ok=True)

    sim_metrics, sim_missing = load_simulation_metrics(
        sim_root=args.sim_root,
        day=args.day,
        window=args.window,
        ha_thresh=args.ha_thresh,
        eps=args.eps,
    )
    exp_metrics = load_experimental_metrics(args.exp_metrics)

    sim_metrics_path = args.out / "sanity11_sim_metrics.csv"
    sim_metrics.to_csv(sim_metrics_path, index=False)

    exp_metrics_path = args.out / "sanity11_exp_metrics_subset.csv"
    exp_metrics.to_csv(exp_metrics_path, index=False)

    sim_summary = summarize_by_condition(sim_metrics, n_col="n_sim")
    exp_summary = (
        exp_metrics.groupby("cond", as_index=False)
        .agg(n_exp=("cond", "count"), cv_mean=("cv_intensity", "mean"), cv_sd=("cv_intensity", "std"), h_spatial_mean=("h_spatial", "mean"), h_spatial_sd=("h_spatial", "std"))
        .sort_values("cond")
    )
    sim_summary_path = args.out / "sanity11_sim_summary_by_condition.csv"
    exp_summary_path = args.out / "sanity11_exp_summary_by_condition.csv"
    sim_summary.to_csv(sim_summary_path, index=False)
    exp_summary.to_csv(exp_summary_path, index=False)

    compare = pd.merge(
        sim_summary.rename(
            columns={
                "cv_mean": "sim_cv_mean",
                "cv_sd": "sim_cv_sd",
                "h_spatial_mean": "sim_h_spatial_mean",
                "h_spatial_sd": "sim_h_spatial_sd",
            }
        ),
        exp_summary.rename(
            columns={
                "cv_mean": "exp_cv_mean",
                "cv_sd": "exp_cv_sd",
                "h_spatial_mean": "exp_h_spatial_mean",
                "h_spatial_sd": "exp_h_spatial_sd",
            }
        ),
        on="cond",
        how="outer",
    ).sort_values("cond")
    compare_path = args.out / "sanity11_compare_condition_means.csv"
    compare.to_csv(compare_path, index=False)
    rel_table = build_relative_effect_table(compare, ref_cond=args.ref_cond)
    rel_table_path = args.out / f"sanity11_relative_effects_vs_{args.ref_cond.lower()}.csv"
    rel_table.to_csv(rel_table_path, index=False)

    sim_h_map = {c: sim_metrics.loc[sim_metrics["cond"] == c, "h_spatial"].dropna().to_numpy(dtype=float) for c in COND_ORDER}
    sim_cv_map = {c: sim_metrics.loc[sim_metrics["cond"] == c, "cv_intensity"].dropna().to_numpy(dtype=float) for c in COND_ORDER}
    exp_h_map = {c: exp_metrics.loc[exp_metrics["cond"] == c, "h_spatial"].dropna().to_numpy(dtype=float) for c in COND_ORDER}
    exp_cv_map = {c: exp_metrics.loc[exp_metrics["cond"] == c, "cv_intensity"].dropna().to_numpy(dtype=float) for c in COND_ORDER}

    write_stats_txt(args.out / "sanity11_sim_h_spatial_stats.txt", "Simulation H_spatial (day 28, thresholded HAp mask)", sim_h_map, args.alpha)
    write_stats_txt(args.out / "sanity11_sim_cv_stats.txt", "Simulation CV (day 28, thresholded HAp mask)", sim_cv_map, args.alpha)
    write_stats_txt(args.out / "sanity11_exp_h_spatial_stats.txt", "Experimental H_spatial (ARS threshold mask)", exp_h_map, args.alpha)
    write_stats_txt(args.out / "sanity11_exp_cv_stats.txt", "Experimental CV (ARS threshold mask)", exp_cv_map, args.alpha)

    plot_metric_compare(
        sim_df=sim_metrics,
        exp_df=exp_metrics,
        value_col="h_spatial",
        ylabel=r"Spatial heterogeneity ($H_{spatial}$)",
        title="Sanity 11: Day-28 Spatial Heterogeneity (Simulation vs Experiment)",
        out_png=args.out / "sanity11_h_spatial_compare.png",
        out_pdf=args.out / "sanity11_h_spatial_compare.pdf",
    )
    plot_metric_compare(
        sim_df=sim_metrics,
        exp_df=exp_metrics,
        value_col="cv_intensity",
        ylabel="Mask heterogeneity (CV)",
        title="Sanity 11: Day-28 CV (Simulation vs Experiment)",
        out_png=args.out / "sanity11_cv_compare.png",
        out_pdf=args.out / "sanity11_cv_compare.pdf",
    )
    rel_scatter_png = args.out / f"sanity11_relative_effect_scatter_vs_{args.ref_cond.lower()}.png"
    rel_scatter_pdf = args.out / f"sanity11_relative_effect_scatter_vs_{args.ref_cond.lower()}.pdf"
    plot_relative_effect_scatter(rel_table, args.ref_cond, rel_scatter_png, rel_scatter_pdf)

    report_path = args.out / "sanity11_report.txt"
    shared_h = compare[["cond", "sim_h_spatial_mean", "exp_h_spatial_mean"]].dropna()
    shared_cv = compare[["cond", "sim_cv_mean", "exp_cv_mean"]].dropna()

    with report_path.open("w", encoding="utf-8") as f:
        f.write("Sanity 11 report\n")
        f.write("================\n")
        f.write(f"Simulation root: {args.sim_root}\n")
        f.write(f"Experimental metrics: {args.exp_metrics}\n")
        f.write(f"Day: {args.day}\n")
        f.write(f"Window: {args.window}\n")
        f.write(f"HAp threshold: {args.ha_thresh}\n\n")
        f.write(f"Relative-effect reference condition: {args.ref_cond}\n\n")

        f.write("Detected simulation replicates per condition\n")
        for c in COND_ORDER:
            n = int((sim_metrics["cond"] == c).sum()) if not sim_metrics.empty else 0
            f.write(f"  {c}: n={n}\n")
        f.write("\n")

        if sim_missing:
            f.write("Missing simulation data notes\n")
            for line in sim_missing:
                f.write(f"  - {line}\n")
            f.write("\n")

        if len(shared_h) >= 2:
            r_h, p_h = stats.pearsonr(shared_h["sim_h_spatial_mean"], shared_h["exp_h_spatial_mean"])
            rho_h, p_s_h = stats.spearmanr(shared_h["sim_h_spatial_mean"], shared_h["exp_h_spatial_mean"])
            f.write(f"H_spatial mean correlation across shared conditions (n={len(shared_h)}):\n")
            f.write(f"  Pearson r={r_h:.4g}, p={p_h:.4g}\n")
            f.write(f"  Spearman rho={rho_h:.4g}, p={p_s_h:.4g}\n\n")
        else:
            f.write("H_spatial mean correlation: insufficient shared conditions (need >= 2)\n\n")

        if len(shared_cv) >= 2:
            r_cv, p_cv = stats.pearsonr(shared_cv["sim_cv_mean"], shared_cv["exp_cv_mean"])
            rho_cv, p_s_cv = stats.spearmanr(shared_cv["sim_cv_mean"], shared_cv["exp_cv_mean"])
            f.write(f"CV mean correlation across shared conditions (n={len(shared_cv)}):\n")
            f.write(f"  Pearson r={r_cv:.4g}, p={p_cv:.4g}\n")
            f.write(f"  Spearman rho={rho_cv:.4g}, p={p_s_cv:.4g}\n")
        else:
            f.write("CV mean correlation: insufficient shared conditions (need >= 2)\n")
        f.write("\n")
        for metric, label in [("h_spatial", "H_spatial"), ("cv", "CV")]:
            m = agreement_metrics(rel_table, metric, args.ref_cond)
            f.write(f"{label} relative effect agreement vs {args.ref_cond} (non-reference n={m['n']}):\n")
            f.write(f"  Kendall tau={m['kendall_tau']:.4g}\n")
            f.write(f"  Sign agreement={100.0 * m['sign_agreement']:.2f}%\n")
            f.write(f"  MAE={m['mae']:.4g}\n")

    print(f"Saved: {sim_metrics_path}")
    print(f"Saved: {exp_metrics_path}")
    print(f"Saved: {sim_summary_path}")
    print(f"Saved: {exp_summary_path}")
    print(f"Saved: {compare_path}")
    print(f"Saved: {args.out / 'sanity11_h_spatial_compare.png'}")
    print(f"Saved: {args.out / 'sanity11_h_spatial_compare.pdf'}")
    print(f"Saved: {args.out / 'sanity11_cv_compare.png'}")
    print(f"Saved: {args.out / 'sanity11_cv_compare.pdf'}")
    print(f"Saved: {rel_table_path}")
    print(f"Saved: {rel_scatter_png}")
    print(f"Saved: {rel_scatter_pdf}")
    print(f"Saved: {report_path}")


if __name__ == "__main__":
    main()
