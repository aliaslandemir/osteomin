#!/usr/bin/env python3

#python sanity/sanity_check6.py --samples 256



from __future__ import annotations

import argparse
import csv
import json
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

matplotlib.use("Agg")
import matplotlib.pyplot as plt


BASE_OUT = SCRIPT_DIR / "out" / "sanity_check6"
ALP_DAY = 10
COLLAGEN_DAY = 14
ARS_DAY = 28

BASE_ARGS = [
    str(PROJECT_ROOT / "main.py"),
    "--hours",
    "672",
    "--cells",
    "72",
    "--seed-mode",
    "random",
    "--w",
    "33",
    "--h",
    "33",
    "--nodules",
    "40n",
    "--dex-conc",
    "0",
    "--mk4-conc",
    "0",
    "--boundary-mode",
    "periodic",
    "--media-interval",
    "48",
    "--random-move",
    "--move-prob",
    "0.27",
    "--max-moves",
    "1",
    "--proliferation",
    "--ob-div-prob",
    "0.0",
    "--apoptosis",
    "--apoptosis-prob",
    "0.0010",
    "--apoptosis-osteo",
    "0.0",
    "--osteogenic-commitment",
    "--osteo-induction-hours",
    "72",
    "--commit-delay",
    "48",
    "--commit-k",
    "0.0125",
    "--aa0",
    "50.0",
    "--aa-half-life",
    "8",
    "--collagen-aa-km",
    "5.0",
    "--min-div-hours",
    "38",
    "--max-div-hours",
    "52",
    "--secretion",
    "--alp-rb",
    "0.0949",
    "--alp-ro",
    "0.810",
    "--collagen-rb",
    "0.00336",
    "--collagen-ro",
    "0.0130",
    "--alp-dex-ref",
    "22.7",
    "--collagen-dex-ref",
    "6.22",
    "--ars-dex-ref",
    "46.6",
    "--alp-mk4-ref",
    "0.784",
    "--collagen-mk4-ref",
    "0.349",
    "--ars-mk4-ref",
    "0.667",
    "--ha-model",
    "minimal",
    "--ha-ro",
    "0.102",
    "--alp-a-d",
    "3.72",
    "--alp-b-d",
    "1.75",
    "--collagen-a-d",
    "2.69",
    "--collagen-b-d",
    "2.04",
    "--ha-a-d",
    "1.22",
    "--ha-b-d",
    "0.679",
    "--alp-a-k",
    "1.10",
    "--alp-b-k",
    "1.06",
    "--collagen-a-k",
    "4.66",
    "--collagen-b-k",
    "2.15",
    "--ha-a-k",
    "4.66",
    "--ha-b-k",
    "2.15",
    "--ha-tau",
    "384",
    "--ha-hill-n",
    "2",
    "--spatial-every",
    "0",
    "--no-snapshots",
    "--no-unique-id",
    "--alp-plot-hours",
    "240",
    "--collagen-plot-hours",
    "336",
    "--mineral-plot-hours",
    "672",
    "--no-hourly-exports",
    "--csv",
    "--no-xlsx",
]

PARAMETERS: list[dict[str, Any]] = [
    {"name": "alp_rb", "flag": "--alp-rb", "bounds": [0.07, 0.13], "baseline": 0.0949, "label": "ALP basal rate"},
    {"name": "alp_ro", "flag": "--alp-ro", "bounds": [0.60, 1.05], "baseline": 0.810, "label": "ALP osteogenic rate"},
    {"name": "collagen_rb", "flag": "--collagen-rb", "bounds": [0.0020, 0.0055], "baseline": 0.00336, "label": "Collagen basal rate"},
    {"name": "collagen_ro", "flag": "--collagen-ro", "bounds": [0.0080, 0.0200], "baseline": 0.0130, "label": "Collagen osteogenic rate"},
    {"name": "ha_ro", "flag": "--ha-ro", "bounds": [0.075, 0.150], "baseline": 0.102, "label": "HAp osteogenic rate"},
    {"name": "ha_tau", "flag": "--ha-tau", "bounds": [288.0, 480.0], "baseline": 384.0, "label": "HAp Hill tau"},
    {"name": "ha_hill_n", "flag": "--ha-hill-n", "bounds": [1.5, 3.5], "baseline": 2.0, "label": "HAp Hill exponent"},
    {"name": "commit_k", "flag": "--commit-k", "bounds": [0.0075, 0.0200], "baseline": 0.0125, "label": "Commitment k"},
    {"name": "apoptosis_prob", "flag": "--apoptosis-prob", "bounds": [0.0005, 0.0015], "baseline": 0.0010, "label": "Apoptosis probability"},
    {"name": "move_prob", "flag": "--move-prob", "bounds": [0.15, 0.40], "baseline": 0.27, "label": "Random-move probability"},
    {"name": "aa0", "flag": "--aa0", "bounds": [35.0, 65.0], "baseline": 50.0, "label": "AA initial concentration"},
    {"name": "aa_half_life", "flag": "--aa-half-life", "bounds": [4.0, 12.0], "baseline": 8.0, "label": "AA half-life (h)"},
    {"name": "collagen_aa_km", "flag": "--collagen-aa-km", "bounds": [2.5, 8.0], "baseline": 5.0, "label": "Collagen AA half-sat"},
    {"name": "commit_delay", "flag": "--commit-delay", "bounds": [24.0, 72.0], "baseline": 48.0, "label": "Commitment delay (h)", "kind": "int"},
    {"name": "min_div_hours", "flag": "--min-div-hours", "bounds": [30.0, 46.0], "baseline": 38.0, "label": "Minimum division interval (h)", "kind": "int"},
    {"name": "div_gap_hours", "flag": None, "bounds": [8.0, 24.0], "baseline": 14.0, "label": "Division interval gap (h)", "kind": "int"},
    {"name": "proliferation_chance", "flag": "--proliferation-chance", "bounds": [0.75, 1.0], "baseline": 1.0, "label": "Division attempt chance"},
    {"name": "ob_apoptosis_prob", "flag": "--apoptosis-osteo", "bounds": [0.0, 0.0010], "baseline": 0.0, "label": "OB apoptosis probability"},
    {"name": "osteo_induction_hours", "flag": "--osteo-induction-hours", "bounds": [48.0, 96.0], "baseline": 72.0, "label": "Osteogenic induction time (h)", "kind": "int"},
    {"name": "media_interval", "flag": "--media-interval", "bounds": [24.0, 72.0], "baseline": 48.0, "label": "Media interval (h)", "kind": "int"},
]

ENDPOINTS = [
    {"name": "alp_240h", "title": "ALP endpoint (240 h)", "color": "#c75a1b"},
    {"name": "collagen_336h", "title": "Collagen endpoint (336 h)", "color": "#159570"},
    {"name": "ars_672h", "title": "HAp endpoint / ARS proxy (672 h)", "color": "#2a5f9e"},
]

COLORS = {
    "ink": "#111827",
    "grid": "#d7dee7",
    "bg": "#ffffff",
    "accent": "#8a5b2a",
    "baseline": "#7a1f1f",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.sans-serif": ["DejaVu Sans", "Arial", "Liberation Sans", "sans-serif"],
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 13.0,
        "axes.titleweight": "semibold",
        "axes.labelsize": 11.0,
        "xtick.labelsize": 10.0,
        "ytick.labelsize": 10.0,
        "xtick.color": COLORS["ink"],
        "ytick.color": COLORS["ink"],
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sobol sensitivity analysis for OsteoMin endpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--out", type=Path, default=BASE_OUT, help="output root directory")
    parser.add_argument("--python", type=str, default=sys.executable, help="Python interpreter for main.py")
    parser.add_argument("--samples", type=int, default=1024, help="Sobol base sample size N (total runs ~= N*(2D+2))")
    parser.add_argument("--base-seed", type=int, default=12345, help="base random seed for model runs")
    parser.add_argument("--sobol-seed", type=int, default=24680, help="Sobol sampler seed")
    parser.add_argument("--overwrite", action="store_true", help="rerun simulations even if existing outputs are present")
    parser.add_argument("--analyze-only", action="store_true", help="skip simulation and analyze existing CSV outputs")
    parser.add_argument("--convergence", action="store_true", help="run Sobol analyses at N = 128, 256, 512, and 1024")
    parser.add_argument("--dry-run", action="store_true", help="print commands without launching runs")
    parser.add_argument("--runtime", action="store_true", help="pass --runtime to main.py")
    parser.add_argument("--second-order", action="store_true", help="compute second-order Sobol indices")
    parser.set_defaults(second_order=False)
    args = parser.parse_args()
    if args.samples <= 0:
        raise SystemExit("--samples must be >= 1")
    if args.samples & (args.samples - 1) != 0:
        print("[warn] --samples should preferably be a power of 2 (128, 256, 512, 1024, ...)")
    return args


def style_axes(ax: plt.Axes, grid_axis: str | None = "y", grid_alpha: float = 0.4) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=4.2, width=0.8)
    if grid_axis:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=grid_alpha)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=600, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def load_salib() -> tuple[Any, Any]:
    try:
        from SALib.analyze import sobol as sobol_analyzer
    except Exception as exc:
        raise SystemExit(
            "SALib is required. Install with: pip install SALib\n"
            f"Import error: {exc}"
        ) from exc
    return sobol_analyzer, None


def generate_samples(problem: dict[str, Any], n: int, second_order: bool, seed: int) -> np.ndarray:
    try:
        from SALib.sample import sobol as sobol_sampler

        try:
            return sobol_sampler.sample(
                problem,
                n,
                calc_second_order=second_order,
                scramble=True,
                seed=seed,
            )
        except TypeError:
            np.random.seed(seed)
            return sobol_sampler.sample(problem, n, calc_second_order=second_order)
    except Exception:
        from SALib.sample import saltelli

        np.random.seed(seed)
        return saltelli.sample(problem, n, calc_second_order=second_order)


def extract_endpoints(timeseries_path: Path) -> dict[str, float]:
    rows_by_day: dict[int, dict[str, str]] = {}
    with timeseries_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows_by_day[int(float(row["day"]))] = row

    missing = [d for d in (ALP_DAY, COLLAGEN_DAY, ARS_DAY) if d not in rows_by_day]
    if missing:
        raise ValueError(f"missing required day rows {missing} in {timeseries_path}")

    return {
        "alp_240h": float(rows_by_day[ALP_DAY]["alp_total"]),
        "collagen_336h": float(rows_by_day[COLLAGEN_DAY]["collagen_total"]),
        "ars_672h": float(rows_by_day[ARS_DAY]["ha_total"]),
        "n_total_672h": float(rows_by_day[ARS_DAY]["n_total"]),
    }


def build_command(
    python_exe: str,
    sample_values: dict[str, float],
    seed: int,
    out_dir: Path,
    runtime: bool,
) -> tuple[list[str], dict[str, float], dict[str, float]]:
    cmd = [python_exe, *BASE_ARGS, "--seed", str(seed), "--out", str(out_dir)]
    effective_inputs = dict(sample_values)
    effective_inputs["commit_delay"] = max(0, int(round(float(effective_inputs["commit_delay"]))))
    effective_inputs["min_div_hours"] = max(1, int(round(float(effective_inputs["min_div_hours"]))))
    effective_inputs["div_gap_hours"] = max(0, int(round(float(effective_inputs["div_gap_hours"]))))
    effective_inputs["osteo_induction_hours"] = max(
        0, int(round(float(effective_inputs["osteo_induction_hours"])))
    )
    effective_inputs["media_interval"] = max(1, int(round(float(effective_inputs["media_interval"]))))

    model_inputs = dict(effective_inputs)
    model_inputs["max_div_hours"] = int(model_inputs["min_div_hours"] + model_inputs["div_gap_hours"])

    for spec in PARAMETERS:
        if spec["flag"] is None:
            continue
        val = model_inputs[spec["name"]]
        if spec.get("kind") == "int":
            cmd.extend([spec["flag"], str(int(val))])
        else:
            cmd.extend([spec["flag"], f"{float(val):.10g}"])
    cmd.extend(["--max-div-hours", str(int(model_inputs["max_div_hours"]))])
    if runtime:
        cmd.append("--runtime")
    return cmd, effective_inputs, model_inputs


def run_sample(
    sample_index: int,
    row: np.ndarray,
    args: argparse.Namespace,
    runs_dir: Path,
) -> dict[str, Any]:
    run_dir = runs_dir / f"s{sample_index:05d}"
    run_dir.mkdir(parents=True, exist_ok=True)
    timeseries_path = run_dir / "timeseries.csv"

    sample_values = {PARAMETERS[i]["name"]: float(row[i]) for i in range(len(PARAMETERS))}
    seed = int(args.base_seed + sample_index)
    cmd, effective_inputs, model_inputs = build_command(args.python, sample_values, seed, run_dir, args.runtime)
    command_str = shlex.join(cmd)

    if args.dry_run:
        print(command_str)
        return {
            "sample_index": sample_index,
            "seed": seed,
            "run_dir": str(run_dir),
            "success": True,
            "returncode": 0,
            "command": command_str,
            **{f"raw_{name}": value for name, value in sample_values.items()},
            **effective_inputs,
            "max_div_hours_used": float(model_inputs["max_div_hours"]),
        }

    if timeseries_path.exists() and not args.overwrite:
        endpoints = extract_endpoints(timeseries_path)
        return {
            "sample_index": sample_index,
            "seed": seed,
            "run_dir": str(run_dir),
            "success": True,
            "returncode": 0,
            "command": command_str,
            **{f"raw_{name}": value for name, value in sample_values.items()},
            **effective_inputs,
            "max_div_hours_used": float(model_inputs["max_div_hours"]),
            **endpoints,
        }

    completed = subprocess.run(cmd, check=False)
    success = completed.returncode == 0
    row_out: dict[str, Any] = {
        "sample_index": sample_index,
        "seed": seed,
        "run_dir": str(run_dir),
        "success": bool(success),
        "returncode": int(completed.returncode),
        "command": command_str,
        **{f"raw_{name}": value for name, value in sample_values.items()},
        **effective_inputs,
        "max_div_hours_used": float(model_inputs["max_div_hours"]),
    }
    if success:
        row_out.update(extract_endpoints(timeseries_path))
    return row_out


def problem_definition() -> dict[str, Any]:
    return {
        "num_vars": len(PARAMETERS),
        "names": [str(spec["name"]) for spec in PARAMETERS],
        "bounds": [list(spec["bounds"]) for spec in PARAMETERS],
    }


def write_samples(samples: np.ndarray, out_dir: Path) -> pd.DataFrame:
    df = pd.DataFrame(samples, columns=[spec["name"] for spec in PARAMETERS])
    df.insert(0, "sample_index", np.arange(1, len(df) + 1))
    df.to_csv(out_dir / "sample_matrix.csv", index=False)
    return df


def write_effective_samples(endpoint_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    cols = ["sample_index", *[str(spec["name"]) for spec in PARAMETERS], "max_div_hours_used"]
    effective_df = endpoint_df.loc[:, cols].copy()
    effective_df.to_csv(out_dir / "effective_sample_matrix.csv", index=False)
    return effective_df


def run_or_load_results(args: argparse.Namespace, samples: np.ndarray, out_dir: Path) -> pd.DataFrame:
    runs_dir = out_dir / "runs"
    runs_dir.mkdir(parents=True, exist_ok=True)

    if args.analyze_only:
        endpoint_path = out_dir / "endpoint_results.csv"
        if not endpoint_path.exists():
            raise SystemExit(f"--analyze-only requested but missing {endpoint_path}")
        return pd.read_csv(endpoint_path)

    results: list[dict[str, Any]] = []
    for i, row in enumerate(samples, start=1):
        print(f"[sanity_check6] run {i}/{len(samples)}")
        results.append(run_sample(i, row, args, runs_dir))

    df = pd.DataFrame(results)
    df.to_csv(out_dir / "run_summary.csv", index=False)

    if args.dry_run:
        return df

    failed = df[~df["success"].astype(bool)]
    if not failed.empty:
        fail_path = out_dir / "failed_runs.csv"
        failed.to_csv(fail_path, index=False)
        raise SystemExit(f"{len(failed)} run(s) failed. See {fail_path}")

    df.to_csv(out_dir / "endpoint_results.csv", index=False)
    return df


def sobol_analysis(
    problem: dict[str, Any],
    endpoint_df: pd.DataFrame,
    out_dir: Path,
    *,
    second_order: bool,
) -> tuple[pd.DataFrame, dict[str, dict[str, Any]]]:
    sobol_analyzer, _ = load_salib()
    names = [str(n) for n in problem["names"]]

    sobol_rows: list[dict[str, Any]] = []
    raw: dict[str, dict[str, Any]] = {}

    for endpoint in ENDPOINTS:
        ep = endpoint["name"]
        y = endpoint_df.sort_values("sample_index")[ep].to_numpy(dtype=float)
        si = sobol_analyzer.analyze(problem, y, calc_second_order=second_order, print_to_console=False)
        raw[ep] = si
        for i, name in enumerate(names):
            sobol_rows.append(
                {
                    "endpoint": ep,
                    "parameter": name,
                    "S1": float(si["S1"][i]),
                    "S1_conf": float(si["S1_conf"][i]),
                    "ST": float(si["ST"][i]),
                    "ST_conf": float(si["ST_conf"][i]),
                }
            )

        pd.DataFrame(
            {
                "parameter": names,
                "S1": si["S1"],
                "S1_conf": si["S1_conf"],
                "ST": si["ST"],
                "ST_conf": si["ST_conf"],
            }
        ).to_csv(out_dir / f"sobol_{ep}.csv", index=False)

        if second_order and "S2" in si:
            s2 = np.asarray(si["S2"], dtype=float)
            pd.DataFrame(s2, index=names, columns=names).to_csv(out_dir / f"sobol_{ep}_S2_matrix.csv")

    sobol_df = pd.DataFrame(sobol_rows)
    sobol_df.to_csv(out_dir / "sobol_indices_all_endpoints.csv", index=False)

    ranking = (
        sobol_df.groupby("parameter", as_index=False)["ST"]
        .mean()
        .sort_values("ST", ascending=False)
        .rename(columns={"ST": "mean_ST_across_endpoints"})
    )
    ranking.to_csv(out_dir / "parameter_ranking.csv", index=False)

    return sobol_df, raw


def clean_sobol_for_plot(vals: np.ndarray, err: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    vals = np.asarray(vals, dtype=float).copy()
    err = np.asarray(err, dtype=float).copy()

    vals = np.clip(vals, 0.0, 1.0)
    lo = np.maximum(0.0, vals - err)
    hi = np.minimum(1.0, vals + err)
    err_left = vals - lo
    err_right = hi - vals
    return vals, np.vstack([err_left, err_right])


def endpoint_color(name: str) -> str:
    for ep in ENDPOINTS:
        if ep["name"] == name:
            return str(ep["color"])
    return "#334155"


def endpoint_title(name: str) -> str:
    for ep in ENDPOINTS:
        if ep["name"] == name:
            return str(ep["title"])
    return name


def plot_sampling_coverage(sample_df: pd.DataFrame, out_fig: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    for idx, spec in enumerate(PARAMETERS):
        vals = sample_df[spec["name"]].to_numpy(dtype=float)
        y = np.full(vals.shape[0], idx, dtype=float)
        jitter = (np.random.default_rng(100 + idx).random(vals.shape[0]) - 0.5) * 0.26
        ax.scatter(vals, y + jitter, s=13, alpha=0.52, color=COLORS["accent"], linewidths=0)
        ax.plot(
            [spec["bounds"][0], spec["bounds"][1]],
            [idx, idx],
            color="#94a3b8",
            lw=1.2,
            solid_capstyle="round",
            zorder=0,
        )
        ax.scatter([spec["baseline"]], [idx], marker="D", s=24, color=COLORS["baseline"], zorder=3)

    ax.set_yticks(range(len(PARAMETERS)))
    ax.set_yticklabels([spec["name"] for spec in PARAMETERS])
    ax.set_xlabel("effective parameter value")
    ax.set_title("Sobol effective-input coverage")
    style_axes(ax, "x", 0.3)
    save_figure(fig, out_fig, "figure_1_sampling_coverage")


def plot_s1_st(sobol_df: pd.DataFrame, out_fig: Path, col: str, stem: str, title: str) -> None:
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(16.5, 5.3), sharey=False)
    for ax, endpoint in zip(axes, ENDPOINTS):
        ep = endpoint["name"]
        sub = sobol_df[sobol_df["endpoint"] == ep].sort_values("ST", ascending=False).reset_index(drop=True)
        order = sub["parameter"].tolist()
        vals_raw = sub[col].to_numpy(dtype=float)
        err_raw = sub[f"{col}_conf"].to_numpy(dtype=float)
        vals, xerr = clean_sobol_for_plot(vals_raw, err_raw)
        ypos = np.arange(len(order))
        ax.barh(ypos, vals, xerr=xerr, color=endpoint["color"], alpha=0.9, ecolor="#334155", capsize=2.0)
        ax.set_title(endpoint_title(ep))
        ax.set_yticks(ypos)
        ax.set_yticklabels(order)
        ax.invert_yaxis()
        ax.set_xlabel(col)
        ax.axvline(0.0, color="#64748b", lw=1.0)
        ax.set_xlim(0.0, 1.0)
        style_axes(ax, "x", 0.35)
    fig.suptitle(title, fontsize=14.0, fontweight="semibold")
    save_figure(fig, out_fig, stem)


def plot_s2_heatmaps(raw_sobol: dict[str, dict[str, Any]], out_fig: Path) -> None:
    names = [spec["name"] for spec in PARAMETERS]
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(16.5, 5.0))
    vlim = 0.0
    matrices: list[np.ndarray] = []
    for endpoint in ENDPOINTS:
        s2 = raw_sobol.get(endpoint["name"], {}).get("S2")
        if s2 is None:
            s2m = np.zeros((len(names), len(names)), dtype=float)
        else:
            s2m = np.nan_to_num(np.asarray(s2, dtype=float), nan=0.0)
        matrices.append(s2m)
        vlim = max(vlim, float(np.max(np.abs(s2m))))
    vlim = max(vlim, 1e-6)

    for ax, endpoint, s2m in zip(axes, ENDPOINTS, matrices):
        im = ax.imshow(s2m, cmap="RdBu_r", vmin=-vlim, vmax=vlim, interpolation="nearest")
        ax.set_title(endpoint_title(endpoint["name"]))
        ax.set_xticks(range(len(names)))
        ax.set_yticks(range(len(names)))
        ax.set_xticklabels(names, rotation=90)
        ax.set_yticklabels(names)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.85, pad=0.02)
    cbar.set_label("S2 interaction effect")
    fig.suptitle("Second-order Sobol interaction matrices", fontsize=14.0, fontweight="semibold")
    save_figure(fig, out_fig, "figure_4_sobol_s2_heatmaps")


def plot_endpoint_effects(endpoint_df: pd.DataFrame, sobol_df: pd.DataFrame, out_fig: Path) -> None:
    fig, axes = plt.subplots(1, len(ENDPOINTS), figsize=(16.5, 4.8))
    for ax, endpoint in zip(axes, ENDPOINTS):
        ep = endpoint["name"]
        top_param = (
            sobol_df[sobol_df["endpoint"] == ep]
            .sort_values("ST", ascending=False)
            .iloc[0]["parameter"]
        )
        x = endpoint_df[top_param].to_numpy(dtype=float)
        y = endpoint_df[ep].to_numpy(dtype=float)
        order = np.argsort(x)
        xs = x[order]
        ys = y[order]
        ax.scatter(x, y, s=17, alpha=0.42, color=endpoint["color"], edgecolors="none")
        if len(xs) >= 6:
            bins = np.array_split(np.arange(len(xs)), 12)
            x_smooth = np.array([float(np.mean(xs[b])) for b in bins if len(b) > 0], dtype=float)
            y_smooth = np.array([float(np.mean(ys[b])) for b in bins if len(b) > 0], dtype=float)
            ax.plot(x_smooth, y_smooth, color=COLORS["ink"], lw=1.8, alpha=0.95)
        ax.set_xlabel(top_param)
        ax.set_ylabel(ep)
        ax.set_title(f"{endpoint_title(ep)}\n(top parameter: {top_param})")
        style_axes(ax, "both", 0.32)
    fig.suptitle("Endpoint response versus top Sobol parameter", fontsize=14.0, fontweight="semibold")
    save_figure(fig, out_fig, "figure_5_endpoint_driver_effects")


def export_plots(
    sample_df: pd.DataFrame,
    endpoint_df: pd.DataFrame,
    sobol_df: pd.DataFrame,
    raw_sobol: dict[str, dict[str, Any]],
    out_dir: Path,
) -> None:
    fig_dir = out_dir / "figures"
    plot_sampling_coverage(sample_df, fig_dir)
    plot_s1_st(
        sobol_df,
        fig_dir,
        col="S1",
        stem="figure_2_sobol_first_order",
        title="First-order Sobol indices (S1)",
    )
    plot_s1_st(
        sobol_df,
        fig_dir,
        col="ST",
        stem="figure_3_sobol_total_order",
        title="Total-order Sobol indices (ST)",
    )
    if any(raw_sobol.get(endpoint["name"], {}).get("S2") is not None for endpoint in ENDPOINTS):
        plot_s2_heatmaps(raw_sobol, fig_dir)
    plot_endpoint_effects(endpoint_df, sobol_df, fig_dir)


def expected_runs(n: int, second_order: bool, d: int) -> int:
    return n * ((2 * d + 2) if second_order else (d + 2))


def run_single_analysis(
    args: argparse.Namespace,
    out_dir: Path,
    *,
    n_samples: int,
    sobol_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    problem = problem_definition()
    problem_path = out_dir / "sobol_problem.json"
    problem_path.write_text(json.dumps(problem, indent=2), encoding="utf-8")

    if args.analyze_only:
        sample_path = out_dir / "sample_matrix.csv"
        if not sample_path.exists():
            raise SystemExit(f"--analyze-only requested but missing {sample_path}")
        sample_df = pd.read_csv(sample_path)
        samples = sample_df[[spec["name"] for spec in PARAMETERS]].to_numpy(dtype=float)
    else:
        samples = generate_samples(problem, n_samples, args.second_order, sobol_seed)
        sample_df = write_samples(samples, out_dir)

    d = len(PARAMETERS)
    print(f"[sanity_check6] parameters: {d}")
    print(f"[sanity_check6] expected runs: {expected_runs(n_samples, args.second_order, d)} (N={n_samples}, second_order={args.second_order})")

    endpoint_df = run_or_load_results(args, samples, out_dir)
    if args.dry_run:
        print("[sanity_check6] dry run complete; no Sobol analysis or figures generated")
        return sample_df, endpoint_df, {}

    required_cols = {"sample_index", "alp_240h", "collagen_336h", "ars_672h"}
    if not required_cols.issubset(set(endpoint_df.columns)):
        missing = sorted(required_cols.difference(set(endpoint_df.columns)))
        raise SystemExit(f"endpoint_results missing columns: {missing}")

    endpoint_df = endpoint_df.sort_values("sample_index").reset_index(drop=True)
    effective_df = write_effective_samples(endpoint_df, out_dir)
    sobol_df, raw_sobol = sobol_analysis(problem, endpoint_df, out_dir, second_order=args.second_order)
    export_plots(effective_df, endpoint_df, sobol_df, raw_sobol, out_dir)
    return effective_df, sobol_df, raw_sobol


def run_convergence(args: argparse.Namespace) -> None:
    convergence_ns = [128, 256, 512, 1024]
    rows: list[dict[str, Any]] = []
    for offset, n_samples in enumerate(convergence_ns):
        run_out = args.out / "convergence" / f"N{n_samples:04d}"
        _, sobol_df, _ = run_single_analysis(args, run_out, n_samples=n_samples, sobol_seed=args.sobol_seed + offset)
        if args.dry_run:
            continue
        for endpoint in ENDPOINTS:
            ep = endpoint["name"]
            top = (
                sobol_df[sobol_df["endpoint"] == ep]
                .sort_values("ST", ascending=False)
                .head(5)
            )
            for _, record in top.iterrows():
                rows.append(
                    {
                        "N": n_samples,
                        "endpoint": ep,
                        "parameter": str(record["parameter"]),
                        "ST": float(record["ST"]),
                        "ST_conf": float(record["ST_conf"]),
                        "S1": float(record["S1"]),
                        "S1_conf": float(record["S1_conf"]),
                    }
                )
    if rows:
        pd.DataFrame(rows).to_csv(args.out / "convergence_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.convergence:
        run_convergence(args)
        print(f"[sanity_check6] convergence root: {args.out / 'convergence'}")
        if not args.dry_run:
            print(f"[sanity_check6] convergence summary: {args.out / 'convergence_summary.csv'}")
        return

    sample_df, sobol_df, raw_sobol = run_single_analysis(
        args,
        args.out,
        n_samples=args.samples,
        sobol_seed=args.sobol_seed,
    )
    if args.dry_run:
        return

    figure_count = 4 + int(any(raw_sobol.get(endpoint["name"], {}).get("S2") is not None for endpoint in ENDPOINTS))
    print(f"[sanity_check6] samples: {len(sample_df)}")
    print(f"[sanity_check6] endpoint table: {args.out / 'endpoint_results.csv'}")
    print(f"[sanity_check6] effective sample table: {args.out / 'effective_sample_matrix.csv'}")
    print(f"[sanity_check6] Sobol table: {args.out / 'sobol_indices_all_endpoints.csv'}")
    print(f"[sanity_check6] ranking: {args.out / 'parameter_ranking.csv'}")
    print(f"[sanity_check6] figures: {args.out / 'figures'} ({figure_count} figures, png+pdf)")


if __name__ == "__main__":
    main()
