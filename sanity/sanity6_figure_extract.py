#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = Path(__file__).resolve().parent

ENDPOINTS = [
    {
        "key": "alp_240h",
        "field": "alp_total",
        "title": "ALP (240 h)",
        "color": "#c75a1b",
    },
    {
        "key": "collagen_336h",
        "field": "collagen_total",
        "title": "Collagen (336 h)",
        "color": "#1b9e77",
    },
    {
        "key": "ars_672h",
        "field": "ha_total",
        "title": "HAp / ARS proxy (672 h)",
        "color": "#2b6cb0",
    },
]

COLORS = {
    "ink": "#0f172a",
    "muted": "#64748b",
    "grid": "#d8dee9",
    "bg": "#ffffff",
    "highlight": "#b45309",
    "low": "#94a3b8",
    "high": "#0f766e",
}

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["ink"],
        "axes.linewidth": 0.9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "semibold",
        "axes.titlesize": 12.5,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
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
        description="Publication-grade figure extraction for sanity_check6 sensitivity outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default=str(SCRIPT_DIR / "out" / "sanity_check6"),
        help="root folder with Sobol/output CSVs",
    )
    parser.add_argument(
        "--runs",
        type=str,
        default=str(SCRIPT_DIR / "out" / "sanity_check6" / "runs"),
        help="runs folder containing sXXXXX/timeseries.csv",
    )
    parser.add_argument(
        "--out",
        type=str,
        default="",
        help="output directory; default is <input>/figure_extract_6_pub",
    )
    parser.add_argument(
        "--clip-for-display",
        action="store_true",
        help="clip Sobol indices/CIs for plotting readability; raw CSV values are unchanged",
    )
    return parser.parse_args()


def normalize_path(raw: str) -> Path:
    txt = str(raw).strip()
    if txt.startswith("\\\\wsl$\\"):
        slash = txt.replace("\\", "/")
        # Convert //wsl$/Distro/home/... to /home/...
        parts = slash.split("/", 4)
        if len(parts) >= 5:
            return Path("/" + parts[4])
    return Path(txt)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y", alpha: float = 0.45) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=3.8, width=0.8)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=alpha)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=500, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=500, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def load_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoint = pd.read_csv(input_dir / "endpoint_results.csv")
    sobol = pd.read_csv(input_dir / "sobol_indices_all_endpoints.csv")
    rank = pd.read_csv(input_dir / "parameter_ranking.csv")
    return endpoint, sobol, rank


def prepare_sobol_for_plot(
    sobol_df: pd.DataFrame, *, clip_for_display: bool
) -> pd.DataFrame:
    df = sobol_df.copy()
    # Always keep non-negative ST for ranking/normalization.
    df["ST_nonneg"] = df["ST"].clip(lower=0.0)
    max_by_endpoint = (
        df.groupby("endpoint")["ST_nonneg"].transform("max").replace(0.0, np.nan)
    )
    df["ST_norm"] = (df["ST_nonneg"] / max_by_endpoint).fillna(0.0).clip(0.0, 1.0)
    if clip_for_display:
        df["S1_plot"] = df["S1"].clip(lower=-0.2, upper=1.2)
        df["ST_plot"] = df["ST_nonneg"].clip(lower=0.0, upper=1.2)
        df["S1_conf_plot"] = df["S1_conf"].abs().clip(lower=0.0, upper=0.6)
        df["ST_conf_plot"] = df["ST_conf"].abs().clip(lower=0.0, upper=0.6)
    else:
        df["S1_plot"] = df["S1"]
        df["ST_plot"] = df["ST_nonneg"]
        df["S1_conf_plot"] = df["S1_conf"].abs()
        df["ST_conf_plot"] = df["ST_conf"].abs()
    return df


def endpoint_order(sobol_df: pd.DataFrame) -> list[str]:
    metric = "ST_nonneg" if "ST_nonneg" in sobol_df.columns else "ST"
    order = (
        sobol_df.groupby("parameter", as_index=False)[metric]
        .mean()
        .sort_values(metric, ascending=False)["parameter"]
        .tolist()
    )
    return order


def endpoint_title(key: str) -> str:
    for e in ENDPOINTS:
        if e["key"] == key:
            return str(e["title"])
    return key


def endpoint_color(key: str) -> str:
    for e in ENDPOINTS:
        if e["key"] == key:
            return str(e["color"])
    return COLORS["highlight"]


def endpoint_field(key: str) -> str:
    for e in ENDPOINTS:
        if e["key"] == key:
            return str(e["field"])
    return ""


def endpoint_short_ylabel(key: str) -> str:
    mapping = {
        "alp_240h": "ALP",
        "collagen_336h": "Collagen",
        "ars_672h": "HAp/ARS",
    }
    return mapping.get(key, key)


def validate_sobol_table(sobol_df: pd.DataFrame) -> None:
    expected_eps = {e["key"] for e in ENDPOINTS}
    expected_pairs = {
        (p, ep) for p in sobol_df["parameter"].unique() for ep in expected_eps
    }
    observed_pairs = {
        (str(r["parameter"]), str(r["endpoint"])) for _, r in sobol_df.iterrows()
    }
    missing = sorted(expected_pairs.difference(observed_pairs))
    if missing:
        raise SystemExit(
            f"Missing parameter-endpoint combinations in Sobol table: {missing[:10]}"
        )

    dup = sobol_df.duplicated(subset=["parameter", "endpoint"], keep=False)
    if bool(dup.any()):
        d = sobol_df.loc[dup, ["parameter", "endpoint"]].drop_duplicates()
        raise SystemExit(
            f"Duplicate parameter-endpoint rows in Sobol table: {d.to_dict(orient='records')}"
        )


def fig1_total_effect_heatmap(sobol_df: pd.DataFrame, out_dir: Path) -> None:
    params = endpoint_order(sobol_df)
    eps = [e["key"] for e in ENDPOINTS]
    mat = np.zeros((len(params), len(eps)), dtype=float)
    for i, p in enumerate(params):
        for j, ep in enumerate(eps):
            row = sobol_df[
                (sobol_df["parameter"] == p) & (sobol_df["endpoint"] == ep)
            ].iloc[0]
            mat[i, j] = float(row["ST_norm"])

    cmap = LinearSegmentedColormap.from_list(
        "stmap", ["#f7fbff", "#bdd7e7", "#6baed6", "#2171b5", "#084594"]
    )

    fig, ax = plt.subplots(
        figsize=(7.2, max(4.2, 0.32 * len(params) + 1.4)), constrained_layout=True
    )
    im = ax.imshow(mat, aspect="auto", cmap=cmap, vmin=0.0, vmax=1.0)
    ax.set_xticks(np.arange(len(eps)))
    ax.set_xticklabels([endpoint_title(e) for e in eps], rotation=12, ha="right")
    ax.set_yticks(np.arange(len(params)))
    ax.set_yticklabels(params)
    ax.set_title("Total-order sensitivity heatmap (normalized ST)")
    ax.set_xlabel("Endpoint")
    ax.set_ylabel("Parameter")
    for i in range(len(params)):
        for j in range(len(eps)):
            v = mat[i, j]
            txt = f"{v:.2f}"
            color = "white" if v > 0.55 else COLORS["ink"]
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Normalized ST (0-1 within each endpoint)")
    save_figure(fig, out_dir, "figure_1_total_effect_heatmap")


def fig2_s1_vs_st(
    sobol_df: pd.DataFrame, out_dir: Path, *, clip_for_display: bool
) -> None:
    params = endpoint_order(sobol_df)[:10]
    fig, axes = plt.subplots(
        1, len(ENDPOINTS), figsize=(14.8, 5.3), sharey=True, constrained_layout=True
    )
    for ax, ep in zip(axes, [e["key"] for e in ENDPOINTS]):
        sub = (
            sobol_df[sobol_df["endpoint"] == ep]
            .set_index("parameter")
            .loc[params]
            .reset_index()
        )
        y = np.arange(len(params))
        s1 = sub["S1_plot"].to_numpy(float)
        st = sub["ST_plot"].to_numpy(float)
        s1c = sub["S1_conf_plot"].to_numpy(float)
        stc = sub["ST_conf_plot"].to_numpy(float)
        ax.hlines(y, s1, st, color="#94a3b8", lw=1.8, alpha=0.9)
        ax.errorbar(
            s1,
            y,
            xerr=s1c,
            fmt="o",
            color="#475569",
            markersize=4.5,
            capsize=2.0,
            label="S1",
        )
        ax.errorbar(
            st,
            y,
            xerr=stc,
            fmt="s",
            color=endpoint_color(ep),
            markersize=4.6,
            capsize=2.0,
            label="ST",
        )
        ax.axvline(0.0, color="#64748b", lw=1.0)
        if clip_for_display:
            ax.set_xlim(-0.2, 1.2)
        ax.set_title(endpoint_title(ep))
        ax.set_xlabel("Sensitivity index")
        ax.set_yticks(y)
        ax.set_yticklabels(params)
        ax.invert_yaxis()
        style_axes(ax, "x", 0.35)
    axes[-1].legend(frameon=False, loc="lower right", fontsize=8.8)
    fig.suptitle(
        "First-order vs total-order Sobol indices", fontsize=13.2, fontweight="semibold"
    )
    note = ""
    if clip_for_display:
        note += ""
    fig.text(
        0.5, 0.01, note, ha="center", va="bottom", fontsize=8.6, color=COLORS["muted"]
    )
    save_figure(fig, out_dir, "figure_2_s1_vs_st")


def fig3_parameter_ranking(rank_df: pd.DataFrame, out_dir: Path) -> None:
    rank = rank_df.sort_values("mean_ST_across_endpoints", ascending=True)
    fig, ax = plt.subplots(
        figsize=(7.0, max(4.1, 0.32 * len(rank) + 1.2)), constrained_layout=True
    )
    bars = ax.barh(
        rank["parameter"],
        rank["mean_ST_across_endpoints"],
        color="#2563eb",
        alpha=0.9,
        edgecolor="none",
    )
    top_name = str(rank.iloc[-1]["parameter"])
    for b, p in zip(bars, rank["parameter"]):
        if str(p) == top_name:
            b.set_color(COLORS["highlight"])
    ax.set_xlabel("Mean ST across endpoints")
    ax.set_title("Global parameter importance ranking")
    style_axes(ax, "x", 0.35)
    save_figure(fig, out_dir, "figure_3_parameter_ranking")


def fig4_top2_response_maps(
    endpoint_df: pd.DataFrame,
    sobol_plot_df: pd.DataFrame,
    sobol_raw_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1, len(ENDPOINTS), figsize=(15.2, 4.9), constrained_layout=True
    )
    for ax, ep in zip(axes, [e["key"] for e in ENDPOINTS]):
        top = (
            sobol_raw_df[sobol_raw_df["endpoint"] == ep]
            .assign(ST_nonneg=lambda d: d["ST"].clip(lower=0.0))
            .sort_values("ST_nonneg", ascending=False)
            .head(2)["parameter"]
            .tolist()
        )
        x, y = top[0], top[1]
        c = endpoint_df[ep].to_numpy(float)
        p = ax.scatter(
            endpoint_df[x].to_numpy(float),
            endpoint_df[y].to_numpy(float),
            c=c,
            cmap="viridis",
            s=28,
            alpha=0.86,
            edgecolors="none",
        )
        ax.set_title(f"{endpoint_title(ep)}\n(top pair: {x}, {y})")
        ax.set_xlabel(x)
        ax.set_ylabel(y)
        style_axes(ax, "both", 0.25)
        cbar = fig.colorbar(p, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(ep)
        sub_plot = sobol_plot_df[sobol_plot_df["endpoint"] == ep].set_index("parameter")
        st1 = float(sub_plot.loc[x, "ST_norm"]) if x in sub_plot.index else float("nan")
        st2 = float(sub_plot.loc[y, "ST_norm"]) if y in sub_plot.index else float("nan")
        ax.text(
            0.02,
            0.98,
            f"norm ST: {x}={st1:.2f}, {y}={st2:.2f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=8.0,
            color=COLORS["muted"],
        )
    fig.suptitle(
        "Top-two driver maps: sampled runs colored by endpoint",
        fontsize=13.2,
        fontweight="semibold",
    )
    save_figure(fig, out_dir, "figure_4_top2_response_maps")


def collect_group_trajectories(
    run_dirs: Iterable[Path], field: str
) -> tuple[np.ndarray, np.ndarray]:
    xs: list[np.ndarray] = []
    ys: list[np.ndarray] = []
    for rdir in run_dirs:
        ts = rdir / "timeseries.csv"
        if not ts.exists():
            continue
        df = pd.read_csv(ts, usecols=["day", field])
        xs.append(df["day"].to_numpy(float))
        ys.append(df[field].to_numpy(float))
    if not xs:
        return np.array([], dtype=float), np.empty((0, 0), dtype=float)
    xref = xs[0]
    bad = [
        i
        for i, x in enumerate(xs)
        if not (len(x) == len(xref) and np.allclose(x, xref))
    ]
    if bad:
        raise SystemExit(
            f"Mismatched time grids detected in trajectory group for field '{field}'. "
            f"Indices failing alignment: {bad}"
        )
    arr = np.vstack(ys)
    return xref, arr


def fig5_trajectory_bands(
    endpoint_df: pd.DataFrame, sobol_df: pd.DataFrame, runs_dir: Path, out_dir: Path
) -> None:
    fig, axes = plt.subplots(
        1, len(ENDPOINTS), figsize=(15.4, 5.1), sharex=True, constrained_layout=True
    )
    for ax, ep in zip(axes, [e["key"] for e in ENDPOINTS]):
        top_param = (
            sobol_df[sobol_df["endpoint"] == ep]
            .sort_values("ST_nonneg", ascending=False)
            .iloc[0]["parameter"]
        )
        q25 = float(endpoint_df[top_param].quantile(0.25))
        q75 = float(endpoint_df[top_param].quantile(0.75))
        low = endpoint_df[endpoint_df[top_param] <= q25]
        high = endpoint_df[endpoint_df[top_param] >= q75]
        if low.empty or high.empty:
            ax.text(
                0.5,
                0.5,
                f"Empty quartile group\n({top_param})",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=9.6,
            )
            ax.set_title(endpoint_title(ep))
            continue

        low_dirs = [runs_dir / Path(str(p)).name for p in low["run_dir"]]
        high_dirs = [runs_dir / Path(str(p)).name for p in high["run_dir"]]
        field = endpoint_field(ep)

        x_low, y_low = collect_group_trajectories(low_dirs, field)
        x_high, y_high = collect_group_trajectories(high_dirs, field)
        if y_low.size == 0 or y_high.size == 0:
            ax.text(
                0.5,
                0.5,
                "Insufficient runs",
                transform=ax.transAxes,
                ha="center",
                va="center",
                fontsize=10,
            )
            ax.set_title(endpoint_title(ep))
            continue

        low_med = np.median(y_low, axis=0)
        low_q1 = np.quantile(y_low, 0.25, axis=0)
        low_q3 = np.quantile(y_low, 0.75, axis=0)
        hi_med = np.median(y_high, axis=0)
        hi_q1 = np.quantile(y_high, 0.25, axis=0)
        hi_q3 = np.quantile(y_high, 0.75, axis=0)

        ax.fill_between(
            x_low, low_q1, low_q3, color=COLORS["low"], alpha=0.28, linewidth=0
        )
        ax.plot(
            x_low,
            low_med,
            color=COLORS["low"],
            lw=2.0,
            label=f"{top_param} low quartile",
        )
        ax.fill_between(
            x_high, hi_q1, hi_q3, color=COLORS["high"], alpha=0.24, linewidth=0
        )
        ax.plot(
            x_high,
            hi_med,
            color=COLORS["high"],
            lw=2.0,
            label=f"{top_param} high quartile",
        )
        ax.set_title(endpoint_title(ep))
        ax.set_xlabel("Day")
        ax.set_ylabel(endpoint_short_ylabel(ep))
        style_axes(ax, "both", 0.25)
        ax.legend(frameon=False, fontsize=7.8, loc="upper left")

    fig.suptitle(
        "Time-course shift by dominant sensitivity driver",
        fontsize=13.2,
        fontweight="semibold",
    )
    save_figure(fig, out_dir, "figure_5_trajectory_quartiles")


def write_caption_file(out_dir: Path, *, clip_for_display: bool) -> None:
    caption = (
        "Figure set for sanity_check6 global sensitivity analysis. "
        "Figure 1 shows total Sobol effects as endpoint-normalized ST (0-1 per endpoint); "
        "Figure 2 compares S1 vs ST with confidence intervals; negative S1 values and CI overflow are estimator-uncertainty artifacts. "
        "Figure 3 ranks parameters by mean ST; Figure 4 maps endpoint "
        "response across the top two endpoint-specific drivers as sampled runs colored by endpoint value; Figure 5 shows endpoint time-course "
        "bands (median and interquartile range) for low vs high quartiles of each endpoint's top driver."
    )
    if clip_for_display:
        caption += (
            " Display clipping is enabled via --clip-for-display for readability only."
        )
    else:
        caption += " No display clipping was applied."
    (out_dir / "figure_captions.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = normalize_path(args.input)
    runs_dir = normalize_path(args.runs)
    out_dir = (
        normalize_path(args.out)
        if args.out.strip()
        else (input_dir / "figure_extract_6_pub")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_df, sobol_df, rank_df = load_tables(input_dir)
    endpoint_df = endpoint_df[endpoint_df["success"].astype(bool)].copy()
    validate_sobol_table(sobol_df)
    sobol_plot_df = prepare_sobol_for_plot(
        sobol_df, clip_for_display=args.clip_for_display
    )

    fig1_total_effect_heatmap(sobol_plot_df, out_dir)
    fig2_s1_vs_st(sobol_plot_df, out_dir, clip_for_display=args.clip_for_display)
    fig3_parameter_ranking(rank_df, out_dir)
    fig4_top2_response_maps(endpoint_df, sobol_plot_df, sobol_df, out_dir)
    fig5_trajectory_bands(endpoint_df, sobol_plot_df, runs_dir, out_dir)
    write_caption_file(out_dir, clip_for_display=args.clip_for_display)

    print(f"[sanity6_figure_extract] input: {input_dir}")
    print(f"[sanity6_figure_extract] runs: {runs_dir}")
    print(f"[sanity6_figure_extract] wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
