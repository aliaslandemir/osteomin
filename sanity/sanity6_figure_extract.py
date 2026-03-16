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
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable

SCRIPT_DIR = Path(__file__).resolve().parent

ENDPOINTS = [
    {
        "key": "alp_240h",
        "field": "alp_total",
        "title": "ALP",
        "color": "#c75a1b",
    },
    {
        "key": "collagen_336h",
        "field": "collagen_total",
        "title": "Collagen",
        "color": "#1b9e77",
    },
    {
        "key": "ars_672h",
        "field": "ha_total",
        "title": "HAp",
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

PARAM_LABELS = {
    "alp_rb": "ALP basal rate",
    "alp_ro": "ALP osteogenic rate",
    "alp_a_d": "ALP Dex ampl.",
    "alp_b_d": "ALP Dex curve",
    "alp_a_k": "ALP MK-4 ampl.",
    "alp_b_k": "ALP MK-4 curve",
    "collagen_rb": "Collagen basal rate",
    "collagen_ro": "Collagen osteogenic rate",
    "collagen_a_d": "Collagen Dex ampl.",
    "collagen_b_d": "Collagen Dex curve",
    "collagen_a_k": "Collagen MK-4 ampl.",
    "collagen_b_k": "Collagen MK-4 curve",
    "ha_ro": "HAp osteogenic rate",
    "ha_a_d": "HAp Dex ampl.",
    "ha_b_d": "HAp Dex curve",
    "ha_a_k": "HAp MK-4 ampl.",
    "ha_b_k": "HAp MK-4 curve",
    "ha_tau": "HAp tau",
    "ha_hill_n": "HAp Hill n",
    "commit_k": "Commitment k",
    "commit_k_growth": "Commitment k growth",
    "apoptosis_prob": "Apoptosis prob",
    "move_prob": "Motility prob",
    "aa0": "Ascorbate A0",
    "aa_half_life": "Ascorbate half-life",
    "collagen_aa_km": "Collagen AA Km",
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
        description="Figure extraction for sanity_check6 sensitivity outputs.",
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
        default="",
        help="runs folder containing sXXXXX/timeseries.csv (default: <input>/runs)",
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
    if txt.startswith("\\\\wsl$\\") or txt.startswith("\\\\wsl.localhost\\"):
        # Map Windows UNC WSL paths like:
        # \\wsl$\Ubuntu-24.04\home\user\repo
        # \\wsl.localhost\Ubuntu-24.04\home\user\repo
        slash = txt.replace("\\", "/")
        parts = slash.split("/", 4)
        if len(parts) >= 5:
            return Path("/" + parts[4])
    return Path(txt)


def style_axes(ax: plt.Axes, grid_axis: str | None = "y", alpha: float = 0.45) -> None:
    ax.set_facecolor("white")
    ax.tick_params(direction="out", length=3.8, width=0.8)
    if grid_axis is not None:
        ax.grid(axis=grid_axis, color=COLORS["grid"], linewidth=0.8, alpha=alpha)


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, *, dpi: int = 500) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.png", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    fig.savefig(out_dir / f"{stem}.pdf", dpi=dpi, bbox_inches="tight", pad_inches=0.1)
    plt.close(fig)


def pretty_param(name: str) -> str:
    return PARAM_LABELS.get(name, name.replace("_", " "))


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=12.5,
        fontweight="bold",
        color=COLORS["ink"],
    )


def load_tables(input_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    endpoint = pd.read_csv(input_dir / "endpoint_results.csv")
    sobol = pd.read_csv(input_dir / "sobol_indices_all_endpoints.csv")
    rank = pd.read_csv(input_dir / "parameter_ranking.csv")
    return endpoint, sobol, rank


def load_s2_matrices(input_dir: Path) -> dict[str, pd.DataFrame]:
    mats: dict[str, pd.DataFrame] = {}
    for ep in [e["key"] for e in ENDPOINTS]:
        path = input_dir / f"sobol_{ep}_S2_matrix.csv"
        if not path.exists():
            continue
        mats[ep] = pd.read_csv(path, index_col=0)
    return mats


def prepare_sobol_for_plot(
    sobol_df: pd.DataFrame, *, clip_for_display: bool
) -> pd.DataFrame:
    df = sobol_df.copy()
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


def fig2_parameter_order(sobol_df: pd.DataFrame) -> list[str]:
    exclude_suffixes = ("_b_d", "_b_k")
    exclude_exact = {"alp_a_d", "alp_a_k", "collagen_a_d", "collagen_a_k"}
    params = endpoint_order(sobol_df)
    return [
        p
        for p in params
        if (not any(p.endswith(sfx) for sfx in exclude_suffixes)) and (p not in exclude_exact)
    ]


def endpoint_title(key: str) -> str:
    for e in ENDPOINTS:
        if e["key"] == key:
            return str(e["title"])
    return key


def endpoint_panel_title(key: str) -> str:
    mapping = {
        "alp_240h": "ALP activity endpoint (Day 10)",
        "collagen_336h": "Collagen endpoint (Day 14)",
        "ars_672h": "HAp proxy endpoint (Day 28)",
    }
    return mapping.get(key, endpoint_title(key))


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
        "ars_672h": "HAp",
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
    ax.set_yticklabels([pretty_param(p) for p in params])
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
    add_panel_label(ax, "A")
    save_figure(fig, out_dir, "figure_1_total_effect_heatmap")


def fig2_s1_vs_st(
    sobol_df: pd.DataFrame, out_dir: Path, *, clip_for_display: bool
) -> None:
    legend_handles = None
    global_x_min = 0.0
    global_x_max = 1.0
    if clip_for_display:
        global_x_min, global_x_max = -0.2, 1.2
    else:
        x1 = sobol_df["S1_plot"] - sobol_df["S1_conf_plot"]
        x2 = sobol_df["ST_plot"] + sobol_df["ST_conf_plot"]
        global_x_min = min(-0.05, float(np.nanmin(x1)))
        global_x_max = max(1.0, float(np.nanmax(x2)))
        span = global_x_max - global_x_min
        global_x_min -= 0.05 * span
        global_x_max += 0.05 * span
    params = fig2_parameter_order(sobol_df)
    fig_h = max(8.0, 0.30 * len(params) + 1.6)
    fig = plt.figure(figsize=(16.8, fig_h), constrained_layout=False)
    gs = fig.add_gridspec(1, 4, width_ratios=[1.55, 4.25, 4.25, 4.25], wspace=0.07)
    label_ax = fig.add_subplot(gs[0, 0])
    axes = [fig.add_subplot(gs[0, i]) for i in range(1, 4)]
    fig.subplots_adjust(left=0.012, right=0.985, top=0.89, bottom=0.08)

    y = np.arange(len(params))
    label_ax.set_xlim(0.0, 1.0)
    label_ax.set_ylim(len(params) - 0.5, -0.5)
    label_ax.axis("off")
    for i, p in enumerate(params):
        label_ax.text(
            0.99,
            y[i],
            pretty_param(p),
            ha="right",
            va="center",
            fontsize=10.0,
            color=COLORS["ink"],
        )

    for idx, (ax, ep) in enumerate(zip(axes, [e["key"] for e in ENDPOINTS])):
        sub = (
            sobol_df[sobol_df["endpoint"] == ep]
            .set_index("parameter")
            .loc[params]
            .reset_index()
        )
        s1 = sub["S1_plot"].to_numpy(float)
        st = sub["ST_plot"].to_numpy(float)
        s1c = sub["S1_conf_plot"].to_numpy(float)
        stc = sub["ST_conf_plot"].to_numpy(float)
        for row_idx in range(len(params)):
            if row_idx % 2 == 0:
                ax.axhspan(
                    row_idx - 0.5,
                    row_idx + 0.5,
                    color="#f1f5f9",
                    alpha=1.0,
                    zorder=0,
                )
        ax.hlines(y, s1, st, color="#cbd5e1", lw=1.9, alpha=1.0)
        h1 = ax.errorbar(
            s1,
            y,
            xerr=s1c,
            fmt="o",
            color="#111827",
            markersize=6.4,
            markeredgewidth=0.0,
            capsize=2.6,
            elinewidth=1.35,
            label="S1",
        )
        h2 = ax.errorbar(
            st,
            y,
            xerr=stc,
            fmt="s",
            color=endpoint_color(ep),
            markersize=7.1,
            markeredgewidth=0.0,
            capsize=2.6,
            elinewidth=1.35,
            label="ST",
        )
        ax.axvline(0.0, color="#64748b", lw=1.0)
        ax.set_xlim(global_x_min, global_x_max)
        ax.set_ylim(len(params) - 0.5, -0.5)
        ax.set_title(endpoint_panel_title(ep), fontsize=11.8, x=0.53)
        ax.set_xlabel("Sobol sensitivity index")
        ax.set_yticks(y)
        ax.set_yticklabels([""] * len(params))
        style_axes(ax, "x", 0.35)
        if legend_handles is None:
            legend_handles = (h1, h2)
        ax.text(
            -0.06,
            1.01,
            chr(ord("A") + idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.5,
            fontweight="bold",
            color=COLORS["ink"],
        )
    if legend_handles is not None:
        fig.legend(
            legend_handles,
            ["First-order (S1)", "Total-order (ST)"],
            frameon=False,
            loc="upper right",
            bbox_to_anchor=(0.985, 0.997),
            ncol=2,
            fontsize=10.0,
            handlelength=1.4,
            columnspacing=1.6,
        )
    fig.suptitle(
        "First-order vs total-order Sobol indices",
        fontsize=16.0,
        fontweight="semibold",
        y=0.995,
    )
    save_figure(fig, out_dir, "figure_2_s1_vs_st")


def fig6_s1_ploscb_style(sobol_df: pd.DataFrame, out_dir: Path) -> None:
    # PLOS CB style companion panel: first-order effects only (S1 +/- CI),
    # focusing on the strongest parameters for readability in print layouts.
    params = (
        sobol_df.assign(S1_abs=lambda d: d["S1"].abs())
        .groupby("parameter", as_index=False)["S1_abs"]
        .mean()
        .sort_values("S1_abs", ascending=False)["parameter"]
        .tolist()
    )
    top_n = min(12, len(params))
    params = params[:top_n]

    fig, axes = plt.subplots(
        1, len(ENDPOINTS), figsize=(13.8, max(4.8, 0.33 * top_n + 1.6)), sharey=True, constrained_layout=True
    )

    for idx, (ax, ep) in enumerate(zip(axes, [e["key"] for e in ENDPOINTS])):
        sub = (
            sobol_df[sobol_df["endpoint"] == ep]
            .set_index("parameter")
            .loc[params]
            .reset_index()
        )
        y = np.arange(len(params))
        s1 = sub["S1_plot"].to_numpy(float)
        s1c = sub["S1_conf_plot"].to_numpy(float)

        for row_idx in range(len(params)):
            if row_idx % 2 == 0:
                ax.axhspan(row_idx - 0.5, row_idx + 0.5, color="#f8fafc", zorder=0)

        ax.barh(y, s1, color=endpoint_color(ep), alpha=0.92, edgecolor="none", zorder=2)
        ax.errorbar(
            s1,
            y,
            xerr=s1c,
            fmt="none",
            ecolor=COLORS["ink"],
            elinewidth=1.0,
            capsize=2.5,
            zorder=3,
        )
        ax.axvline(0.0, color=COLORS["muted"], lw=1.0, zorder=1)
        ax.set_title(endpoint_panel_title(ep), fontsize=11.2)
        ax.set_xlabel("First-order Sobol index (S1)")
        ax.set_yticks(y)
        if idx == 0:
            ax.set_yticklabels([pretty_param(p) for p in params], fontsize=8.8)
            ax.set_ylabel("Parameter")
        else:
            ax.set_yticklabels([])
        style_axes(ax, "x", 0.30)
        ax.invert_yaxis()
        ax.text(
            -0.08,
            1.01,
            chr(ord("L") + idx),
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color=COLORS["ink"],
        )

    fig.suptitle(
        "Publication-style first-order Sobol sensitivities (top parameters)",
        fontsize=13.6,
        fontweight="semibold",
    )
    save_figure(fig, out_dir, "figure_6_sobol_first_order_ploscb")


def fig7_integrated_sobol_final(
    sobol_plot_df: pd.DataFrame,
    sobol_raw_df: pd.DataFrame,
    s2_mats: dict[str, pd.DataFrame],
    out_dir: Path,
) -> None:
    if not s2_mats:
        return

    # Consistent visual language across all rows.
    s1_color = "#1f3b73"  # dark blue
    st_color = "#d97706"  # orange
    connector_color = "#9ca3af"

    endpoint_name = {
        "alp_240h": "ALP ",
        "collagen_336h": "Collagen ",
        "ars_672h": "HAp ",
    }

    short_alias = {
        "osteo_induction_hours": "induction time",
        "max_div_hours": "max div",
        "min_div_hours": "min div",
        "apoptosis_prob": "apoptosis",
        "ob_apoptosis_prob": "OB apoptosis",
        "aa_half_life": "AA half-life",
        "collagen_aa_km": "collagen AA Km",
        "proliferation_chance": "division chance",
        "collagen_ro": "collagen osteo rate",
        "collagen_rb": "collagen basal rate",
        "alp_ro": "ALP osteo rate",
        "alp_rb": "ALP basal rate",
        "ha_hill_n": "HAp Hill n",
        "ha_tau": "HAp tau",
        "commit_k": "commitment k",
        "commit_delay": "commit delay",
        "move_prob": "motility",
        "aa0": "AA initial",
    }

    def short_label(name: str, max_len: int = 18) -> str:
        lbl = short_alias.get(name, pretty_param(name))
        if len(lbl) <= max_len:
            return lbl
        return lbl[: max_len - 1].rstrip() + "..."

    def short_label_s2(name: str, max_len: int = 12) -> str:
        return short_label(name, max_len=max_len)

    endpoint_keys = [e["key"] for e in ENDPOINTS if e["key"] in s2_mats]
    if not endpoint_keys:
        return

    # Keep the main-effect side concise while letting S2 summarize the
    # strongest interaction-driven terms for each endpoint separately.
    top_n = 8
    s2_top_n = 8
    left_top_by_ep: dict[str, list[str]] = {}
    right_top_by_ep: dict[str, list[str]] = {}
    s2_vmax_by_ep: dict[str, float] = {}
    for ep in endpoint_keys:
        raw_ep = sobol_raw_df[sobol_raw_df["endpoint"] == ep].copy()
        raw_ep["ST_nonneg"] = raw_ep["ST"].clip(lower=0.0)
        top_main = (
            raw_ep.sort_values("ST_nonneg", ascending=False)["parameter"]
            .head(top_n)
            .tolist()
        )
        if top_main:
            left_top_by_ep[ep] = top_main

        s2_df = s2_mats[ep].copy()
        pair_scores: list[tuple[float, str, str]] = []
        cols = list(s2_df.columns)
        idxs = list(s2_df.index)
        for i, rname in enumerate(idxs):
            for j, cname in enumerate(cols):
                if j <= i:
                    continue
                val = s2_df.loc[rname, cname]
                if pd.isna(val):
                    continue
                pair_scores.append((abs(float(val)), str(rname), str(cname)))
        pair_scores.sort(key=lambda x: x[0], reverse=True)

        picked: list[str] = []
        for _, p1, p2 in pair_scores:
            if p1 not in picked:
                picked.append(p1)
            if len(picked) >= s2_top_n:
                break
            if p2 not in picked:
                picked.append(p2)
            if len(picked) >= s2_top_n:
                break
        if picked:
            right_top_by_ep[ep] = picked[:s2_top_n]
    if not left_top_by_ep or not right_top_by_ep:
        return

    # Global scales improve comparability across endpoints.
    x_max = 1.0
    for ep, top in left_top_by_ep.items():
        sub = (
            sobol_plot_df[sobol_plot_df["endpoint"] == ep]
            .set_index("parameter")
            .reindex(top)
        )
        if sub.empty:
            continue
        x_hi = np.nanmax((sub["ST_plot"] + sub["ST_conf_plot"]).to_numpy(float))
        if np.isfinite(x_hi):
            x_max = max(x_max, float(x_hi))
    for ep, top in right_top_by_ep.items():
        s2 = s2_mats[ep].reindex(index=top, columns=top).to_numpy(float)
        finite = np.abs(s2[np.isfinite(s2)])
        if finite.size:
            # Clip the display range for readability while preserving the raw CSV values.
            s2_vmax_by_ep[ep] = max(float(np.nanpercentile(finite, 90)), 0.08)
        else:
            s2_vmax_by_ep[ep] = 0.08

    fig = plt.figure(figsize=(17.2, 13.2), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.92, 1.08],
        wspace=-0.20,
        hspace=0.44,
    )

    # Softer diverging map: keep contrast, avoid near-black/navy and oversaturated reds.
    s2_cmap = LinearSegmentedColormap.from_list(
        "soft_s2",
        ["#4f88b8", "#a9cbe3", "#f7f7f5", "#f6b38d", "#cf5c4a"],
    ).copy()
    s2_cmap.set_bad("#f3f4f6")

    x_right = x_max * 1.02
    panel_letters = ["A", "B", "C", "D", "E", "F"]
    for row, ep in enumerate(endpoint_keys):

        left_top_params = left_top_by_ep[ep]
        right_top_params = right_top_by_ep[ep]

        ax_l = fig.add_subplot(gs[row, 0])
        sub = (
            sobol_plot_df[sobol_plot_df["endpoint"] == ep]
            .set_index("parameter")
            .reindex(left_top_params)
            .reset_index()
        )
        y = np.arange(len(left_top_params))
        s1 = sub["S1_plot"].to_numpy(float)
        st = sub["ST_plot"].to_numpy(float)
        s1c = sub["S1_conf_plot"].to_numpy(float)
        stc = sub["ST_conf_plot"].to_numpy(float)

        ax_l.hlines(y, s1, st, color="#64748b", lw=1.25, zorder=1, alpha=0.95)
        ax_l.errorbar(
            s1,
            y,
            xerr=s1c,
            fmt="o",
            color=s1_color,
            markersize=5.6,
            markerfacecolor=s1_color,
            markeredgecolor=s1_color,
            capsize=2.3,
            elinewidth=1.15,
            label="S1",
            zorder=3,
        )
        ax_l.errorbar(
            st,
            y,
            xerr=stc,
            fmt="s",
            color=st_color,
            markersize=5.7,
            markerfacecolor="white",
            markeredgecolor=st_color,
            markeredgewidth=1.2,
            capsize=2.3,
            elinewidth=1.15,
            label="ST",
            zorder=3,
        )
        ax_l.axvline(0.0, color="#475569", lw=1.1)
        ax_l.set_yticks(y)
        ax_l.set_yticklabels([short_label(p) for p in left_top_params], fontsize=8.0)
        ax_l.set_xlim(-0.04, x_right)
        ax_l.set_xlabel("Sobol index", fontsize=9.8)
        ax_l.set_title(
            f"{endpoint_name.get(ep, endpoint_title(ep))}: main effects",
            fontsize=10.4,
            pad=7.0,
        )
        ax_l.invert_yaxis()
        style_axes(ax_l, "x", 0.18)
        ax_l.tick_params(axis="x", labelsize=8.0)
        if row == 0:
            ax_l.legend(
                frameon=False,
                fontsize=8.2,
                ncol=2,
                loc="lower right",
                handletextpad=0.6,
                columnspacing=1.3,
            )
        ax_l.text(
            -0.12,
            1.04,
            panel_letters[row * 2],
            transform=ax_l.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color=COLORS["ink"],
            clip_on=False,
        )

        ax_r = fig.add_subplot(gs[row, 1])
        s2 = s2_mats[ep].copy()
        s2 = s2.reindex(index=right_top_params, columns=right_top_params)
        disp = s2.to_numpy(float)
        np.fill_diagonal(disp, np.nan)
        s2_vmax = s2_vmax_by_ep.get(ep, 0.08)
        s2_norm = TwoSlopeNorm(vmin=-s2_vmax, vcenter=0.0, vmax=s2_vmax)
        im = ax_r.imshow(
            disp,
            cmap=s2_cmap,
            norm=s2_norm,
            aspect="equal",
            interpolation="nearest",
        )
        # Cell boundaries improve readability in dense labels.
        n = len(right_top_params)
        for k in range(n + 1):
            ax_r.axhline(k - 0.5, color="#edf1f5", lw=0.45, zorder=2)
            ax_r.axvline(k - 0.5, color="#edf1f5", lw=0.45, zorder=2)
        # Annotate only cells that are both finite and among the strongest displayed effects.
        for i in range(n):
            for j in range(n):
                v = disp[i, j]
                if np.isfinite(v) and abs(v) >= max(0.08, 0.22 * s2_vmax):
                    ax_r.text(
                        j,
                        i,
                        f"{v:+.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.4,
                        color="#1f2937",
                        zorder=3,
                    )
        ax_r.set_xticks(np.arange(len(right_top_params)))
        ax_r.set_yticks(np.arange(len(right_top_params)))
        ax_r.set_xticklabels(
            [short_label_s2(p, max_len=10) for p in right_top_params],
            rotation=35,
            ha="right",
            fontsize=6.4,
        )
        ax_r.set_yticklabels([short_label_s2(p, max_len=10) for p in right_top_params], fontsize=6.4)
        ax_r.set_facecolor("#f8fafc")
        ax_r.set_title(
            f"{endpoint_name.get(ep, endpoint_title(ep))}: second-order effects (S2)",
            fontsize=9.8,
            pad=5.0,
        )
        ax_r.set_xlabel("")
        ax_r.set_ylabel("")
        style_axes(ax_r, None, 0.0)
        ax_r.tick_params(axis="x", labelsize=6.4, pad=1.2)
        ax_r.tick_params(axis="y", labelsize=6.4)
        ax_r.text(
            -0.19,
            1.02,
            panel_letters[row * 2 + 1],
            transform=ax_r.transAxes,
            ha="left",
            va="bottom",
            fontsize=12.0,
            fontweight="bold",
            color=COLORS["ink"],
            clip_on=False,
        )

        divider = make_axes_locatable(ax_r)
        cax = divider.append_axes("right", size="3.8%", pad=0.025)
        cbar = fig.colorbar(im, cax=cax)
        cbar.set_label("S2", fontsize=8.4)
        cbar.ax.tick_params(labelsize=7.0)


    fig.suptitle(
        "Integrated Sobol Sensitivity",
        fontsize=14.0,
        fontweight="semibold",
        x=0.4,
        ha="center",
        y=0.985,
    )
    fig.subplots_adjust(left=0.105, right=0.955, top=0.93, bottom=0.085)
    save_figure(fig, out_dir, "figure_7_sobol_integrated_final", dpi=600)


def export_parameter_label_table(sobol_df: pd.DataFrame, out_dir: Path) -> None:
    params = fig2_parameter_order(sobol_df)
    table = pd.DataFrame(
        {
            "parameter_code": params,
            "article_label": [pretty_param(p) for p in params],
        }
    )
    table.to_csv(out_dir / "figure_2_parameter_labels.csv", index=False)


def fig3_parameter_ranking(rank_df: pd.DataFrame, out_dir: Path) -> None:
    rank = rank_df.sort_values("mean_ST_across_endpoints", ascending=True)
    labels = [pretty_param(str(p)) for p in rank["parameter"]]
    ypos = np.arange(len(rank))
    fig, ax = plt.subplots(
        figsize=(7.0, max(4.1, 0.32 * len(rank) + 1.2)), constrained_layout=True
    )
    bars = ax.barh(
        ypos,
        rank["mean_ST_across_endpoints"],
        color="#2563eb",
        alpha=0.9,
        edgecolor="none",
    )
    top3 = set(rank.tail(3)["parameter"].astype(str).tolist())
    for b, p in zip(bars, rank["parameter"]):
        if str(p) in top3:
            b.set_color(COLORS["highlight"])
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels)
    for b in bars:
        w = float(b.get_width())
        ax.text(w + 0.01, b.get_y() + b.get_height() * 0.5, f"{w:.2f}", va="center", ha="left", fontsize=8.0, color=COLORS["muted"])
    ax.set_xlabel("Mean ST across endpoints")
    ax.set_title("Global parameter importance ranking")
    style_axes(ax, "x", 0.35)
    add_panel_label(ax, "E")
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
        ax.set_title(f"{endpoint_title(ep)}\n(top pair: {pretty_param(x)}, {pretty_param(y)})")
        ax.set_xlabel(pretty_param(x))
        ax.set_ylabel(pretty_param(y))
        style_axes(ax, "both", 0.25)
        cbar = fig.colorbar(p, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(endpoint_title(ep))
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
        add_panel_label(ax, chr(ord("F") + [e["key"] for e in ENDPOINTS].index(ep)))
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
            label=f"{pretty_param(top_param)} low quartile (n={len(y_low)})",
        )
        ax.fill_between(
            x_high, hi_q1, hi_q3, color=COLORS["high"], alpha=0.24, linewidth=0
        )
        ax.plot(
            x_high,
            hi_med,
            color=COLORS["high"],
            lw=2.0,
            label=f"{pretty_param(top_param)} high quartile (n={len(y_high)})",
        )
        ax.set_title(endpoint_title(ep))
        ax.set_xlabel("Day")
        ax.set_ylabel(endpoint_short_ylabel(ep))
        style_axes(ax, "both", 0.25)
        ax.legend(frameon=False, fontsize=7.8, loc="upper left")
        add_panel_label(ax, chr(ord("I") + [e["key"] for e in ENDPOINTS].index(ep)))

    fig.suptitle(
        "Time-course shift by dominant sensitivity driver",
        fontsize=13.2,
        fontweight="semibold",
    )
    save_figure(fig, out_dir, "figure_5_trajectory_quartiles")


def write_caption_file(out_dir: Path, *, clip_for_display: bool) -> None:
    caption = (
        "Figure outputs for sanity_check6 Sobol sensitivity analysis. "
        "Figure 1: endpoint-normalized total-order indices (ST). "
        "Figure 2: first-order (S1) and total-order (ST) indices with confidence intervals "
        "for selected parameters (curvature terms and ALP/Collagen response amplitudes omitted). "
        "Figure 3: mean ST ranking across endpoints. "
        "Figure 4: endpoint values plotted against the top two endpoint-specific drivers. "
        "Figure 5: endpoint trajectories grouped by low and high quartiles of each endpoint's top driver. "
        "Figure 6: publication-style first-order Sobol panel (S1 +/- CI) for top parameters. "
        "Figure 7: integrated final Sobol panel combining S1, ST, and second-order interactions (S2)."
    )
    if clip_for_display:
        caption += " Display clipping was enabled."
    else:
        caption += " Display clipping was not enabled."
    (out_dir / "figure_captions.txt").write_text(caption + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    input_dir = normalize_path(args.input)
    runs_dir = normalize_path(args.runs) if args.runs.strip() else (input_dir / "runs")
    out_dir = (
        normalize_path(args.out)
        if args.out.strip()
        else (input_dir / "figure_extract_6_pub")
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    endpoint_df, sobol_df, rank_df = load_tables(input_dir)
    s2_mats = load_s2_matrices(input_dir)
    endpoint_df = endpoint_df[endpoint_df["success"].astype(bool)].copy()
    validate_sobol_table(sobol_df)
    sobol_plot_df = prepare_sobol_for_plot(
        sobol_df, clip_for_display=args.clip_for_display
    )

    fig1_total_effect_heatmap(sobol_plot_df, out_dir)
    fig2_s1_vs_st(sobol_plot_df, out_dir, clip_for_display=args.clip_for_display)
    export_parameter_label_table(sobol_plot_df, out_dir)
    fig3_parameter_ranking(rank_df, out_dir)
    fig4_top2_response_maps(endpoint_df, sobol_plot_df, sobol_df, out_dir)
    fig5_trajectory_bands(endpoint_df, sobol_plot_df, runs_dir, out_dir)
    fig6_s1_ploscb_style(sobol_plot_df, out_dir)
    fig7_integrated_sobol_final(sobol_plot_df, sobol_df, s2_mats, out_dir)
    write_caption_file(out_dir, clip_for_display=args.clip_for_display)

    print(f"[sanity6_figure_extract] input: {input_dir}")
    print(f"[sanity6_figure_extract] runs: {runs_dir}")
    print(f"[sanity6_figure_extract] wrote figures to: {out_dir}")


if __name__ == "__main__":
    main()
