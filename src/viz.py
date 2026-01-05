from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb, PowerNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .agent import CellState

__all__ = [
    "plot_summary",
    "export_spatial_for_day",
    "export_spatial_hourly",
    "export_overlay_collagen_ha",
    "export_overlay_collagen_ha_from_npy",
    "export_collagen_ha_snapshot",
]

COLORS = {
    "preob": "#0072B2",  # colorblind-safe blue
    "ob": "#D55E00",  # colorblind-safe vermillion
    "dead_preob": "#E69F00",
    "dead_ob": "#CC79A7",
    "text": "#111111",
    "bg": "#ffffff",
    "grid": "#eaeaea",
}
PLOT_STYLE = {
    "title_size": 12,
    "label_size": 10,
    "tick_size": 9,
    "line_width": 2.2,
    "grid_alpha": 0.18,
}

FIGURE_SIZE_SPATIAL = (8, 7)  # Shared size to keep spatial and collagen PDFs matching.

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.titleweight": "semibold",
        "axes.titlesize": PLOT_STYLE["title_size"],
        "axes.labelsize": PLOT_STYLE["label_size"],
        "xtick.labelsize": PLOT_STYLE["tick_size"],
        "ytick.labelsize": PLOT_STYLE["tick_size"],
        "legend.fontsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _set_white(ax):
    ax.set_facecolor(COLORS["bg"])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")
        spine.set_linewidth(0.8)
    ax.tick_params(direction="out", length=4, width=0.8, colors="#222222")
    ax.grid(alpha=PLOT_STYLE["grid_alpha"], linestyle="--", linewidth=0.7)


def _limit_days(df: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if hours is None or hours <= 0:
        return df
    max_day = float(hours) / 24.0
    return df[df["day"] <= max_day].copy()

def _normalize_cumulative(series: np.ndarray, denom: float) -> np.ndarray:
    if denom <= 0.0:
        return series
    return series / denom


def _format_xlabel(ax: plt.Axes, days: np.ndarray) -> None:
    if days.size == 0:
        ax.set_xlabel("Day")
        return
    max_day = float(np.max(days))
    max_hours = max_day * 24.0
    ax.set_xlabel(f"Day (0–{int(round(max_day))}; {int(round(max_hours))} h)")
    # force integer day ticks including last day
    end_tick = int(np.ceil(max_day))
    if end_tick <= 0:
        ax.set_xticks([0])
        return
    step = max(1, int(np.ceil(end_tick / 7)))
    ticks = list(range(0, end_tick + 1, step))
    if ticks[-1] != end_tick:
        ticks.append(end_tick)
    ax.set_xticks(ticks)


def _save_figure(fig: plt.Figure, outdir: Path, basename: str) -> None:
    png_path = outdir / f"{basename}.png"
    pdf_dir = outdir / "figures_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = pdf_dir / f"{basename}.pdf"
    fig.savefig(png_path, dpi=100, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")

def _plot_cumulative_components(
    df: pd.DataFrame,
    outdir: Path,
    *,
    base_col: str,
    ob_col: str,
    total_col: str,
    plot_hours: int | None,
    filename: str,
    title: str,
    ylabel: str,
) -> None:
    df_plot = _limit_days(df, plot_hours)
    if df_plot.empty:
        return
    days_plot = df_plot["day"].to_numpy()
    base_vals = df_plot[base_col].to_numpy()
    ob_vals = df_plot[ob_col].to_numpy()
    total_vals = df_plot[total_col].to_numpy()
    denom = float(total_vals[-1]) if total_vals.size else 0.0

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(
        days_plot,
        _normalize_cumulative(base_vals, denom),
        label="preOB contribution",
        lw=PLOT_STYLE["line_width"],
        color="#0ea5e9",
    )
    ax.plot(
        days_plot,
        _normalize_cumulative(ob_vals, denom),
        label="OB contribution",
        lw=PLOT_STYLE["line_width"],
        color="#f97316",
    )
    ax.plot(
        days_plot,
        _normalize_cumulative(total_vals, denom),
        label="Total",
        lw=PLOT_STYLE["line_width"] + 0.3,
        ls="--",
        color="#111827",
    )
    if total_vals.size:
        ax.plot(
            days_plot[-1],
            _normalize_cumulative(total_vals[-1], denom),
            marker="o",
            markersize=6,
            markerfacecolor="#111827",
            markeredgecolor="#111827",
            linestyle="None",
        )
    if days_plot.size:
        ax.set_xlim(days_plot[0], days_plot[-1])
        ticks = list(ax.get_xticks())
        last_day = float(days_plot[-1])
        if all(abs(t - last_day) > 1e-6 for t in ticks):
            ticks.append(last_day)
            ticks = sorted(ticks)
            ax.set_xticks(ticks)
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(
        frameon=False,
        ncol=1,
        loc="upper right",
        bbox_to_anchor=(0.99, 0.99),
    )
    _format_xlabel(ax, days_plot)
    fig.tight_layout(pad=0.6)
    _save_figure(fig, outdir, Path(filename).stem)
    plt.close(fig)


def plot_summary(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    alp_plot_hours: int | None = None,
    collagen_plot_hours: int | None = None,
    mineral_plot_hours: int | None = None,
) -> None:
    outdir = Path(outdir)
    img_dir = outdir / "collagen_ha"
    img_dir.mkdir(parents=True, exist_ok=True)
    days = df["day"].to_numpy()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(
        days,
        df["n_preob"],
        label="preOB",
        color=COLORS["preob"],
        lw=PLOT_STYLE["line_width"],
    )
    ax.plot(
        days,
        df["n_osteoblast"],
        label="Osteoblast",
        color=COLORS["ob"],
        lw=PLOT_STYLE["line_width"],
    )
    ax.plot(
        days,
        df["n_total"],
        label="Total",
        color="#333",
        lw=1.6,
        ls="--",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Cells")
    ax.set_title("Cell states over time", pad=10)
    ax.legend(frameon=False, ncol=2, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    _format_xlabel(ax, days)
    fig.tight_layout(pad=0.6)
    _save_figure(fig, outdir, "01_states_counts")
    plt.close(fig)

    tmp = df.copy()
    if "dead_total_today" not in tmp.columns:
        tmp["dead_total_today"] = tmp["dead_preob_today"] + tmp["dead_ob_today"]
    tmp["dead_preob_cum"] = tmp["dead_preob_today"].cumsum()
    tmp["dead_ob_cum"] = tmp["dead_ob_today"].cumsum()
    tmp["dead_total_cum"] = tmp["dead_total_today"].cumsum()

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(
        days,
        tmp["dead_preob_cum"],
        label="preOB→Dead (cum)",
        color=COLORS["dead_preob"],
        lw=PLOT_STYLE["line_width"],
    )
    ax.plot(
        days,
        tmp["dead_ob_cum"],
        label="OB→Dead (cum)",
        color=COLORS["dead_ob"],
        lw=PLOT_STYLE["line_width"],
    )
    ax.plot(
        days,
        tmp["dead_total_cum"],
        label="All deaths (cum)",
        color="#444",
        lw=1.6,
        ls="--",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Cumulative events", pad=10)
    ax.legend(frameon=False, loc="upper right", bbox_to_anchor=(0.99, 0.99), ncol=1)
    _format_xlabel(ax, days)
    fig.tight_layout(pad=0.6)
    _save_figure(fig, outdir, "02_events_cumulative")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.bar(
        days,
        df["committed_today"],
        label="Committed (today)",
        color="#fdae61",
        edgecolor="#9a3412",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.bar(
        days,
        df["dead_preob_today"],
        bottom=df["committed_today"],
        label="Died (preOB)",
        color=COLORS["dead_preob"],
        edgecolor="#b45309",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.bar(
        days,
        df["dead_ob_today"],
        bottom=df["committed_today"] + df["dead_preob_today"],
        label="Died (osteoblast)",
        color=COLORS["dead_ob"],
        edgecolor="#9d174d",
        linewidth=0.5,
        alpha=0.9,
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.set_title("Daily events", pad=10)
    ax.legend(frameon=False, ncol=1, loc="upper right", bbox_to_anchor=(0.99, 0.99))
    _format_xlabel(ax, days)
    fig.tight_layout(pad=0.6)
    _save_figure(fig, outdir, "04_events_daily")
    plt.close(fig)

    if all(
        col in df.columns
        for col in ("alp_preob_total", "alp_ob_total", "alp_total")
    ):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="alp_preob_total",
            ob_col="alp_ob_total",
            total_col="alp_total",
            plot_hours=alp_plot_hours,
            filename="06_alp_total.png",
            title="ALP accumulation (normalized cumulative)",
            ylabel="Fraction of total",
        )

    if all(
        col in df.columns
        for col in ("collagen_preob_total", "collagen_ob_total", "collagen_total")
    ):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="collagen_preob_total",
            ob_col="collagen_ob_total",
            total_col="collagen_total",
            plot_hours=collagen_plot_hours,
            filename="07_collagen_total.png",
            title="Collagen accumulation (normalized cumulative)",
            ylabel="Fraction of total",
        )

    if all(
        col in df.columns
        for col in ("mineral_preob_total", "mineral_ob_total", "ha_total")
    ):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="mineral_preob_total",
            ob_col="mineral_ob_total",
            total_col="ha_total",
            plot_hours=mineral_plot_hours,
            filename="05_minerals_cumulative.png",
            title="Mineral accumulation (normalized cumulative)",
            ylabel="Fraction of total",
        )


def _render_spatial_image(
    img_path: Path,
    title_label: str,
    state_grid: np.ndarray,
    deaths_preob_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    dead_total_cum: int,
) -> None:
    """Shared renderer for spatial cell snapshots (daily or hourly)."""
    h, w = state_grid.shape
    canvas = np.zeros((h, w, 3), dtype=float) + 1.0
    preob_mask = state_grid == CellState.PRE_OB
    ob_mask = state_grid == CellState.OSTEOBLAST
    preob_color = to_rgb(COLORS["preob"])
    ob_color = to_rgb(COLORS["ob"])
    canvas[preob_mask] = preob_color
    canvas[ob_mask] = ob_color
    canvas[deaths_preob_mask.astype(bool)] = to_rgb(COLORS["dead_preob"])
    canvas[deaths_ob_mask.astype(bool)] = to_rgb(COLORS["dead_ob"])

    n_preob = int(preob_mask.sum())
    n_ob = int(ob_mask.sum())
    n_da = int(deaths_preob_mask.sum())
    n_do = int(deaths_ob_mask.sum())

    fig, ax = plt.subplots(figsize=FIGURE_SIZE_SPATIAL)
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"{title_label}  |  Total {n_preob + n_ob} live  |  "
        f"preOB: {n_preob}, osteo: {n_ob}, dead today: {n_da + n_do}, dead cum: {dead_total_cum}",
        loc="center",
        fontsize=12,
        fontweight="bold",
    )
    legend_elems = [
        Patch(facecolor=COLORS["preob"], label=f"preOB ({n_preob})"),
        Patch(facecolor=COLORS["ob"], label=f"Osteoblast ({n_ob})"),
        Patch(facecolor=COLORS["dead_preob"], label=f"Died while preOB ({n_da})"),
        Patch(facecolor=COLORS["dead_ob"], label=f"Died while osteoblast ({n_do})"),
    ]
    legend = ax.legend(
        handles=legend_elems, frameon=True, loc="upper right", ncol=1, fontsize=9
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.6)
    ax.set_aspect("equal")
    fig.subplots_adjust(top=0.88, right=0.95, left=0.05, bottom=0.04)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(img_path, dpi=300)
    plt.close(fig)


def export_spatial_hourly(
    outdir: str | Path,
    day: int,
    hour: int,
    state_grid: np.ndarray,
    deaths_preob_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    dead_total_cum: int,
) -> None:
    target = Path(outdir) / "day_hor" / f"day{day}" / "spatial_cells"
    img_path = target / f"hour_{hour:02d}.pdf"
    title_label = f"Day {day:02d} — Hour {hour:02d}"
    _render_spatial_image(
        img_path,
        title_label,
        state_grid,
        deaths_preob_mask,
        deaths_ob_mask,
        dead_total_cum,
    )


def export_spatial_for_day(
    outdir: str | Path,
    day: int,
    state_grid: np.ndarray,
    deaths_preob_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    dead_total_cum: int,
) -> None:
    target = Path(outdir) / "spatial_cells"
    img_path = target / f"spatial_day_{day:03d}.pdf"
    _render_spatial_image(
        img_path,
        f"Day {day:02d}",
        state_grid,
        deaths_preob_mask,
        deaths_ob_mask,
        dead_total_cum,
    )


def _robust_norm(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
    """Normalize to [0,1] using a robust upper bound (percentile)."""
    if arr.size == 0:
        return arr
    vmax = np.percentile(arr[arr > 0], p) if (arr > 0).any() else 1.0
    vmax = max(vmax, 1e-9)
    out = np.clip(arr / vmax, 0.0, 1.0)
    return out


def _render_collagen_ha_overlay(
    img_path: Path,
    title_label: str,
    col: np.ndarray,
    ha: np.ndarray,
    *,
    n_preob: int | None = None,
    n_ob: int | None = None,
    dead_today: int | None = None,
    nodule_positions: np.ndarray | None = None,
    nodule_cum: int = 0,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if nodule_positions is None:
        nodule_positions = np.empty((0, 2), dtype=float)

    if not col.any() and not ha.any():
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
        _set_white(ax)
        ax.text(
            0.5, 0.5, "No collagen or HA yet", ha="center", va="center", fontsize=14
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{title_label} — No collagen/HA", fontsize=12, pad=10)
        if nodule_cum:
            stats_bits: list[str] = []
            if n_preob is not None and n_ob is not None:
                stats_bits.append(f"preOB: {n_preob}")
                stats_bits.append(f"OB: {n_ob}")
            if dead_today is not None:
                stats_bits.append(f"Dead: {dead_today}")
            stats_bits.append(f"Nodules: {nodule_cum}")
            ax.text(
                0.5,
                0.08,
                " | ".join(stats_bits),
                transform=ax.transAxes,
                ha="center",
                va="bottom",
                fontsize=9,
                color=COLORS["text"],
            )
        fig.savefig(img_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    def _p99(x: np.ndarray) -> float:
        return float(np.percentile(x[x > 0], 99)) if (x > 0).any() else 1.0

    if global_vmax_col is not None:
        vmax_col = max(global_vmax_col, 1e-9)
    else:
        vmax_col = max(_p99(col), 1e-9)
    if global_vmax_ha is not None:
        vmax_ha = max(global_vmax_ha, 1e-9)
    else:
        vmax_ha = max(_p99(ha), 1e-9)

    col_norm = PowerNorm(gamma=0.7, vmin=0.0, vmax=vmax_col)
    ha_norm = PowerNorm(gamma=0.7, vmin=0.0, vmax=vmax_ha)

    fig = plt.figure(figsize=FIGURE_SIZE_SPATIAL)
    ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
    _set_white(ax)
    ax.imshow(
        np.ones((*col.shape, 3)) * 0.97,
        origin="lower",
        interpolation="nearest",
    )
    im_col = ax.imshow(
        col,
        cmap="YlGn",
        norm=col_norm,
        origin="lower",
        interpolation="nearest",
        alpha=0.9,
    )
    im_ha = ax.imshow(
        ha,
        cmap="PuRd",
        norm=ha_norm,
        origin="lower",
        interpolation="nearest",
        alpha=0.65,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(
        f"{title_label} — Collagen (yellow-green), HA (magenta/purple)",
        fontsize=12,
        y=0.97,
    )
    hx: list[float] = []
    hy: list[float] = []
    if nodule_positions.size:
        hx.extend(nodule_positions[:, 0].tolist())
        hy.extend(nodule_positions[:, 1].tolist())
    nodule_handle = None
    if hx:
        nodule_handle = ax.scatter(
            hx,
            hy,
            marker="*",
            s=45,
            c="#8b1d8b",
            edgecolors="#ffffff",
            linewidths=0.5,
            alpha=0.85,
            zorder=5,
        )

    stats_bits: list[str] = []
    if n_preob is not None and n_ob is not None:
        stats_bits.append(f"preOB: {n_preob}")
        stats_bits.append(f"OB: {n_ob}")
    if dead_today is not None:
        stats_bits.append(f"Dead: {dead_today}")
    stats_bits.append(f"Nodules: {nodule_cum}")
    ax.text(
        0.5,
        1.05,
        " | ".join(stats_bits),
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=10,
        color=COLORS["text"],
    )

    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="2.5%", pad=0.02)
    cax_right = divider.append_axes("right", size="2.5%", pad=0.02)
    cbar_left = fig.colorbar(im_col, cax=cax_left, orientation="vertical")
    cbar_left.ax.yaxis.set_label_position("left")
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_left.set_label("Collagen-I (a.u.)", fontsize=9)
    cbar_left.ax.tick_params(labelsize=8, colors="black")
    cbar_right = fig.colorbar(im_ha, cax=cax_right, orientation="vertical")
    cbar_right.set_label("Hydroxyapatite (a.u.)", fontsize=9)
    cbar_right.ax.tick_params(labelsize=8, colors="black")
    legend_handle = None
    if nodule_handle is not None:
        legend_handle = matplotlib.lines.Line2D(
            [],
            [],
            marker="*",
            linestyle="None",
            markersize=6,
            color="#8b1d8b",
            label="Nodule",
        )
    if legend_handle is not None:
        legend = ax.legend(
            handles=[legend_handle], loc="lower left", frameon=True, fontsize=8
        )
        legend.get_frame().set_edgecolor("#333333")
        legend.get_frame().set_linewidth(0.6)

    fig.savefig(img_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def export_overlay_collagen_ha(
    img_path: str | Path,
    title_label: str,
    col: np.ndarray,
    ha: np.ndarray,
    *,
    n_preob: int | None = None,
    n_ob: int | None = None,
    dead_today: int | None = None,
    nodule_positions: np.ndarray | None = None,
    nodule_cum: int = 0,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    _render_collagen_ha_overlay(
        Path(img_path),
        title_label,
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        nodule_positions=nodule_positions,
        nodule_cum=nodule_cum,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
    )


def export_collagen_ha_snapshot(
    outdir: str | Path,
    day: int,
    hour: int,
    col: np.ndarray,
    ha: np.ndarray,
    state_grid: np.ndarray,
    deaths_preob_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    *,
    nodule_positions: np.ndarray | None = None,
    nodule_cum: int = 0,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    target = Path(outdir) / "day_hor" / f"day{day}" / "collagen_ha"
    img_path = target / f"hour_{hour:02d}.pdf"
    n_preob = int((state_grid == CellState.PRE_OB).sum())
    n_ob = int((state_grid == CellState.OSTEOBLAST).sum())
    dead_today = int(deaths_preob_mask.sum() + deaths_ob_mask.sum())
    export_overlay_collagen_ha(
        img_path,
        f"Day {day:02d} — Hour {hour:02d}",
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        nodule_positions=nodule_positions,
        nodule_cum=nodule_cum,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
    )


def export_overlay_collagen_ha_from_npy(
    grids_dir: str | Path,
    outdir: str | Path,
    day: int,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    """
    Overlay using saved per-day arrays from /grids:
      - collagen_day_XXX.npy
      - ha_day_XXX.npy
    """
    grids_dir = Path(grids_dir)
    outdir = Path(outdir)
    img_dir = outdir / "collagen_ha"
    img_dir.mkdir(parents=True, exist_ok=True)
    col = np.load(grids_dir / f"collagen_day_{day:03d}.npy")
    ha = np.load(grids_dir / f"ha_day_{day:03d}.npy")
    state_path = grids_dir / f"state_day_{day:03d}.npy"
    deaths_path = grids_dir / f"deaths_day_{day:03d}.npy"
    n_preob = n_ob = dead_today = None
    if state_path.exists():
        state_grid = np.load(state_path)
        n_preob = int((state_grid == CellState.PRE_OB).sum())
        n_ob = int((state_grid == CellState.OSTEOBLAST).sum())
    if deaths_path.exists():
        deaths_grid = np.load(deaths_path)
        dead_today = int(np.asarray(deaths_grid).sum())
    nodules_path = grids_dir / f"nodules_day_{day:03d}.npz"
    nodule_positions = np.empty((0, 2), dtype=float)
    nodule_cum = 0

    if nodules_path.exists():
        nodules_data = np.load(nodules_path, allow_pickle=False)
        positions = np.asarray(nodules_data["positions"], dtype=float)
        if positions.size:
            nodule_positions = positions.reshape(-1, 2)
        nodule_cum = int(nodules_data["count"])

    export_overlay_collagen_ha(
        img_dir / f"collagen_ha_day_{day:03d}.pdf",
        f"Day {day:02d}",
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        nodule_positions=nodule_positions,
        nodule_cum=nodule_cum,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
    )
