from __future__ import annotations
from pathlib import Path
from functools import lru_cache
import numpy as np
import pandas as pd
import matplotlib
from scipy import ndimage as ndi
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb, PowerNorm
from matplotlib.cm import ScalarMappable
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .agent import CellState

__all__ = [
    "plot_summary",
    "detect_nodules",
    "export_spatial_for_day",
    "export_spatial_hourly",
    "export_overlay_collagen_ha",
    "export_overlay_collagen_ha_from_npy",
    "export_collagen_ha_snapshot",
]

COLORS = {
    "preob": "#0072B2",
    "ob": "#D55E00",
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

FIGURE_SIZE_SPATIAL = (8, 7)

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


def detect_nodules(
    state_grid: np.ndarray,
    ha: np.ndarray,
    *,
    connectivity: str = "8n",
    boundary_mode: str = "clamp",
) -> tuple[list[tuple[float, float]], np.ndarray]:
    """
    Detect nodule-core candidates where center + required neighbors are all
    osteoblast and mineralized (HA > 0), then return one center per connected
    component under the selected neighborhood connectivity.
    Returns:
      - marker centers (x, y in pixel coords)
      - a bool mask of candidate nodule-core cells
    """
    conn = str(connectivity).lower().strip()
    bmode = str(boundary_mode).lower().strip()
    ob_ha = (state_grid == CellState.OSTEOBLAST) & (ha > 0.0)
    structure, offsets = _kernel_and_offsets(conn)
    if bmode == "periodic":
        neighbor_count = ndi.convolve(
            ob_ha.astype(np.uint8), structure, mode="wrap"
        )
        required = np.full_like(neighbor_count, int(structure.sum()), dtype=np.int32)
    else:
        neighbor_count = ndi.convolve(
            ob_ha.astype(np.uint8), structure, mode="constant", cval=0
        )
        valid = np.ones_like(ob_ha, dtype=np.uint8)
        required = ndi.convolve(valid, structure, mode="constant", cval=0).astype(np.int32)
    candidate_core = ob_ha & (neighbor_count == required)

    centers = _component_centers(candidate_core, offsets, bmode)
    return centers, candidate_core


@lru_cache(maxsize=None)
def _kernel_and_offsets(connectivity: str) -> tuple[np.ndarray, tuple[tuple[int, int], ...]]:
    conn = str(connectivity).lower().strip()

    def _manhattan_structure(radius: int) -> np.ndarray:
        size = 2 * radius + 1
        cy = cx = radius
        st = np.zeros((size, size), dtype=np.uint8)
        for yy in range(size):
            for xx in range(size):
                if abs(yy - cy) + abs(xx - cx) <= radius:
                    st[yy, xx] = 1
        return st

    manhattan_radius_map = {
        "4n": 1,
        "40n": 4,
        "60n": 5,
        "84n": 6,
        "112n": 7,
        "144n": 8,
    }
    if conn in manhattan_radius_map:
        structure = _manhattan_structure(manhattan_radius_map[conn])
    elif conn == "8n":
        structure = np.ones((3, 3), dtype=np.uint8)
    elif conn == "24n":
        structure = np.ones((5, 5), dtype=np.uint8)
    else:
        raise ValueError(
            f"Unsupported nodule connectivity '{connectivity}'. Choose one of: 4n, 8n, 24n, 40n, 60n, 84n, 112n, 144n."
        )

    cy = structure.shape[0] // 2
    cx = structure.shape[1] // 2
    offsets: list[tuple[int, int]] = []
    for oy, ox in zip(*np.where(structure > 0)):
        dy = int(oy - cy)
        dx = int(ox - cx)
        if dy == 0 and dx == 0:
            continue
        if dy > 0 or (dy == 0 and dx > 0):
            offsets.append((dy, dx))
    return structure, tuple(offsets)


def _component_centers(
    mask: np.ndarray,
    offsets: tuple[tuple[int, int], ...],
    boundary_mode: str,
) -> list[tuple[float, float]]:
    h, w = mask.shape
    ys, xs = np.where(mask)
    n_nodes = xs.size
    if n_nodes == 0:
        return []

    id_grid = np.full((h, w), -1, dtype=np.int32)
    id_grid[ys, xs] = np.arange(n_nodes, dtype=np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []

    for dy, dx in offsets:
        if boundary_mode == "periodic":
            src_mask = mask & np.roll(mask, shift=(-dy, -dx), axis=(0, 1))
            if not src_mask.any():
                continue
            src_ids = id_grid[src_mask]
            dst_ids = np.roll(id_grid, shift=(-dy, -dx), axis=(0, 1))[src_mask]
        else:
            ys0 = max(0, -dy)
            ys1 = min(h, h - dy)
            xs0 = max(0, -dx)
            xs1 = min(w, w - dx)
            src = mask[ys0:ys1, xs0:xs1]
            dst = mask[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx]
            both = src & dst
            if not both.any():
                continue
            src_ids = id_grid[ys0:ys1, xs0:xs1][both]
            dst_ids = id_grid[ys0 + dy:ys1 + dy, xs0 + dx:xs1 + dx][both]
        rows.append(src_ids)
        cols.append(dst_ids)

    if rows:
        r = np.concatenate(rows + cols)
        c = np.concatenate(cols + rows)
        data = np.ones(r.shape[0], dtype=np.uint8)
        graph = coo_matrix((data, (r, c)), shape=(n_nodes, n_nodes))
    else:
        graph = coo_matrix((n_nodes, n_nodes), dtype=np.uint8)

    _, labels = connected_components(graph, directed=False, return_labels=True)
    n_comp = int(labels.max()) + 1 if labels.size else 0

    def _circular_mean(vals: np.ndarray, period: int) -> float:
        if vals.size == 0:
            return 0.0
        ang = (2.0 * np.pi * vals) / float(period)
        s = float(np.mean(np.sin(ang)))
        c = float(np.mean(np.cos(ang)))
        if abs(s) < 1e-12 and abs(c) < 1e-12:
            return float(np.mean(vals) % float(period))
        theta = float(np.arctan2(s, c))
        if theta < 0.0:
            theta += 2.0 * np.pi
        return float((theta / (2.0 * np.pi)) * float(period))

    centers: list[tuple[float, float]] = []
    for i in range(n_comp):
        sel = labels == i
        if not np.any(sel):
            continue
        x_comp = xs[sel].astype(float)
        y_comp = ys[sel].astype(float)
        if boundary_mode == "periodic":
            cx = _circular_mean(x_comp, w)
            cy = _circular_mean(y_comp, h)
        else:
            cx = float(np.mean(x_comp))
            cy = float(np.mean(y_comp))
        centers.append((cx, cy))
    return centers

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

    if "nodule_count" in df.columns:
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        _set_white(ax)
        ax.plot(
            days,
            df["nodule_count"],
            label="Nodules",
            color="#0891b2",
            lw=PLOT_STYLE["line_width"],
            marker="o",
            markersize=4,
            markerfacecolor="#0891b2",
            markeredgecolor="#0e7490",
        )
        ax.set_xlabel("Day")
        ax.set_ylabel("Count")
        ax.set_title("Nodule count over time", pad=10)
        ax.legend(frameon=False, loc="upper left")
        _format_xlabel(ax, days)
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "08_nodule_count")
        plt.close(fig)


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
    nodule_count: int | None = None,
    nodule_centers: list[tuple[float, float]] | None = None,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)

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
    col_scaled = col_norm(col)
    ha_scaled = ha_norm(ha)

    bg = 0.96
    mix = np.zeros((*col.shape, 3), dtype=float)
    mix[..., 0] = bg - 0.78 * col_scaled + 0.76 * ha_scaled
    mix[..., 1] = bg + 0.82 * col_scaled - 0.72 * ha_scaled
    mix[..., 2] = bg - 0.76 * col_scaled + 0.78 * ha_scaled
    mix = np.clip(mix, 0.0, 1.0)

    fig = plt.figure(figsize=FIGURE_SIZE_SPATIAL)
    ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
    _set_white(ax)
    ax.imshow(mix, origin="lower", interpolation="nearest")
    if nodule_centers:
        xs = [c[0] for c in nodule_centers]
        ys = [c[1] for c in nodule_centers]
        ax.scatter(
            xs,
            ys,
            s=52,
            c="#00c2ff",
            edgecolors="#083344",
            linewidths=0.8,
            marker="o",
            label="Nodule center",
            zorder=6,
        )
        legend = ax.legend(
            loc="upper right",
            frameon=True,
            fontsize=9,
            borderpad=0.4,
            handletextpad=0.4,
        )
        legend.get_frame().set_edgecolor("black")
        legend.get_frame().set_linewidth(0.6)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.suptitle(
        f"{title_label} — Collagen (green), HA (magenta), overlap (bright), nodules (cyan)",
        fontsize=12,
        y=0.97,
    )

    stats_bits: list[str] = []
    if n_preob is not None and n_ob is not None:
        stats_bits.append(f"preOB: {n_preob}")
        stats_bits.append(f"OB: {n_ob}")
    if dead_today is not None:
        stats_bits.append(f"Dead: {dead_today}")
    if nodule_count is not None:
        stats_bits.append(f"Nodules: {nodule_count}")
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
    sm_col = ScalarMappable(norm=col_norm, cmap="YlGn")
    sm_ha = ScalarMappable(norm=ha_norm, cmap="PuRd")
    cbar_left = fig.colorbar(sm_col, cax=cax_left, orientation="vertical")
    cbar_left.ax.yaxis.set_label_position("left")
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_left.set_label("Collagen-I (a.u.)", fontsize=9)
    cbar_left.ax.tick_params(labelsize=8, colors="black")
    cbar_right = fig.colorbar(sm_ha, cax=cax_right, orientation="vertical")
    cbar_right.set_label("Hydroxyapatite (a.u.)", fontsize=9)
    cbar_right.ax.tick_params(labelsize=8, colors="black")

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
    nodule_count: int | None = None,
    nodule_centers: list[tuple[float, float]] | None = None,
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
        nodule_count=nodule_count,
        nodule_centers=nodule_centers,
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
    nodule_connectivity: str = "8n",
    boundary_mode: str = "clamp",
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
) -> None:
    target = Path(outdir) / "day_hor" / f"day{day}" / "collagen_ha"
    img_path = target / f"hour_{hour:02d}.pdf"
    n_preob = int((state_grid == CellState.PRE_OB).sum())
    n_ob = int((state_grid == CellState.OSTEOBLAST).sum())
    dead_today = int(deaths_preob_mask.sum() + deaths_ob_mask.sum())
    nodule_centers, _ = detect_nodules(
        state_grid,
        ha,
        connectivity=nodule_connectivity,
        boundary_mode=boundary_mode,
    )
    export_overlay_collagen_ha(
        img_path,
        f"Day {day:02d} — Hour {hour:02d}",
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        nodule_count=len(nodule_centers),
        nodule_centers=nodule_centers,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
    )


def export_overlay_collagen_ha_from_npy(
    grids_dir: str | Path,
    outdir: str | Path,
    day: int,
    nodule_connectivity: str = "8n",
    boundary_mode: str = "clamp",
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
    n_preob = n_ob = dead_today = nodule_count = None
    nodule_centers: list[tuple[float, float]] | None = None
    if state_path.exists():
        state_grid = np.load(state_path)
        n_preob = int((state_grid == CellState.PRE_OB).sum())
        n_ob = int((state_grid == CellState.OSTEOBLAST).sum())
        nodule_centers, _ = detect_nodules(
            state_grid,
            ha,
            connectivity=nodule_connectivity,
            boundary_mode=boundary_mode,
        )
        nodule_count = len(nodule_centers)
    if deaths_path.exists():
        deaths_grid = np.load(deaths_path)
        dead_today = int(np.asarray(deaths_grid).sum())

    export_overlay_collagen_ha(
        img_dir / f"collagen_ha_day_{day:03d}.pdf",
        f"Day {day:02d}",
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        nodule_count=nodule_count,
        nodule_centers=nodule_centers,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
    )
