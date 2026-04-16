from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import LinearSegmentedColormap, PowerNorm, to_rgb
from .agent import CellState
from .modulatoreff import plot_drug_modulation_summary

__all__ = [
    "plot_summary",
    "export_spatial_for_day",
    "export_spatial_hourly",
    "export_alp_field_snapshot",
    "export_overlay_collagen_ha",
    "export_overlay_collagen_ha_from_npy",
    "export_collagen_ha_snapshot",
]

COLORS = {
    "preob": "#0072B2",
    "ob": "#D55E00",
    "dead_preob": "#E45756",
    "dead_ob": "#B279A2",
    "text": "#111827",
    "bg": "#ffffff",
    "grid": "#d6dee8",
    "muted": "#64748b",
    "ink": "#1f2937",
}
PLOT_STYLE = {
    "title_size": 12,
    "label_size": 10,
    "tick_size": 9,
    "line_width": 2.0,
    "grid_alpha": 0.12,
    "spine_width": 0.9,
    "major_tick_length": 4.5,
    "major_tick_width": 0.85,
}

FIGURE_SIZE_MAP = (10.6, 9.2)
MAP_AX_POS = [0.13, 0.035, 0.74, 0.92]
CBAR_LEFT_POS = [0.098, 0.095, 0.026, 0.81]
CBAR_RIGHT_POS = [0.875, 0.095, 0.026, 0.81]
SPATIAL_DEAD_COLOR = "#2f2f2f"
COLLAGEN_BASE_COLOR = "#0f9d94"
HAP_BASE_COLOR = "#c2185b"
SNAPSHOT_TEXT_SCALE = 1.5


def _snap_fs(size: float) -> float:
    return float(size) * SNAPSHOT_TEXT_SCALE


def _sequential_map(name: str, color: str) -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(name, ["#ffffff", color])


COLLAGEN_CMAP = _sequential_map("osteomin_collagen", COLLAGEN_BASE_COLOR)
HAP_CMAP = _sequential_map("osteomin_hap", HAP_BASE_COLOR)
ALP_CMAP = LinearSegmentedColormap.from_list(
    "osteomin_alp",
    ["#ffffff", "#fff2b3", "#f9c22e", "#f28e1c", "#d73027"],
)

plt.rcParams.update(
    {
        "font.family": "DejaVu Serif",
        "axes.titleweight": "semibold",
        "axes.titlesize": PLOT_STYLE["title_size"],
        "axes.labelsize": PLOT_STYLE["label_size"],
        "xtick.labelsize": PLOT_STYLE["tick_size"],
        "ytick.labelsize": PLOT_STYLE["tick_size"],
        "legend.fontsize": 9,
        "axes.edgecolor": COLORS["ink"],
        "axes.labelcolor": COLORS["text"],
        "axes.linewidth": PLOT_STYLE["spine_width"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "text.color": COLORS["text"],
        "figure.facecolor": COLORS["bg"],
        "savefig.facecolor": COLORS["bg"],
        "savefig.edgecolor": COLORS["bg"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _set_white(ax):
    ax.set_facecolor("white")
    for spine in ax.spines.values():
        spine.set_color(COLORS["ink"])
        spine.set_linewidth(PLOT_STYLE["spine_width"])
    ax.tick_params(
        direction="out",
        length=PLOT_STYLE["major_tick_length"],
        width=PLOT_STYLE["major_tick_width"],
        colors=COLORS["text"],
        pad=3,
    )
    ax.grid(axis="y", alpha=PLOT_STYLE["grid_alpha"], linestyle="-", linewidth=0.8, color=COLORS["grid"])
    ax.grid(axis="x", alpha=0.0)


def _set_map_header(
    fig: plt.Figure,
    title: str,
    subtitle: str | None = None,
    *,
    title_y: float = 0.985,
    subtitle_y: float = 0.955,
    title_size: float = 12,
) -> None:
    fig.suptitle(title, fontsize=_snap_fs(title_size), y=title_y)
    if subtitle:
        fig.text(
            0.5,
            subtitle_y,
            subtitle,
            ha="center",
            va="center",
            fontsize=_snap_fs(10),
            color=COLORS["text"],
        )


def _style_timecourse(ax: plt.Axes, days: np.ndarray, ylabel: str, title: str, *, legend_loc: str = "upper left") -> None:
    ax.set_xlabel("Day")
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=10)
    ax.legend(frameon=False, loc=legend_loc, handlelength=2.8, labelspacing=0.4)
    _format_xlabel(ax, days)


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
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")


def _save_snapshot_figure(fig: plt.Figure, img_path: Path) -> None:
    fig.savefig(
        img_path,
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.005,
        facecolor="white",
        edgecolor="white",
        transparent=False,
    )


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
        label="Basal (live-cell) contribution",
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
    if days_plot.size:
        ax.set_xlim(days_plot[0], days_plot[-1])
        ticks = list(ax.get_xticks())
        last_day = float(days_plot[-1])
        if all(abs(t - last_day) > 1e-6 for t in ticks):
            ticks.append(last_day)
            ticks = sorted(ticks)
            ax.set_xticks(ticks)
    _style_timecourse(ax, days_plot, ylabel, title, legend_loc="upper right")
    fig.tight_layout(pad=0.6)
    _save_figure(fig, outdir, Path(filename).stem)
    plt.close(fig)


def plot_summary(
    df: pd.DataFrame,
    outdir: str | Path,
    *,
    field_cfg: object | None = None,
    modulator_df: pd.DataFrame | None = None,
    total_hours: int | None = None,
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
        color=COLORS["ink"],
        lw=2.1,
        ls="--",
    )
    _style_timecourse(ax, days, "Cells", "Cell states over time", legend_loc="upper right")
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
        color=COLORS["ink"],
        lw=2.1,
        ls="--",
    )
    _style_timecourse(ax, days, "Cumulative count", "Cumulative events", legend_loc="upper right")
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
    _style_timecourse(ax, days, "Count", "Daily events", legend_loc="upper right")
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
            title="HAp accumulation (normalized cumulative)",
            ylabel="Fraction of total",
        )

    if all(col in df.columns for col in ("msd_effective", "avg_move_count")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days, df["msd_effective"], label="Effective MSD", color="#0f766e", lw=2.0)
        ax.plot(days, df["avg_move_count"], label="Average move count", color="#7c3aed", lw=2.0)
        _style_timecourse(ax, days, "Value", "Cell motility summaries")
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "03_cell_motility")
        plt.close(fig)

    if all(col in df.columns for col in ("alp_total", "collagen_total", "ha_total")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days, df["alp_total"], label="ALP total", color="#2563eb", lw=2.0)
        ax.plot(days, df["collagen_total"], label="Collagen total", color="#059669", lw=2.0)
        ax.plot(days, df["ha_total"], label="HAp total", color="#b45309", lw=2.0)
        _style_timecourse(ax, days, "Cumulative total", "Field totals over time")
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "09_field_totals_absolute")
        plt.close(fig)

    if all(col in df.columns for col in ("avg_division_interval", "avg_live_age_hours")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(
            days,
            df["avg_division_interval"],
            label="Average division interval",
            color="#ea580c",
            lw=2.0,
        )
        ax.plot(
            days,
            df["avg_live_age_hours"],
            label="Average live-cell age",
            color="#475569",
            lw=2.0,
        )
        _style_timecourse(ax, days, "Hours", "Cell cycle summaries")
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "12_cell_cycle_summary")
        plt.close(fig)

    if field_cfg is not None and total_hours is not None and int(total_hours) > 0:
        hours = np.arange(int(total_hours) + 1, dtype=float)
        days_hr = hours / 24.0
        induction_h = max(0.0, float(getattr(field_cfg, "osteo_induction_hours", 0.0)))
        media_interval = int(getattr(field_cfg, "media_interval", 0))
        aa_initial = max(0.0, float(getattr(field_cfg, "aa_initial", 0.0)))
        aa_half_life = max(1e-9, float(getattr(field_cfg, "aa_half_life_hours", 1.0)))
        aa_km = max(1e-9, float(getattr(field_cfg, "collagen_aa_half_sat", 1.0)))

        aa = np.zeros_like(hours, dtype=float)
        if aa_initial > 0.0:
            k = np.log(2.0) / aa_half_life
            for i, hour in enumerate(hours):
                if hour < induction_h:
                    continue
                if media_interval > 0:
                    since = (hour - induction_h) % float(media_interval)
                else:
                    since = hour - induction_h
                aa[i] = aa_initial * np.exp(-k * since)

        aa_gate = np.zeros_like(aa, dtype=float)
        valid = aa > 0.0
        aa_gate[valid] = aa[valid] / (aa_km + aa[valid])

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days_hr, aa, color="#0f766e", lw=2.0, label="AA(t)")
        ax.axvline(
            induction_h / 24.0,
            color="#64748b",
            lw=1.2,
            ls="--",
            label="Osteogenic induction",
        )
        _style_timecourse(ax, days_hr, "AA concentration", "Ascorbate pulse-decay schedule", legend_loc="upper right")
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "10_ascorbate_schedule")
        plt.close(fig)

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days_hr, aa_gate, color="#16a34a", lw=2.0, label="A/(K_m + A)")
        ax.plot(days_hr, aa, color="#94a3b8", lw=1.2, ls="--", label="AA(t)")
        ax.axvline(induction_h / 24.0, color="#64748b", lw=1.2, ls=":")
        _style_timecourse(ax, days_hr, "Collagen gate", "Ascorbate-gated collagen activation", legend_loc="upper right")
        fig.tight_layout(pad=0.6)
        _save_figure(fig, outdir, "11_ascorbate_collagen_gate")
        plt.close(fig)

        plot_drug_modulation_summary(
            outdir,
            field_cfg,
            modulator_df=modulator_df,
            alp_plot_hours=alp_plot_hours,
            collagen_plot_hours=collagen_plot_hours,
            mineral_plot_hours=mineral_plot_hours,
            save_figure=_save_figure,
            set_white=_set_white,
            colors=COLORS,
        )


def _render_spatial_image(
    img_path: Path,
    title_label: str,
    state_grid: np.ndarray,
    deaths_preob_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    dead_total_cum: int,
) -> None:
    h, w = state_grid.shape
    canvas = np.zeros((h, w, 4), dtype=float)
    canvas[..., :3] = 1.0
    canvas[..., 3] = 1.0
    preob_mask = state_grid == CellState.PRE_OB
    ob_mask = state_grid == CellState.OSTEOBLAST
    dead_mask = deaths_preob_mask.astype(bool) | deaths_ob_mask.astype(bool)
    canvas[preob_mask, :3] = to_rgb(COLORS["preob"])
    canvas[ob_mask, :3] = to_rgb(COLORS["ob"])
    canvas[dead_mask, :3] = to_rgb(SPATIAL_DEAD_COLOR)
    canvas[dead_mask, 3] = 0.6

    n_preob = int(preob_mask.sum())
    n_ob = int(ob_mask.sum())
    n_da = int(deaths_preob_mask.sum())
    n_do = int(deaths_ob_mask.sum())

    fig = plt.figure(figsize=FIGURE_SIZE_MAP)
    ax = fig.add_axes(MAP_AX_POS)
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    _set_map_header(
        fig,
        f"{title_label} — spatial cells",
        title_y=0.958,
        title_size=16,
    )
    legend_elems = [
        Patch(facecolor=COLORS["preob"], label=f"preOB ({n_preob})"),
        Patch(facecolor=COLORS["ob"], label=f"Osteoblast ({n_ob})"),
        Patch(facecolor=SPATIAL_DEAD_COLOR, label=f"Dead ({n_da + n_do})"),
    ]
    legend = ax.legend(
        handles=legend_elems,
        frameon=True,
        loc="upper right",
        ncol=1,
        fontsize=_snap_fs(11.0),
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.6)
    ax.set_aspect("equal")
    img_path.parent.mkdir(parents=True, exist_ok=True)
    _save_snapshot_figure(fig, img_path)
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
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
    state_grid: np.ndarray | None = None,
    ha_baseline: float = 0.0,
) -> None:
    img_path.parent.mkdir(parents=True, exist_ok=True)

    ha_baseline = max(0.0, float(ha_baseline))
    ha_disp = np.maximum(0.0, ha - ha_baseline)

    if not col.any() and not ha_disp.any():
        fig = plt.figure(figsize=FIGURE_SIZE_MAP)
        ax = fig.add_axes(MAP_AX_POS)
        _set_white(ax)
        ax.text(
            0.5,
            0.5,
            "No collagen or HAp yet",
            ha="center",
            va="center",
            fontsize=_snap_fs(14),
        )
        ax.set_xticks([])
        ax.set_yticks([])
        _set_map_header(
            fig,
            f"{title_label} — collagen/HAp overlay",
            title_y=0.958,
            title_size=16,
        )
        _save_snapshot_figure(fig, img_path)
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
        vmax_ha = max(_p99(ha_disp), 1e-9)

    col_norm = PowerNorm(gamma=0.62, vmin=0.0, vmax=vmax_col)
    ha_norm = PowerNorm(gamma=0.7, vmin=0.0, vmax=vmax_ha)
    ha_scaled = ha_norm(ha_disp)

    fig = plt.figure(figsize=FIGURE_SIZE_MAP)
    ax = fig.add_axes(MAP_AX_POS)
    _set_white(ax)
    im_col = ax.imshow(
        col,
        origin="lower",
        interpolation="nearest",
        cmap=COLLAGEN_CMAP,
        norm=col_norm,
    )
    ha_rgba = HAP_CMAP(ha_scaled)
    ha_rgba[..., 3] = np.where(ha_disp > 0.0, 0.14 + 0.56 * np.power(ha_scaled, 1.05), 0.0)
    ax.imshow(
        ha_rgba,
        origin="lower",
        interpolation="nearest",
        zorder=3,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    _set_map_header(
        fig,
        f"{title_label} — collagen/HAp overlay",
        title_y=0.958,
        title_size=16,
    )

    sm_col = plt.cm.ScalarMappable(norm=col_norm, cmap=COLLAGEN_CMAP)
    sm_ha = plt.cm.ScalarMappable(norm=ha_norm, cmap=HAP_CMAP)
    cax_col = fig.add_axes(CBAR_LEFT_POS)
    cbar_col = fig.colorbar(im_col, cax=cax_col)
    cbar_col.ax.yaxis.set_label_position("left")
    cbar_col.ax.yaxis.set_ticks_position("left")
    cbar_col.set_label("Collagen-I (a.u.)", fontsize=_snap_fs(13.5), labelpad=12)
    cbar_col.ax.tick_params(
        labelsize=_snap_fs(11.5), colors="black", width=0.9, length=4.0
    )

    cax_ha = fig.add_axes(CBAR_RIGHT_POS)
    cbar_ha = fig.colorbar(sm_ha, cax=cax_ha)
    cbar_ha.set_label("Hydroxyapatite (a.u.)", fontsize=_snap_fs(13.5), labelpad=12)
    cbar_ha.ax.tick_params(
        labelsize=_snap_fs(11.5), colors="black", width=0.9, length=4.0
    )

    _save_snapshot_figure(fig, img_path)
    plt.close(fig)


def export_alp_field_snapshot(
    img_path: str | Path,
    title_label: str,
    alp: np.ndarray,
    *,
    global_vmax_alp: float | None = None,
    state_grid: np.ndarray | None = None,
) -> None:
    img_path = Path(img_path)
    img_path.parent.mkdir(parents=True, exist_ok=True)
    if alp.size == 0:
        return
    alp_disp = np.maximum(0.0, alp)
    vmax = (
        max(float(global_vmax_alp), 1e-9)
        if global_vmax_alp is not None
        else max(float(np.nanpercentile(alp_disp, 99)), 1e-9)
    )
    floor = max(vmax * 0.02, 1e-12)
    alp_plot = alp_disp.copy()
    alp_plot[alp_plot < floor] = 0.0
    norm = PowerNorm(gamma=0.5, vmin=0.0, vmax=vmax)
    fig = plt.figure(figsize=FIGURE_SIZE_MAP)
    ax = fig.add_axes(MAP_AX_POS)
    _set_white(ax)
    im = ax.imshow(
        alp_plot,
        origin="lower",
        interpolation="nearest",
        cmap=ALP_CMAP,
        norm=norm,
    )
    if state_grid is not None:
        live_mask = (state_grid == CellState.PRE_OB) | (state_grid == CellState.OSTEOBLAST)
        if np.any(live_mask):
            ys, xs = np.where(live_mask)
            ax.scatter(
                xs,
                ys,
                s=6,
                c="#111827",
                alpha=0.14,
                linewidths=0.0,
                zorder=4,
            )
    ax.set_xticks([])
    ax.set_yticks([])
    _set_map_header(
        fig,
        f"{title_label} — ALP field",
        title_y=0.958,
        title_size=16,
    )
    cax = fig.add_axes(CBAR_LEFT_POS)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.yaxis.set_label_position("left")
    cbar.ax.yaxis.set_ticks_position("left")
    tick_vals = np.linspace(0.0, vmax, 5)
    cbar.set_ticks(tick_vals)
    cbar.set_ticklabels([f"{int(round(v))}" for v in tick_vals])
    cbar.set_label("ALP signal (a.u.)", fontsize=_snap_fs(13.5), labelpad=12)
    cbar.ax.tick_params(
        labelsize=_snap_fs(11.5), colors="black", width=0.9, length=4.0
    )
    _save_snapshot_figure(fig, img_path)
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
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
    state_grid: np.ndarray | None = None,
    ha_baseline: float = 0.0,
) -> None:
    _render_collagen_ha_overlay(
        Path(img_path),
        title_label,
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
        state_grid=state_grid,
        ha_baseline=ha_baseline,
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
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
    ha_baseline: float = 0.0,
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
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
        state_grid=state_grid,
        ha_baseline=ha_baseline,
    )


def export_overlay_collagen_ha_from_npy(
    grids_dir: str | Path,
    outdir: str | Path,
    day: int,
    global_vmax_col: float | None = None,
    global_vmax_ha: float | None = None,
    ha_baseline: float = 0.0,
) -> None:
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

    export_overlay_collagen_ha(
        img_dir / f"collagen_ha_day_{day:03d}.pdf",
        f"Day {day:02d}",
        col,
        ha,
        n_preob=n_preob,
        n_ob=n_ob,
        dead_today=dead_today,
        global_vmax_col=global_vmax_col,
        global_vmax_ha=global_vmax_ha,
        state_grid=state_grid if state_path.exists() else None,
        ha_baseline=ha_baseline,
    )
