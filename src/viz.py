from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # safe non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.colors import to_rgb
from mpl_toolkits.axes_grid1 import make_axes_locatable

__all__ = [
    "plot_summary",
    "export_spatial_for_day",
    "export_overlay_collagen_ha",
    "export_overlay_collagen_ha_from_npy",
]

COLORS = {
    "active": "#0072B2",  # colorblind-safe blue
    "ob": "#D55E00",  # colorblind-safe vermillion
    "dead_active": "#E69F00",
    "dead_ob": "#CC79A7",
    "text": "#111111",
    "bg": "#ffffff",
    "grid": "#eaeaea",
}


def _set_white(ax):
    ax.set_facecolor(COLORS["bg"])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")


def plot_summary(df: pd.DataFrame, outdir: str | Path) -> None:
    outdir = Path(outdir)
    img_dir = outdir / "collagen_ha"
    img_dir.mkdir(parents=True, exist_ok=True)
    days = df["day"].to_numpy()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _set_white(ax)
    ax.plot(days, df["n_active"], label="Active", color=COLORS["active"], lw=2)
    ax.plot(days, df["n_osteoblast"], label="Osteoblast", color=COLORS["ob"], lw=2)
    ax.plot(days, df["n_total"], label="Total", color="#333", lw=1.3, ls="--")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cells")
    ax.set_title("Cell states over time")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.6)
    fig.savefig(outdir / "01_states_counts.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    tmp = df.copy()
    if "dead_total_today" not in tmp.columns:
        tmp["dead_total_today"] = tmp["dead_active_today"] + tmp["dead_ob_today"]
    tmp["dead_active_cum"] = tmp["dead_active_today"].cumsum()
    tmp["dead_ob_cum"] = tmp["dead_ob_today"].cumsum()
    tmp["dead_total_cum"] = tmp["dead_total_today"].cumsum()

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _set_white(ax)
    ax.plot(
        days,
        tmp["dead_active_cum"],
        label="Active→Dead (cum)",
        color=COLORS["dead_active"],
        lw=2,
    )
    ax.plot(
        days, tmp["dead_ob_cum"], label="OB→Dead (cum)", color=COLORS["dead_ob"], lw=2
    )
    ax.plot(
        days,
        tmp["dead_total_cum"],
        label="All deaths (cum)",
        color="#444",
        lw=1.5,
        ls="--",
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Cumulative events")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.6)
    fig.savefig(outdir / "02_events_cumulative.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # --- Biochemistry ions: three stacked panels (one figure) ---
    ca_used = df["ca_used_total"] if "ca_used_total" in df.columns else None
    ca_supplied = df["ca_supplied_total"] if "ca_supplied_total" in df.columns else None
    cell_ca = df["cell_ca"] if "cell_ca" in df.columns else None
    ca_media = df["ca_media"] if "ca_media" in df.columns else None

    pi_used = df["pi_used_total"] if "pi_used_total" in df.columns else None
    pi_generated = (
        df["pi_generated_total"] if "pi_generated_total" in df.columns else None
    )
    bgp_consumed = (
        df["bgp_consumed_total"] if "bgp_consumed_total" in df.columns else None
    )
    bgp_pool = df["bgp_pool"] if "bgp_pool" in df.columns else None
    cell_pi = df["cell_pi"] if "cell_pi" in df.columns else None

    fig, axes = plt.subplots(3, 1, figsize=(6, 7.2), sharex=True)

    # Panel 1: free media + cell pools
    ax_free = axes[0]
    _set_white(ax_free)
    handles: list = []
    labels: list = []
    if ca_media is not None:
        handles.append(
            ax_free.plot(
                days, ca_media, label="media Ca (free)", lw=2, color="#1f77b4"
            )[0]
        )
        labels.append("media Ca (free)")
    if cell_ca is not None:
        handles.append(
            ax_free.plot(
                days,
                cell_ca,
                label="cell-processed Ca (per area)",
                lw=2,
                color="#0d9488",
            )[0]
        )
        labels.append("cell-processed Ca (per area)")
    if bgp_pool is not None:
        handles.append(
            ax_free.plot(
                days,
                bgp_pool,
                label="β-GP pool (free)",
                lw=2,
                ls="--",
                color="#f97316",
            )[0]
        )
        labels.append("β-GP pool (free)")
    if cell_pi is not None:
        handles.append(
            ax_free.plot(
                days,
                cell_pi,
                label="cell-generated Pi (mean field)",
                lw=2,
                color="#ff7f0e",
            )[0]
        )
        labels.append("cell-generated Pi (mean field)")
    ax_free.set_ylabel("Free pools (a.u.)")
    ax_free.set_title("Free media & cell pools")
    if handles:
        ax_free.legend(handles, labels, frameon=False, ncol=2)
    ax_free.grid(alpha=0.25)

    # Panel 2: cumulative calcium budget (separate scales)
    ax_ca_left = axes[1]
    _set_white(ax_ca_left)
    left_handles = []
    left_labels = []
    if ca_used is not None:
        left_handles.append(
            ax_ca_left.plot(
                days,
                ca_used,
                label="Ca used (cum)",
                lw=2,
                color="#7c3aed",
            )[0]
        )
        left_labels.append("Ca used (cum)")
    ax_ca_left.set_ylabel("Ca used (a.u.)")
    ax_ca_left.set_title("Calcium budget (cumulative)")
    ax_ca_right = ax_ca_left.twinx()
    right_handles = []
    right_labels = []
    if ca_supplied is not None:
        right_handles.append(
            ax_ca_right.plot(
                days,
                ca_supplied,
                label="media Ca supplied (cum)",
                lw=2,
                ls=":",
                color="#268bd2",
            )[0]
        )
        right_labels.append("media Ca supplied (cum)")
    if right_handles:
        ax_ca_right.set_ylabel("Ca supplied (a.u.)")
    handles = left_handles + right_handles
    labels = left_labels + right_labels
    if handles:
        ax_ca_left.legend(handles, labels, frameon=False)
    ax_ca_left.grid(alpha=0.25)

    # Panel 3: cumulative phosphate / β-GP budget (separate scales)
    ax_pi_left = axes[2]
    _set_white(ax_pi_left)
    left_handles = []
    left_labels = []
    if pi_used is not None:
        left_handles.append(
            ax_pi_left.plot(
                days,
                pi_used,
                label="Pi used (cum)",
                lw=2,
                color="#6a51a3",
            )[0]
        )
        left_labels.append("Pi used (cum)")
    ax_pi_left.set_ylabel("Pi used (a.u.)")
    ax_pi_left.set_title("Phosphate / β-GP budget (cumulative)")
    ax_pi_right = ax_pi_left.twinx()
    right_handles = []
    right_labels = []
    if pi_generated is not None:
        right_handles.append(
            ax_pi_right.plot(
                days,
                pi_generated,
                label="Pi generated (cum)",
                lw=2,
                ls="--",
                color="#9467bd",
            )[0]
        )
        right_labels.append("Pi generated (cum)")
    if bgp_consumed is not None:
        right_handles.append(
            ax_pi_right.plot(
                days,
                bgp_consumed,
                label="β-GP consumed (cum)",
                lw=2,
                ls="-.",
                color="#d95f02",
            )[0]
        )
        right_labels.append("β-GP consumed (cum)")
    if right_handles:
        ax_pi_right.set_ylabel("Generated / β-GP (a.u.)")
    handles = left_handles + right_handles
    labels = left_labels + right_labels
    ax_pi_left.set_xlabel("Day")
    if handles:
        ax_pi_left.legend(handles, labels, frameon=False, ncol=2)
    ax_pi_left.grid(alpha=0.25)

    fig.tight_layout(pad=0.8)
    fig.savefig(outdir / "03_biochem.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    _set_white(ax)
    ax.bar(
        days,
        df["committed_today"],
        label="Committed (today)",
        color="#fdae61",
        alpha=0.9,
    )
    ax.bar(
        days,
        df["dead_active_today"],
        bottom=df["committed_today"],
        label="Died (active)",
        color=COLORS["dead_active"],
        alpha=0.9,
    )
    ax.bar(
        days,
        df["dead_ob_today"],
        bottom=df["committed_today"] + df["dead_active_today"],
        label="Died (osteoblast)",
        color=COLORS["dead_ob"],
        alpha=0.9,
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.set_title("Daily events")
    ax.legend(frameon=False, ncol=2)
    ax.grid(alpha=0.25)
    fig.tight_layout(pad=0.6)
    fig.savefig(outdir / "04_events_daily.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    supp_cols = [
        ("supp_aa", "Ascorbic acid (AA)", "#0d9488"),
        ("supp_bgp", "β-GP", "#f97316"),
        ("supp_mk4", "MK-4", "#6366f1"),
        ("supp_dexa", "Dexa", "#dc2626"),
    ]
    if all(col in df.columns for col, _, _ in supp_cols):
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _set_white(ax)
        for col, label, color in supp_cols:
            ax.plot(days, df[col], label=label, lw=2, color=color)
        ax.set_xlabel("Day")
        ax.set_ylabel("Supplement scale (a.u.)")
        ax.set_title("Supplements schedule")
        ax.legend(frameon=False, ncol=2)
        ax.grid(alpha=0.25)
        fig.tight_layout(pad=0.6)
        fig.savefig(outdir / "05_supplements.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    acp_total = df["acp_total"] if "acp_total" in df.columns else None
    ha_total = df["ha_total"] if "ha_total" in df.columns else None
    if acp_total is not None or ha_total is not None:
        fig, ax = plt.subplots(figsize=(6, 3.5))
        _set_white(ax)
        handles = []
        labels = []
        if acp_total is not None:
            handles.append(
                ax.plot(
                    days,
                    acp_total,
                    label="ACP (cum)",
                    lw=2,
                    color="#fb923c",
                )[0]
            )
            labels.append("ACP (cum)")
        if ha_total is not None:
            handles.append(
                ax.plot(
                    days,
                    ha_total,
                    label="HA (cum)",
                    lw=2,
                    color="#0ea5e9",
                )[0]
            )
            labels.append("HA (cum)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Cumulative mass (a.u.)")
        ax.set_title("ACP and HA accumulation")
        if handles:
            ax.legend(handles, labels, frameon=False)
        ax.grid(alpha=0.25)
        fig.tight_layout(pad=0.6)
        fig.savefig(outdir / "06_minerals_cumulative.png", dpi=300, bbox_inches="tight")
        plt.close(fig)


def export_spatial_for_day(
    outdir: str | Path,
    day: int,
    state_grid: np.ndarray,
    deaths_active_mask: np.ndarray,
    deaths_ob_mask: np.ndarray,
    dead_total_cum: int,
) -> None:
    h, w = state_grid.shape
    canvas = np.zeros((h, w, 3), dtype=float) + 1.0  # white BG

    # live cells
    active_mask = state_grid == 1
    ob_mask = state_grid == 3
    active_color = to_rgb(COLORS["active"])
    ob_color = to_rgb(COLORS["ob"])
    canvas[active_mask] = active_color
    canvas[ob_mask] = ob_color

    # deaths today
    canvas[deaths_active_mask.astype(bool)] = to_rgb(COLORS["dead_active"])
    canvas[deaths_ob_mask.astype(bool)] = to_rgb(COLORS["dead_ob"])

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    n_act = int((state_grid == 1).sum())
    n_ob = int((state_grid == 3).sum())
    n_da = int(deaths_active_mask.sum())
    n_do = int(deaths_ob_mask.sum())
    ax.set_title(
        f"Day {day:02d}  |  Total {n_act + n_ob} live  |  "
        f"act: {n_act}, osteo: {n_ob}, dead today: {n_da + n_do}, dead cum: {dead_total_cum}",
        loc="center",
        fontsize=12,
        fontweight="bold",
    )
    legend_elems = [
        Patch(facecolor=COLORS["active"], label=f"Active ({n_act})"),
        Patch(facecolor=COLORS["ob"], label=f"Osteoblast ({n_ob})"),
        Patch(facecolor=COLORS["dead_active"], label=f"Died while active ({n_da})"),
        Patch(facecolor=COLORS["dead_ob"], label=f"Died while osteoblast ({n_do})"),
    ]
    legend = ax.legend(
        handles=legend_elems, frameon=True, loc="upper right", ncol=1, fontsize=9
    )
    legend.get_frame().set_edgecolor("black")
    legend.get_frame().set_linewidth(0.6)
    ax.set_aspect("equal")
    fig.subplots_adjust(top=0.88, right=0.95, left=0.05, bottom=0.04)
    outdir = Path(outdir)
    img_dir = outdir / "spatial_cells"
    img_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(img_dir / f"spatial_day_{day:03d}.png", dpi=300)
    plt.close(fig)


def _robust_norm(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
    """Normalize to [0,1] using a robust upper bound (percentile)."""
    if arr.size == 0:
        return arr
    vmax = np.percentile(arr[arr > 0], p) if (arr > 0).any() else 1.0
    vmax = max(vmax, 1e-9)
    out = np.clip(arr / vmax, 0.0, 1.0)
    return out


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
    if not col.any() and not ha.any():
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
        _set_white(ax)
        ax.text(0.5, 0.5, "No collagen or HA yet", ha="center", va="center", fontsize=14)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"Day {day:02d} — No collagen/HA", fontsize=12, pad=10)
        fig.savefig(img_dir / f"collagen_ha_day_{day:03d}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        return

    # robust p99 scaling for alpha
    def _p99(x):
        return float(np.percentile(x[x > 0], 99)) if (x > 0).any() else 1.0

    if global_vmax_col is not None:
        vmax_col = max(global_vmax_col, 1e-9)
    else:
        vmax_col = max(_p99(col), 1e-9)
    if global_vmax_ha is not None:
        vmax_ha = max(global_vmax_ha, 1e-9)
    else:
        vmax_ha = max(_p99(ha), 1e-9)
    norm_col = np.clip(col / vmax_col, 0, 1)
    norm_ha = np.clip(ha / vmax_ha, 0, 1)
    a_col = norm_col * 0.85
    a_ha = norm_ha * 0.85
    coloc = np.minimum(norm_col, norm_ha)
    a_coloc = np.clip(coloc * 0.9, 0, 1)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_axes([0.14, 0.12, 0.72, 0.74])
    _set_white(ax)
    ax.imshow(np.ones((*col.shape, 3)), origin="lower", interpolation="nearest")
    im_col = ax.imshow(
        col,
        cmap="Greens",
        vmin=0,
        vmax=vmax_col,
        origin="lower",
        interpolation="nearest",
        alpha=a_col,
    )
    im_ha = ax.imshow(
        ha,
        cmap="Blues",
        vmin=0,
        vmax=vmax_ha,
        origin="lower",
        interpolation="nearest",
        alpha=a_ha,
    )
    im_coloc = ax.imshow(
        coloc,
        cmap="Reds",
        vmin=0,
        vmax=1.0,
        origin="lower",
        interpolation="nearest",
        alpha=a_coloc,
    )
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(
        f"Day {day:02d} — Collagen (green), HA (blue), colocalized mineral (red)",
        fontsize=12,
        pad=10,
    )
    ax.set_aspect("equal")

    divider = make_axes_locatable(ax)
    cax_left = divider.append_axes("left", size="2.5%", pad=0.02)
    cax_right = divider.append_axes("right", size="2.5%", pad=0.02)
    cax_bottom = divider.append_axes("bottom", size="2.5%", pad=0.45)
    cbar_left = fig.colorbar(im_col, cax=cax_left, orientation="vertical")
    cbar_left.ax.yaxis.set_label_position("left")
    cbar_left.ax.yaxis.set_ticks_position("left")
    cbar_left.set_label("Collagen-I (a.u.)", fontsize=9)
    cbar_left.ax.tick_params(labelsize=8, colors="black")
    cbar_right = fig.colorbar(im_ha, cax=cax_right, orientation="vertical")
    cbar_right.set_label("Hydroxyapatite (a.u.)", fontsize=9)
    cbar_right.ax.tick_params(labelsize=8, colors="black")
    cbar_bottom = fig.colorbar(im_coloc, cax=cax_bottom, orientation="horizontal")
    cbar_bottom.set_label("Colocalized mineral (intensity)", fontsize=9)
    cbar_bottom.ax.tick_params(labelsize=8, colors="black")

    fig.savefig(img_dir / f"collagen_ha_day_{day:03d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
