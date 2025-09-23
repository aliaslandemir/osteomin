from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # safe non-GUI backend
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = ["plot_summary", "export_spatial_for_day", "export_overlay_collagen_ha", "export_overlay_collagen_ha_from_npy"]

COLORS = {
    "active":      "#2ca25f",
    "ob":          "#67a9cf",
    "dead_active": "#fdb863",
    "dead_ob":     "#8073ac",
    "text":        "#111111",
    "bg":          "#ffffff",
    "grid":        "#eaeaea",
}

def _set_white(ax):
    ax.set_facecolor(COLORS["bg"])
    for spine in ax.spines.values():
        spine.set_color("#cccccc")

def plot_summary(df: pd.DataFrame, outdir: str | Path) -> None:
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    days = df["day"].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5))
    _set_white(ax)
    ax.plot(days, df["n_active"], label="Active", color=COLORS["active"], lw=2)
    ax.plot(days, df["n_osteoblast"], label="Osteoblast", color=COLORS["ob"], lw=2)
    ax.plot(days, df["n_total"], label="Total", color="#333", lw=1.3, ls="--")
    ax.set_xlabel("Day"); ax.set_ylabel("Cells"); ax.set_title("Cell states over time")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(outdir / "01_states_counts.png", dpi=200); plt.close(fig)

    tmp = df.copy()
    if "dead_total_today" not in tmp.columns:
        tmp["dead_total_today"] = tmp["dead_active_today"] + tmp["dead_ob_today"]
    tmp["dead_active_cum"] = tmp["dead_active_today"].cumsum()
    tmp["dead_ob_cum"] = tmp["dead_ob_today"].cumsum()
    tmp["dead_total_cum"] = tmp["dead_total_today"].cumsum()

    fig, ax = plt.subplots(figsize=(10, 5))
    _set_white(ax)
    ax.plot(days, tmp["dead_active_cum"], label="Active→Dead (cum)", color=COLORS["dead_active"], lw=2)
    ax.plot(days, tmp["dead_ob_cum"], label="OB→Dead (cum)", color=COLORS["dead_ob"], lw=2)
    ax.plot(days, tmp["dead_total_cum"], label="All deaths (cum)", color="#444", lw=1.5, ls="--")
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative count"); ax.set_title("Cumulative events")
    ax.legend(frameon=False)
    fig.savefig(outdir / "02_events_cumulative.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    _set_white(ax)
    ax.plot(days, df["collagen_total"], label="Collagen (total)", lw=2, color=COLORS["ob"])
    ax.plot(days, df["alp_total"], label="ALP (total)", lw=2, color="#d33682")
    ax2 = ax.twinx()
    ax2.plot(days, df["free_ca"], label="free Ca", lw=1.8, color="#1f77b4", alpha=0.8)
    ax2.plot(days, df["free_pi"], label="free Pi", lw=1.8, color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("Day"); ax.set_ylabel("ECM (a.u.)"); ax2.set_ylabel("Ions (a.u.)")
    ax.set_title("Matrix & biochem")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, frameon=False, loc="upper left", ncol=2)
    fig.savefig(outdir / "03_matrix_biochem.png", dpi=200); plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    _set_white(ax)
    ax.bar(days, df["committed_today"], label="Committed (today)", color="#fdae61", alpha=0.9)
    ax.bar(days, df["dead_active_today"], bottom=df["committed_today"],
           label="Died (active)", color=COLORS["dead_active"], alpha=0.9)
    ax.bar(days, df["dead_ob_today"], bottom=df["committed_today"] + df["dead_active_today"],
           label="Died (osteoblast)", color=COLORS["dead_ob"], alpha=0.9)
    ax.set_xlabel("Day"); ax.set_ylabel("Count"); ax.set_title("Daily events")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(outdir / "04_events_daily.png", dpi=200); plt.close(fig)



def export_spatial_for_day(outdir: str | Path, day: int,
                           state_grid: np.ndarray,
                           deaths_active_mask: np.ndarray,
                           deaths_ob_mask: np.ndarray,
                           dead_total_cum: int) -> None:
    h, w = state_grid.shape
    canvas = np.zeros((h, w, 3), dtype=float) + 1.0  # white BG

    # live cells
    active_mask = (state_grid == 1)
    ob_mask = (state_grid == 3)
    canvas[active_mask] = [0.17, 0.64, 0.37]  # green-ish
    canvas[ob_mask] = [0.40, 0.66, 0.81]      # blue-ish

    # deaths today
    canvas[deaths_active_mask.astype(bool)] = [0.99, 0.72, 0.39]
    canvas[deaths_ob_mask.astype(bool)] = [0.50, 0.45, 0.67]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    n_act = int((state_grid == 1).sum())
    n_ob = int((state_grid == 3).sum())
    n_da = int(deaths_active_mask.sum())
    n_do = int(deaths_ob_mask.sum())
    ax.set_title(
        f"Day {day:02d}  |  Total {n_act + n_ob} live  |  "
        f"act: {n_act}, osteo: {n_ob}, dead today: {n_da + n_do}, dead cum: {dead_total_cum}",
        loc="left", fontsize=16, fontweight="bold"
    )
    legend_elems = [
        Patch(facecolor=COLORS["active"], label=f"Active ({n_act})"),
        Patch(facecolor=COLORS["ob"], label=f"Osteoblast ({n_ob})"),
        Patch(facecolor=COLORS["dead_active"], label=f"Died while active ({n_da})"),
        Patch(facecolor=COLORS["dead_ob"], label=f"Died while osteoblast ({n_do})"),
    ]
    ax.legend(handles=legend_elems, frameon=False, loc="center left", bbox_to_anchor=(1.02, 0.5))
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(outdir / f"spatial_day_{day:03d}.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _robust_norm(arr: np.ndarray, p: float = 99.0) -> np.ndarray:
    """Normalize to [0,1] using a robust upper bound (percentile)."""
    if arr.size == 0:
        return arr
    vmax = np.percentile(arr[arr > 0], p) if (arr > 0).any() else 1.0
    vmax = max(vmax, 1e-9)
    out = np.clip(arr / vmax, 0.0, 1.0)
    return out

def export_overlay_collagen_ha_from_npy(grids_dir: str | Path, outdir: str | Path, day: int) -> None:
    """
    Overlay using saved per-day arrays from /grids:
      - collagen_day_XXX.npy
      - ha_day_XXX.npy
    """
    grids_dir = Path(grids_dir)
    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    col = np.load(grids_dir / f"collagen_day_{day:03d}.npy")
    ha  = np.load(grids_dir / f"ha_day_{day:03d}.npy")

    # robust p99 scaling for alpha
    def _p99(x): return float(np.percentile(x[x > 0], 99)) if (x > 0).any() else 1.0
    vmax_col = max(_p99(col), 1e-9)
    vmax_ha  = max(_p99(ha),  1e-9)
    a_col = np.clip(col / vmax_col, 0, 1) * 0.70
    a_ha  = np.clip(ha  / vmax_ha,  0, 1) * 0.70

    fig, ax = plt.subplots(figsize=(7, 7))
    _set_white(ax)
    ax.imshow(np.ones((*col.shape, 3)), origin="lower", interpolation="nearest")
    im_col = ax.imshow(col, cmap="Reds",   vmin=0, vmax=vmax_col, origin="lower",
                       interpolation="nearest", alpha=a_col)
    im_ha  = ax.imshow(ha,  cmap="Greens", vmin=0, vmax=vmax_ha,  origin="lower",
                       interpolation="nearest", alpha=a_ha)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"Day {day:02d} — Collagen (red, α-scaled) over HAp (green, α-scaled)")

    cbar_left  = fig.colorbar(im_col, ax=ax, orientation="vertical", fraction=0.04, pad=0.02, location="left")
    cbar_left.set_label("Collagen-I (a.u.)")
    cbar_right = fig.colorbar(im_ha,  ax=ax, orientation="vertical", fraction=0.04, pad=0.02, location="right")
    cbar_right.set_label("Hydroxyapatite (a.u.)")

    plt.tight_layout()
    fig.savefig(outdir / f"collagen_ha_day_{day:03d}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
