from __future__ import annotations
from pathlib import Path
from typing import List, Tuple
import polars as pl
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch

__all__ = ["plot_summary", "export_single_panels"]

# --------------------- colors (Active ↔ Osteoblast swapped) ---------------------
COLORS = {
    "total":        "#000000",
    "active":       "#009E73",  # green
    "osteoblast":   "#56B4E9",  # blue
    "apoptotic":    "#D55E00",  # orange/red
    "cum committed":"#E69F00",
    "cum died":     "#D55E00",
    "collagen":     "#0072B2",
    "alp":          "#CC79A7",
    "ha":           "#666666",
    "commit":       "#E69F00",
    "death":        "#D55E00",
}

# distinct overlay tints for split deaths
OVERLAY_DEAD_ACTIVE = (0.90, 0.45, 0.00, 0.80)   # orange-ish
OVERLAY_DEAD_OB     = (0.25, 0.20, 0.55, 0.80)   # purple-ish
OVERLAY_DEAD_TOTAL  = (0.84, 0.37, 0.00, 0.80)   # fallback single mask

# --------------------- helpers ---------------------
def _to_pandas(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.to_pandas()
    if {"free_ca", "free_pi"}.issubset(pdf.columns):
        pdf["ha_proxy"] = (1.0 - pdf[["free_ca", "free_pi"]].min(axis=1)).clip(lower=0.0)
    else:
        if "ha_proxy" not in pdf.columns:
            pdf["ha_proxy"] = np.nan
    return pdf

def _find(paths_dir: Path, pattern: str) -> List[Path]:
    return sorted((paths_dir.glob(pattern))) if paths_dir.exists() else []

def _nearest_upsample(arr: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(arr, np.ones((scale, scale), dtype=arr.dtype))

def _load(path: Path) -> np.ndarray | None:
    try:
        return np.load(path) if path.exists() else None
    except Exception as e:
        print(f"[viz] Failed to load {path}: {e}")
        return None

def _state_colormap() -> Tuple[ListedColormap, BoundaryNorm, dict]:
    """0=empty, 1=ACTIVE, 2=APOPTOTIC, 3=OSTEOBLAST"""
    colors = ["#FFFFFF", COLORS["active"], COLORS["apoptotic"], COLORS["osteoblast"]]
    cmap = ListedColormap(colors)
    bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
    norm = BoundaryNorm(bounds, cmap.N)
    legend = {"Active": COLORS["active"], "Apoptotic": COLORS["apoptotic"], "Osteoblast": COLORS["osteoblast"]}
    return cmap, norm, legend

def _load_cumulative_any(gdir: Path, day: int) -> np.ndarray | None:
    """OR all deaths_day_000..day masks; returns None if none exist."""
    cum = None
    for d in range(day + 1):
        p = gdir / f"deaths_day_{d:03d}.npy"
        if p.exists():
            arr = _load(p)
            if arr is None:
                continue
            arr = arr.astype(np.uint8)
            cum = arr if cum is None else np.maximum(cum, arr)
    return cum

def _load_cumulative_split(gdir: Path, day: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Optional split: deaths_active_day_###.npy and deaths_ob_day_###.npy.
    Returns (cum_dead_active, cum_dead_ob) or (None, None) if not available.
    """
    cum_a, cum_ob = None, None
    found_any = False
    for d in range(day + 1):
        pa = gdir / f"deaths_active_day_{d:03d}.npy"
        po = gdir / f"deaths_ob_day_{d:03d}.npy"
        if pa.exists():
            found_any = True
            aa = _load(pa)
            if aa is not None:
                aa = aa.astype(np.uint8)
                cum_a = aa if cum_a is None else np.maximum(cum_a, aa)
        if po.exists():
            found_any = True
            oo = _load(po)
            if oo is not None:
                oo = oo.astype(np.uint8)
                cum_ob = oo if cum_ob is None else np.maximum(cum_ob, oo)
    if not found_any:
        return None, None
    return cum_a, cum_ob

def _finish_ax(ax, title: str, xlabel: str, ylabel: str, legend_loc="best"):
    ax.set_title(title, fontsize=12, weight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, which="major", color="#EAEAEA")
    ax.grid(True, which="minor", color="#F3F3F3", alpha=0.6)
    ax.legend(loc=legend_loc, frameon=False, fontsize=9)

def _save_fig(fig: plt.Figure, path: Path, dpi=450):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

# --------------------- SUMMARY FIGURES (matplotlib only) ---------------------
def plot_summary(df: pl.DataFrame, outdir: str | Path) -> Path:
    out = Path(outdir); out.mkdir(parents=True, exist_ok=True)
    pdf = _to_pandas(df).copy()

    # Safe defaults
    for c in ["n_total", "n_active", "n_osteoblast", "cum_committed", "cum_died",
              "collagen_total", "alp_total", "ha_proxy", "committed_today", "died_today"]:
        if c not in pdf.columns:
            pdf[c] = 0 if "cum" in c or "n_" in c or "today" in c else np.nan

    x = pdf["day"].to_numpy()

    # 01) Absolute counts
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x, pdf["n_total"], label="Total", linewidth=2.0, color=COLORS["total"])
    ax.plot(x, pdf["n_active"], label="Active", linewidth=2.0, color=COLORS["active"])
    ax.plot(x, pdf["n_osteoblast"], label="Osteoblast", linewidth=2.0, color=COLORS["osteoblast"])
    ax.set_ylim(bottom=0)
    _finish_ax(ax, "Agent-based cell counts", "Day", "Cells")
    _save_fig(fig, out / "01_states_counts.png")

    # 02) Cumulative events
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x, pdf["cum_committed"], label="Committed (cum)", linewidth=2.0, color=COLORS["cum committed"])
    ax.plot(x, pdf["cum_died"], label="Died (cum)", linewidth=2.0, color=COLORS["cum died"], linestyle="--")
    ax.set_ylim(bottom=0)
    _finish_ax(ax, "Cumulative events", "Day", "Cumulative count")
    _save_fig(fig, out / "02_events_cumulative.png")

    # 03) Matrix/biochem + HA proxy
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    ax.plot(x, pdf["collagen_total"], label="Collagen (total)", linewidth=2.0, color=COLORS["collagen"])
    ax.plot(x, pdf["alp_total"], label="ALP (total)", linewidth=2.0, color=COLORS["alp"])
    ax.plot(x, pdf["ha_proxy"], label="HA proxy (1−min)", linewidth=2.0, color=COLORS["ha"])
    _finish_ax(ax, "Matrix & biochem", "Day", "a.u.")
    _save_fig(fig, out / "03_matrix_biochem.png")

    # 04) Daily events (bars)
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    width = 0.45
    ax.bar(x - width/2, pdf["committed_today"], width=width, label="Committed (today)", color=COLORS["commit"])
    ax.bar(x + width/2, pdf["died_today"],      width=width, label="Died (today)",      color=COLORS["death"])
    _finish_ax(ax, "Daily events", "Day", "Count")
    _save_fig(fig, out / "04_events_daily.png")

    return out / "01_states_counts.png"

# --------------------- SPATIAL MAPS (true states + split/cumulative deaths overlay) ---------------------
def _render_spatial_panel(
    state_grid: np.ndarray,
    deaths_cum_any: np.ndarray | None,
    deaths_cum_active: np.ndarray | None,
    deaths_cum_ob: np.ndarray | None,
    out_png: Path,
    day: int,
    dpi: int = 500,
    scale: int = 8
) -> None:
    cmap, norm, legend = _state_colormap()

    if state_grid is None:
        print(f"[viz] Missing state grid for day {day}, skipped.")
        return

    # Alive counts from state grid
    bc = np.bincount(state_grid.ravel().astype(np.int16), minlength=4)
    active_alive, apoptotic_alive, ob_alive = int(bc[1]), int(bc[2]), int(bc[3])

    # Split death counts if present, else use the any-mask
    n_dead_act = int((deaths_cum_active > 0).sum()) if deaths_cum_active is not None else 0
    n_dead_ob  = int((deaths_cum_ob > 0).sum()) if deaths_cum_ob is not None else 0
    n_dead_any = int((deaths_cum_any > 0).sum()) if deaths_cum_any is not None else (n_dead_act + n_dead_ob)

    # Prefer split totals if available
    died_total = (n_dead_act + n_dead_ob) if (deaths_cum_active is not None or deaths_cum_ob is not None) else n_dead_any

    # Total = alive(active+OB) + Apoptotic total
    total_all = active_alive + ob_alive + died_total

    # Upscale for crisp viewing
    big = _nearest_upsample(state_grid, scale=scale)
    h, w = big.shape

    fig = plt.figure(figsize=(w/130, h/130), dpi=130)
    ax = plt.axes([0.02, 0.08, 0.76, 0.90])  # room for legend
    ax.imshow(big, cmap=cmap, norm=norm, interpolation="nearest", origin="upper")
    ax.set_axis_off()

    # --- overlays ---
    # Split overlays if available
    if deaths_cum_active is not None or deaths_cum_ob is not None:
        if deaths_cum_active is not None:
            overlay_a = _nearest_upsample((deaths_cum_active > 0).astype(np.uint8), scale=scale)
            cmap_a = ListedColormap([(1, 1, 1, 0), OVERLAY_DEAD_ACTIVE])
            ax.imshow(overlay_a, cmap=cmap_a, interpolation="nearest", origin="upper")
        if deaths_cum_ob is not None:
            overlay_ob = _nearest_upsample((deaths_cum_ob > 0).astype(np.uint8), scale=scale)
            cmap_ob = ListedColormap([(1, 1, 1, 0), OVERLAY_DEAD_OB])
            ax.imshow(overlay_ob, cmap=cmap_ob, interpolation="nearest", origin="upper")
    # Fallback overlay (single cumulative mask)
    elif deaths_cum_any is not None:
        overlay_any = _nearest_upsample((deaths_cum_any > 0).astype(np.uint8), scale=scale)
        cmap_any = ListedColormap([(1, 1, 1, 0), OVERLAY_DEAD_TOTAL])
        ax.imshow(overlay_any, cmap=cmap_any, interpolation="nearest", origin="upper")

    # Legend (outside)
    handles = [
        Patch(facecolor=legend["Active"], edgecolor="#222222", label=f"Active ({active_alive})"),
        Patch(facecolor=legend["Osteoblast"], edgecolor="#222222", label=f"Osteoblast ({ob_alive})"),
    ]

    # Add split death legend if present
    if deaths_cum_active is not None or deaths_cum_ob is not None:
        if deaths_cum_active is not None:
            handles.append(Patch(facecolor=OVERLAY_DEAD_ACTIVE, edgecolor="#222222",
                                 label=f"Died while active ({n_dead_act})"))
        if deaths_cum_ob is not None:
            handles.append(Patch(facecolor=OVERLAY_DEAD_OB, edgecolor="#222222",
                                 label=f"Died while osteoblast ({n_dead_ob})"))
        handles.append(Patch(facecolor=(0.85, 0.85, 0.85, 0.7), edgecolor="#222222",
                             label=f"Apoptotic total ({died_total})"))
    else:
        # Fallback: only total
        handles.append(Patch(facecolor=COLORS["cum died"], edgecolor="#222222",
                             label=f"Apoptotic total ({n_dead_any})"))

    leg_ax = fig.add_axes([0.80, 0.23, 0.18, 0.54]); leg_ax.axis("off")
    leg_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=10)

    # Title with exact breakdown
    title = (
        f"Day {day}  |  Total {total_all} cells  |  "
        f"act: {active_alive}, osteo: {ob_alive}, dead: {died_total}"
    )
    fig.suptitle(title, fontsize=14, fontweight="bold", x=0.02, ha="left", y=0.99)

    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=dpi, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)

def export_single_panels(df: pl.DataFrame, outdir: str | Path) -> None:
    """
    Renders spatial maps for first, middle, last day using:
      <out>/grids/state_day_###.npy
      Optional split overlays:
        <out>/grids/deaths_active_day_###.npy
        <out>/grids/deaths_ob_day_###.npy
      Fallback single overlay:
        <out>/grids/deaths_day_###.npy
    """
    out = Path(outdir)
    gdir = out / "grids"
    states = _find(gdir, "state_day_*.npy")
    if not states:
        print(f"[viz] No state grids at {gdir}. Skipping spatial maps.")
        return

    picks = [states[0], states[len(states)//2], states[-1]]

    for spath in picks:
        try:
            day = int(spath.stem.split("_")[-1])
        except Exception:
            print(f"[viz] Could not parse day from {spath.name}, skipping.")
            continue

        sgrid = _load(spath)
        if sgrid is None:
            print(f"[viz] Invalid state grid at {spath}, skipping.")
            continue

        # Try split; if not available, fall back to single cumulative mask
        cum_a, cum_ob = _load_cumulative_split(gdir, day)
        cum_any = _load_cumulative_any(gdir, day) if (cum_a is None and cum_ob is None) else None

        out_png = out / f"spatial_day_{day:03d}.png"
        _render_spatial_panel(
            sgrid,
            cum_any,
            cum_a,
            cum_ob,
            out_png,
            day=day,
            dpi=500,
            scale=8
        )
        print(f"[viz] wrote {out_png}")
