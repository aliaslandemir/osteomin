from __future__ import annotations
from pathlib import Path
from typing import Sequence, Dict, Tuple
import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.lines import Line2D

__all__ = ["plot_summary", "export_single_panels"]

# ------------------------------- Styling -------------------------------------
plt.rcParams.update({
    "figure.dpi": 300, "savefig.dpi": 300,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "black", "axes.linewidth": 0.8,
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.titlesize": 11, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
})

# state_id: (label, color, marker)
STATE_INFO: Dict[int, Tuple[str, str, str]] = {
    1: ("Active",     "#2ca02c", "o"),
    2: ("Apoptotic",  "#d62728", "X"),
    3: ("Osteoblast", "#1f77b4", "s"),
}

CMAP = {
    "collagen": "Greens",
    "alp":      "Reds",
    "acp":      "Blues",
    "ha":       "Purples",
    "density":  "YlGnBu",
    "nodule":   "Greens",
}

# ----------------------------- Helpers ---------------------------------------
def _vlines(ax, xs=(7, 14, 21)):
    for x in xs:
        ax.axvline(x, ls="--", lw=1, color="#8a8a8a", alpha=0.6)

def _read_table(pathlike):
    p = Path(pathlike)
    ext = p.suffix.lower()
    if ext in (".parquet", ".pq"):
        return pl.read_parquet(p)
    if ext in (".xlsx", ".xls"):
        return pl.read_excel(p)
    if ext == ".csv":
        return pl.read_csv(p)
    for reader in (pl.read_excel, pl.read_parquet, pl.read_csv):
        try:
            return reader(p)
        except Exception:
            continue
    raise ValueError(f"Unsupported results file: {p}")

def _load_cumulative_deaths(src_dir: Path, up_to_day: int) -> tuple[np.ndarray, np.ndarray]:
    """Load and stack all deaths_{d}.csv for d ≤ up_to_day. Returns (xs, ys) or (empty, empty)."""
    xs_all, ys_all = [], []
    for d in range(1, up_to_day + 1):
        f = src_dir / f"deaths_{d}.csv"
        if f.exists():
            try:
                df = pl.read_csv(f)
                if {"x", "y"}.issubset(df.columns):
                    xs_all.append(df["x"].to_numpy())
                    ys_all.append(df["y"].to_numpy())
            except Exception:
                pass
    if xs_all:
        return np.concatenate(xs_all), np.concatenate(ys_all)
    return np.array([], dtype=int), np.array([], dtype=int)

# ---------------------------- Time-series figure -----------------------------
def plot_summary(results_path: str | Path, output: str | Path = "summary.png") -> None:
    """Time-series summary figure from results.xlsx/.csv/.parquet."""
    df = _read_table(results_path)
    days = df["Day"].to_numpy()

    fig, axarr = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)

    # cell counts (+ cumulative)
    ax = axarr[0, 0]
    for lbl, col, _ in STATE_INFO.values():
        colname = f"Cells_{lbl}"
        if colname in df.columns:
            pretty = "Apoptotic (daily)" if lbl == "Apoptotic" else lbl
            ax.plot(days, df[colname].to_numpy(), lw=2.2, color=col, label=pretty,
                    path_effects=[pe.Stroke(linewidth=3, foreground="white"), pe.Normal()])
    if "Cells_Osteoblast_Cum" in df.columns and "Cells_Apoptotic_Cum" in df.columns:
        ax.plot(days, df["Cells_Osteoblast_Cum"].to_numpy(), ls="--", lw=1.6,
                color=STATE_INFO[3][1], label="Osteoblast (cum)")
        ax.plot(days, df["Cells_Apoptotic_Cum"].to_numpy(), ls="--", lw=1.6,
                color=STATE_INFO[2][1], label="Apoptotic (cum)")
    _vlines(ax); ax.set(title="Cell counts", xlabel="Day", ylabel="Count")
    ax.legend(frameon=False, ncol=3, loc="upper left")

    panels = [
        (axarr[0,1], "Collagen_Avg", "Collagen (ECM)"),
        (axarr[0,2], "ALP_Avg", "ALP"),
        (axarr[1,0], "ACP_Avg", "Amorphous CaP"),
        (axarr[1,1], "HA_Avg",  "Hydroxyapatite"),
        (axarr[1,2], "Nodule_Index", "Nodule index (clustered HA mass)"),
        (axarr[2,0], "Ca_Avg",  "Ca²⁺"),
        (axarr[2,1], "PO4_Avg", "PO₄³⁻"),
        (axarr[2,2], "Supersat_Avg", "Supersaturation ⟨S⟩"),
    ]
    for a, col, title in panels:
        if col in df.columns:
            a.plot(days, df[col].to_numpy(), lw=2, color="black")
            _vlines(a)
            a.set(title=title, xlabel="Day")
        else:
            a.axis("off")

    fig.savefig(output)
    plt.close(fig)
    print(f"[viz] summary → {output}")

# ------------------------- Single-panel export utils -------------------------
def _save_field_imshow(arr: np.ndarray, title: str, outpath: Path,
                       cmap: str, log: bool = False,
                       vmin: float | None = None, vmax: float | None = None) -> None:
    """Save a single field as PNG with robust scaling."""
    fig, ax = plt.subplots(figsize=(4.8, 4.8))
    ax.set_title(title, pad=6)

    if log:
        if np.any(arr > 0):
            vmin = max(vmin or np.percentile(arr[arr > 0], 1), 1e-9)
            vmax = vmax or np.percentile(arr, 99.9)
            vmax = max(vmax, vmin * 1.01)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
            im = ax.imshow(arr, cmap=cmap, origin="lower", norm=norm)
        else:
            im = ax.imshow(arr, cmap=cmap, origin="lower")
    else:
        if vmin is None: vmin = np.percentile(arr, 1)
        if vmax is None: vmax = np.percentile(arr, 99.5)
        if vmin == vmax: vmax = vmin + 1e-9
        im = ax.imshow(arr, cmap=cmap, origin="lower", vmin=vmin, vmax=vmax)

    ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    out = Path(outpath).with_suffix(".png")
    fig.savefig(out)
    plt.close(fig)

def _compute_nodule_map(ha: np.ndarray) -> np.ndarray:
    """Robust nodule mask from HA (progressive threshold + local mean; fallback Otsu)."""
    if not np.any(ha > 0):
        return np.zeros_like(ha, dtype=float)

    nz = ha[ha > 0.0]
    H, W = ha.shape
    for q, local_frac in [(99.9, 0.6), (99.5, 0.55), (99.0, 0.5), (97.5, 0.45), (95.0, 0.40)]:
        tau = float(np.quantile(nz, q / 100.0))
        sum9 = (ha + np.roll(ha,1,0)+np.roll(ha,-1,0)+np.roll(ha,1,1)+np.roll(ha,-1,1)+
                np.roll(np.roll(ha,1,0),1,1)+np.roll(np.roll(ha,1,0),-1,1)+
                np.roll(np.roll(ha,-1,0),1,1)+np.roll(np.roll(ha,-1,0),-1,1))
        local_mean = sum9 / 9.0
        mask = ((ha > tau) & (local_mean > local_frac * tau))
        if int(mask.sum()) >= max(5, int(0.0005 * H * W)):
            return mask.astype(float)

    x = np.log1p(ha)
    hist, edges = np.histogram(x, bins=256)
    w0, w1 = np.cumsum(hist), np.cumsum(hist[::-1])[::-1]
    mu = np.cumsum(hist * (edges[:-1] + edges[1:]) / 2)
    mu_t = mu[-1]
    sigma_b2 = (mu_t * w0 - mu) ** 2 / (w0 * (w1 + 1e-12) + 1e-12)
    t_idx = int(np.nanargmax(sigma_b2))
    thr = (edges[t_idx] + edges[t_idx + 1]) / 2
    return (x > thr).astype(float)

def _plot_cell_states(cells_df: "pl.DataFrame", shape_hw: tuple[int, int],
                      day: int, outpath: Path, src_dir: Path | None = None,
                      max_points_per_state: int = 2000) -> None:
    """
    Scatter agents with crisp markers and counts per state (PNG only).
    If src_dir is provided, overlays **Apoptotic (cumulative)** using all
    deaths_{d}.csv for d ≤ day.
    """
    rng = np.random.default_rng(42)
    H, W = shape_hw
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.set_title(f"Cell states — Day {day}", pad=6)
    ax.set_xlim(0, W-1); ax.set_ylim(0, H-1)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)

    handles, labels = [], []

    # Live states
    for st, (lbl, colr, marker) in STATE_INFO.items():
        sub = cells_df.filter(pl.col("state") == st)
        if sub.height == 0:
            continue
        xs = sub["x"].to_numpy(); ys = sub["y"].to_numpy()
        if sub.height > max_points_per_state:
            idx = rng.choice(sub.height, size=max_points_per_state, replace=False)
            xs, ys = xs[idx], ys[idx]
        ax.scatter(xs, ys, c=colr, marker=marker, s=18,
                   edgecolor="black", linewidths=0.25, alpha=0.95, zorder=3)
        lbl_pretty = "Apoptotic (present)" if lbl == "Apoptotic" else lbl
        handles.append(Line2D([], [], ls="", marker=marker, markersize=6,
                              markerfacecolor=colr, markeredgecolor="black"))
        labels.append(f"{lbl_pretty} (n={sub.height})")

    # Cumulative deaths overlay (X markers, slightly larger)
    if src_dir is not None:
        xs, ys = _load_cumulative_deaths(src_dir, day)
        if xs.size > 0:
            lbl, colr, marker = STATE_INFO[2]
            ax.scatter(xs, ys, c=colr, marker=marker, s=26,
                       edgecolor="black", linewidths=0.4, alpha=0.95, zorder=4)
            handles.append(Line2D([], [], ls="", marker=marker, markersize=6,
                                  markerfacecolor=colr, markeredgecolor="black"))
            labels.append(f"Apoptotic (n={xs.size})")

    ax.legend(handles, labels, frameon=False, loc="lower right", ncol=1)
    fig.tight_layout()
    out = Path(outpath).with_suffix(".png")
    fig.savefig(out)
    plt.close(fig)

# ----------------------------- Public API ------------------------------------
def export_single_panels(snapshot_dir: str | Path,
                         days: Sequence[int],
                         out_dir: str | Path = "panels_pub") -> None:
    """
    Exports SEPARATE publication-ready panels for each requested day (PNG only):
      - cell_states_day{d}.png   (Active, Osteoblast, and Apoptotic cumulative overlay)
      - nodule_day{d}.png
      - ha_day{d}.png            [log scale]
      - collagen_day{d}.png
      - density_day{d}.png
    """
    src, dst = Path(snapshot_dir), Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)
    for day in days:
        try:
            col = np.load(src / f"col_{day}.npy")
            ha  = np.load(src / f"ha_{day}.npy")
            dens = np.load(src / f"density_{day}.npy")
            cells = pl.read_csv(src / f"cells_{day}.csv")
        except FileNotFoundError:
            print(f"[viz] panels: missing snapshot day {day}")
            continue

        # 1) Cell states (with cumulative deaths overlay)
        _plot_cell_states(cells, shape_hw=ha.shape, day=day,
                          outpath=dst / f"cell_states_day{day}",
                          src_dir=src)

        # 2) Nodule map (robust, non-cumulative; or switch to cumulative logic if desired)
        nodule = _compute_nodule_map(ha)
        _save_field_imshow(nodule, f"Nodule map — Day {day}",
                           dst / f"nodule_day{day}", cmap=CMAP["nodule"],
                           log=False, vmin=0.0, vmax=1.0)

        # 3) HA (log scale)
        _save_field_imshow(ha, f"Hydroxyapatite (HA) — Day {day}",
                           dst / f"ha_day{day}", cmap=CMAP["ha"], log=True)

        # 4) Collagen
        _save_field_imshow(col, f"Collagen — Day {day}",
                           dst / f"collagen_day{day}", cmap=CMAP["collagen"], log=False)

        # 5) Cell density (live occupancy)
        _save_field_imshow(dens.astype(float), f"Cell density — Day {day}",
                           dst / f"density_day{day}", cmap=CMAP["density"], log=False)

        print(f"[viz] wrote panels for day {day} → {dst.resolve()}")
