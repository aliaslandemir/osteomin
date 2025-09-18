from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

__all__ = ["plot_summary",  "export_spatial_for_day"]

# --- color palette (white background friendly) ---
COLORS = {
    "active":      "#2ca25f",  # green
    "ob":          "#67a9cf",  # blue
    "dead_active": "#fdb863",  # orange
    "dead_ob":     "#8073ac",  # purple
    "text":        "#111111",
}


def _set_white():
    plt.style.use("default")
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.edgecolor": "#333333",
        "grid.color": "#eaeaea",
        "axes.grid": False,
        "font.size": 12,
    })


def plot_summary(df: pd.DataFrame, outdir: Path) -> None:
    _set_white()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    days = df["day"].values

    # 1) states counts
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, df["n_active"], label="Active", color=COLORS["active"], lw=2)
    ax.plot(days, df["n_osteoblast"], label="Osteoblast", color=COLORS["ob"], lw=2)
    ax.plot(days, df["n_total"], label="Total", color="black", lw=1.5, alpha=0.7)
    ax.set_xlabel("Day"); ax.set_ylabel("Cells"); ax.set_title("Agent-based cell counts")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(outdir / "01_states_counts.png", dpi=200); plt.close(fig)

    # 2) cumulative events (compute if not present)
    if "dead_total_cum" not in df.columns:
        tmp = df.copy()
        tmp["dead_total_today"] = tmp["dead_active_today"] + tmp["dead_ob_today"]
        tmp["dead_active_cum"] = tmp["dead_active_today"].cumsum()
        tmp["dead_ob_cum"] = tmp["dead_ob_today"].cumsum()
        tmp["dead_total_cum"] = tmp["dead_total_today"].cumsum()
    else:
        tmp = df

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, tmp["dead_active_cum"], label="Active→Dead (cum)", color=COLORS["dead_active"], lw=2)
    ax.plot(days, tmp["dead_ob_cum"], label="Osteoblast→Dead (cum)", color=COLORS["dead_ob"], lw=2)
    ax.plot(days, tmp["dead_total_cum"], label="All deaths (cum)", color="#444", lw=1.5, ls="--")
    ax.set_xlabel("Day"); ax.set_ylabel("Cumulative count"); ax.set_title("Cumulative events")
    ax.legend(frameon=False)
    fig.savefig(outdir / "02_events_cumulative.png", dpi=200); plt.close(fig)

    # 3) matrix + biochem
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(days, df["collagen_total"], label="Collagen (total)", lw=2, color=COLORS["ob"])
    ax.plot(days, df["alp_total"], label="ALP (total)", lw=2, color="#d33682")
    ax2 = ax.twinx()
    ax2.plot(days, df["free_ca"], label="free Ca", lw=1.8, color="#1f77b4", alpha=0.8)
    ax2.plot(days, df["free_pi"], label="free Pi", lw=1.8, color="#ff7f0e", alpha=0.8)
    ax.set_xlabel("Day"); ax.set_ylabel("ECM (a.u.)"); ax2.set_ylabel("Ions (a.u.)")
    ax.set_title("Matrix & biochem")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, frameon=False, loc="upper left")
    fig.savefig(outdir / "03_matrix_biochem.png", dpi=200); plt.close(fig)

    # 4) daily events
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(days, df["committed_today"], label="Committed (today)", color="#fdae61", alpha=0.9)
    ax.bar(days, df["dead_active_today"], bottom=df["committed_today"],
           label="Died (active)", color=COLORS["dead_active"], alpha=0.9)
    ax.bar(days, df["dead_ob_today"], bottom=df["committed_today"] + df["dead_active_today"],
           label="Died (osteoblast)", color=COLORS["dead_ob"], alpha=0.9)
    ax.set_xlabel("Day"); ax.set_ylabel("Count"); ax.set_title("Daily events")
    ax.legend(frameon=False, ncol=3)
    fig.savefig(outdir / "04_events_daily.png", dpi=200); plt.close(fig)

    # Roll-up figure
    fig = plt.figure(figsize=(10, 10))
    axes = [fig.add_subplot(3, 1, i + 1) for i in range(3)]
    axes[0].plot(days, df["n_active"], color=COLORS["active"], lw=2, label="Active")
    axes[0].plot(days, df["n_osteoblast"], color=COLORS["ob"], lw=2, label="Osteoblast")
    axes[0].plot(days, df["n_total"], color="black", lw=1.5, alpha=0.7, label="Total")
    axes[0].legend(frameon=False); axes[0].set_ylabel("Cells"); axes[0].set_title("Cell states over time")

    axes[1].plot(days, tmp["dead_total_cum"], color="#444", lw=1.5, ls="--", label="All deaths (cum)")
    axes[1].plot(days, tmp["dead_active_cum"], color=COLORS["dead_active"], lw=2, label="Active→Dead")
    axes[1].plot(days, tmp["dead_ob_cum"], color=COLORS["dead_ob"], lw=2, label="OB→Dead")
    axes[1].legend(frameon=False); axes[1].set_ylabel("Cumulative")

    axes[2].plot(days, df["free_ca"], color="#1f77b4", lw=1.8, label="free Ca")
    axes[2].plot(days, df["free_pi"], color="#ff7f0e", lw=1.8, label="free Pi")
    axes[2].legend(frameon=False); axes[2].set_xlabel("Day"); axes[2].set_ylabel("Ions (a.u.)")

    fig.tight_layout()
    fig.savefig(outdir / "summary.png", dpi=220); plt.close(fig)


def _rgba_canvas(h, w, bg=(1, 1, 1)):
    arr = np.ones((h, w, 3), dtype=float)
    arr[..., 0] *= bg[0]; arr[..., 1] *= bg[1]; arr[..., 2] *= bg[2]
    return arr


def export_spatial_for_day(
    outdir: Path,
    day: int,
    state: np.ndarray,
    deaths_active: np.ndarray,
    deaths_ob: np.ndarray,
    dead_total_cum: int = 0,  # NEW: cumulative total to show in title
) -> None:
    _set_white()
    h, w = state.shape
    canvas = _rgba_canvas(h, w, (1, 1, 1))

    # paint base states
    active_mask = (state == 1)
    ob_mask = (state == 3)
    canvas[active_mask] = (44/255, 162/255, 95/255)    # green
    canvas[ob_mask]     = (103/255, 169/255, 207/255)  # blue

    # paint deaths (overlay for today's deaths)
    canvas[deaths_active.astype(bool)] = (253/255, 184/255, 99/255)  # orange
    canvas[deaths_ob.astype(bool)]     = (128/255, 115/255, 172/255) # purple

    # counts
    n_act = int(active_mask.sum())
    n_ob  = int(ob_mask.sum())
    n_da  = int(deaths_active.sum())
    n_do  = int(deaths_ob.sum())

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.imshow(canvas, origin="lower", interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])

    ax.set_title(
        f"Day {day:02d}  |  Total {n_act + n_ob} live  |  "
        f"act: {n_act}, osteo: {n_ob}, dead today: {n_da + n_do}, dead cum: {dead_total_cum}",
        loc="left", fontsize=18, fontweight="bold"
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


