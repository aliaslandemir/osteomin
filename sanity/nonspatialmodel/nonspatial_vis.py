#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.modulatoreff import drug_response_summary


COLORS = {
    "preob": "#0072B2",
    "ob": "#D55E00",
    "total": "#1f2937",
    "dead_preob": "#E45756",
    "dead_ob": "#B279A2",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Plot nonspatial OsteoMin outputs (timeseries.csv) into main_out-like essentials."
    )
    p.add_argument(
        "--in-csv",
        type=str,
        default=str(SCRIPT_DIR / "out" / "timeseries.csv"),
        help="input timeseries CSV from nonspatialmodel.py",
    )
    p.add_argument(
        "--run-json",
        type=str,
        default=str(SCRIPT_DIR / "out" / "run.json"),
        help="optional run config JSON produced by nonspatialmodel.py",
    )
    p.add_argument(
        "--outdir",
        type=str,
        default=str(SCRIPT_DIR / "out"),
        help="output directory for generated figures",
    )
    p.add_argument("--alp-plot-hours", type=int, default=240)
    p.add_argument("--collagen-plot-hours", type=int, default=336)
    p.add_argument("--mineral-plot-hours", type=int, default=672)
    return p.parse_args()


def _set_white(ax: plt.Axes) -> None:
    ax.set_facecolor("white")
    ax.grid(axis="y", alpha=0.12)
    ax.grid(axis="x", alpha=0.0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _save_figure(fig: plt.Figure, outdir: Path, basename: str) -> None:
    outdir.mkdir(parents=True, exist_ok=True)
    pdf_dir = outdir / "figures_pdf"
    pdf_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / f"{basename}.png", dpi=300, bbox_inches="tight")
    fig.savefig(pdf_dir / f"{basename}.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _limit_days(df: pd.DataFrame, hours: int | None) -> pd.DataFrame:
    if hours is None or hours <= 0:
        return df
    max_day = float(hours) / 24.0
    return df[df["day"] <= max_day].copy()


def _plot_cumulative_components(
    df: pd.DataFrame,
    outdir: Path,
    *,
    base_col: str,
    ob_col: str,
    total_col: str,
    plot_hours: int | None,
    basename: str,
    title: str,
) -> None:
    d = _limit_days(df, plot_hours)
    if d.empty:
        return
    days = d["day"].to_numpy()
    base = d[base_col].to_numpy(dtype=float)
    ob = d[ob_col].to_numpy(dtype=float)
    total = d[total_col].to_numpy(dtype=float)
    denom = float(total[-1]) if len(total) else 0.0
    if denom > 0:
        base = base / denom
        ob = ob / denom

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(days, base, lw=2.0, color="#0ea5e9", label="Basal/live-cell")
    ax.plot(days, ob, lw=2.0, color="#f97316", label="OB")
    ax.set_xlabel("Day")
    ax.set_ylabel("Fraction of final total")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, outdir, basename)


def _load_field_cfg(run_json_path: Path) -> object | None:
    if not run_json_path.exists():
        return None
    payload = json.loads(run_json_path.read_text(encoding="utf-8"))
    cfg = payload.get("config", {})
    abm_cfg = {}
    if not isinstance(cfg, dict) or not cfg:
        cfg = payload.get("field_cfg", {})
        abm_cfg = payload.get("abm_cfg", {})
    if not isinstance(cfg, dict):
        return None
    if not isinstance(abm_cfg, dict):
        abm_cfg = {}
    return SimpleNamespace(
        osteo_induction_hours=abm_cfg.get("osteo_induction_hours", cfg.get("osteo_induction_hours", 72)),
        media_interval=cfg.get("media_interval", 48),
        aa_initial=cfg.get("aa_initial", 50.0),
        aa_half_life_hours=cfg.get("aa_half_life_hours", 48.0),
        collagen_aa_half_sat=cfg.get("collagen_aa_half_sat", 10.0),
        dex_conc=cfg.get("dex_conc", 0.0),
        mk4_conc=cfg.get("mk4_conc", 0.0),
        alp_osteo_rate=cfg.get("alp_osteo_rate", 0.0),
        collagen_osteo_rate=cfg.get("collagen_osteo_rate", 0.0),
        ha_osteo_rate=cfg.get("ha_osteo_rate", 0.0),
        alp_dex_ref_conc=cfg.get("alp_dex_ref_conc", 20.0),
        collagen_dex_ref_conc=cfg.get("collagen_dex_ref_conc", 60.0),
        ha_dex_ref_conc=cfg.get("ha_dex_ref_conc", 50.0),
        alp_mk4_ref_conc=cfg.get("alp_mk4_ref_conc", 0.6),
        collagen_mk4_ref_conc=cfg.get("collagen_mk4_ref_conc", 8.0),
        ha_mk4_ref_conc=cfg.get("ha_mk4_ref_conc", 0.7),
        alp_a_d=cfg.get("alp_a_d", 0.0),
        alp_b_d=cfg.get("alp_b_d", 0.0),
        collagen_a_d=cfg.get("collagen_a_d", 0.0),
        collagen_b_d=cfg.get("collagen_b_d", 0.0),
        ha_a_d=cfg.get("ha_a_d", 0.0),
        ha_b_d=cfg.get("ha_b_d", 0.0),
        alp_a_k=cfg.get("alp_a_k", 0.0),
        alp_b_k=cfg.get("alp_b_k", 0.0),
        collagen_a_k=cfg.get("collagen_a_k", 0.0),
        collagen_b_k=cfg.get("collagen_b_k", 0.0),
        ha_a_k=cfg.get("ha_a_k", 0.0),
        ha_b_k=cfg.get("ha_b_k", 0.0),
        dex_ref_conc=20.0,
        mk4_ref_conc=0.6,
    )


def main() -> None:
    args = parse_args()
    in_csv = Path(args.in_csv).expanduser()
    if not in_csv.is_absolute():
        in_csv = (PROJECT_ROOT / in_csv).resolve()
    if not in_csv.exists():
        raise SystemExit(f"missing input CSV: {in_csv}")
    outdir = Path(args.outdir).expanduser()
    if not outdir.is_absolute():
        outdir = (PROJECT_ROOT / outdir).resolve()

    df = pd.read_csv(in_csv)
    needed = [
        "day",
        "n_preob",
        "n_osteoblast",
        "n_total",
        "dead_preob_today",
        "dead_ob_today",
        "committed_today",
        "alp_total",
        "collagen_total",
        "ha_total",
    ]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"input CSV missing columns: {missing}")

    if "dead_total_today" not in df.columns:
        df["dead_total_today"] = df["dead_preob_today"] + df["dead_ob_today"]
    if "dead_preob_cum" not in df.columns:
        df["dead_preob_cum"] = df["dead_preob_today"].cumsum()
    if "dead_ob_cum" not in df.columns:
        df["dead_ob_cum"] = df["dead_ob_today"].cumsum()
    if "dead_total_cum" not in df.columns:
        df["dead_total_cum"] = df["dead_total_today"].cumsum()

    days = df["day"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(days, df["n_preob"], lw=2.0, label="preOB", color=COLORS["preob"])
    ax.plot(days, df["n_osteoblast"], lw=2.0, label="Osteoblast", color=COLORS["ob"])
    ax.plot(days, df["n_total"], lw=2.0, ls="--", label="Total", color=COLORS["total"])
    ax.set_xlabel("Day")
    ax.set_ylabel("Cells")
    ax.set_title("Cell states over time")
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, outdir, "01_states_counts")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.plot(days, df["dead_preob_cum"], lw=2.0, label="preOB→Dead (cum)", color=COLORS["dead_preob"])
    ax.plot(days, df["dead_ob_cum"], lw=2.0, label="OB→Dead (cum)", color=COLORS["dead_ob"])
    ax.plot(days, df["dead_total_cum"], lw=2.0, ls="--", label="All deaths (cum)", color="#111827")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative count")
    ax.set_title("Cumulative events")
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, outdir, "02_events_cumulative")

    fig, ax = plt.subplots(figsize=(6.2, 3.6))
    _set_white(ax)
    ax.bar(days, df["committed_today"], label="Committed (today)", color="#fdae61", alpha=0.9)
    ax.bar(days, df["dead_preob_today"], bottom=df["committed_today"], label="Died (preOB)", color=COLORS["dead_preob"], alpha=0.9)
    ax.bar(
        days,
        df["dead_ob_today"],
        bottom=df["committed_today"] + df["dead_preob_today"],
        label="Died (OB)",
        color=COLORS["dead_ob"],
        alpha=0.9,
    )
    ax.set_xlabel("Day")
    ax.set_ylabel("Count")
    ax.set_title("Daily events")
    ax.legend(frameon=False, loc="upper right")
    _save_figure(fig, outdir, "04_events_daily")

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    _set_white(ax)
    ax.plot(days, df["alp_total"], lw=2.0, color="#2563eb", label="ALP total")
    ax.plot(days, df["collagen_total"], lw=2.0, color="#059669", label="Collagen total")
    ax.plot(days, df["ha_total"], lw=2.0, color="#b45309", label="HAp total")
    ax.set_xlabel("Day")
    ax.set_ylabel("Cumulative total")
    ax.set_title("Field totals over time")
    ax.legend(frameon=False, loc="upper left")
    _save_figure(fig, outdir, "09_field_totals_absolute")

    if all(c in df.columns for c in ("alp_preob_total", "alp_ob_total", "alp_total")):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="alp_preob_total",
            ob_col="alp_ob_total",
            total_col="alp_total",
            plot_hours=args.alp_plot_hours,
            basename="06_alp_total",
            title="ALP accumulation (normalized cumulative)",
        )
    if all(c in df.columns for c in ("collagen_preob_total", "collagen_ob_total", "collagen_total")):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="collagen_preob_total",
            ob_col="collagen_ob_total",
            total_col="collagen_total",
            plot_hours=args.collagen_plot_hours,
            basename="07_collagen_total",
            title="Collagen accumulation (normalized cumulative)",
        )
    if all(c in df.columns for c in ("mineral_preob_total", "mineral_ob_total", "ha_total")):
        _plot_cumulative_components(
            df,
            outdir,
            base_col="mineral_preob_total",
            ob_col="mineral_ob_total",
            total_col="ha_total",
            plot_hours=args.mineral_plot_hours,
            basename="05_minerals_cumulative",
            title="HAp accumulation (normalized cumulative)",
        )

    if "nodule_count" in df.columns:
        nod = df["nodule_count_ever"] if "nodule_count_ever" in df.columns else df["nodule_count"]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        _set_white(ax)
        ax.plot(days, nod, lw=2.0, marker="o", ms=4, color="#0891b2", label="Nodules (ever formed)")
        if "nodule_count_current" in df.columns:
            ax.plot(days, df["nodule_count_current"], lw=1.5, ls="--", color="#64748b", label="Nodules (current)")
        ax.set_xlabel("Day")
        ax.set_ylabel("Count")
        ax.set_title("Nodule count over time")
        ax.legend(frameon=False, loc="upper left")
        _save_figure(fig, outdir, "08_nodule_count")

    if all(c in df.columns for c in ("msd_effective", "avg_move_count")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days, df["msd_effective"], lw=2.0, color="#0f766e", label="Effective MSD")
        ax.plot(days, df["avg_move_count"], lw=2.0, color="#7c3aed", label="Average move count")
        ax.set_xlabel("Day")
        ax.set_ylabel("Value")
        ax.set_title("Cell motility summaries")
        ax.legend(frameon=False, loc="upper right")
        _save_figure(fig, outdir, "03_cell_motility")

    if all(c in df.columns for c in ("avg_division_interval", "avg_live_age_hours")):
        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days, df["avg_division_interval"], lw=2.0, color="#ea580c", label="Average division interval")
        ax.plot(days, df["avg_live_age_hours"], lw=2.0, color="#475569", label="Average live-cell age")
        ax.set_xlabel("Day")
        ax.set_ylabel("Hours")
        ax.set_title("Cell cycle summaries")
        ax.legend(frameon=False, loc="upper left")
        _save_figure(fig, outdir, "12_cell_cycle_summary")

    run_json = Path(args.run_json).expanduser()
    if not run_json.is_absolute():
        run_json = (PROJECT_ROOT / run_json).resolve()
    field_cfg = _load_field_cfg(run_json)
    if field_cfg is not None:
        total_hours = int(round(float(days.max()) * 24.0)) if len(days) else 0
        hours = np.arange(total_hours + 1, dtype=float)
        aa = np.zeros_like(hours, dtype=float)
        induction_h = int(getattr(field_cfg, "osteo_induction_hours", 72))
        media_interval = int(getattr(field_cfg, "media_interval", 48))
        aa_initial = max(0.0, float(getattr(field_cfg, "aa_initial", 50.0)))
        aa_half_life = max(1e-9, float(getattr(field_cfg, "aa_half_life_hours", 48.0)))
        aa_km = max(1e-9, float(getattr(field_cfg, "collagen_aa_half_sat", 10.0)))
        if aa_initial > 0.0:
            k = np.log(2.0) / aa_half_life
            for i, h in enumerate(hours):
                if h < induction_h:
                    continue
                since = (h - induction_h) if media_interval <= 0 else ((h - induction_h) % media_interval)
                aa[i] = aa_initial * np.exp(-k * since)
        aa_gate = np.where(aa > 0.0, aa / (aa_km + aa), 0.0)
        days_hr = hours / 24.0

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days_hr, aa, lw=2.0, color="#0f766e", label="AA(t)")
        ax.axvline(induction_h / 24.0, color="#64748b", lw=1.2, ls="--", label="Osteogenic induction")
        ax.set_xlabel("Day")
        ax.set_ylabel("AA concentration")
        ax.set_title("Ascorbate pulse-decay schedule")
        ax.legend(frameon=False, loc="upper right")
        _save_figure(fig, outdir, "10_ascorbate_schedule")

        fig, ax = plt.subplots(figsize=(6.4, 3.8))
        _set_white(ax)
        ax.plot(days_hr, aa_gate, lw=2.0, color="#16a34a", label="A/(K_m + A)")
        ax.plot(days_hr, aa, lw=1.2, ls="--", color="#94a3b8", label="AA(t)")
        ax.axvline(induction_h / 24.0, color="#64748b", lw=1.2, ls=":")
        ax.set_xlabel("Day")
        ax.set_ylabel("Collagen gate")
        ax.set_title("Ascorbate-gated collagen activation")
        ax.legend(frameon=False, loc="upper right")
        _save_figure(fig, outdir, "11_ascorbate_collagen_gate")

        summary = drug_response_summary(field_cfg)
        labels = ["ALP", "Collagen", "HAp"]
        m_d = [summary["alp_m_dex"], summary["collagen_m_dex"], summary["ha_m_dex"]]
        m_k = [summary["alp_m_mk4"], summary["collagen_m_mk4"], summary["ha_m_mk4"]]
        m_t = [summary["alp_m_total"], summary["collagen_m_total"], summary["ha_m_total"]]
        x = np.arange(len(labels), dtype=float)
        w = 0.25
        fig, ax = plt.subplots(figsize=(7.2, 4.0))
        _set_white(ax)
        ax.bar(x - w, m_d, width=w, color="#ef4444", label="Dex factor")
        ax.bar(x, m_k, width=w, color="#22c55e", label="MK-4 factor")
        ax.bar(x + w, m_t, width=w, color="#2563eb", label="Combined factor")
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        ax.set_ylabel("Multiplier")
        ax.set_title("Drug modulation summary")
        ax.legend(frameon=False, loc="upper right")
        _save_figure(fig, outdir, "13_drug_modulation_summary")

    print(f"[nonspatial_vis] wrote figures to {outdir}")


if __name__ == "__main__":
    main()
