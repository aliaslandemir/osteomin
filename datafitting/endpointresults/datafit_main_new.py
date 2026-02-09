#!/usr/bin/env python3
"""
Build linear endpoint-based datafit curves and dose modifiers.

Inputs (endpoints):
  - alp_result.txt       (C1, C2 at 240 h)
  - collagen_result.txt  (C1, C2 at 336 h)
  - ars_result.txt       (C1, C2 at 672 h)

Outputs (NPZ):
  1) datafit_endpoint.npz
     - time_hours
     - alp_preob_hourly, alp_ob_hourly (per-hour increments)
     - collagen_preob_hourly, collagen_ob_hourly
     - mineral_preob_hourly, mineral_ob_hourly
       (arrays are [n_conditions, n_time])
     - alp_cond_C1..C8, collagen_cond_C1..C8, mineral_cond_C1..C8 (cumulative totals)
     - alp_basal, alp_osteo (baseline components)
     - collagen_C1, collagen_C2 (basal and osteo increment)
     - mineral_C1, mineral_C2 (basal and osteo increment)
     - conditions, D_nM, K_nM, dose_effects_* and rates_*
     - meta: measurement hours + rates + gating params

  2) dk_modifiers.npz
     - alp_modifiers, collagen_modifiers, ars_modifiers
       (arrays: [bD_low, bD_high, bK_low, bK_high])

Notes:
  - Basal = C1; Osteo increment = (C2 - C1)
  - Osteogenic supplements start at t_osteo (default 72 h)
  - Collagen osteo increment is AA-gated (pulse + decay)
  - Mineral uses collagen gate + Hill-type accumulation
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _read_endpoint_table(path: Path) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] < 2:
            raise ValueError
    except Exception:
        df = pd.read_csv(path, sep=r"\s+", engine="python")

    df.columns = [c.strip() for c in df.columns]
    colmap = {}
    for c in df.columns:
        cl = c.lower()
        if cl == "condition":
            colmap[c] = "Condition"
        elif cl == "mean":
            colmap[c] = "Mean"
    df = df.rename(columns=colmap)

    if "Condition" not in df.columns or "Mean" not in df.columns:
        raise ValueError(f"{path}: must include Condition and Mean.")

    df["Condition"] = df["Condition"].astype(str).str.strip()
    df["Mean"] = pd.to_numeric(df["Mean"], errors="coerce")
    if df["Mean"].isna().any():
        bad = df[df["Mean"].isna()]
        raise ValueError(f"{path}: unparsable Mean values:\n{bad}")
    return df


def _endpoint_c1_c2(df: pd.DataFrame, name: str) -> tuple[float, float]:
    def _get(cond: str) -> float:
        v = df.loc[df["Condition"] == cond, "Mean"]
        if v.empty:
            raise ValueError(f"{name}: missing {cond}")
        return float(v.iloc[0])

    c1 = _get("C1")
    c2 = _get("C2")
    if np.isclose(c2 - c1, 0.0):
        raise ValueError(f"{name}: C2 - C1 is ~0")
    return c1, c2


def _linear_curve(rate: float, hours: int, time_hours: np.ndarray) -> np.ndarray:
    return rate * np.minimum(time_hours, float(hours))


def _build_curves(
    c1: float, c2: float, hours: int, time_hours: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    basal_rate = c1 / float(hours)
    osteo_rate = (c2 - c1) / float(hours)
    basal = _linear_curve(basal_rate, hours, time_hours)
    osteo = _linear_curve(osteo_rate, hours, time_hours)
    total = basal + osteo
    return basal, osteo, total


def _osteo_linear_component(
    target: float, hours: int, time_hours: np.ndarray, t_osteo: int
) -> np.ndarray:
    """Linear osteo component starting at t_osteo and ending at hours."""
    out = np.zeros_like(time_hours, dtype=float)
    start = max(0, int(t_osteo))
    end = max(start, int(hours))
    if end <= start:
        return out
    rate = target / float(end - start)
    for i, t in enumerate(time_hours):
        if t < start:
            out[i] = 0.0
        elif t <= end:
            out[i] = rate * (t - start)
        else:
            out[i] = rate * (end - start)
    return out


def _aa_profile(
    time_hours: np.ndarray,
    t_osteo: int,
    media_interval_h: int,
    half_life_h: float,
    A0: float,
) -> np.ndarray:
    """Pulse + exponential decay AA profile after osteogenic start."""
    A = np.zeros_like(time_hours, dtype=float)
    if media_interval_h <= 0 or half_life_h <= 0:
        return A
    k = np.log(2.0) / float(half_life_h)
    for i, t in enumerate(time_hours):
        if t < t_osteo:
            A[i] = 0.0
        else:
            rel = t - t_osteo
            since = rel % float(media_interval_h)
            A[i] = A0 * np.exp(-k * since)
    return A


def _osteo_gated_component(
    target: float,
    hours: int,
    time_hours: np.ndarray,
    t_osteo: int,
    media_interval_h: int,
    half_life_h: float,
    A0: float,
    Km: float,
) -> np.ndarray:
    """Osteo component scaled by AA gating so endpoint matches target."""
    out = np.zeros_like(time_hours, dtype=float)
    A = _aa_profile(time_hours, t_osteo, media_interval_h, half_life_h, A0)
    gate = A / (Km + A) if Km > 0 else np.ones_like(A)
    start = max(0, int(t_osteo))
    end = max(start, int(hours))
    if end <= start:
        return out
    mask = (time_hours >= start) & (time_hours <= end)
    area = np.sum(gate[mask])
    if area <= 0:
        return out
    rate = target / float(area)
    acc = 0.0
    for i, t in enumerate(time_hours):
        if t < start:
            out[i] = 0.0
        elif t <= end:
            acc += rate * gate[i]
            out[i] = acc
        else:
            out[i] = acc
    return out


def _hill_curve(
    time_hours: np.ndarray,
    t_start: float,
    t_end: float,
    n: float,
    tau: float,
    target: float,
) -> np.ndarray:
    """Cumulative Hill-shaped curve that reaches target at t_end."""
    out = np.zeros_like(time_hours, dtype=float)
    if t_start >= t_end:
        return out
    tau = max(float(tau), 1.0)
    n = max(float(n), 1.0)
    # rate shape
    def _rate(t):
        x = max(t - t_start, 0.0)
        return (x**n) / (tau**n + x**n)

    # compute area for scaling
    area = 0.0
    for t in time_hours:
        if t < t_start or t > t_end:
            continue
        area += _rate(t)
    if area <= 0:
        return out
    beta = target / area

    acc = 0.0
    for i, t in enumerate(time_hours):
        if t < t_start:
            out[i] = 0.0
        elif t <= t_end:
            acc += beta * _rate(t)
            out[i] = acc
        else:
            out[i] = acc
    return out


def _load_modifiers(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        return {}
    df = pd.read_csv(path)
    out: dict[str, np.ndarray] = {}
    for _, row in df.iterrows():
        assay = str(row["Assay"]).strip().lower()
        arr = np.array(
            [
                float(row["bD_low"]),
                float(row["bD_high"]),
                float(row["bK_low"]),
                float(row["bK_high"]),
            ],
            dtype=float,
        )
        out[assay] = arr
    return out


def _condition_table() -> pd.DataFrame:
    cond = pd.DataFrame(
        {
            "Condition": ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"],
            "D_nM": [0, 0, 10, 10, 100, 100, 100, 0],
            "K_uM": [0, 0, 0.1, 10, 0.1, 10, 0, 10],
        }
    )
    cond["K_nM"] = cond["K_uM"] * 1000.0
    return cond


def _dose_effects_for_conditions(mods: np.ndarray, cond: pd.DataFrame) -> np.ndarray:
    bD_low, bD_high, bK_low, bK_high = mods
    d = cond["D_nM"].to_numpy(float)
    k = cond["K_nM"].to_numpy(float)
    d_low = (np.isclose(d, 10.0)).astype(float)
    d_high = (np.isclose(d, 100.0)).astype(float)
    k_low = (np.isclose(k, 100.0)).astype(float)
    k_high = (np.isclose(k, 10000.0)).astype(float)
    return bD_low * d_low + bD_high * d_high + bK_low * k_low + bK_high * k_high


def _plot_assay_curves(
    name: str,
    time_hours: np.ndarray,
    curves: dict[str, np.ndarray],
    outdir: Path,
) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Serif",
            "font.size": 9.5,
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "axes.linewidth": 0.9,
            "lines.linewidth": 1.2,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    colors = {
        "C1": "#7a7a7a",
        "C2": "#3b6fb6",
        "C3": "#2a9d8f",
        "C4": "#e0a100",
        "C5": "#b565a7",
        "C6": "#c84c3a",
        "C7": "#6aaed6",
        "C8": "#8c6d31",
    }
    t_days = time_hours / 24.0
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.set_facecolor("#FBFBFB")
    ax.grid(alpha=0.15, linestyle="--", linewidth=0.6)
    prefix = f"{name.lower()}_cond_"
    for cond, y in curves.items():
        if not cond.startswith(prefix):
            continue
        cid = cond.replace(prefix, "")
        lw = 2.2 if cid in ("C1", "C2") else 1.1
        alpha = 0.95 if cid in ("C1", "C2") else 0.7
        ax.plot(t_days, y, color=colors.get(cid, "#444444"), lw=lw, alpha=alpha, label=cid)
    ax.set_xlabel("Day")
    ax.set_ylabel(f"{name} (a.u.)")
    ax.set_title(f"{name}: endpoint-based curves (C1–C8)")
    ax.legend(frameon=False, ncol=4, fontsize=8, loc="upper left", bbox_to_anchor=(0.0, 1.02))
    fig.tight_layout()
    fig.savefig(outdir / f"{name.lower()}_curves.png", dpi=300)
    fig.savefig(outdir / f"{name.lower()}_curves.pdf")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--alp", default="alp_result.txt", help="ALP endpoint table")
    p.add_argument("--collagen", default="collagen_result.txt", help="Collagen endpoint table")
    p.add_argument("--ars", default="ars_result.txt", help="ARS endpoint table")
    p.add_argument("--alp_hours", type=int, default=240)
    p.add_argument("--collagen_hours", type=int, default=336)
    p.add_argument("--ars_hours", type=int, default=672)
    p.add_argument("--osteo_start_hours", type=int, default=72)
    p.add_argument("--media_interval_hours", type=int, default=48)
    p.add_argument("--aa_half_life_hours", type=float, default=3.0)
    p.add_argument("--aa0", type=float, default=1.0)
    p.add_argument("--km_collagen", type=float, default=1.0)
    p.add_argument("--collagen_threshold_frac", type=float, default=0.5)
    p.add_argument("--mineral_tmin_hours", type=int, default=336)
    p.add_argument("--mineral_hill_n", type=float, default=2.0)
    p.add_argument("--fit_summary", default="fit_outputs/dk_fit_summary.csv")
    p.add_argument("--outdir", default="fit_outputs")
    args = p.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    alp_df = _read_endpoint_table(Path(args.alp))
    col_df = _read_endpoint_table(Path(args.collagen))
    ars_df = _read_endpoint_table(Path(args.ars))

    alp_c1, alp_c2 = _endpoint_c1_c2(alp_df, "ALP")
    col_c1, col_c2 = _endpoint_c1_c2(col_df, "Collagen")
    ars_c1, ars_c2 = _endpoint_c1_c2(ars_df, "ARS")

    max_h = max(args.alp_hours, args.collagen_hours, args.ars_hours)
    time_hours = np.arange(0, max_h + 1, dtype=float)

    alp_basal = _linear_curve(alp_c1 / args.alp_hours, args.alp_hours, time_hours)
    alp_osteo = _osteo_linear_component(
        alp_c2 - alp_c1, args.alp_hours, time_hours, args.osteo_start_hours
    )
    alp_total = alp_basal + alp_osteo

    col_basal = _linear_curve(col_c1 / args.collagen_hours, args.collagen_hours, time_hours)
    col_osteo = _osteo_gated_component(
        col_c2 - col_c1,
        args.collagen_hours,
        time_hours,
        args.osteo_start_hours,
        args.media_interval_hours,
        args.aa_half_life_hours,
        args.aa0,
        args.km_collagen,
    )
    col_total = col_basal + col_osteo

    # mineral baseline (preOB/background) and osteo increment (Hill gated)
    mineral_preob_cum = _linear_curve(ars_c1 / args.ars_hours, args.ars_hours, time_hours)

    rates = {
        "alp_basal_rate": alp_c1 / args.alp_hours,
        "alp_osteo_rate": (alp_c2 - alp_c1) / args.alp_hours,
        "collagen_basal_rate": col_c1 / args.collagen_hours,
        "collagen_osteo_rate": (col_c2 - col_c1) / args.collagen_hours,
        "mineral_basal_rate": ars_c1 / args.ars_hours,
        "mineral_osteo_rate": (ars_c2 - ars_c1) / args.ars_hours,
    }

    cond = _condition_table()
    conds = cond["Condition"].tolist()

    modifiers = _load_modifiers(Path(args.fit_summary))
    alp_mod = modifiers.get("alp")
    col_mod = modifiers.get("collagen")
    ars_mod = modifiers.get("ars")

    curves: dict[str, np.ndarray] = {}

    def _apply_assay(name: str, basal: np.ndarray, osteo: np.ndarray, hours: int, mods: np.ndarray | None):
        if mods is None:
            mods = np.zeros(4, dtype=float)
        effects = _dose_effects_for_conditions(mods, cond)
        scales = 1.0 + effects
        # C1 has no osteogenic increment
        scales = np.where(cond["Condition"].to_numpy() == "C1", 0.0, scales)

        basal_rate = float(basal[-1]) / float(hours)
        osteo_rate = float(osteo[-1]) / float(hours)
        total_rate = basal_rate + osteo_rate * scales

        for cid, scale in zip(conds, scales):
            curves[f"{name.lower()}_cond_{cid}"] = basal + osteo * scale

        return effects, scales, basal_rate, osteo_rate, total_rate

    alp_eff, alp_scale, alp_b_rate, alp_o_rate, alp_rates = _apply_assay(
        "ALP", alp_basal, alp_osteo, args.alp_hours, alp_mod
    )
    col_eff, col_scale, col_b_rate, col_o_rate, col_rates = _apply_assay(
        "Collagen", col_basal, col_osteo, args.collagen_hours, col_mod
    )
    # mineral per-condition curves with collagen gate + Hill increase
    ars_eff, ars_scale, ars_b_rate, ars_o_rate, ars_rates = _apply_assay(
        "Mineral", mineral_preob_cum, np.zeros_like(mineral_preob_cum), args.ars_hours, ars_mod
    )
    mineral_curves: dict[str, np.ndarray] = {}
    mineral_ob_curves: dict[str, np.ndarray] = {}
    t_end = float(args.ars_hours)
    for cid in conds:
        c_curve = curves[f"collagen_cond_{cid}"]
        c_end = c_curve[int(args.collagen_hours)]
        c_th = float(args.collagen_threshold_frac) * float(c_end)
        t_cross = None
        for i, t in enumerate(time_hours):
            if t <= args.collagen_hours and c_curve[i] >= c_th:
                t_cross = float(t)
                break
        if t_cross is None:
            t_cross = float(args.collagen_hours)
        t_start = max(float(args.mineral_tmin_hours), float(args.osteo_start_hours), t_cross)
        # target osteogenic mineral increment for condition
        scale = float(ars_scale[conds.index(cid)])
        target_osteo = (ars_c2 - ars_c1) * scale
        tau = max(1.0, (t_end - t_start) / 2.0)
        hill = _hill_curve(time_hours, t_start, t_end, args.mineral_hill_n, tau, target_osteo)
        mineral_ob_curves[cid] = hill
        total = mineral_preob_cum + hill
        mineral_curves[cid] = total
        curves[f"mineral_cond_{cid}"] = total

    # use gated C1 as basal and gated C2 osteo increment
    mineral_c1_curve = mineral_preob_cum
    mineral_c2_curve = mineral_ob_curves["C2"]
    ars_total = mineral_preob_cum + mineral_c2_curve

    # per-hour increments (preOB/OB)
    alp_preob_inc = np.diff(alp_basal, prepend=0.0)
    alp_ob_inc_base = np.diff(alp_osteo, prepend=0.0)
    col_preob_inc = np.diff(col_basal, prepend=0.0)
    col_ob_inc_base = np.diff(col_osteo, prepend=0.0)
    min_preob_inc = np.diff(mineral_preob_cum, prepend=0.0)

    alp_preob_hourly = np.vstack([alp_preob_inc for _ in conds])
    col_preob_hourly = np.vstack([col_preob_inc for _ in conds])
    min_preob_hourly = np.vstack([min_preob_inc for _ in conds])

    alp_ob_hourly = np.vstack([alp_ob_inc_base * s for s in alp_scale])
    col_ob_hourly = np.vstack([col_ob_inc_base * s for s in col_scale])
    min_ob_hourly = np.vstack([np.diff(mineral_ob_curves[cid], prepend=0.0) for cid in conds])

    np.savez(
        outdir / "datafit_endpoint.npz",
        time_hours=time_hours,
        alp_preob_hourly=alp_preob_hourly,
        alp_ob_hourly=alp_ob_hourly,
        collagen_preob_hourly=col_preob_hourly,
        collagen_ob_hourly=col_ob_hourly,
        mineral_preob_hourly=min_preob_hourly,
        mineral_ob_hourly=min_ob_hourly,
        **curves,
        alp_basal=alp_basal,
        alp_osteo=alp_osteo,
        collagen_C1=col_basal,
        collagen_C2=col_osteo,
        mineral_C1=mineral_c1_curve,
        mineral_C2=mineral_c2_curve,
        conditions=np.array(conds, dtype="U2"),
        D_nM=cond["D_nM"].to_numpy(float),
        K_nM=cond["K_nM"].to_numpy(float),
        alp_dose_effects=alp_eff,
        collagen_dose_effects=col_eff,
        mineral_dose_effects=ars_eff,
        alp_rates=alp_rates,
        collagen_rates=col_rates,
        mineral_rates=ars_rates,
        meta={
            "alp_hours": int(args.alp_hours),
            "collagen_hours": int(args.collagen_hours),
            "ars_hours": int(args.ars_hours),
            "osteo_start_hours": int(args.osteo_start_hours),
            "media_interval_hours": int(args.media_interval_hours),
            "aa_half_life_hours": float(args.aa_half_life_hours),
            "km_collagen": float(args.km_collagen),
            "collagen_threshold_frac": float(args.collagen_threshold_frac),
            "mineral_tmin_hours": int(args.mineral_tmin_hours),
            "mineral_hill_n": float(args.mineral_hill_n),
            "note": "Hourly arrays are per-hour increments (preOB vs OB).",
            **rates,
        },
    )

    if modifiers:
        np.savez(
            outdir / "dk_modifiers.npz",
            alp_modifiers=alp_mod if alp_mod is not None else np.zeros(4),
            collagen_modifiers=col_mod if col_mod is not None else np.zeros(4),
            ars_modifiers=ars_mod if ars_mod is not None else np.zeros(4),
        )

    # CSV summary for quick inspection
    summary = pd.DataFrame(
        [
            {
                "Assay": "ALP",
                "C1": alp_c1,
                "C2": alp_c2,
                "hours": args.alp_hours,
                "basal_rate": alp_b_rate,
                "osteo_rate": alp_o_rate,
            },
            {
                "Assay": "Collagen",
                "C1": col_c1,
                "C2": col_c2,
                "hours": args.collagen_hours,
                "basal_rate": col_b_rate,
                "osteo_rate": col_o_rate,
            },
            {
                "Assay": "ARS",
                "C1": ars_c1,
                "C2": ars_c2,
                "hours": args.ars_hours,
                "basal_rate": ars_b_rate,
                "osteo_rate": ars_o_rate,
            },
        ]
    )
    summary.to_csv(outdir / "endpoint_rates_summary.csv", index=False)

    alp_rates_cond = [curves[f"alp_cond_{cid}"][-1] / args.alp_hours for cid in conds]
    col_rates_cond = [curves[f"collagen_cond_{cid}"][-1] / args.collagen_hours for cid in conds]
    min_rates_cond = [curves[f"mineral_cond_{cid}"][-1] / args.ars_hours for cid in conds]

    cond_summary = pd.DataFrame(
        {
            "Condition": conds,
            "D_nM": cond["D_nM"],
            "K_nM": cond["K_nM"],
            "alp_scale": alp_scale,
            "collagen_scale": col_scale,
            "mineral_scale": ars_scale,
            "alp_rate": alp_rates_cond,
            "collagen_rate": col_rates_cond,
            "mineral_rate": min_rates_cond,
        }
    )
    cond_summary.to_csv(outdir / "endpoint_condition_rates.csv", index=False)

    print(f"Saved: {outdir / 'datafit_endpoint.npz'}")
    if modifiers:
        print(f"Saved: {outdir / 'dk_modifiers.npz'}")
    print(f"Saved: {outdir / 'endpoint_rates_summary.csv'}")
    print(f"Saved: {outdir / 'endpoint_condition_rates.csv'}")

    _plot_assay_curves("ALP", time_hours, curves, outdir)
    _plot_assay_curves("Collagen", time_hours, curves, outdir)
    _plot_assay_curves("Mineral", time_hours, curves, outdir)


if __name__ == "__main__":
    main()
