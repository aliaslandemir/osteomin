from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "log_concentration_offset",
    "drug_response_summary",
    "print_drug_response_summary",
    "plot_drug_modulation_summary",
]


def log_concentration_offset(conc: float, ref_conc: float) -> float:
    if conc <= 0.0:
        return 0.0
    return float(np.log10(conc / ref_conc))


def drug_response_summary(field_cfg: object) -> dict[str, float | str]:
    x_d = log_concentration_offset(
        float(getattr(field_cfg, "dex_conc", 0.0)),
        max(1e-12, float(getattr(field_cfg, "dex_ref_conc", 31.622776601683793))),
    )
    x_k = log_concentration_offset(
        float(getattr(field_cfg, "mk4_conc", 0.0)),
        max(1e-12, float(getattr(field_cfg, "mk4_ref_conc", 1.0))),
    )

    alp_base = float(getattr(field_cfg, "alp_osteo_rate", 0.0))
    collagen_base = float(getattr(field_cfg, "collagen_osteo_rate", 0.0))
    ha_base = float(getattr(field_cfg, "ha_osteo_rate", 0.0))

    alp_delta_d = float(getattr(field_cfg, "alp_kappa_d", 0.0)) * x_d
    alp_delta_k = float(getattr(field_cfg, "alp_kappa_k", 0.0)) * x_k
    collagen_delta_d = float(getattr(field_cfg, "collagen_kappa_d", 0.0)) * x_d
    collagen_delta_k = float(getattr(field_cfg, "collagen_kappa_k", 0.0)) * x_k
    ha_delta_d = float(getattr(field_cfg, "ha_kappa_d", 0.0)) * x_d
    ha_delta_k = float(getattr(field_cfg, "ha_kappa_k", 0.0)) * x_k

    return {
        "dex_conc_nM": float(getattr(field_cfg, "dex_conc", 0.0)),
        "mk4_conc_uM": float(getattr(field_cfg, "mk4_conc", 0.0)),
        "dex_ref_nM": float(getattr(field_cfg, "dex_ref_conc", 31.622776601683793)),
        "mk4_ref_uM": float(getattr(field_cfg, "mk4_ref_conc", 1.0)),
        "x_D": float(x_d),
        "x_K": float(x_k),
        "alp_base": alp_base,
        "alp_delta_dex": alp_delta_d,
        "alp_delta_mk4": alp_delta_k,
        "alp_effective": float(max(0.0, alp_base + alp_delta_d + alp_delta_k)),
        "collagen_base": collagen_base,
        "collagen_delta_dex": collagen_delta_d,
        "collagen_delta_mk4": collagen_delta_k,
        "collagen_effective_pre_aa": float(
            max(0.0, collagen_base + collagen_delta_d + collagen_delta_k)
        ),
        "ha_base": ha_base,
        "ha_delta_dex": ha_delta_d,
        "ha_delta_mk4": ha_delta_k,
        "ha_effective_pre_gates": float(max(0.0, ha_base + ha_delta_d + ha_delta_k)),
        "notes": (
            "Collagen effective rate is shown before AA gating; HAp effective rate is shown "
            "before Hill-time and matured-collagen gates."
        ),
    }


def print_drug_response_summary(field_cfg: object) -> None:
    summary = drug_response_summary(field_cfg)
    print("[drug-response]")
    print(
        f"Dex: c_D={summary['dex_conc_nM']:.6g} nM, c_D0={summary['dex_ref_nM']:.6g} nM, "
        f"x_D=log10(c_D/c_D0)={summary['x_D']:.6g}"
    )
    print(
        f"MK-4: c_K={summary['mk4_conc_uM']:.6g} uM, c_K0={summary['mk4_ref_uM']:.6g} uM, "
        f"x_K=log10(c_K/c_K0)={summary['x_K']:.6g}"
    )
    print(
        f"ALP osteogenic term: {summary['alp_base']:.6g} + "
        f"({summary['alp_delta_dex']:.6g}) + ({summary['alp_delta_mk4']:.6g}) = "
        f"{summary['alp_effective']:.6g}"
    )
    print(
        f"Collagen osteogenic term pre-AA: {summary['collagen_base']:.6g} + "
        f"({summary['collagen_delta_dex']:.6g}) + ({summary['collagen_delta_mk4']:.6g}) = "
        f"{summary['collagen_effective_pre_aa']:.6g}"
    )
    print(
        f"HAp osteogenic term pre-gates: {summary['ha_base']:.6g} + "
        f"({summary['ha_delta_dex']:.6g}) + ({summary['ha_delta_mk4']:.6g}) = "
        f"{summary['ha_effective_pre_gates']:.6g}"
    )
    print(summary["notes"])


def plot_drug_modulation_summary(
    outdir: str | Path,
    field_cfg: object,
    *,
    modulator_df,
    alp_plot_hours: int | None,
    collagen_plot_hours: int | None,
    mineral_plot_hours: int | None,
    save_figure,
    set_white,
    colors,
) -> None:
    outdir = Path(outdir)
    summary = drug_response_summary(field_cfg)
    if modulator_df is None or len(modulator_df) == 0:
        return

    def _window_value(prefix: str, component: str, hour_limit: int) -> float:
        target = int(max(0, min(hour_limit, int(modulator_df["hour"].max()))))
        row = modulator_df.loc[modulator_df["hour"] == target]
        if row.empty:
            row = modulator_df.iloc[[-1]]
        return float(row.iloc[0][f"{prefix}_{component}_cum"])

    windows = [
        ("ALP", "alp", int(alp_plot_hours or 0)),
        ("Collagen", "collagen", int(collagen_plot_hours or 0)),
        ("HAp", "ha", int(mineral_plot_hours or 0)),
    ]
    records: list[dict[str, float | str | int]] = []
    for label, prefix, raw_hour in windows:
        hour_limit = raw_hour if raw_hour > 0 else int(modulator_df["hour"].max())
        base = _window_value(prefix, "base", hour_limit)
        dex = _window_value(prefix, "dex", hour_limit)
        mk4 = _window_value(prefix, "mk4", hour_limit)
        total = base + dex + mk4
        fold = total / base if abs(base) > 1e-12 else 1.0
        pct = 100.0 * (total - base) / base if abs(base) > 1e-12 else 0.0
        dex_pct = 100.0 * dex / base if abs(base) > 1e-12 else 0.0
        mk4_pct = 100.0 * mk4 / base if abs(base) > 1e-12 else 0.0
        records.append(
            {
                "label": label,
                "hour_limit": int(hour_limit),
                "base": float(base),
                "dex": float(dex),
                "mk4": float(mk4),
                "total": float(total),
                "fold": float(fold),
                "pct_total": float(pct),
                "pct_dex": float(dex_pct),
                "pct_mk4": float(mk4_pct),
            }
        )

    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.8), sharey=True)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    base_color = "#cbd5e1"
    dex_color = "#d97706"
    mk4_color = "#0f766e"
    total_color = colors["ink"]
    for ax, rec in zip(axes, records):
        set_white(ax)
        ax.axhline(1.0, color="#64748b", lw=1.1, ls="--", alpha=0.9)
        ax.bar([0], [1.0], width=0.54, color=base_color, edgecolor="#475569", linewidth=1.0)
        ax.bar([1], [rec["fold"]], width=0.54, color="#dbeafe", edgecolor="#1d4ed8", linewidth=1.0)
        ax.scatter([1], [rec["fold"]], s=84, color=total_color, edgecolor="#ffffff", linewidth=1.2, zorder=3)

        ax.annotate(
            "",
            xy=(1, rec["fold"]),
            xytext=(0, 1.0),
            arrowprops=dict(arrowstyle="-|>", lw=2.0, color=total_color, shrinkA=8, shrinkB=8),
        )

        ax.text(
            0,
            1.0,
            "1.00x\nbase",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
        )
        ax.text(
            1,
            rec["fold"],
            f"{rec['fold']:.2f}x\n{rec['pct_total']:+.1f}%",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0f172a",
        )
        ax.text(
            0.5,
            max(0.15, 0.5 * (1.0 + rec["fold"])),
            f"Dex {rec['pct_dex']:+.1f}%\nMK-4 {rec['pct_mk4']:+.1f}%",
            ha="center",
            va="center",
            fontsize=9,
            color="#0f172a",
            bbox=dict(boxstyle="round,pad=0.28", facecolor="#ffffff", edgecolor="#cbd5e1", alpha=0.9),
        )
        ax.text(
            0.5,
            0.05,
            f"window: 0-{rec['hour_limit']} h\ncum base={rec['base']:.4g}\ncum total={rec['total']:.4g}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#334155",
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Base", "With Dex+MK-4"])
        ax.set_title(str(rec["label"]), pad=10)
        ax.grid(axis="x", alpha=0.0)
        ax.set_xlim(-0.45, 1.45)

    ymax = max(1.08, max(float(rec["fold"]) for rec in records) * 1.18)
    for ax in axes:
        ax.set_ylim(0.0, ymax)
    axes[0].set_ylabel("Cumulative osteogenic drive fold vs base")
    fig.suptitle(
        (
            f"Dex/MK-4 shaping of cumulative osteogenic drive | "
            f"c_D={summary['dex_conc_nM']:.4g} nM, c_K={summary['mk4_conc_uM']:.4g} uM | "
            f"x_D={summary['x_D']:.3g}, x_K={summary['x_K']:.3g}"
        ),
        y=0.98,
        fontsize=14,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.01,
        "Each panel uses the actual simulated hourly gates over its assay window; annotations show Dex and MK-4 contributions relative to the osteogenic base drive.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.92), pad=0.9)
    save_figure(fig, outdir, "13_drug_modulation_summary")
    plt.close(fig)
