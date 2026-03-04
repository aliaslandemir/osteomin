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
        max(1e-12, float(getattr(field_cfg, "dex_ref_conc", 10.0))),
    )
    x_k = log_concentration_offset(
        float(getattr(field_cfg, "mk4_conc", 0.0)),
        max(1e-12, float(getattr(field_cfg, "mk4_ref_conc", 1.0))),
    )

    alp_base = float(getattr(field_cfg, "alp_osteo_rate", 0.0))
    collagen_base = float(getattr(field_cfg, "collagen_osteo_rate", 0.0))
    ha_base = float(getattr(field_cfg, "ha_osteo_rate", 0.0))
    x_d_sq = x_d * x_d
    x_k_sq = x_k * x_k

    alp_phi_d_lin = float(getattr(field_cfg, "alp_kappa_d", 0.0)) * x_d
    alp_phi_d_quad = float(getattr(field_cfg, "alp_kappa_d2", 0.0)) * x_d_sq
    alp_phi_d = alp_phi_d_lin + alp_phi_d_quad
    alp_phi_k_lin = float(getattr(field_cfg, "alp_kappa_k", 0.0)) * x_k
    alp_phi_k_quad = float(getattr(field_cfg, "alp_kappa_k2", 0.0)) * x_k_sq
    collagen_phi_d_lin = float(getattr(field_cfg, "collagen_kappa_d", 0.0)) * x_d
    collagen_phi_d_quad = float(getattr(field_cfg, "collagen_kappa_d2", 0.0)) * x_d_sq
    collagen_phi_d = collagen_phi_d_lin + collagen_phi_d_quad
    collagen_phi_k_lin = float(getattr(field_cfg, "collagen_kappa_k", 0.0)) * x_k
    collagen_phi_k_quad = float(getattr(field_cfg, "collagen_kappa_k2", 0.0)) * x_k_sq
    ha_phi_d_lin = float(getattr(field_cfg, "ha_kappa_d", 0.0)) * x_d
    ha_phi_d_quad = float(getattr(field_cfg, "ha_kappa_d2", 0.0)) * x_d_sq
    ha_phi_d = ha_phi_d_lin + ha_phi_d_quad
    ha_phi_k_lin = float(getattr(field_cfg, "ha_kappa_k", 0.0)) * x_k
    ha_phi_k_quad = float(getattr(field_cfg, "ha_kappa_k2", 0.0)) * x_k_sq
    alp_phi_k = alp_phi_k_lin + alp_phi_k_quad
    collagen_phi_k = collagen_phi_k_lin + collagen_phi_k_quad
    ha_phi_k = ha_phi_k_lin + ha_phi_k_quad
    alp_delta_d = alp_base * alp_phi_d
    alp_delta_k = alp_base * alp_phi_k
    collagen_delta_d = collagen_base * collagen_phi_d
    collagen_delta_k = collagen_base * collagen_phi_k
    ha_delta_d = ha_base * ha_phi_d
    ha_delta_k = ha_base * ha_phi_k

    return {
        "dex_conc_nM": float(getattr(field_cfg, "dex_conc", 0.0)),
        "mk4_conc_uM": float(getattr(field_cfg, "mk4_conc", 0.0)),
        "dex_ref_nM": float(getattr(field_cfg, "dex_ref_conc", 10.0)),
        "mk4_ref_uM": float(getattr(field_cfg, "mk4_ref_conc", 1.0)),
        "x_D": float(x_d),
        "x_D_sq": float(x_d_sq),
        "x_K": float(x_k),
        "x_K_sq": float(x_k_sq),
        "alp_base": alp_base,
        "alp_phi_dex_linear": alp_phi_d_lin,
        "alp_phi_dex_quadratic": alp_phi_d_quad,
        "alp_phi_dex": alp_phi_d,
        "alp_delta_dex": alp_delta_d,
        "alp_phi_mk4_linear": alp_phi_k_lin,
        "alp_phi_mk4_quadratic": alp_phi_k_quad,
        "alp_phi_mk4": alp_phi_k,
        "alp_delta_mk4_linear": alp_base * alp_phi_k_lin,
        "alp_delta_mk4_quadratic": alp_base * alp_phi_k_quad,
        "alp_delta_mk4": alp_delta_k,
        "alp_effective": float(max(0.0, alp_base * (1.0 + alp_phi_d + alp_phi_k))),
        "collagen_base": collagen_base,
        "collagen_phi_dex_linear": collagen_phi_d_lin,
        "collagen_phi_dex_quadratic": collagen_phi_d_quad,
        "collagen_phi_dex": collagen_phi_d,
        "collagen_delta_dex": collagen_delta_d,
        "collagen_phi_mk4_linear": collagen_phi_k_lin,
        "collagen_phi_mk4_quadratic": collagen_phi_k_quad,
        "collagen_phi_mk4": collagen_phi_k,
        "collagen_delta_mk4_linear": collagen_base * collagen_phi_k_lin,
        "collagen_delta_mk4_quadratic": collagen_base * collagen_phi_k_quad,
        "collagen_delta_mk4": collagen_delta_k,
        "collagen_effective_pre_aa": float(
            max(0.0, collagen_base * (1.0 + collagen_phi_d + collagen_phi_k))
        ),
        "ha_base": ha_base,
        "ha_phi_dex_linear": ha_phi_d_lin,
        "ha_phi_dex_quadratic": ha_phi_d_quad,
        "ha_phi_dex": ha_phi_d,
        "ha_delta_dex": ha_delta_d,
        "ha_phi_mk4_linear": ha_phi_k_lin,
        "ha_phi_mk4_quadratic": ha_phi_k_quad,
        "ha_phi_mk4": ha_phi_k,
        "ha_delta_mk4_linear": ha_base * ha_phi_k_lin,
        "ha_delta_mk4_quadratic": ha_base * ha_phi_k_quad,
        "ha_delta_mk4": ha_delta_k,
        "ha_effective_pre_gates": float(max(0.0, ha_base * (1.0 + ha_phi_d + ha_phi_k))),
        "notes": (
            "Collagen effective rate is shown before AA gating; HAp effective rate is shown "
            "before Hill-time and matured-collagen gates. Drug effects scale the osteogenic "
            "rate multiplicatively as r_o * (1 + phi_D + phi_K), with quadratic log-dose "
            "terms for both Dex and MK-4."
        ),
    }


def print_drug_response_summary(field_cfg: object) -> None:
    summary = drug_response_summary(field_cfg)
    print("[drug-response]")
    print(
        f"Dex: c_D={summary['dex_conc_nM']:.6g} nM, c_D0={summary['dex_ref_nM']:.6g} nM, "
        f"x_D=log10(c_D/c_D0)={summary['x_D']:.6g}, x_D^2={summary['x_D_sq']:.6g}"
    )
    print(
        f"MK-4: c_K={summary['mk4_conc_uM']:.6g} uM, c_K0={summary['mk4_ref_uM']:.6g} uM, "
        f"x_K=log10(c_K/c_K0)={summary['x_K']:.6g}, x_K^2={summary['x_K_sq']:.6g}"
    )
    print(
        f"ALP osteogenic term: {summary['alp_base']:.6g} + "
        f"({summary['alp_base']:.6g} * ({summary['alp_phi_dex_linear']:.6g} + {summary['alp_phi_dex_quadratic']:.6g})) + "
        f"({summary['alp_delta_mk4_linear']:.6g} + {summary['alp_delta_mk4_quadratic']:.6g}) = "
        f"{summary['alp_effective']:.6g}"
    )
    print(
        f"Collagen osteogenic term pre-AA: {summary['collagen_base']:.6g} + "
        f"({summary['collagen_base']:.6g} * ({summary['collagen_phi_dex_linear']:.6g} + {summary['collagen_phi_dex_quadratic']:.6g})) + "
        f"({summary['collagen_delta_mk4_linear']:.6g} + {summary['collagen_delta_mk4_quadratic']:.6g}) = "
        f"{summary['collagen_effective_pre_aa']:.6g}"
    )
    print(
        f"HAp osteogenic term pre-gates: {summary['ha_base']:.6g} + "
        f"({summary['ha_base']:.6g} * ({summary['ha_phi_dex_linear']:.6g} + {summary['ha_phi_dex_quadratic']:.6g})) + "
        f"({summary['ha_delta_mk4_linear']:.6g} + {summary['ha_delta_mk4_quadratic']:.6g}) = "
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
    if modulator_df is None or len(modulator_df) == 0:
        return
    dex_conc = float(getattr(field_cfg, "dex_conc", 0.0))
    mk4_conc = float(getattr(field_cfg, "mk4_conc", 0.0))

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
                "prefix": prefix,
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

    def _fmt_signed(value: float) -> str:
        return f"{value:+.2f}".rstrip("0").rstrip(".")

    def _fmt_value(value: float) -> str:
        if abs(value) >= 100:
            return f"{value:.0f}"
        if abs(value) >= 10:
            return f"{value:.1f}".rstrip("0").rstrip(".")
        return f"{value:.2f}".rstrip("0").rstrip(".")

    fig, axes = plt.subplots(1, 3, figsize=(13.4, 5.6), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    base_color = "#cfd6de"
    dex_color = "#d55e00"
    mk4_color = "#009e73"
    total_color = "#1f3b5c"
    baseline_color = "#94a3b8"
    grid_color = "#e2e8f0"
    for ax, rec in zip(axes, records):
        set_white(ax)
        base = float(rec["base"])
        dex = float(rec["dex"])
        mk4 = float(rec["mk4"])
        total = float(rec["total"])
        panel_ymax = max(0.25, max(base, base + dex, total) * 1.24)
        panel_ymin = min(
            0.0, min(0.0, base, base + dex, base + dex + mk4, total) * 1.15
        )

        starts = [0.0, base, base + dex, 0.0]
        heights = [base, dex, mk4, total]
        colors_local = [base_color, dex_color, mk4_color, total_color]
        labels = ["Base", "Dex", "MK-4", "Total"]
        xpos = np.arange(4)

        ax.bar(
            xpos,
            heights,
            bottom=starts,
            width=0.62,
            color=colors_local,
            edgecolor="#334155",
            linewidth=0.9,
            zorder=3,
        )
        ax.plot(
            [0, 1, 2, 3],
            [base, base, base + dex, total],
            color=baseline_color,
            lw=1.2,
            zorder=2,
        )
        ax.axhline(0.0, color="#475569", lw=0.9, zorder=1)
        ax.grid(axis="y", color=grid_color, lw=0.8, zorder=0)

        bar_centers = [
            base / 2.0,
            base + dex / 2.0,
            base + dex + mk4 / 2.0,
            total / 2.0,
        ]
        value_labels = [
            _fmt_value(base),
            _fmt_signed(dex),
            _fmt_signed(mk4),
            _fmt_value(total),
        ]
        for x, y, text in zip(xpos, bar_centers, value_labels):
            ax.text(
                x,
                y,
                text,
                ha="center",
                va="center",
                fontsize=9,
                color="#0f172a",
                zorder=4,
            )

        top_anchor = max(base, base + dex, total)
        ax.text(
            1.5,
            top_anchor,
            f"{rec['fold']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=10,
            color="#0f172a",
            zorder=5,
        )
        total_change = dex + mk4
        if abs(total_change) > 1e-12:
            dex_share = 100.0 * dex / total_change
            mk4_share = 100.0 * mk4 / total_change
            ax.text(
                1.5,
                panel_ymin + 0.06 * (panel_ymax - panel_ymin),
                f"Dex {dex_share:.0f}%   MK-4 {mk4_share:.0f}%",
                ha="center",
                va="bottom",
                fontsize=8.5,
                color="#334155",
            )

        ax.set_xticks(xpos)
        ax.set_xticklabels(labels)
        ax.set_title(str(rec["label"]), pad=10)
        if str(rec["prefix"]) == "alp":
            ax.set_ylabel("Cumulative biomarker drive (a.u.)")
        ax.grid(axis="x", alpha=0.0)
        ax.set_xlim(-0.55, 3.55)
        ax.set_ylim(panel_ymin, panel_ymax)
    fig.suptitle(
        "Dex and MK-4 contributions to cumulative osteogenic drive",
        y=0.985,
        fontsize=13,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.925,
        f"Dex concentration: {dex_conc:.4g} nM   |   MK-4 concentration: {mk4_conc:.4g} uM",
        ha="center",
        va="center",
        fontsize=9,
        color="#475569",
    )
    fig.text(
        0.5,
        0.01,
        "Independent y-scales because biomarkers accumulate on different magnitudes.",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#475569",
    )
    fig.tight_layout(rect=(0.02, 0.05, 0.98, 0.9), pad=0.9)
    save_figure(fig, outdir, "13_drug_modulation_summary")
    plt.close(fig)
