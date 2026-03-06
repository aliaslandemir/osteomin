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
    # Keep reporting consistent with simulation logic for no-drug baselines.
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

    i_d = 1.0 if float(getattr(field_cfg, "dex_conc", 0.0)) > 0.0 else 0.0
    i_k = 1.0 if float(getattr(field_cfg, "mk4_conc", 0.0)) > 0.0 else 0.0
    x_d_sq = x_d * x_d
    x_k_sq = x_k * x_k

    def _factors(prefix: str) -> tuple[float, float, float]:
        a_d = float(getattr(field_cfg, f"{prefix}_a_d", 0.0))
        b_d = float(getattr(field_cfg, f"{prefix}_b_d", 0.0))
        a_k = float(getattr(field_cfg, f"{prefix}_a_k", 0.0))
        b_k = float(getattr(field_cfg, f"{prefix}_b_k", 0.0))
        m_d = max(0.0, 1.0 + i_d * (a_d - b_d * x_d_sq))
        m_k = max(0.0, 1.0 + i_k * (a_k - b_k * x_k_sq))
        return m_d, m_k, m_d * m_k

    alp_base = float(getattr(field_cfg, "alp_osteo_rate", 0.0))
    collagen_base = float(getattr(field_cfg, "collagen_osteo_rate", 0.0))
    ha_base = float(getattr(field_cfg, "ha_osteo_rate", 0.0))
    alp_m_d, alp_m_k, alp_m = _factors("alp")
    collagen_m_d, collagen_m_k, collagen_m = _factors("collagen")
    ha_m_d, ha_m_k, ha_m = _factors("ha")

    return {
        "dex_conc_nM": float(getattr(field_cfg, "dex_conc", 0.0)),
        "mk4_conc_uM": float(getattr(field_cfg, "mk4_conc", 0.0)),
        "dex_ref_nM": float(getattr(field_cfg, "dex_ref_conc", 10.0)),
        "mk4_ref_uM": float(getattr(field_cfg, "mk4_ref_conc", 1.0)),
        "x_D": float(x_d),
        "x_D_sq": float(x_d_sq),
        "x_K": float(x_k),
        "x_K_sq": float(x_k_sq),
        "I_D": float(i_d),
        "I_K": float(i_k),
        "alp_base": alp_base,
        "alp_a_d": float(getattr(field_cfg, "alp_a_d", 0.0)),
        "alp_b_d": float(getattr(field_cfg, "alp_b_d", 0.0)),
        "alp_a_k": float(getattr(field_cfg, "alp_a_k", 0.0)),
        "alp_b_k": float(getattr(field_cfg, "alp_b_k", 0.0)),
        "alp_m_dex": float(alp_m_d),
        "alp_m_mk4": float(alp_m_k),
        "alp_m_total": float(alp_m),
        "alp_effective": float(max(0.0, alp_base * alp_m)),
        "collagen_base": collagen_base,
        "collagen_a_d": float(getattr(field_cfg, "collagen_a_d", 0.0)),
        "collagen_b_d": float(getattr(field_cfg, "collagen_b_d", 0.0)),
        "collagen_a_k": float(getattr(field_cfg, "collagen_a_k", 0.0)),
        "collagen_b_k": float(getattr(field_cfg, "collagen_b_k", 0.0)),
        "collagen_m_dex": float(collagen_m_d),
        "collagen_m_mk4": float(collagen_m_k),
        "collagen_m_total": float(collagen_m),
        "collagen_effective_pre_aa": float(max(0.0, collagen_base * collagen_m)),
        "ha_base": ha_base,
        "ha_a_d": float(getattr(field_cfg, "ha_a_d", 0.0)),
        "ha_b_d": float(getattr(field_cfg, "ha_b_d", 0.0)),
        "ha_a_k": float(getattr(field_cfg, "ha_a_k", 0.0)),
        "ha_b_k": float(getattr(field_cfg, "ha_b_k", 0.0)),
        "ha_m_dex": float(ha_m_d),
        "ha_m_mk4": float(ha_m_k),
        "ha_m_total": float(ha_m),
        "ha_effective_pre_gates": float(max(0.0, ha_base * ha_m)),
        "notes": (
            "Collagen effective rate is shown before AA gating; HAp effective rate is shown "
            "before Hill-time and matured-collagen gates. Drug effects scale the osteogenic "
            "rate multiplicatively as r_o * m_D(c_D) * m_K(c_K), where each m term is "
            "a peaked quadratic response in log-dose and clipped to stay non-negative."
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
        f"ALP osteogenic term: {summary['alp_base']:.6g} * "
        f"m_D={summary['alp_m_dex']:.6g} * m_K={summary['alp_m_mk4']:.6g} "
        f"(m={summary['alp_m_total']:.6g}) = "
        f"{summary['alp_effective']:.6g}"
    )
    print(
        f"Collagen osteogenic term pre-AA: {summary['collagen_base']:.6g} * "
        f"m_D={summary['collagen_m_dex']:.6g} * m_K={summary['collagen_m_mk4']:.6g} "
        f"(m={summary['collagen_m_total']:.6g}) = "
        f"{summary['collagen_effective_pre_aa']:.6g}"
    )
    print(
        f"HAp osteogenic term pre-gates: {summary['ha_base']:.6g} * "
        f"m_D={summary['ha_m_dex']:.6g} * m_K={summary['ha_m_mk4']:.6g} "
        f"(m={summary['ha_m_total']:.6g}) = "
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
        interaction = (
            _window_value(prefix, "interaction", hour_limit)
            if f"{prefix}_interaction_cum" in modulator_df.columns
            else 0.0
        )
        total = base + dex + mk4 + interaction
        fold = total / base if abs(base) > 1e-12 else 1.0
        pct = 100.0 * (total - base) / base if abs(base) > 1e-12 else 0.0
        dex_pct = 100.0 * dex / base if abs(base) > 1e-12 else 0.0
        mk4_pct = 100.0 * mk4 / base if abs(base) > 1e-12 else 0.0
        interaction_pct = 100.0 * interaction / base if abs(base) > 1e-12 else 0.0
        records.append(
            {
                "label": label,
                "prefix": prefix,
                "hour_limit": int(hour_limit),
                "base": float(base),
                "dex": float(dex),
                "mk4": float(mk4),
                "interaction": float(interaction),
                "total": float(total),
                "fold": float(fold),
                "pct_total": float(pct),
                "pct_dex": float(dex_pct),
                "pct_mk4": float(mk4_pct),
                "pct_interaction": float(interaction_pct),
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

    fig, axes = plt.subplots(1, 3, figsize=(12.6, 4.8), sharey=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    base_color = "#b0b9c6"
    dex_color = "#c44e52"
    mk4_color = "#4c9f70"
    interaction_color = "#8172b3"
    total_color = "#2f4b7c"
    grid_color = "#e5e7eb"
    edge_color = "#2f3e4f"
    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=base_color, edgecolor=edge_color, linewidth=0.8, label="Base"),
        plt.Rectangle((0, 0), 1, 1, facecolor=dex_color, edgecolor=edge_color, linewidth=0.8, label="Dex"),
        plt.Rectangle((0, 0), 1, 1, facecolor=mk4_color, edgecolor=edge_color, linewidth=0.8, label="MK-4"),
        plt.Rectangle((0, 0), 1, 1, facecolor=interaction_color, edgecolor=edge_color, linewidth=0.8, label="Interaction"),
        plt.Rectangle((0, 0), 1, 1, facecolor=total_color, edgecolor=edge_color, linewidth=0.8, label="Total"),
    ]
    for ax, rec in zip(axes, records):
        set_white(ax)
        base = float(rec["base"])
        dex = float(rec["dex"])
        mk4 = float(rec["mk4"])
        interaction = float(rec["interaction"])
        total = float(rec["total"])
        panel_ymax = max(0.25, max(base, base + dex + mk4 + interaction, total) * 1.24)
        panel_ymin = min(
            0.0,
            min(
                0.0,
                base,
                base + dex,
                base + dex + mk4,
                base + dex + mk4 + interaction,
                total,
            )
            * 1.15,
        )

        starts = [0.0, base, base + dex, base + dex + mk4, 0.0]
        heights = [base, dex, mk4, interaction, total]
        colors_local = [base_color, dex_color, mk4_color, interaction_color, total_color]
        labels = ["Base", "Dex", "MK-4", "Interaction", "Total"]
        xpos = np.arange(5)

        ax.bar(
            xpos,
            heights,
            bottom=starts,
            width=0.58,
            color=colors_local,
            edgecolor=edge_color,
            linewidth=0.8,
            zorder=3,
        )
        ax.axhline(0.0, color="#6b7280", lw=0.9, zorder=1)
        ax.grid(axis="y", color=grid_color, lw=0.8, zorder=0)

        top_anchor = max(base, base + dex + mk4 + interaction, total)
        info_text = "\n".join(
            [
                f"Fold change: {rec['fold']:.2f}x",
                f"Delta vs base: {_fmt_signed(total - base)}",
                f"Total: {_fmt_value(total)}",
            ]
        )
        ax.text(
            0.98,
            0.96,
            info_text,
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8.2,
            color="#1f2937",
            bbox=dict(
                boxstyle="round,pad=0.22",
                facecolor="white",
                edgecolor="#d1d5db",
                alpha=0.96,
            ),
        )

        ax.set_xticks(xpos)
        ax.set_xticklabels(labels, fontsize=8.5)
        ax.set_title(str(rec["label"]), pad=8, fontsize=10.5)
        if str(rec["prefix"]) == "alp":
            ax.set_ylabel("Cumulative biomarker drive (a.u.)", fontsize=9.5)
        ax.grid(axis="x", alpha=0.0)
        ax.set_xlim(-0.55, 4.55)
        ax.set_ylim(panel_ymin, panel_ymax)
        ax.tick_params(axis="y", labelsize=8.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(0.8)
        ax.spines["bottom"].set_linewidth(0.8)

    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=5,
        frameon=False,
        fontsize=8.4,
        bbox_to_anchor=(0.5, 0.955),
    )
    fig.suptitle(
        "Dex and MK-4 contributions to cumulative osteogenic drive",
        y=0.995,
        fontsize=12.5,
        fontweight="semibold",
    )
    fig.text(
        0.5,
        0.90,
        f"Dex concentration: {dex_conc:.4g} nM   |   MK-4 concentration: {mk4_conc:.4g} uM",
        ha="center",
        va="center",
        fontsize=8.6,
        color="#4b5563",
    )
    fig.text(
        0.5,
        0.02,
        "Independent y-scales because biomarkers accumulate on different magnitudes.",
        ha="center",
        va="bottom",
        fontsize=8.4,
        color="#6b7280",
    )
    fig.tight_layout(rect=(0.02, 0.06, 0.98, 0.86), pad=0.8)
    save_figure(fig, outdir, "13_drug_modulation_summary")
    plt.close(fig)
