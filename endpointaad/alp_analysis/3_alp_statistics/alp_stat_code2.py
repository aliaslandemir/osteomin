#!/usr/bin/env python3
"""



Generate the ALP summary figure and mean table.

python alp_stat_code2.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output2

"""

import argparse
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats

from pipeline import parse_metadata, read_values_xlsx


# =========================
# GLOBAL PLOT SETTINGS
# =========================
def apply_pub_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12.5,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


# =========================
# CONDITIONS
# =========================
CONDITIONS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]

FAM_A_REF = "C2"
FAM_A_TARGETS = ["C1", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C2
ASSAY_COLORS = {
    "alp": "#D55E00",
    "collagen": "#009E73",
    "mineralization": "#0072B2",
}
USE_REFERENCE_GREY = False
REFERENCE_COLOR = "#374151"

MEAN_COLOR = "#8b1e1e"
SD_COLOR = "#5a5a5a"

STAR_COLOR_A = "#000000"


# =========================
# CONDITION TAGS
# =========================
def two_line_label(name: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": "Osteo",
        "Osteo + D_LK_L": r"Osteo" + "\n" + r"$D_LK_L$",
        "Osteo + D_LK_H": r"Osteo" + "\n" + r"$D_LK_H$",
        "Osteo + D_HK_L": r"Osteo" + "\n" + r"$D_HK_L$",
        "Osteo + D_HK_H": r"Osteo" + "\n" + r"$D_HK_H$",
        "Osteo + D_HK_0": r"Osteo" + "\n" + r"$D_HK_0$",
        "Osteo + D_0K_H": r"Osteo" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


def cond_id_to_pretty_name(c: str) -> str:
    m = {
        "C1": "Basal",
        "C2": "Osteo",
        "C3": "Osteo + D_LK_L",
        "C4": "Osteo + D_LK_H",
        "C5": "Osteo + D_HK_L",
        "C6": "Osteo + D_HK_H",
        "C7": "Osteo + D_HK_0",
        "C8": "Osteo + D_0K_H",
    }
    return m.get(c, c)


def cond_id_to_tag(c: str) -> str:
    label_letter = chr(ord("a") + int(c[1:]) - 1) if c.startswith("C") else "?"
    return f"({label_letter}) " + two_line_label(cond_id_to_pretty_name(c))


# =========================
# STATS HELPERS
# =========================
def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


def star_from_p(p: float) -> str:
    if not np.isfinite(p):
        return "-"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""


def pairwise_vs_ref_BH(
    cond_to_vals: Dict[str, np.ndarray],
    ref: str,
    targets: List[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    ref_vals = np.asarray(cond_to_vals.get(ref, np.array([])), dtype=float)
    ref_vals = ref_vals[np.isfinite(ref_vals)]

    out: Dict[str, Dict[str, float]] = {}
    names: List[str] = []
    p_raws: List[float] = []

    for c in targets:
        vals = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]

        if ref_vals.size == 0 or vals.size == 0:
            out[c] = {"p_raw": np.nan, "p_adj": np.nan, "sig": False}
            continue

        _, p_raw = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
        out[c] = {"p_raw": float(p_raw)}
        names.append(c)
        p_raws.append(float(p_raw))

    p_adjs = benjamini_hochberg(p_raws)
    for c, padj in zip(names, p_adjs):
        out[c]["p_adj"] = float(padj)
        out[c]["sig"] = bool(padj < alpha)
    return out


# =========================
# FIGURE 2
# =========================
def _mean_sd(
    vals: np.ndarray,
) -> tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean, np.nan
    sd = float(np.std(vals, ddof=1))
    return mean, sd


def plot_figure2_tags_bar_delta_sig(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    suptitle: str,
    xlab: str,
    seed: int = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)

    data = [
        df.loc[df["condition"] == c, value_col].dropna().astype(float).values
        for c in conds
    ]

    fig, axB = plt.subplots(figsize=(6.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.305, top=0.885)

    x_step = 0.56
    x = np.arange(len(conds), dtype=float) * x_step

    means: List[float] = []
    sds: List[float] = []
    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        bar_color = ASSAY_COLORS["alp"]
        if USE_REFERENCE_GREY and c == FAM_A_REF:
            bar_color = REFERENCE_COLOR
        mean, sd = _mean_sd(vals)
        means.append(mean)
        sds.append(sd)
        axB.bar(
            x[i],
            mean,
            width=0.40,
            color=bar_color,
            edgecolor="#444444",
            linewidth=0.8,
            alpha=0.28,
            zorder=2,
        )
        if np.isfinite(mean):
            if np.isfinite(sd) and sd > 0.0:
                axB.errorbar(
                    x[i],
                    mean,
                    yerr=sd,
                    fmt="none",
                    ecolor="#202020",
                    elinewidth=1.25,
                    capsize=4.0,
                    capthick=1.25,
                    zorder=4,
                )
            xj = x[i] + rng.normal(0, 0.020, size=vals.size)
            axB.scatter(
                xj,
                vals,
                s=32,
                marker="o",
                color="#8B3A0E",
                edgecolor="white",
                linewidth=0.9,
                alpha=0.9,
                zorder=5,
            )
            axB.scatter(
                x[i],
                mean,
                s=60,
                marker="o",
                color="#111111",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )

    finite_vals = np.concatenate([vals[np.isfinite(vals)] for vals in data if vals.size])
    finite_means = np.asarray([v for v in means if np.isfinite(v)], dtype=float)
    finite_sds = np.asarray([v for v in sds if np.isfinite(v)], dtype=float)
    ymax = 1.0
    if finite_vals.size and finite_means.size:
        ymax = max(
            float(np.max(finite_vals)),
            float(np.max(finite_means) + np.max(finite_sds) if finite_sds.size else np.max(finite_means)),
        )
    axB.set_ylim(0.0, 1.5)
    axB.set_xlim(x[0] - 0.25, x[-1] + 0.25)

    labels = [two_line_label(cond_id_to_pretty_name(c)) for c in conds]
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=0, ha="center")
    axB.set_ylabel(xlab, fontsize=15.5)
    axB.set_xlabel("")
    axB.grid(False)
    axB.tick_params(axis="x", labelsize=12.0, pad=8)
    axB.tick_params(axis="y", labelsize=13.5)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_linewidth(0.8)
    axB.spines["bottom"].set_linewidth(0.8)

    y_sig = 1.50
    for i, c in enumerate(conds):
        st = "ref" if c == "C2" else star_from_p(famA.get(c, {}).get("p_adj", np.nan))
        axB.text(x[i], y_sig, st, ha="center", va="center", fontsize=13.8, color=STAR_COLOR_A)

    fig.suptitle(suptitle, fontsize=20, y=0.965)
    return fig


# =========================
# CLI / MAIN
# =========================
def parse_args():
    p = argparse.ArgumentParser(
        description="ALP Figure 2 only (condition tags + bar plot + significance)."
    )
    p.add_argument("--metadata", required=True, help="Metadata text file.")
    p.add_argument("--data", required=True, help="Raw ALP data xlsx.")
    p.add_argument("--blank", type=float, default=0.066, help="Blank absorbance.")
    p.add_argument(
        "--conversion", type=float, required=True, help="Conversion net abs -> mU/mL."
    )
    p.add_argument("--alpha", type=float, default=0.05, help="BH alpha.")
    p.add_argument("--out_prefix", type=str, default="alp_raw", help="Output prefix.")
    p.add_argument(
        "--out_dir", type=Path, default=Path("alp_output"), help="Output directory."
    )
    p.add_argument("--seed", type=int, default=0, help="Seed for jitter.")
    return p.parse_args()


def main():
    apply_pub_rcparams()

    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

    rows = []
    for cond, vals in cond_to_values.items():
        if cond not in CONDITIONS:
            continue
        for i, v in enumerate(vals):
            try:
                v = float(v)
            except Exception:
                continue
            v_net = v - float(args.blank)
            alp = max(v_net * float(args.conversion), 0.0)
            rows.append({"condition": cond, "well_index": i + 1, "ALP_mU_mL": alp})

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid ALP values found for conditions C1–C8.")

    conds = [c for c in cond_order if c in CONDITIONS and (df["condition"] == c).any()]
    if "C2" not in conds:
        raise ValueError("C2 must be present.")

    cond_to_vals: Dict[str, np.ndarray] = {
        c: df.loc[df["condition"] == c, "ALP_mU_mL"].dropna().astype(float).values
        for c in conds
    }

    means_txt = []
    for c in conds:
        vals = cond_to_vals.get(c, np.array([]))
        if vals.size:
            mean_val = float(np.mean(vals))
            sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
            n_val = int(vals.size)
        else:
            mean_val = float("nan")
            sd_val = float("nan")
            n_val = 0
        means_txt.append((c, mean_val, sd_val, n_val))

    famA = pairwise_vs_ref_BH(
        cond_to_vals,
        ref=FAM_A_REF,
        targets=[c for c in FAM_A_TARGETS if c in conds],
        alpha=args.alpha,
    )
    famB: Dict[str, Dict[str, float]] = {}

    fig2 = plot_figure2_tags_bar_delta_sig(
        df=df,
        value_col="ALP_mU_mL",
        conds=conds,
        famA=famA,
        famB=famB,
        suptitle="ALP activity / Day 10",
        xlab=r"ALP activity (mU·mL$^{-1}$)",
        seed=args.seed,
    )

    out_pdf = args.out_dir / f"{args.out_prefix}_ALP_fig2_tags_bar_delta_sig.pdf"
    out_png = args.out_dir / f"{args.out_prefix}_ALP_fig2_tags_bar_delta_sig.png"
    fig2.savefig(out_pdf)
    fig2.savefig(out_png)
    plt.close(fig2)

    out_txt = args.out_dir / f"{args.out_prefix}_ALP_means.txt"
    with out_txt.open("w", encoding="utf-8") as f:
        f.write("Condition\tMean\tSD\tN\n")
        for c, mean_val, sd_val, n_val in means_txt:
            f.write(f"{c}\t{mean_val:.6g}\t{sd_val:.6g}\t{n_val}\n")

    print(f"Saved: {out_pdf}")
    print(f"Saved: {out_png}")
    print(f"Saved: {out_txt}")


if __name__ == "__main__":
    main()
