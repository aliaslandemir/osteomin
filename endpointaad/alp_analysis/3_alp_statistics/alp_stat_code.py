#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd  # not strictly needed, but handy
from scipy import stats

from pipeline import parse_metadata, read_values_xlsx

# -------------------- CONDITIONS & LABELS --------------------
COND_CODE = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo+D/MK4 (L/L)",
    "C4": "Osteo+D/MK4 (L/H)",
    "C5": "Osteo+D/MK4 (H/L)",
    "C6": "Osteo+D/MK4 (H/H)",
    "C7": "Osteo+Dex (H)",
    "C8": "Osteo+MK-4 (H)",
}
CONDITIONS = list(COND_CODE.keys())


def two_line_label(name: str) -> str:
    """Two-line labels for conditions."""
    m = {
        "Basal": "Basal",
        "Osteo": r"AA+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)": r"AA+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


def save_figures(fig, pdf_path: Optional[Path], png_path: Optional[Path]):
    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if png_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")


# ----------------- STATS HELPERS ----------------- #

def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta (non-parametric effect size)."""
    a = np.asarray(a)
    b = np.asarray(b)
    greater = 0
    lesser = 0
    for x in a:
        greater += np.sum(x > b)
        lesser += np.sum(x < b)
    n_tot = a.size * b.size
    if n_tot == 0:
        return np.nan
    return (greater - lesser) / n_tot


def benjamini_hochberg(pvals: List[float]) -> List[float]:
    """BH FDR correction, returns list of adjusted p-values."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []

    order = np.argsort(p)
    ranked_p = p[order]
    adj = ranked_p * m / (np.arange(m) + 1)

    # enforce monotone decreasing of adjusted p when going from largest to smallest p
    adj_rev = np.minimum.accumulate(adj[::-1])[::-1]
    adj_final = np.clip(adj_rev, 0.0, 1.0)

    p_adj = np.empty_like(adj_final)
    p_adj[order] = adj_final
    return p_adj.tolist()


def mannwhitney_z(a: np.ndarray, b: np.ndarray) -> float:
    """Approximate z-score for Mann–Whitney U test (no tie correction)."""
    a = np.asarray(a)
    b = np.asarray(b)
    n1 = a.size
    n2 = b.size
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    mean_U = n1 * n2 / 2.0
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12.0)
    if std_U == 0:
        return np.nan
    return (U - mean_U) / std_U


def alp_stats(
    cond_to_mU: Dict[str, List[float]],
    conds_order: List[str],
    basal: str = "C1",
    alpha: float = 0.05,
) -> Tuple[float, float, Dict[str, bool], Dict[str, Dict[str, float]]]:
    """
    Run Kruskal–Wallis across all conditions,
    then Dunn-style pairwise Mann–Whitney vs basal (BH-corrected).

    Returns:
        H, p_kw, sig_dict (BH-significant vs basal),
        diff_dict: per-condition stats (mean, n, p_raw, p_adj, z, delta).
    """
    # Prepare data in the requested order
    groups = []
    conds_used = []
    for c in conds_order:
        vals = np.asarray(cond_to_mU.get(c, []), dtype=float)
        if vals.size > 0:
            groups.append(vals)
            conds_used.append(c)

    if len(groups) < 2:
        return np.nan, np.nan, {}, {}

    # Kruskal–Wallis
    H, p_kw = stats.kruskal(*groups)

    if basal not in cond_to_mU or len(cond_to_mU[basal]) == 0:
        raise ValueError(f"Basal condition '{basal}' has no data.")

    basal_vals = np.asarray(cond_to_mU[basal], dtype=float)

    # Pairwise vs basal
    names = []
    p_raws = []
    zs = []
    deltas = []
    ns = []
    means = []

    # basal stats
    diff_dict: Dict[str, Dict[str, float]] = {}
    diff_dict[basal] = {
        "mean": float(np.mean(basal_vals)) if basal_vals.size > 0 else np.nan,
        "n": int(basal_vals.size),
        "p_raw": np.nan,
        "p_adj": np.nan,
        "z": np.nan,
        "delta": np.nan,
    }

    for c in conds_order:
        vals = np.asarray(cond_to_mU.get(c, []), dtype=float)
        if c == basal:
            continue
        if vals.size == 0:
            diff_dict[c] = {
                "mean": np.nan,
                "n": 0,
                "p_raw": np.nan,
                "p_adj": np.nan,
                "z": np.nan,
                "delta": np.nan,
            }
            continue

        _, p_raw = stats.mannwhitneyu(basal_vals, vals, alternative="two-sided")
        z = mannwhitney_z(basal_vals, vals)
        delta = cliffs_delta(basal_vals, vals)

        names.append(c)
        p_raws.append(p_raw)
        zs.append(z)
        deltas.append(delta)
        ns.append(int(vals.size))
        means.append(float(np.mean(vals)))

    # BH correction over all non-basal tests
    p_adjs = benjamini_hochberg(p_raws) if p_raws else []

    sig_dict: Dict[str, bool] = {}

    for i, c in enumerate(names):
        diff_dict[c] = {
            "mean": means[i],
            "n": ns[i],
            "p_raw": p_raws[i],
            "p_adj": p_adjs[i],
            "z": zs[i],
            "delta": deltas[i],
        }
        sig_dict[c] = p_adjs[i] < alpha

    return H, p_kw, sig_dict, diff_dict


# ----------------- ARG PARSING ----------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Blank-correct ALP values and convert to mU/mL, then run KW + Dunn-style vs basal."
    )
    p.add_argument("--pipeline", default="pipeline.py")
    p.add_argument("--metadata", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--blank", type=float, default=0.066)
    p.add_argument("--conversion", type=float, required=True)
    p.add_argument("--out", help="Optional output PDF path")
    p.add_argument(
        "--png",
        help="Optional PNG path; defaults to the same stem as --out with .png.",
    )
    p.add_argument(
        "--basal",
        default="C1",
        help="Code of basal condition (default: C1).",
    )
    return p.parse_args()


# ----------------- MAIN ----------------- #

def main():
    args = parse_args()

    # read metadata + xlsx
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

    # blank-correct + convert to mU/mL
    cond_to_mU: Dict[str, List[float]] = {}
    for cond, vals in cond_to_values.items():
        out = []
        for v in vals:
            v_net = v - args.blank
            mU = max(v_net * args.conversion, 0.0)
            out.append(mU)
        cond_to_mU[cond] = out

    # only C1–C8 (in metadata order)
    conds_c = [c for c in cond_order if c in CONDITIONS]

    xs, ys = [], []
    for i, cond in enumerate(conds_c, start=1):
        for v in cond_to_mU.get(cond, []):
            xs.append(i)
            ys.append(v)

    if not xs or not ys:
        print("No data found for C1–C8, exiting.")
        return

    # ---------- STATS ----------
    H, p_kw, sig_dict, diff_dict = alp_stats(
        cond_to_mU, conds_c, basal=args.basal
    )

    print("\n===== Kruskal–Wallis across all conditions =====")
    print(f"H = {H:.3f}, p = {p_kw:.4g}")

    print("\n===== Per-condition stats (vs basal) =====")
    basal_mean = diff_dict.get(args.basal, {}).get("mean", np.nan)
    for cond in conds_c:
        info = diff_dict.get(cond, {})
        mean_val = info.get("mean", np.nan)
        n = int(info.get("n", 0))
        delta = info.get("delta", np.nan)
        if cond == args.basal:
            print(f"{cond} (Basal): mean = {mean_val:.4g}, n = {n}")
        else:
            p_raw = info.get("p_raw", np.nan)
            p_adj = info.get("p_adj", np.nan)
            z = info.get("z", np.nan)
            print(
                f"{cond}: mean = {mean_val:.4g}, n = {n}, "
                f"Δmean (vs basal) = {mean_val - basal_mean:+.4g}, "
                f"Cliff δ = {delta:+.3f}, "
                f"z = {z:.3f}, p_raw = {p_raw:.4g}, p_BH = {p_adj:.4g}, "
                f"significant = {sig_dict.get(cond, False)}"
            )

    means = {c: diff_dict.get(c, {}).get("mean", np.nan) for c in conds_c}
    mean_basal = means.get(args.basal, np.nan)

    # ---------- PLOT ----------
    fig, ax = plt.subplots(figsize=(8, 4))

    # scatter + mean lines
    for i, cond in enumerate(conds_c, start=1):
        vals = cond_to_mU.get(cond, [])
        if len(vals) > 0:
            x_jitter = np.random.normal(loc=i, scale=0.05, size=len(vals))
            ax.scatter(x_jitter, vals, s=30)
            mean_val = np.mean(vals)
            ax.hlines(mean_val, i - 0.2, i + 0.2, linewidth=1, color="0.2")

    ax.set_xticks(range(1, len(conds_c) + 1))
    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
    ax.set_xticklabels(labels)

    ax.set_ylabel("ALP activity (mU·mL⁻¹, 30-min pNPP assay)")

    # y-range + annotation heights
    y_min, y_max = min(ys), max(ys)
    yr = y_max - y_min if y_max > y_min else 1.0
    label_y = y_max + 0.05 * yr
    star_y = y_max + 0.12 * yr

    # Δ-mean labels + stars vs basal
    for i, cond in enumerate(conds_c, start=1):
        if cond == args.basal:
            ax.text(i, label_y, "baseline", ha="center", fontsize=7)
            continue

        diff = means[cond] - mean_basal
        if np.isnan(diff):
            lbl = "NA"
        else:
            arrow = "↑" if diff > 0 else "↓"
            lbl = f"{diff:+.2f} {arrow}"

        ax.text(i, label_y, lbl, ha="center", fontsize=7)

        if sig_dict.get(cond, False):
            ax.text(i, star_y, "*", ha="center", fontsize=12)

    ax.set_title(
        "ALP activity (mU/mL, 30-min pNPP assay)\n"
        "Stars mark BH-corrected Mann–Whitney significance vs Basal (n=3 technical)"
    )

    ax.set_ylim(y_min - 0.05 * yr, y_max + 0.25 * yr)
    fig.tight_layout()

    # save / show
    if args.out or args.png:
        pdf_path = Path(args.out) if args.out else None
        png_path = (
            Path(args.png)
            if args.png
            else pdf_path.with_suffix(".png") if pdf_path else None
        )
        save_figures(fig, pdf_path, png_path)
    else:
        plt.show()


if __name__ == "__main__":
    main()


"""
Example usage:

python alp_stat_code.py \
    --metadata alpmetadata.txt \
    --data alp_raw_data.xlsx \
    --blank 0.066 \
    --conversion 11.01 \
    --out alp_activity_stat.pdf
"""
