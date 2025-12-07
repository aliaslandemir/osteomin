#!/usr/bin/env python3

import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    """Benjamini–Hochberg FDR correction."""
    m = len(pvals)
    order = np.argsort(pvals)
    ranked = pvals[order]
    adj = np.empty(m)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj[i] = min(prev, 1.0)
    out = np.empty(m)
    out[order] = adj
    return out


def alp_stats(cond_to_mU, conds_c, basal="C1"):
    """Compute Kruskal–Wallis + pairwise MWU vs basal with BH correction."""
    rows = []
    for cond in conds_c:
        for v in cond_to_mU.get(cond, []):
            rows.append({"condition": cond, "value": v})
    df = pd.DataFrame(rows)

    # Kruskal–Wallis
    groups = [df[df["condition"] == c]["value"].dropna().values for c in conds_c]
    H, p_kw = stats.kruskal(*groups)

    # Pairwise vs basal
    basal_vals = df[df["condition"] == basal]["value"].dropna().values
    p_raw, cond_list, diffs = [], [], []

    for cond in conds_c:
        if cond == basal:
            continue
        vals = df[df["condition"] == cond]["value"].dropna().values
        if len(vals) == 0:
            continue

        U, p_raw_val = stats.mannwhitneyu(basal_vals, vals, alternative="two-sided")
        p_raw.append(p_raw_val)
        cond_list.append(cond)

        diffs.append(np.mean(vals) - np.mean(basal_vals))

    if p_raw:
        p_adj = benjamini_hochberg(np.array(p_raw))
        sig_dict = {c: (pa < 0.05) for c, pa in zip(cond_list, p_adj)}
        diff_dict = {c: d for c, d in zip(cond_list, diffs)}
    else:
        sig_dict, diff_dict = {}, {}

    return H, p_kw, sig_dict, diff_dict


# ----------------- ARG PARSING ----------------- #

def parse_args():
    p = argparse.ArgumentParser(
        description="Blank-correct ALP values and convert to mU/mL."
    )
    p.add_argument("--pipeline", default="pipeline.py")
    p.add_argument("--metadata", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--blank", type=float, default=0.066)
    p.add_argument("--conversion", type=float, required=True)
    p.add_argument("--out", help="Optional output PDF/PNG")
    p.add_argument(
        "--png",
        help="Optional PNG path; defaults to the same stem as --out with .png.",
    )
    return p.parse_args()


# ----------------- MAIN ----------------- #

def main():
    args = parse_args()

    # read metadata + xlsx
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

    # blank-correct + convert
    cond_to_mU = {}
    for cond, vals in cond_to_values.items():
        out = []
        for v in vals:
            v_net = v - args.blank
            mU = max(v_net * args.conversion, 0.0)
            out.append(mU)
        cond_to_mU[cond] = out

    # only C1–C8
    conds_c = [c for c in cond_order if c.startswith("C")]

    xs, ys = [], []
    for i, cond in enumerate(conds_c, start=1):
        for v in cond_to_mU.get(cond, []):
            xs.append(i)
            ys.append(v)

    if xs and ys:
        # ---------- STATS ----------
        H, p_kw, sig_dict, diff_dict = alp_stats(cond_to_mU, conds_c, basal="C1")

        means = {c: np.mean(cond_to_mU.get(c, [])) for c in conds_c}
        mean_basal = means["C1"]

        # ---------- PLOT ----------
        fig, ax = plt.subplots(figsize=(8, 4))

        # scatter + mean lines
        for i, cond in enumerate(conds_c, start=1):
            vals = cond_to_mU.get(cond, [])
            if len(vals) > 0:
                ax.scatter([i] * len(vals), vals, s=30)
                mean_val = np.mean(vals)
                ax.hlines(mean_val, i - 0.2, i + 0.2, linewidth=1, color="0.2")

        ax.set_xticks(range(1, len(conds_c) + 1))
        labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
        ax.set_xticklabels(labels)

        ax.set_ylabel("ALP activity (mU/mL)")

        # y-range
        y_min, y_max = min(ys), max(ys)
        yr = y_max - y_min if y_max > y_min else 1
        label_y = y_max + 0.05 * yr
        star_y = y_max + 0.12 * yr

        # Δ-mean labels + stars
        for i, cond in enumerate(conds_c, start=1):
            if cond == "C1":
                ax.text(i, label_y, "baseline", ha="center", fontsize=7)
                continue

            diff = means[cond] - mean_basal
            ad = abs(diff)

            if ad == 0:
                lbl = "0"
            else:
                level = "strong" if ad > 0.3 else "mod" if ad >= 0.15 else "mild"
                arrow = "↑" if diff > 0 else "↓"
                lbl = f"{diff:+.2f} {arrow} {level}"

            ax.text(i, label_y, lbl, ha="center", fontsize=7)

            if sig_dict.get(cond, False):
                ax.text(i, star_y, "*", ha="center", fontsize=12)

        ax.set_title(
            "ALP activity per condition\n"
            "Stars indicate Benjamini–Hochberg–adjusted p < 0.05 vs basal"
        )

        ax.set_ylim(y_min - 0.05 * yr, y_max + 0.25 * yr)
        fig.tight_layout()

        if args.out or args.png:
            pdf_path = Path(args.out) if args.out else None
            png_path = (
                Path(args.png)
                if args.png
                else pdf_path.with_suffix(".png")
                if pdf_path
                else None
            )
            save_figures(fig, pdf_path, png_path)
        else:
            plt.show()


if __name__ == "__main__":
    main()



# To run:


"""


python alp_stat_code.py \
    --metadata alpmetadata.txt \
    --data alp_raw_data.xlsx \
    --blank 0.066 \
    --conversion 11.01 \
    --out alp_activity_stat.pdf

    
"""
