#!/usr/bin/env python3

"""
Simple ALP stats/plot assuming technical replicates and parametric tests.

- Blank-corrects and converts ALP readings to mU/mL
- Computes mean ± SEM for each condition
- Runs Welch's t-tests vs Basal (C1), treating technical replicates as independent
- Plots jittered individual points + mean ± SEM and marks * for p < 0.05

NOTE: This is a "simple" parametric pipeline; it ignores
      (i) non-normality, (ii) tiny n, and (iii) multiple-testing correction.
      Use mainly as a quick visual / sanity check.
"""

import argparse
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import matplotlib.pyplot as plt
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
    """Two-line labels for conditions for plotting."""
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
    """Save figure as PDF/PNG if paths are provided."""
    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if png_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")


# ----------------- ARG PARSING ----------------- #


def parse_args():
    p = argparse.ArgumentParser(
        description="Simple ALP stats/plot using Welch t-tests vs Basal (technical replicates)."
    )
    p.add_argument("--metadata", required=True, help="Metadata text file.")
    p.add_argument("--data", required=True, help="Raw ALP data xlsx.")
    p.add_argument(
        "--blank", type=float, default=0.066, help="Blank absorbance (default: 0.066)."
    )
    p.add_argument(
        "--conversion",
        type=float,
        required=True,
        help="Conversion factor from net absorbance to mU/mL.",
    )
    p.add_argument("--out", help="Optional output PDF path.")
    p.add_argument(
        "--png",
        help="Optional PNG path; defaults to same stem as --out with .png if not given.",
    )
    p.add_argument(
        "--basal", default="C1", help="Code of basal condition (default: C1)."
    )
    p.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance threshold for stars (default: 0.05).",
    )
    return p.parse_args()


# ----------------- MAIN ----------------- #


def main():
    args = parse_args()

    # 1) Read metadata and raw values
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

    # 2) Blank-correct and convert to mU/mL
    cond_to_mU: Dict[str, List[float]] = {}
    for cond, vals in cond_to_values.items():
        out = []
        for v in vals:
            v_net = v - args.blank
            mU = max(v_net * args.conversion, 0.0)
            out.append(mU)
        cond_to_mU[cond] = out

    # 3) Restrict to C1–C8 in metadata order
    conds_c = [c for c in cond_order if c in CONDITIONS]
    if not conds_c:
        print("No conditions C1–C8 found in metadata; exiting.")
        return

    # 4) Compute mean, SEM, and Welch t-tests vs Basal
    if args.basal not in cond_to_mU or len(cond_to_mU[args.basal]) == 0:
        raise ValueError(f"Basal condition '{args.basal}' has no data.")

    basal_vals = np.asarray(cond_to_mU[args.basal], dtype=float)
    mean_dict: Dict[str, float] = {}
    sem_dict: Dict[str, float] = {}
    pvals: Dict[str, float] = {}

    for cond in conds_c:
        vals = np.asarray(cond_to_mU.get(cond, []), dtype=float)
        if vals.size == 0:
            mean_dict[cond] = np.nan
            sem_dict[cond] = np.nan
            pvals[cond] = np.nan
            continue

        mean = float(np.mean(vals))
        sem = (
            float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            if vals.size > 1
            else np.nan
        )
        mean_dict[cond] = mean
        sem_dict[cond] = sem

        if cond == args.basal:
            pvals[cond] = np.nan
        else:
            # Welch's t-test (does not assume equal variance)
            t_stat, p = stats.ttest_ind(basal_vals, vals, equal_var=False)
            pvals[cond] = p

    # 5) Print a small stats summary
    print("\n===== Simple Welch t-tests vs Basal (technical replicates) =====")
    print(f"Basal condition: {args.basal}")
    for cond in conds_c:
        mean = mean_dict[cond]
        sem = sem_dict[cond]
        p = pvals[cond]
        n = len(cond_to_mU.get(cond, []))
        if cond == args.basal:
            print(f"{cond} (Basal): mean = {mean:.4g}, SEM = {sem:.4g}, n = {n}")
        else:
            p_str = "NA" if np.isnan(p) else f"{p:.4g}"
            print(
                f"{cond}: mean = {mean:.4g}, SEM = {sem:.4g}, n = {n}, "
                f"p (Welch vs Basal) = {p_str}"
            )

    # 6) Scatter + mean/SEM plot instead of bar plot
    x_positions = np.arange(1, len(conds_c) + 1)

    # Prepare means and sems
    means = [mean_dict[c] for c in conds_c]
    sems = [sem_dict[c] for c in conds_c]

    # Y-limits based on all individual values
    all_vals = []
    for c in conds_c:
        all_vals.extend(cond_to_mU.get(c, []))

    if all_vals:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
    else:
        y_min, y_max = 0.0, 1.0

    yr = y_max - y_min if y_max > y_min else 1.0

    fig, ax = plt.subplots(figsize=(8, 4))

    # Reproducible jitter
    rng = np.random.default_rng(42)
    jitter_width = 0.20

    # Plot individual points (technical replicates)
    for x, cond in zip(x_positions, conds_c):
        vals = np.asarray(cond_to_mU.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
        ax.scatter(
            np.full_like(vals, x, dtype=float) + jitter,
            vals,
            s=28,
            alpha=0.8,
            edgecolor="black",
            linewidth=0.4,
        )

    # Plot mean ± SEM as horizontal lines + whiskers
    for x, cond, m, se in zip(x_positions, conds_c, means, sems):
        if np.isnan(m):
            continue
        # mean line
        ax.hlines(m, x - 0.28, x + 0.28, linewidth=2.0)
        # SEM whiskers if available
        if not np.isnan(se):
            ax.vlines(x, m - se, m + se, linewidth=1.2)
            ax.hlines(m - se, x - 0.12, x + 0.12, linewidth=1.0)
            ax.hlines(m + se, x - 0.12, x + 0.12, linewidth=1.0)

    # x-axis labels
    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels)

    ax.set_ylabel("ALP activity (mU·mL⁻¹, 30-min pNPP assay)")

    # Stars for p < alpha (placed above max point of each group)
    for x, cond in zip(x_positions, conds_c):
        if cond == args.basal:
            continue
        p = pvals[cond]
        if p is None or np.isnan(p) or p >= args.alpha:
            continue

        vals = np.asarray(cond_to_mU.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        group_max = float(np.max(vals))
        star_y = group_max + 0.08 * yr
        ax.text(x, star_y, "*", ha="center", va="bottom", fontsize=12)

    ax.set_ylim(y_min - 0.05 * yr, y_max + 0.25 * yr)

    ax.set_title(
        "ALP activity (mU/mL, 30-min pNPP assay)\n"
        "Dots = technical replicates, lines = mean±SEM, "
        "stars = Welch t-test p < α vs Basal\n"
        "(technical replicates treated as independent; no multiple-testing correction)"
    )

    fig.tight_layout()

    # 7) Save or show
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


# To Run:


"""
python alp_ttest_scatter.py \
    --metadata alpmetadata.txt \
    --data alp_raw_data.xlsx \
    --blank 0.066 \
    --conversion 11.01 \
    --out alp_activity_scatter.pdf
"""
