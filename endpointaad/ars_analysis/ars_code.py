#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARS quantification using R–G threshold + Welch t-tests vs Basal.

Features:
- Builds R–G threshold from Basal (98th percentile).
- Computes ARS-positive pixel area (% and µm²).
- Welch t-test vs Basal (simple exploratory stats).
- Better plotting layout with clear significance stars.
- Exports CSV + PNG + PDF.

Use:
python ars_rg_ttest.py -i ../ars --px 1.62 --out ars_results --pdf ars_results.pdf
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, morphology
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------
# Conditions
# ------------------------------------------------------

COND_RE = re.compile(r"^(C[1-8])", re.IGNORECASE)
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

def two_line_label(name: str):
    m = {
        "Basal": "Basal",
        "Osteo": r"AA+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": "AA+$β$-GP\nD_LK_L",
        "Osteo+D/MK4 (L/H)": "AA+$β$-GP\nD_LK_H",
        "Osteo+D/MK4 (H/L)": "AA+$β$-GP\nD_HK_L",
        "Osteo+D/MK4 (H/H)": "AA+$β$-GP\nD_HK_H",
        "Osteo+Dex (H)": "AA+$β$-GP\nD_HK_0",
        "Osteo+MK-4 (H)": "AA+$β$-GP\nD_0K_H",
    }
    return m.get(name, name)

def detect_condition(path: Path):
    parent = path.parent.name.upper()
    if parent in COND_CODE:
        return parent
    m = COND_RE.match(path.name)
    return m.group(1).upper() if m else "UNKNOWN"


# ------------------------------------------------------
# Image Loading
# ------------------------------------------------------

def load_rgb_float(p: Path):
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img]*3, -1)
    if img.ndim == 3 and img.shape[0] in (3,4) and img.shape[-1] not in (3,4):
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] > 3:
        img = img[...,:3]
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img


# ------------------------------------------------------
# MAIN
# ------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="ars_rg")
    ap.add_argument("--glob", default="**/*.tif")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--px", type=float, default=None, help="Pixel size in µm (optional).")
    ap.add_argument("--pdf", type=str, help="Optional PDF export for plot.")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    files = sorted(Path(args.input).glob(args.glob))
    rows = []

    # --------------------------------------------------
    # 1. Build Basal R–G distribution
    # --------------------------------------------------
    basal_vals = []

    for p in files:
        if detect_condition(p) != "C1":
            continue
        rgb = load_rgb_float(p)
        R, G = rgb[...,0], rgb[...,1]
        D = (R - G).ravel()
        basal_vals.append(D)

    if not basal_vals:
        raise RuntimeError("No basal (C1) images found!")

    basal_vals = np.concatenate(basal_vals)
    threshold = np.percentile(basal_vals, 98)
    print("Basal 98th percentile threshold (R-G):", threshold)

    # --------------------------------------------------
    # 2. Process all images
    # --------------------------------------------------

    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue

        rgb = load_rgb_float(p)
        R, G = rgb[...,0], rgb[...,1]
        D = R - G

        mask = D > threshold
        mask = morphology.remove_small_objects(mask, min_size=20)

        stained_px = mask.sum()
        total_px = mask.size
        frac_percent = stained_px / total_px * 100.0

        # Compute µm² area if px size provided
        area_um2 = stained_px * (args.px**2) if args.px else None

        rows.append({
            "image": p.name,
            "cond": cond,
            "stained_px": stained_px,
            "total_px": total_px,
            "fraction_percent": frac_percent,
            "area_um2": area_um2
        })

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{args.out}_results.csv", index=False)

    # --------------------------------------------------
    # 3. Welch t-test vs Basal
    # --------------------------------------------------

    conds = sorted(df["cond"].unique(), key=lambda x: int(x[1:]))
    cond_to_vals = {c: df[df["cond"] == c]["fraction_percent"].values for c in conds}
    basal_vals = cond_to_vals["C1"]

    means = {c: np.mean(cond_to_vals[c]) for c in conds}
    sems  = {c: np.std(cond_to_vals[c], ddof=1)/np.sqrt(len(cond_to_vals[c]))
             if len(cond_to_vals[c]) > 1 else np.nan for c in conds}

    pvals = {}
    for c in conds:
        if c == "C1":
            pvals[c] = np.nan
        else:
            t, p = stats.ttest_ind(basal_vals, cond_to_vals[c], equal_var=False)
            pvals[c] = p

    print("\n===== Welch t-tests vs Basal =====")
    for c in conds:
        print(f"{c}: mean={means[c]:.2f}%, n={len(cond_to_vals[c])}, p={pvals[c]}")

    # --------------------------------------------------
    # 4. Plot (improved layout)
    # --------------------------------------------------

    x = np.arange(1, len(conds)+1)
    fig, ax = plt.subplots(figsize=(14, 6))

    jitter_rng = np.random.default_rng(42)

    # Scatter points
    for xi, c in zip(x, conds):
        vals = cond_to_vals[c]
        jit = jitter_rng.uniform(-0.22, 0.22, len(vals))
        ax.scatter(xi + jit, vals, s=40, alpha=0.9, edgecolor="black", linewidth=0.4)

    # Mean ± SEM
    for xi, c in zip(x, conds):
        m = means[c]
        se = sems[c]
        ax.hlines(m, xi - 0.3, xi + 0.3, linewidth=2)
        if not np.isnan(se):
            ax.vlines(xi, m - se, m + se, linewidth=1)
            ax.hlines(m - se, xi - 0.12, xi + 0.12)
            ax.hlines(m + se, xi - 0.12, xi + 0.12)

    # Significance stars
    allv = np.concatenate(list(cond_to_vals.values()))
    y_min, y_max = np.min(allv), np.max(allv)
    y_range = y_max - y_min if y_max > y_min else 1

    star_level = y_max + 0.12 * y_range

    for xi, c in zip(x, conds):
        p = pvals[c]
        if c != "C1" and p < args.alpha:
            ax.text(xi, star_level, "*", ha="center", va="bottom", fontsize=18)

    # Axis formatting
    ax.set_xticks(x)
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in conds],
                       rotation=45, ha="right")
    ax.set_ylabel("ARS-stained area (%)")

    ax.set_ylim(y_min - 0.05*y_range, y_max + 0.25*y_range)

    ax.set_title(
        "ARS area fraction (%), R–G threshold from Basal\n"
        "Welch t-tests vs Basal (technical replicates)"
    )


    # --------------------------------------------------
    # Δ-mean labels (difference from Basal) + stars
    # --------------------------------------------------

    basal_mean = means["C1"]

    # y positions
    label_y = y_max + 0.05 * y_range
    star_y = y_max + 0.12 * y_range

    for xi, c in zip(x, conds):
        if c == "C1":
            ax.text(xi, label_y, "baseline", ha="center", fontsize=8)
            continue

        diff = means[c] - basal_mean
        if np.isnan(diff):
            lbl = "NA"
        else:
            arrow = "↑" if diff > 0 else "↓"
            lbl = f"{diff:+.2f} {arrow}"

        ax.text(xi, label_y, lbl, ha="center", fontsize=8)


    fig.tight_layout()
    fig.savefig(outdir / f"{args.out}_plot.png", dpi=300)

    if args.pdf:
        fig.savefig(args.pdf, dpi=300)

    plt.close(fig)
    print("\nSaved:", outdir)


if __name__ == "__main__":
    main()
    
    
    
    
# To Run:

# python ars_code.py -i ../ars --px 1.62 --out ars_results --pdf ars_plot.pdf