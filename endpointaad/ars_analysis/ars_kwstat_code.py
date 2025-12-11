#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARS quantification with R–G threshold + non-parametric stats (KW + Mann–Whitney + BH).

Pipeline:
1. Build R–G threshold (98th percentile of Basal C1).
2. Compute ARS-positive area (% of pixels); also µm² if --px provided.
3. Kruskal–Wallis across all conditions C1–C8.
4. Pairwise Mann–Whitney vs Basal (BH-corrected).
5. Compute effect sizes (Cliff’s delta and MW z-score).
6. Scatter + mean±SEM + corrected significance stars.
7. Export CSV + PNG + optional PDF.

Usage:
python ars_kwstat_code.py -i ../ars --px 1.62 --out ars_kw --pdf ars_kw.pdf
"""

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io, morphology
import matplotlib.pyplot as plt
from scipy import stats

# --------------------------------------------------
# CONDITIONS
# --------------------------------------------------

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


# --------------------------------------------------
# IMAGE LOADING
# --------------------------------------------------


def load_rgb_float(p: Path):
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img


# --------------------------------------------------
# NON-PARAMETRIC STATS HELPERS
# --------------------------------------------------


def cliffs_delta(a, b):
    a, b = np.asarray(a), np.asarray(b)
    greater, lesser = 0, 0
    for x in a:
        greater += np.sum(x > b)
        lesser += np.sum(x < b)
    n = a.size * b.size
    return (greater - lesser) / n if n > 0 else np.nan


def mannwhitney_z(a, b):
    a, b = np.asarray(a), np.asarray(b)
    n1, n2 = a.size, b.size
    if n1 == 0 or n2 == 0:
        return np.nan
    U, _ = stats.mannwhitneyu(a, b, alternative="two-sided")
    mean_U = n1 * n2 / 2
    std_U = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
    return (U - mean_U) / std_U if std_U > 0 else np.nan


def benjamini_hochberg(pvals):
    p = np.array(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj_rev = np.minimum.accumulate(ranked[::-1])[::-1]
    adj = np.clip(adj_rev, 0, 1)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()


# --------------------------------------------------
# MAIN
# --------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="ars_kw")
    ap.add_argument("--glob", default="**/*.tif")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--px", type=float, default=None)
    ap.add_argument("--pdf", help="Export PDF")
    args = ap.parse_args()

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    files = sorted(Path(args.input).glob(args.glob))
    rows = []

    # --------------------------------------------------
    # 1. Basal threshold
    # --------------------------------------------------

    basal_vals = []
    for p in files:
        if detect_condition(p) == "C1":
            rgb = load_rgb_float(p)
            D = (rgb[..., 0] - rgb[..., 1]).ravel()
            basal_vals.append(D)

    if not basal_vals:
        raise RuntimeError("No Basal (C1) images found!")

    basal_vals = np.concatenate(basal_vals)
    threshold = np.percentile(basal_vals, 98)
    print("Threshold (98th percentile R–G):", threshold)

    # --------------------------------------------------
    # 2. Compute ARS area
    # --------------------------------------------------

    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue

        rgb = load_rgb_float(p)
        D = rgb[..., 0] - rgb[..., 1]
        mask = D > threshold
        mask = morphology.remove_small_objects(mask, min_size=20)

        stained = mask.sum()
        total = mask.size
        frac = stained / total * 100.0
        area = stained * (args.px**2) if args.px else None

        rows.append(
            {
                "image": p.name,
                "cond": cond,
                "fraction_percent": frac,
                "stained_px": stained,
                "total_px": total,
                "area_um2": area,
            }
        )

    df = pd.DataFrame(rows)
    df.to_csv(outdir / f"{args.out}_results.csv", index=False)

    # --------------------------------------------------
    # 3. Kruskal–Wallis + MW vs Basal
    # --------------------------------------------------

    conds = sorted(df["cond"].unique(), key=lambda x: int(x[1:]))
    cond_to_vals = {c: df[df["cond"] == c]["fraction_percent"].values for c in conds}

    groups = [cond_to_vals[c] for c in conds]
    H, p_kw = stats.kruskal(*groups)
    print("\nKW test: H =", H, ", p =", p_kw)

    # Pairwise vs Basal
    basal = cond_to_vals["C1"]
    p_raw, zvals, deltas = [], [], []
    names = []

    for c in conds:
        if c == "C1":
            continue
        vals = cond_to_vals[c]
        U, p = stats.mannwhitneyu(basal, vals, alternative="two-sided")
        p_raw.append(p)
        names.append(c)
        zvals.append(mannwhitney_z(basal, vals))
        deltas.append(cliffs_delta(basal, vals))

    p_adj = benjamini_hochberg(p_raw) if p_raw else []

    sig = {c: False for c in conds}
    for c, p in zip(names, p_adj):
        if p < args.alpha:
            sig[c] = True

    # --------------------------------------------------
    # 4. Plot
    # --------------------------------------------------

    x = np.arange(1, len(conds) + 1)
    fig, ax = plt.subplots(figsize=(14, 6))

    jitter_rng = np.random.default_rng(42)

    # scatter
    for xi, c in zip(x, conds):
        vals = cond_to_vals[c]
        jit = jitter_rng.uniform(-0.22, 0.22, len(vals))
        ax.scatter(xi + jit, vals, s=40, alpha=0.85, edgecolor="black", linewidth=0.4)

    # mean ± SEM
    means = {c: np.mean(cond_to_vals[c]) for c in conds}
    sems = {
        c: (
            np.std(cond_to_vals[c], ddof=1) / np.sqrt(len(cond_to_vals[c]))
            if len(cond_to_vals[c]) > 1
            else np.nan
        )
        for c in conds
    }

    for xi, c in zip(x, conds):
        m, se = means[c], sems[c]
        ax.hlines(m, xi - 0.3, xi + 0.3, linewidth=2)
        if not np.isnan(se):
            ax.vlines(xi, m - se, m + se, linewidth=1)
            ax.hlines(m - se, xi - 0.12, xi + 0.12)
            ax.hlines(m + se, xi - 0.12, xi + 0.12)

    # significance stars
    allv = np.concatenate(groups)
    y_min, y_max = np.min(allv), np.max(allv)
    y_range = y_max - y_min if y_max > y_min else 1
    star_y = y_max + 0.12 * y_range

    for xi, c in zip(x, conds):
        if sig.get(c, False) and c != "C1":
            ax.text(xi, star_y, "*", ha="center", va="bottom", fontsize=18)

    ax.set_xticks(x)
    ax.set_xticklabels(
        [two_line_label(COND_CODE[c]) for c in conds], rotation=45, ha="right"
    )
    ax.set_ylabel("ARS-stained area (%)")
    ax.set_ylim(y_min - 0.05 * y_range, y_max + 0.25 * y_range)

    ax.set_title(
        "ARS area fraction (%), R–G threshold from Basal\n"
        "Kruskal–Wallis + BH-corrected Mann–Whitney vs Basal"
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

    print("\nDone. Output saved to:", outdir)


if __name__ == "__main__":
    main()


# To Run:

# python ars_kwstat_code.py -i ../ars --px 1.62 --out ars_kw --pdf ars_kw.pdf
