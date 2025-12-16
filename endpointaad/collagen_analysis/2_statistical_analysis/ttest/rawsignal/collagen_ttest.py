#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARS quantification using R–G threshold + Welch t-tests vs Basal.

Adds:
- --percentile to control basal percentile threshold (default 98)
- --overlay to export cyan mask overlays (fast PNG, 100 dpi metadata)
- collagen-style plot: clean axes, jittered points, mean±SEM whiskers, stars

Example:
python ars_rg_ttest.py -i ../ars --px 1.62 --percentile 98 --out ars_results --pdf ars_results.pdf --overlay
"""

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
from skimage import io, morphology
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image  # for fast PNG + dpi metadata


# ---------------- configurable ----------------
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
CONDITIONS = list(COND_CODE.keys())


def two_line_label(name: str) -> str:
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


def detect_condition(path: Path) -> str:
    parent = path.parent.name.upper()
    if parent in COND_CODE:
        return parent
    m = COND_RE.match(path.name)
    return m.group(1).upper() if m else "UNKNOWN"


def load_rgb_float(p: Path) -> np.ndarray:
    """Load an RGB image as float32 in [0,1], channel-last."""
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # channel-first -> channel-last
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)

    # drop alpha if present
    if img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    if img.max() > 1:
        img /= 255.0
    return img


def save_png_with_dpi(arr_uint8: np.ndarray, out_path: Path, dpi: int = 100):
    """Save uint8 RGB array as PNG with dpi metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr_uint8, mode="RGB")
    im.save(str(out_path), dpi=(dpi, dpi))


def make_cyan_overlay(rgb01: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """
    Create cyan overlay on top of RGB image.
    rgb01: float image [0,1], shape (H,W,3)
    mask: boolean mask (H,W)
    alpha: blend factor for cyan
    Returns uint8 RGB image.
    """
    rgb = np.clip(rgb01, 0, 1).copy()
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)  # requested cyan
    m = mask.astype(bool)

    # Blend only where mask is True
    rgb[m] = (1.0 - alpha) * rgb[m] + alpha * cyan
    out = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out


def save_figures(fig, pdf_path: Path | None, png_path: Path | None):
    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if png_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Input folder containing images.")
    ap.add_argument("-o", "--out", default="ars_rg", help="Output folder name.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument("--alpha", type=float, default=0.05, help="Star threshold (p < alpha).")
    ap.add_argument("--px", type=float, default=None, help="Pixel size in µm (optional).")
    ap.add_argument("--pdf", type=str, help="Optional PDF export for plot.")
    ap.add_argument("--png", type=str, help="Optional PNG export path for plot (else saved in outdir).")

    ap.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="Basal percentile for (R-G) threshold (e.g., 95, 98, 99). Default: 98.",
    )
    ap.add_argument(
        "--min_size",
        type=int,
        default=20,
        help="Remove ARS-positive objects smaller than this (pixels). Default: 20.",
    )

    # overlay export
    ap.add_argument(
        "--overlay",
        action="store_true",
        help="If set, export cyan overlay PNGs showing ARS-positive pixels.",
    )
    ap.add_argument(
        "--overlay_alpha",
        type=float,
        default=0.45,
        help="Cyan overlay blending strength (0..1). Default: 0.45.",
    )
    ap.add_argument(
        "--overlay_dpi",
        type=int,
        default=100,
        help="DPI metadata for overlay PNG export. Default: 100.",
    )

    return ap.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be between 0 and 100 (exclusive).")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be between 0 and 1.")

    outdir = Path(args.out)
    outdir.mkdir(exist_ok=True)

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise RuntimeError(f"No files found: {args.input} with glob '{args.glob}'")

    # --------------------------------------------------
    # 1) Build Basal R–G distribution
    # --------------------------------------------------
    basal_vals = []
    for p in files:
        if detect_condition(p) != "C1":
            continue
        rgb = load_rgb_float(p)
        D = (rgb[..., 0] - rgb[..., 1]).ravel()
        basal_vals.append(D)

    if not basal_vals:
        raise RuntimeError("No basal (C1) images found!")

    basal_vals = np.concatenate(basal_vals)
    threshold = float(np.percentile(basal_vals, args.percentile))
    print(f"Basal {args.percentile:g}th percentile threshold (R-G): {threshold:.6g}")

    # --------------------------------------------------
    # 2) Process all images
    # --------------------------------------------------
    rows = []

    overlay_root = outdir / "overlays"
    if args.overlay:
        overlay_root.mkdir(exist_ok=True)

    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue

        rgb = load_rgb_float(p)
        D = rgb[..., 0] - rgb[..., 1]

        mask = D > threshold
        mask = morphology.remove_small_objects(mask, min_size=args.min_size)

        stained_px = int(mask.sum())
        total_px = int(mask.size)
        frac_percent = (stained_px / total_px) * 100.0 if total_px > 0 else np.nan
        area_um2 = stained_px * (args.px ** 2) if args.px else np.nan

        rows.append(
            {
                "image": p.name,
                "cond": cond,
                "threshold_percentile": args.percentile,
                "threshold_value_rg": threshold,
                "stained_px": stained_px,
                "total_px": total_px,
                "fraction_percent": frac_percent,
                "area_um2": area_um2,
            }
        )

        # overlays (fast)
        if args.overlay:
            overlay_uint8 = make_cyan_overlay(rgb, mask, alpha=args.overlay_alpha)
            out_path = overlay_root / cond / f"{Path(p.name).stem}_overlay.png"
            save_png_with_dpi(overlay_uint8, out_path, dpi=args.overlay_dpi)

    df = pd.DataFrame(rows)
    csv_path = outdir / f"{args.out}_results.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    if args.overlay:
        print("Saved overlays to:", overlay_root)

    # --------------------------------------------------
    # 3) Welch t-test vs Basal (on fraction_percent)
    # --------------------------------------------------
    conds = sorted(df["cond"].unique(), key=lambda x: int(x[1:]))
    cond_to_vals = {
        c: df.loc[df["cond"] == c, "fraction_percent"].dropna().astype(float).values
        for c in conds
    }

    if "C1" not in cond_to_vals or len(cond_to_vals["C1"]) == 0:
        raise RuntimeError("Basal (C1) has no usable fraction_percent values.")

    basal = cond_to_vals["C1"]

    mean_dict = {c: (float(np.mean(cond_to_vals[c])) if len(cond_to_vals[c]) else np.nan) for c in conds}
    sem_dict = {
        c: (float(np.std(cond_to_vals[c], ddof=1) / np.sqrt(len(cond_to_vals[c])))
            if len(cond_to_vals[c]) > 1 else np.nan)
        for c in conds
    }

    pvals = {}
    for c in conds:
        if c == "C1" or len(cond_to_vals[c]) == 0:
            pvals[c] = np.nan
        else:
            _, p = stats.ttest_ind(basal, cond_to_vals[c], equal_var=False)
            pvals[c] = float(p)

    print("\n===== Welch t-tests vs Basal (fraction_percent) =====")
    for c in conds:
        n = len(cond_to_vals[c])
        p = pvals[c]
        p_str = "NA" if np.isnan(p) else f"{p:.4g}"
        print(f"{c}: mean={mean_dict[c]:.3g}%, SEM={sem_dict[c]:.3g}, n={n}, p={p_str}")

    # --------------------------------------------------
    # 4) Plot (collagen-style)
    # --------------------------------------------------
    conds_c = [c for c in conds if len(cond_to_vals.get(c, [])) > 0]
    x_positions = np.arange(1, len(conds_c) + 1)
    means = [mean_dict[c] for c in conds_c]
    sems = [sem_dict[c] for c in conds_c]

    all_vals = []
    for c in conds_c:
        all_vals.extend(cond_to_vals.get(c, []).tolist())

    if all_vals:
        y_min = float(np.min(all_vals))
        y_max = float(np.max(all_vals))
    else:
        y_min, y_max = 0.0, 1.0
    yr = max(y_max - y_min, 1e-6)

    fig, ax = plt.subplots(figsize=(8, 3.5))

    # clean axes (like your collagen plot)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1)
    ax.spines["bottom"].set_linewidth(1)
    ax.tick_params(axis="both", width=1, labelsize=9)

    rng = np.random.default_rng(42)
    jitter_width = 0.10

    # dots
    for x, cond in zip(x_positions, conds_c):
        vals = np.asarray(cond_to_vals.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        jitter = rng.uniform(-jitter_width, jitter_width, size=vals.size)
        ax.scatter(
            np.full_like(vals, x, dtype=float) + jitter,
            vals,
            s=28,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.4,
            zorder=2,
        )

    # mean + SEM whiskers
    for x, m, se in zip(x_positions, means, sems):
        if np.isnan(m):
            continue
        ax.hlines(m, x - 0.22, x + 0.22, linewidth=2.0, zorder=3)
        if not np.isnan(se):
            ax.vlines(x, m - se, m + se, linewidth=1.0)
            ax.hlines(m - se, x - 0.10, x + 0.10, linewidth=1.0)
            ax.hlines(m + se, x - 0.10, x + 0.10, linewidth=1.0)

    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in conds_c]
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("ARS-stained area (%)", fontsize=10)

    # stars (p < alpha vs basal)
    for x, cond in zip(x_positions, conds_c):
        if cond == "C1":
            continue
        p = pvals.get(cond, np.nan)
        if np.isnan(p) or p >= args.alpha:
            continue
        vals = np.asarray(cond_to_vals.get(cond, []), dtype=float)
        if vals.size == 0:
            continue
        group_max = float(np.max(vals))
        star_y = group_max + 0.08 * yr
        ax.text(x, star_y, "*", ha="center", va="bottom", fontsize=13)

    ax.set_ylim(y_min - 0.06 * yr, y_max + 0.22 * yr)
    ax.set_title(
        f"ARS area fraction (%), R–G threshold from Basal ({args.percentile:g}th percentile)\n"
        "Welch t-tests vs Basal (technical replicates)"
    )
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.23, top=0.92)

    # save
    pdf_path = Path(args.pdf) if args.pdf else None
    png_path = Path(args.png) if args.png else (outdir / f"{args.out}_plot.png")
    save_figures(fig, pdf_path, png_path)
    plt.close(fig)

    print("Done. Output saved to:", outdir)


if __name__ == "__main__":
    main()
