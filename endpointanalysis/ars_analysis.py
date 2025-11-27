#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Alizarin Red S area — memory-safe, two-pass thresholds, clean overlays.

Method (default):
- Pass 1: for each image, compute Otsu on a downscaled (thumb) (R-G) image.
          Global T_rg = median of per-image Otsu values (robust, low RAM).
- Pass 2: per image, mask = (R-G > T_rg) AND (L* < Lcap). No cleanup by design.

Options:
- --mode global  (default): use median of per-image Otsu thresholds
- --mode perimg: compute Otsu per image (on a thumbnail) and apply to that image
- --thumb controls downscale used for Otsu estimation only (default 0.25)

Condition detection:
- From file prefix C1_..C8_ or parent folder c1..c8.

Outputs:
  <out>/
    masks/*.png        (downscaled for speed)
    overlays/*.png     (downscaled RGB with blue edges)
    tables/ars_area_by_image.csv
    tables/threshold_summary.csv
    figs/violin_percent_area.png
    figs/bar_mean_sd_percent_area.png

Usage:
  python ars_analysis.py --input arsdata --out ars_out
  python ars_analysis.py --input arsdata --output ars_out
"""

from __future__ import annotations
import argparse, re
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from skimage import io, color, segmentation, img_as_ubyte, transform
from skimage.filters import threshold_otsu

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -------- Colors / Labels --------
VIOLIN_COLOR = "#E69F00"
DOT_COLOR = "#0072B2"
MEAN_COLOR = "#009E73"
MEDIAN_COLOR = MEAN_COLOR
MEAN_TEXT_FMT = "{:.2f}"
MEAN_TEXT_OFFSET = 0.14
_JITTER_RNG = np.random.default_rng(42)

COND_CODE: Dict[str, str] = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo+D/MK4 (L/L)",
    "C4": "Osteo+D/MK4 (L/H)",
    "C5": "Osteo+D/MK4 (H/L)",
    "C6": "Osteo+D/MK4 (H/H)",
    "C7": "Osteo+Dex (H)",
    "C8": "Osteo+MK-4 (H)",
}
ORDER_C: List[str] = list(COND_CODE.values())
LABELS_2LINE: Dict[str, str] = {
    "Basal": "Basal",
    "Osteo": r"A+$\beta$-GP",
    "Osteo+D/MK4 (L/L)": r"A+$\beta$-GP" + "\n" + r"$D_LK_L$",
    "Osteo+D/MK4 (L/H)": r"A+$\beta$-GP" + "\n" + r"$D_LK_H$",
    "Osteo+D/MK4 (H/L)": r"A+$\beta$-GP" + "\n" + r"$D_HK_L$",
    "Osteo+D/MK4 (H/H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_H$",
    "Osteo+Dex (H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_0$",
    "Osteo+MK-4 (H)": r"A+$\beta$-GP" + "\n" + r"$D_0K_H$",
}

COND_RE = re.compile(r"\b(C[1-8])[_\-]", re.IGNORECASE)


def two_line_label(label: str) -> str:
    """Return a two-line version of the condition label if available."""
    return LABELS_2LINE.get(label, label)


# -------- Helpers --------
def detect_condition(path: Path) -> Tuple[str, str]:
    m = COND_RE.search(path.name)
    if m:
        code = m.group(1).upper()
    else:
        parent = path.parent.name.lower()
        if parent in [f"c{i}" for i in range(1, 9)]:
            code = parent.upper()
        else:
            gp = path.parent.parent.name.lower() if path.parent.parent else ""
            code = gp.upper() if gp in [f"c{i}" for i in range(1, 9)] else "UNKNOWN"
    return code, COND_CODE.get(code, "Unknown")


def load_rgb_float(p: Path) -> np.ndarray:
    """Load as float32 in [0,1], handle 2D, HWC/CHW, and RGBA."""
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img, img, img], axis=-1)
    elif img.ndim == 3:
        if img.shape[-1] in (3, 4):  # HWC
            img = img[..., :3]
        elif img.shape[0] in (3, 4):  # CHW
            img = np.moveaxis(img, 0, -1)[..., :3]
        else:
            raise ValueError(f"Unexpected image shape for {p}: {img.shape}")
    else:
        raise ValueError(f"Unsupported ndim for {p}: {img.ndim}")
    img = img.astype(np.float32)
    if img.max() > 1.01:  # 0-255 → 0-1
        img /= 255.0
    return np.clip(img, 0, 1)


def resize_safe(arr: np.ndarray, scale: float, order: int, aa: bool) -> np.ndarray:
    if scale == 1.0:
        return arr
    h, w = arr.shape[:2]
    new_hw = (max(1, int(h * scale)), max(1, int(w * scale)))
    if arr.ndim == 2:
        return transform.resize(
            arr, new_hw, order=order, preserve_range=True, anti_aliasing=aa
        ).astype(np.float32)
    else:
        return transform.resize(
            arr,
            (new_hw[0], new_hw[1], arr.shape[2]),
            order=order,
            preserve_range=True,
            anti_aliasing=aa,
        ).astype(np.float32)


def save_downscaled_png(path: Path, img: np.ndarray, scale: float):
    if img.ndim == 2:
        img_small = resize_safe(img, scale, order=0, aa=False)
    else:
        img_small = resize_safe(img, scale, order=1, aa=True)
    io.imsave(str(path), img_as_ubyte(np.clip(img_small, 0, 1)), check_contrast=False)


def overlay_edges(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    edges = segmentation.find_boundaries(mask, mode="outer")
    out = rgb.copy()
    out[edges] = [0, 0, 1.0]  # blue
    return out


def quick_rg_threshold(p: Path, thumb: float) -> float:
    """Compute Otsu on (R-G) of a thumbnail. Low RAM."""
    rgb = load_rgb_float(p)
    if thumb != 1.0:
        rgb = resize_safe(rgb, thumb, order=1, aa=True)
    rg_small = (rgb[..., 0] - rgb[..., 1]).astype(np.float32)
    return float(threshold_otsu(rg_small))


# -------- Plots --------
def jittered_scatter(ax, x_center, values, width=0.3):
    if len(values) == 0:
        return
    x = x_center + _JITTER_RNG.uniform(-width, width, size=len(values))
    ax.scatter(
        x,
        values,
        s=18,
        color=DOT_COLOR,
        edgecolor="black",
        linewidth=0.3,
        alpha=0.85,
        zorder=3,
    )


def violin_with_median(df: pd.DataFrame, outpath: Path):
    feat = "percent_area"
    groups = [g for g in ORDER_C if g in df["condition"].unique()]
    data = [df.loc[df["condition"] == g, feat].dropna().values for g in groups]

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    ax.yaxis.grid(True, linestyle="--", linewidth=0.6, alpha=0.4)
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for body in parts["bodies"]:
        body.set_facecolor(VIOLIN_COLOR)
        body.set_edgecolor("black")
        body.set_alpha(0.55)
        body.set_linewidth(0.7)

    for idx, values in enumerate(data, start=1):
        jittered_scatter(ax, idx, values)

    for i, g in enumerate(groups, start=1):
        y = df.loc[df["condition"] == g, feat].dropna().values
        if len(y) == 0:
            continue
        mu = float(np.mean(y))
        med = float(np.median(y))
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
        ci = 1.96 * sd / np.sqrt(len(y)) if len(y) > 1 else np.nan

        ax.plot([i], [med], "s", ms=5.0, color="black", mec="black", mew=0.5, zorder=5)
        ax.plot(
            [i], [mu], "D", ms=6.2, color=MEAN_COLOR, mec="black", mew=0.5, zorder=6
        )
        if np.isfinite(sd):
            ax.errorbar(
                [i],
                [mu],
                yerr=sd,
                fmt="none",
                ecolor="black",
                elinewidth=1.0,
                capsize=4,
                zorder=4,
            )
        if np.isfinite(ci):
            ax.errorbar(
                [i],
                [mu],
                yerr=ci,
                fmt="none",
                ecolor="#444444",
                elinewidth=2.0,
                alpha=0.7,
                zorder=3,
            )
        ax.text(
            i + MEAN_TEXT_OFFSET,
            mu,
            MEAN_TEXT_FMT.format(mu),
            ha="left",
            va="center",
            fontsize=8.5,
            color=MEAN_COLOR,
            fontweight="bold",
        )

    ax.set_xticks(range(1, len(groups) + 1))
    ax.set_xticklabels([two_line_label(g) for g in groups])
    ax.set_ylabel("ARS-stained area (%)")
    ax.set_title("Alizarin Red S — Area Fraction per Condition")
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


def bar_mean_sd_with_text(df: pd.DataFrame, outpath: Path):
    groups = [g for g in ORDER_C if g in df["condition"].unique()]
    means = [df.loc[df["condition"] == g, "percent_area"].mean() for g in groups]
    sds = [df.loc[df["condition"] == g, "percent_area"].std(ddof=1) for g in groups]

    fig, ax = plt.subplots(figsize=(12, 4))
    x = np.arange(len(groups))
    ax.bar(
        x, means, yerr=sds, capsize=4, color=VIOLIN_COLOR, edgecolor="black", alpha=0.75
    )
    ax.plot(
        x,
        means,
        linestyle="none",
        marker="D",
        color=MEDIAN_COLOR,
        markeredgecolor="black",
        markeredgewidth=0.4,
    )
    for xi, m in zip(x, means):
        ax.text(
            xi,
            m,
            f"{m:.1f}",
            ha="center",
            va="bottom",
            color=MEDIAN_COLOR,
            fontsize=9,
            weight="bold",
        )
    ax.set_xticks(x)
    ax.set_xticklabels([two_line_label(g) for g in groups])
    ax.set_ylabel("ARS-stained area (%)")
    ax.set_title("Alizarin Red S — Mean ± SD (with mean value)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=300)
    plt.close(fig)


# -------- Main --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True, help="Root folder with images.")
    ap.add_argument(
        "-o",
        "--out",
        "--output",
        dest="out",
        default="ars_out",
        help="Base output folder for masks, overlays, tables, and figures.",
    )
    ap.add_argument("--glob", default="**/*.tif", help="Recursive glob for images.")
    ap.add_argument(
        "--lcap",
        type=float,
        default=75.0,
        help="L* cap: only pixels darker than this are allowed.",
    )
    ap.add_argument(
        "--save-scale", type=float, default=0.5, help="Downscale factor for saved PNGs."
    )
    ap.add_argument(
        "--thumb",
        type=float,
        default=0.25,
        help="Downscale used for Otsu estimation (0.1–0.5 good).",
    )
    ap.add_argument(
        "--mode",
        choices=["global", "perimg"],
        default="global",
        help="Threshold strategy: median-of-per-image (global) or per-image.",
    )
    args = ap.parse_args()

    in_root = Path(args.input)
    out_root = Path(args.out)
    d_masks = out_root / "masks"
    d_ovl = out_root / "overlays"
    d_tabs = out_root / "tables"
    d_figs = out_root / "figs"
    for d in [d_masks, d_ovl, d_tabs, d_figs]:
        d.mkdir(parents=True, exist_ok=True)

    # 1) Discover images & conditions
    files = sorted(in_root.glob(args.glob))
    items = []
    for p in files:
        code, label = detect_condition(p)
        if label != "Unknown":
            items.append({"path": p, "cond_code": code, "condition": label})
    if not items:
        raise SystemExit("No C1..C8 images found. Check names/folders.")
    df_idx = pd.DataFrame(items)

    # 2) Build thresholds (lightweight)
    if args.mode == "global":
        t_list: List[float] = []
        for p in df_idx["path"]:
            try:
                t = quick_rg_threshold(p, thumb=args.thumb)
                t_list.append(t)
            except Exception as e:
                print(f"[skip T] {p.name}: {e}")
        if not t_list:
            raise SystemExit("No thresholds computed.")
        T_rg_global = float(np.median(t_list))
        print(
            f"[info] Global T_rg (median of per-image Otsu on thumb={args.thumb}): {T_rg_global:.5f}"
        )
    else:
        T_rg_global = np.nan  # unused

    # 3) Process images one-by-one (no caching)
    out_rows = []
    for p, cond_code, condition in df_idx[
        ["path", "cond_code", "condition"]
    ].itertuples(index=False):
        try:
            rgb = load_rgb_float(p)
            lab = color.rgb2lab(rgb).astype(np.float32)  # L* in [0..100]
            L = lab[..., 0]
            rg = (rgb[..., 0] - rgb[..., 1]).astype(np.float32)

            if args.mode == "global":
                T_rg = T_rg_global
            else:
                # per-image threshold from a thumbnail
                T_rg = quick_rg_threshold(p, thumb=args.thumb)

            mask = (rg > T_rg) & (L < args.lcap)

            stained = int(mask.sum())
            total = mask.size
            frac = stained / total * 100.0

            stem = p.stem
            save_downscaled_png(
                d_masks / f"{stem}_mask.png", mask.astype(np.float32), args.save_scale
            )
            ov = overlay_edges(rgb, mask)
            save_downscaled_png(d_ovl / f"{stem}_overlay.png", ov, args.save_scale)

            out_rows.append(
                {
                    "image": stem,
                    "path": str(p),
                    "cond_code": cond_code,
                    "condition": condition,
                    "percent_area": frac,
                    "stained_pixels": stained,
                    "total_pixels": total,
                    "T_rg_used": float(T_rg),
                }
            )
        except Exception as e:
            print(f"[error] {p.name}: {e}")

    if not out_rows:
        raise SystemExit("No images processed successfully.")

    df_res = pd.DataFrame(out_rows)
    df_res.to_csv(d_tabs / "ars_area_by_image.csv", index=False)

    # 4) Threshold summary
    if args.mode == "global":
        pd.DataFrame(
            [
                {
                    "rule": "mask = (R-G > T_rg) AND (L* < Lcap)",
                    "mode": "global-median",
                    "T_rg_global": float(T_rg_global),
                    "Lcap": float(args.lcap),
                    "thumb": float(args.thumb),
                    "n_images": int(len(df_idx)),
                }
            ]
        ).to_csv(d_tabs / "threshold_summary.csv", index=False)
    else:
        pd.DataFrame(
            [
                {
                    "rule": "mask = (R-G > T_rg_img) AND (L* < Lcap)",
                    "mode": "per-image",
                    "Lcap": float(args.lcap),
                    "thumb": float(args.thumb),
                    "n_images": int(len(df_idx)),
                }
            ]
        ).to_csv(d_tabs / "threshold_summary.csv", index=False)

    # 5) Figures
    violin_with_median(df_res, d_figs / "violin_percent_area.pdf")
    bar_mean_sd_with_text(df_res, d_figs / "bar_mean_sd_percent_area.pdf")

    print("✅ Done:", out_root)
    print("  Masks:", d_masks)
    print("  Overlays:", d_ovl)
    print(
        "  Tables:",
        d_tabs / "ars_area_by_image.csv",
        ",",
        d_tabs / "threshold_summary.csv",
    )
    print(
        "  Figures:",
        d_figs / "violin_percent_area.png",
        ",",
        d_figs / "bar_mean_sd_percent_area.png",
    )


if __name__ == "__main__":
    main()
