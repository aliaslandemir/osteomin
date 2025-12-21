#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARS RGB channels (separate) + monochrome exports + per-image scatter (publication style)

Folder layout:
  ../ars/
    C1/   (basal images)
    C2/   (AA+β-GP images)

Run:
  python ars_channel.py --input ../ars --out ars_rgb_scatter.pdf --norm unit --stat mean --save_mono

Notes:
- Reads with PIL -> convert("RGB") (consistent).
- Per-image channel summary = mean or median over all pixels.
- norm:
    raw        -> 0..255
    unit       -> 0..1
    fraction   -> channel_mean / (R_mean+G_mean+B_mean)
    rg_over_gb -> R_mean / (G_mean + B_mean)  (and analogous for G, B in panels)
- Plot shows only: per-image dots + condition mean marker (NO SEM).
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


IMG_EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg", ".bmp")


def list_images_in_dir(d: Path):
    if not d.exists():
        return []
    return [p for p in sorted(d.iterdir()) if p.is_file() and p.suffix.lower() in IMG_EXTS]


def read_rgb_255(path: Path) -> np.ndarray:
    """RGB float32 in [0,255], via PIL convert('RGB')."""
    img = Image.open(str(path)).convert("RGB")
    return np.asarray(img, dtype=np.float32)


def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def save_mono_channels(rgb_255: np.ndarray, out_prefix: Path):
    """Save R/G/B as grayscale PNG (0..255)."""
    r = np.clip(rgb_255[..., 0], 0, 255).astype(np.uint8)
    g = np.clip(rgb_255[..., 1], 0, 255).astype(np.uint8)
    b = np.clip(rgb_255[..., 2], 0, 255).astype(np.uint8)

    Image.fromarray(r, mode="L").save(str(out_prefix) + "_R.png")
    Image.fromarray(g, mode="L").save(str(out_prefix) + "_G.png")
    Image.fromarray(b, mode="L").save(str(out_prefix) + "_B.png")


def summarize_channel(ch: np.ndarray, stat: str) -> float:
    flat = ch.reshape(-1)
    if stat == "mean":
        return float(np.mean(flat))
    if stat == "median":
        return float(np.median(flat))
    raise ValueError(f"Unknown stat: {stat}")


def safe_div(a: float, b: float, eps: float = 1e-12) -> float:
    return float(a) / float(b + eps)


def jitter(n: int, width: float, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(n) - 0.5) * 2.0 * width


def find_subdir_case_insensitive(root: Path, wanted: str) -> Path:
    """If root/wanted doesn't exist, try case-insensitive match among subdirs."""
    p = root / wanted
    if p.exists():
        return p
    wanted_lower = wanted.lower()
    for child in root.iterdir():
        if child.is_dir() and child.name.lower() == wanted_lower:
            return child
    return p  # will fail later with a clear error


def write_csv(rows, out_csv: Path):
    fieldnames = ["condition", "id", "path", "R", "G", "B", "R_over_GB"]
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-input", "--input", "-i", required=True, help="Root folder containing C1/ and C2/ subfolders")
    ap.add_argument("--c1_dir", default="C1", help="Subfolder name for basal (default: C1)")
    ap.add_argument("--c2_dir", default="C2", help="Subfolder name for AA+β-GP (default: C2)")

    ap.add_argument("--c1_name", default="C1 (Basal)")
    ap.add_argument("--c2_name", default="C2 (AA+β-GP)")

    ap.add_argument("--out", default="ars_rgb_scatter.pdf")
    ap.add_argument("--out_csv", default="ars_rgb_table.csv")

    ap.add_argument("--stat", choices=["mean", "median"], default="mean")
    ap.add_argument("--norm", choices=["raw", "unit", "fraction", "rg_over_gb"], default="unit")

    ap.add_argument("--save_mono", action="store_true")
    ap.add_argument("--mono_dir", default="mono_channels")

    ap.add_argument("--max_images", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)

    root = Path(args.input)
    if not root.exists():
        raise SystemExit(f"--input does not exist: {root}")

    c1_path = find_subdir_case_insensitive(root, args.c1_dir)
    c2_path = find_subdir_case_insensitive(root, args.c2_dir)

    c1_files = list_images_in_dir(c1_path)
    c2_files = list_images_in_dir(c2_path)

    if len(c1_files) == 0:
        raise SystemExit(f"No images found in: {c1_path} (supported: {IMG_EXTS})")
    if len(c2_files) == 0:
        raise SystemExit(f"No images found in: {c2_path} (supported: {IMG_EXTS})")

    if args.max_images and args.max_images > 0:
        c1_files = c1_files[: args.max_images]
        c2_files = c2_files[: args.max_images]

    mono_root = Path(args.mono_dir)
    if args.save_mono:
        ensure_dir(mono_root / "C1")
        ensure_dir(mono_root / "C2")

    def compute_rows(files, cond_label: str):
        out = []
        for p in files:
            rgb = read_rgb_255(p)

            r_m = summarize_channel(rgb[..., 0], args.stat)
            g_m = summarize_channel(rgb[..., 1], args.stat)
            b_m = summarize_channel(rgb[..., 2], args.stat)

            r_over_gb = safe_div(r_m, (g_m + b_m))

            if args.norm == "raw":
                R, G, B = r_m, g_m, b_m
            elif args.norm == "unit":
                R, G, B = r_m / 255.0, g_m / 255.0, b_m / 255.0
            elif args.norm == "fraction":
                denom = max(r_m + g_m + b_m, 1e-12)
                R, G, B = r_m / denom, g_m / denom, b_m / denom
            elif args.norm == "rg_over_gb":
                R = safe_div(r_m, (g_m + b_m))
                G = safe_div(g_m, (r_m + b_m))
                B = safe_div(b_m, (r_m + g_m))
            else:
                raise ValueError("Bad --norm")

            if args.save_mono:
                out_prefix = mono_root / cond_label / p.stem
                save_mono_channels(rgb, out_prefix)

            out.append(
                {
                    "condition": cond_label,
                    "id": p.stem,
                    "path": str(p),
                    "R": float(R),
                    "G": float(G),
                    "B": float(B),
                    "R_over_GB": float(r_over_gb),
                }
            )
        return out

    rows_c1 = compute_rows(c1_files, "C1")
    rows_c2 = compute_rows(c2_files, "C2")
    all_rows = rows_c1 + rows_c2

    out_csv = Path(args.out_csv)
    write_csv(all_rows, out_csv)

    # ---------- Plot ----------
    channel_specs = [
        ("R", "Red channel", "#D62728"),
        ("G", "Green channel", "#2CA02C"),
        ("B", "Blue channel", "#1F77B4"),
    ]

    def y_label():
        if args.norm == "raw":
            return f"{args.stat} intensity (0–255)"
        if args.norm == "unit":
            return f"{args.stat} intensity (0–1)"
        if args.norm == "fraction":
            return f"{args.stat} fraction (ch/sum)"
        return f"{args.stat} ratio (ch / others)"

    n1, n2 = len(rows_c1), len(rows_c2)
    x1, x2 = 0.0, 1.0
    jwidth = 0.10

    # global rc tweaks (small, clean)
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)

    for ax, (key, title, col) in zip(axes, channel_specs):
        y1 = np.array([r[key] for r in rows_c1], dtype=float)
        y2 = np.array([r[key] for r in rows_c2], dtype=float)

        j1 = jitter(len(y1), jwidth, rng)
        j2 = jitter(len(y2), jwidth, rng)

        # per-image dots
        ax.scatter(
            x1 + j1, y1,
            s=28, c=col, alpha=0.55,
            edgecolors="none", zorder=2
        )
        ax.scatter(
            x2 + j2, y2,
            s=28, c=col, alpha=0.55,
            edgecolors="none", zorder=2
        )

        # mean marker only (no SEM)
        m1, m2 = float(np.mean(y1)), float(np.mean(y2))
        ax.scatter(
            [x1, x2], [m1, m2],
            s=46, marker="D",
            facecolors="black", edgecolors="black", linewidths=1.2, alpha=0.75,
            zorder=4
        )

        ax.set_title(title)
        ax.set_xticks([x1, x2])
        ax.set_xticklabels([args.c1_name, args.c2_name], rotation=20, ha="right")
        ax.set_xlim(-0.45, 1.45)
        ax.set_ylabel(y_label())

        ax.grid(axis="y", alpha=0.25, zorder=1)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        # a bit of headroom
        ymin, ymax = ax.get_ylim()
        pad = 0.07 * (ymax - ymin + 1e-12)
        ax.set_ylim(ymin, ymax + pad)

    # one clean legend for "dots" vs "mean"
    legend_handles = [
        Line2D([0], [0], marker="o", linestyle="None", markersize=6,
               markerfacecolor="0.35", markeredgewidth=0, alpha=0.6,
               label="Per-image value"),
        Line2D([0], [0], marker="D", linestyle="None", markersize=6,
               markerfacecolor="black", markeredgecolor="black",
               label="Condition mean"),
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        ncol=2,
        frameon=False,
        bbox_to_anchor=(0.5, 1.08),
    )

    # better title (short, publication-ish)
    norm_label = {
        "raw": "raw (0–255)",
        "unit": "unit (0–1)",
        "fraction": "fraction (ch/sum)",
        "rg_over_gb": "ratio (ch/others)",
    }[args.norm]
    fig.suptitle(
        f"Alizarin Red S (ARS): per-image RGB summaries ({args.stat}); normalization = {norm_label}",
        y=1.12,
        fontsize=11,
    )

    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Saved plot:", args.out)
    print("Saved table:", str(out_csv))
    if args.save_mono:
        print("Saved monochrome channels to:", str(mono_root))


if __name__ == "__main__":
    main()




"""
python ars_channel.py --input ../ars --out ars_rgb_scatter.pdf --norm unit --stat mean --save_mono


"""