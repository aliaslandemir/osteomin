#!/usr/bin/env python3
"""
Threshold-sensitivity supplementary analysis for ARS in one condition (default: C2).

Example:
python ars_threshold.py \
  -input ../ars \
  --condition C2 \
  --th-min 0.0 \
  --th-max 1.0 \
  --n-thresh 101 \
  --selected-thresh 0.55 \
  --out_prefix ars_threshold \
  --out_dir ars_output_2
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from ars_code2 import apply_pub_rcparams, detect_condition, load_rgb_float01, ars_score_ratio


def parse_args():
    ap = argparse.ArgumentParser(
        description="Sweep ARS ratio threshold and quantify sensitivity for one condition."
    )
    ap.add_argument("-input", "--input", required=True, help="Input folder containing ARS images.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument("--condition", default="C2", help="Condition to analyze (default: C2).")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for score R/(G+B+eps).")
    ap.add_argument("--th-min", type=float, default=0.0, help="Minimum threshold.")
    ap.add_argument("--th-max", type=float, default=1.0, help="Maximum threshold.")
    ap.add_argument("--n-thresh", type=int, default=101, help="Number of threshold points.")
    ap.add_argument("--selected-thresh", type=float, default=0.55, help="Reference selected threshold.")
    ap.add_argument("--out_prefix", type=str, default="ars_threshold", help="Output filename prefix.")
    ap.add_argument("--out_dir", type=Path, default=Path("ars_output_2"), help="Output directory.")
    return ap.parse_args()


def _fraction_percent_from_scores(scores: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
    """Fast monotone evaluation using sorted scores and searchsorted."""
    s = np.sort(np.asarray(scores, dtype=np.float64).ravel())
    n = float(s.size)
    idx = np.searchsorted(s, thresholds, side="left")
    return (n - idx) / n * 100.0


def _plot_threshold_supp(
    thresholds: np.ndarray,
    per_image_frac: np.ndarray,
    selected_idx: int,
    selected_thr: float,
    condition: str,
) -> plt.Figure:
    mean = np.mean(per_image_frac, axis=0)
    std = np.std(per_image_frac, axis=0, ddof=1) if per_image_frac.shape[0] > 1 else np.zeros_like(mean)
    sem = std / max(np.sqrt(per_image_frac.shape[0]), 1.0)
    ci95 = 1.96 * sem

    fig, ax = plt.subplots(figsize=(7.2, 5.6))

    base_line = "#4b5563"
    base_fill = "#CBD5E1"
    per_img_color = "#9CA3AF"
    selected_red = "#D62828"

    for i in range(per_image_frac.shape[0]):
        ax.plot(thresholds, per_image_frac[i], color=per_img_color, alpha=0.18, linewidth=1.0, zorder=1)

    ax.fill_between(thresholds, mean - ci95, mean + ci95, color=base_fill, alpha=0.55, linewidth=0, zorder=2)
    ax.plot(thresholds, mean, color=base_line, linewidth=2.0, zorder=3)
    ax.scatter(thresholds, mean, s=24, color=base_line, alpha=0.28, edgecolor="none", zorder=4)

    ax.axvline(selected_thr, color=selected_red, linestyle=(0, (4, 2)), linewidth=1.8, zorder=5)
    ax.scatter(
        [selected_thr],
        [mean[selected_idx]],
        s=86,
        color=selected_red,
        edgecolor="white",
        linewidth=0.9,
        zorder=6,
    )

    ax.set_ylabel("ARS-positive area (% of field)")
    ax.set_xlabel(r"Ratio threshold on $R/(G+B)$")
    ax.set_title(f"Threshold Sensitivity in {condition} (Day 28 ARS)", fontsize=16.5, pad=8)
    ax.text(
        0.98,
        0.96,
        f"Red = selected threshold ({selected_thr:.2f})\nArea @ selected = {mean[selected_idx]:.2f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.5,
        color="#111111",
    )

    ax.set_xlim(float(np.min(thresholds)), float(np.max(thresholds)))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(labelsize=11.5)
    ax.grid(axis="y", color="#e5e7eb", linewidth=0.65, alpha=0.85)

    return fig


def main():
    apply_pub_rcparams()
    args = parse_args()

    if args.n_thresh < 3:
        raise ValueError("--n-thresh must be >= 3")
    if not (args.th_min < args.th_max):
        raise ValueError("--th-min must be smaller than --th-max")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    root = Path(args.input)
    files = sorted(root.glob(args.glob))
    if args.glob.lower().endswith(".tif"):
        files += sorted(root.glob(args.glob[:-4] + ".tiff"))
    files = sorted(set(files))
    if not files:
        raise RuntimeError(f"No files found in {root} with pattern {args.glob}")

    cond = args.condition.upper()
    cond_files = [p for p in files if detect_condition(p) == cond]
    if not cond_files:
        raise RuntimeError(f"No files found for condition {cond}")

    thresholds = np.linspace(args.th_min, args.th_max, args.n_thresh, dtype=np.float64)

    per_image_frac = []
    image_names = []
    for p in cond_files:
        rgb = load_rgb_float01(p)
        scores = ars_score_ratio(rgb, eps=args.eps)
        frac = _fraction_percent_from_scores(scores, thresholds)
        per_image_frac.append(frac)
        image_names.append(p.name)

    per_image_frac = np.asarray(per_image_frac, dtype=np.float64)
    selected_idx = int(np.argmin(np.abs(thresholds - float(args.selected_thresh))))
    selected_thr = float(thresholds[selected_idx])

    mean = np.mean(per_image_frac, axis=0)
    std = np.std(per_image_frac, axis=0, ddof=1) if per_image_frac.shape[0] > 1 else np.zeros_like(mean)
    sem = std / max(np.sqrt(per_image_frac.shape[0]), 1.0)
    ci95 = 1.96 * sem
    delta = mean - mean[selected_idx]

    summary = pd.DataFrame(
        {
            "threshold": thresholds,
            "mean_fraction_percent": mean,
            "std_fraction_percent": std,
            "sem_fraction_percent": sem,
            "ci95_fraction_percent": ci95,
            "delta_vs_selected_pp": delta,
            "selected_threshold_used": selected_thr,
        }
    )
    summary_path = args.out_dir / f"{args.out_prefix}_{cond}_threshold_scan_summary.csv"
    summary.to_csv(summary_path, index=False)

    long_rows = []
    for i, name in enumerate(image_names):
        for j, thr in enumerate(thresholds):
            long_rows.append(
                {
                    "image": name,
                    "condition": cond,
                    "threshold": float(thr),
                    "fraction_percent": float(per_image_frac[i, j]),
                }
            )
    long_df = pd.DataFrame(long_rows)
    long_path = args.out_dir / f"{args.out_prefix}_{cond}_threshold_scan_per_image.csv"
    long_df.to_csv(long_path, index=False)

    fig = _plot_threshold_supp(
        thresholds=thresholds,
        per_image_frac=per_image_frac,
        selected_idx=selected_idx,
        selected_thr=selected_thr,
        condition=cond,
    )

    png = args.out_dir / f"{args.out_prefix}_{cond}_supp_threshold_sensitivity.png"
    pdf = args.out_dir / f"{args.out_prefix}_{cond}_supp_threshold_sensitivity.pdf"
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)

    print(f"Condition: {cond}")
    print(f"N images: {len(cond_files)}")
    print(f"Selected threshold (grid snapped): {selected_thr:.4f}")
    print(f"Mean @ selected: {mean[selected_idx]:.4f}%")
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")
    print(f"Saved: {summary_path}")
    print(f"Saved: {long_path}")


if __name__ == "__main__":
    main()
