#!/usr/bin/env python3
"""
Publication-ready collagen metrics — RAW / ABSOLUTE (NO fold normalization)
Dual-family stats + collagen/ALP-style plot

Based on your current collagen script :contentReference[oaicite:0]{index=0}, but UPDATED to:
- REMOVE fold vs C1 median completely
- Run stats on RAW metrics (absolute)
- Plot RAW metrics (absolute) with same violin + points + mean±SEM + dual star rows + Δmean row

Per image (RAW):
- C_sum   = sum(collagen intensities)
- A_total = total # pixels
- D_area  = # pixels where DAPI > 0  (assumes true background pixels are exactly 0)

RAW metrics:
- M1_field = C_sum / A_total   (collagen density per field; intensity/pixel)
- M2_cov   = C_sum / D_area    (collagen per DAPI-positive area; intensity/pixel in nuclei+cell regions proxy)
- f_cov    = D_area / A_total  (coverage proxy; fraction of pixels DAPI-positive)
- M3_dsum  = C_sum / D_sum     (collagen per total DAPI intensity; intensity/intensity)

Stats (per RAW metric):
- Global Kruskal–Wallis across all present conditions (reported)
- Family A (lower, black): C1 vs (C2–C8), Mann–Whitney U + BH
- Family B (upper, red):   C2 vs (C3–C8), Mann–Whitney U + BH (excluding C1)

Outputs:
- CSV metrics (RAW)
- one stats report TXT per RAW metric
- figures (PDF + PNG): M1_field, M2_cov, M3_dsum, f_cov

Run:
python collagen_publishable_dualref_RAW_pro.py \
  -input ../../../confocal \
  --alpha 0.05 \
  --out_prefix collagen_raw \
  --out_dir collagen_output
"""

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import tifffile as tf
from plot_style import (
    CONDITIONS,
    FAM_A_REF,
    FAM_A_TARGETS,
    FAM_B_REF,
    FAM_B_TARGETS,
    kw_all_conditions,
    pairwise_vs_ref_BH,
    plot_violin,
)
import matplotlib.pyplot as plt


# ---------------- defaults ----------------
INPUT_DIR = Path("input_images")
OUTPUT_DIR = Path("collagen_output")

DAPI_CHANNEL = 0
COLLAGEN_CHANNEL = 2


# ---------------- IO ----------------
def get_condition_from_name(stem: str) -> str:
    import re
    m = re.match(r"^(C[1-8])[_-]", stem, re.IGNORECASE)
    return m.group(1).upper() if m else "Unknown"


def read_channels(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    with tf.TiffFile(path) as tif:
        arr = tif.asarray()

    if arr.ndim == 2:
        raise ValueError("2D TIFF found; expected multichannel.")

    if arr.ndim == 3:
        # (C,H,W)
        if arr.shape[0] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError("Not enough channels.")
        dapi = arr[DAPI_CHANNEL].astype(np.float64)
        col = arr[COLLAGEN_CHANNEL].astype(np.float64)
    elif arr.ndim == 4:
        # (T/Z,C,H,W) -> first plane
        if arr.shape[1] <= max(DAPI_CHANNEL, COLLAGEN_CHANNEL):
            raise ValueError("Not enough channels.")
        dapi = arr[0, DAPI_CHANNEL].astype(np.float64)
        col = arr[0, COLLAGEN_CHANNEL].astype(np.float64)
    else:
        raise ValueError(f"Unsupported shape {arr.shape}")

    return dapi, col


def compute_row(path: Path) -> dict:
    dapi_raw, col_raw = read_channels(path)

    A_total = float(dapi_raw.size)
    C_sum = float(np.sum(col_raw))
    D_sum = float(np.sum(dapi_raw))
    D_area = float(np.count_nonzero(dapi_raw > 0))

    M1_field = (C_sum / A_total) if A_total > 0 else np.nan
    M2_cov = (C_sum / D_area) if D_area > 0 else np.nan
    f_cov = (D_area / A_total) if A_total > 0 else np.nan
    M3_dsum = (C_sum / D_sum) if D_sum > 0 else np.nan

    return {
        "image": path.name,
        "condition": get_condition_from_name(path.stem),
        "C_sum": C_sum,
        "A_total": A_total,
        "D_area": D_area,
        "D_sum": D_sum,
        "M1_field": M1_field,
        "M2_cov": M2_cov,
        "M3_dsum": M3_dsum,
        "f_cov": f_cov,
    }


# ---------------- main ----------------
def parse_args():
    p = argparse.ArgumentParser(description="Collagen RAW metrics + dual-family MWU+BH + pro plots (no fold).")
    p.add_argument("-input", type=Path, default=INPUT_DIR, help="Folder containing TIFF images.")
    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")
    p.add_argument("--out_prefix", type=str, default="collagen_raw", help="Output filename prefix.")
    p.add_argument("--out_dir", type=Path, default=OUTPUT_DIR, help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(list(args.input.glob("*.tif")) + list(args.input.glob("*.tiff")))
    if not paths:
        print("No images found in", args.input)
        return

    rows = []
    for pth in paths:
        try:
            rows.append(compute_row(pth))
        except Exception as e:
            print(f"[warn] {pth.name}: {e}")

    df = pd.DataFrame(rows)
    df = df[df["condition"].isin(CONDITIONS)].copy()
    if df.empty:
        print("No valid C1–C8 images after filtering.")
        return

    conds = [c for c in CONDITIONS if (df["condition"] == c).any()]
    print("Conditions present:", conds)

    if "C1" not in conds:
        raise ValueError("Missing C1 (Basal).")
    if "C2" not in conds:
        raise ValueError("Missing C2 (Osteo).")

    # save metrics (RAW)
    csv_path = args.out_dir / f"{args.out_prefix}_metrics_RAW.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    def cond_to_vals(col: str) -> Dict[str, np.ndarray]:
        out: Dict[str, np.ndarray] = {}
        for c in conds:
            v = df.loc[df["condition"] == c, col].dropna().astype(float).values
            if v.size:
                out[c] = v
        return out

    # ---- RAW plots ----
    # NOTE: units are "intensity per pixel" for M1/M2, "ratio" for f_cov, "intensity/intensity" for M3
    plot_specs = [
        ("M1_field", "Collagen density per field", "Mean collagen intensity per pixel"),
        ("M2_cov", "Collagen per DAPI-positive area", "Collagen intensity / DAPI-positive pixel"),
        ("M3_dsum", "Collagen per total DAPI intensity", "Collagen sum / DAPI sum (a.u.)"),
        ("f_cov", "Coverage: DAPI-positive fraction", "DAPI-positive area fraction"),
    ]

    for col, title, ylab in plot_specs:
        cmap = cond_to_vals(col)

        H, pkw = kw_all_conditions(cmap, conds)

        famA = pairwise_vs_ref_BH(
            cmap,
            ref=FAM_A_REF,
            targets=[c for c in FAM_A_TARGETS if c in conds],
            alpha=args.alpha,
        )
        famB = pairwise_vs_ref_BH(
            cmap,
            ref=FAM_B_REF,
            targets=[c for c in FAM_B_TARGETS if c in conds],
            alpha=args.alpha,
        )

        # stats report per plot
        rpt = args.out_dir / f"{args.out_prefix}_{col}_RAW_stats.txt"
        with open(rpt, "w", encoding="utf-8") as f:
            f.write(f"{title} (RAW)\n")
            f.write(f"Metric column: {col}\n")
            f.write("Test: Mann–Whitney U (two-sided)\n")
            f.write("Multiple testing: Benjamini–Hochberg within each family\n")
            f.write(f"Alpha: {args.alpha}\n\n")
            f.write(f"Kruskal–Wallis across all conditions: H={H:.4g}, p={pkw:.4g}\n\n")

            f.write("Family A (black): C1 vs targets\n")
            for c in [x for x in FAM_A_TARGETS if x in conds]:
                f.write(
                    f"  {c} vs C1: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                    f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n"
                )

            f.write("\nFamily B (red): C2 vs targets (excluding C1)\n")
            for c in [x for x in FAM_B_TARGETS if x in conds]:
                f.write(
                    f"  {c} vs C2: p_raw={famB.get(c, {}).get('p_raw', np.nan):.4g}, "
                    f"p_adj={famB.get(c, {}).get('p_adj', np.nan):.4g}\n"
                )

        fig = plot_violin(
            df=df,
            value_col=col,
            conds=conds,
            title=title,
            ylab=ylab,
            famA=famA,
            famB=famB,
            seed=args.seed,
        )

        pdf = args.out_dir / f"{args.out_prefix}_{col}_RAW.pdf"
        png = args.out_dir / f"{args.out_prefix}_{col}_RAW.png"
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {pdf}")
        print(f"Saved: {png}")
        print(f"Saved: {rpt}")


if __name__ == "__main__":
    main()


"""
Run:

python collagen_kw_stat.py \
  -input ../../confocal \
  --alpha 0.05 \
  --out_prefix collagen_raw \
  --out_dir collagen_output
  
"""
