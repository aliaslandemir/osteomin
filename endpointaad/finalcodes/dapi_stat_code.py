#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPI nuclei segmentation + shape analysis + ALP-style stats/plots.

Plot (per feature):
- violins + jittered per-nucleus points
- mean marker (red dot) + median marker (blue diamond)
- stars in two rows:
    * black: C1 vs others (MWU + BH)
    * red:   C2 vs others excluding C1 (MWU + BH)
- separators: C2–C3 and C6–C7
- Δmean row: mean difference vs C1 (same units as feature)
- compact header + legend

Segmentation:
Gaussian blur + CLAHE -> Otsu -> fill holes -> distance transform -> peak seeds -> watershed
Measurements use ORIGINAL DAPI intensities (not CLAHE image).

Integrated intensity:
    integrated_intensity = mean_intensity * area_px
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
from pathlib import Path
import re
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tf

from plot_style import (
    COND_CODE,
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
from scipy import ndimage
from skimage import exposure, filters, measure, util, segmentation, morphology, feature


# ========================== DAPI CONFIG ==========================
DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_OUTPUT_ROOT = Path("dapi_output")

DEFAULT_PIXEL_SIZE_UM = 0.21
DAPI_CHANNEL_INDEX = 0

MIN_NUCLEUS_SIZE_PX = 100
CLAHE_CLIP_LIMIT = 0.02
GAUSSIAN_BLUR_SIGMA = 3.0
WATERSHED_PEAK_MIN_DIST = 10

SHAPE_PROPERTIES = [
    "label", "area", "perimeter", "minor_axis_length",
    "major_axis_length", "eccentricity", "solidity", "mean_intensity"
]

FEATURES_TO_PLOT = [
    "area_um2",
    "equivalent_diameter_um",
    "aspect_ratio",
    "circularity",
    "eccentricity",
    "solidity",
    "integrated_intensity",
]

DEFAULT_MIN_AREA_UM2 = 140.0
DEFAULT_MAX_AREA_UM2 = 300.0
DEFAULT_MIN_CIRC = 0.6
DEFAULT_MAX_CIRC = 1.2


# ========================== OUTPUT DIRS ==========================
def setup_output_dirs(out_root: Path) -> Dict[str, Path]:
    out_root.mkdir(parents=True, exist_ok=True)
    d = {
        "root": out_root,
        "overlays_all": out_root / "overlays",
        "overlays_kept": out_root / "overlays_analyzed",
        "labelmaps": out_root / "labelmaps",
        "figs": out_root / "figs_violin",
        "csv_all": out_root / "nuclei_measurements_all.csv",
        "csv_kept": out_root / "nuclei_measurements_kept.csv",
        "stats_txt": out_root / "stats_report_DAPI_features.txt",
    }
    for k in ["overlays_all", "overlays_kept", "labelmaps", "figs"]:
        d[k].mkdir(exist_ok=True)
    return d


# ========================== METADATA HELPERS ==========================
def get_condition_code_from_stem(stem: str) -> str | None:
    m = re.match(r"^(C[1-8])[_-]", stem, flags=re.IGNORECASE)
    return m.group(1).upper() if m else None

def condition_from_path(p: Path) -> str:
    code = get_condition_code_from_stem(p.stem)
    return code if (code in COND_CODE) else "Unknown"

def load_dapi_channel(image_path: Path) -> np.ndarray:
    with tf.TiffFile(image_path) as tif:
        data = tif.asarray()
    if data.ndim == 4:        # Z, C, Y, X
        dapi = data[0, DAPI_CHANNEL_INDEX, :, :]
    elif data.ndim == 3:      # C, Y, X
        dapi = data[DAPI_CHANNEL_INDEX, :, :]
    else:                     # Y, X
        dapi = data
    return np.asarray(dapi).squeeze()

def derive_shape_metrics(df: pd.DataFrame, pixel_size_um: float) -> pd.DataFrame:
    eps = 1e-12
    df = df.copy()

    df["aspect_ratio"] = (df["major_axis_length"] + eps) / (df["minor_axis_length"] + eps)
    df["circularity"] = (4 * np.pi * df["area"]) / (np.maximum(df["perimeter"], eps) ** 2)
    df["equivalent_diameter_px"] = 2 * np.sqrt(df["area"] / np.pi)

    s = float(pixel_size_um)
    df["area_um2"] = df["area"] * (s ** 2)
    df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * s

    # Integrated intensity in (a.u.·px): sum of raw intensities in the nucleus
    df["integrated_intensity"] = df["mean_intensity"] * df["area"]

    return df


def segment_and_measure_one(image_path: Path, out_dirs: Dict[str, Path]) -> pd.DataFrame | None:
    try:
        dapi_original = load_dapi_channel(image_path)
        dapi_float = util.img_as_float(dapi_original)

        # preprocess for segmentation
        dapi_blurred = filters.gaussian(dapi_float, sigma=GAUSSIAN_BLUR_SIGMA)
        dapi_eq = exposure.equalize_adapthist(dapi_blurred, clip_limit=CLAHE_CLIP_LIMIT)

        thresh = filters.threshold_otsu(dapi_eq)
        binary = dapi_eq > thresh
        binary_filled = ndimage.binary_fill_holes(binary)

        dist = ndimage.distance_transform_edt(binary_filled)
        coords = feature.peak_local_max(
            dist,
            min_distance=WATERSHED_PEAK_MIN_DIST,
            labels=binary_filled,
            exclude_border=False,
        )
        mask_markers = np.zeros(dist.shape, dtype=bool)
        if coords.size > 0:
            mask_markers[tuple(coords.T)] = True
        markers = measure.label(mask_markers)

        labeled = segmentation.watershed(-dist, markers, mask=binary_filled)
        labeled = segmentation.clear_border(labeled)
        labeled = morphology.remove_small_objects(labeled, min_size=MIN_NUCLEUS_SIZE_PX)

        labeled_final, num = measure.label(labeled > 0, return_num=True)
        if num == 0:
            print(f"> No nuclei found in {image_path.name}")
            return None

        # Save labelmap
        label_path = out_dirs["labelmaps"] / f"{image_path.stem}_labels.tif"
        tf.imwrite(label_path, labeled_final.astype(np.uint16), photometric="minisblack")

        # Measure features using ORIGINAL intensities
        props = measure.regionprops_table(
            labeled_final,
            intensity_image=dapi_original,
            properties=SHAPE_PROPERTIES,
        )
        df = pd.DataFrame(props)
        df["image_filename"] = image_path.name
        df["condition"] = condition_from_path(image_path)

        # Red overlay (all nuclei)
        disp = exposure.rescale_intensity(dapi_original, out_range=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray")
        ax.contour(labeled_final > 0, levels=[0.5], colors="red", linewidths=0.9)
        ax.set_axis_off()
        fig.savefig(out_dirs["overlays_all"] / f"{image_path.stem}_overlay.png",
                    dpi=80, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

        return df

    except Exception as e:
        print(f"!! ERROR {image_path.name}: {e}")
        return None


# ========================== MAIN ==========================
def parse_args():
    p = argparse.ArgumentParser(description="DAPI nuclei shape analysis (seg + ALP-style stats/plots).")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT_DIR, help="Folder with .tif/.tiff images")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root folder")
    p.add_argument("--pixel_um", type=float, default=DEFAULT_PIXEL_SIZE_UM, help="Pixel size (µm/px)")

    p.add_argument("--min_area_um2", type=float, default=DEFAULT_MIN_AREA_UM2)
    p.add_argument("--max_area_um2", type=float, default=DEFAULT_MAX_AREA_UM2)
    p.add_argument("--min_circ", type=float, default=DEFAULT_MIN_CIRC)
    p.add_argument("--max_circ", type=float, default=DEFAULT_MAX_CIRC)

    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter")
    return p.parse_args()


def main():
    args = parse_args()
    out_dirs = setup_output_dirs(args.out)

    paths = sorted(list(args.input.glob("*.tif")) + list(args.input.glob("*.tiff")))
    if not paths:
        print(f"No images found in {args.input}")
        raise SystemExit(0)

    all_rows = []
    try:
        from tqdm import tqdm
        iterator = tqdm(paths, desc="Processing Images")
    except Exception:
        iterator = paths

    for pth in iterator:
        df_one = segment_and_measure_one(pth, out_dirs)
        if df_one is not None and not df_one.empty:
            all_rows.append(df_one)

    if not all_rows:
        print("No nuclei processed.")
        raise SystemExit(0)

    df_all = pd.concat(all_rows, ignore_index=True)
    df_all = derive_shape_metrics(df_all, pixel_size_um=float(args.pixel_um))
    df_all.to_csv(out_dirs["csv_all"], index=False)
    print(f"[saved] {out_dirs['csv_all']}")

    # ---------- POST FILTERS ----------
    before_n = len(df_all)
    keep_mask = (
        df_all["area_um2"].between(args.min_area_um2, args.max_area_um2, inclusive="both")
        & df_all["circularity"].between(args.min_circ, args.max_circ, inclusive="both")
    )
    df_kept = df_all.loc[keep_mask].copy()
    after_n = len(df_kept)
    kept_pct = (after_n / before_n * 100.0) if before_n else 0.0
    print(
        f"Filtered nuclei: kept {after_n}/{before_n} ({kept_pct:.1f}%) "
        f"[area {args.min_area_um2}-{args.max_area_um2} µm², circularity {args.min_circ}-{args.max_circ}]"
    )
    df_kept.to_csv(out_dirs["csv_kept"], index=False)
    print(f"[saved] {out_dirs['csv_kept']}")

    # ---------- Blue overlays (kept nuclei) ----------
    kept_by_img = (
        df_kept.groupby("image_filename")["label"]
        .apply(lambda s: set(map(int, s.values)))
        .to_dict()
    )
    for img_name, kept_labels in kept_by_img.items():
        if not kept_labels:
            continue
        img_path = next((p for p in paths if p.name == img_name), None)
        if img_path is None:
            continue
        label_path = out_dirs["labelmaps"] / f"{Path(img_name).stem}_labels.tif"
        if not label_path.exists():
            continue

        dapi_original = load_dapi_channel(img_path)
        labeled_final = tf.imread(label_path).astype(np.int32)
        kept_mask_img = np.isin(labeled_final, list(kept_labels))

        disp = exposure.rescale_intensity(dapi_original, out_range=(0, 1))
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.imshow(disp, cmap="gray")
        ax.contour(kept_mask_img, levels=[0.5], colors="deepskyblue", linewidths=1.0)
        ax.set_axis_off()
        fig.savefig(out_dirs["overlays_kept"] / f"{Path(img_name).stem}_overlay_analyzed.png",
                    dpi=150, bbox_inches="tight", pad_inches=0)
        plt.close(fig)

    # ---------- Conditions present (ordered) ----------
    conds_present = [c for c in CONDITIONS if (df_kept["condition"] == c).any()]
    if not conds_present:
        raise RuntimeError("No recognized conditions (C1–C8) found after filtering.")
    if "C1" not in conds_present:
        raise RuntimeError("Missing C1 (Basal) after filtering.")
    if "C2" not in conds_present:
        raise RuntimeError("Missing C2 (Osteo) after filtering.")

    # ---------- Stats + plots per feature ----------
    with open(out_dirs["stats_txt"], "w", encoding="utf-8") as f:
        f.write("DAPI nuclei shape metrics — KEPT nuclei only\n")
        f.write("Segmentation: Gaussian+CLAHE -> Otsu -> watershed; clear_border; remove_small_objects\n")
        f.write(f"Pixel size: {args.pixel_um} µm/px\n")
        f.write(
            f"Filters: area {args.min_area_um2}-{args.max_area_um2} µm²; "
            f"circularity {args.min_circ}-{args.max_circ}\n"
        )
        f.write("Stats: MWU (two-sided) + BH within each family\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write("Integrated intensity: mean_intensity × area_px (=sum of raw intensities in nucleus)\n\n")

        for feat in FEATURES_TO_PLOT:
            if feat not in df_kept.columns:
                continue

            cond_to_vals: Dict[str, np.ndarray] = {}
            for c in conds_present:
                v = df_kept.loc[df_kept["condition"] == c, feat].dropna().astype(float).values
                if v.size:
                    cond_to_vals[c] = v

            H, pkw = kw_all_conditions(cond_to_vals, conds_present)

            famA = pairwise_vs_ref_BH(
                cond_to_vals,
                ref=FAM_A_REF,
                targets=[c for c in FAM_A_TARGETS if c in conds_present],
                alpha=float(args.alpha),
            )
            famB = pairwise_vs_ref_BH(
                cond_to_vals,
                ref=FAM_B_REF,
                targets=[c for c in FAM_B_TARGETS if c in conds_present],
                alpha=float(args.alpha),
            )

            f.write(f"=== Feature: {feat} ===\n")
            f.write(f"Kruskal–Wallis across all present conditions: H={H:.4g}, p={pkw:.4g}\n")
            f.write("Family A (black): C1 vs targets\n")
            for c in [x for x in FAM_A_TARGETS if x in conds_present]:
                f.write(f"  {c} vs C1: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                        f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n")
            f.write("Family B (red): C2 vs targets (excluding C1)\n")
            for c in [x for x in FAM_B_TARGETS if x in conds_present]:
                f.write(f"  {c} vs C2: p_raw={famB.get(c, {}).get('p_raw', np.nan):.4g}, "
                        f"p_adj={famB.get(c, {}).get('p_adj', np.nan):.4g}\n")
            f.write("\n")

            ylab_map = {
                "area_um2": r"Area ($\mu$m$^2$)",
                "equivalent_diameter_um": r"Equivalent diameter ($\mu$m)",
                "aspect_ratio": "Aspect ratio (major/minor)",
                "circularity": "Circularity",
                "eccentricity": "Eccentricity",
                "solidity": "Solidity",
                "integrated_intensity": "Integrated intensity (a.u.·px)",
            }
            ylab = ylab_map.get(feat, feat.replace("_", " "))

            title_map = {
                "area_um2": "DAPI nuclei — area",
                "integrated_intensity": "DAPI nuclei — integrated intensity",
            }
            title = title_map.get(feat, f"DAPI nuclei — {feat.replace('_', ' ')}")

            fig = plot_violin(
                df=df_kept,
                value_col=feat,
                conds=conds_present,
                title=title,
                ylab=ylab,
                famA=famA,
                famB=famB,
                seed=int(args.seed),
            )

            pdf = out_dirs["figs"] / f"dapi_{feat}.pdf"
            png = out_dirs["figs"] / f"dapi_{feat}.png"
            fig.savefig(pdf, dpi=300, bbox_inches="tight")
            fig.savefig(png, dpi=300, bbox_inches="tight")
            plt.close(fig)

    print(f"[saved] {out_dirs['stats_txt']}")
    print("\n✅ Done.")
    print(" - All-nuclei overlays (red):", out_dirs["overlays_all"])
    print(" - Kept-nuclei overlays (blue):", out_dirs["overlays_kept"])
    print(" - Label maps:", out_dirs["labelmaps"])
    print(" - Figures:", out_dirs["figs"])
    print(" - Tables:", out_dirs["csv_all"], "and", out_dirs["csv_kept"])


if __name__ == "__main__":
    main()



"""
To Run:

python dapi_stat_code.py \
  --input ../confocal \
  --out dapi_analysis_output \
  --pixel_um 0.21 \
  --alpha 0.05 \
  --seed 0
  
"""
