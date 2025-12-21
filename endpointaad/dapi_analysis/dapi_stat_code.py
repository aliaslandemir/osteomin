#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPI nuclei segmentation + shape analysis (DAPI framework) +
ALP-style stats + plotting (dual reference MWU+BH, gaps, stars, Δmean row).

NEW (per request):
- Smaller dots in violin plots
- Added Integrated Intensity:
      Integrated Intensity = Mean Intensity × Area (in pixels)
  -> computed from regionprops mean_intensity and area (px)

Inputs:
- Folder of .tif/.tiff with condition prefix in filename: C1_, C2_, ... (case-insensitive)

Pipeline:
1) Segment nuclei from DAPI channel (Otsu + watershed)
2) Regionprops + derived metrics (area_um2, circ, aspect ratio, etc.)
3) Post-filter nuclei by area (µm²) and circularity
4) Stats on KEPT nuclei:
   - Global Kruskal–Wallis across all present conditions
   - Family A (black stars): C1 vs C2–C8, Mann–Whitney U (two-sided) + BH
   - Family B (red stars):   C2 vs C3–C8, Mann–Whitney U (two-sided) + BH (excluding C1)
5) Plot per feature:
   - clean violins + colored points + black mean±SEM
   - stars in two rows (black vs C1, red vs C2)
   - separators (C1–C2 and C6–C7)
   - Δmean row (mean difference vs C1) at bottom (in same units as feature)
   - descriptive stats header

Outputs (under --out):
- overlays/ (red all nuclei contours)
- overlays_analyzed/ (blue kept nuclei contours)
- labelmaps/
- figs_violin/ (PDF+PNG per feature)
- nuclei_measurements_all.csv
- nuclei_measurements_kept.csv
- stats_report_DAPI_features.txt
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

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea

from scipy import stats, ndimage
from skimage import exposure, filters, measure, util, segmentation, morphology, feature


# ========================== GLOBAL STYLE ==========================
rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 300,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 1.2,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.size": 10,
    "font.family": "DejaVu Sans",
    "legend.frameon": False,
    "figure.autolayout": True,
})


# ========================== CONDITIONS / LABELS (ALP style) ==========================
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

FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]

FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]

# Publication-safe palette (same idea as ALP)
COND_COLORS = {
    "C1": "#2B2B2B",  # charcoal
    "C2": "#0072B2",  # blue
    "C3": "#009E73",  # green
    "C4": "#E69F00",  # orange
    "C5": "#CC79A7",  # purple/pink
    "C6": "#D55E00",  # vermillion
    "C7": "#56B4E9",  # light blue
    "C8": "#8C6D31",  # brown/olive
}
VIOLIN_FACE = "#E6E6E6"
VIOLIN_EDGE = "#B3B3B3"
STAR_COLOR_A = "#000000"   # black (vs C1)
STAR_COLOR_B = "#B22222"   # dark red (vs C2)
SEPARATOR_LINE = "#F2F2F2"


def two_line_label(name: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": r"AA+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"AA+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"AA+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)":     r"AA+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)":    r"AA+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


# ========================== DAPI CONFIG (framework) ==========================
DEFAULT_INPUT_DIR = Path("input_images")
DEFAULT_OUTPUT_ROOT = Path("dapi_output")

# Imaging
DEFAULT_PIXEL_SIZE_UM = 0.21  # µm/px (set your microscope value)
DAPI_CHANNEL_INDEX = 0

# Segmentation params
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
    "integrated_intensity",  # NEW
]

# Post-filter thresholds (µm² and circularity)
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

    # Derived geometric metrics
    df["aspect_ratio"] = (df["major_axis_length"] + eps) / (df["minor_axis_length"] + eps)
    df["circularity"] = (4 * np.pi * df["area"]) / (np.maximum(df["perimeter"], eps) ** 2)
    df["equivalent_diameter_px"] = 2 * np.sqrt(df["area"] / np.pi)

    # Unit conversions
    s = float(pixel_size_um)
    df["area_um2"] = df["area"] * (s ** 2)
    df["equivalent_diameter_um"] = df["equivalent_diameter_px"] * s

    # NEW: integrated intensity (mean_intensity × area_px)
    # mean_intensity from regionprops_table is average intensity inside the nucleus
    # area is in pixels -> integrated intensity in "intensity·px"
    df["integrated_intensity"] = df["mean_intensity"] * df["area"]

    return df


# ========================== STATS HELPERS (ALP style) ==========================
def benjamini_hochberg(pvals: List[float]) -> List[float]:
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * m / (np.arange(m) + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    out = np.empty_like(adj)
    out[order] = adj
    return out.tolist()

def star_from_p(p: float) -> str:
    if not np.isfinite(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return ""

def sem(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size <= 1:
        return np.nan
    return float(np.std(x, ddof=1) / np.sqrt(x.size))

def kw_all_conditions(cond_to_vals: Dict[str, np.ndarray], conds_order: List[str]) -> Tuple[float, float]:
    groups = []
    for c in conds_order:
        v = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            groups.append(v)
    if len(groups) < 2:
        return np.nan, np.nan
    H, p = stats.kruskal(*groups)
    return float(H), float(p)

def pairwise_vs_ref_BH(
    cond_to_vals: Dict[str, np.ndarray],
    ref: str,
    targets: List[str],
    alpha: float,
) -> Dict[str, Dict[str, float]]:
    ref_vals = np.asarray(cond_to_vals.get(ref, np.array([])), dtype=float)
    ref_vals = ref_vals[np.isfinite(ref_vals)]

    out: Dict[str, Dict[str, float]] = {}
    names: List[str] = []
    p_raws: List[float] = []

    for c in targets:
        vals = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        vals = vals[np.isfinite(vals)]

        if ref_vals.size == 0 or vals.size == 0:
            out[c] = {"p_raw": np.nan, "p_adj": np.nan, "sig": False}
            continue

        _, p_raw = stats.mannwhitneyu(ref_vals, vals, alternative="two-sided")
        out[c] = {"p_raw": float(p_raw)}
        names.append(c)
        p_raws.append(float(p_raw))

    p_adjs = benjamini_hochberg(p_raws)
    for c, padj in zip(names, p_adjs):
        out[c]["p_adj"] = float(padj)
        out[c]["sig"] = bool(padj < alpha)
    return out


# ========================== PLOT HELPERS (ALP style) ==========================
def make_positions_with_gaps(conds: List[str]) -> Tuple[List[float], List[float]]:
    xs = []
    base = 0.0
    for c in conds:
        base += 1.0

        if c == "C3":
            base += 0.30   # gap between C2–C3

        if c == "C7":
            base += 0.35   # gap between C6–C7

        xs.append(base)

    idx = {c: i for i, c in enumerate(conds)}
    seps = []

    # separator C2–C3
    if "C2" in idx and "C3" in idx:
        seps.append((xs[idx["C2"]] + xs[idx["C3"]]) / 2.0)

    # separator C6–C7
    if "C6" in idx and "C7" in idx:
        seps.append((xs[idx["C6"]] + xs[idx["C7"]]) / 2.0)

    return xs, seps


def mean_diff_labels_vs_c1(df: pd.DataFrame, value_col: str, conds: List[str]) -> List[str]:
    ref_mean = df.loc[df["condition"] == "C1", value_col].mean()
    out = []
    for c in conds:
        if c == "C1":
            out.append("baseline")
        else:
            m = df.loc[df["condition"] == c, value_col].mean()
            d = m - ref_mean
            arrow = "↑" if d > 0 else "↓"
            if np.isfinite(d) and abs(d) < 0.0005:
                d = 0.0
            if np.isfinite(d):
                out.append(f"{d:+.2g} {arrow}")
            else:
                out.append("")
    return out

def add_stats_header(ax, fs=8.6, grey="0.35", red="#B22222"):
    t0 = TextArea("Stars: * p<0.05, ** p<0.01, *** p<0.001 (BH)",
                  textprops=dict(color=grey, fontsize=fs))
    s0 = TextArea("   |   ", textprops=dict(color=grey, fontsize=fs))
    t1 = TextArea("Lower (black): vs C1 (MWU + BH)",
                  textprops=dict(color=grey, fontsize=fs))
    s1 = TextArea("   |   ", textprops=dict(color=grey, fontsize=fs))
    t2 = TextArea("Upper (red): vs C2 (MWU + BH, excl. C1)",
                  textprops=dict(color=red, fontsize=fs))

    row = HPacker(children=[t0, s0, t1, s1, t2], align="center", pad=0, sep=0)
    box = AnchoredOffsetbox(
        loc="upper center",
        child=row,
        frameon=False,
        bbox_to_anchor=(0.5, 1.0),
        bbox_transform=ax.transAxes,
        borderpad=0.0,
    )
    box.set_clip_on(False)
    ax.add_artist(box)

def plot_violin(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    title: str,
    ylab: str,
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    seed: int = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    xs, seps = make_positions_with_gaps(conds)
    data = [df.loc[df["condition"] == c, value_col].dropna().astype(float).values for c in conds]

    any_nonempty = any(v.size > 0 for v in data)
    allv = np.concatenate([v for v in data if v.size > 0]) if any_nonempty else np.array([0.0])
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    yr = (y_max - y_min) if y_max > y_min else 1.0

    yA = y_max + 0.045 * yr   # black stars
    yB = y_max + 0.080 * yr   # red stars

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.12)

    vp = ax.violinplot(
        data,
        positions=xs,
        widths=0.72,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_alpha(1.0)
        body.set_linewidth(1.0)

    for s in seps:
        ax.axvline(s, color="white", linewidth=12, zorder=1)
        ax.axvline(s, color=SEPARATOR_LINE, linewidth=1.5, zorder=2)

    for x, c, vals in zip(xs, conds, data):
        if vals.size == 0:
            continue
        jitter = rng.normal(0, 0.055, size=vals.size)
        ax.scatter(
            x + jitter,
            vals,
            s=22,  # <<< smaller dots (was 40)
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        m = float(np.mean(vals))
        s_ = sem(vals)
        ax.errorbar(
            [x], [m], yerr=[s_],
            fmt="o",
            color="black",
            markersize=4.5,
            capsize=3.2,
            linewidth=1.1,
            zorder=4,
        )

    for x, c in zip(xs, conds):
        if c in FAM_A_TARGETS:
            st = star_from_p(famA.get(c, {}).get("p_adj", np.nan))
            if st:
                ax.text(x, yA + 0.015 * yr, st, ha="center", va="bottom",
                        fontsize=12, color=STAR_COLOR_A, zorder=6)
        if c in FAM_B_TARGETS:
            st = star_from_p(famB.get(c, {}).get("p_adj", np.nan))
            if st:
                ax.text(x, yB + 0.015 * yr, st, ha="center", va="bottom",
                        fontsize=12, color=STAR_COLOR_B, zorder=6)

    delta_labels = mean_diff_labels_vs_c1(df, value_col, conds)
    y_delta = y_min - 0.11 * yr
    for x, lab in zip(xs, delta_labels):
        ax.text(x, y_delta, lab, ha="center", va="top", fontsize=8.2, color="0.35")

    ax.text(xs[0] - 0.75, y_delta, r"$\Delta$mean", ha="left", va="top",
            fontsize=8.2, color="0.35", clip_on=False)

    ax.set_xticks(xs)
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in conds], fontsize=9)

    ax.set_ylabel(ylab)
    ax.set_title(title, fontsize=13)
    ax.set_ylim(y_min - 0.17 * yr, y_max + 0.26 * yr)

    fig.tight_layout()
    add_stats_header(ax)
    return fig


# ========================== SEGMENTATION (DAPI framework) ==========================
def segment_and_measure_one(image_path: Path, out_dirs: Dict[str, Path]) -> pd.DataFrame | None:
    try:
        dapi_original = load_dapi_channel(image_path)
        dapi_float = util.img_as_float(dapi_original)

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
    p = argparse.ArgumentParser(description="DAPI nuclei shape analysis (DAPI seg + ALP-style stats/plots).")
    p.add_argument("-i", "--input", type=Path, default=DEFAULT_INPUT_DIR, help="Folder with .tif/.tiff images")
    p.add_argument("-o", "--out", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Output root folder")
    p.add_argument("--pixel_um", type=float, default=DEFAULT_PIXEL_SIZE_UM, help="Pixel size (µm/px)")

    # post-filter thresholds
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
        f.write("Segmentation: Otsu + watershed; clear_border; remove_small_objects\n")
        f.write(f"Pixel size: {args.pixel_um} µm/px\n")
        f.write(
            f"Filters: area {args.min_area_um2}-{args.max_area_um2} µm²; "
            f"circularity {args.min_circ}-{args.max_circ}\n"
        )
        f.write("Stats: MWU (two-sided) + BH within each family\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write("Integrated intensity: mean_intensity × area_px\n\n")

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
                "area_um2": "DAPI nuclei — area um2",
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
