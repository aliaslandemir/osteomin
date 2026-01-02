#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DAPI nuclei segmentation + shape analysis + ALP Figure-2 master plot style (locked).

Analysis (unchanged):
- Segmentation: Gaussian blur + CLAHE -> Otsu -> fill holes -> distance transform
  -> peak seeds -> watershed -> clear border -> remove small objects
- Measurements use ORIGINAL DAPI intensities (not CLAHE image).
- Integrated intensity: mean_intensity * area_px
- Post-filter: area_um2 + circularity bounds
- Stats: Kruskal–Wallis overall + MWU two-sided with BH in two families
  * Family A: C1 vs C2–C8
  * Family B: C2 vs C3–C8

Plot (changed only to match ALP Figure-2 master):
- Column 1: Condition tags (two-line)
- Column 2: ID (C1..C8)
- Column 3: Horizontal violins + jittered points + mean(red circle) + median(black diamond)
- Column 4: Δmean (vs C1) with arrows
- Column 5: Significance stars (blank = ns), dual-family: black vs C1; red vs C2



python dapi_stat_code.py \
  --input ../confocal \
  --out dapi_output \
  --pixel_um 0.21 \
  --min_area_um2 140 \
  --max_area_um2 300 \
  --min_circ 0.6 \
  --max_circ 1.2 \
  --alpha 0.05 \
  --seed 0



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

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter

from scipy import stats, ndimage
from skimage import exposure, filters, measure, util, segmentation, morphology, feature


# =========================
# GLOBAL PLOT SETTINGS (identical to ALP master)
# =========================
def apply_pub_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


# ========================== CONDITIONS / LABELS ==========================
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

COND_COLORS = {
    "C1": "#BDCC16",
    "C2": "#1F5AA6",
    "C3": "#1B7F5A",
    "C4": "#C97A00",
    "C5": "#A65C8A",
    "C6": "#B24C1A",
    "C7": "#6AAED6",
    "C8": "#7A6A3A",
}

# ALP-style cosmetics (locked)
VIOLIN_FACE = "#E9E9E9"
VIOLIN_EDGE = "#B0B0B0"
MEAN_COLOR = "#B22222"
MEDIAN_COLOR = "#1F1F1F"
STAR_COLOR_A = "#000000"
STAR_COLOR_B = "#B22222"


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


def cond_id_to_pretty_name(c: str) -> str:
    return COND_CODE.get(c, c)


def cond_id_to_tag(c: str) -> str:
    return two_line_label(cond_id_to_pretty_name(c))


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
        "figs": out_root / "figs_fig2_style",
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

    df["integrated_intensity"] = df["mean_intensity"] * df["area"]
    return df


# ========================== STATS HELPERS ==========================
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
    return ""  # blank = ns

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


# =========================
# ALP FIGURE-2 MASTER PLOT (for DAPI features)
# =========================
def plot_fig2_tags_violin_delta_sig(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    famA: Dict[str, Dict[str, float]],
    famB: Dict[str, Dict[str, float]],
    suptitle: str,
    xlab: str,
    seed: int = 0,
) -> plt.Figure:
    rng = np.random.default_rng(seed)

    data = [
        df.loc[df["condition"] == c, value_col].dropna().astype(float).values
        for c in conds
    ]

    ref_vals = df.loc[df["condition"] == "C1", value_col].dropna().astype(float).values
    ref_mean = float(np.mean(ref_vals)) if ref_vals.size else np.nan

    dmeans: List[float] = []
    for c in conds:
        vals = df.loc[df["condition"] == c, value_col].dropna().astype(float).values
        if vals.size == 0 or not np.isfinite(ref_mean):
            dmeans.append(np.nan)
        else:
            dmeans.append(float(np.mean(vals) - ref_mean))

    def fmt_dmean(d: float) -> str:
        if not np.isfinite(d):
            return ""
        if abs(d) < 0.0005:
            d = 0.0
        arrow = "↑" if d > 0 else ("↓" if d < 0 else "→")
        return f"{d:+.2g} {arrow}"

    # xlim from pooled data with padding
    nonempty = [v for v in data if v.size > 0]
    if nonempty:
        allv = np.concatenate(nonempty)
        xmin = float(np.min(allv))
        xmax = float(np.max(allv))
        xr = (xmax - xmin) if xmax > xmin else (abs(xmax) if xmax != 0 else 1.0)
        pad = 0.12 * xr
        xlim = (xmin - pad, xmax + pad)
    else:
        xlim = (0.0, 1.0)

    fig = plt.figure(figsize=(15.0, 6.3))
    fig.subplots_adjust(left=0.03, right=0.995, bottom=0.12, top=0.86, wspace=0.16)

    gs = GridSpec(
        1, 5,
        width_ratios=[1.10, 0.60, 5.20, 0.80, 1.00],
        wspace=0.12,
        figure=fig,
    )

    axT  = fig.add_subplot(gs[0, 0])
    axID = fig.add_subplot(gs[0, 1], sharey=axT)
    axV  = fig.add_subplot(gs[0, 2], sharey=axT)
    axD  = fig.add_subplot(gs[0, 3], sharey=axT)
    axS  = fig.add_subplot(gs[0, 4], sharey=axT)

    y = np.arange(len(conds))
    axT.set_ylim(len(conds) - 0.5, -0.5)

    # row guides
    row_color, row_alpha, row_lw = "#000000", 0.10, 0.8
    for yy in y:
        for ax in (axT, axID, axV, axD, axS):
            ax.axhline(yy, color=row_color, alpha=row_alpha, linewidth=row_lw, zorder=0)

    title_pad = 10

    # Condition column
    axT.set_xlim(0, 1)
    axT.set_xticks([])
    axT.set_yticks(y)
    axT.set_yticklabels([])
    axT.set_title("Condition", pad=title_pad)
    for i, c in enumerate(conds):
        axT.text(
            0.5, i, cond_id_to_tag(c),
            ha="center", va="center",
            fontsize=12, color="#1a1a1a",
            linespacing=1.10,
        )
    for s in axT.spines.values():
        s.set_visible(False)

    # ID column
    axID.set_xlim(0, 1)
    axID.set_xticks([])
    axID.set_yticks(y)
    axID.set_yticklabels([])
    axID.set_title("ID", pad=title_pad)
    for i, c in enumerate(conds):
        axID.text(
            0.5, i, c,
            ha="center", va="center",
            fontsize=12, fontweight="bold",
            color="#1a1a1a",
        )
    for s in axID.spines.values():
        s.set_visible(False)

    # Violins
    vp = axV.violinplot(
        data,
        positions=y,
        vert=False,
        widths=0.80,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body in vp["bodies"]:
        body.set_facecolor(VIOLIN_FACE)
        body.set_edgecolor(VIOLIN_EDGE)
        body.set_alpha(1.0)
        body.set_linewidth(1.0)
    vp["cmedians"].set_color("#2b2b2b")
    vp["cmedians"].set_linewidth(1.0)

    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            continue
        yj = i + rng.normal(0, 0.06, size=vals.size)
        axV.scatter(
            vals, yj,
            s=22,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.6,
            alpha=0.92,
            zorder=3,
        )
        axV.scatter(
            float(np.mean(vals)), i,
            s=60, marker="o",
            color=MEAN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.70,
            zorder=5,
        )
        axV.scatter(
            float(np.median(vals)), i,
            s=60, marker="D",
            color=MEDIAN_COLOR,
            edgecolor="white",
            linewidth=0.8,
            alpha=0.70,
            zorder=5,
        )

    axV.set_xlim(*xlim)
    axV.set_xlabel(xlab)
    axV.set_title("Distribution", pad=14)
    axV.grid(axis="x", alpha=0.15)
    axV.set_yticks([])
    axV.yaxis.set_major_formatter(NullFormatter())
    axV.tick_params(axis="y", length=0)
    for s in axV.spines.values():
        s.set_visible(False)

    # Δmean
    axD.set_xlim(0, 1)
    axD.set_xticks([])
    axD.set_yticks([])
    axD.yaxis.set_major_formatter(NullFormatter())
    axD.tick_params(axis="y", length=0)
    axD.set_title("Δmean", pad=title_pad)
    axD.text(
        0.5, -0.085, "(vs C1)",
        transform=axD.transAxes,
        ha="center", va="top",
        fontsize=10, color="0.65",
    )
    for i, (c, d) in enumerate(zip(conds, dmeans)):
        if c == "C1":
            txt, col = "baseline", "0.45"
        else:
            txt, col = fmt_dmean(d), "0.25"
        axD.text(0.5, i, txt, ha="center", va="center", fontsize=10, color=col)
    for s in axD.spines.values():
        s.set_visible(False)

    # Significance
    axS.set_xlim(-0.6, 1.6)
    axS.set_yticks([])
    axS.yaxis.set_major_formatter(NullFormatter())
    axS.tick_params(axis="y", length=0)
    axS.set_xticks([])
    axS.set_title("Significance\n(blank = ns)", pad=title_pad - 2)
    axS.axvspan(-0.6, 0.5, color="0.0", alpha=0.015, zorder=0)
    axS.axvline(0.5, color="0.0", alpha=0.18, lw=1.0, zorder=1)
    axS.text(0.25, -0.09, "vs C1", transform=axS.transAxes,
             ha="center", va="top", fontsize=10)
    axS.text(0.75, -0.09, "vs C2", transform=axS.transAxes,
             ha="center", va="top", fontsize=10)

    for i, c in enumerate(conds):
        if c == "C1":
            axS.text(0.0, i, "ref", ha="center", va="center", fontsize=10, color="0.5")
        else:
            st = star_from_p(famA.get(c, {}).get("p_adj", np.nan))
            axS.text(0.0, i, st, ha="center", va="center", fontsize=10, color=STAR_COLOR_A)

        if c == "C2":
            axS.text(1.0, i, "ref", ha="center", va="center", fontsize=10, color="0.5")
        elif c != "C1":
            st = star_from_p(famB.get(c, {}).get("p_adj", np.nan))
            axS.text(1.0, i, st, ha="center", va="center", fontsize=10, color=STAR_COLOR_B)

    for s in axS.spines.values():
        s.set_visible(False)

    fig.suptitle(suptitle, fontsize=14, y=0.975)
    return fig


# ========================== SEGMENTATION ==========================
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

        # overlay (all nuclei, red contours)
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
    p = argparse.ArgumentParser(description="DAPI nuclei shape analysis (seg + ALP Figure-2 style plots).")
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
    apply_pub_rcparams()
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

        ylab_map = {
            "area_um2": r"Area ($\mu$m$^2$)",
            "equivalent_diameter_um": r"Equivalent diameter ($\mu$m)",
            "aspect_ratio": "Aspect ratio (major/minor)",
            "circularity": "Circularity",
            "eccentricity": "Eccentricity",
            "solidity": "Solidity",
            "integrated_intensity": "Integrated intensity (a.u.·px)",
        }

        title_map = {
            "area_um2": "DAPI nuclei — area",
            "equivalent_diameter_um": "DAPI nuclei — equivalent diameter",
            "aspect_ratio": "DAPI nuclei — aspect ratio",
            "circularity": "DAPI nuclei — circularity",
            "eccentricity": "DAPI nuclei — eccentricity",
            "solidity": "DAPI nuclei — solidity",
            "integrated_intensity": "DAPI nuclei — integrated intensity",
        }

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

            fig = plot_fig2_tags_violin_delta_sig(
                df=df_kept,
                value_col=feat,
                conds=conds_present,
                famA=famA,
                famB=famB,
                suptitle=title_map.get(feat, f"DAPI nuclei — {feat.replace('_',' ')}"),
                xlab=ylab_map.get(feat, feat.replace("_", " ")),
                seed=int(args.seed),
            )

            pdf = out_dirs["figs"] / f"dapi_{feat}_fig2.pdf"
            png = out_dirs["figs"] / f"dapi_{feat}_fig2.png"
            fig.savefig(pdf)  # vector PDF; savefig.dpi applies only to embedded rasters
            fig.savefig(png)  # 300 dpi from rcParams
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
