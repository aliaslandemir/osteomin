#!/usr/bin/env python3
"""
Actin coherency — ALP-style publication stats + plots

What you get
- Actin processing framework (from your actin script):
    * read multichannel images
    * pick actin + dapi channels
    * CLAHE on actin, normalize dapi
    * structure-tensor coherency map
    * coherent-mask threshold + remove_small_objects
    * fast skeleton length (µm) -> length density (µm/mm²)
    * DAPI area fraction and DAPI-normalized metrics
    * optional overlay PNGs

- ALP-style framework (from your ALP script):
    * condition codes C1..C8
    * dual-family MWU + BH:
        Family A: C1 vs C2..C8  -> black stars (lower row)
        Family B: C2 vs C3..C8  -> red stars (upper row)
    * global Kruskal–Wallis
    * pro violin plot with:
        violins (neutral), colored dots, mean±SEM, separators, Δmean vs C1 row, header legend

Outputs
- CSV: per-image metrics
- TXT: stats report per feature
- PDF/PNG: one plot per feature (ALP-like style)
- Overlays: optional

Run example
python actin_publishable_alpstyle.py \
  --input input_images \
  --out actin_output \
  --actin_channel 1 \
  --dapi_channel 0 \
  --pix 0.21 \
  --sigma 1.5 \
  --coh_t 0.6 \
  --i_t 0.12 \
  --min_coh_area 16 \
  --len_ds 2 \
  --alpha 0.05 \
  --seed 0 \
  --save_overlays
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import tifffile as tf

from scipy import stats
from scipy.ndimage import convolve
from skimage import exposure, morphology
from skimage.feature import structure_tensor
from skimage.morphology import thin
from skimage.measure import block_reduce

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


# -------------------- CONDITIONS & LABELS (ALP-style) --------------------

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

# Publication-safe palette (same mapping philosophy as your ALP script)
COND_COLORS = {
    "C1": "#2B2B2B",
    "C2": "#0072B2",
    "C3": "#009E73",
    "C4": "#E69F00",
    "C5": "#CC79A7",
    "C6": "#D55E00",
    "C7": "#56B4E9",
    "C8": "#8C6D31",
}
VIOLIN_FACE = "#E6E6E6"
VIOLIN_EDGE = "#B3B3B3"
STAR_COLOR_A = "#000000"
STAR_COLOR_B = "#B22222"
SEPARATOR_LINE = "#F2F2F2"


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


# -------------------- IO / discovery --------------------

def list_images(indir: Path) -> List[Path]:
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return [p for p in indir.rglob("*") if p.suffix.lower() in exts]


def detect_condition_code(p: Path) -> str:
    """
    Returns condition code 'C1'..'C8' if found, else 'Unknown'.
    Supports:
      - filename prefix: C1_..., C2-..., etc.
      - parent folder: .../C3/... etc.
    """
    m = re.match(r"^(C[1-8])[_-]", p.name, flags=re.IGNORECASE)
    if m:
        return m.group(1).upper()
    m2 = re.search(r"(?:^|[\\/])(C[1-8])(?:[\\/]|$)", str(p.parent), flags=re.IGNORECASE)
    if m2:
        return m2.group(1).upper()
    return "Unknown"


def setup_dirs(output_root: Path):
    d_over = output_root / "overlays"
    d_figs = output_root / "figs"
    d_stats = output_root / "stats"
    output_root.mkdir(parents=True, exist_ok=True)
    d_over.mkdir(exist_ok=True)
    d_figs.mkdir(exist_ok=True)
    d_stats.mkdir(exist_ok=True)
    return d_over, d_figs, d_stats, (output_root / "actin_metrics_per_image.csv")


# -------------------- Channel picking (actin framework) --------------------

def pick_channel(arr: np.ndarray, ch_index: int) -> np.ndarray:
    """
    Robust channel picker for tifffile stacks.
    Tries common layouts:
      - (H,W)
      - (H,W,C) where C small
      - (C,H,W) where C small
      - (Z,C,H,W) or (C,Z,H,W), etc.
    Falls back to max projections until 2D.
    """
    shp = arr.shape
    if arr.ndim == 2:
        return arr

    if arr.ndim == 3:
        if shp[-1] <= 6:
            return arr[..., ch_index]
        if shp[0] <= 6:
            return arr[ch_index, ...]
        return arr.max(axis=0)

    if arr.ndim == 4:
        # find a "channel axis" (size <=6)
        ch_axes = [i for i, s in enumerate(shp) if s <= 6]
        if ch_axes:
            cax = ch_axes[0]
            sl = [slice(None)] * arr.ndim
            sl[cax] = ch_index
            sub = arr[tuple(sl)]
            if sub.ndim == 3:
                sub = sub.max(axis=0)
            return sub
        # else assume last axis is channel-ish
        return arr[..., ch_index].max(axis=0)

    while arr.ndim > 2:
        arr = arr.max(axis=0)
    return arr


# -------------------- Coherency + overlay (actin framework) --------------------

def compute_coherency(img: np.ndarray, sigma: float) -> np.ndarray:
    Axx, Axy, Ayy = structure_tensor(img, sigma=sigma)
    tmp = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy**2)
    l1 = 0.5 * (Axx + Ayy + tmp)
    l2 = 0.5 * (Axx + Ayy - tmp)
    return np.clip((l1 - l2) / (l1 + l2 + 1e-12), 0, 1)


def build_overlay_white_cyan(
    act_gray: np.ndarray,
    coher: np.ndarray,
    white_gain: float,
    alpha: float,
    coh_t: float,
    i_t: float,
    gamma_c: float,
    gamma_i: float,
) -> np.ndarray:
    base = act_gray.astype(np.float32)
    lo, hi = np.percentile(base, (1, 99.5))
    base = np.clip((base - lo) / max(hi - lo, 1e-6), 0, 1)

    gray = np.power(base, gamma_i) * 0.3
    base_rgb = np.stack([gray, gray, gray], axis=-1)

    c = np.power(np.clip(coher, 0, 1), gamma_c)
    mask = (c >= coh_t) & (base >= i_t)
    white = np.zeros_like(base)
    white[mask] = c[mask] * base[mask] * white_gain

    cyan = np.zeros((*base.shape, 3), dtype=np.float32)
    cyan[..., 1:] = white[..., None]
    return np.clip(base_rgb + alpha * cyan, 0, 1)


# -------------------- Fast skeleton length (actin framework) --------------------

def length_um_fast(mask_bool: np.ndarray, pix_um: float, ds: int) -> float:
    """
    Approx skeleton length in µm (HV edges + diag edges).
    Downsamples mask by block_reduce if ds>1 for speed.
    """
    if ds > 1:
        mask_ds = block_reduce(mask_bool, (ds, ds), func=np.max).astype(bool)
        pix_um_eff = pix_um * ds
    else:
        mask_ds = mask_bool
        pix_um_eff = pix_um

    if not np.any(mask_ds):
        return 0.0

    skel = thin(mask_ds)
    hv_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    diag_kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)

    sk = skel.astype(np.uint8)
    hv_pairs = 0.5 * np.sum(sk * convolve(sk, hv_kernel, mode="constant"))
    diag_pairs = 0.5 * np.sum(sk * convolve(sk, diag_kernel, mode="constant"))

    return float(hv_pairs * pix_um_eff + diag_pairs * (pix_um_eff * np.sqrt(2)))


# -------------------- Stats helpers (ALP-style) --------------------

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


# -------------------- Plot helpers (ALP-style) --------------------

def make_positions_with_gaps(conds: List[str]) -> Tuple[List[float], List[float]]:
    xs = []
    base = 0.0
    for c in conds:
        base += 1.0
        if c == "C2":
            base += 0.30  # gap after C1
        if c == "C7":
            base += 0.35  # gap after C6
        xs.append(base)

    idx = {c: i for i, c in enumerate(conds)}
    seps = []
    if "C1" in idx and "C2" in idx:
        seps.append((xs[idx["C1"]] + xs[idx["C2"]]) / 2.0)
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
            if abs(d) < 1e-12:
                d = 0.0
            out.append(f"{d:+.3g} {arrow}")
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


def plot_violin_pro(
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

    allv = np.concatenate([v for v in data if v.size > 0]) if any(v.size > 0 for v in data) else np.array([0.0])
    y_min = float(np.min(allv))
    y_max = float(np.max(allv))
    yr = (y_max - y_min) if y_max > y_min else 1.0

    yA = y_max + 0.045 * yr   # black stars
    yB = y_max + 0.080 * yr   # red stars

    fig, ax = plt.subplots(figsize=(10.4, 4.9))
    ax.set_axisbelow(True)
    ax.grid(axis="y", alpha=0.12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

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
            s=40,
            color=COND_COLORS.get(c, "#444444"),
            edgecolor="white",
            linewidth=0.7,
            alpha=0.95,
            zorder=3,
        )
        m = float(np.mean(vals))
        s = sem(vals)
        ax.errorbar(
            [x], [m], yerr=[s],
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


# -------------------- Per-image processing --------------------

def process_one(p: Path, args, d_over: Path) -> Dict:
    try:
        arr = tf.imread(p)

        act = pick_channel(arr, args.actin_channel).astype(np.float32)
        dapi = pick_channel(arr, args.dapi_channel).astype(np.float32)

        # Actin normalize + CLAHE (your actin framework)
        lo_a, hi_a = np.percentile(act, (0.5, 99.5))

        # Simple intensity normalization (no CLAHE)
        lo_a, hi_a = np.percentile(act, (1, 99))
        act_norm = np.clip((act - lo_a) / (hi_a - lo_a + 1e-6), 0, 1)


        # DAPI normalize (simple)
        lo_d, hi_d = np.percentile(dapi, (1, 99))
        dapi_norm = np.clip((dapi - lo_d) / (hi_d - lo_d + 1e-6), 0, 1)

        coh = compute_coherency(act_norm, sigma=args.sigma)
        cond = detect_condition_code(p)

        # mask
        mask = (coh >= args.coh_t) & (act_norm >= args.i_t)

        if args.min_coh_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=args.min_coh_area)

        # geometry
        H, W = act_norm.shape
        area_mm2 = (H * W * (args.pix ** 2)) / 1e6

        # DAPI area fraction (keep same threshold logic)
        dapi_mask = dapi_norm > args.dapi_thr
        dapi_area_fraction = float(np.mean(dapi_mask))

        # length density (µm/mm²)
        total_len_um = length_um_fast(mask, args.pix, args.len_ds)
        length_density_um_per_mm2 = total_len_um / max(area_mm2, 1e-12)

        # DAPI normalized
        dapi_norm_factor = dapi_area_fraction if dapi_area_fraction > 1e-6 else np.nan
        mean_coh = float(np.mean(coh))
        mean_coh_normDAPI = mean_coh / dapi_norm_factor
        len_density_normDAPI = float(length_density_um_per_mm2) / dapi_norm_factor

        if args.save_overlays:
            ov = build_overlay_white_cyan(
                act_norm, coh,
                white_gain=args.white_gain,
                alpha=args.overlay_alpha,
                coh_t=args.coh_t,
                i_t=args.i_t,
                gamma_c=args.gamma_c,
                gamma_i=args.gamma_i,
            )

            plt.imsave((d_over / f"{p.stem}_overlay.png"), ov)

        return {
            "file": p.name,
            "path": str(p),
            "condition": cond,
            "mean_coherency": mean_coh,
            "length_density_um_per_mm2": float(length_density_um_per_mm2),
            "dapi_area_fraction": dapi_area_fraction,
            "mean_coherency_normDAPI": float(mean_coh_normDAPI),
            "length_density_um_per_mm2_normDAPI": float(len_density_normDAPI),
        }

    except Exception as e:
        return {"file": p.name, "path": str(p), "condition": "ERROR", "error": str(e)}


# -------------------- Main --------------------

FEATURE_SPECS = [
    ("mean_coherency", "Mean coherency (a.u.)", "Actin coherency"),
    ("length_density_um_per_mm2", r"Length density ($\mu$m mm$^{-2}$)", "Actin coherent length density"),
    ("mean_coherency_normDAPI", "Mean coherency / DAPI area fraction (a.u.)", "Actin coherency (DAPI-normalized)"),
    ("length_density_um_per_mm2_normDAPI", r"Length density / DAPI area fraction ($\mu$m mm$^{-2}$)",
     "Actin length density (DAPI-normalized)"),
]


def parse_args():
    p = argparse.ArgumentParser(description="Actin analysis with ALP-style plots + MWU(BH) dual-family stats.")
    p.add_argument("--input", type=str, default="input_images")
    p.add_argument("--out", type=str, default="actin_output")

    # imaging params
    p.add_argument("--actin_channel", type=int, default=1)
    p.add_argument("--dapi_channel", type=int, default=0)
    p.add_argument("--pix", type=float, default=0.21, help="Pixel size in µm.")
    p.add_argument("--sigma", type=float, default=1.5, help="Structure tensor sigma.")
    p.add_argument("--coh_t", type=float, default=0.6, help="Coherency threshold.")
    p.add_argument("--i_t", type=float, default=0.12, help="Actin intensity threshold (on CLAHE image).")
    p.add_argument("--min_coh_area", type=int, default=16, help="Remove coherent objects smaller than this (px).")
    p.add_argument("--len_ds", type=int, default=2, help="Downsample factor for length estimation.")

    # DAPI
    p.add_argument("--dapi_thr", type=float, default=0.3, help="Threshold on normalized DAPI for area fraction.")

    # overlay
    p.add_argument("--save_overlays", action="store_true", default=False)
    p.add_argument("--white_gain", type=float, default=0.9)
    p.add_argument("--overlay_alpha", type=float, default=0.55)
    p.add_argument("--gamma_c", type=float, default=0.85)
    p.add_argument("--gamma_i", type=float, default=1.25)

    # stats/plot
    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold.")
    p.add_argument("--seed", type=int, default=0, help="Jitter seed.")
    p.add_argument("--workers", type=int, default=0, help="0 = use default ProcessPoolExecutor().")

    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input)
    out_dir = Path(args.out)
    d_over, d_figs, d_stats, csv_path = setup_dirs(out_dir)

    images = list_images(input_dir)
    if not images:
        print(f"No images found in: {input_dir}")
        return

    # Parallel processing
    rows = []
    if args.workers and args.workers > 0:
        ex_kwargs = dict(max_workers=args.workers)
    else:
        ex_kwargs = {}

    with ProcessPoolExecutor(**ex_kwargs) as ex:
        futs = {ex.submit(process_one, p, args, d_over): p for p in images}
        for fut in as_completed(futs):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    if "error" in df.columns:
        df_ok = df[df["error"].isna()].copy()
    else:
        df_ok = df.copy()

    if df_ok.empty:
        print("All images failed. Check errors in the CSV.")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV (with errors): {csv_path}")
        return

    # keep only valid conditions C1..C8
    df_ok = df_ok[df_ok["condition"].isin(CONDITIONS)].copy()
    if df_ok.empty:
        print("No images matched conditions C1..C8 (check naming: C1_*, folders, etc.)")
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV (with errors): {csv_path}")
        return

    # Save CSV (include also error rows for debugging)
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV: {csv_path}")

    # Condition order: only those present, in C1..C8 order
    conds = [c for c in CONDITIONS if (df_ok["condition"] == c).any()]
    if "C1" not in conds or "C2" not in conds:
        raise ValueError("Need at least C1 and C2 present to run dual-family comparisons.")

    # For each feature: stats + ALP-style plot
    for feat, ylab, title in FEATURE_SPECS:
        # build condition->vals
        cond_to_vals: Dict[str, np.ndarray] = {}
        for c in conds:
            v = df_ok.loc[df_ok["condition"] == c, feat].dropna().astype(float).values
            if v.size:
                cond_to_vals[c] = v

        H, pkw = kw_all_conditions(cond_to_vals, conds)

        famA = pairwise_vs_ref_BH(
            cond_to_vals,
            ref=FAM_A_REF,
            targets=[c for c in FAM_A_TARGETS if c in conds],
            alpha=args.alpha,
        )
        famB = pairwise_vs_ref_BH(
            cond_to_vals,
            ref=FAM_B_REF,
            targets=[c for c in FAM_B_TARGETS if c in conds],
            alpha=args.alpha,
        )

        # stats report
        rpt = d_stats / f"{feat}_stats.txt"
        with open(rpt, "w", encoding="utf-8") as f:
            f.write("Actin analysis — ALP-style stats\n")
            f.write(f"Feature: {feat}\n")
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

        # plot
        fig = plot_violin_pro(
            df=df_ok,
            value_col=feat,
            conds=conds,
            title=title,
            ylab=ylab,
            famA=famA,
            famB=famB,
            seed=args.seed,
        )
        pdf = d_figs / f"{feat}_ALPstyle.pdf"
        png = d_figs / f"{feat}_ALPstyle.png"
        fig.savefig(pdf, dpi=300, bbox_inches="tight")
        fig.savefig(png, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"[OK] {feat}: plot -> {pdf}")
        print(f"[OK] {feat}: stats -> {rpt}")

    if args.save_overlays:
        print(f"[OK] Overlays: {d_over}")
    print(f"[OK] Figures:  {d_figs}")
    print(f"[OK] Stats:    {d_stats}")


if __name__ == "__main__":
    main()



"""

To Run:



python actin_stat_code.py \
    --input ../confocal \
    --out actin_output \
    --actin_channel 1 \
    --dapi_channel 0 \
    --pix 0.21 \
    --save_overlays


"""