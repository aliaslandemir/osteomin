#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Actin coherency — publication version (fast length density + DAPI normalization)
Unified plotting + two-line condition labels.
"""

import warnings; warnings.filterwarnings("ignore", category=UserWarning)

import argparse, re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import tifffile as tf
from scipy.ndimage import convolve
from skimage import exposure, morphology
from skimage.feature import structure_tensor
from skimage.morphology import thin
from skimage.measure import block_reduce

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# ---------------------- STYLE ----------------------
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
VIOLIN_COLOR = "#E69F00"
DOT_COLOR = "#0072B2"
MEAN_COLOR = "#009E73"
MEAN_TEXT_FMT = "{:.2f}"
MEAN_TEXT_OFFSET = 0.14

# ---------------------- Defaults ----------------------
DEFAULT_INPUT = Path("input_images")
DEFAULT_OUTPUT = Path("actin_results")
DEFAULT_SIGMA = 1.5
DEFAULT_PIXSIZE = 0.21
DEFAULT_ACTIN_CHANNEL = 1
DEFAULT_DAPI_CHANNEL = 0

DEFAULT_WHITE_GAIN = 0.9
DEFAULT_ALPHA = 0.55
DEFAULT_COH_T = 0.6
DEFAULT_I_T = 0.12
DEFAULT_GAMMA_C = 0.85
DEFAULT_GAMMA_I = 1.25

DEFAULT_LEN_DS = 2
DEFAULT_MIN_COH_AREA = 16

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
ORDER_C = list(COND_CODE.values())

def two_line_label(pretty: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": r"A+$\beta$-GP",
        "Osteo+D/MK4 (L/L)": r"A+$\beta$-GP" + "\n" + r"$D_LK_L$",
        "Osteo+D/MK4 (L/H)": r"A+$\beta$-GP" + "\n" + r"$D_LK_H$",
        "Osteo+D/MK4 (H/L)": r"A+$\beta$-GP" + "\n" + r"$D_HK_L$",
        "Osteo+D/MK4 (H/H)": r"A+$\beta$-GP" + "\n" + r"$D_HK_H$",
        "Osteo+Dex (H)":       r"A+$\beta$-GP" + "\n" + r"$D_HK_0$",
        "Osteo+MK-4 (H)":      r"A+$\beta$-GP" + "\n" + r"$D_0K_H$",
    }
    return m.get(pretty, pretty)

# ---------------------- IO helpers ----------------------
def setup_dirs(output_root: Path):
    d_over = output_root / "overlays"
    d_figs = output_root / "figs"
    output_root.mkdir(parents=True, exist_ok=True)
    d_over.mkdir(exist_ok=True)
    d_figs.mkdir(exist_ok=True)
    return d_over, d_figs, (output_root / "actin_coherency_summary.csv")

def list_images(indir: Path):
    exts = {".tif", ".tiff", ".png", ".jpg", ".jpeg"}
    return [p for p in indir.rglob("*") if p.suffix.lower() in exts]

def detect_condition(p: Path) -> str:
    m = re.match(r"^(C[1-8])[_-]", p.name, flags=re.IGNORECASE)
    if m:
        return COND_CODE.get(m.group(1).upper(), m.group(1))
    m2 = re.search(r"(?:^|[\\/])(C[1-8])(?:[\\/]|$)", str(p.parent), flags=re.IGNORECASE)
    if m2:
        return COND_CODE.get(m2.group(1).upper(), m2.group(1))
    return "Unknown"

def pick_channel(arr, ch_index: int):
    shp = arr.shape
    if arr.ndim == 2: return arr
    if arr.ndim == 3:
        if shp[-1] <= 6: return arr[..., ch_index]
        if shp[0] <= 6:  return arr[ch_index, ...]
        return arr.max(axis=0)
    if arr.ndim == 4:
        ch_axes = [i for i, s in enumerate(shp) if s <= 6]
        if ch_axes:
            cax = ch_axes[0]
            sl = [slice(None)] * arr.ndim
            sl[cax] = ch_index
            sub = arr[tuple(sl)]
            if sub.ndim == 3: sub = sub.max(axis=0)
            return sub
        return arr[..., ch_index].max(axis=0)
    while arr.ndim > 2:
        arr = arr.max(axis=0)
    return arr

def compute_coherency(img, sigma: float):
    Axx, Axy, Ayy = structure_tensor(img, sigma=sigma)
    tmp = np.sqrt((Axx - Ayy) ** 2 + 4 * Axy**2)
    l1 = 0.5 * (Axx + Ayy + tmp)
    l2 = 0.5 * (Axx + Ayy - tmp)
    return np.clip((l1 - l2) / (l1 + l2 + 1e-12), 0, 1)

# ---------------------- Overlays ----------------------
def build_overlay_white_cyan(act_gray, coher, white_gain, alpha, coh_t, i_t, gamma_c, gamma_i):
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

# ---------------------- FAST coherent length density ----------------------
from scipy.ndimage import convolve
from skimage.morphology import thin
from skimage.measure import block_reduce

_JITTER_RNG = np.random.default_rng(42)

def length_density_fast(mask_bool: np.ndarray, pix_um: float, ds: int) -> float:
    if ds > 1:
        mask_ds = block_reduce(mask_bool, (ds, ds), func=np.max).astype(bool)
        pix_um_eff = pix_um * ds
    else:
        mask_ds = mask_bool
        pix_um_eff = pix_um
    if not np.any(mask_ds): return 0.0
    skel = thin(mask_ds)
    hv_kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.uint8)
    diag_kernel = np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=np.uint8)
    sk = skel.astype(np.uint8)
    hv_pairs = 0.5 * np.sum(sk * convolve(sk, hv_kernel, mode="constant"))
    diag_pairs = 0.5 * np.sum(sk * convolve(sk, diag_kernel, mode="constant"))
    length_um = hv_pairs * pix_um_eff + diag_pairs * (pix_um_eff * np.sqrt(2))
    return float(length_um)

# ---------------------- Per-image pipeline ----------------------
def process_one(p: Path, args, d_over: Path):
    try:
        arr = tf.imread(p)
        act = pick_channel(arr, args.actin_channel).astype(np.float32)
        dapi = pick_channel(arr, args.dapi_channel).astype(np.float32)

        lo_a, hi_a = np.percentile(act, (0.5, 99.5))
        act_eq = exposure.equalize_adapthist(
            np.clip((act - lo_a) / (hi_a - lo_a + 1e-6), 0, 1), clip_limit=0.02
        )
        lo_d, hi_d = np.percentile(dapi, (1, 99))
        dapi_norm = np.clip((dapi - lo_d) / (hi_d - lo_d + 1e-6), 0, 1)

        coh = compute_coherency(act_eq, sigma=args.sigma)
        cond = detect_condition(p)

        if args.save_overlays:
            ov = build_overlay_white_cyan(
                act_eq, coh, args.white_gain, args.alpha, args.coh_t, args.i_t, args.gamma_c, args.gamma_i
            )
            plt.imsave((d_over / f"{p.stem}_overlay.png"), ov)

        mask = (coh >= args.coh_t) & (act_eq >= args.i_t)
        if args.min_coh_area > 0:
            mask = morphology.remove_small_objects(mask, min_size=args.min_coh_area)

        H, W = act_eq.shape
        area_mm2 = (H * W * (args.pix**2)) / 1e6

        dapi_mask = dapi_norm > 0.3
        dapi_area_fraction = float(np.mean(dapi_mask))

        total_len_um = length_density_fast(mask, args.pix, args.len_ds)
        length_density = total_len_um / max(area_mm2, 1e-12)

        dapi_norm_factor = dapi_area_fraction if dapi_area_fraction > 1e-6 else np.nan
        mean_coh_norm = float(coh.mean()) / dapi_norm_factor
        len_density_norm = float(length_density) / dapi_norm_factor

        return {
            "file": p.name,
            "condition": cond,
            "mean_coherency": float(coh.mean()),
            "length_density_um_per_mm2": float(length_density),
            "dapi_area_fraction": dapi_area_fraction,
            "mean_coherency_normDAPI": float(mean_coh_norm),
            "length_density_um_per_mm2_normDAPI": float(len_density_norm),
        }
    except Exception as e:
        return {"file": p.name, "condition": "ERROR", "error": str(e)}

# ---------------------- Plot ----------------------
def jittered_scatter(ax, x_center, y, width=0.07, s=10, alpha=0.75):
    if len(y) == 0:
        return
    x = x_center + _JITTER_RNG.uniform(-width, width, size=len(y))
    ax.scatter(x, y, s=s, color=DOT_COLOR, edgecolor="none", alpha=alpha, zorder=5)

def violin_plot(df, feature, outpath):
    conds = [c for c in ORDER_C if c in df["condition"].unique()]
    data = [df.loc[df["condition"] == c, feature].dropna().values for c in conds]

    fig, ax = plt.subplots(figsize=(12.5, 5.2))
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for b in parts["bodies"]:
        b.set_facecolor(VIOLIN_COLOR)
        b.set_edgecolor("black")
        b.set_alpha(0.55)

    for i, y in enumerate(data, 1):
        jittered_scatter(ax, i, y)
        if len(y) == 0:
            continue
        mu = float(np.mean(y))
        sd = float(np.std(y, ddof=1)) if len(y) > 1 else np.nan
        ci = 1.96 * sd / np.sqrt(len(y)) if len(y) > 1 else np.nan
        med = float(np.median(y))
        ax.plot([i], [med], "s", ms=5.0, color="black", mec="black", mew=0.5, zorder=5)
        ax.plot([i], [mu], "D", color=MEAN_COLOR, ms=6, mec="black", mew=0.5, zorder=7)
        if np.isfinite(sd):
            ax.errorbar([i], [mu], yerr=sd, fmt="none", ecolor="black", elinewidth=1, capsize=3, zorder=5)
        if np.isfinite(ci):
            ax.errorbar([i], [mu], yerr=ci, fmt="none", ecolor="#444444", elinewidth=2, capsize=3, zorder=6)
        label = MEAN_TEXT_FMT.format(mu)
        ax.text(
            i + MEAN_TEXT_OFFSET,
            mu,
            label,
            color=MEAN_COLOR,
            fontsize=8.5,
            fontweight="bold",
            ha="left",
            va="center",
            zorder=8,
        )

    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels([two_line_label(c) for c in conds])
    ax.set_ylabel(feature.replace("_", " "))
    ax.set_title("")
    fig.tight_layout()
    fig.savefig(outpath, bbox_inches="tight", dpi=300)
    plt.close(fig)

# ---------------------- Main ----------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--input", default=str(DEFAULT_INPUT))
    parser.add_argument("-o", "--out", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sigma", type=float, default=DEFAULT_SIGMA)
    parser.add_argument("--pix", type=float, default=DEFAULT_PIXSIZE)
    parser.add_argument("--actin_channel", type=int, default=DEFAULT_ACTIN_CHANNEL)
    parser.add_argument("--dapi_channel", type=int, default=DEFAULT_DAPI_CHANNEL)
    parser.add_argument("--save_overlays", action="store_true", default=True)
    parser.add_argument("--white_gain", type=float, default=DEFAULT_WHITE_GAIN)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--coh_t", type=float, default=DEFAULT_COH_T)
    parser.add_argument("--i_t", type=float, default=DEFAULT_I_T)
    parser.add_argument("--gamma_c", type=float, default=DEFAULT_GAMMA_C)
    parser.add_argument("--gamma_i", type=float, default=DEFAULT_GAMMA_I)
    parser.add_argument("--len_ds", type=int, default=DEFAULT_LEN_DS)
    parser.add_argument("--min_coh_area", type=int, default=DEFAULT_MIN_COH_AREA)
    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.out)
    d_over, d_figs, csv_path = setup_dirs(output_dir)

    images = list_images(input_dir)
    rows = []
    with ProcessPoolExecutor() as ex:
        futs = {ex.submit(process_one, p, args, d_over): p for p in images}
        for fut in as_completed(futs):
            rows.append(fut.result())

    df = pd.DataFrame(rows)
    if "error" in df.columns:
        df = df[df["error"].isna()].drop(columns=["error"], errors="ignore")
    df.to_csv(csv_path, index=False)

    violin_plot(df, "mean_coherency", d_figs / "violin_mean_coherency.pdf")
    violin_plot(df, "length_density_um_per_mm2", d_figs / "violin_length_density.pdf")
    violin_plot(df, "mean_coherency_normDAPI", d_figs / "violin_mean_coherency_normDAPI.pdf")
    violin_plot(
        df,
        "length_density_um_per_mm2_normDAPI",
        d_figs / "violin_length_density_normDAPI.pdf",
    )

    print(f"[OK] Overlays  → {d_over}")
    print(f"[OK] Figures   → {d_figs}")
    print(f"[OK] CSV       → {csv_path}")

if __name__ == "__main__":
    main()
