#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ARS quantification (R–G threshold) — ABSOLUTE units (publication-friendly) + collagen/ALP-style stats/plot

Keeps reading + segmentation logic the same:
- detect_condition() from folder or filename (C1–C8)
- basal-derived threshold on (R-G) using --percentile
- ARS mask = (R-G) > threshold
- per-image metrics:
    * fraction_percent = 100 * stained_px / total_px        (ABSOLUTE % area of field)
    * stained_area_um2 = stained_px * (px_um)^2             (ABSOLUTE µm²; only if --px provided)
    * total_area_um2   = total_px * (px_um)^2               (ABSOLUTE field area; only if --px provided)

UPDATED analysis + plot style (same as collagen/ALP):
- Stats are done on the ABSOLUTE metric you choose via --metric:
    * Global Kruskal–Wallis across all present conditions (reported)
    * Family A (lower, black): C1 vs C2–C8, MWU (two-sided) + BH
    * Family B (upper, red):   C2 vs C3–C8, MWU (two-sided) + BH (excluding C1)
- Plot:
    * clean violins + colored points + black mean±SEM
    * two star rows (black vs C1, red vs C2)
    * separators (C1–C2 and C6–C7)
    * Δmean row (mean difference vs C1) at bottom (in the SAME ABSOLUTE units)
    * descriptive stats header

Outputs:
- CSV: per-image metrics (absolute + derived)
- TXT: stats report for chosen metric
- PDF + PNG: plot for chosen metric

Examples:
1) Absolute % area (recommended default):
python ars_abs_dualref_pro.py -input ../ars --percentile 98 --metric fraction_percent \
  --out_prefix ars_abs --out_dir ars_output --overlay

2) Absolute µm² (requires px):
python ars_abs_dualref_pro.py -input ../ars --percentile 98 --px 1.62 --metric stained_area_um2 \
  --out_prefix ars_um2 --out_dir ars_output --overlay
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea
from PIL import Image  # fast JPEG overlays


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

# Comparison families (match collagen/ALP)
FAM_A_REF = "C1"
FAM_A_TARGETS = ["C2", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C1
FAM_B_REF = "C2"
FAM_B_TARGETS = ["C3", "C4", "C5", "C6", "C7", "C8"]        # vs C2 (exclude C1)

# ---- Publication-safe palette (Okabe–Ito-ish; distinct) ----
COND_COLORS = {
    "C1": "#2B2B2B",  # Basal: charcoal
    "C2": "#0072B2",  # Osteo ref: blue
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


# ---------------- labels ----------------
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


# ---------------- detection + IO (same behavior) ----------------
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


def make_cyan_overlay(rgb01: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    rgb = np.clip(rgb01, 0, 1).copy()
    cyan = np.array([0.0, 1.0, 1.0], dtype=np.float32)
    m = mask.astype(bool)
    rgb[m] = (1.0 - alpha) * rgb[m] + alpha * cyan
    out = (np.clip(rgb, 0, 1) * 255.0 + 0.5).astype(np.uint8)
    return out


def save_overlay_jpeg(arr_uint8: np.ndarray, out_path: Path, scale: float = 0.5):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im = Image.fromarray(arr_uint8, mode="RGB")
    if scale != 1.0:
        w, h = im.size
        im = im.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
    im.save(
        str(out_path.with_suffix(".jpg")),
        quality=80,
        subsampling=2,
        optimize=True,
    )


# ---------------- stats helpers (match collagen/ALP) ----------------
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


# ---------------- plotting (match collagen/ALP) ----------------
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
    ref_mean = df.loc[df["cond"] == "C1", value_col].mean()
    out = []
    for c in conds:
        if c == "C1":
            out.append("baseline")
        else:
            m = df.loc[df["cond"] == c, value_col].mean()
            d = m - ref_mean
            arrow = "↑" if d > 0 else "↓"
            # for large units (µm²), don't show too many decimals
            if abs(d) >= 1000:
                out.append(f"{d:+.0f} {arrow}")
            elif abs(d) >= 10:
                out.append(f"{d:+.1f} {arrow}")
            else:
                out.append(f"{d:+.2f} {arrow}")
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
    data = [df.loc[df["cond"] == c, value_col].dropna().astype(float).values for c in conds]

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


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-input", "--input", required=True, help="Input folder containing images.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")

    ap.add_argument("--px", type=float, default=None, help="Pixel size in µm (optional, required for µm² metric).")
    ap.add_argument(
        "--percentile",
        type=float,
        default=98.0,
        help="Basal percentile for (R-G) threshold (e.g., 95, 98, 99). Default: 98.",
    )

    ap.add_argument(
        "--metric",
        type=str,
        default="fraction_percent",
        choices=["fraction_percent", "stained_area_um2", "total_area_um2"],
        help="ABSOLUTE metric to plot/stats: fraction_percent (%% area) or stained_area_um2 (µm²) or total_area_um2 (µm²).",
    )

    # overlay export (QC)
    ap.add_argument("--overlay", action="store_true", help="Export cyan overlay JPEGs.")
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="Cyan overlay strength (0..1).")
    ap.add_argument("--overlay_scale", type=float, default=0.5, help="Downscale factor for overlay export.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")

    # outputs
    ap.add_argument("--out_prefix", type=str, default="ars_abs", help="Output filename prefix.")
    ap.add_argument("--out_dir", type=Path, default=Path("ars_output"), help="Output directory.")
    return ap.parse_args()


def main():
    args = parse_args()

    if not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be between 0 and 100 (exclusive).")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be between 0 and 1.")
    if args.metric.endswith("_um2") and (args.px is None or args.px <= 0):
        raise ValueError("--metric stained_area_um2/total_area_um2 requires a positive --px (µm per pixel).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise RuntimeError(f"No files found: {args.input} with glob '{args.glob}'")

    # 1) Build Basal R–G distribution
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

    # 2) Process images
    rows = []
    overlay_root = args.out_dir / "overlays"
    if args.overlay:
        overlay_root.mkdir(exist_ok=True)

    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue

        rgb = load_rgb_float(p)
        D = rgb[..., 0] - rgb[..., 1]
        mask = D > threshold

        stained_px = int(mask.sum())
        total_px = int(mask.size)
        fraction_percent = (stained_px / total_px) * 100.0 if total_px > 0 else np.nan

        if args.px is not None and args.px > 0:
            stained_area_um2 = stained_px * (args.px ** 2)
            total_area_um2 = total_px * (args.px ** 2)
        else:
            stained_area_um2 = np.nan
            total_area_um2 = np.nan

        rows.append(
            dict(
                image=p.name,
                cond=cond,
                threshold_percentile=args.percentile,
                threshold_value_rg=threshold,
                stained_px=stained_px,
                total_px=total_px,
                fraction_percent=fraction_percent,
                stained_area_um2=stained_area_um2,
                total_area_um2=total_area_um2,
            )
        )

        if args.overlay:
            overlay_uint8 = make_cyan_overlay(rgb, mask, alpha=args.overlay_alpha)
            out_path = overlay_root / cond / f"{Path(p.name).stem}_overlay"
            save_overlay_jpeg(overlay_uint8, out_path, scale=args.overlay_scale)

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid C1–C8 images processed.")

    conds = [c for c in CONDITIONS if (df["cond"] == c).any()]
    if "C1" not in conds:
        raise RuntimeError("Missing C1 (Basal).")
    if "C2" not in conds:
        raise RuntimeError("Missing C2 (Osteo).")

    # Save CSV (always includes all absolute metrics)
    csv_path = args.out_dir / f"{args.out_prefix}_ars_metrics_absolute.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    if args.overlay:
        print("Saved overlays to:", overlay_root)

    # 3) Stats on the chosen ABSOLUTE metric
    value_col = args.metric
    cond_to_vals: Dict[str, np.ndarray] = {
        c: df.loc[df["cond"] == c, value_col].dropna().astype(float).values for c in conds
    }

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

    # metric labeling
    if value_col == "fraction_percent":
        ylab = "ARS-positive area (% of field)"
    elif value_col == "stained_area_um2":
        ylab = r"ARS-positive area ($\mu m^2$)"
    else:
        ylab = r"Field area ($\mu m^2$)"

    title = f"ARS area fraction (R–G threshold from Basal, {args.percentile:g}th percentile)"

    rpt = args.out_dir / f"{args.out_prefix}_{value_col}_stats.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"{title}\n")
        f.write(f"Metric column: {value_col}\n")
        f.write("Test: Mann–Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini–Hochberg within each family\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Basal threshold: {args.percentile:g}th percentile of (R-G) in C1; value={threshold:.6g}\n")
        if args.px is not None:
            f.write(f"Pixel size: {args.px:g} µm/px\n")
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

    print("Saved stats:", rpt)

    # 4) Plot
    fig = plot_violin_pro(
        df=df,
        value_col=value_col,
        conds=conds,
        title=title,
        ylab=ylab,
        famA=famA,
        famB=famB,
        seed=args.seed,
    )

    pdf = args.out_dir / f"{args.out_prefix}_{value_col}.pdf"
    png = args.out_dir / f"{args.out_prefix}_{value_col}.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", pdf)
    print("Saved:", png)
    print("Done. Output saved to:", args.out_dir)


if __name__ == "__main__":
    main()



"""

python ars_code.py -input ../ars --percentile 98 --metric fraction_percent \
  --out_prefix ars_abs --out_dir ars_output --overlay


"""