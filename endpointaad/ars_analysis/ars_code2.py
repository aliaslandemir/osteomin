
"""


Quantify ARS signal and generate summary outputs.

python ars_code2.py -input ../ars --mode ratio_fixed --ratio_thresh 0.55 \
  --metric fraction_percent --out_prefix ars_ratio1 --out_dir ars_output_2 --overlay
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from skimage import io
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy import stats
from PIL import Image


# ---------------- configurable ----------------
COND_RE = re.compile(r"^(C[1-8])", re.IGNORECASE)
COND_CODE = {
    "C1": "Basal",
    "C2": "Osteo",
    "C3": "Osteo + D_LK_L",
    "C4": "Osteo + D_LK_H",
    "C5": "Osteo + D_HK_L",
    "C6": "Osteo + D_HK_H",
    "C7": "Osteo + D_HK_0",
    "C8": "Osteo + D_0K_H",
}
CONDITIONS = list(COND_CODE.keys())

FAM_A_REF = "C2"
FAM_A_TARGETS = ["C1", "C3", "C4", "C5", "C6", "C7", "C8"]  # vs C2

# Assay colors (journal style, colorblind-safe)
BAR_COLOR = "#0072B2"
POINT_COLOR = "#1F4E79"

# Plot colors
MEAN_COLOR = "#B22222"
SD_COLOR = "#5A5A5A"
STAR_COLOR_A = "#000000"  # vs C2
 
# =========================
# GLOBAL PLOT SETTINGS
# =========================
def apply_pub_rcparams() -> None:
    mpl.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 12.5,
            "axes.titlesize": 14.0,
            "axes.labelsize": 13.0,
            "xtick.labelsize": 12.0,
            "ytick.labelsize": 12.0,
            "axes.linewidth": 0.8,
            "lines.linewidth": 1.0,
            "grid.linewidth": 0.6,
        }
    )


# ---------------- labels ----------------
def two_line_label(name: str) -> str:
    m = {
        "Basal": "Basal",
        "Osteo": "Osteo",
        "Osteo + D_LK_L": r"Osteo" + "\n" + r"$D_LK_L$",
        "Osteo + D_LK_H": r"Osteo" + "\n" + r"$D_LK_H$",
        "Osteo + D_HK_L": r"Osteo" + "\n" + r"$D_HK_L$",
        "Osteo + D_HK_H": r"Osteo" + "\n" + r"$D_HK_H$",
        "Osteo + D_HK_0": r"Osteo" + "\n" + r"$D_HK_0$",
        "Osteo + D_0K_H": r"Osteo" + "\n" + r"$D_0K_H$",
    }
    return m.get(name, name)


def cond_id_to_pretty_name(c: str) -> str:
    return COND_CODE.get(c, c)


def cond_id_to_tag(c: str) -> str:
    return two_line_label(cond_id_to_pretty_name(c))

def _mean_sd(vals: np.ndarray) -> Tuple[float, float]:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return np.nan, np.nan
    mean = float(np.mean(vals))
    if vals.size < 2:
        return mean, np.nan
    sd = float(np.std(vals, ddof=1))
    return mean, sd


# ---------------- detection + IO ----------------
def detect_condition(path: Path) -> str:
    parent = path.parent.name.upper()
    if parent in COND_CODE:
        return parent
    m = COND_RE.match(path.name)
    return m.group(1).upper() if m else "UNKNOWN"


def load_rgb_float01(p: Path) -> np.ndarray:
    """
    Load RGB image as float32 in [0,1], channel-last.
    Handles:
      - grayscale -> replicates
      - channel-first -> moves
      - alpha -> drops
      - uint8/uint16 -> rescales by observed max
    """
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # channel-first -> channel-last
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)

    if img.shape[-1] > 3:
        img = img[..., :3]

    img = img.astype(np.float32)
    maxv = float(np.max(img)) if img.size else 1.0
    if maxv > 1.0:
        img = img / maxv
    return np.clip(img, 0.0, 1.0)


def ars_score_ratio(rgb01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """
    Paper-aligned score for red stains:
        score = R / (G + B)
    """
    R = rgb01[..., 0]
    G = rgb01[..., 1]
    B = rgb01[..., 2]
    return R / (G + B + eps)


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


# ---------------- stats helpers ----------------
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
# FIGURE
# =========================
def plot_fig2_tags_bar_delta_sig(
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
        df.loc[df["cond"] == c, value_col].dropna().astype(float).values
        for c in conds
    ]

    fig, axB = plt.subplots(figsize=(6.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.305, top=0.885)

    x_step = 0.56
    x = np.arange(len(conds), dtype=float) * x_step

    means: List[float] = []
    sds: List[float] = []
    for i, (c, vals) in enumerate(zip(conds, data)):
        if vals.size == 0:
            means.append(np.nan)
            sds.append(np.nan)
            continue
        mean, sd = _mean_sd(vals)
        means.append(mean)
        sds.append(sd)
        axB.bar(
            x[i],
            mean,
            width=0.40,
            color=BAR_COLOR,
            edgecolor="#444444",
            linewidth=0.8,
            alpha=0.28,
            zorder=2,
        )
        if np.isfinite(mean):
            if np.isfinite(sd) and sd > 0.0:
                axB.errorbar(
                    x[i],
                    mean,
                    yerr=sd,
                    fmt="none",
                    ecolor="#222222",
                    elinewidth=1.0,
                    capsize=4.0,
                    capthick=1.0,
                    zorder=4,
                )
            xj = x[i] + rng.normal(0, 0.020, size=vals.size)
            axB.scatter(
                xj,
                vals,
                s=32,
                marker="o",
                color=POINT_COLOR,
                edgecolor="white",
                linewidth=0.6,
                alpha=0.9,
                zorder=5,
            )
            axB.scatter(
                x[i],
                mean,
                s=55,
                marker="o",
                color="#111111",
                edgecolor="white",
                linewidth=0.5,
                zorder=6,
            )

    finite_vals = np.concatenate([vals[np.isfinite(vals)] for vals in data if vals.size])
    finite_means = np.asarray([v for v in means if np.isfinite(v)], dtype=float)
    finite_sds = np.asarray([v for v in sds if np.isfinite(v)], dtype=float)
    ymax = 1.0
    if finite_vals.size and finite_means.size:
        ymax = max(
            float(np.max(finite_vals)),
            float(np.max(finite_means) + np.max(finite_sds) if finite_sds.size else np.max(finite_means)),
        )
    ypad_top = max(ymax * 0.22, 0.12)
    axB.set_ylim(0.0, ymax + ypad_top)
    axB.set_xlim(x[0] - 0.25, x[-1] + 0.25)

    labels = [two_line_label(cond_id_to_pretty_name(c)) for c in conds]
    axB.set_xticks(x)
    axB.set_xticklabels(labels, rotation=0, ha="center")
    axB.set_ylabel(xlab, fontsize=15.5)
    axB.set_xlabel("")
    axB.grid(False)
    axB.tick_params(axis="x", labelsize=12.0, pad=8)
    axB.tick_params(axis="y", labelsize=13.5)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)
    axB.spines["left"].set_linewidth(0.8)
    axB.spines["bottom"].set_linewidth(0.8)

    y_sig = ymax + ypad_top * 0.60
    for i, c in enumerate(conds):
        st = "ref" if c == "C2" else star_from_p(famA.get(c, {}).get("p_adj", np.nan))
        axB.text(x[i], y_sig, st, ha="center", va="center", fontsize=13.8, color=STAR_COLOR_A)

    fig.suptitle(suptitle, fontsize=20, y=0.965)
    return fig


# ---------------- CLI ----------------
def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("-input", "--input", required=True, help="Input folder containing images.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")

    ap.add_argument("--px", type=float, default=None, help="Pixel size in µm (required for µm² metrics).")

    # Thresholding options
    ap.add_argument("--mode", choices=["ratio_fixed", "ratio_basal_percentile"], default="ratio_fixed",
                    help="How to set threshold on ratio score.")
    ap.add_argument("--ratio_thresh", type=float, default=1.0,
                    help="Fixed threshold for score=R/(G+B). Default 1.0 (paper-style).")
    ap.add_argument("--percentile", type=float, default=98.0,
                    help="If --mode ratio_basal_percentile: C1 percentile for ratio threshold. Default 98.")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon to avoid division by zero.")

    ap.add_argument(
        "--metric",
        type=str,
        default="fraction_percent",
        choices=["fraction_percent", "stained_area_um2", "total_area_um2"],
        help="Metric used for plots and statistics.",
    )

    ap.add_argument("--overlay", action="store_true", help="Export cyan overlay JPEGs.")
    ap.add_argument("--overlay_alpha", type=float, default=0.45, help="Cyan overlay strength (0..1).")
    ap.add_argument("--overlay_scale", type=float, default=0.5, help="Downscale factor for overlay export.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")

    # outputs
    ap.add_argument("--out_prefix", type=str, default="ars_abs", help="Output filename prefix.")
    ap.add_argument("--out_dir", type=Path, default=Path("ars_output"), help="Output directory.")
    return ap.parse_args()


def main():
    apply_pub_rcparams()

    args = parse_args()

    if args.mode == "ratio_basal_percentile" and not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be between 0 and 100 (exclusive).")
    if not (0.0 <= args.overlay_alpha <= 1.0):
        raise ValueError("--overlay_alpha must be between 0 and 1.")
    if args.metric.endswith("_um2") and (args.px is None or args.px <= 0):
        raise ValueError("--metric stained_area_um2/total_area_um2 requires a positive --px (µm per pixel).")
    if args.ratio_thresh <= 0:
        raise ValueError("--ratio_thresh must be > 0.")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise RuntimeError(f"No files found: {args.input} with glob '{args.glob}'")

    # 1) Determine threshold on ratio score
    if args.mode == "ratio_fixed":
        threshold = float(args.ratio_thresh)
        thr_note = f"Fixed threshold on R/(G+B): {threshold:.6g}"
    else:
        basal_vals = []
        for p in files:
            if detect_condition(p) != "C1":
                continue
            rgb = load_rgb_float01(p)
            score = ars_score_ratio(rgb, eps=args.eps).ravel()
            basal_vals.append(score)

        if not basal_vals:
            raise RuntimeError("No basal (C1) images found to build percentile threshold.")

        basal_vals = np.concatenate(basal_vals)
        threshold = float(np.percentile(basal_vals, args.percentile))
        thr_note = f"Basal-derived {args.percentile:g}th percentile threshold on R/(G+B): {threshold:.6g}"

    print(thr_note)

    # 2) Process images
    rows = []
    overlay_root = args.out_dir / "overlays"
    if args.overlay:
        overlay_root.mkdir(exist_ok=True)

    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue

        rgb = load_rgb_float01(p)
        score = ars_score_ratio(rgb, eps=args.eps)
        mask = score >= threshold

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
                mode=args.mode,
                threshold_value=threshold,
                threshold_note=thr_note,
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
    if "C2" not in conds:
        raise RuntimeError("Missing C2 (Osteo).")

    csv_path = args.out_dir / f"{args.out_prefix}_ars_metrics_absolute.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)
    if args.overlay:
        print("Saved overlays to:", overlay_root)

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
    famB: Dict[str, Dict[str, float]] = {}

    if value_col == "fraction_percent":
        xlab = "ARS-positive area (% of field)"
    elif value_col == "stained_area_um2":
        xlab = r"ARS-positive area ($\mu m^2$)"
    else:
        xlab = r"Field area ($\mu m^2$)"

    suptitle = "ARS area fraction / Day 28"

    rpt = args.out_dir / f"{args.out_prefix}_{value_col}_stats.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write(f"{suptitle}\n")
        f.write(f"Metric column: {value_col}\n")
        f.write("Score: R/(G+B)\n")
        f.write(f"Thresholding: {thr_note}\n")
        f.write("Test: Mann–Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini–Hochberg across C2-vs-target comparisons\n")
        f.write(f"Alpha: {args.alpha}\n")
        if args.px is not None:
            f.write(f"Pixel size: {args.px:g} µm/px\n")
        f.write(f"Kruskal–Wallis across all conditions: H={H:.4g}, p={pkw:.4g}\n\n")

        f.write("Pairwise comparisons vs C2\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )

    print("Saved stats:", rpt)

    fig = plot_fig2_tags_bar_delta_sig(
        df=df,
        value_col=value_col,
        conds=conds,
        famA=famA,
        famB=famB,
        suptitle=suptitle,
        xlab=xlab,
        seed=args.seed,
    )

    pdf = args.out_dir / f"{args.out_prefix}_{value_col}_fig2.pdf"
    png = args.out_dir / f"{args.out_prefix}_{value_col}_fig2.png"
    fig.savefig(pdf)   # vector PDF; savefig.dpi applies to embedded rasters
    fig.savefig(png)   # 300 dpi because rcParams['savefig.dpi']=300
    plt.close(fig)
    print("Saved:", pdf)
    print("Saved:", png)

    means_path = args.out_dir / f"{args.out_prefix}_{value_col}_means.txt"
    with means_path.open("w", encoding="utf-8") as f:
        f.write("Condition\tMean\tSD\tN\n")
        for c in conds:
            vals = cond_to_vals.get(c, np.array([]))
            if vals.size:
                mean_val = float(np.mean(vals))
                sd_val = float(np.std(vals, ddof=1)) if vals.size > 1 else float("nan")
                n_val = int(vals.size)
            else:
                mean_val = float("nan")
                sd_val = float("nan")
                n_val = 0
            f.write(f"{c}\t{mean_val:.6g}\t{sd_val:.6g}\t{n_val}\n")

    print("Saved:", means_path)
    print("Done. Output saved to:", args.out_dir)


if __name__ == "__main__":
    main()
