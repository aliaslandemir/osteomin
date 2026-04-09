"""
Compute ARS lacunarity from binary mineral masks across C1-C8.

Definition:
    Lambda(s) = var(M_s) / mean(M_s)^2 + 1
where M_s is the mineral-pixel count in sliding s x s windows.

Default mask mode matches ars_code2.py:
    score = R / (G + B), mineral if score >= ratio_thresh

Example:
.venv/bin/python endpointaad/ars_analysis/ars_lacunarity.py \
  --input endpointaad/ars \
  --glob "**/*.tif" \
  --mode ratio_fixed \
  --ratio_thresh 0.55 \
  --s_min 2 \
  --s_max 128 \
  --n_scales 20 \
  --use_log_spacing \
  --out_prefix ars_lacunarity \
  --out_dir endpointaad/ars_analysis/ars_output_lacunarity
"""

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from skimage import io


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
FAM_A_TARGETS = ["C1", "C3", "C4", "C5", "C6", "C7", "C8"]

BAR_COLOR = "#C0392B"
POINT_COLOR = "#7F1D1D"
MEAN_COLOR = "#111111"
STAR_COLOR_A = "#000000"


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


def detect_condition(path: Path) -> str:
    parent = path.parent.name.upper()
    if parent in COND_CODE:
        return parent
    m = COND_RE.match(path.name)
    return m.group(1).upper() if m else "UNKNOWN"


def load_rgb_float01(p: Path) -> np.ndarray:
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
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
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    return r / (g + b + eps)


def build_mask_ratio(rgb01: np.ndarray, threshold: float, eps: float) -> np.ndarray:
    score = ars_score_ratio(rgb01, eps=eps)
    return (score >= threshold).astype(np.uint8)


def compute_lacunarity(
    img: np.ndarray,
    s_min: int = 2,
    s_max: int = None,
    n_scales: int = 20,
    use_log_spacing: bool = True,
    window_step: int = 1,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    img = np.asarray(img)
    if img.ndim != 2:
        raise ValueError("Input image must be 2D.")
    if not np.array_equal(img, img.astype(bool)):
        raise ValueError("Input image must be binary (0/1).")

    h, w = img.shape
    if s_max is None:
        s_max = min(h, w) // 2
    s_min = max(2, int(s_min))
    s_max = max(s_min, int(s_max))
    if s_min >= s_max:
        raise ValueError("s_min must be < s_max and at least 2 pixels.")
    if window_step < 1:
        raise ValueError("window_step must be >= 1.")

    if use_log_spacing:
        s_vals = np.unique(np.round(np.logspace(np.log10(s_min), np.log10(s_max), num=n_scales)).astype(int))
    else:
        s_vals = np.unique(np.round(np.linspace(s_min, s_max, num=n_scales)).astype(int))

    integral = np.zeros((h + 1, w + 1), dtype=float)
    integral[1:, 1:] = np.cumsum(np.cumsum(img, axis=0), axis=1)

    def window_sums(s: int):
        h2 = h - s + 1
        w2 = w - s + 1
        if h2 <= 0 or w2 <= 0:
            return None
        i = integral
        br = i[s:, s:]
        tr = i[:-s, s:]
        bl = i[s:, :-s]
        tl = i[:-s, :-s]
        sums = br - tr - bl + tl
        if window_step > 1:
            sums = sums[::window_step, ::window_step]
        return sums.reshape(-1)

    lambda_vals = []
    valid_s_vals = []
    for s in s_vals:
        sums = window_sums(int(s))
        if sums is None:
            continue
        mu = float(np.mean(sums))
        var = float(np.var(sums))
        if mu <= 0:
            lam = np.nan
        else:
            lam = var / (mu * mu) + 1.0
        lambda_vals.append(lam)
        valid_s_vals.append(s)

    s_vals = np.asarray(valid_s_vals, dtype=float)
    lambda_vals = np.asarray(lambda_vals, dtype=float)

    if lambda_vals.size == 0 or np.all(np.isnan(lambda_vals)):
        lambda_max = np.nan
        s_at_max = np.nan
    else:
        idx_max = int(np.nanargmax(lambda_vals))
        lambda_max = float(lambda_vals[idx_max])
        s_at_max = float(s_vals[idx_max])

    return s_vals, lambda_vals, lambda_max, s_at_max


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


def pairwise_vs_ref_bh(
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


def kw_all_conditions(cond_to_vals: Dict[str, np.ndarray], conds_order: List[str]) -> Tuple[float, float]:
    groups = []
    for c in conds_order:
        v = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            groups.append(v)
    if len(groups) < 2:
        return np.nan, np.nan
    try:
        h, p = stats.kruskal(*groups)
    except ValueError:
        # SciPy raises when all values are identical across all groups.
        return 0.0, 1.0
    return float(h), float(p)


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


def plot_points_bars(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    famA: Dict[str, Dict[str, float]],
    ylab: str,
    title: str,
    seed: int,
    log_y: bool = False,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    data = [df.loc[df["cond"] == c, value_col].dropna().astype(float).values for c in conds]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.305, top=0.885)

    x_step = 0.56
    x = np.arange(len(conds), dtype=float) * x_step
    ymax = 1.0
    for i, vals in enumerate(data):
        if vals.size == 0:
            continue
        mean, sd = _mean_sd(vals)
        low = max(mean - sd, 0.0) if np.isfinite(sd) and sd > 0 else max(mean, 0.0)
        high = mean + sd if np.isfinite(sd) and sd > 0 else mean
        ymax = max(ymax, float(np.max(vals)), high)

        ax.bar(
            x[i],
            max(high - low, 0.0),
            bottom=low,
            width=0.42,
            color=BAR_COLOR,
            edgecolor="#1f2937",
            linewidth=1.15,
            alpha=0.48,
            zorder=2,
        )
        xj = x[i] + rng.normal(0, 0.020, size=vals.size)
        ax.scatter(xj, vals, s=36, marker="o", color=POINT_COLOR, edgecolor="#111111", linewidth=0.45, alpha=0.95, zorder=5)
        ax.hlines(mean, x[i] - 0.16, x[i] + 0.16, color=MEAN_COLOR, linewidth=1.6, zorder=6)
        ax.scatter(x[i], mean, s=58, marker="o", color=MEAN_COLOR, edgecolor="white", linewidth=0.65, zorder=7)

    if log_y:
        all_vals = np.concatenate([vals for vals in data if vals.size > 0])
        all_pos = all_vals[all_vals > 0]
        lo = max(float(np.min(all_pos)) * 0.8, 1e-6) if all_pos.size else 1e-6
        hi = max(float(np.max(all_pos)) * 1.35, lo * 10.0) if all_pos.size else 10.0
        ax.set_yscale("log")
        ax.set_ylim(lo, hi)
    else:
        yrange = max(ymax, 1e-9)
        pad = max(0.12 * yrange, 0.02 * max(abs(ymax), 1.0))
        ax.set_ylim(0.0, ymax + pad)
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)
    ax.set_xticks(x)
    ax.set_xticklabels([two_line_label(cond_id_to_pretty_name(c)) for c in conds], rotation=0, ha="center")
    ax.set_ylabel(ylab, fontsize=15.5)
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=12.0, pad=8)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    if log_y:
        y0, y1 = ax.get_ylim()
        y_sig = 10 ** (np.log10(y0) + 0.92 * (np.log10(y1) - np.log10(y0)))
    else:
        yrange = max(ymax, 1e-9)
        pad = max(0.12 * yrange, 0.02 * max(abs(ymax), 1.0))
        y_sig = ymax + pad * 0.45
    for i, c in enumerate(conds):
        st = "ref" if c == "C2" else star_from_p(famA.get(c, {}).get("p_adj", np.nan))
        ax.text(x[i], y_sig, st, ha="center", va="center", fontsize=13.8, color=STAR_COLOR_A)

    fig.suptitle(title, fontsize=18.5, y=0.965)
    return fig


def plot_lacunarity_curves(
    df_long: pd.DataFrame,
    conds: List[str],
    log_y: bool = True,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7.0, 5.8))
    cmap = plt.cm.tab10
    for i, c in enumerate(conds):
        sub = df_long[df_long["cond"] == c]
        if sub.empty:
            continue
        grp = sub.groupby("s", as_index=False)["lambda"].agg(["mean", "std"]).reset_index()
        s = grp["s"].to_numpy(dtype=float)
        m = grp["mean"].to_numpy(dtype=float)
        sd = grp["std"].to_numpy(dtype=float)
        color = cmap(i % 10)
        ax.plot(s, m, color=color, lw=1.8, label=c)
        lo = np.where(np.isfinite(sd), m - sd, m)
        hi = np.where(np.isfinite(sd), m + sd, m)
        ax.fill_between(s, lo, hi, color=color, alpha=0.18, linewidth=0.0)

    ax.set_xscale("log")
    ax.set_xlabel("Box size, s (px)")
    ax.set_ylabel("Lacunarity, Lambda(s)")
    if log_y:
        ax.set_yscale("log")
    ax.set_title("ARS lacunarity curves / Day 28")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(False)
    ax.legend(frameon=False, ncol=2, fontsize=9)
    return fig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder containing ARS images.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument("--mode", choices=["ratio_fixed", "ratio_basal_percentile"], default="ratio_fixed")
    ap.add_argument("--ratio_thresh", type=float, default=0.55, help="Fixed ratio threshold for R/(G+B).")
    ap.add_argument("--percentile", type=float, default=98.0, help="If basal percentile mode, percentile from C1.")
    ap.add_argument("--eps", type=float, default=1e-6, help="Epsilon for ratio denominator.")
    ap.add_argument("--s_min", type=int, default=2, help="Minimum box size for lacunarity.")
    ap.add_argument("--s_max", type=int, default=128, help="Maximum box size for lacunarity.")
    ap.add_argument("--n_scales", type=int, default=20, help="Number of s values.")
    ap.add_argument("--use_log_spacing", action="store_true", help="Use log spacing for box sizes.")
    ap.add_argument("--window_step", type=int, default=1, help="Stride for sliding windows (>=1).")
    ap.add_argument("--downsample", type=int, default=1, help="Downsample binary mask by this factor before lacunarity.")
    ap.add_argument("--workers", type=int, default=1, help="Parallel worker processes for per-image lacunarity.")
    ap.add_argument("--s_ref", type=float, default=32.0, help="Reference box size for fixed-scale Lambda summary.")
    ap.add_argument("--alpha", type=float, default=0.05, help="BH threshold for pairwise tests vs C2.")
    ap.add_argument("--seed", type=int, default=0, help="Seed for plot jitter.")
    ap.add_argument(
        "--figure_conds",
        type=str,
        default="all",
        help='Comma-separated conditions for figures (e.g., "C1,C2"). Use "all" for all detected conditions.',
    )
    ap.add_argument("--out_prefix", default="ars_lacunarity", help="Output filename prefix.")
    ap.add_argument("--out_dir", type=Path, default=Path("ars_output_lacunarity"), help="Output directory.")
    return ap.parse_args()


def process_one_image(
    p_str: str,
    threshold: float,
    eps: float,
    s_min: int,
    s_max: int,
    n_scales: int,
    use_log_spacing: bool,
    window_step: int,
    downsample: int,
    mode: str,
    thr_note: str,
):
    p = Path(p_str)
    cond = detect_condition(p)
    if cond not in COND_CODE:
        return None, []

    rgb = load_rgb_float01(p)
    mask = build_mask_ratio(rgb, threshold=threshold, eps=eps)
    if downsample > 1:
        mask = mask[::downsample, ::downsample]
    p_mask = float(np.mean(mask))

    s_max_eff = min(s_max, min(mask.shape) - 1)
    if s_max_eff <= s_min:
        return None, []

    s_vals, lambda_vals, lambda_max, s_at_max = compute_lacunarity(
        mask,
        s_min=s_min,
        s_max=s_max_eff,
        n_scales=n_scales,
        use_log_spacing=use_log_spacing,
        window_step=window_step,
    )

    row_summary = {
        "image": p.name,
        "cond": cond,
        "mode": mode,
        "threshold_value": threshold,
        "threshold_note": thr_note,
        "s_min": s_min,
        "s_max": s_max_eff,
        "n_scales": n_scales,
        "use_log_spacing": bool(use_log_spacing),
        "window_step": int(window_step),
        "downsample": int(downsample),
        "p_mask": p_mask,
        "lambda_max": lambda_max,
        "s_at_max": s_at_max,
    }
    row_long = [{"image": p.name, "cond": cond, "s": float(s), "lambda": float(lam)} for s, lam in zip(s_vals, lambda_vals)]
    return row_summary, row_long


def main():
    apply_pub_rcparams()
    args = parse_args()

    if args.s_min < 2:
        raise ValueError("--s_min must be >= 2")
    if args.s_max <= args.s_min:
        raise ValueError("--s_max must be > --s_min")
    if args.n_scales < 3:
        raise ValueError("--n_scales must be >= 3")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0")
    if args.window_step < 1:
        raise ValueError("--window_step must be >= 1")
    if args.downsample < 1:
        raise ValueError("--downsample must be >= 1")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise RuntimeError(f"No files found: {args.input} with glob '{args.glob}'")

    if args.mode == "ratio_basal_percentile" and not (0.0 < args.percentile < 100.0):
        raise ValueError("--percentile must be between 0 and 100.")

    if args.mode == "ratio_fixed":
        threshold = float(args.ratio_thresh)
        thr_note = f"Fixed threshold on R/(G+B): {threshold:.6g}"
    else:
        basal_vals = []
        for p in files:
            if detect_condition(p) != "C1":
                continue
            rgb = load_rgb_float01(p)
            basal_vals.append(ars_score_ratio(rgb, eps=args.eps).ravel())
        if not basal_vals:
            raise RuntimeError("No C1 images found for basal percentile threshold.")
        basal_vals = np.concatenate(basal_vals)
        threshold = float(np.percentile(basal_vals, args.percentile))
        thr_note = f"Basal-derived {args.percentile:g}th percentile threshold on R/(G+B): {threshold:.6g}"

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows_summary = []
    rows_long = []

    valid_files = [p for p in files if detect_condition(p) in COND_CODE]
    if args.workers == 1:
        for p in valid_files:
            row_summary, row_long = process_one_image(
                p_str=str(p),
                threshold=threshold,
                eps=args.eps,
                s_min=args.s_min,
                s_max=args.s_max,
                n_scales=args.n_scales,
                use_log_spacing=args.use_log_spacing,
                window_step=args.window_step,
                downsample=args.downsample,
                mode=args.mode,
                thr_note=thr_note,
            )
            if row_summary is not None:
                rows_summary.append(row_summary)
                rows_long.extend(row_long)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [
                ex.submit(
                    process_one_image,
                    str(p),
                    threshold,
                    args.eps,
                    args.s_min,
                    args.s_max,
                    args.n_scales,
                    args.use_log_spacing,
                    args.window_step,
                    args.downsample,
                    args.mode,
                    thr_note,
                )
                for p in valid_files
            ]
            for fut in as_completed(futures):
                row_summary, row_long = fut.result()
                if row_summary is not None:
                    rows_summary.append(row_summary)
                    rows_long.extend(row_long)

    df = pd.DataFrame(rows_summary).sort_values(["cond", "image"]).reset_index(drop=True)
    df_long = pd.DataFrame(rows_long)
    if df.empty:
        raise RuntimeError("No valid C1-C8 images processed.")

    conds = [c for c in CONDITIONS if (df["cond"] == c).any()]
    if "C2" not in conds:
        raise RuntimeError("Missing C2 (Osteo).")

    if args.figure_conds.strip().lower() == "all":
        conds_fig = conds
    else:
        req = [x.strip().upper() for x in args.figure_conds.split(",") if x.strip()]
        conds_fig = [c for c in CONDITIONS if c in req and c in conds]
        if not conds_fig:
            raise RuntimeError("--figure_conds did not match any detected conditions.")

    # Add per-image fixed-scale lacunarity at s_ref (nearest available s in each image curve).
    if not df_long.empty:
        sref = float(args.s_ref)
        pick_rows = []
        for (image, cond), sub in df_long.groupby(["image", "cond"], as_index=False):
            idx = int(np.argmin(np.abs(sub["s"].to_numpy(dtype=float) - sref)))
            row = sub.iloc[idx]
            pick_rows.append({"image": image, "cond": cond, "s_ref_used": float(row["s"]), "lambda_sref": float(row["lambda"])})
        df_sref = pd.DataFrame(pick_rows)
        df = df.merge(df_sref, on=["image", "cond"], how="left")
    else:
        df["s_ref_used"] = np.nan
        df["lambda_sref"] = np.nan

    # Occupancy-corrected lacunarity to reduce low-fill inflation:
    # Lambda_adj = (Lambda - 1) * p + 1, where p is mineral fill fraction.
    df["lambda_max_adj"] = (df["lambda_max"] - 1.0) * df["p_mask"] + 1.0
    df["lambda_sref_adj"] = (df["lambda_sref"] - 1.0) * df["p_mask"] + 1.0

    csv_img = args.out_dir / f"{args.out_prefix}_per_image.csv"
    csv_long = args.out_dir / f"{args.out_prefix}_per_scale.csv"
    df.to_csv(csv_img, index=False)
    df_long.to_csv(csv_long, index=False)
    print("Saved:", csv_img)
    print("Saved:", csv_long)

    summary = (
        df.groupby("cond", as_index=False)
        .agg(
            n_images=("image", "count"),
            p_mask_mean=("p_mask", "mean"),
            p_mask_sd=("p_mask", "std"),
            lambda_max_mean=("lambda_max", "mean"),
            lambda_max_sd=("lambda_max", "std"),
            lambda_max_adj_mean=("lambda_max_adj", "mean"),
            lambda_max_adj_sd=("lambda_max_adj", "std"),
            lambda_sref_mean=("lambda_sref", "mean"),
            lambda_sref_sd=("lambda_sref", "std"),
            lambda_sref_adj_mean=("lambda_sref_adj", "mean"),
            lambda_sref_adj_sd=("lambda_sref_adj", "std"),
            s_at_max_mean=("s_at_max", "mean"),
            s_at_max_sd=("s_at_max", "std"),
        )
        .sort_values(by="cond")
    )
    csv_cond = args.out_dir / f"{args.out_prefix}_summary_by_condition.csv"
    summary.to_csv(csv_cond, index=False)
    print("Saved:", csv_cond)

    cond_to_lmax = {c: df.loc[df["cond"] == c, "lambda_max"].dropna().astype(float).values for c in conds}
    cond_to_lmax_adj = {c: df.loc[df["cond"] == c, "lambda_max_adj"].dropna().astype(float).values for c in conds}
    cond_to_lref = {c: df.loc[df["cond"] == c, "lambda_sref"].dropna().astype(float).values for c in conds}
    cond_to_lref_adj = {c: df.loc[df["cond"] == c, "lambda_sref_adj"].dropna().astype(float).values for c in conds}
    cond_to_smax = {c: df.loc[df["cond"] == c, "s_at_max"].dropna().astype(float).values for c in conds}
    famA_lmax = pairwise_vs_ref_bh(cond_to_lmax, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    famA_lmax_adj = pairwise_vs_ref_bh(cond_to_lmax_adj, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    famA_lref = pairwise_vs_ref_bh(cond_to_lref, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    famA_lref_adj = pairwise_vs_ref_bh(cond_to_lref_adj, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    famA_smax = pairwise_vs_ref_bh(cond_to_smax, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    h_lmax, p_lmax = kw_all_conditions(cond_to_lmax, conds)
    h_lmax_adj, p_lmax_adj = kw_all_conditions(cond_to_lmax_adj, conds)
    h_lref, p_lref = kw_all_conditions(cond_to_lref, conds)
    h_lref_adj, p_lref_adj = kw_all_conditions(cond_to_lref_adj, conds)
    h_smax, p_smax = kw_all_conditions(cond_to_smax, conds)

    stats_path = args.out_dir / f"{args.out_prefix}_stats.txt"
    with stats_path.open("w", encoding="utf-8") as f:
        f.write("ARS Lacunarity / Day 28\n")
        f.write(f"Masking: {thr_note}\n")
        f.write(f"s_min={args.s_min}, s_max={args.s_max}, n_scales={args.n_scales}, use_log_spacing={bool(args.use_log_spacing)}\n")
        f.write(f"window_step={args.window_step}, downsample={args.downsample}, workers={args.workers}\n")
        f.write("Test: Mann-Whitney U (two-sided), BH across C2-vs-target comparisons\n")
        f.write(f"Alpha: {args.alpha}\n\n")
        f.write(f"Lambda_max: Kruskal-Wallis H={h_lmax:.4g}, p={p_lmax:.4g}\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2 (Lambda_max): p_raw={famA_lmax.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA_lmax.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )
        f.write("\n")
        f.write(f"Lambda_max_adj: Kruskal-Wallis H={h_lmax_adj:.4g}, p={p_lmax_adj:.4g}\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2 (Lambda_max_adj): p_raw={famA_lmax_adj.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA_lmax_adj.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )
        f.write("\n")
        f.write(f"Lambda(s_ref~{args.s_ref:g}px): Kruskal-Wallis H={h_lref:.4g}, p={p_lref:.4g}\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2 (Lambda_sref): p_raw={famA_lref.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA_lref.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )
        f.write("\n")
        f.write(f"Lambda_sref_adj: Kruskal-Wallis H={h_lref_adj:.4g}, p={p_lref_adj:.4g}\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2 (Lambda_sref_adj): p_raw={famA_lref_adj.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA_lref_adj.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )
        f.write("\n")
        f.write(f"s_at_max: Kruskal-Wallis H={h_smax:.4g}, p={p_smax:.4g}\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2 (s_at_max): p_raw={famA_smax.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA_smax.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )
    print("Saved:", stats_path)

    fig_curve = plot_lacunarity_curves(df_long=df_long, conds=conds_fig, log_y=True)
    curve_pdf = args.out_dir / f"{args.out_prefix}_curves.pdf"
    curve_png = args.out_dir / f"{args.out_prefix}_curves.png"
    fig_curve.savefig(curve_pdf)
    fig_curve.savefig(curve_png)
    plt.close(fig_curve)
    print("Saved:", curve_pdf)
    print("Saved:", curve_png)

    fig_lmax = plot_points_bars(
        df=df,
        value_col="lambda_max",
        conds=conds_fig,
        famA=famA_lmax,
        ylab="Lacunarity peak (Lambda_max, log scale)",
        title="ARS lacunarity peak / Day 28",
        seed=args.seed,
        log_y=True,
    )
    lmax_pdf = args.out_dir / f"{args.out_prefix}_lambda_max_fig2.pdf"
    lmax_png = args.out_dir / f"{args.out_prefix}_lambda_max_fig2.png"
    fig_lmax.savefig(lmax_pdf)
    fig_lmax.savefig(lmax_png)
    plt.close(fig_lmax)
    print("Saved:", lmax_pdf)
    print("Saved:", lmax_png)

    fig_lref = plot_points_bars(
        df=df,
        value_col="lambda_sref",
        conds=conds_fig,
        famA=famA_lref,
        ylab=f"Lacunarity at s~{args.s_ref:g}px",
        title=f"ARS lacunarity at fixed scale (s~{args.s_ref:g}px) / Day 28",
        seed=args.seed + 1,
        log_y=True,
    )
    lref_pdf = args.out_dir / f"{args.out_prefix}_lambda_sref_fig2.pdf"
    lref_png = args.out_dir / f"{args.out_prefix}_lambda_sref_fig2.png"
    fig_lref.savefig(lref_pdf)
    fig_lref.savefig(lref_png)
    plt.close(fig_lref)
    print("Saved:", lref_pdf)
    print("Saved:", lref_png)

    fig_lref_adj = plot_points_bars(
        df=df,
        value_col="lambda_sref_adj",
        conds=conds_fig,
        famA=famA_lref_adj,
        ylab=f"Occupancy-adjusted lacunarity at s~{args.s_ref:g}px",
        title=f"ARS adjusted lacunarity (s~{args.s_ref:g}px) / Day 28",
        seed=args.seed + 2,
        log_y=False,
    )
    lref_adj_pdf = args.out_dir / f"{args.out_prefix}_lambda_sref_adj_fig2.pdf"
    lref_adj_png = args.out_dir / f"{args.out_prefix}_lambda_sref_adj_fig2.png"
    fig_lref_adj.savefig(lref_adj_pdf)
    fig_lref_adj.savefig(lref_adj_png)
    plt.close(fig_lref_adj)
    print("Saved:", lref_adj_pdf)
    print("Saved:", lref_adj_png)

    print("Done. Output saved to:", args.out_dir)


if __name__ == "__main__":
    main()
