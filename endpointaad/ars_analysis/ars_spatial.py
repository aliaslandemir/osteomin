"""
Quantify ARS heterogeneity with:
1) Global intensity spread: CV = std(I) / mean(I)
2) Spatial heterogeneity: H_spatial = mean(local variance)

Example:
python ars_spatial.py \
  --input ../ars \
  --glob "**/*.tif" \
  --channel red \
  --norm_mode minmax \
  --window 64 \
  --out_prefix ars_spatial \
  --out_dir ars_output_spatial
"""

import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import uniform_filter
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

# PLOS-friendly colorblind-safe styling (Okabe-Ito palette family)
BAR_COLOR = "#56B4E9"
POINT_COLOR = "#0072B2"
MEAN_COLOR = "#1A1A1A"
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


def cond_id_to_pretty_name(c: str) -> str:
    return COND_CODE.get(c, c)


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


def intensity_from_channel(rgb01: np.ndarray, channel: str) -> np.ndarray:
    if channel == "red":
        return rgb01[..., 0].astype(np.float32)
    if channel == "green":
        return rgb01[..., 1].astype(np.float32)
    if channel == "inv_green":
        return (1.0 - rgb01[..., 1]).astype(np.float32)
    # luminance-like grayscale
    return (0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]).astype(np.float32)


def normalize_intensity(
    img: np.ndarray,
    mode: str,
    p_low: float,
    p_high: float,
    eps: float,
) -> np.ndarray:
    arr = img.astype(np.float32)
    if mode == "minmax":
        lo = float(np.min(arr))
        hi = float(np.max(arr))
    else:
        lo = float(np.percentile(arr, p_low))
        hi = float(np.percentile(arr, p_high))

    den = max(hi - lo, eps)
    out = (arr - lo) / den
    return np.clip(out, 0.0, 1.0)


def ars_score_ratio(rgb01: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    return r / (g + b + eps)


def compute_cv(img01: np.ndarray, eps: float) -> float:
    mu = float(np.mean(img01))
    sigma = float(np.std(img01, ddof=0))
    return float(sigma / max(mu, eps))


def compute_spatial_heterogeneity(img01: np.ndarray, window: int) -> float:
    local_mean = uniform_filter(img01, size=window, mode="reflect")
    local_mean_sq = uniform_filter(img01 * img01, size=window, mode="reflect")
    local_var = np.clip(local_mean_sq - local_mean * local_mean, 0.0, None)
    return float(np.mean(local_var))


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


def kw_all_conditions(cond_to_vals: Dict[str, np.ndarray], conds_order: List[str]) -> Tuple[float, float]:
    groups = []
    for c in conds_order:
        v = np.asarray(cond_to_vals.get(c, np.array([])), dtype=float)
        v = v[np.isfinite(v)]
        if v.size > 0:
            groups.append(v)
    if len(groups) < 2:
        return np.nan, np.nan
    h, p = stats.kruskal(*groups)
    return float(h), float(p)


def plot_metric_by_condition(
    df: pd.DataFrame,
    value_col: str,
    conds: List[str],
    famA: Dict[str, Dict[str, float]],
    ylab: str,
    title: str,
    seed: int,
) -> plt.Figure:
    rng = np.random.default_rng(seed)
    data = [df.loc[df["cond"] == c, value_col].dropna().astype(float).values for c in conds]

    fig, ax = plt.subplots(figsize=(6.2, 6.0))
    fig.subplots_adjust(left=0.10, right=0.99, bottom=0.305, top=0.885)

    x_step = 0.56
    x = np.arange(len(conds), dtype=float) * x_step

    ymin_raw = np.nan
    ymax_raw = np.nan
    for i, vals in enumerate(data):
        if vals.size == 0:
            continue
        mean, sd = _mean_sd(vals)
        low = mean - sd if np.isfinite(sd) and sd > 0 else mean
        high = mean + sd if np.isfinite(sd) and sd > 0 else mean
        ymin_raw = np.nanmin([ymin_raw, np.min(vals), low])
        ymax_raw = np.nanmax([ymax_raw, np.max(vals), high])

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
        ax.scatter(
            xj,
            vals,
            s=36,
            marker="o",
            color=POINT_COLOR,
            edgecolor="#111111",
            linewidth=0.45,
            alpha=0.95,
            zorder=5,
        )
        ax.hlines(mean, x[i] - 0.16, x[i] + 0.16, color=MEAN_COLOR, linewidth=1.6, zorder=6)
        ax.scatter(x[i], mean, s=58, marker="o", color=MEAN_COLOR, edgecolor="white", linewidth=0.65, zorder=7)

    if not np.isfinite(ymax_raw) or not np.isfinite(ymin_raw):
        ymin_raw = 0.0
        ymax_raw = 1.0

    ymin = float(ymin_raw)
    ymax = float(ymax_raw)
    yrange = max(ymax - ymin, 1e-9)
    pad = max(0.10 * yrange, 0.02 * max(abs(ymax), abs(ymin), 1.0))
    ylo = ymin - 0.45 * pad
    yhi = ymax + pad
    ax.set_ylim(ylo, yhi)
    ax.set_xlim(x[0] - 0.25, x[-1] + 0.25)

    labels = [two_line_label(cond_id_to_pretty_name(c)) for c in conds]
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center")
    ax.set_ylabel(ylab, fontsize=15.5)
    ax.set_xlabel("")
    ax.grid(False)
    ax.tick_params(axis="x", labelsize=12.0, pad=8)
    ax.tick_params(axis="y", labelsize=13.5)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)

    y_sig = ymax + pad * 0.45
    for i, c in enumerate(conds):
        st = "ref" if c == "C2" else star_from_p(famA.get(c, {}).get("p_adj", np.nan))
        ax.text(x[i], y_sig, st, ha="center", va="center", fontsize=13.8, color=STAR_COLOR_A)

    fig.suptitle(title, fontsize=18.5, y=0.965)
    return fig


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input folder containing ARS images.")
    ap.add_argument("--glob", default="**/*.tif", help="Glob pattern inside input folder.")
    ap.add_argument(
        "--channel",
        choices=["red", "green", "inv_green", "gray"],
        default="red",
        help="Intensity channel for analysis.",
    )
    ap.add_argument("--norm_mode", choices=["minmax", "percentile"], default="minmax", help="Intensity normalization mode.")
    ap.add_argument("--norm_p_low", type=float, default=1.0, help="Low percentile for --norm_mode percentile.")
    ap.add_argument("--norm_p_high", type=float, default=99.0, help="High percentile for --norm_mode percentile.")
    ap.add_argument("--window", type=int, default=64, help="Sliding-window size in pixels for local variance.")
    ap.add_argument(
        "--ratio_thresh",
        type=float,
        default=None,
        help="If set, compute metrics on thresholded ARS mask where R/(G+B) >= ratio_thresh.",
    )
    ap.add_argument("--alpha", type=float, default=0.05, help="BH threshold for pairwise tests vs C2.")
    ap.add_argument("--eps", type=float, default=1e-8, help="Small constant for robust division.")
    ap.add_argument("--seed", type=int, default=0, help="Random seed for jitter in plots.")
    ap.add_argument("--out_prefix", type=str, default="ars_spatial", help="Output filename prefix.")
    ap.add_argument("--out_dir", type=Path, default=Path("ars_output_spatial"), help="Output directory.")
    return ap.parse_args()


def write_metric_stats(
    out_path: Path,
    metric_name: str,
    conds: List[str],
    cond_to_vals: Dict[str, np.ndarray],
    famA: Dict[str, Dict[str, float]],
    alpha: float,
    config_note: str,
) -> None:
    h, pkw = kw_all_conditions(cond_to_vals, conds)
    with out_path.open("w", encoding="utf-8") as f:
        f.write(f"{metric_name}\n")
        f.write(f"{config_note}\n")
        f.write("Test: Mann-Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini-Hochberg across C2-vs-target comparisons\n")
        f.write(f"Alpha: {alpha}\n")
        f.write(f"Kruskal-Wallis across all conditions: H={h:.4g}, p={pkw:.4g}\n\n")
        f.write("Pairwise comparisons vs C2\n")
        for c in [x for x in FAM_A_TARGETS if x in conds]:
            f.write(
                f"  {c} vs C2: p_raw={famA.get(c, {}).get('p_raw', np.nan):.4g}, "
                f"p_adj={famA.get(c, {}).get('p_adj', np.nan):.4g}\n"
            )


def main() -> None:
    apply_pub_rcparams()
    args = parse_args()

    if args.window < 2:
        raise ValueError("--window must be >= 2.")
    if args.eps <= 0:
        raise ValueError("--eps must be > 0.")
    if args.ratio_thresh is not None and args.ratio_thresh <= 0:
        raise ValueError("--ratio_thresh must be > 0 when provided.")
    if args.norm_mode == "percentile":
        if not (0.0 <= args.norm_p_low < args.norm_p_high <= 100.0):
            raise ValueError("--norm_p_low and --norm_p_high must satisfy 0 <= low < high <= 100.")

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise RuntimeError(f"No files found: {args.input} with glob '{args.glob}'")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in files:
        cond = detect_condition(p)
        if cond not in COND_CODE:
            continue
        rgb = load_rgb_float01(p)
        intensity = intensity_from_channel(rgb, args.channel)
        intensity_01 = normalize_intensity(
            intensity,
            mode=args.norm_mode,
            p_low=args.norm_p_low,
            p_high=args.norm_p_high,
            eps=args.eps,
        )

        if args.ratio_thresh is None:
            signal = intensity_01
            ratio_thresh_used = np.nan
            signal_mode = "intensity"
        else:
            ratio = ars_score_ratio(rgb, eps=args.eps)
            signal = (ratio >= float(args.ratio_thresh)).astype(np.float32)
            ratio_thresh_used = float(args.ratio_thresh)
            signal_mode = "ratio_mask"

        cv_intensity = compute_cv(signal, eps=args.eps)
        h_spatial = compute_spatial_heterogeneity(signal, window=args.window)
        rows.append(
            {
                "image": p.name,
                "cond": cond,
                "channel": args.channel,
                "norm_mode": args.norm_mode,
                "norm_p_low": args.norm_p_low,
                "norm_p_high": args.norm_p_high,
                "window_px": args.window,
                "signal_mode": signal_mode,
                "ratio_thresh": ratio_thresh_used,
                "cv_intensity": cv_intensity,
                "h_spatial": h_spatial,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No valid C1-C8 images processed.")

    conds = [c for c in CONDITIONS if (df["cond"] == c).any()]
    if "C2" not in conds:
        raise RuntimeError("Missing C2 (Osteo).")

    csv_path = args.out_dir / f"{args.out_prefix}_metrics.csv"
    df.to_csv(csv_path, index=False)
    print("Saved:", csv_path)

    summary = (
        df.groupby("cond", as_index=False)
        .agg(
            n_images=("image", "count"),
            cv_mean=("cv_intensity", "mean"),
            cv_sd=("cv_intensity", "std"),
            h_spatial_mean=("h_spatial", "mean"),
            h_spatial_sd=("h_spatial", "std"),
        )
        .sort_values(by="cond")
    )
    summary_path = args.out_dir / f"{args.out_prefix}_summary_by_condition.csv"
    summary.to_csv(summary_path, index=False)
    print("Saved:", summary_path)

    if args.ratio_thresh is None:
        config_note = (
            f"Signal=intensity ({args.channel}); normalization={args.norm_mode} "
            f"(p_low={args.norm_p_low:g}, p_high={args.norm_p_high:g}); window={args.window}px"
        )
        cv_metric_name = "Global heterogeneity (CV = sigma_I / mu_I)"
        h_metric_name = "Spatial heterogeneity (H_spatial = mean(local variance))"
        cv_ylab = r"Global heterogeneity (CV = $\sigma_I/\mu_I$)"
        h_ylab = r"Spatial heterogeneity ($H_{spatial}$)"
        cv_title = "ARS global heterogeneity / Day 28"
        h_title = "ARS spatial heterogeneity / Day 28"
    else:
        config_note = (
            f"Signal=thresholded ratio mask [R/(G+B)>={args.ratio_thresh:g}]; "
            f"window={args.window}px"
        )
        cv_metric_name = "Global heterogeneity on thresholded ratio mask"
        h_metric_name = "Spatial heterogeneity on thresholded ratio mask"
        cv_ylab = r"Mask heterogeneity (CV)"
        h_ylab = r"Mask spatial heterogeneity ($H_{spatial}$)"
        cv_title = "ARS mask heterogeneity / Day 28"
        h_title = "ARS mask spatial heterogeneity / Day 28"

    cond_to_cv = {c: df.loc[df["cond"] == c, "cv_intensity"].dropna().astype(float).values for c in conds}
    famA_cv = pairwise_vs_ref_bh(cond_to_cv, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    cv_stats_path = args.out_dir / f"{args.out_prefix}_cv_stats.txt"
    write_metric_stats(
        out_path=cv_stats_path,
        metric_name=cv_metric_name,
        conds=conds,
        cond_to_vals=cond_to_cv,
        famA=famA_cv,
        alpha=args.alpha,
        config_note=config_note,
    )
    print("Saved:", cv_stats_path)

    cond_to_h = {c: df.loc[df["cond"] == c, "h_spatial"].dropna().astype(float).values for c in conds}
    famA_h = pairwise_vs_ref_bh(cond_to_h, ref=FAM_A_REF, targets=[c for c in FAM_A_TARGETS if c in conds], alpha=args.alpha)
    h_stats_path = args.out_dir / f"{args.out_prefix}_h_spatial_stats.txt"
    write_metric_stats(
        out_path=h_stats_path,
        metric_name=h_metric_name,
        conds=conds,
        cond_to_vals=cond_to_h,
        famA=famA_h,
        alpha=args.alpha,
        config_note=config_note,
    )
    print("Saved:", h_stats_path)

    fig_cv = plot_metric_by_condition(
        df=df,
        value_col="cv_intensity",
        conds=conds,
        famA=famA_cv,
        ylab=cv_ylab,
        title=cv_title,
        seed=args.seed,
    )
    cv_pdf = args.out_dir / f"{args.out_prefix}_cv_fig2.pdf"
    cv_png = args.out_dir / f"{args.out_prefix}_cv_fig2.png"
    fig_cv.savefig(cv_pdf)
    fig_cv.savefig(cv_png)
    plt.close(fig_cv)
    print("Saved:", cv_pdf)
    print("Saved:", cv_png)

    fig_h = plot_metric_by_condition(
        df=df,
        value_col="h_spatial",
        conds=conds,
        famA=famA_h,
        ylab=h_ylab,
        title=h_title,
        seed=args.seed + 1,
    )
    h_pdf = args.out_dir / f"{args.out_prefix}_h_spatial_fig2.pdf"
    h_png = args.out_dir / f"{args.out_prefix}_h_spatial_fig2.png"
    fig_h.savefig(h_pdf)
    fig_h.savefig(h_png)
    plt.close(fig_h)
    print("Saved:", h_pdf)
    print("Saved:", h_png)

    print("Done. Output saved to:", args.out_dir)


if __name__ == "__main__":
    main()
