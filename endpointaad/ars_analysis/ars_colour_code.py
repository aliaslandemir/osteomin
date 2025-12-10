#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re
from pathlib import Path
import numpy as np
import pandas as pd
from skimage import io
import matplotlib.pyplot as plt

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


def two_line_label(name: str) -> str:
    """Two-line labels for conditions."""
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


def load_rgb_float(p: Path):
    """Load image as float32 RGB in [0, 1]."""
    img = io.imread(str(p))
    if img.ndim == 2:
        img = np.stack([img] * 3, -1)
    if img.ndim == 3 and img.shape[0] in (3, 4) and img.shape[-1] not in (3, 4):
        img = np.moveaxis(img, 0, -1)
    if img.shape[-1] > 3:
        img = img[..., :3]
    img = img.astype(np.float32)
    if img.max() > 1.01:
        img /= 255.0
    return img


def detect_condition(p: Path):
    """Detect condition from parent folder (C1..C8) or filename."""
    parent = p.parent.name.lower()
    if parent in [f"c{i}" for i in range(1, 9)]:
        return parent.upper()
    m = COND_RE.match(p.name)
    return m.group(1).upper() if m else "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="ars_channels")
    ap.add_argument("--glob", default="**/*.tif")
    args = ap.parse_args()

    files = sorted(Path(args.input).glob(args.glob))
    if not files:
        raise SystemExit(
            f"No files found under {args.input!r} with pattern {args.glob!r}"
        )

    rows = []

    # collect per-condition channel values
    per_cond = {c: {"R": [], "G": [], "B": []} for c in CONDITIONS}

    for p in files:
        cond = detect_condition(p)
        if cond not in CONDITIONS:
            continue
        rgb = load_rgb_float(p)
        # NOTE: 0=Red, 1=Green, 2=Blue
        mean_r = rgb[..., 0].mean()
        mean_g = rgb[..., 1].mean()
        mean_b = rgb[..., 2].mean()

        rows.append(
            {
                "image": p.name,
                "path": str(p),
                "cond": cond,
                "cond_label": COND_CODE[cond],
                "mean_R": mean_r,
                "mean_G": mean_g,
                "mean_B": mean_b,
            }
        )

        per_cond[cond]["R"].append(mean_r)
        per_cond[cond]["G"].append(mean_g)
        per_cond[cond]["B"].append(mean_b)

    # ------------------------------------------------------------------
    # 1) Save per-image channel means
    # ------------------------------------------------------------------
    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}_per_image_channels.csv", index=False)

    # ------------------------------------------------------------------
    # 2) Save per-condition channel means (mean, std, n)
    # ------------------------------------------------------------------
    cond_rows = []
    for cond in CONDITIONS:
        for ch in ["R", "G", "B"]:
            vals = np.array(per_cond[cond][ch], dtype=float)
            if vals.size == 0:
                continue
            cond_rows.append(
                {
                    "cond": cond,
                    "cond_label": COND_CODE[cond],
                    "channel": ch,
                    "n": vals.size,
                    "mean": float(vals.mean()),
                    "std": float(vals.std(ddof=1)) if vals.size > 1 else np.nan,
                }
            )
    dfc = pd.DataFrame(cond_rows)
    dfc.to_csv(f"{args.out}_per_condition_channels.csv", index=False)

    # ------------------------------------------------------------------
    # 3) Basal (C1) vs C2 figure: show all 9 points per channel + mean
    # ------------------------------------------------------------------
    def get_vals(cond, ch):
        return np.array(per_cond[cond][ch], dtype=float)

    c1, c2 = "C1", "C2"
    if not per_cond[c1]["R"] or not per_cond[c2]["R"]:
        print("Warning: C1 or C2 has no data, skipping basal vs C2 plot.")
        return

    channels = [("R", "Red channel"), ("G", "Green channel"), ("B", "Blue channel")]

    # 3 rows, 1 column for a more paper-ish layout
    fig, axes = plt.subplots(3, 1, figsize=(4, 8), sharex=True)

    for ax, (ch, ch_label) in zip(axes, channels):
        vals_c1 = get_vals(c1, ch)
        vals_c2 = get_vals(c2, ch)

        # base x positions for the two conditions
        x_base = np.array([0.0, 1.0])
        # small jitter for individual points
        jitter_c1 = x_base[0] + (np.random.rand(len(vals_c1)) - 0.5) * 0.1
        jitter_c2 = x_base[1] + (np.random.rand(len(vals_c2)) - 0.5) * 0.1

        # scatter individual points
        s1 = ax.scatter(jitter_c1, vals_c1, alpha=0.7, label="Basal")
        s2 = ax.scatter(jitter_c2, vals_c2, alpha=0.7, label="AA+$\\beta$-GP")

        # mean markers (no jitter)
        mean_c1 = vals_c1.mean()
        mean_c2 = vals_c2.mean()
        m = ax.scatter(
            x_base,
            [mean_c1, mean_c2],
            marker="D",
            s=80,
            edgecolor="k",
            linewidth=0.8,
            label="Mean",
        )

        ax.set_ylabel("Mean intensity")
        ax.set_title(ch_label)
        ax.grid(True, linestyle=":", alpha=0.3)

    # shared x-axis labels only on bottom
    axes[-1].set_xticks([0, 1])
    axes[-1].set_xticklabels(
        [two_line_label(COND_CODE[c1]), two_line_label(COND_CODE[c2])]
    )

    # put a single legend on the bottom axis
    axes[-1].legend(loc="upper right", fontsize=8, frameon=False)

    fig.suptitle("Basal (C1) vs C2 — RGB mean channel intensities", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{args.out}_C1_vs_C2_channels.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()

# Example usage:
#   python ars_channels.py --input ../ars --out ars_channels
