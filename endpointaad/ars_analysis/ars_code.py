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
    parent = p.parent.name.lower()
    if parent in [f"c{i}" for i in range(1, 9)]:
        return parent.upper()
    m = COND_RE.match(p.name)
    return m.group(1).upper() if m else "UNKNOWN"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--input", required=True)
    ap.add_argument("-o", "--out", default="ars_rg")
    ap.add_argument("--glob", default="**/*.tif")
    args = ap.parse_args()

    files = sorted(Path(args.input).glob(args.glob))
    cond_to_rg = {f"C{i}": [] for i in range(1, 9)}
    rows = []

    for p in files:
        cond = detect_condition(p)
        if cond not in cond_to_rg:
            continue
        rgb = load_rgb_float(p)
        rg = (rgb[..., 0] - rgb[..., 1]).mean()
        cond_to_rg[cond].append(rg)
        rows.append({"image": p.name, "cond": cond, "RminusG": rg})

    df = pd.DataFrame(rows)
    df.to_csv(f"{args.out}_rg_values.csv", index=False)

    conds = [f"C{i}" for i in range(1, 9) if len(cond_to_rg[f"C{i}"]) > 0]
    xs, ys = [], []
    for i, c in enumerate(conds, 1):
        for v in cond_to_rg[c]:
            xs.append(i)
            ys.append(v)

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.scatter(xs, ys)
    ax.set_xticks(range(1, len(conds) + 1))
    ax.set_xticklabels([two_line_label(COND_CODE[c]) for c in conds])
    ax.set_ylabel("Mean (R - G)")
    ax.set_title("Mean (R - G) per condition")
    fig.tight_layout()
    fig.savefig(f"{args.out}_rg_scatter.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()


# To run:

#  python ars_code.py --input ../ars --out ars_norm
