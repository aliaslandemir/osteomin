#!/usr/bin/env python3


import argparse
import pandas as pd
import matplotlib.pyplot as plt


def excel_cell_to_rowcol(cell):
    """
    Convert Excel cell like 'C5' → (row='C', col=5)
    """
    row = cell[0].upper()
    col = int(cell[1:])
    return row, col


def parse_range(rng):
    """
    'C5:C8' → list ['C5','C6','C7','C8']
    """
    start, end = rng.split(":")
    r1, c1 = excel_cell_to_rowcol(start)
    r2, c2 = excel_cell_to_rowcol(end)

    if r1 != r2:
        raise ValueError("Range must be within the same row, e.g. C5:C8")

    return [f"{r1}{c}" for c in range(c1, c2 + 1)]


def read_val(df, cell):
    row, col = excel_cell_to_rowcol(cell)
    return float(df.loc[row, col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("xlsx")
    ap.add_argument("range", help="Excel-like range, e.g. C5:C8")
    ap.add_argument("-o", "--out", default="cip.png")
    args = ap.parse_args()

    # load excel
    df = pd.read_excel(args.xlsx).set_index("Sample")

    # convert range → list of wells
    wells = parse_range(args.range)

    # read values
    vals = [read_val(df, w) for w in wells]

    # print to console
    for w, v in zip(wells, vals):
        print(f"{w}: {v:.4f}")

    # plot
    plt.figure(figsize=(4, 3))
    plt.plot(range(len(vals)), vals, "o-")
    plt.xticks(range(len(vals)), wells)
    plt.ylabel("A405")
    plt.title("CIP Absorbance")
    plt.tight_layout()
    plt.savefig(args.out, dpi=300)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()



"""
Plot absorbance values from a Excel range like 'G3:G6' (G6 is blank).

Usage:
    python cip_raw_curve.py alp_raw_data.xlsx "G3:G6" -o cip.png
"""