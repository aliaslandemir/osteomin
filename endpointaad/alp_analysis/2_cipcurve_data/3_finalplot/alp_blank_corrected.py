#!/usr/bin/env python3


import argparse
import matplotlib.pyplot as plt

from pipeline import parse_metadata, read_values_xlsx


def parse_args():
    p = argparse.ArgumentParser(
        description="Blank-correct ALP A405 data and convert to mU/mL ( )."
    )
    p.add_argument(
        "--pipeline",
        default="pipeline.py",
        help="Path to pipeline.py (not used directly, only for bookkeeping).",
    )
    p.add_argument(
        "--metadata",
        required=True,
        help="Metadata file (well -> condition), e.g. alpmetada.txt",
    )
    p.add_argument(
        "--data",
        required=True,
        help="Excel file with raw A405 values, e.g. alp_raw_data.xlsx",
    )
    p.add_argument(
        "--blank",
        type=float,
        default=0.066,
        help="Blank A405 value to subtract (default: 0.066)",
    )
    p.add_argument(
        "--conversion",
        type=float,
        required=True,
        help="Conversion factor (mU/mL per unit net A405), e.g. 11.01",
    )
    p.add_argument(
        "--out",
        help="Optional output PNG/PDF for the scatter plot",
    )
    return p.parse_args()


def main():
    args = parse_args()

    # --- read plate + map wells to conditions
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)

    # --- blank-correct and convert to mU/mL
    cond_to_mU = {}
    for cond, vals in cond_to_values.items():
        converted = []
        for v in vals:
            a_net = v - args.blank
            mU_mL = max(a_net * args.conversion, 0.0)  # clip negatives to 0
            converted.append(mU_mL)
        cond_to_mU[cond] = converted

    # ---   scatter only for C1..C8
    conds_c = [c for c in cond_order if c.startswith("C")]
    xs, ys = [], []
    for i, cond in enumerate(conds_c, start=1):
        for v in cond_to_mU.get(cond, []):
            xs.append(i)
            ys.append(v)

    if xs and ys:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.scatter(xs, ys)
        ax.set_xticks(range(1, len(conds_c) + 1))
        ax.set_xticklabels(conds_c)
        ax.set_ylabel("ALP activity (mU/mL)")
        ax.set_title("ALP activity per condition (blank-corrected)")
        fig.tight_layout()

        if args.out:
            fig.savefig(args.out, dpi=300, bbox_inches="tight")
            print(f"Saved: {args.out}")
        else:
            plt.show()


if __name__ == "__main__":
    main()


# To run:


"""

python alp_blank_corrected.py \
    --pipeline pipeline.py \
    --metadata alpmetadata.txt \
    --data alp_raw_data.xlsx \
    --blank 0.066 \
    --conversion 11.01 \
    --out alp_activity.pdf
    
"""
