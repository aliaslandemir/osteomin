#!/usr/bin/env python3
"""
Publication-ready ALP (technical replicates) — RAW / ABSOLUTE mU/mL
MATCHES collagen-style plot + stats, but WITHOUT fold normalization.

Keeps the SAME IO as your old ALP script:
- parse_metadata(metadata.txt)  -> well_to_cond, cond_order
- read_values_xlsx(data.xlsx, well_to_cond) -> cond_to_values (raw absorbance)

Pipeline:
1) blank-correct + convert to mU/mL (clamped to >=0)
2) Stats on RAW ALP_mU_mL (absolute):
   - Global Kruskal–Wallis across all present conditions (reported)
   - Family A (lower black stars): C1 vs C2–C8, Mann–Whitney U (two-sided) + BH
   - Family B (upper red stars):   C2 vs C3–C8, Mann–Whitney U (two-sided) + BH (excluding C1)
3) Plot:
   - clean violins + colored points + black mean±SEM
   - bracket-less stars in two rows (black vs C1, red vs C2)
   - separators (C1–C2 and C6–C7)
   - Δmean row (mean difference vs C1) at bottom (IN mU/mL)
   - descriptive stats header

Outputs:
- CSV: per-well absorbance + ALP_mU_mL
- TXT: stats report
- Figure: PDF + PNG (RAW)

Run:
python alp_publishable_dualref_RAW_pro.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output
"""

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from plot_style import (
    CONDITIONS,
    FAM_A_REF,
    FAM_A_TARGETS,
    FAM_B_REF,
    FAM_B_TARGETS,
    kw_all_conditions,
    pairwise_vs_ref_BH,
    plot_violin,
)

from pipeline import parse_metadata, read_values_xlsx


# -------------------- CLI --------------------

def parse_args():
    p = argparse.ArgumentParser(description="ALP RAW mU/mL stats + pro violin plot (MWU+BH dual-family).")
    p.add_argument("--metadata", required=True, help="Metadata text file (same format as before).")
    p.add_argument("--data", required=True, help="Raw ALP data xlsx (same as before).")
    p.add_argument("--blank", type=float, default=0.066, help="Blank absorbance (default: 0.066).")
    p.add_argument("--conversion", type=float, required=True,
                   help="Conversion factor from net absorbance to mU/mL.")
    p.add_argument("--alpha", type=float, default=0.05, help="BH threshold (default 0.05).")
    p.add_argument("--out_prefix", type=str, default="alp_raw", help="Output filename prefix.")
    p.add_argument("--out_dir", type=Path, default=Path("alp_output"), help="Output directory.")
    p.add_argument("--seed", type=int, default=0, help="Random seed for jitter.")
    return p.parse_args()


def main():
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Read metadata and raw values (UNCHANGED)
    well_to_cond, cond_order = parse_metadata(args.metadata)
    cond_to_values = read_values_xlsx(args.data, well_to_cond)  # raw absorbance

    # 2) Convert to per-well mU/mL
    rows = []
    for cond, vals in cond_to_values.items():
        if cond not in CONDITIONS:
            continue
        for i, v in enumerate(vals):
            try:
                v = float(v)
            except Exception:
                continue
            v_net = v - float(args.blank)
            mU = max(v_net * float(args.conversion), 0.0)
            rows.append({
                "condition": cond,
                "well_index": i + 1,
                "abs_raw": v,
                "abs_net": v_net,
                "ALP_mU_mL": mU,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No valid ALP values found for conditions C1–C8.")
        return

    # 3) Restrict to conditions present in metadata order
    conds = [c for c in cond_order if (c in CONDITIONS and (df["condition"] == c).any())]
    if not conds:
        print("No conditions C1–C8 found in metadata order; exiting.")
        return

    if "C1" not in conds:
        raise ValueError("Missing C1 (Basal).")
    if "C2" not in conds:
        raise ValueError("Missing C2 (Osteo).")

    # Save CSV (per-well, RAW)
    csv_path = args.out_dir / f"{args.out_prefix}_alp_metrics_raw.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved CSV: {csv_path}")

    # Convenience: map condition->values for RAW column
    cond_to_vals: Dict[str, np.ndarray] = {}
    for c in conds:
        v = df.loc[df["condition"] == c, "ALP_mU_mL"].dropna().astype(float).values
        if v.size:
            cond_to_vals[c] = v

    # 4) Global KW + pairwise MWU+BH families
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

    # 5) Stats report
    rpt = args.out_dir / f"{args.out_prefix}_ALP_mU_mL_stats.txt"
    with open(rpt, "w", encoding="utf-8") as f:
        f.write("ALP activity — RAW mU/mL (30-min pNPP assay)\n")
        f.write("Metric column: ALP_mU_mL\n")
        f.write("Per-well values treated as independent technical replicates.\n")
        f.write("Test: Mann–Whitney U (two-sided)\n")
        f.write("Multiple testing: Benjamini–Hochberg within each family\n")
        f.write(f"Alpha: {args.alpha}\n")
        f.write(f"Blank absorbance: {args.blank}\n")
        f.write(f"Conversion factor: {args.conversion} (net abs → mU/mL)\n\n")
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

    print(f"Saved stats: {rpt}")

    # 6) Plot (RAW mU/mL)
    fig = plot_violin(
        df=df,
        value_col="ALP_mU_mL",
        conds=conds,
        title="ALP activity (30-min pNPP assay)",
        ylab=r"ALP activity (mU·mL$^{-1}$)",
        famA=famA,
        famB=famB,
        seed=args.seed,
    )

    pdf = args.out_dir / f"{args.out_prefix}_ALP_mU_mL.pdf"
    png = args.out_dir / f"{args.out_prefix}_ALP_mU_mL.png"
    fig.savefig(pdf, dpi=300, bbox_inches="tight")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {pdf}")
    print(f"Saved: {png}")


if __name__ == "__main__":
    main()


"""
Run:

python alp_stat_code.py \
  --metadata alpmetadata.txt \
  --data alp_raw_data.xlsx \
  --blank 0.066 \
  --conversion 11.01 \
  --alpha 0.05 \
  --out_prefix alp_raw \
  --out_dir alp_output
  
"""
