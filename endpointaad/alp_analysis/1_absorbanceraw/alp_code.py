import argparse
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import pipeline
from pipeline import parse_metadata, read_values_xlsx, make_scatter_data


# -------------------- CONDITIONS & LABELS --------------------
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


def save_figures(fig, pdf_path: Optional[Path], png_path: Optional[Path]):
    """Save figure to both PDF and PNG when requested."""
    if pdf_path:
        fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {pdf_path}")
    if png_path:
        fig.savefig(png_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {png_path}")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--out")
    p.add_argument(
        "--png",
        help="Optional PNG path; defaults to the same stem as --out with .png.",
    )
    return p.parse_args()


def main():
    args = parse_args()

    if args.pipeline:
        print(pipeline.__file__)
        return

    wells, cond_order = parse_metadata(args.meta)
    cond_vals = read_values_xlsx(args.data, wells)
    xs, ys = make_scatter_data(cond_vals, cond_order)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(xs, ys)
    ax.set_xticks(range(1, len(cond_order) + 1))
    labels = [two_line_label(COND_CODE.get(cond, cond)) for cond in cond_order]
    ax.set_xticklabels(labels)
    ax.set_ylabel("Absorbance (405 nm)")
    fig.tight_layout()

    if args.out or args.png:
        pdf_path = Path(args.out) if args.out else None
        png_path = (
            Path(args.png)
            if args.png
            else pdf_path.with_suffix(".png")
            if pdf_path
            else None
        )
        save_figures(fig, pdf_path, png_path)
    plt.show()


if __name__ == "__main__":
    main()



### Run example:
### python alp_code.py --meta alpmetada.txt --data alp_raw_data.xlsx --out alpplot.pdf
