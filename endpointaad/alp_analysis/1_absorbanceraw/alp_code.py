import argparse
import matplotlib.pyplot as plt
import pipeline
from pipeline import parse_metadata, read_values_xlsx, make_scatter_data


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--meta", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--pipeline", action="store_true")
    p.add_argument("--out")
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
    ax.set_xticklabels(cond_order)
    ax.set_ylabel("Absorbance (405 nm)")
    fig.tight_layout()

    plt.show()

    if args.out:
        fig.savefig(args.out, dpi=300, bbox_inches="tight")
        print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()



### Run example:
### python alp_code.py --meta alpmetada.txt --data alp_raw_data.xlsx --out alpplot.pdf