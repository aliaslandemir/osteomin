# alp_code.py
import argparse
from pipeline import read_xlsx_columns, visualize


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)

    return parser.parse_args()


def main():
    args = parse_args()
    df = read_xlsx_columns(args.input)
    visualize(df, args.xcol, args.ycol)

    print(df)


if __name__ == "__main__":
    main()
