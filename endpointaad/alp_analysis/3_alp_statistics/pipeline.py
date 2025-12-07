import pandas as pd


def parse_metadata(meta_path):
    """
    Read lines like `b1 = C1` and return:
      - well_to_cond: {("b", 1): "C1", ...}
      - cond_order:   ["C1", "C2", ...] in first-seen order
    """
    well_to_cond = {}
    cond_order = []

    with open(meta_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue

            left, right = map(str.strip, line.split("=", 1))
            well = left.lower()  # e.g. "b1"
            row = well[0]  # "b"
            col = int(well[1:])  # 1..12

            cond = right.split()[0]  # "C1", "PNPP_plus_lysate", ...

            well_to_cond[(row, col)] = cond
            if cond not in cond_order:
                cond_order.append(cond)

    return well_to_cond, cond_order


def read_values_xlsx(xlsx_path, well_to_cond):
    """
    Excel layout:

      col0: "Sample"  -> A,B,C,D,...
      col1-12:        -> 1..12

    Returns: {"C1": [v1,v2,v3], ...}
    """
    df = pd.read_excel(xlsx_path).set_index("Sample")

    cond_to_values = {}
    for (row, col), cond in well_to_cond.items():
        val = float(df.loc[row.upper(), col])  # row "B"/"C"/"D", column 1..12
        cond_to_values.setdefault(cond, []).append(val)

    return cond_to_values


def make_scatter_data(cond_to_values, cond_order):
    xs, ys = [], []
    for i, cond in enumerate(cond_order, start=1):
        vals = cond_to_values.get(cond, [])
        xs.extend([i] * len(vals))
        ys.extend(vals)
    return xs, ys
