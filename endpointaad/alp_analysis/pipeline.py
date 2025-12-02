import pandas as pd
import matplotlib.pyplot as plt


def read_xlsx_columns(path):
    df = pd.read_excel(path)
    df.columns = df.columns.astype(str)
    return df


def visualize(df, x_col, y_col=None):
    if y_col is None:
        df[x_col].plot(kind="hist")
    else:
        plt.scatter(df[x_col], df[y_col])
    plt.show()
