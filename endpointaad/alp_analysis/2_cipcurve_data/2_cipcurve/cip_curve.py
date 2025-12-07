from pathlib import Path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------
# USER INPUT
# -------------------------------
excel_file = "alp_raw_data.xlsx"

# CIP wells (last one = blank)
# CIP wells (G3=0.00020, G4=0.00010, G5=0.00005, G6=blank)

cip_wells = ["G3", "G4", "G5", "G6"]

# Activities (U/well)
cip_U = np.array([0.00020, 0.00010, 0.00005, 0.0])

# -------------------------------
# LOAD EXCEL
# -------------------------------
df = pd.read_excel(excel_file, header=None)
df = df.set_index(df.columns[0])  # first column = row IDs (A,B,C,...)


def read(cell):
    row = cell[0].upper()
    col = int(cell[1:])
    return float(df.loc[row, col])


A_raw = np.array([read(w) for w in cip_wells], dtype=float)

A_blank = A_raw[-1]
A_net = A_raw - A_blank

# -------------------------------
# FIT (origin-constrained)
# -------------------------------
mask = cip_U > 0
U_fit = cip_U[mask]
A_fit = A_net[mask]

m = (U_fit @ A_fit) / (U_fit @ U_fit)
r2 = 1 - np.sum((A_fit - m * U_fit) ** 2) / np.sum((A_fit - A_fit.mean()) ** 2)


# Convert A_net → mU/mL:
# U/well = A_net / m
# U/mL  = U/well / 0.025   (25 µL lysate per well)
# mU/mL = U/mL * 1000
# → factor = 40000 / m


# conversion factor
factor_mU_per_mL = 40000.0 / m

# -------------------------------
# PLOT
# -------------------------------
plt.figure(figsize=(4.5, 3.5))

# scatter
plt.scatter(U_fit, A_fit, s=60, edgecolor="black", zorder=3)

# fit line
U_line = np.linspace(0, U_fit.max() * 1.1, 200)
plt.plot(U_line, m * U_line, "r", lw=2, zorder=2)

# textbox (top-left)
txt = (
    f"Slope m = {m:.2e}\n"
    f"R² = {r2:.4f}\n"
    f"Blank = {A_blank:.3f}\n"
    f"mU/mL per A = {factor_mU_per_mL:.2f}"
)
plt.text(
    0.02 * U_fit.max(),
    0.95 * A_fit.max(),
    txt,
    ha="left",
    va="top",
    fontsize=8,
    bbox=dict(facecolor="white", edgecolor="gray", alpha=0.9),
)

plt.xlabel("CIP (U/well)")
plt.ylabel("Net A405")
plt.title("CIP Calibration Curve")

plt.tight_layout()
fig = plt.gcf()
base = Path("cip_calibration_curve")
pdf_out = base.with_suffix(".pdf")
png_out = base.with_suffix(".png")
fig.savefig(pdf_out, dpi=300)
fig.savefig(png_out, dpi=300)
print(f"Saved {pdf_out}")
print(f"Saved {png_out}")
plt.show()


# To run:

# python cip_curve.py
