
---

# OSTEOGENIC PROGRESSION MODEL (OsteoMin)

---

# 1. Fixed Parameters (Not Fitted)

## 1.1 Protocol Inputs

* **t_osteo** : osteogenic induction time (protocol-defined)
* **Δt_med**  : medium refresh interval (protocol-defined)
* **Ethanol carrier**: 0.10 % (v/v) in all conditions

---

## 1.2 Ascorbic Acid (AA) Dynamics

* **Initial AA after refresh (A₀)** = 0.284 mM (50 µg/mL)
* **Half-life (τ_A)** ≈ 6 h
* **Decay constant (k_A)** = ln(2) / τ_A ≈ 0.11 h⁻¹

AA follows first-order decay between medium refresh events.

---

## 1.3 AA Gating Function (Fixed)

AA modulates osteogenic production amplitudes via:

[
g_A(A) = \frac{A_{\mathrm{eff}}}{K_A^C + A_{\mathrm{eff}}}
]

* **K_A^C** : fixed half-saturation constant
* No separate κ_A terms are included (avoids redundancy).

---

## 1.4 Mineralization Time Gate (Fixed)

Mineral deposition is gated by differentiation time:

[
h(t_{\mathrm{eff}}) =
\frac{t_{\mathrm{eff}}^{n}}{\tau_H^{n} + t_{\mathrm{eff}}^{n}}
]

* **τ_H** = 16 days
* **n** = 4

Both fixed from independent ARS time-course fitting.

---

## 1.5 Collagen Maturation Gate (Fixed)

Newly deposited collagen is not assumed to become mineralizable
immediately. Instead, each lattice site carries a matured-collagen state
that relaxes toward total collagen with a first-order maturation timescale:

[
\frac{d C^{\mathrm{mat}}}{dt}
=
\frac{C - C^{\mathrm{mat}}}{\tau_{\mathrm{col}}}
]

Mineralization is then modulated by a local matrix-readiness gate:

[
g_{\mathrm{col}}(C^{\mathrm{mat}})
=
\frac{C^{\mathrm{mat}}}{K_{\mathrm{col}} + C^{\mathrm{mat}}}
]

* **τ_col** : collagen maturation timescale
* **K_col** : matured-collagen half-saturation constant for mineralization

---

## 1.6 Fixed Production Amplitudes

The following amplitudes are fixed from literature calibration:

* r_basal_P
* r_osteo_P
* r_basal_C
* r_osteo_C
* r_osteo_H

These define the constrained osteogenic baseline.

---

# 2. Free Parameters (6 Total)

Only drug modifiers are fitted.

## 2.1 ALP Modifiers (2)

* κ_DP : Dex modifier on osteogenic ALP
* κ_KP : MK-4 modifier on osteogenic ALP

## 2.2 Collagen Modifiers (2)

* κ_DC : Dex modifier on osteogenic collagen
* κ_KC : MK-4 modifier on osteogenic collagen

## 2.3 Mineral / ARS Modifiers (2)

* κ_DH : Dex modifier on osteogenic mineral
* κ_KH : MK-4 modifier on osteogenic mineral

---

### Free Parameter Vector

[
\theta_{\text{free}} =
[
\kappa_{DP},;
\kappa_{KP},;
\kappa_{DC},;
\kappa_{KC},;
\kappa_{DH},;
\kappa_{KH}
]
]

**Total free parameters = 6**

---

# 3. Experimental Conditions (C1–C8)

### C1: Basal

AA=0 | β-GP=0 | Dex=0 | MK-4=0

### C2: AA + β-GP

AA=0.284 mM | β-GP=10 mM | Dex=0 | MK-4=0

### C3: AA + β-GP + D_L + K_L

AA=0.284 mM | β-GP=10 mM | Dex=10 nM | MK-4=0.1 µM

### C4: AA + β-GP + D_L + K_H

AA=0.284 mM | β-GP=10 mM | Dex=10 nM | MK-4=10 µM

### C5: AA + β-GP + D_H + K_L

AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=0.1 µM

### C6: AA + β-GP + D_H + K_H

AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=10 µM

### C7: AA + β-GP + D_H

AA=0.284 mM | β-GP=10 mM | Dex=100 nM | MK-4=0

### C8: AA + β-GP + K_H

AA=0.284 mM | β-GP=10 mM | Dex=0 | MK-4=10 µM

---

# 4. Replicate Structure

Per condition:

* **ALP**: 3 replicates
* **Collagen**: 5 replicates
* **ARS**: 9 replicates

Total conditions: 8
