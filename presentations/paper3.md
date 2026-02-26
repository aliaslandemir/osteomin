# Endpoint Perturbations Quantify but Do Not Resolve Mechanistic Identifiability in Osteogenic Differentiation

## Hybrid Cellular Automaton Tested by Endpoint Perturbations

---

## 1. Introduction

- **Problem:** Osteogenic assays provide population-averaged endpoints (ALP, collagen, ARS).
- **Limitation:** Endpoints compress spatial + temporal structure.
- **Gap:** Mechanistic identifiability under pharmacological perturbation is unclear.
- **Aim:** Test a literature-constrained hybrid CA against structured Dex × MK-4 perturbations to quantify identifiability limits.

---

## 2. Experimental System

- **Cells:** MC3T3-E1 (2D differentiation).
- **Design:** Factorial Dex × MK-4 (8 conditions) + AA + β-GP baseline.
- **Endpoints:**
  - ALP (d10)
  - Collagen I (d14)
  - ARS (d28)
- **Statistics:** Main + interaction effects; corrected multiple comparisons.

### Figure 1 — Experimental Grounding
- **Fig 1A:** Confocal IF (morphology + markers)
- **Fig 1B:** ALP, Collagen, ARS across 8 conditions
- **Table I:** Drug concentrations

---

## 3. Hybrid CA Model

### 3.1 Architecture
- Agents: pre-OB → OB → death
- Fields: cumulative ALP, collagen, mineral
- Readouts: global field sums mapped to assays

### 3.2 Rules
- Motility (pre-OB)
- Proliferation (timer)
- Stochastic commitment (delay-gated)
  - Osteogenic induction baseline: day 3 (`t_osteo = 72 h`)
  - Commitment gate: `t >= t_osteo + t_d`, with default `t_d = 48 h`
  - Time-varying hazard: `k(t) = k0 + gk * (t - t_osteo - t_d)`, defaults `k0 = 0.06 day^-1`, `gk = 0`
  - Hourly commitment probability: `p_hour(t) = 1 - exp(-k(t)/24)`
- Apoptosis
- OBs deposit matrix/mineral

### 3.3 Field Logic
- Local accumulation
- No diffusion/degradation
- Delayed mineral gating

### 3.4 Parameters
- Fixed: baseline kinetics (literature-constrained)
- Fitted: Dex, MK-4 coefficients (per readout)
- Analysis: sensitivity + profile likelihood

### Figure 2 — Model Architecture
- Grid
- Cell states
- Field layers
- Update loop

### Table II — Parameters

---

## 4. Calibration

- Data: 8-condition endpoint matrix
- Sequential fitting: baseline → commitment → mineral gate → drugs
- Metric: weighted SSE / RMSE

### Figure 3 — Calibration Fit
- Model vs experiment
- Residuals
- Goodness-of-fit
- Practical identifiability

---

## 5. Validation

- Cross-condition prediction
- Rank-order accuracy
- Predictive error

---

## 6. Identifiability Limits

- Baseline ordering reproduced (ALP → collagen → mineral)
- Perturbation responses captured in endpoint space
- Non-uniqueness: distinct trajectories → identical endpoints
- Weak constraint on spatial/temporal structure

### Figure 4 — Non-Uniqueness
- Two parameter sets
- Same endpoints
- Different spatial dynamics

### Figure 5 — Sensitivity / Identifiability
- Parameter ranking
- Correlation matrix
- Non-identifiable directions

---

## 7. Discussion

- Endpoint perturbations constrain totals, not mechanisms.
- Dex/MK-4 effects partially underdetermined.
- Model functions as a constraint-mapping tool.
- Time-resolved + spatial data required to collapse degeneracy.

---

## 8. Conclusion

- Hybrid CA links cell rules to assay-level outputs.
- Structured perturbations test mechanistic identifiability.
- Endpoint equivalence ≠ mechanistic equivalence.

---

## Core Logic

Biology → Hybrid CA → Calibration → Validation → Identifiability Limits
