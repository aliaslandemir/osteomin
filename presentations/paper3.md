

# Endpoint Perturbations Quantify but Do Not Resolve Mechanistic Identifiability in Osteogenic Differentiation

---

## *Hybrid Cellular Automaton Tested by Endpoint Perturbations*

---

## 1. Introduction

* **Biological problem:** Osteogenic differentiation is assessed using endpoint-only assays.
* **Limitation:** Endpoints obscure spatial dynamics and timing.
* **Gap:** It is unclear whether pharmacological perturbations (Dex, MK-4) are mechanistically identifiable from endpoints.
* **Aim:** Test a **literature-derived CA model** against endpoint perturbations to evaluate identifiability and predictive power.

---

## 2. Experimental Methods

* **System:** MC3T3-E1 osteogenic differentiation (2D).
* **Perturbations:** Factorial Dex × MK-4 (8 conditions).
* **Measurements:** ALP (d10), collagen I (d14), ARS (d28).
* **Analysis:** Normalized endpoints; interaction effects (Dex×MK-4).

---

## 3. Computational Model

### 3.1 Model Overview

* Hybrid CA from literature:

  * **Agents:** cells.
  * **Fields:** cumulative ALP, collagen, mineral (Not real concentrations! Representative units).
* Model outputs mapped to assay readouts.

### 3.2 Cellular Rules

* Movement, proliferation, stochastic commitment, apoptosis.
* Fixed update order.

### 3.3 Microenvironment Dynamics

* Local accumulation (no diffusion).
* Designed to represent assay recovery, not molecular transport.

### 3.4 Parameterisation

* **Fixed:** basal and osteogenic kinetics from literature.
* **Fitted:** Dex and MK-4 perturbation coefficients.
* Identifiability assessed via sensitivity / profiles.

---

## 4. Model Calibration

* Fitted to endpoint data across 8 perturbation conditions.
* Sequential fitting strategy.
* Metrics: RMSE, weighted SSE.

---

## 5. Model Validation

* Withheld perturbation condition or cross-validation.
* Test: prediction of unseen Dex/MK-4 responses.
* Metrics: predictive error, rank-order accuracy.

---

## 6. Results & Mechanistic Insights

* **Baseline:** recovers expected differentiation sequence.
* **Perturbations:** nonlinear Dex/MK-4 effects captured.
* **Non-uniqueness:** identical endpoints arise from distinct spatial trajectories.
* **Sensitivity:** endpoints constrain totals but not spatial dynamics.

---

## 7. Discussion

* Endpoint perturbations are necessary but insufficient for mechanistic inference.
* Dex/MK-4 effects remain partially underdetermined.
* Model provides a **constraint-based inference tool**, not a unique mechanism.

---

## 8. Conclusion

* Built: literature-grounded hybrid CA.
* Tested: against structured endpoint perturbations.
* Learned:
  **Endpoint equivalence ≠ mechanistic equivalence.**

---

### Core Logic

**Literature biology → constrained CA → perturbation stress-test → identifiability limits revealed**

