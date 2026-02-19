# 📗 Paper 2

## Brüningk et al. (2019)


**“A cellular automaton model for spheroid response to radiation and hyperthermia treatments”**

---

## 🎯 Aim

To develop a **3D cellular automaton model** that:

* Simulates tumour spheroid growth
* Predicts response to:

  * Radiation (RT)
  * Hyperthermia (HT)
  * Combined RT + HT

### Central Question

> Can differences in cell death mechanisms explain observed spheroid growth dynamics after treatment?

---

## 🧱 Paper Structure

### 1. Experimental Section

* HCT116 tumour spheroids grown in vitro
* Radiation (0–10 Gy)
* Hyperthermia (0–240 CEM43)
* Growth monitored over 3 weeks
* Clonogenic survival measured

This provides the data used for calibration and validation.

---

### 2. Computational Model

**3D Cellular Automaton**

* Each voxel = one cell
* Explicit cell cycle stages
* Oxygen diffusion solved via PDE
* Hypoxia-induced necrosis
* Quiescence under spatial constraints

---

### 3. Treatment Response Mechanisms

#### Radiation

* Linear–Quadratic model
* Oxygen enhancement ratio
* Cell death is **proliferation-dependent**
* Mitotic catastrophe
* Delayed shrinkage

#### Hyperthermia

* Clonogenic survival-based
* Cell death is:

  * Rapid
  * Proliferation-independent
* Immediate clearance
* Reoxygenation possible

---

### 4. Parameter Fitting & Validation

* Fit parameters using 10 Gy curve
* Validate on 2 Gy and 5 Gy
* Fit 240 CEM43
* Validate 40 & 80 CEM43
* Combination treatments predicted without new parameters

R² values > 0.85 reported.

---

## 📊 Main Results

### 1️⃣ Oxygen Gradients Reproduced

Simulated oxygen maps match analytical model and histology.

---

### 2️⃣ Radiation vs Hyperthermia Produce Different Dynamics

Despite similar clonogenic survival:

* Radiation → delayed shrinkage
* Hyperthermia → rapid collapse + reoxygenation

Mechanism matters.

---

### 3️⃣ Combined Treatment Predictive

The model predicts multimodal treatment response without additional fitting.

---

## 🔎 Interpretation

This is a **model + experiment paper**.

Strengths:

* Quantitative calibration
* Mechanistic cell death modelling
* Predictive validation

Limitations:

* No evolutionary dynamics
* No genotype modelling

---

---

# 🔬 Direct Comparison

| Feature                 | Gerlee (2007)   | Brüningk (2019) |
| ----------------------- | --------------- | --------------- |
| Model type              | Hybrid CA + PDE | 3D CA + PDE     |
| Evolution               | Yes             | No              |
| Neural network genotype | Yes             | No              |
| Experimental validation | No              | Yes             |
| Treatment modelling     | No              | Yes             |
| Oxygen-driven effects   | Yes             | Yes             |
| Quantitative fitting    | No              | Yes             |

---

# 🧠 Conceptual Difference

### Gerlee (2007)

Focuses on:

> Evolutionary tumour ecology

Microenvironment → selection → phenotype diversification

---

### Brüningk (2019)

Focuses on:

> Mechanistic treatment response

Microenvironment → treatment sensitivity → growth dynamics

---



