
---

# 📘 Paper 1

## Gerlee & Anderson (2007)


**“An evolutionary hybrid cellular automaton model of solid tumour growth”**

---

## 🎯 Aim

To develop a **hybrid cellular automaton model** that explicitly incorporates:

* Clonal evolution
* Genotype–phenotype mapping
* Microenvironmental selection

### Central Question

> How does tissue oxygen concentration influence tumour morphology and evolutionary dynamics during the avascular stage?

The goal is to mechanistically link:

**Hypoxia → Selection pressure → Evolution → Tumour morphology**

---

## 🧱 Paper Structure

### 1. Biological Motivation

* Cancer as an evolutionary process
* Role of hypoxia in tumour aggressiveness
* Need for evolutionary modelling (not static phenotypes)

### 2. Model Formulation

**Hybrid CA framework:**

* **Discrete layer:** Cells on a 2D lattice
* **Continuous layer:** PDEs for

  * Oxygen
  * Glucose
  * H⁺ concentration

Reaction–diffusion equations govern nutrient dynamics.

---

### 3. Core Innovation: Evolvable Neural Network per Cell

Each cell contains a **feed-forward neural network**:

**Input layer**

* Oxygen
* Glucose
* Acidity
* Neighbour count

**Hidden layer**

* Regulatory “genes”

**Output layer**

* Proliferation
* Quiescence
* Apoptosis
* Metabolism
* Movement

The network weights:

* Mutate during cell division
* Represent genotype
* Determine phenotype

This creates true evolutionary dynamics.

---

### 4. Simulation Experiments

Different tissue oxygen concentrations are imposed.

Measured outcomes:

* Tumour morphology
* Growth rate
* Clonal diversity (Shannon index)
* Phenotype distribution

---

## 📊 Main Results

### 1️⃣ Morphology Depends on Oxygen

* **Low oxygen**

  * Fingered / invasive morphology
  * Aggressive phenotypes
  * Reduced apoptotic response

* **High oxygen**

  * Compact, round tumours
  * Less aggressive clones

---

### 2️⃣ Selection Pressure Increases Under Hypoxia

* Faster diversification
* Stronger clonal selection
* Emergence of glycolytic phenotypes

---

### 3️⃣ Microenvironment Shapes Evolution

Hypoxia does not just affect growth —
it actively drives evolutionary trajectories.

---

## 🔎 Interpretation

This is a **theoretical framework paper**.

Strengths:

* Explicit genotype–phenotype mapping
* Evolution embedded naturally
* Clear biological narrative

Limitations:

* No direct experimental calibration
* Neural network remains abstract

---
