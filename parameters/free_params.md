
---

# OsteoMin Free Parameters

This document lists **all tunable (free) parameters** in the OsteoMin model.
Parameters constrained by literature and fixed in the PDF are **explicitly excluded**.

**Rule:**

* *PDF = biologically fixed*
* *Markdown here = calibrated / exploratory / modeling choices*

---

## 🕒 Time & Resolution (fixed elsewhere)

* Hourly solver (1 h timestep)
* Cell-sized lattice (~30 µm)
  → **Not free**, defined by model resolution

---

## 🧭 CLI-Exposed Free Parameters

### 🧪 Core Simulation

* `--days`
  Simulation duration (bookkeeping only)
* `--cells`
  Initial seeded cell number
* `--media-interval`
  Media refresh period (days → converted to hours)
* `--seed`
  RNG seed

---

### 🧩 Lattice & Boundary

* `--w`, `--h`
  Grid dimensions
* `--boundary-mode`
  `periodic` vs `clamp`

---

### 🧴 Supplements (scaling, not absolute biology)

* `--aa`
  Ascorbic acid availability scaling
* `--bgp`
  β-glycerophosphate availability scaling

---

### 🧬 Cell Motility & Fate (ABM)

#### Motility

* `--random-move / --no-random-move`
* `--move-prob`
  Probability a preOB attempts movement per hour
* `--max-moves`
  Max random-walk steps per hour

#### Proliferation

* `--proliferation / --no-proliferation`
* `--proliferation-chance`
  Chance to divide when division window opens
* `min_div_days`, `max_div_days` *(code-level)*
  Division interval bounds

#### Differentiation

* `--osteogenic-commitment / --no-osteogenic-commitment`
* `base_commit_prob` *(code-level baseline)*
* `--commit-growth`
  **Free slope** (order of magnitude fixed, exact value calibrated)
* `--commit-delay`
  Fixed at 72 h (PDF), **not free**

#### Apoptosis

* `--apoptosis / --no-apoptosis`
* `--apoptosis-prob` *(scaled from PDF but still tunable)*
* `--apoptosis-osteo` *(scaled from PDF but still tunable)*

---

### 🧱 Mineralization & Chemistry

#### Nucleation & Maturation

* `--s-thresh`
  Supersaturation threshold
* `--k-nuc`
  ACP nucleation rate
* `--k-ha`
  ACP → HA maturation rate

#### Nodules

* `--enable-nodules / --no-nodules`
* `--nodule-window`

#### Stoichiometry (model assumptions)

* `ca_per_acp`
* `pi_per_acp`
* `acp_per_ha_reaction`
* `ca_per_reaction`
* `ha_per_reaction`

---

### 🧬 Collagen / ALP / AA Coupling

* `alp_per_ob`
* `alp_to_pi_rate`
* `base_collagen_rate`
* `k_aa`
  AA saturation constant
* `aa_usage_per_collagen`
* `bgp_usage_per_pi`
* `collagen_nucleation_threshold`

*(All free: no direct quantitative literature constraints)*

---

### 🧯 Internal (Hard-coded but Conceptually Free)

These are **not exposed via CLI** but are not literature-fixed:

* Internal move target probability (`_move_prob`)
* Axial vs diagonal movement weights
* H⁺ / OH⁻ baseline constants (`kw`, `h_base`, `oh_base`)

---

## ✅ Explicitly NOT Free (defined in PDF)

* Cell-cycle time (~48 h)
* Apoptosis order of magnitude
* Commitment delay (72 h)
* Terminal differentiation
* Motility order (~1 move/h)
* Temporal resolution (1 h)

---

## 📌 Summary (one sentence)

**All kinetic rates, thresholds, efficiencies, and scalings are free; only biological time windows and qualitative behaviors are fixed.**

