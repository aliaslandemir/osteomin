---

# OsteoMin Free Parameters

This document summarizes the knobs you can tune in the OsteoMin model. Parameters highlighted in **blue** in the attached LaTeX table are fixed by the literature/PDF; the **red** entries are the free, calibration-oriented choices you can adjust.

**Rule:**

* *PDF = biologically fixed (blue)*
* *Markdown here = calibrated / exploratory (red)*

---

## 🧬 Proliferation

* **Fixed (blue):** `min_div_hours` and `max_div_hours` bound cell-cycle durations to the MC3T3-E1 doubling-time window (≈ 38–52 h). Each daughter cell samples an integer interval uniformly from that range before it becomes eligible to divide again.
* **Free (red):**
  * `--proliferation / --no-proliferation` – master toggle for allowing the division loop.
  * `--proliferation-chance` – per-hour probability applied once the division window has elapsed; it governs how aggressively PRE\_OB cells attempt to spawn.

---

## 🧩 Differentiation & Commitment

* **Fixed (blue):**
  * `commit_delay` is anchored at 72 h after seeding/induction before any cell becomes eligible for commitment.
  * Differentiation is irreversible (osteoblast phenotype is terminal).
* **Free (red):**
  * `base_commit_prob` – the per-hour intercept of the commitment probability once the delay has passed.
  * `--commit-growth` (`commit_prob_growth`) – linear slope added every hour, forming
    \[
    p_{\text{commit}}(t)=\min(1,\;p_0 + g\,t)
    \]
    where \(t\) is hours since the first eligible hour after the last division.

---

## 🦴 Apoptosis

* **Free (red):**
  * `--apoptosis-prob` – PRE\_OB baseline death rate (scaled from the 2–8 % / 24 h literature range).
  * `--apoptosis-osteo` – OB-specific death rate (typically an order of magnitude lower).

---

## 🏃 Motility & Lattice

* **Fixed (blue):**
  * The lattice uses a Moore neighborhood with axial:diagonal weights \(1:1/\sqrt{2}\).
  * Nominal per-hour motility is set to ≈1 move/h based on migration speed literature.
* **Free (red):**
  * `--random-move / --no-random-move` – enable or disable PRE\_OB random walks.
  * `--move-prob` – Bernoulli gating of each hour’s movement attempt (higher value both increases hop chance and reduces the implicit stay-put bias).
  * `--max-moves` – caps the number of hop-attempts per hour when a movement attempt is triggered.
  * **Conceptual stay-put bias (`p_{\text{stay}}`):** the mass left by `1 - move_prob` can be interpreted as an independent free knob because it controls how often PES\_OB cells simply stay in place instead of stepping.

---

## ⚗️ ACP formation & HAp maturation

* **Fixed (blue):** stoichiometry (3 ACP + Ca²⁺ → 2 HAp) and the decision to route through an ACP precursor state.
* **Free (red):**
  * `--s-thresh` – local supersaturation threshold \(S_{\mathrm{thresh}}\) that gates ACP production.
  * `--k-nuc` – linear nucleation rate for ACP when \(S > S_{\mathrm{thresh}}\).
  * `--k-ha` – surface-controlled ACP→HAp conversion rate (often expressed with a saturating \(K_{\mathrm{sat}}\) denominator).

---

## 🧴 Collagen / ALP / AA coupling

* **Free (red):** all collagen/ALP scaling parameters remain tunable because they lack direct quantitative literature anchors:
  * `alp_per_ob`
  * `alp_to_pi_rate`
  * `base_collagen_rate`
  * `k_aa`
  * `aa_usage_per_collagen`
  * `bgp_usage_per_pi`
  * `collagen_nucleation_threshold`

---

## 🧯 Internal (Hard-coded but conceptually free)

These knobs are not exposed on the CLI, but they are not locked by literature either:

* `_move_prob` – internal move-target probability that biases the stay-put mass described above.
* Axial vs diagonal move weights in the Moore neighborhood.
* \(h^+\)/OH⁻ baseline constants (`kw`, `h_base`, `oh_base`) reused by the field chemistry logic.

---

## ✅ Explicitly NOT Free (defined in PDF)

* Cell-cycle time (~48 h, via `min_div_hours`/`max_div_hours`).
* Commitment delay (72 h) and irreversibility.
* Baseline apoptosis order of magnitude.
* Motility order (~1 move/h) and the hourly solver resolution (1 h).

---

## 📌 Summary

**The table’s “red” entries are captured here: you can tweak commitment kinetics, apoptosis rates, movement gating/budgets, and the mineralization kinetics; everything colored “blue” in the table (fixed biological time windows, stoichiometry, irreversibility) stays locked to the PDF.**

---

## Comprehensive Parameter Table (fixed vs literature vs experiment-matched)

Notes:
- "Fixed" means locked in the model (not tuned).
- "From literature" means the value/behavior is constrained by published ranges or qualitative biology.
- "Matched to experiments" means calibrated or intended to be calibrated against data (free parameters).

| Process | Parameter / Rule | Model key or CLI | Value / Range / Formula | Status | Evidence / Source | Matched to experiments |
|---|---|---|---|---|---|---|
| Proliferation | Cell-cycle window | `min_div_hours`, `max_div_hours` | 38–52 h | Fixed | Literature (MC3T3-E1 doubling time) | No |
| Proliferation | Division toggle | `--proliferation` / `--no-proliferation` | On/Off | Free | Model control | No |
| Proliferation | Division attempt rate | `--proliferation-chance` | Per-hour probability after window | Free | Model control | No |
| Proliferation | Post-commitment division | Rule | No division after OB | Fixed | Biology assumption | No |
| Apoptosis | PRE_OB baseline death | `--apoptosis-prob` | Per-hour (scaled from ~2–8%/day) | Free | Literature range used as guidance | No |
| Apoptosis | OB baseline death | `--apoptosis-osteo` | Lower than PRE_OB | Free | Literature trend | No |
| Differentiation | Induction timing | `commit_delay` | 72 h after seeding | Fixed | Protocol timing | No |
| Differentiation | Commitment rule | `base_commit_prob`, `commit_prob_growth` | `p_commit(t)=min(1, p0 + g*t)` | Free | Model choice | No |
| Differentiation | Eligibility constraint | Rule | Only if ≥24 h after last division | Fixed | Biology-informed rule | No |
| Differentiation | Reversibility | Rule | Irreversible OB state | Fixed | Biology assumption | No |
| Motility | Lattice neighborhood | Rule | Moore (8 neighbors) | Fixed | Standard ABM choice | No |
| Motility | Axial/diagonal weights | Rule | 1 vs 1/sqrt(2) | Fixed | Literature (grid walk) | No |
| Motility | Move gating | `--move-prob` | Per-hour Bernoulli | Free | Model control | No |
| Motility | Move attempts cap | `--max-moves` | Max hops per hour | Free | Model control | No |
| Motility | Random-walk toggle | `--random-move` / `--no-random-move` | On/Off | Free | Model control | No |
| Motility | Stay-put bias | `p_stay` (conceptual) | Implicit mass `1 - move_prob` | Free | Model control | No |
| Spatial/Temporal | Lattice spacing and timestep | Rule | Cell-sized grid, 1 h updates | Fixed | Model discretization | No |
| ACP/HAp | Pathway choice | Rule | ACP precursor state | Fixed | Literature (non-classical CaP) | No |
| ACP/HAp | Stoichiometry | Rule | 3 ACP + Ca2+ → 2 HAp | Fixed | Model assumption | No |
| ACP/HAp | Supersaturation threshold | `--s-thresh` | `S_thresh` | Free | Model control | Yes |
| ACP/HAp | ACP nucleation rate | `--k-nuc` | `k_nuc` | Free | Model control | Yes |
| ACP/HAp | HAp conversion rate | `--k-ha` | `k_HA` | Free | Model control | Yes |
| ACP/HAp | Saturation constant | `K_sat` | In `ACP/(K_sat + ACP)` | Free | Model control | Yes |
| Collagen/ALP/AA | ALP per OB | `alp_per_ob` | Scaling factor | Free | Model control | Yes |
| Collagen/ALP/AA | ALP → Pi rate | `alp_to_pi_rate` | Rate constant | Free | Model control | Yes |
| Collagen/ALP/AA | Basal collagen rate | `base_collagen_rate` | Rate constant | Free | Model control | Yes |
| Collagen/ALP/AA | AA coupling | `k_aa` | Scaling factor | Free | Model control | Yes |
| Collagen/ALP/AA | AA usage per collagen | `aa_usage_per_collagen` | Stoichiometric use | Free | Model control | Yes |
| Collagen/ALP/AA | BGP usage per Pi | `bgp_usage_per_pi` | Stoichiometric use | Free | Model control | Yes |
| Collagen/ALP/AA | Collagen mineralization threshold | `collagen_nucleation_threshold` | Threshold level | Free | Model control | Yes |
| Collagen | Mineralization criterion | `C_thresh` (conceptual) | Basal day-14 collagen | Fixed | Literature anchor | No |
| Internal | Move-target bias | `_move_prob` | Internal bias | Free (internal) | Model control | No |
| Internal | Lattice move weights | Axial/diagonal weights | 1 vs 1/sqrt(2) | Fixed | Model choice | No |
| Internal | Chemistry constants | `kw`, `h_base`, `oh_base` | Field baselines | Free (internal) | Model control | No |
