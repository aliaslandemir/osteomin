---
theme: default
background: '#f8fafc'
class: text-slate-900
transition: fade
title: OsteoMin — Understanding Biomineralization of Pre-osteoblast Cells
info: |
  ## Ali Aslan Demir, PhD Candidate  
  University of Oslo
  **10.11.25 - Work Discussion**
colorSchema: light
fonts:
  sans: 'Inter'
  serif: 'IBM Plex Serif'
  mono: 'JetBrains Mono'
autoAnimate: false
duration: 20min
drawings:
  persist: false
mdc: true
addons:
  - '@slidev/theme-default'
---

---
layout: cover
transition: fade
class: text-center bg-white
---


# OsteoMin  
## Modelling the baheviour of MC3T3-E1 cells

 
**Ali Aslan Demir**  


<div class="mt-8 text-sm text-slate-500">
General Discussion on Model Behaviour

12.11.2025
</div>


---
layout: cover
transition: fade
class: text-center bg-white
---




# Model Initialization
- `OsteoModel` (`src/model.py`) combine ABM and the field manager, structure the model.
- `seed_random(n, w, h)` scatters up to `w×h` cells onto free lattice sites and records them in the grid.
- `ABMConfig` (width, height, boundary mode) dictates lattice size and behaviors; the field manager starts with uniform Ca/Pi and zero collagen/mineral.
- Purpose of the field manager; in case of diffusion plans.
- Daily storing of masks, density grids, and optional `.npy` chemistry snapshots.
::right::

<div class="border border-slate-300 rounded-lg p-3 text-xs font-mono leading-relaxed bg-white/80 shadow-sm">
  <div>seed coords</div>
  <div class="text-center">⬇︎</div>
  <div>occupancy grid</div>
  <div class="text-center">⬇︎</div>
  <div>ABM ↔ field manager share RNG + masks</div>
  <div class="text-center">⬇︎</div>
  <div>daily exports (state, chem .npy, overlays)</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# preOB Cell Behaviour
- Every agent begins in `CellState.PRE_OB` (`src/agent.py`) and can `random_move`, divide, commit, secrete, and die in that order each day.
- `random_move` uses `_random_move_step` if enabled; division clocks (`last_div_day`, `division_interval`) so proliferation waits 1 days.
- Commitment uses `base_commit_prob + day * commit_prob_growth`, but only starts once `day >= commit_delay`, so you can hold off osteoblast differentiation for the first few days (3).
- preOB cells contribute to apoptosis statistics and can transition to osteoblasts.
::right::

<div class="border border-emerald-200 bg-emerald-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>preOB</div>
  <div class="text-center">↓ random_move?</div>
  <div>division attempt (interval + chance)</div>
  <div class="text-center">↓</div>
  <div>commitment test → OB or stay preOB</div>
  <div class="text-center">↓</div>
  <div>apoptosis cleanup releases lattice</div>
</div>

---
transition: slide-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Osteoblast Behaviour
- `mark_committed` flips a cell to `CellState.OSTEOBLAST`, tracks `day_committed`, and flags `ever_ob` for history.
- Osteoblasts secrete collagen and ALP in `_secretion_step`, scaled per-cell (`collagen_per_ob`, `alp_per_ob`) before apoptosis runs.
- A boolean OB mask is emitted to the field manager each day so matrix and mineral chemistry stays localized to living osteoblast pixels.
- OBs remain subject to apoptosis, with deaths logged separately (`dead_ob_coords`) for overlays and statistics. 
- OB-specific death probability (`ob_apoptosis_prob`, default 0.001) keeps committed cells alive longer than preOB cells (20x less apoptotic).
::right::

<div class="border border-orange-200 bg-orange-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>preOB + commit ✔ → mark OB</div>
  <div class="text-center">↓</div>
  <div>emit OB mask → field manager uses pixels</div>
  <div class="text-center">↓</div>
  <div>secrete collagen + ALP per OB</div>
  <div class="text-center">↓</div>
  <div>OB apoptosis ⇒ clear mask + log death</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Random Movement
- When `enable_random_move` (the `random_move` toggle) is true, preOB cells can attempt up to `max_moves_per_day` hops.
- `_movement_target` biases a stay-put probability (0.6) and samples 8-neighbor offsets with axial vs diagonal weights.
- `boundary_mode` controls wrapping: `periodic` wraps coordinates; `clamp` aborts moves that leave the lattice.
- Moves swap occupancy only if the destination is empty.
::right::

<div class="border border-sky-200 bg-sky-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>stay prob = 0.6</div>
  <div class="text-center">│</div>
  <div>else pick 8-neighbor offset</div>
  <div class="text-center">↓ boundary</div>
  <div>periodic ⇒ wrap / clamp ⇒ reject OOB</div>
  <div class="text-center">↓</div>
  <div>move only if target cell empty</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Proliferation
- `_proliferation_step` enforces per-cell `division_interval` drawn between `min_div_days` and `max_div_days`.
- Cells must also pass `proliferation_chance`; osteoblast division is further throttled by `ob_division_prob`.
- Daughter cells spawn in the first free neighbor sampled; failure to place bumps `last_div_day` so the parent retries later.
- After division, the occupancy grid is rebuilt so downstream steps (commitment, field coupling) see consistent positions.
::right::

<div class="border border-purple-200 bg-purple-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>interval met?</div>
  <div class="text-center">↓ yes</div>
  <div>chance pass?</div>
  <div class="text-center">↓ yes</div>
  <div>search neighbor → empty slot?</div>
  <div class="text-center">↓</div>
  <div>spawn preOB daughter / reset parent clock</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Collagen Production
- Osteoblast counts feed `self.field.add_collagen(n_ob * collagen_per_ob)` before apoptosis.
- AA availability multiplies matrix deposition via `AA_TO_COLLAGEN_RATE`.
- If a field manager osteoblast mask exists, collagen is deposited only on those pixels; otherwise it spreads uniformly.
- `collagen_total` accumulates globally and is written to timeseries/chemistry snapshots for plotting.
::right::

<div class="border border-rose-200 bg-rose-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>n_OB × per-cell yield</div>
  <div class="text-center">×</div>
  <div>AA pool scale (0 ⇒ stall)</div>
  <div class="text-center">↓</div>
  <div>deposit on OB mask / uniform grid</div>
  <div class="text-center">↓</div>
  <div>_consume_aa reduces supplement</div>
</div>

---
transition: slide-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Supplement Conservation
- AA is reduced each time collagen is secreted via `_consume_aa`, so the `aa_pool` drops between media refreshes until the next dose.
- ALP converts β-GP into Pi (`ALP_TO_PI_RATE * bgp_pool`), and `_consume_bgp` subtracts that usage so β-GP acts as a conserved resource.
- Calcium is pulled from the media (`CA0`) and consumed during ACP/HA formation, so mineral reflects ionic availability rather than secreted Ca.
::right::

<div class="border border-amber-200 bg-amber-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>AA pool ──(collagen)──► decreases</div>
  <div>β-GP pool ──(ALP)──► Pi</div>
  <div class="text-center">Ca0 media → ACP/HA uses Ca/Pi</div>
  <div class="text-center">↺</div>
  <div>media refresh ⇒ reset AA/β-GP + ions</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Cell Media Workflow
- Fresh media loads Ca but no Pi; Pi is generated only when ALP converts β-GP.
- ACP nucleation checks the local product `Ca × Pi ≥ S_THRESH` on osteoblast pixels before consuming ions.
- ACP that exceeds `ACP_TO_HA_THRESHOLD` converts to HA using additional Ca, releasing protons and updating the usage counters.

::right::

<div class="border border-slate-200 bg-white/80 rounded-lg p-3 text-xs font-mono leading-relaxed space-y-1">
  <div>β-GP + ALP ──► Pi (local)</div>
  <div>Pi × Ca (≥ S_THRESH) ──► ACP</div>
  <div>ACP + Ca ──► HA + H⁺</div>
  <div>usage logs ⇒ `ca_used_total`, `pi_used_total`</div>
  <div>media refresh ⇒ Reset ions + supplements</div>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Hydroxyapatite Production
- Field manager chemistry (`src/field_manager.py`) first raises Pi via ALP (`ALP_TO_PI_RATE * β-GP`) to drive supersaturation.
- ACP nucleates where `Ca * Pi >= S_THRESH`, consuming ions with stoichiometric caps and staying confined to the OB mask.
- Once local ACP exceeds `ACP_TO_HA_THRESHOLD`, a fraction converts to HA at `K_HA`, consuming more Ca and releasing H⁺.

::right::

<div class="border border-indigo-200 bg-indigo-50/70 rounded-lg p-3 text-xs leading-relaxed">
  <p class="font-semibold text-indigo-800 mb-1">Mineral pipeline</p>
  <p>ALP → Pi spike (β-GP limited)</p>
  <p class="mt-1">Supersat ≥ S_THRESH ⇒ ACP nucleation</p>
  <p class="mt-1">ACP mass + Ca ⇒ HA + H⁺, export overlays</p>
</div>

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Running the Code
- Entry point `main.py` flags for days, cell count, supplements, export cadence, and mineralization kinetics.
- Outputs: CSV/XLSX timeseries, PNG overlays (per `--spatial-every` or `--spatial-days`), and raw `.npy` grids when `--snapshots` is on.
- Common run:

README.md
::right::

<div class="border border-slate-300 bg-white/70 rounded-lg p-3 text-xs leading-relaxed">
  <p class="font-semibold text-slate-800 mb-1">Execution map</p>
  <p>CLI args → config objects</p>
  <p class="mt-1">Run → dataframe → XLSX/PNG/NPY outputs</p>
  <p class="mt-1">Check `out/spatial_cells` + `out/collagen_ha` folders</p>
</div>



---
transition: slide-left
layout: default
class: grid-cols-[2fr_1fr] items-start
---


# Things to be discussed
- Linear or Hill? (collagen + hap + Pi Production)
- 1st: Collagen production simple function
- 2nd: ACP production (Nucleation logic / no surface to volume growth)
- 3rd: Abiotic CaP / Extravesicular Mineralization - Link to Nodule formations.
- No diffusion?
- 2 step osteoblast behaviour?
- Do osteoblast cells trigger each other to differentiate?
- For nucleation; 8+16+1 (nodule look for 25 block to decide threshold?)
- 
