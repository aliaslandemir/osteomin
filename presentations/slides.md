---
theme: default
background: '#f8fafc'
class: text-slate-900
transition: fade
title: OsteoMin — Understanding Biomineralization of Pre-osteoblast Cells
info: |
  ## Ali Aslan Demir, PhD Candidate  
  University of Oslo
  **18.11.25 - Work Discussion**
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

18.11.2025
</div>




---
layout: cover
transition: fade
class: text-center bg-white
---

# Model Initialization
- `OsteoModel` (`src/model.py`) combine ABM and Field/PDE, structure the model.
- `seed_random(n, w, h)` scatters up to `w×h` cells onto free lattice sites and records them in the grid.
- It places stochastically; giving them unique ID & checking the occupancy.
- Update of each cycle (ABM_manager) follows the order.
- Every daughter (proliferation); also follow up this order.
- `ABMConfig` (width, height, boundary mode) dictates lattice size and behaviors.
- Purpose of the PDE field; in case of diffusion plans.
- Daily storing of masks, density grids, and optional `.npy` chemistry snapshots.
::right::

<div class="border border-slate-300 rounded-lg p-3 text-xs font-mono leading-relaxed bg-white/80 shadow-sm">
  <div>seed coords</div>
  <div class="text-center">⬇︎</div>
  <div>occupancy grid</div>
  <div class="text-center">⬇︎</div>
  <div>ABM ↔ PDE share RNG + masks</div>
  <div class="text-center">⬇︎</div>
  <div>daily exports (state, chem .npy, overlays)</div>
</div>

---
transition: fade-left
---

# Active Cell Behaviour
- Every agent begins in `CellState.ACTIVE` (`src/agent.py`) and can `random_move`, divide, commit, secrete, and die in that order each day.
- `random_move` uses `_random_move_step` if enabled; division clocks (`last_div_day`, `division_interval`) so proliferation waits 1 days.
- Commitment uses `base_commit_prob + day * commit_prob_growth`, but only starts once `day >= commit_delay`, so you can hold off osteoblast differentiation for the first few days (3).
- Active cells contribute to apoptosis statistics and can transition to osteoblasts.
::right::

---
transition: slide-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Osteoblast Behaviour
- `mark_committed` flips a cell to `CellState.OSTEOBLAST`, tracks `day_committed`, and flags `ever_ob` for history.
- Osteoblasts secrete collagen and ALP in `_secretion_step`, scaled per-cell (`collagen_per_ob`, `alp_per_ob`) before apoptosis runs.
- A boolean OB mask is emitted to the PDE each day so matrix and mineral chemistry stays localized to living osteoblast pixels.
- OBs remain subject to apoptosis, with deaths logged separately (`dead_ob_coords`) for overlays and statistics. 
 - OB-specific death probability (`ob_apoptosis_prob`, default 0.001) keeps committed cells alive longer than ACTIVE cells (20x less apoptotic).
::right::

<div class="border border-orange-200 bg-orange-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>ACTIVE + commit ✔ → mark OB</div>
  <div class="text-center">↓</div>
  <div>emit OB mask → PDE uses pixels</div>
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
- When `enable_random_move` (the `random_move` toggle) is true, ACTIVE cells can attempt up to `max_moves_per_day` hops.
- `_movement_target` biases a stay-put probability (0.6) and samples 8-neighbor offsets with axial vs diagonal weights.
- `boundary_mode` controls wrapping: `periodic` wraps coordinates; `clamp` aborts moves that leave the lattice.
- Moves swap occupancy only if the destination is empty.
::right::

<div class="border border-sky-200 bg-sky-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>stay prob = 0.5</div>
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
- After division, the occupancy grid is rebuilt so downstream steps (commitment, PDE coupling) see consistent positions.
::right::

<div class="border border-purple-200 bg-purple-50/80 rounded-lg p-3 text-xs font-mono leading-relaxed">
  <div>interval met?</div>
  <div class="text-center">↓ yes</div>
  <div>chance pass?</div>
  <div class="text-center">↓ yes</div>
  <div>search neighbor → empty slot?</div>
  <div class="text-center">↓</div>
  <div>spawn ACTIVE daughter / reset parent clock</div>
</div>



---
transition: slide-left
---


# What to fit? #1

- Placement of the grid of cells (stochastic!)

- Random move, Chances of move (only one move!), Proliferation, Apoptosis are liteture finding. No need to fit!


- Collagen secretion! AA to Collagen. Example models are in the bib list. I will summarize them.

- What is important here is the half-saturation constant; which determines how well the collagen matrix or osteogenesis happens.

- Cell crowding to collagen? Is it necessary?



---
transition: slide-left
---

# What to fit? #2

- Isotropic diffusion & boundary handling to match cell behaviour - update within each cycle!

- Diffusion for ions (no interaction), time steps to match daily rates!

- Dose-dependent biphasic effect for Dexamethasone?

- MK4; synergy or anthogony to Dex?

- Density dependent responses; disregard - nodule detection to capture synthetically?

---
transition: fade
---

### 🔧 Parameters Selected for Fitting

#### 🟠 AA → Collagen Module

| Parameter | Role | Reason to Fit |
|----------|------|----------------|
| **base_collagen_rate** | Baseline collagen production (AA−) | Sets absolute collagen/ARS level in AA− |
| **k_aa** | AA sensitivity / gain | Controls fold-change from AA− → AA+ |
| *aa_usage_per_collagen* (optional) | AA consumed per collagen | Fit only if AA depletion matters |

---

#### 🔵 βGP → Pi Module

| Parameter | Role | Reason to Fit |
|----------|------|----------------|
| **alp_to_pi_rate** | ALP-driven βGP → Pi conversion | Controls Pi buildup & mineralization onset |
| **bgp_usage_per_pi** | βGP consumed per Pi | Fit only when βGP+ vs βGP− gap is strong |


-
-

#### 🟣 Mineralization (fit **one** only)

| Parameter | Role | Reason to Fit |
|----------|------|----------------|
| **S_THRESH** *(recommended)* | Supersaturation threshold for ACP nucleation | Shifts mineral onset timing/intensity |
| *or* **collagen_nucleation_threshold** | Collagen needed before nucleation | Couples matrix accumulation → mineral start |



---
transition: fade-left
---

### 🌊 Adding Diffusion — What Should Be Fitted?

#### 🧱 Parameters to **Fix** (physics / design)

- Individual diffusion coefficients:  
  `D_Ca`, `D_Pi`, `D_AA`, `D_BGP`, `D_ALP`  
  <span class="text-xs text-slate-500">Use literature values for small ions / molecules.</span>
- Boundary condition type  
  <span class="text-xs text-slate-500">(Dirichlet / Neumann / periodic)</span>
- Numerical settings: `dt`, `dx`  
  <span class="text-xs text-slate-500">Affect stability, not biology.</span>


#### 🔧 Minimal Extra Parameters You *Can* Fit

| Parameter | Meaning | Why Fit? |
|----------|---------|----------|
| **D_scale_ions** | Global multiplier on Ca/Pi diffusion | Captures effective slowing in ECM vs fluid |
| *leak_rate* (optional) | Relaxation to boundary/media | Only if spatial gradients matter experimentally |


---

### 🎯 Practical Recommendation (endpoint ALP/ARS)

- Keep all raw `D_x` values **fixed**  
- Fit **only** `D_scale_ions` if diffusion noticeably affects outputs  
- Core curve-fit set remains unchanged:  
  - `base_collagen_rate`, `k_aa`  
  - `alp_to_pi_rate`, `bgp_usage_per_pi`  
  - **One** mineralization knob: `S_THRESH` *or* `collagen_nucleation_threshold`

- Isotropic diffusion

---
transition: fade-left
layout: two-cols
class: grid-cols-[2fr_1fr] items-start
---


# Collagen Production
- Osteoblast counts feed `self.pde.add_collagen(n_ob * collagen_per_ob)` before apoptosis.
- AA availability multiplies matrix deposition via `AA_TO_COLLAGEN_RATE`.
- If a PDE osteoblast mask exists, collagen is deposited only on those pixels; otherwise it spreads uniformly.
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
- PDE chemistry (`src/pde_manager.py`) first raises Pi via ALP (`ALP_TO_PI_RATE * β-GP`) to drive supersaturation.
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




<div class="relative w-[85%] mx-auto group" data-id="frame">
  <div  
    class="absolute inset-0 rounded-[36px] border border-white/20 
           shadow-[0_35px_65px_-35px_rgba(16,185,129,0.7)] 
           transition-transform duration-700 ease-out group-hover:scale-[1.02]">
  </div>
  <img
    src="./images/flowchartoverall.png"
    class="relative w-full h-full object-cover rounded-[36px] 
           transition-transform duration-700 ease-out group-hover:scale-[1.02]"
    data-id="hero"
  />
</div>



---
transition: slide-left
layout: default
class: grid-cols-[2fr_1fr] items-start
---

# Nucleation Scaffold & HAp formation

<br><br>
<br><br>




<div class="relative w-[85 %] mx-auto" data-id="frame">
  <div
    class="absolute inset-0 rounded-[36px] border border-white/20 
               shadow-[0_35px_65px_-35px_rgba(203,15,198,0.7)] 
               transition-transform duration-700 ease-out hover:scale-[1.02]">
  </div>
  <img
    src="./images/overall2.png"
    class="relative w-full h-full object-cover rounded-[36px] 
               transition-all"
    data-id="hero"
  />
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




### Discussion meeting 18.11.2025 w/ Aurelie


**• Pre-OB seeding**  
Yes, all cells start as pre-osteoblasts.

**• Rule order**  
Rules run sequentially: move → proliferate → differentiate → secrete → apoptotic → nucleate

**• Proliferation → OB?**  
A cell that proliferates have to wait >= 1 day to become OB.

**• Grid sweep**  
Grid updates snapwise implemented; each time to whole grid. Only nodule search sliding window.

**• Rule 3 timer (>1 day)**  
Once proliferated; each cell have to wait one day before OB chance.


**• Movement probabilities**  
BIO-LGCA paper. Stay / axial / diagonal rules + isotrophy on grid.

---
layout: cover
transition: fade
class: text-center bg-white
---

**• Rule 4 ("g" growth)**  
“g” = daily increase in commitment (linear); longer undifferentiated = higher probability.\
Supplementing (media 2 days) with AA+B-GP pushes cells to differentiate.

**• Rule 5 – n_OB** 
This part is a bit tricky. I define the collagen production on a grid level; collective. Because cells feels each other through actins. 
They do produce same amount of collagen; overall level distributed to each cell uniformly. AA depletion (half life time) defines collective collagen production.
MK4 effect through SXR (steroid/X-receptor) (10.1074/jbc.M600896200) (10.1111/j.1600-0838.2009.00986.x)

**• Collagen secretion**  
If AA is depleted locally → secretion stops.  
Collagen depends only on AA + MK4.

**• ACP**  
Amorphous calcium phosphate — early mineral precursor.  
I want to include this part particularly.

**• Supplements without diffusion**  
No diffusion → no gradients yeap. Everything handled locally.  
Through ionic availability, half life time of AA we are assuming they are not excess in the medium.
