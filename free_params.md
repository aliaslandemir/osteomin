# OsteoMin Free Parameters

This document summarizes the tunable ("free") parameters for OsteoMin, where they live, and how they are used in the simulation. Icons are included for quick scanning.

---

## 🕒 Time scaling

- The simulator is now hourly-native: all CLI rates/probabilities/intervals are specified per hour (`--hours` is the primary time budget; `--days` is only a shorthand that multiplies by 24).
- Defaults have been rescaled to keep similar behaviour to the old daily setup (e.g., apoptosis defaults are ~0.00042/hour, commitment growth ~0.000017/hour, media interval 48 hours).
- Daily summaries/plots still emit once every 24 simulated hours, and the optional `--hourly-exports` flag controls extra per-hour spatial/collagen snapshots under `out/day_hor/dayX/`.

## 🧭 Entry Point (CLI Parameters)

These are passed via `python main.py ...` and wired into the runtime configs and model.

### 🧪 Core Simulation
- `--days` (default: 30)
  - **Used in**: `main.py` (model runtime), `src/model.py` (loop length).
  - **Effect**: Number of simulated days.
- `--cells` (default: 72)
  - **Used in**: `main.py` -> `model.seed_random(...)`.
  - **Effect**: Initial number of seeded cells.
- `--media-interval` (default: 2)
  - **Used in**: `FieldConfig.media_interval`, `FieldManager.step_hour()`.
  - **Effect**: Days between media refresh (resets Ca/Pi and supplement pools); converted to hours internally before the hourly chemistry tick.
- `--seed` (default: None)
  - **Used in**: `OsteoModel` RNG seed.
  - **Effect**: Reproducibility of stochastic steps.

### 🧩 Lattice and Boundary
- `--w` (default: 33), `--h` (default: 33)
  - **Used in**: `ABMConfig.width/height`, `FieldConfig.width/height`.
  - **Effect**: Grid dimensions for ABM and fields.
- `--boundary-mode` (default: periodic)
  - **Used in**: `ABMManager` (neighbor resolution).
  - **Effect**: Edge handling; `clamp` = hard edges, `periodic` = wrap.

### 🧴 Supplements
- `--aa` (default: 1.0)
  - **Used in**: `supplements` dict -> `FieldManager` AA pool.
  - **Effect**: Ascorbic acid scaling; limits collagen production.
- `--bgp` (default: 1.0)
  - **Used in**: `supplements` dict -> `FieldManager` BGP pool.
  - **Effect**: Beta-glycerophosphate supply; limits Pi generation.
- `--mk4` (default: 0.0)
  - **Used in**: `FieldManager` MK-4 effect.
  - **Effect**: Boosts collagen output and nucleation/maturation rates.
- `--dex` (default: None)
  - **Used in**: `ABMConfig.dex_dose`, `ABMManager` commitment/apoptosis scaling.
  - **Effect**: Dex dose (0-1). If absent, no Dex effect.
- `--dex-k` (default: 1.0)
  - **Used in**: `ABMConfig.dex_k`.
  - **Effect**: Strength of Dex effect on commitment probability.

### 🧬 Cell-Level Behavior
- `--random-move` / `--no-random-move` (default: enabled)
  - **Used in**: `ABMConfig.enable_random_move`, `ABMManager._random_move_step()`.
  - **Effect**: Toggles random motility.
- `--move-prob` (default: 1.0)
  - **Used in**: `ABMConfig.move_prob`.
  - **Effect**: Chance a preOB attempts a move per hour.
- `--max-moves` (default: 1)
  - **Used in**: `ABMConfig.max_moves_per_day`.
  - **Effect**: Max random-walk steps per hour.
- `--ob-div-prob` (default: 0.0)
  - **Used in**: `ABMConfig.ob_division_prob`.
  - **Effect**: Osteoblast division probability.

- `--osteogenic-commitment` / `--no-osteogenic-commitment` (default: enabled)
  - **Used in**: `ABMConfig.enable_commitment`, `ABMManager.step_hour()` (hourly solver).
  - **Effect**: Toggles preOB -> OB differentiation.
- `--commit-growth` (default: 0.01 / 576)
  - **Used in**: `ABMConfig.commit_prob_growth`.
  - **Effect**: Per-hour increase in commitment probability (legacy 0.01/day slope).
- `--commit-delay` (default: 72)
  - **Used in**: `ABMConfig.commit_delay`.
  - **Effect**: Minimum hours before commitment can start (legacy 3 days).

- `--apoptosis` / `--no-apoptosis` (default: enabled)
  - **Used in**: `ABMConfig.enable_apoptosis`, `ABMManager._apoptosis_step()`.
  - **Effect**: Toggles apoptosis.
- `--apoptosis-prob` (default: 0.01 / 24)
  - **Used in**: `ABMConfig.apoptosis_prob`.
  - **Effect**: Per-cell hourly apoptosis probability (legacy 0.01/day).
- `--apoptosis-osteo` (default: 0.001 / 24)
  - **Used in**: `ABMConfig.ob_apoptosis_prob`.
  - **Effect**: Per-osteoblast hourly apoptosis probability (legacy 0.001/day).

- `--proliferation` / `--no-proliferation` (default: enabled)
  - **Used in**: `ABMConfig.enable_proliferation`, `ABMManager._proliferation_step()`.
  - **Effect**: Toggles division events.
- `--proliferation-chance` (default: 1.0)
  - **Used in**: `ABMConfig.proliferation_chance`.
  - **Effect**: Per-cell chance to attempt division when interval passes.

- `--secretion` / `--no-secretion` (default: enabled)
  - **Used in**: `ABMConfig.enable_secretion`, `ABMManager._secretion_step()`.
  - **Effect**: Toggles OB secretion of collagen/ALP.

### 🧱 Mineralization
- `--s-thresh` (default: 0.01)
  - **Used in**: `FieldConfig.s_thresh`, `FieldManager.S_THRESH`.
  - **Effect**: Supersaturation threshold for ACP nucleation.
- `--k-nuc` (default: 0.025)
  - **Used in**: `FieldConfig.k_nuc`, `FieldManager.K_NUC` (converted to hourly rate).
  - **Effect**: ACP nucleation rate (per-day value scaled to per-hour before each chemistry tick).
- `--k-ha` (default: 0.06)
  - **Used in**: `FieldConfig.k_ha`, `FieldManager.K_HA` (converted to hourly rate).
  - **Effect**: ACP -> HA maturation rate (per-day value scaled to per-hour before each chemistry tick).
- `--enable-nodules` / `--no-nodules` (default: enabled)
  - **Used in**: `FieldConfig.enable_nodules`, `FieldManager.detect_nodules()`.
  - **Effect**: Toggles nodule detection.
- `--nodule-window` (default: 4)
  - **Used in**: `FieldConfig.nodule_window`, `FieldManager.detect_nodules()`.
  - **Effect**: Sliding window size for nodule scanning.
  - **Used in**: `FieldConfig.enable_nucleation`, `FieldManager.step_hour()`.
  - **Effect**: Toggles Pi -> ACP nucleation and HA maturation.

### 📦 Output / Export
- `--spatial-every` (default: 1)
  - **Used in**: `OsteoModel._should_export_spatial()`.
  - **Effect**: Export spatial PNG every N days (0 disables frequency).
- `--hourly-exports` / `--no-hourly-exports` (default: disabled)
  - **Used in**: `OsteoModel` hourly loop.
  - **Effect**: Toggles per-hour spatial/collagen snapshots under `out/day_hor/dayX/`.
- `--spatial-days` (default: "")
  - **Used in**: `main.py` -> `spatial_days` list.
  - **Effect**: Explicit export days, comma-separated.
- `--out` (default: out)
  - **Used in**: `OsteoModel` output directory.
  - **Effect**: Output directory path.
- `--xlsx` / `--no-xlsx` (default: enabled)
  - **Used in**: `OsteoModel.save_timeseries_excel()`.
  - **Effect**: Toggles Excel timeseries output.
- `--csv` (default: disabled)
  - **Used in**: `OsteoModel.save_timeseries_csv()`.
  - **Effect**: Toggles CSV timeseries output.
- `--snapshots` / `--no-snapshots` (default: enabled)
  - **Used in**: `OsteoModel` daily grid saves; overlays depend on snapshots.
  - **Effect**: Toggles .npy grid dumps.
- `--uniqueID` / `--no-unique-id` (default: enabled)
  - **Used in**: `OsteoModel.save_cell_history_excel()`.
  - **Effect**: Toggles per-cell history tracking/export.

---

## ⚙️ Config Defaults (Non-CLI but Tunable in Code)

These are in `src/config.py` and define defaults or parameters not surfaced in the CLI.

### 🔬 ABMConfig (Cell Behavior Defaults)
- `base_commit_prob` = 0.0001
  - **Used in**: `ABMManager.step_hour()` base commitment probability (monthly rate is updated hourly).
- `min_div_days` = 48, `max_div_days` = 96
  - **Used in**: `ABMManager._spawn_cell()` to randomize division interval (hours; legacy 2–4 days).
- `collagen_per_ob` = 1.5
  - **Used in**: Legacy placeholder (collagen now in `FieldManager`, ALP still uses `alp_per_ob`).
- `alp_per_ob` = 1.2
  - **Used in**: `ABMManager._secretion_step()` -> `FieldManager.add_alp()`.

### 🧪 FieldConfig (Field Chemistry Defaults)
- `ca0` = 1.0, `pi0` = 0.0
  - **Used in**: `FieldManager.refresh_media()` (baseline media).
- `alp_to_pi_rate` = 0.01
  - **Used in**: `FieldManager._produce_pi_from_alp()`.
- `ca_per_acp` = 3.0, `pi_per_acp` = 2.0
  - **Used in**: ACP nucleation mass balance.
- `acp_to_ha_threshold` = 0.05
  - **Used in**: `FieldManager` maturation threshold.
- `collagen_nucleation_threshold` = 1e-6
  - **Used in**: Limits HA maturation to collagen-positive sites.
- `acp_per_ha_reaction` = 3.0, `ca_per_reaction` = 1.0, `ha_per_reaction` = 2.0
  - **Used in**: HA maturation stoichiometry.
- `h_plus_per_ha` = 1.0, `kw` = 1.0, `h_base` = 1.0, `oh_base` = 1.0
  - **Used in**: H+/OH- equilibrium and baseline.
- `aa_usage_per_collagen` = 0.05
  - **Used in**: AA consumption per collagen unit.
- `bgp_usage_per_pi` = 3e-4
  - **Used in**: BGP consumption per Pi unit.
- `base_collagen_rate` = 0.001
  - **Used in**: `FieldManager._apply_collagen_secretion()`.
- `k_aa` = 0.2
  - **Used in**: AA saturation response in collagen secretion.

---

## 🧯 Internal Fixed Tuning (Not Exposed Yet)

These are fixed in code unless you edit the class directly.

- `ABMManager._stay_prob` = 0.5
  - **Used in**: `ABMManager._movement_target()`.
- Movement direction weights (axial vs diagonal)
  - **Used in**: Random walk probabilities.
- MK-4 piecewise multipliers (1.0 / 1.3 / 1.6)
  - **Used in**: `FieldManager._apply_collagen_secretion()` and nucleation/maturation scaling.
- Dex piecewise apoptosis scaling
  - **Used in**: `ABMManager._apoptosis_step()`.

---

## 📍 Source Pointers

- CLI and runtime wiring: `main.py`
- ABM defaults: `src/config.py` (`ABMConfig`)
- Field defaults: `src/config.py` (`FieldConfig`)
- ABM mechanics: `src/abm_manager.py`
- Field chemistry: `src/field_manager.py`
- Model wiring + export: `src/model.py`
