from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import os
import platform
import socket
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd

from .abm_manager import ABMManager
from .config import ABMConfig, FieldConfig
from .field_manager import FieldManager
from .modulatoreff import drug_response_summary
from .agent import CellState
from .io_utils import (
    cell_grids,
    coords_to_mask,
    save_cell_history_excel,
    save_timeseries_csv,
    save_timeseries_excel,
)
from .time_utils import HOURS_PER_DAY
from .viz import (
    export_spatial_for_day,
    export_spatial_hourly,
    export_collagen_ha_snapshot,
)


__all__ = ["OsteoModel"]

class OsteoModel:
    def __init__(
        self,
        outdir: str | Path,
        total_hours: int,
        abm_cfg: ABMConfig,
        field_cfg: FieldConfig,
        seed: Optional[int] = None,
        spatial_every: int = 0,
        spatial_days: Optional[Iterable[int]] = None,
        snapshots_enabled: bool = False,
        unique_id_tracking: bool = False,
        hourly_exports: bool = False,
        show_runtime: bool = False,
        runtime_every_days: int = 1,
    ):
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.total_hours = max(0, int(total_hours))
        self.hours_per_day = HOURS_PER_DAY
        self.day_blocks = (
            (self.total_hours + self.hours_per_day - 1) // self.hours_per_day
            if self.total_hours > 0
            else 0
        )
        self.days = 1 + self.day_blocks
        self.grids_dir = self.outdir / "grids"
        self.grids_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots_enabled = bool(snapshots_enabled)
        self.enable_unique_ids = bool(unique_id_tracking)

        self.rng = np.random.default_rng(seed)
        self.abm = ABMManager(abm_cfg, self.rng, track_cells=self.enable_unique_ids)

        self.field = FieldManager(
            width=field_cfg.width,
            height=field_cfg.height,
            total_hours=self.total_hours,
            media_interval=field_cfg.media_interval,
            enable_secretion=field_cfg.enable_secretion,
            osteo_induction_hours=field_cfg.osteo_induction_hours,
            dex_conc=field_cfg.dex_conc,
            mk4_conc=field_cfg.mk4_conc,
            dex_ref_conc=field_cfg.dex_ref_conc,
            mk4_ref_conc=field_cfg.mk4_ref_conc,
            alp_basal_rate=field_cfg.alp_basal_rate,
            alp_osteo_rate=field_cfg.alp_osteo_rate,
            alp_initial=field_cfg.alp_initial,
            alp_dex_ref_conc=field_cfg.alp_dex_ref_conc,
            alp_mk4_ref_conc=field_cfg.alp_mk4_ref_conc,
            alp_a_d=field_cfg.alp_a_d,
            alp_b_d=field_cfg.alp_b_d,
            alp_a_k=field_cfg.alp_a_k,
            alp_b_k=field_cfg.alp_b_k,
            alp_alpha_d=field_cfg.alp_alpha_d,
            alp_alpha_k=field_cfg.alp_alpha_k,
            alp_alpha_dk=field_cfg.alp_alpha_dk,
            collagen_basal_rate=field_cfg.collagen_basal_rate,
            collagen_osteo_rate=field_cfg.collagen_osteo_rate,
            collagen_dex_ref_conc=field_cfg.collagen_dex_ref_conc,
            collagen_mk4_ref_conc=field_cfg.collagen_mk4_ref_conc,
            collagen_a_d=field_cfg.collagen_a_d,
            collagen_b_d=field_cfg.collagen_b_d,
            collagen_a_k=field_cfg.collagen_a_k,
            collagen_b_k=field_cfg.collagen_b_k,
            collagen_alpha_d=field_cfg.collagen_alpha_d,
            collagen_alpha_k=field_cfg.collagen_alpha_k,
            collagen_alpha_dk=field_cfg.collagen_alpha_dk,
            aa_initial=field_cfg.aa_initial,
            aa_half_life_hours=field_cfg.aa_half_life_hours,
            collagen_aa_half_sat=field_cfg.collagen_aa_half_sat,
            ha_model=field_cfg.ha_model,
            ha_initial=field_cfg.ha_initial,
            ha_osteo_rate=field_cfg.ha_osteo_rate,
            ha_dex_ref_conc=field_cfg.ha_dex_ref_conc,
            ha_mk4_ref_conc=field_cfg.ha_mk4_ref_conc,
            ha_a_d=field_cfg.ha_a_d,
            ha_b_d=field_cfg.ha_b_d,
            ha_a_k=field_cfg.ha_a_k,
            ha_b_k=field_cfg.ha_b_k,
            ha_alpha_d=field_cfg.ha_alpha_d,
            ha_alpha_k=field_cfg.ha_alpha_k,
            ha_alpha_dk=field_cfg.ha_alpha_dk,
            ha_tau_hours=field_cfg.ha_tau_hours,
            ha_hill_n=field_cfg.ha_hill_n,
            ha_t_commit_hours=field_cfg.ha_t_commit_hours,
            ha_collagen_maturation_tau_hours=field_cfg.ha_collagen_maturation_tau_hours,
            ha_collagen_half_sat=field_cfg.ha_collagen_half_sat,
        )
        self.abm.attach_field(self.field)

        self._rows: List[Dict] = []
        self._cell_history: List[Dict[str, Any]] = []
        self._daily_dead_preob = 0
        self._daily_dead_ob = 0
        self._daily_committed = 0
        self._daily_dead_cells: List[Tuple[int, int]] = []
        self._daily_dead_preob_coords: List[Tuple[int, int]] = []
        self._daily_dead_ob_coords: List[Tuple[int, int]] = []
        self.hourly_exports = bool(hourly_exports)
        self.show_runtime = bool(show_runtime)
        self.runtime_every_days = max(1, int(runtime_every_days))

        self.spatial_every = max(0, int(spatial_every))
        base_days = set((spatial_days or []))
        base_days.add(self.days - 1)
        self.spatial_days = {d for d in base_days if 0 <= d < self.days}

        self._run_meta = {
            "total_hours": self.total_hours,
            "hours_per_day": self.hours_per_day,
            "days": self.days,
            "seed": seed,
            "abm_cfg": __import__("dataclasses").asdict(abm_cfg),
            "field_cfg": __import__("dataclasses").asdict(field_cfg),
            "drug_response": drug_response_summary(field_cfg),
            "spatial_every_hours": self.spatial_every,
            "spatial_days": sorted(list(self.spatial_days)),
            "snapshots_enabled": self.snapshots_enabled,
            "show_runtime": self.show_runtime,
            "runtime_every_days": self.runtime_every_days,
            "runtime_environment": {
                "hostname": socket.gethostname(),
                "platform": platform.platform(),
                "system": platform.system(),
                "release": platform.release(),
                "machine": platform.machine(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "logical_cpu_count": os.cpu_count(),
            },
            "model_size": {
                "grid_width": int(self.abm.width),
                "grid_height": int(self.abm.height),
                "grid_sites": int(self.abm.width * self.abm.height),
                "initial_cell_count": int(len(self.abm.cells)),
            },
            "timing": {
                "started_utc": datetime.now(timezone.utc).isoformat(),
            },
        }
        self._write_run_meta()

    def _write_run_meta(self) -> None:
        (self.outdir / "run.json").write_text(
            json.dumps(self._run_meta, indent=2), encoding="utf-8"
        )

    def _performance_summary(self, elapsed_seconds: float) -> Dict[str, Any]:
        elapsed_seconds = max(float(elapsed_seconds), 1e-12)
        wall_hours = elapsed_seconds / 3600.0
        sim_hours = float(self.total_hours)
        sim_days = sim_hours / float(self.hours_per_day)
        return {
            "wallclock_seconds": elapsed_seconds,
            "wallclock_minutes": elapsed_seconds / 60.0,
            "wallclock_hours": wall_hours,
            "simulated_hours": sim_hours,
            "simulated_days": sim_days,
            "simulated_hours_per_wallclock_second": sim_hours / elapsed_seconds,
            "simulated_hours_per_wallclock_hour": sim_hours / wall_hours,
            "simulated_days_per_wallclock_hour": sim_days / wall_hours,
        }

    def seed_random(self, n: int, w: int, h: int) -> None:
        sites = [(x, y) for y in range(h) for x in range(w)]
        n = min(int(n), len(sites))
        idx = self.rng.choice(len(sites), size=n, replace=False)
        coords = [sites[i] for i in idx]
        self.abm.seed_from_coords(coords, day0=0)

    def _should_export_spatial(self, day: int, hour: int) -> bool:
        if day in self.spatial_days:
            return True
        if self.spatial_every > 0 and (hour % self.spatial_every) == 0:
            return True
        return False

    def run(self) -> pd.DataFrame:
        t0 = time.perf_counter()
        deaths_mask = np.zeros((self.abm.height, self.abm.width), dtype=np.uint8)
        deaths_preob_mask = deaths_mask.copy()
        deaths_ob_mask = deaths_mask.copy()

        dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
        self._record_cell_snapshot(0)
        if self.snapshots_enabled:
            np.save(self.grids_dir / f"density_day_000.npy", dens.astype(np.uint16))
            np.save(self.grids_dir / f"state_day_000.npy",   state.astype(np.uint8))
            np.save(self.grids_dir / f"deaths_day_000.npy",  deaths_mask)
            np.save(self.grids_dir / f"deaths_preob_day_000.npy", deaths_preob_mask)
            np.save(self.grids_dir / f"deaths_ob_day_000.npy",     deaths_ob_mask)
            np.save(self.grids_dir / f"alp_day_000.npy",      self.field.alp.astype(np.float32))
            np.save(self.grids_dir / f"collagen_day_000.npy", self.field.collagen.astype(np.float32))
            np.save(self.grids_dir / f"ha_day_000.npy",       self.field.ha.astype(np.float32))

        self._append_row(0, state_grid=state)
        dead_total_cum = 0
        if self._should_export_spatial(0, 0):
            export_spatial_for_day(
                self.outdir,
                0,
                state,
                deaths_preob_mask,
                deaths_ob_mask,
                dead_total_cum=dead_total_cum,
            )

        if self.total_hours == 0:
            elapsed = time.perf_counter() - t0
            self._run_meta["timing"]["finished_utc"] = datetime.now(
                timezone.utc
            ).isoformat()
            self._run_meta["performance"] = self._performance_summary(elapsed)
            self._write_run_meta()
            return pd.DataFrame(self._rows)

        final_dens: np.ndarray | None = None
        final_state: np.ndarray | None = None
        for hour in range(1, self.total_hours + 1):
            hour_in_day = (hour - 1) % self.hours_per_day
            day_index = (hour - 1) // self.hours_per_day + 1
            if hour_in_day == 0:
                self._reset_daily_accumulators()

            self.abm.step_hour(hour)
            self._record_dead_cell_rows(hour)

            dens_hour, state_hour = cell_grids(
                self.abm.height, self.abm.width, self.abm.cells
            )
            deaths_preob_mask_hour = coords_to_mask(
                self.abm.height, self.abm.width, self.abm.dead_preob_coords
            )
            deaths_ob_mask_hour = coords_to_mask(
                self.abm.height, self.abm.width, self.abm.dead_ob_coords
            )
            self._daily_dead_preob += len(self.abm.dead_preob_coords)
            self._daily_dead_ob += len(self.abm.dead_ob_coords)
            self._daily_committed += self.abm.committed_today
            self._daily_dead_cells.extend(self.abm.dead_cells)
            self._daily_dead_preob_coords.extend(self.abm.dead_preob_coords)
            self._daily_dead_ob_coords.extend(self.abm.dead_ob_coords)

            dead_total_day = self._daily_dead_preob + self._daily_dead_ob
            if self.hourly_exports:
                dead_total_cum_hour = dead_total_cum + dead_total_day
                export_spatial_hourly(
                    self.outdir,
                    day_index,
                    hour_in_day,
                    state_hour,
                    deaths_preob_mask_hour,
                    deaths_ob_mask_hour,
                    dead_total_cum=dead_total_cum_hour,
                )
                export_collagen_ha_snapshot(
                    self.outdir,
                    day_index,
                    hour_in_day,
                    self.field.collagen.copy(),
                    self.field.ha.copy(),
                    state_hour,
                    deaths_preob_mask_hour,
                    deaths_ob_mask_hour,
                )

            final_dens, final_state = dens_hour, state_hour

            is_day_end = (
                hour_in_day == self.hours_per_day - 1 or hour == self.total_hours
            )
            if not is_day_end:
                continue

            deaths_mask = coords_to_mask(
                self.abm.height, self.abm.width, self._daily_dead_cells
            )
            deaths_preob_mask = coords_to_mask(
                self.abm.height, self.abm.width, self._daily_dead_preob_coords
            )
            deaths_ob_mask = coords_to_mask(
                self.abm.height, self.abm.width, self._daily_dead_ob_coords
            )

            daily_dead_total = self._daily_dead_preob + self._daily_dead_ob
            self._record_cell_snapshot(day_index)
            if self.snapshots_enabled:
                np.save(
                    self.grids_dir / f"density_day_{day_index:03d}.npy",
                    final_dens.astype(np.uint16),
                )
                np.save(
                    self.grids_dir / f"state_day_{day_index:03d}.npy",
                    final_state.astype(np.uint8),
                )
                np.save(
                    self.grids_dir / f"deaths_day_{day_index:03d}.npy",
                    deaths_mask,
                )
                np.save(
                    self.grids_dir / f"deaths_preob_day_{day_index:03d}.npy",
                    deaths_preob_mask.astype(np.uint8),
                )
                np.save(
                    self.grids_dir / f"deaths_ob_day_{day_index:03d}.npy",
                    deaths_ob_mask.astype(np.uint8),
                )
                np.save(
                    self.grids_dir / f"alp_day_{day_index:03d}.npy",
                    self.field.alp.astype(np.float32),
                )
                np.save(
                    self.grids_dir / f"collagen_day_{day_index:03d}.npy",
                    self.field.collagen.astype(np.float32),
                )
                np.save(
                    self.grids_dir / f"ha_day_{day_index:03d}.npy",
                    self.field.ha.astype(np.float32),
                )

            self._append_row(
                day_index,
                dead_preob_today=self._daily_dead_preob,
                dead_ob_today=self._daily_dead_ob,
                committed_today=self._daily_committed,
                state_grid=final_state,
            )

            dead_total_cum += daily_dead_total

            if self._should_export_spatial(day_index, hour):
                export_spatial_for_day(
                    self.outdir,
                    day_index,
                    final_state,
                    deaths_preob_mask,
                    deaths_ob_mask,
                    dead_total_cum=dead_total_cum,
                )

            if self.show_runtime and (
                day_index == self.days - 1 or day_index % self.runtime_every_days == 0
            ):
                elapsed = time.perf_counter() - t0
                frac = hour / max(1, self.total_hours)
                remain = (elapsed / frac - elapsed) if frac > 0.0 else 0.0
                print(
                    f"[runtime] day {day_index}/{self.days - 1} "
                    f"hour {hour}/{self.total_hours} "
                    f"elapsed {elapsed:.2f}s "
                    f"eta {max(0.0, remain):.2f}s"
                )

        if self.show_runtime:
            elapsed = time.perf_counter() - t0
            print(f"[runtime] finished in {elapsed:.2f}s")

        elapsed = time.perf_counter() - t0
        self._run_meta["timing"]["finished_utc"] = datetime.now(
            timezone.utc
        ).isoformat()
        self._run_meta["performance"] = self._performance_summary(elapsed)
        self._write_run_meta()

        return pd.DataFrame(self._rows)
    

    def _record_cell_snapshot(self, day: int) -> None:
        if not self.enable_unique_ids:
            return
        for c in self.abm.cells:
            row = {
                "day": int(day),
                "cell_id": int(c.unique_id),
                "row": int(c.y),
                "col": int(c.x),
                "state": int(c.state),
                "state_name": c.state.name,
                "alive": True,
                "ever_ob": bool(c.ever_ob),
                "birth_day": int(c.birth_day),
                "last_div_day": int(c.last_div_day),
                "day_committed": int(c.day_committed)
                if c.day_committed is not None
                else None,
                "day_died": int(c.day_died) if c.day_died is not None else None,
                "move_count": int(c.move_count),
            }
            self._cell_history.append(row)

    def _record_dead_cell_rows(self, day: int) -> None:
        if not self.enable_unique_ids:
            return
        rows = self.abm.drain_death_rows()
        if not rows:
            return
        self._cell_history.extend(rows)

    def _reset_daily_accumulators(self) -> None:
        self._daily_dead_preob = 0
        self._daily_dead_ob = 0
        self._daily_committed = 0
        self._daily_dead_cells = []
        self._daily_dead_preob_coords = []
        self._daily_dead_ob_coords = []

    def _append_row(
        self,
        day: int,
        *,
        dead_preob_today: int = 0,
        dead_ob_today: int = 0,
        committed_today: int = 0,
        state_grid: np.ndarray | None = None,
    ) -> None:
        n_preob = sum(1 for c in self.abm.cells if c.state == CellState.PRE_OB)
        n_ob = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)
        if state_grid is None:
            _, state_grid = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
        msd_effective = self._effective_msd_live_cells()
        avg_division_interval = self._average_division_interval_live_cells()
        avg_move_count = self._average_move_count_live_cells()
        avg_live_age_hours = self._average_live_age_hours(day)

        if day == 0:
            # Keep day-0 mineral consistent with initialized field state.
            scalars0 = self.field.snapshot_scalars()
            row = {
                "day": int(day),
                "n_preob": int(n_preob),
                "n_osteoblast": int(n_ob),
                "n_total": int(n_preob + n_ob),
                "dead_preob_today": 0,
                "dead_ob_today": 0,
                "committed_today": 0,
                "collagen_total": float(self.field.total_collagen),
                "alp_total": float(self.field.total_alp),
                "ha_total": float(scalars0.get("ha_total", 0.0)),
                "collagen_preob_total": float(self.field.collagen_preob_total),
                "collagen_ob_total": float(self.field.collagen_ob_total),
                "alp_preob_total": float(self.field.alp_preob_total),
                "alp_ob_total": float(self.field.alp_ob_total),
                "mineral_preob_total": float(self.field.mineral_preob_total),
                "mineral_ob_total": float(self.field.mineral_ob_total),
                "msd_effective": float(msd_effective),
                "avg_division_interval": float(avg_division_interval),
                "avg_move_count": float(avg_move_count),
                "avg_live_age_hours": float(avg_live_age_hours),
            }
            self._rows.append(row)
            return

        scalars = self.field.snapshot_scalars()
        row = {
            "day": int(day),
            "n_preob": int(n_preob),
            "n_osteoblast": int(n_ob),
            "n_total": int(n_preob + n_ob),
            "dead_preob_today": int(dead_preob_today),
            "dead_ob_today": int(dead_ob_today),
            "committed_today": int(committed_today),
            "collagen_total": float(scalars["collagen_total"]),
            "alp_total": float(scalars["alp_total"]),
            "ha_total": float(scalars.get("ha_total", 0.0)),
            "collagen_preob_total": float(scalars.get("collagen_preob_total", 0.0)),
            "collagen_ob_total": float(scalars.get("collagen_ob_total", 0.0)),
            "alp_preob_total": float(scalars.get("alp_preob_total", 0.0)),
            "alp_ob_total": float(scalars.get("alp_ob_total", 0.0)),
            "mineral_preob_total": float(scalars.get("mineral_preob_total", 0.0)),
            "mineral_ob_total": float(scalars.get("mineral_ob_total", 0.0)),
            "msd_effective": float(msd_effective),
            "avg_division_interval": float(avg_division_interval),
            "avg_move_count": float(avg_move_count),
            "avg_live_age_hours": float(avg_live_age_hours),
        }
        self._rows.append(row)

    def _effective_msd_live_cells(self) -> float:
        if not self.abm.cells:
            return 0.0
        vals: list[float] = []
        width = float(self.abm.width)
        height = float(self.abm.height)
        periodic = str(self.abm.boundary_mode).lower() == "periodic"
        for c in self.abm.cells:
            x0, y0 = c.trajectory[0]
            dx = float(c.x - x0)
            dy = float(c.y - y0)
            if periodic:
                adx = abs(dx)
                ady = abs(dy)
                dx = min(adx, width - adx)
                dy = min(ady, height - ady)
            vals.append(dx * dx + dy * dy)
        return float(np.mean(vals)) if vals else 0.0

    def _average_division_interval_live_cells(self) -> float:
        if not self.abm.cells:
            return 0.0
        return float(np.mean([float(c.division_interval) for c in self.abm.cells]))

    def _average_move_count_live_cells(self) -> float:
        if not self.abm.cells:
            return 0.0
        return float(np.mean([float(c.move_count) for c in self.abm.cells]))

    def _average_live_age_hours(self, day: int) -> float:
        if not self.abm.cells:
            return 0.0
        hour = int(day) * HOURS_PER_DAY
        ages = [max(0.0, float(hour - c.birth_day)) for c in self.abm.cells]
        return float(np.mean(ages)) if ages else 0.0

    def save_timeseries_excel(self, df: pd.DataFrame, basename: str = "timeseries") -> Path:
        out = self.outdir / f"{basename}.xlsx"
        save_timeseries_excel(df, out)
        return out

    def save_timeseries_csv(self, df: pd.DataFrame, basename: str = "timeseries") -> Path:
        out = self.outdir / f"{basename}.csv"
        save_timeseries_csv(df, out)
        return out

    def save_cell_history_excel(self, basename: str = "cell_history") -> Path:
        out = self.outdir / f"{basename}.xlsx"
        df = pd.DataFrame(self._cell_history)
        save_cell_history_excel(df, out)
        return out
