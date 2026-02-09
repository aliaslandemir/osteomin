from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple
import json
import numpy as np
import pandas as pd

from .abm_manager import ABMManager
from .config import ABMConfig, FieldConfig
from .field_manager import FieldManager
from .agent import CellState
from .datafit import DatafitSchedule
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
        datafit_path: Path | None = None,
        datafit_condition: str | int | None = None,
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

        datafit_schedule: DatafitSchedule | None = None
        if datafit_path is not None:
            resolved = Path(datafit_path)
            if not resolved.exists():
                raise SystemExit(f"--datafit file not found: {resolved}")
            datafit_schedule = DatafitSchedule.from_path(
                resolved, self.total_hours, condition=datafit_condition
            )
        self.datafit_schedule = datafit_schedule

        self.field = FieldManager(
            width=field_cfg.width,
            height=field_cfg.height,
            media_interval=field_cfg.media_interval,
            enable_nodules=getattr(field_cfg, "enable_nodules", False),
            nodule_window=getattr(field_cfg, "nodule_window", 10),
            datafit_schedule=datafit_schedule,
        )
        self.abm.attach_field(self.field)

        self._rows: List[Dict] = []
        self._nodule_history: dict[int, dict] = {}
        self._cell_history: List[Dict[str, Any]] = []
        self._daily_dead_preob = 0
        self._daily_dead_ob = 0
        self._daily_committed = 0
        self._daily_dead_cells: List[Tuple[int, int]] = []
        self._daily_dead_preob_coords: List[Tuple[int, int]] = []
        self._daily_dead_ob_coords: List[Tuple[int, int]] = []
        self.hourly_exports = bool(hourly_exports)

        self.spatial_every = max(0, int(spatial_every))
        base_days = set((spatial_days or []))
        base_days.add(self.days - 1)  # always export the latest day index
        self.spatial_days = {d for d in base_days if 0 <= d < self.days}

        meta = {
            "total_hours": self.total_hours,
            "hours_per_day": self.hours_per_day,
            "days": self.days,
            "seed": seed,
            "abm_cfg": __import__("dataclasses").asdict(abm_cfg),
            "field_cfg": __import__("dataclasses").asdict(field_cfg),
            "spatial_every": self.spatial_every,
            "spatial_days": sorted(list(self.spatial_days)),
            "snapshots_enabled": self.snapshots_enabled,
            "datafit_path": str(datafit_path) if datafit_path is not None else None,
            "datafit_condition": datafit_condition,
        }
        (self.outdir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def seed_random(self, n: int, w: int, h: int) -> None:
        sites = [(x, y) for y in range(h) for x in range(w)]
        n = min(int(n), len(sites))
        idx = self.rng.choice(len(sites), size=n, replace=False)
        coords = [sites[i] for i in idx]
        self.abm.seed_from_coords(coords, day0=0)

    def _should_export_spatial(self, day: int) -> bool:
        if day in self.spatial_days:
            return True
        if self.spatial_every > 0 and (day % self.spatial_every) == 0:
            return True
        return False

    def run(self) -> pd.DataFrame:
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
            np.save(self.grids_dir / f"collagen_day_000.npy", self.field.collagen.astype(np.float32))
            np.save(self.grids_dir / f"ha_day_000.npy",       self.field.ha.astype(np.float32))

        self._handle_nodules(0, state)
        self._append_row(0)
        dead_total_cum = 0
        if self._should_export_spatial(0):
            export_spatial_for_day(
                self.outdir,
                0,
                state,
                deaths_preob_mask,
                deaths_ob_mask,
                dead_total_cum=dead_total_cum,
            )

        if self.total_hours == 0:
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
                    self.grids_dir / f"collagen_day_{day_index:03d}.npy",
                    self.field.collagen.astype(np.float32),
                )
                np.save(
                    self.grids_dir / f"ha_day_{day_index:03d}.npy",
                    self.field.ha.astype(np.float32),
                )

            self._handle_nodules(day_index, final_state)
            self._append_row(
                day_index,
                dead_preob_today=self._daily_dead_preob,
                dead_ob_today=self._daily_dead_ob,
                committed_today=self._daily_committed,
            )

            dead_total_cum += daily_dead_total

            if self._should_export_spatial(day_index):
                export_spatial_for_day(
                    self.outdir,
                    day_index,
                    final_state,
                    deaths_preob_mask,
                    deaths_ob_mask,
                    dead_total_cum=dead_total_cum,
                )

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
    ) -> None:
        n_preob = sum(1 for c in self.abm.cells if c.state == CellState.PRE_OB)
        n_ob = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)

        if day == 0:
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
                "ha_total": 0.0,
                "collagen_preob_total": float(self.field.collagen_preob_total),
                "collagen_ob_total": float(self.field.collagen_ob_total),
                "alp_preob_total": float(self.field.alp_preob_total),
                "alp_ob_total": float(self.field.alp_ob_total),
                "mineral_preob_total": float(self.field.mineral_preob_total),
                "mineral_ob_total": float(self.field.mineral_ob_total),
            }
            row["nodule_count"] = int(
                self.field.nodule_data_for_day(day)[0]
            ) if hasattr(self.field, "nodule_data_for_day") else 0
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
        }
        row["nodule_count"] = int(
            self.field.nodule_data_for_day(day)[0]
        ) if hasattr(self.field, "nodule_data_for_day") else 0
        self._rows.append(row)


    def _handle_nodules(self, day: int, state_grid: np.ndarray) -> None:
        if not getattr(self.field, "enable_nodules", False):
            return
        ob_mask = state_grid == CellState.OSTEOBLAST
        count, positions = self.field.detect_nodules(day, ob_mask)
        self._nodule_history[day] = {"count": count, "positions": positions}
        self._save_nodules_snapshot(day, positions)

    def _save_nodules_snapshot(
        self, day: int, positions: List[Tuple[float, float]]
    ) -> None:
        if not getattr(self.field, "enable_nodules", False):
            return
        path = self.grids_dir / f"nodules_day_{day:03d}.npz"
        prev_positions = np.empty((0, 2), dtype=np.float32)
        prev_path = self.grids_dir / f"nodules_day_{day-1:03d}.npz"
        if day > 0 and prev_path.exists():
            prev = np.load(prev_path, allow_pickle=False)
            prev_positions = np.asarray(prev["positions"], dtype=np.float32)
        arr = np.asarray(positions, dtype=np.float32)
        if arr.size == 0:
            arr = arr.reshape(0, 2)
        combined = arr
        if prev_positions.size:
            if combined.size:
                combined = np.vstack([prev_positions, combined])
            else:
                combined = prev_positions
        if combined.size:
            tol = max(1.0, float(getattr(self.field, "nodule_window", 1)) * 0.5)
            unique: List[Tuple[float, float]] = []
            for x, y in combined:
                keep = True
                for ux, uy in unique:
                    if abs(float(ux) - float(x)) <= tol and abs(float(uy) - float(y)) <= tol:
                        keep = False
                        break
                if keep:
                    unique.append((x, y))
            combined = np.asarray(unique, dtype=np.float32)
        else:
            combined = combined.reshape(0, 2)
        np.savez(
            path,
            count=np.int32(len(combined)),
            positions=combined,
        )

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
