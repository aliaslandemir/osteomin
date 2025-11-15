from __future__ import annotations
from pathlib import Path
from typing import Dict, List, Iterable, Optional, Tuple
import json
import numpy as np
import pandas as pd

from .abm_manager import ABMManager
from .config import ABMConfig, FieldConfig
from .field_manager import FieldManager
from .agent import CellState
from .io_utils import cell_grids, save_timeseries_excel, save_timeseries_csv, coords_to_mask
from .viz import export_spatial_for_day


__all__ = ["OsteoModel"]

class OsteoModel:
    def __init__(self,
                 outdir: str | Path,
                 days: int,
                 abm_cfg: ABMConfig,
                 field_cfg: FieldConfig,
                 supplements: dict,
                 seed: Optional[int] = None,
                 spatial_every: int = 0,
                 spatial_days: Optional[Iterable[int]] = None,
                 snapshots_enabled: bool = False):
        self.outdir = Path(outdir); self.outdir.mkdir(parents=True, exist_ok=True)
        self.days = int(days)
        self.grids_dir = self.outdir / "grids"
        self.grids_dir.mkdir(parents=True, exist_ok=True)

        self.snapshots_enabled = bool(snapshots_enabled)

        self.rng = np.random.default_rng(seed)
        self.abm = ABMManager(abm_cfg, self.rng)
        self.field = FieldManager(
            width=field_cfg.width,
            height=field_cfg.height,
            media_interval=field_cfg.media_interval,
            supplements=supplements,
            enable_nucleation=field_cfg.enable_nucleation,
            enable_nodules=getattr(field_cfg, "enable_nodules", False),
            s_thresh=getattr(field_cfg, "s_thresh", 0.006),
            k_nuc=getattr(field_cfg, "k_nuc", 0.05),
            k_ha=getattr(field_cfg, "k_ha", 0.60),
            nodule_window=getattr(field_cfg, "nodule_window", 10),
            ca0=getattr(field_cfg, "ca0", 1.0),
            pi0=getattr(field_cfg, "pi0", 0.0),
            alp_to_pi_rate=getattr(field_cfg, "alp_to_pi_rate", 0.01),
            ca_per_acp=getattr(field_cfg, "ca_per_acp", 3.0),
            pi_per_acp=getattr(field_cfg, "pi_per_acp", 2.0),
            acp_to_ha_threshold=getattr(field_cfg, "acp_to_ha_threshold", 0.20),
            acp_per_ha_reaction=getattr(field_cfg, "acp_per_ha_reaction", 3.0),
            ca_per_reaction=getattr(field_cfg, "ca_per_reaction", 1.0),
            ha_per_reaction=getattr(field_cfg, "ha_per_reaction", 2.0),
            h_plus_per_ha=getattr(field_cfg, "h_plus_per_ha", 1.0),
            kw=getattr(field_cfg, "kw", 1.0),
            h_base=getattr(field_cfg, "h_base", 1.0),
            oh_base=getattr(field_cfg, "oh_base", 1.0),
            aa_usage_per_collagen=getattr(field_cfg, "aa_usage_per_collagen", 0.05),
            bgp_usage_per_pi=getattr(field_cfg, "bgp_usage_per_pi", 3e-4),
            base_collagen_rate=getattr(field_cfg, "base_collagen_rate", 0.001),
            k_aa=getattr(field_cfg, "k_aa", 0.2),
        )
        self.abm.attach_field(self.field)

        self._rows: List[Dict] = []
        self._nodule_history: dict[int, dict] = {}

        # Spatial export policy
        self.spatial_every = max(0, int(spatial_every))
        base_days = set((spatial_days or []))
        base_days.add(self.days - 1)  # always export last day
        self.spatial_days = {d for d in base_days if 0 <= d < self.days}

        # Metadata
        meta = {
            "days": self.days,
            "seed": seed,
            "abm_cfg": __import__("dataclasses").asdict(abm_cfg),
            "field_cfg": __import__("dataclasses").asdict(field_cfg),
            "supplements": supplements,
            "spatial_every": self.spatial_every,
            "spatial_days": sorted(list(self.spatial_days)),
            "snapshots_enabled": self.snapshots_enabled,
        }
        (self.outdir / "run.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    # ---- seeding ----
    def seed_random(self, n: int, w: int, h: int) -> None:
        sites = [(x, y) for y in range(h) for x in range(w)]
        n = min(int(n), len(sites))
        idx = self.rng.choice(len(sites), size=n, replace=False)
        coords = [sites[i] for i in idx]
        self.abm.seed_from_coords(coords, day0=0)

    # ---- export policy ----
    def _should_export_spatial(self, day: int) -> bool:
        if day in self.spatial_days:
            return True
        if self.spatial_every > 0 and (day % self.spatial_every) == 0:
            return True
        return False

    # ---- main run ----
    def run(self) -> pd.DataFrame:
        # ---------- DAY 0 ----------
        deaths_mask = np.zeros((self.abm.height, self.abm.width), dtype=np.uint8)
        deaths_active_mask = deaths_mask.copy()
        deaths_ob_mask = deaths_mask.copy()

        # ---------- DAY 0 ----------
        dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
        if self.snapshots_enabled:
            np.save(self.grids_dir / f"density_day_000.npy", dens.astype(np.uint16))
            np.save(self.grids_dir / f"state_day_000.npy",   state.astype(np.uint8))
            np.save(self.grids_dir / f"deaths_day_000.npy",  deaths_mask)
            np.save(self.grids_dir / f"deaths_active_day_000.npy", deaths_active_mask)
            np.save(self.grids_dir / f"deaths_ob_day_000.npy",     deaths_ob_mask)
            # NEW: chemistry snapshots
            np.save(self.grids_dir / f"collagen_day_000.npy", self.field.collagen.astype(np.float32))
            np.save(self.grids_dir / f"ha_day_000.npy",       self.field.ha.astype(np.float32))

        self._handle_nodules(0, state)
        self._append_row(0)
        dead_total_cum = 0
        if self._should_export_spatial(0):
            export_spatial_for_day(
                self.outdir, 0, state, deaths_active_mask, deaths_ob_mask,
                dead_total_cum=dead_total_cum
            )

        # ---------- DAYS 1..N-1 ----------
        for day in range(1, self.days):
            self.abm.step_day(day)

            deaths_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_cells)
            deaths_active_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_active_coords)
            deaths_ob_mask = coords_to_mask(self.abm.height, self.abm.width, self.abm.dead_ob_coords)

            dens, state = cell_grids(self.abm.height, self.abm.width, self.abm.cells)
            if self.snapshots_enabled:
                np.save(self.grids_dir / f"density_day_{day:03d}.npy", dens.astype(np.uint16))
                np.save(self.grids_dir / f"state_day_{day:03d}.npy",   state.astype(np.uint8))
                np.save(self.grids_dir / f"deaths_day_{day:03d}.npy",  deaths_mask)
                np.save(self.grids_dir / f"deaths_active_day_{day:03d}.npy", deaths_active_mask.astype(np.uint8))
                np.save(self.grids_dir / f"deaths_ob_day_{day:03d}.npy",     deaths_ob_mask.astype(np.uint8))
                # NEW: chemistry snapshots
                np.save(self.grids_dir / f"collagen_day_{day:03d}.npy", self.field.collagen.astype(np.float32))
                np.save(self.grids_dir / f"ha_day_{day:03d}.npy",       self.field.ha.astype(np.float32))


            self._handle_nodules(day, state)
            self._append_row(day)

            dead_total_cum += int(self._rows[-1]["dead_active_today"] + self._rows[-1]["dead_ob_today"])

            if self._should_export_spatial(day):
                export_spatial_for_day(
                    self.outdir, day, state, deaths_active_mask, deaths_ob_mask,
                    dead_total_cum=dead_total_cum
                )

        return pd.DataFrame(self._rows)
    

    def _append_row(self, day: int) -> None:
        n_active = sum(1 for c in self.abm.cells if c.state == CellState.ACTIVE)
        n_ob     = sum(1 for c in self.abm.cells if c.state == CellState.OSTEOBLAST)

        supp_aa = float(getattr(self.field, "aa_pool", 0.0))
        supp_bgp = float(getattr(self.field, "bgp_pool", 0.0))
        supp_mk4 = float(getattr(self.field, "mk4_level", 0.0))
        supp_dexa = float(getattr(self.field, "dexa_level", 0.0))

        if day == 0:
            # use current field scalars at t0
            row = {
                "day": int(day),
                "n_active": int(n_active),
                "n_osteoblast": int(n_ob),
                "n_total": int(n_active + n_ob),
                "dead_active_today": 0,
                "dead_ob_today": 0,
                "committed_today": 0,
                "collagen_total": float(self.field.total_collagen),
                "alp_total": float(self.field.total_alp),
                "acp_total": 0.0,
                "ha_total": 0.0,
                "ca_used_total": 0.0,
                "pi_used_total": 0.0,
                "ca_supplied_total": float(self.field.ca_supplied_total),
                "pi_supplied_total": float(self.field.pi_supplied_total),
                "pi_generated_total": float(self.field.pi_generated_total),
                "bgp_consumed_total": float(self.field.bgp_consumed_total),
                "cell_ca": float(self.field.cell_ca),
                "cell_pi": float(self.field.cell_pi),
                "ca_media": float(self.field.ca.mean()),
                "bgp_pool": float(self.field.bgp_pool),
                "ca_free_total": float(self.field.ca.sum()),
                "pi_free_total": float(self.field.pi.sum()),
                "supp_aa": supp_aa,
                "supp_bgp": supp_bgp,
                "supp_mk4": supp_mk4,
                "supp_dexa": supp_dexa,
            }
            row["nodule_count"] = int(
                self.field.nodule_data_for_day(day)[0]
            ) if hasattr(self.field, "nodule_data_for_day") else 0
            self._rows.append(row)
            return

        scalars = self.field.snapshot_scalars()
        row = {
            "day": int(day),
            "n_active": int(n_active),
            "n_osteoblast": int(n_ob),
            "n_total": int(n_active + n_ob),
            "dead_active_today": int(len(self.abm.dead_active_coords)),
            "dead_ob_today": int(len(self.abm.dead_ob_coords)),
            "committed_today": int(self.abm.committed_today),
            "cell_ca": float(scalars["cell_ca"]),
            "cell_pi": float(scalars["cell_pi"]),
            "ca_media": float(scalars["ca_media"]),
            "bgp_pool": float(scalars.get("bgp_pool", 0.0)),
            "collagen_total": float(scalars["collagen_total"]),
            "alp_total": float(scalars["alp_total"]),
            "acp_total": float(scalars.get("acp_total", 0.0)),
            "ha_total": float(scalars.get("ha_total", 0.0)),
            "ca_used_total": float(scalars.get("ca_used_total", 0.0)),
            "pi_used_total": float(scalars.get("pi_used_total", 0.0)),
            "ca_supplied_total": float(scalars.get("ca_supplied_total", 0.0)),
            "pi_supplied_total": float(scalars.get("pi_supplied_total", 0.0)),
            "pi_generated_total": float(scalars.get("pi_generated_total", 0.0)),
            "bgp_consumed_total": float(scalars.get("bgp_consumed_total", 0.0)),
            "ca_free_total": float(scalars.get("ca_free_total", 0.0)),
            "pi_free_total": float(scalars.get("pi_free_total", 0.0)),
            "supp_aa": supp_aa,
            "supp_bgp": supp_bgp,
            "supp_mk4": supp_mk4,
            "supp_dexa": supp_dexa,
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
