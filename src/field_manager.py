from __future__ import annotations
from typing import Optional
import numpy as np
from .datafit import DatafitSchedule

__all__ = ["FieldManager"]


class FieldManager:
    def __init__(
        self,
        width: int,
        height: int,
        total_hours: int,
        media_interval: int,
        enable_secretion: bool = True,
        osteo_induction_hours: int = 72,
        dex_conc: float = 0.0,
        mk4_conc: float = 0.0,
        alp_basal_rate: float = 0.0,
        alp_osteo_rate: float = 0.0,
        alp_kappa_d: float = 0.0,
        alp_kappa_k: float = 0.0,
        collagen_basal_rate: float = 0.0,
        collagen_osteo_rate: float = 0.0,
        collagen_kappa_d: float = 0.0,
        collagen_kappa_k: float = 0.0,
        aa_initial: float = 1.0,
        aa_half_life_hours: float = 6.0,
        collagen_aa_half_sat: float = 1.0,
        datafit_schedule: DatafitSchedule | None = None,
        ha_model: str = "minimal",
        ha_osteo_rate: float = 0.0,
        ha_kappa_d: float = 0.0,
        ha_kappa_k: float = 0.0,
        ha_tau_hours: float = 384.0,
        ha_hill_n: float = 4.0,
        ha_t_commit_hours: float = 72.0,
        ha_collagen_maturation_tau_hours: float = 48.0,
        ha_collagen_half_sat: float = 1.0,
    ):
        self.width = int(width)
        self.height = int(height)
        self.area = float(self.width * self.height)
        self.total_hours = max(0, int(total_hours))
        self._hour = 0
        self.media_interval_hours = max(0, int(media_interval))
        self.hours_since_media = 0
        self.enable_secretion = bool(enable_secretion)
        self.osteo_induction_hours = max(0, int(osteo_induction_hours))
        self.dex_conc = float(dex_conc)
        self.mk4_conc = float(mk4_conc)
        self.alp_basal_rate = float(alp_basal_rate)
        self.alp_osteo_rate = float(alp_osteo_rate)
        self.alp_kappa_d = float(alp_kappa_d)
        self.alp_kappa_k = float(alp_kappa_k)
        self.collagen_basal_rate = float(collagen_basal_rate)
        self.collagen_osteo_rate = float(collagen_osteo_rate)
        self.collagen_kappa_d = float(collagen_kappa_d)
        self.collagen_kappa_k = float(collagen_kappa_k)
        self.aa_initial = max(0.0, float(aa_initial))
        self.aa_half_life_hours = max(1e-9, float(aa_half_life_hours))
        self.collagen_aa_half_sat = max(1e-9, float(collagen_aa_half_sat))
        self.datafit_schedule = datafit_schedule
        self.ha_model = str(ha_model).lower().strip()
        self.ha_osteo_rate = float(ha_osteo_rate)
        self.ha_kappa_d = float(ha_kappa_d)
        self.ha_kappa_k = float(ha_kappa_k)
        self.ha_tau_hours = max(1e-9, float(ha_tau_hours))
        self.ha_hill_n = max(1.0, float(ha_hill_n))
        self.ha_t_commit_hours = max(0.0, float(ha_t_commit_hours))
        self.ha_collagen_maturation_tau_hours = max(
            1e-9, float(ha_collagen_maturation_tau_hours)
        )
        self.ha_collagen_half_sat = max(1e-9, float(ha_collagen_half_sat))
        self._ha_tau_n = self.ha_tau_hours ** self.ha_hill_n
        self.collagen = np.zeros((self.height, self.width), dtype=float)
        self.collagen_mature = np.zeros((self.height, self.width), dtype=float)
        self.alp = np.zeros((self.height, self.width), dtype=float)
        self.ha = np.zeros((self.height, self.width), dtype=float)

        self.total_collagen = 0.0
        self.total_alp = 0.0
        self.collagen_preob_total = 0.0
        self.collagen_ob_total = 0.0
        self.alp_preob_total = 0.0
        self.alp_ob_total = 0.0
        self.mineral_preob_total = 0.0
        self.mineral_ob_total = 0.0

        self._ob_mask: Optional[np.ndarray] = None
        self._preob_mask: Optional[np.ndarray] = None
        self._ob_commit_hours: Optional[np.ndarray] = None
        self._aa_hourly = self._build_aa_schedule()

    def _osteo_active(self, hour: int) -> float:
        return 1.0 if int(hour) >= self.osteo_induction_hours else 0.0

    def _build_aa_schedule(self) -> np.ndarray:
        aa = np.zeros(self.total_hours + 1, dtype=float)
        if self.aa_initial <= 0.0:
            return aa
        k = np.log(2.0) / self.aa_half_life_hours
        for hour in range(self.total_hours + 1):
            if hour < self.osteo_induction_hours:
                continue
            if self.media_interval_hours <= 0:
                since = max(0.0, float(hour) - self.osteo_induction_hours)
            else:
                since = float((hour - self.osteo_induction_hours) % self.media_interval_hours)
            aa[hour] = self.aa_initial * np.exp(-k * since)
        return aa

    def _aa_concentration(self, hour: int) -> float:
        idx = min(max(0, int(hour)), self.total_hours)
        return float(self._aa_hourly[idx])

    def _deposit_uniform_to_mask(
        self,
        field: np.ndarray,
        mask: np.ndarray | None,
        amount_per_site: float,
    ) -> float:
        if mask is None or not mask.any() or amount_per_site <= 0.0:
            return 0.0
        field[mask] += amount_per_site
        return float(amount_per_site * int(mask.sum()))

    def _update_collagen_maturation(self, dt: float) -> None:
        lag = self.collagen - self.collagen_mature
        if not np.any(lag > 0.0):
            return
        self.collagen_mature += (dt / self.ha_collagen_maturation_tau_hours) * lag
        np.minimum(self.collagen_mature, self.collagen, out=self.collagen_mature)

    def _apply_equation_dynamics(self, hour: int) -> None:
        if not self.enable_secretion:
            return

        dt = 1.0
        pre_mask = self._preob_mask
        ob_mask = self._ob_mask
        u_osteo = self._osteo_active(hour)

        if pre_mask is not None and pre_mask.any():
            alp_added = self._deposit_uniform_to_mask(
                self.alp,
                pre_mask,
                dt * max(0.0, self.alp_basal_rate),
            )
            if alp_added > 0.0:
                self.total_alp += alp_added
                self.alp_preob_total += alp_added

            collagen_added = self._deposit_uniform_to_mask(
                self.collagen,
                pre_mask,
                dt * max(0.0, self.collagen_basal_rate),
            )
            if collagen_added > 0.0:
                self.total_collagen += collagen_added
                self.collagen_preob_total += collagen_added

        if ob_mask is None or not ob_mask.any() or u_osteo <= 0.0:
            self._update_collagen_maturation(dt)
            return

        alp_rate_ob = max(
            0.0,
            self.alp_osteo_rate + self.alp_kappa_d * self.dex_conc + self.alp_kappa_k * self.mk4_conc,
        )
        alp_added = self._deposit_uniform_to_mask(self.alp, ob_mask, dt * alp_rate_ob)
        if alp_added > 0.0:
            self.total_alp += alp_added
            self.alp_ob_total += alp_added

        aa_t = self._aa_concentration(hour)
        aa_gate = aa_t / (self.collagen_aa_half_sat + aa_t) if aa_t > 0.0 else 0.0
        collagen_rate_ob = max(
            0.0,
            (
                self.collagen_osteo_rate
                + self.collagen_kappa_d * self.dex_conc
                + self.collagen_kappa_k * self.mk4_conc
            )
            * aa_gate,
        )
        collagen_added = self._deposit_uniform_to_mask(
            self.collagen, ob_mask, dt * collagen_rate_ob
        )
        if collagen_added > 0.0:
            self.total_collagen += collagen_added
            self.collagen_ob_total += collagen_added

        self._update_collagen_maturation(dt)

        if self.ha_model != "minimal":
            return
        if self._ob_commit_hours is None:
            return

        active_commit = np.asarray(self._ob_commit_hours, dtype=float)
        valid = ob_mask & np.isfinite(active_commit)
        if not valid.any():
            return

        base_rate = max(
            0.0,
            self.ha_osteo_rate + self.ha_kappa_d * self.dex_conc + self.ha_kappa_k * self.mk4_conc,
        )
        if base_rate <= 0.0:
            return

        t_eff = np.maximum(0.0, float(hour) - active_commit)
        teff_n = np.power(t_eff, self.ha_hill_n)
        hill = np.zeros_like(t_eff, dtype=float)
        denom = self._ha_tau_n + teff_n
        np.divide(teff_n, denom, out=hill, where=denom > 0.0)
        collagen_gate = np.zeros_like(self.collagen_mature, dtype=float)
        collagen_denom = self.ha_collagen_half_sat + self.collagen_mature
        np.divide(
            self.collagen_mature,
            collagen_denom,
            out=collagen_gate,
            where=collagen_denom > 0.0,
        )
        delta = dt * base_rate * hill * collagen_gate
        delta[~valid] = 0.0
        added = float(delta.sum())
        if added <= 0.0:
            return
        self.ha += delta
        self.mineral_ob_total += added

    def add_alp(self, amount: float, ob_count: int = 0) -> None:
        amount = float(max(0.0, amount))
        if amount <= 0.0:
            return
        if self._ob_mask is not None and self._ob_mask.any():
            area = int(self._ob_mask.sum())
            self.alp[self._ob_mask] += amount / max(1, area)
        else:
            self.alp += amount / (self.width * self.height)
        self.total_alp += amount

    def _apply_datafit(self, hour: int, preob_count: int, ob_count: int) -> None:
        if self.datafit_schedule is None:
            return
        inc = self.datafit_schedule.hourly_increments(hour)
        pre_mask = self._preob_mask
        ob_mask = self._ob_mask

        collagen_preob = max(0.0, inc.get("collagen_basal", 0.0))
        collagen_ob = max(0.0, inc.get("collagen_osteo_ob", inc.get("collagen_osteo", 0.0)))
        alp_preob = max(0.0, inc.get("alp_basal", 0.0))
        alp_ob = max(0.0, inc.get("alp_osteo_ob", inc.get("alp_osteo", 0.0)))
        mineral_preob = 0.0
        if self.ha_model == "datafit":
            mineral_preob = max(0.0, inc.get("mineral_basal", 0.0))
        mineral_ob = 0.0
        if self.ha_model == "datafit":
            mineral_ob = max(0.0, inc.get("mineral_osteo_ob", inc.get("mineral_osteo", 0.0)))

        if pre_mask is not None and pre_mask.any():
            area = int(pre_mask.sum())
            if collagen_preob > 0.0:
                self.collagen[pre_mask] += collagen_preob / max(1, area)
                self.total_collagen += collagen_preob
                self.collagen_preob_total += collagen_preob
            if alp_preob > 0.0:
                self.alp[pre_mask] += alp_preob / max(1, area)
                self.total_alp += alp_preob
                self.alp_preob_total += alp_preob
            if mineral_preob > 0.0:
                self.ha[pre_mask] += mineral_preob / max(1, area)
                self.mineral_preob_total += mineral_preob
        else:
            if collagen_preob > 0.0:
                self.collagen += collagen_preob / max(1.0, self.area)
                self.total_collagen += collagen_preob
                self.collagen_preob_total += collagen_preob
            if alp_preob > 0.0:
                self.alp += alp_preob / max(1.0, self.area)
                self.total_alp += alp_preob
                self.alp_preob_total += alp_preob
            if mineral_preob > 0.0:
                self.ha += mineral_preob / max(1.0, self.area)
                self.mineral_preob_total += mineral_preob

        if ob_mask is not None and ob_mask.any():
            area = int(ob_mask.sum())
            if collagen_ob > 0.0:
                self.collagen[ob_mask] += collagen_ob / max(1, area)
                self.total_collagen += collagen_ob
                self.collagen_ob_total += collagen_ob
            if alp_ob > 0.0:
                self.alp[ob_mask] += alp_ob / max(1, area)
                self.total_alp += alp_ob
                self.alp_ob_total += alp_ob
            if mineral_ob > 0.0:
                self.ha[ob_mask] += mineral_ob / max(1, area)
                self.mineral_ob_total += mineral_ob
        else:
            if collagen_ob > 0.0:
                self.collagen += collagen_ob / max(1.0, self.area)
                self.total_collagen += collagen_ob
                self.collagen_ob_total += collagen_ob
            if alp_ob > 0.0:
                self.alp += alp_ob / max(1.0, self.area)
                self.total_alp += alp_ob
                self.alp_ob_total += alp_ob

    def set_ob_mask(self, ob_mask: np.ndarray | None) -> None:
        if ob_mask is None:
            self._ob_mask = None
            return
        m = np.asarray(ob_mask, dtype=bool)
        if m.shape != (self.height, self.width):
            raise ValueError(
                f"ob_mask has shape {m.shape}, expected {(self.height, self.width)}"
            )
        self._ob_mask = m

    def set_ob_commit_hours(self, ob_commit_hours: np.ndarray | None) -> None:
        if ob_commit_hours is None:
            self._ob_commit_hours = None
            return
        arr = np.asarray(ob_commit_hours, dtype=float)
        if arr.shape != (self.height, self.width):
            raise ValueError(
                f"ob_commit_hours has shape {arr.shape}, expected {(self.height, self.width)}"
            )
        self._ob_commit_hours = arr

    def set_preob_mask(self, preob_mask: np.ndarray | None) -> None:
        if preob_mask is None:
            self._preob_mask = None
            return
        m = np.asarray(preob_mask, dtype=bool)
        if m.shape != (self.height, self.width):
            raise ValueError(
                f"preob_mask has shape {m.shape}, expected {(self.height, self.width)}"
            )
        self._preob_mask = m

    def step_hour(self, hour: int, preob_count: int, ob_count: int) -> None:
        if self.datafit_schedule is not None:
            self._apply_datafit(hour, preob_count, ob_count)
        else:
            self._apply_equation_dynamics(hour)

        self._hour += 1
        if self.media_interval_hours > 0:
            self.hours_since_media += 1

    def snapshot_scalars(self) -> dict:
        return {
            "collagen_total": float(self.total_collagen),
            "alp_total": float(self.total_alp),
            "ha_total": float(self.ha.sum()),
            "collagen_preob_total": float(self.collagen_preob_total),
            "collagen_ob_total": float(self.collagen_ob_total),
            "alp_preob_total": float(self.alp_preob_total),
            "alp_ob_total": float(self.alp_ob_total),
            "mineral_preob_total": float(self.mineral_preob_total),
            "mineral_ob_total": float(self.mineral_ob_total),
        }
