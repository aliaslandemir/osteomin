from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

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
        dex_ref_conc: float = 20.0,
        mk4_ref_conc: float = 0.6,
        alp_basal_rate: float = 0.0,
        alp_osteo_rate: float = 0.0,
        alp_initial: float = 0.0,
        alp_dex_ref_conc: float = 20.0,
        alp_mk4_ref_conc: float = 0.6,
        alp_a_d: float = 0.0,
        alp_b_d: float = 0.0,
        alp_a_k: float = 0.0,
        alp_b_k: float = 0.0,
        collagen_basal_rate: float = 0.0,
        collagen_osteo_rate: float = 0.0,
        collagen_dex_ref_conc: float = 20.0,
        collagen_mk4_ref_conc: float = 8.0,
        collagen_a_d: float = 0.0,
        collagen_b_d: float = 0.0,
        collagen_a_k: float = 0.0,
        collagen_b_k: float = 0.0,
        aa_initial: float = 1.0,
        aa_half_life_hours: float = 6.0,
        collagen_aa_half_sat: float = 1.0,
        ha_model: str = "minimal",
        ha_initial: float = 0.0,
        ha_osteo_rate: float = 0.0,
        ha_dex_ref_conc: float = 20.0,
        ha_mk4_ref_conc: float = 0.7,
        ha_a_d: float = 0.0,
        ha_b_d: float = 0.0,
        ha_a_k: float = 0.0,
        ha_b_k: float = 0.0,
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
        self.dex_ref_conc = max(1e-12, float(dex_ref_conc))
        self.mk4_ref_conc = max(1e-12, float(mk4_ref_conc))
        self.alp_basal_rate = float(alp_basal_rate)
        self.alp_osteo_rate = float(alp_osteo_rate)
        self.alp_initial = max(0.0, float(alp_initial))
        self.alp_dex_ref_conc = max(1e-12, float(alp_dex_ref_conc))
        self.alp_mk4_ref_conc = max(1e-12, float(alp_mk4_ref_conc))
        self.alp_a_d = float(alp_a_d)
        self.alp_b_d = float(alp_b_d)
        self.alp_a_k = float(alp_a_k)
        self.alp_b_k = float(alp_b_k)
        self.collagen_basal_rate = float(collagen_basal_rate)
        self.collagen_osteo_rate = float(collagen_osteo_rate)
        self.collagen_dex_ref_conc = max(1e-12, float(collagen_dex_ref_conc))
        self.collagen_mk4_ref_conc = max(1e-12, float(collagen_mk4_ref_conc))
        self.collagen_a_d = float(collagen_a_d)
        self.collagen_b_d = float(collagen_b_d)
        self.collagen_a_k = float(collagen_a_k)
        self.collagen_b_k = float(collagen_b_k)
        self.aa_initial = max(0.0, float(aa_initial))
        self.aa_half_life_hours = max(1e-9, float(aa_half_life_hours))
        self.collagen_aa_half_sat = max(1e-9, float(collagen_aa_half_sat))
        self.ha_model = str(ha_model).lower().strip()
        self.ha_initial = max(0.0, float(ha_initial))
        self.ha_osteo_rate = float(ha_osteo_rate)
        self.ha_dex_ref_conc = max(1e-12, float(ha_dex_ref_conc))
        self.ha_mk4_ref_conc = max(1e-12, float(ha_mk4_ref_conc))
        self.ha_a_d = float(ha_a_d)
        self.ha_b_d = float(ha_b_d)
        self.ha_a_k = float(ha_a_k)
        self.ha_b_k = float(ha_b_k)
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
        self.alp = np.full((self.height, self.width), self.alp_initial, dtype=float)
        self.ha = np.full((self.height, self.width), self.ha_initial, dtype=float)

        self.total_collagen = 0.0
        self.total_alp = float(self.alp.sum())
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
        self._modulator_hourly = {
            "alp_base": np.zeros(self.total_hours + 1, dtype=float),
            "alp_dex": np.zeros(self.total_hours + 1, dtype=float),
            "alp_mk4": np.zeros(self.total_hours + 1, dtype=float),
            "alp_interaction": np.zeros(self.total_hours + 1, dtype=float),
            "collagen_base": np.zeros(self.total_hours + 1, dtype=float),
            "collagen_dex": np.zeros(self.total_hours + 1, dtype=float),
            "collagen_mk4": np.zeros(self.total_hours + 1, dtype=float),
            "collagen_interaction": np.zeros(self.total_hours + 1, dtype=float),
            "ha_base": np.zeros(self.total_hours + 1, dtype=float),
            "ha_dex": np.zeros(self.total_hours + 1, dtype=float),
            "ha_mk4": np.zeros(self.total_hours + 1, dtype=float),
            "ha_interaction": np.zeros(self.total_hours + 1, dtype=float),
        }

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

    def _log_concentration_offset(self, conc: float, ref_conc: float) -> float:
        # Baseline/no-drug runs can use conc <= 0; treat those as neutral offset.
        if conc <= 0.0:
            return 0.0
        return float(np.log10(conc / ref_conc))

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

    def _record_modulator_hour(
        self,
        hour: int,
        prefix: str,
        *,
        base: float = 0.0,
        dex: float = 0.0,
        mk4: float = 0.0,
        interaction: float = 0.0,
    ) -> None:
        idx = min(max(0, int(hour)), self.total_hours)
        self._modulator_hourly[f"{prefix}_base"][idx] += float(base)
        self._modulator_hourly[f"{prefix}_dex"][idx] += float(dex)
        self._modulator_hourly[f"{prefix}_mk4"][idx] += float(mk4)
        self._modulator_hourly[f"{prefix}_interaction"][idx] += float(interaction)

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
        if pre_mask is None and ob_mask is None:
            live_mask = None
        elif pre_mask is None:
            live_mask = np.asarray(ob_mask, dtype=bool)
        elif ob_mask is None:
            live_mask = np.asarray(pre_mask, dtype=bool)
        else:
            live_mask = np.asarray(pre_mask, dtype=bool) | np.asarray(ob_mask, dtype=bool)
        u_osteo = self._osteo_active(hour)
        i_d = 1.0 if self.dex_conc > 0.0 else 0.0
        i_k = 1.0 if self.mk4_conc > 0.0 else 0.0

        alp_dex_offset = self._log_concentration_offset(
            self.dex_conc, self.alp_dex_ref_conc
        )
        alp_mk4_offset = self._log_concentration_offset(
            self.mk4_conc, self.alp_mk4_ref_conc
        )
        col_dex_offset = self._log_concentration_offset(
            self.dex_conc, self.collagen_dex_ref_conc
        )
        col_mk4_offset = self._log_concentration_offset(
            self.mk4_conc, self.collagen_mk4_ref_conc
        )
        ha_dex_offset = self._log_concentration_offset(
            self.dex_conc, self.ha_dex_ref_conc
        )
        ha_mk4_offset = self._log_concentration_offset(
            self.mk4_conc, self.ha_mk4_ref_conc
        )
        m_d_alp = max(
            0.0, 1.0 + i_d * (self.alp_a_d - self.alp_b_d * alp_dex_offset**2)
        )
        m_k_alp = max(
            0.0, 1.0 + i_k * (self.alp_a_k - self.alp_b_k * alp_mk4_offset**2)
        )
        m_alp = m_d_alp * m_k_alp
        m_d_col = max(
            0.0,
            1.0 + i_d * (self.collagen_a_d - self.collagen_b_d * col_dex_offset**2),
        )
        m_k_col = max(
            0.0,
            1.0 + i_k * (self.collagen_a_k - self.collagen_b_k * col_mk4_offset**2),
        )
        m_col = m_d_col * m_k_col
        m_d_ha = max(0.0, 1.0 + i_d * (self.ha_a_d - self.ha_b_d * ha_dex_offset**2))
        m_k_ha = max(
            0.0, 1.0 + i_k * (self.ha_a_k - self.ha_b_k * ha_mk4_offset**2)
        )
        m_ha = m_d_ha * m_k_ha

        if live_mask is not None and live_mask.any():
            alp_added = self._deposit_uniform_to_mask(
                self.alp,
                live_mask,
                dt * max(0.0, self.alp_basal_rate),
            )
            if alp_added > 0.0:
                self.total_alp += alp_added
                self.alp_preob_total += alp_added

            collagen_added = self._deposit_uniform_to_mask(
                self.collagen,
                live_mask,
                dt * max(0.0, self.collagen_basal_rate),
            )
            if collagen_added > 0.0:
                self.total_collagen += collagen_added
                self.collagen_preob_total += collagen_added

        if ob_mask is None or not ob_mask.any() or u_osteo <= 0.0:
            self._update_collagen_maturation(dt)
            return

        n_ob_sites = int(ob_mask.sum())
        alp_base_amount = dt * self.alp_osteo_rate * n_ob_sites
        alp_dex_amount = dt * self.alp_osteo_rate * (m_d_alp - 1.0) * n_ob_sites
        alp_mk4_amount = dt * self.alp_osteo_rate * (m_k_alp - 1.0) * n_ob_sites
        alp_interaction_amount = (
            dt
            * self.alp_osteo_rate
            * (m_d_alp - 1.0)
            * (m_k_alp - 1.0)
            * n_ob_sites
        )
        self._record_modulator_hour(
            hour,
            "alp",
            base=alp_base_amount,
            dex=alp_dex_amount,
            mk4=alp_mk4_amount,
            interaction=alp_interaction_amount,
        )
        alp_rate_ob = max(
            0.0,
            self.alp_osteo_rate * m_alp,
        )
        alp_added = self._deposit_uniform_to_mask(self.alp, ob_mask, dt * alp_rate_ob)
        if alp_added > 0.0:
            self.total_alp += alp_added
            self.alp_ob_total += alp_added

        aa_t = self._aa_concentration(hour)
        aa_gate = aa_t / (self.collagen_aa_half_sat + aa_t) if aa_t > 0.0 else 0.0
        collagen_base_amount = dt * self.collagen_osteo_rate * aa_gate * n_ob_sites
        collagen_dex_amount = (
            dt * self.collagen_osteo_rate * (m_d_col - 1.0) * aa_gate * n_ob_sites
        )
        collagen_mk4_amount = (
            dt * self.collagen_osteo_rate * (m_k_col - 1.0) * aa_gate * n_ob_sites
        )
        collagen_interaction_amount = (
            dt
            * self.collagen_osteo_rate
            * (m_d_col - 1.0)
            * (m_k_col - 1.0)
            * aa_gate
            * n_ob_sites
        )
        self._record_modulator_hour(
            hour,
            "collagen",
            base=collagen_base_amount,
            dex=collagen_dex_amount,
            mk4=collagen_mk4_amount,
            interaction=collagen_interaction_amount,
        )
        collagen_rate_ob = max(
            0.0,
            self.collagen_osteo_rate * m_col * aa_gate,
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
            self.ha_osteo_rate * m_ha,
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
        gate_sum = float((hill * collagen_gate * valid).sum())
        ha_base_amount = dt * self.ha_osteo_rate * gate_sum
        ha_dex_amount = dt * self.ha_osteo_rate * (m_d_ha - 1.0) * gate_sum
        ha_mk4_amount = dt * self.ha_osteo_rate * (m_k_ha - 1.0) * gate_sum
        ha_interaction_amount = (
            dt
            * self.ha_osteo_rate
            * (m_d_ha - 1.0)
            * (m_k_ha - 1.0)
            * gate_sum
        )
        self._record_modulator_hour(
            hour,
            "ha",
            base=ha_base_amount,
            dex=ha_dex_amount,
            mk4=ha_mk4_amount,
            interaction=ha_interaction_amount,
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

    def modulator_frame(self) -> pd.DataFrame:
        data: dict[str, np.ndarray] = {
            "hour": np.arange(self.total_hours + 1, dtype=int)
        }
        for key, vals in self._modulator_hourly.items():
            vals_arr = np.asarray(vals, dtype=float)
            data[f"{key}_hourly"] = vals_arr.copy()
            data[f"{key}_cum"] = np.cumsum(vals_arr)
        return pd.DataFrame(data)
