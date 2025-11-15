from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

__all__ = ["FieldManager"]


@dataclass(slots=True)
class _Supplements:
    aa: float = 1.0
    bgp: float = 1.0
    dexa: float = 1.0
    mk4: float = 1.0


class FieldManager:
    """
    Lightweight grid field manager for ions and matrix/mineral fields.

    Preserves scalar behaviours (free_ca/free_pi/collagen_total/alp_total)
    while adding spatial ACP/HA nucleation and maturation.

    Design notes
    ------------
    - Media refresh resets Ca/Pi uniformly.
    - Collagen/ALP accumulate persistently (not reset by media).
    - If an osteoblast mask is provided via set_ob_mask, secretion is deposited
      over that (RAW) mask and nucleation is restricted to the same pixels.
    - ACP nucleates where the Ca*Pi supersaturation exceeds `S_THRESH`;
      ACP matures to HA once the accumulated mass passes `ACP_TO_HA_THRESHOLD`.
    """

    # --- simple chemistry params ---
    CA0: float = 1.0  # media Ca baseline (a.u.)
    PI0: float = 0.0  # media Pi baseline (a.u.); Pi is made via β-GP → ALP
    ALP_TO_PI_RATE: float = 0.01  # Pi increase per ALP (scaled by bgp)
    # --- nucleation / maturation ---|
    S_THRESH: float = 0.01  # Ca*Pi supersaturation threshold
    K_NUC: float = 0.001  # ACP nucleation rate per day (a.u.)
    K_HA: float = 0.06  # ACP -> HA conversion per day

    # --- stoichiometry ---
    CA_PER_ACP: float = 3.0  # stoichiometric Ca per ACP unit
    PI_PER_ACP: float = 2.0  # stoichiometric Pi per ACP unit
    ACP_TO_HA_THRESHOLD: float = 0.20  # required local ACP before HA maturation
    # stoichiometry for the ACP → HA reaction:
    #   3 ACP + 1 Ca²⁺ → 2 HA + 2 H⁺
    ACP_PER_HA_REACTION: float = 3.0
    CA_PER_REACTION: float = 1.0
    HA_PER_REACTION: float = 2.0
    H_PLUS_PER_HA: float = 1.0  # protons released per HA formed
    KW: float = 1.0
    H_BASE: float = 1.0
    OH_BASE: float = 1.0
    # --- supplement depletion ---
    AA_USAGE_PER_COLLAGEN: float = 0.05  # AA units consumed per collagen unit
    BGP_USAGE_PER_PI: float = 3e-4  # β-GP units consumed per Pi unit released
    # --- collagen kinetics ---
    BASE_COLLAGEN_RATE: float = 0.001  # collagen units per OB per day at saturating AA
    K_AA: float = 0.2  # AA level for half-maximal collagen effect

    def __init__(
        self,
        width: int,
        height: int,
        media_interval: int,
        supplements: dict,
        enable_nucleation: bool = True,
    ):
        self.width = int(width)
        self.height = int(height)
        self.area = float(self.width * self.height)
        self._day = 0
        self.media_interval = int(media_interval)
        self.days_since_media = 0
        self.enable_nucleation = bool(enable_nucleation)
        self.ca_supplied_total = 0.0
        self.pi_supplied_total = 0.0
        self.pi_generated_total = 0.0
        self.bgp_consumed_total = 0.0

        # Supplements (clamped to non-negative)
        s = _Supplements(**{k: float(v) for k, v in dict(supplements).items()})
        self.supp = s
        self.aa_pool = max(0.0, s.aa)
        self.bgp_pool = max(0.0, s.bgp)
        self.mk4_level = max(0.0, s.mk4)
        self.dexa_level = max(0.0, s.dexa)

        # Spatial fields
        self.ca = np.zeros((self.height, self.width), dtype=float)
        self.pi = np.zeros((self.height, self.width), dtype=float)
        self.collagen = np.zeros((self.height, self.width), dtype=float)
        self.alp = np.zeros((self.height, self.width), dtype=float)
        self.acp = np.zeros((self.height, self.width), dtype=float)
        self.ha = np.zeros((self.height, self.width), dtype=float)
        self.h_plus = np.full((self.height, self.width), self.H_BASE, dtype=float)
        self.oh_minus = np.full((self.height, self.width), self.OH_BASE, dtype=float)

        # Legacy scalars (for outputs/back-compat)
        self.cell_ca = 0.0
        self.cell_pi = 0.0
        self.total_collagen = 0.0
        self.total_alp = 0.0
        self.ca_consumed_total = 0.0
        self.pi_consumed_total = 0.0

        # Last-day OB mask (RAW, optional)
        self._ob_mask: Optional[np.ndarray] = None

        # pending secretion request (set by ABM prior to chemistry update)
        self._pending_secretion_ob_count: int = 0

        # initialize baseline media so day 0 snapshots show CA0 / PI0
        self.refresh_media()

    # ---------- helpers ----------
    # ---------- public API ----------
    def refresh_media(self) -> None:
        """Reset ionic fields to baseline; keep matrix/minerals."""
        self.ca.fill(self.CA0)
        self.pi.fill(self.PI0)
        self.ca_supplied_total += float(self.ca.sum())
        self.pi_supplied_total += float(self.pi.sum())
        # legacy scalars
        self.cell_ca = 0.0
        self.cell_pi = 0.0
        self.h_plus.fill(self.H_BASE)
        self.oh_minus.fill(self.OH_BASE)
        self.aa_pool = max(0.0, self.supp.aa)
        self.bgp_pool = max(0.0, self.supp.bgp)
        self.days_since_media = 0

    def _consume_aa(self, collagen_added: float) -> None:
        if collagen_added <= 0.0:
            return
        usage = collagen_added * self.AA_USAGE_PER_COLLAGEN
        self.aa_pool = max(0.0, self.aa_pool - usage)

    def _consume_bgp(self, pi_added_total: float) -> None:
        if pi_added_total <= 0.0:
            return
        usage = pi_added_total * self.BGP_USAGE_PER_PI
        self.bgp_pool = max(0.0, self.bgp_pool - usage)
        self.bgp_consumed_total += usage

    def _produce_pi_from_alp(self) -> float:
        """Convert the ALP field to Pi without exceeding β-GP supply."""
        if self.total_alp <= 0.0 or self.bgp_pool <= 0.0:
            return 0.0
        potential_pi = self.ALP_TO_PI_RATE * self.alp
        total_potential = float(potential_pi.sum())
        if total_potential <= 0.0:
            return 0.0
        max_pi = self.bgp_pool / max(1e-12, self.BGP_USAGE_PER_PI)
        if max_pi <= 0.0:
            return 0.0
        scale = min(1.0, max_pi / total_potential)
        pi_add = potential_pi * scale
        self.pi += pi_add
        produced = float(pi_add.sum())
        self._consume_bgp(produced)
        self.pi_generated_total += produced
        return produced

    def request_collagen_secretion(self, ob_count: int) -> None:
        """Queue the number of OBs that should secrete collagen this day."""
        self._pending_secretion_ob_count = max(0, int(ob_count))

    def _apply_collagen_secretion(self) -> None:
        """Apply per-day AA→collagen using a saturating sigmoidal response."""
        count = self._pending_secretion_ob_count
        self._pending_secretion_ob_count = 0
        if count <= 0 or self.BASE_COLLAGEN_RATE <= 0:
            return
        denom = self.aa_pool + self.K_AA
        if denom <= 0.0:
            return
        f_aa = self.aa_pool / denom
        if f_aa <= 0.0:
            return
        amount = count * self.BASE_COLLAGEN_RATE * f_aa
        if amount <= 0.0:
            return
        if self._ob_mask is not None and self._ob_mask.any():
            area = int(self._ob_mask.sum())
            self.collagen[self._ob_mask] += amount / max(1, area)
        else:
            self.collagen += amount / (self.width * self.height)
        self.total_collagen += amount
        self._consume_aa(amount)

    def add_alp(self, amount: float, ob_count: int = 0) -> None:
        """Accumulate ALP; if OB mask known, distribute locally on RAW OB pixels."""
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
        """Provide a boolean mask of current osteoblast locations (H×W)."""
        if ob_mask is None:
            self._ob_mask = None
            return
        m = np.asarray(ob_mask, dtype=bool)
        if m.shape != (self.height, self.width):
            raise ValueError(
                f"ob_mask has shape {m.shape}, expected {(self.height, self.width)}"
            )
        self._ob_mask = m  # store RAW (no dilation here)

    # ---------- daily chemistry ----------
    def step_day(self) -> None:
        """
        Update ions from secreted ALP/collagen and perform nucleation/maturation.
        Media changes occur at the start of a day once the interval elapses so the
        previous day's snapshot captures the depleted state.
        """
        if self.media_interval > 0 and self.days_since_media >= self.media_interval:
            self.refresh_media()

        # 1) ALP raises Pi (limited by β-GP)
        self._produce_pi_from_alp()

        # 2) Collagen deposition from osteoblasts (AA-limited)
        self._apply_collagen_secretion()

        if self.enable_nucleation:
            # 4) ACP nucleation where collagen & supersaturation thresholds met
            supersat = self.ca * self.pi
            if self._ob_mask is not None:
                # nucleation limited to osteoblast positions (no dilation)
                region = self._ob_mask
            else:
                # fallback: full grid
                region = np.ones_like(self.ca, dtype=bool)

            eligible = region & (supersat >= self.S_THRESH)
            if eligible.any():
                eligible_idx = np.where(eligible)
                ca_avail = self.ca[eligible_idx]
                pi_avail = self.pi[eligible_idx]
                max_ca = ca_avail / self.CA_PER_ACP
                max_pi = pi_avail / self.PI_PER_ACP
                max_by_resources = np.minimum(max_ca, max_pi)
                formation = np.minimum(self.K_NUC, max_by_resources)
                ca_needed = formation * self.CA_PER_ACP
                pi_needed = formation * self.PI_PER_ACP
                self.ca[eligible_idx] = np.clip(
                    self.ca[eligible_idx] - ca_needed, 0.0, None
                )
                self.pi[eligible_idx] = np.clip(
                    self.pi[eligible_idx] - pi_needed, 0.0, None
                )
                self.acp[eligible_idx] += formation
                self.ca_consumed_total += float(ca_needed.sum())
                self.pi_consumed_total += float(pi_needed.sum())

            # 4) ACP -> HA maturation once enough ACP is present and Ca is available
            ha_candidates = self.acp >= self.ACP_TO_HA_THRESHOLD
            if ha_candidates.any():
                ha_idx = np.where(ha_candidates)
                available_acp = self.acp[ha_idx]
                acp_lumps = available_acp / self.ACP_PER_HA_REACTION
                possible_conversion = np.minimum(acp_lumps, self.K_HA)
                ca_avail = self.ca[ha_idx]
                ca_capacity = ca_avail / self.CA_PER_REACTION
                conversion = np.minimum(possible_conversion, ca_capacity)
                if conversion.any():
                    acp_consumed = conversion * self.ACP_PER_HA_REACTION
                    ca_consumed = conversion * self.CA_PER_REACTION
                    ha_created = conversion * self.HA_PER_REACTION
                    self.ca[ha_idx] = np.clip(self.ca[ha_idx] - ca_consumed, 0.0, None)
                    self.acp[ha_idx] -= acp_consumed
                    self.ha[ha_idx] += ha_created
                    self.h_plus[ha_idx] += ha_created * self.H_PLUS_PER_HA
                    self.ca_consumed_total += float(ca_consumed.sum())

        # 5) maintain simple OH/H+ equilibrium
        self.oh_minus = np.clip(self.KW / (self.h_plus + 1e-12), 0.0, None)

        # 6) legacy scalars and scheduling
        self.cell_ca = float(self.ca_consumed_total) / max(1.0, self.area)
        self.cell_pi = float(self.pi.mean())

        self._day += 1
        if self.media_interval > 0:
            self.days_since_media += 1

    # ---------- reporting ----------
    def snapshot_scalars(self) -> dict:
        """Return scalars used by the timeseries writer (back-compat keys kept)."""
        return {
            "cell_ca": float(self.cell_ca),
            "cell_pi": float(self.cell_pi),
            "ca_media": float(self.ca.mean()),
            "bgp_pool": float(self.bgp_pool),
            "ca_free_total": float(self.ca.sum()),
            "pi_free_total": float(self.pi.sum()),
            "ca_supplied_total": float(self.ca_supplied_total),
            "pi_supplied_total": float(self.pi_supplied_total),
            "pi_generated_total": float(self.pi_generated_total),
            "bgp_consumed_total": float(self.bgp_consumed_total),
            "collagen_total": float(self.total_collagen),
            "alp_total": float(self.total_alp),
            # new mineral outputs
            "acp_total": float(self.acp.sum()),
            "ha_total": float(self.ha.sum()),
            "h_plus_total": float(self.h_plus.sum()),
            "oh_minus_total": float(self.oh_minus.sum()),
            "ca_used_total": float(self.ca_consumed_total),
            "pi_used_total": float(self.pi_consumed_total),
        }
