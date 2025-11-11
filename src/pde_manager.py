from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np

__all__ = ["PDEManager"]


@dataclass(slots=True)
class _Supplements:
    aa: float = 1.0
    bgp: float = 1.0
    dexa: float = 1.0
    mk4: float = 1.0


class PDEManager:
    """
    Lightweight grid 'PDE' manager for ions and matrix/mineral fields.

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
    AA_TO_COLLAGEN_RATE: float = 0.2  # Collagen-I increase per AA (scaled by aa)
    # --- nucleation / maturation ---|
    S_THRESH: float = 0.01  # Ca*Pi supersaturation threshold
    K_NUC: float = 0.05  # ACP nucleation rate per day (a.u.)
    K_HA: float = 0.05  # ACP -> HA conversion per day

    # --- stoichiometry ---
    CA_PER_ACP: float = 3.0  # stoichiometric Ca per ACP unit
    PI_PER_ACP: float = 2.0  # stoichiometric Pi per ACP unit
    ACP_TO_HA_THRESHOLD: float = 0.25  # required local ACP before HA maturation
    CA_PER_ACP_TO_HA: float = 1.0 / 3.0
    HA_PER_ACP_RATIO: float = 2.0 / 3.0
    H_PLUS_PER_HA: float = 1.0  # protons released per HA formed
    KW: float = 1.0
    H_BASE: float = 1.0
    OH_BASE: float = 1.0
    # --- supplement depletion ---
    AA_USAGE_PER_COLLAGEN: float = 1e-5  # AA units consumed per collagen unit
    BGP_USAGE_PER_PI: float = 3e-4  # β-GP units consumed per Pi unit released

    def __init__(self, width: int, height: int, media_interval: int, supplements: dict):
        self.width = int(width)
        self.height = int(height)
        self._day = 0
        self.media_interval = int(media_interval)
        self.days_since_media = 0

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
        self.free_ca = 0.0
        self.free_pi = 0.0
        self.total_collagen = 0.0
        self.total_alp = 0.0
        self.ca_consumed_total = 0.0
        self.pi_consumed_total = 0.0

        # Last-day OB mask (RAW, optional)
        self._ob_mask: Optional[np.ndarray] = None

        # initialize baseline media so day 0 snapshots show CA0 / PI0
        self.refresh_media()

    # ---------- helpers ----------
    # ---------- public API ----------
    def refresh_media(self) -> None:
        """Reset ionic fields to baseline; keep matrix/minerals."""
        self.ca.fill(self.CA0)
        self.pi.fill(self.PI0)
        # legacy scalars
        self.free_ca = float(self.CA0)
        self.free_pi = float(self.PI0)
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

    def add_collagen(self, amount: float, ob_count: int = 0) -> None:
        """
        Accumulate collagen; AA availability scales how much matrix OBs can lay down.
        If the AA supplement is zero, collagen deposition is stalled.
        """
        aa_factor = max(0.0, self.aa_pool)
        if aa_factor <= 0.0:
            return
        amount = float(max(0.0, amount)) * aa_factor * self.AA_TO_COLLAGEN_RATE
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

        # 1) ALP raises Pi (scaled by β-GP)
        if self.total_alp > 0 and self.bgp_pool > 0:
            pi_delta = self.ALP_TO_PI_RATE * self.bgp_pool * self.alp
            self.pi += pi_delta
            self._consume_bgp(float(pi_delta.sum()))

        # 2) ACP nucleation where collagen & supersaturation thresholds met
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
            possible_conversion = np.minimum(available_acp, self.K_HA)
            ca_avail = self.ca[ha_idx]
            ca_capacity = ca_avail / self.CA_PER_ACP_TO_HA
            conversion = np.minimum(possible_conversion, ca_capacity)
            if conversion.any():
                ca_consumed = conversion * self.CA_PER_ACP_TO_HA
                ha_created = conversion * self.HA_PER_ACP_RATIO
                self.ca[ha_idx] = np.clip(self.ca[ha_idx] - ca_consumed, 0.0, None)
                self.acp[ha_idx] -= conversion
                self.ha[ha_idx] += ha_created
                self.h_plus[ha_idx] += ha_created * self.H_PLUS_PER_HA
                self.ca_consumed_total += float(ca_consumed.sum())

        # 5) maintain simple OH/H+ equilibrium
        self.oh_minus = np.clip(self.KW / (self.h_plus + 1e-12), 0.0, None)

        # 6) legacy scalars and scheduling
        self.free_ca = float(self.ca.mean())
        self.free_pi = float(self.pi.mean())

        self._day += 1
        if self.media_interval > 0:
            self.days_since_media += 1

    # ---------- reporting ----------
    def snapshot_scalars(self) -> dict:
        """Return scalars used by the timeseries writer (back-compat keys kept)."""
        return {
            "free_ca": self.free_ca,
            "free_pi": self.free_pi,
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
