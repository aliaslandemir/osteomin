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
      over that (RAW) mask; nucleation may use a dilated neighborhood.
    - ACP nucleates where (collagen >= C_THRESH) and (Ca*Pi >= S_THRESH) within
      the nucleation region; ACP matures to HA over time.
    """
    # --- simple chemistry params ---
    CA0: float = 1.0                  # media Ca baseline (a.u.)
    PI0: float = 1.0                  # media Pi baseline (a.u.)
    CA_MAX: float = 2.0               # soft clamp
    PI_MAX: float = 2.0
    ALP_TO_PI_RATE: float = 0.01      # Pi increase per ALP (scaled by bgp)
    DEXA_PI_NUDGE: float = 0.002      # small Pi nudge by dexa
    COLLAGEN_CA_SINK: float = 0.0005  # Ca sink per collagen (a.u.)
    COLLAGEN_PI_SINK: float = 0.0002  # Pi sink per collagen (a.u.)
    MK4_PI_SINK: float = 0.001        # extra Pi sink scaled by mk4

    # --- nucleation / maturation ---
    C_THRESH: float = 5.0             # collagen threshold (a.u.)
    S_THRESH: float = 0.15            # Ca*Pi supersaturation threshold
    K_NUC: float = 0.02               # ACP nucleation rate per day (a.u.)
    K_HA: float = 0.05                # ACP -> HA conversion per day
    NUC_RADIUS: int = 1               # dilation for NUCLEATION region (NOT secretion)
    CA_PER_NUC: float = 0.02          # Ca consumed per nucleation step
    PI_PER_NUC: float = 0.02          # Pi consumed per nucleation step

    def __init__(self, width: int, height: int, media_interval: int, supplements: dict):
        self.width = int(width); self.height = int(height)
        self.media_interval = int(media_interval)
        self._day = 0

        # Supplements (clamped to non-negative)
        s = _Supplements(**{k: float(v) for k, v in dict(supplements).items()})
        self.supp = s

        # Spatial fields
        self.ca = np.full((self.height, self.width), self.CA0, dtype=float)
        self.pi = np.full((self.height, self.width), self.PI0, dtype=float)
        self.collagen = np.zeros((self.height, self.width), dtype=float)
        self.alp = np.zeros((self.height, self.width), dtype=float)
        self.acp = np.zeros((self.height, self.width), dtype=float)
        self.ha  = np.zeros((self.height, self.width), dtype=float)

        # Legacy scalars (for outputs/back-compat)
        self.free_ca = float(self.CA0)
        self.free_pi = float(self.PI0)
        self.total_collagen = 0.0
        self.total_alp = 0.0

        # Last-day OB mask (RAW, optional)
        self._ob_mask: Optional[np.ndarray] = None

    # ---------- helpers ----------
    def _dilate(self, mask: np.ndarray, r: int) -> np.ndarray:
        """Binary dilation with a square (2r+1) kernel; no SciPy needed."""
        if r <= 0:
            return mask
        H, W = mask.shape
        pad = np.pad(mask.astype(np.uint8), r, mode="constant")
        out = np.zeros_like(mask, dtype=np.uint8)
        for y in range(H):
            for x in range(W):
                if pad[y:y+2*r+1, x:x+2*r+1].max():
                    out[y, x] = 1
        return out.astype(bool)

    # ---------- public API ----------
    def refresh_media(self) -> None:
        """Reset ionic fields to baseline; keep matrix/minerals."""
        self.ca.fill(self.CA0)
        self.pi.fill(self.PI0)
        # legacy scalars
        self.free_ca = float(self.CA0)
        self.free_pi = float(self.PI0)

    def add_collagen(self, amount: float) -> None:
        """Accumulate collagen; if OB mask known, distribute locally on RAW OB pixels."""
        amount = float(max(0.0, amount))
        if amount <= 0.0:
            return
        if self._ob_mask is not None and self._ob_mask.any():
            area = int(self._ob_mask.sum())
            self.collagen[self._ob_mask] += amount / max(1, area)
        else:
            self.collagen += amount / (self.width * self.height)
        self.total_collagen += amount

    def add_alp(self, amount: float) -> None:
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
            raise ValueError(f"ob_mask has shape {m.shape}, expected {(self.height, self.width)}")
        self._ob_mask = m  # store RAW (no dilation here)

    # ---------- daily chemistry ----------
    def step_day(self) -> None:
        """
        Update ions from secreted ALP/collagen and perform nucleation/maturation,
        then apply media change if scheduled.
        """
        # 1) ALP raises Pi (scaled by β-GP), tiny Dexa nudge
        if self.total_alp > 0:
            self.pi += self.ALP_TO_PI_RATE * self.supp.bgp * self.alp
        self.pi += self.DEXA_PI_NUDGE * self.supp.dexa

        # 2) Collagen acts as a weak sink for Ca/Pi (binding/templating)
        if self.total_collagen > 0:
            self.ca = np.clip(self.ca - self.COLLAGEN_CA_SINK * self.collagen, 0.0, None)
            self.pi = np.clip(
                self.pi - (self.COLLAGEN_PI_SINK + self.MK4_PI_SINK * self.supp.mk4) * self.collagen,
                0.0, None
            )

        # 3) ACP nucleation where collagen & supersaturation thresholds met
        supersat = self.ca * self.pi
        if self._ob_mask is not None:
            # nucleation region: dilated OB neighborhood; secretion stayed on RAW OBs
            region = self._dilate(self._ob_mask, self.NUC_RADIUS)
        else:
            # fallback: anywhere collagen exists
            region = (self.collagen > 0)

        eligible = region & (self.collagen >= self.C_THRESH) & (supersat >= self.S_THRESH)
        if eligible.any():
            # create ACP and consume ions locally
            dACP = np.zeros_like(self.acp)
            dACP[eligible] = self.K_NUC
            self.ca[eligible] = np.clip(self.ca[eligible] - self.CA_PER_NUC, 0.0, None)
            self.pi[eligible] = np.clip(self.pi[eligible] - self.PI_PER_NUC, 0.0, None)
            self.acp += dACP

        # 4) ACP -> HA maturation
        if self.acp.any():
            to_HA = self.K_HA * self.acp
            self.acp -= to_HA
            self.ha  += to_HA

        # 5) legacy scalars and scheduling
        self.free_ca = float(min(self.ca.mean(), self.CA_MAX))
        self.free_pi = float(min(self.pi.mean(), self.PI_MAX))

        self._day += 1
        if (self._day % self.media_interval) == 0:
            self.refresh_media()

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
        }
