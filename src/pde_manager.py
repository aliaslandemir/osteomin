from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Dict


__all__ = ["PDEManager"]


@dataclass(slots=True)
class _Supplements:
    aa: float = 1.0     # ascorbic acid (unitless scaling)
    bgp: float = 1.0    # beta-glycerophosphate (unitless scaling)
    dexa: float = 1.0   # dexamethasone (unitless scaling)
    mk4: float = 1.0    # menaquinone-4 (unitless scaling)


class PDEManager:
    """
    Lightweight placeholder for media & matrix bookkeeping.
    No spatial diffusion here; acts as a set of 'sinks/sources' for ABM coupling.
    """
    def __init__(self, width: int, height: int, media_interval: int,
                 supplements: Optional[Dict[str, float]] = None) -> None:
        self.width = width
        self.height = height
        self.media_interval = max(1, int(media_interval))

        # Supplements (scalars 0..∞), refreshed with media.
        s = supplements or {}
        self.supp = _Supplements(
            aa=float(s.get("aa", 1.0)),
            bgp=float(s.get("bgp", 1.0)),
            dexa=float(s.get("dexa", 1.0)),
            mk4=float(s.get("mk4", 1.0)),
        )

        # Scalar “fields” (totals) accumulated over time
        self.total_collagen: float = 0.0
        self.total_alp: float = 0.0

        # Ion reservoir proxies refreshed with media changes
        self.free_ca: float = 1.0
        self.free_pi: float = 1.0

        # Day counter to manage media refresh
        self._day: int = 0

    # ----- ABM → PDE hooks -----
    def add_collagen(self, amount: float) -> None:
        # AA promotes collagen; simple multiplicative effect
        self.total_collagen += amount * max(0.0, self.supp.aa)

    def add_alp(self, amount: float) -> None:
        # ALP proxy; BGP is substrate source; MK4 can scale maturation effectiveness
        self.total_alp += amount * max(0.0, (self.supp.bgp * (0.5 + 0.5 * self.supp.mk4)))

    # ----- Media handling & per-day bookkeeping -----
    def refresh_media(self) -> None:
        # Reset basic ion proxies and re-apply supplement baselines
        self.free_ca = 1.0
        self.free_pi = 1.0
        # (Supplements themselves are considered constant scalars here.)

    def step_day(self) -> None:
        # Very light pseudo-kinetics: ALP increases effective Pi from BGP each day
        self.free_pi += 0.05 * self.total_alp
        # Collagen scaffolding mildly reduces free ions (nucleation proxy)
        sink = 0.01 * self.total_collagen
        self.free_ca = max(0.0, self.free_ca - sink)
        self.free_pi = max(0.0, self.free_pi - sink)

        self._day += 1
        if (self._day % self.media_interval) == 0:
            self.refresh_media()

    # Convenience snapshot
    def snapshot_scalars(self) -> dict:
        return {
            "free_ca": self.free_ca,
            "free_pi": self.free_pi,
            "collagen_total": self.total_collagen,
            "alp_total": self.total_alp,
        }
