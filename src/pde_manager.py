from __future__ import annotations
from dataclasses import dataclass

__all__ = ["PDEManager"]

@dataclass(slots=True)
class _Supplements:
    aa: float = 1.0
    bgp: float = 1.0
    dexa: float = 1.0
    mk4: float = 1.0

class PDEManager:
    """Scalar 'PDE' accumulator with media refresh + simple clamps."""
    def __init__(self, width: int, height: int, media_interval: int, supplements: dict):
        self.width = int(width); self.height = int(height)
        self.media_interval = int(media_interval)
        self.supp = _Supplements(**supplements)

        self.free_ca = 1.0
        self.free_pi = 1.0
        self.total_collagen = 0.0
        self.total_alp = 0.0

        self.ca_max = 2.0
        self.pi_max = 2.0
        self._day = 0

    def add_collagen(self, amount: float) -> None:
        self.total_collagen += max(0.0, float(amount))

    def add_alp(self, amount: float) -> None:
        self.total_alp += max(0.0, float(amount))

    def refresh_media(self) -> None:
        self.free_ca = 1.0
        self.free_pi = 1.0

    def step_day(self) -> None:
        # ALP liberates Pi from β-GP; Dexa nudges up; collagen sinks both; MK-4 small Pi sink
        self.free_pi += 0.02 * self.total_alp * self.supp.bgp
        self.free_pi += 0.005 * self.supp.dexa
        sink = 0.01 * self.total_collagen * (0.5 * self.supp.aa + 0.5)
        self.free_ca = max(0.0, self.free_ca - sink)
        self.free_pi = max(0.0, self.free_pi - sink)
        self.free_pi = max(0.0, self.free_pi - 0.002 * self.supp.mk4)

        # clamps
        self.free_ca = min(self.free_ca, self.ca_max)
        self.free_pi = min(self.free_pi, self.pi_max)

        # tick + media
        self._day += 1
        if (self._day % self.media_interval) == 0:
            self.refresh_media()

    def snapshot_scalars(self) -> dict:
        return {
            "free_ca": self.free_ca,
            "free_pi": self.free_pi,
            "collagen_total": self.total_collagen,
            "alp_total": self.total_alp,
        }
