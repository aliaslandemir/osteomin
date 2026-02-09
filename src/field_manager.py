from __future__ import annotations
from typing import List, Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from .datafit import DatafitSchedule

__all__ = ["FieldManager"]


class FieldManager:
    """
    Tracks collagen/ALP totals and spatial HA accumulation via datafit input.

    Design notes
    ------------
    - Collagen/ALP accumulate persistently and deposits are distributed over osteoblast-rich regions.
    - Optional nodule detection uses a sliding window over mineralized regions.
    """

    def __init__(
        self,
        width: int,
        height: int,
        media_interval: int,
        enable_nodules: bool = True,
        nodule_window: int = 5,
        base_collagen_rate: float = 0.001,
        datafit_schedule: DatafitSchedule | None = None,
    ):
        self.width = int(width)
        self.height = int(height)
        self.area = float(self.width * self.height)
        self._hour = 0
        self.media_interval_hours = max(0, int(media_interval))
        self.hours_since_media = 0
        self.datafit_schedule = datafit_schedule
        self.BASE_COLLAGEN_RATE = float(base_collagen_rate)
        self.collagen = np.zeros((self.height, self.width), dtype=float)
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

        self._pending_secretion_ob_count: int = 0

        self.enable_nodules = bool(enable_nodules)
        self.nodule_window = int(max(1, nodule_window))
        self._nodule_counts: dict[int, int] = {}
        self._nodule_positions: dict[int, List[Tuple[float, float]]] = {}
        self._latest_nodule_count: int = 0


    def request_collagen_secretion(self, ob_count: int) -> None:
        """Queue the number of OBs that should secrete collagen this day."""
        self._pending_secretion_ob_count = max(0, int(ob_count))

    def _apply_collagen_secretion(self) -> None:
        """Apply per-day collagen deposition from active osteoblasts."""
        count = self._pending_secretion_ob_count
        self._pending_secretion_ob_count = 0
        if count <= 0 or self.BASE_COLLAGEN_RATE <= 0:
            return
        amount = count * self.BASE_COLLAGEN_RATE
        if amount <= 0.0:
            return
        if self._ob_mask is not None and self._ob_mask.any():
            area = int(self._ob_mask.sum())
            self.collagen[self._ob_mask] += amount / max(1, area)
        else:
            self.collagen += amount / (self.width * self.height)
        self.total_collagen += amount

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

    def _apply_datafit(self, hour: int, preob_count: int, ob_count: int) -> None:
        """Apply hourly increments from datafit.npz per site (preOB vs OB)."""
        if self.datafit_schedule is None:
            return
        inc = self.datafit_schedule.hourly_increments(hour)
        pre_mask = self._preob_mask
        ob_mask = self._ob_mask

        collagen_preob = max(0.0, inc.get("collagen_basal", 0.0))
        collagen_ob = max(0.0, inc.get("collagen_osteo_ob", inc.get("collagen_osteo", 0.0)))
        alp_preob = max(0.0, inc.get("alp_basal", 0.0))
        alp_ob = max(0.0, inc.get("alp_osteo_ob", inc.get("alp_osteo", 0.0)))
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
        else:
            if collagen_preob > 0.0:
                self.collagen += collagen_preob / max(1.0, self.area)
                self.total_collagen += collagen_preob
                self.collagen_preob_total += collagen_preob
            if alp_preob > 0.0:
                self.alp += alp_preob / max(1.0, self.area)
                self.total_alp += alp_preob
                self.alp_preob_total += alp_preob

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
        """Provide a boolean mask of current osteoblast locations (HxW)."""
        if ob_mask is None:
            self._ob_mask = None
            return
        m = np.asarray(ob_mask, dtype=bool)
        if m.shape != (self.height, self.width):
            raise ValueError(
                f"ob_mask has shape {m.shape}, expected {(self.height, self.width)}"
            )
        self._ob_mask = m

    def set_preob_mask(self, preob_mask: np.ndarray | None) -> None:
        """Provide a boolean mask of current pre-osteoblast locations (HxW)."""
        if preob_mask is None:
            self._preob_mask = None
            return
        m = np.asarray(preob_mask, dtype=bool)
        if m.shape != (self.height, self.width):
            raise ValueError(
                f"preob_mask has shape {m.shape}, expected {(self.height, self.width)}"
            )
        self._preob_mask = m

    def detect_nodules(
        self, day: int, ob_mask: np.ndarray
    ) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Scan the grid with a window of size `nodule_window` and record nodules that meet:
          - every site is occupied by an osteoblast
          - each site contains mineral (HA > 0) so the window is mineralized.
        """
        if not self.enable_nodules:
            self._latest_nodule_count = 0
            return 0, []

        mask = np.asarray(ob_mask, dtype=bool)
        if mask.shape != (self.height, self.width):
            raise ValueError(
                f"ob_mask has shape {mask.shape}, expected {(self.height, self.width)}"
            )
        win = self.nodule_window
        h, w = mask.shape
        positions: List[Tuple[float, float]] = []
        if h >= win and w >= win:
            ob_view = sliding_window_view(mask.astype(np.uint8), (win, win))
            ob_counts = ob_view.sum(axis=(-2, -1))
            ha_view = sliding_window_view(self.ha, (win, win))
            min_ha = ha_view.min(axis=(-2, -1))
            required_ob = win * win
            hits = (ob_counts == required_ob) & (min_ha > 0.0)
            if hits.any():
                positions = self._cluster_nodule_hits(hits)

        count = len(positions)
        self._nodule_counts[day] = count
        self._nodule_positions[day] = positions
        self._latest_nodule_count = count
        return count, positions

    def _cluster_nodule_hits(self, hits: np.ndarray) -> List[Tuple[float, float]]:
        """
        Merge overlapping positive windows (8-neighbour connectivity) and return
        the approximate center of each nodule as (x, y) floats.
        """
        win = self.nodule_window
        visited = np.zeros_like(hits, dtype=bool)
        positions: List[Tuple[float, float]] = []
        offsets = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
        for y in range(hits.shape[0]):
            for x in range(hits.shape[1]):
                if not hits[y, x] or visited[y, x]:
                    continue
                queue = [(y, x)]
                visited[y, x] = True
                members: List[Tuple[int, int]] = []
                while queue:
                    cy, cx = queue.pop()
                    members.append((cy, cx))
                    for dy, dx in offsets:
                        ny = cy + dy
                        nx = cx + dx
                        if (
                            0 <= ny < hits.shape[0]
                            and 0 <= nx < hits.shape[1]
                            and hits[ny, nx]
                            and not visited[ny, nx]
                        ):
                            visited[ny, nx] = True
                            queue.append((ny, nx))
                if not members:
                    continue
                avg_top = sum(m[0] for m in members) / len(members)
                avg_left = sum(m[1] for m in members) / len(members)
                center_y = avg_top + win / 2 - 0.5
                center_x = avg_left + win / 2 - 0.5
                positions.append((center_x, center_y))
        return positions

    def nodule_data_for_day(self, day: int) -> Tuple[int, List[Tuple[float, float]]]:
        count = self._nodule_counts.get(day, 0)
        positions = self._nodule_positions.get(day, [])
        return count, positions

    def step_hour(self, hour: int, preob_count: int, ob_count: int) -> None:
        """
        Update fields for a single hour.
        """
        if self.datafit_schedule is not None:
            self._apply_datafit(hour, preob_count, ob_count)
        else:
            self._apply_collagen_secretion()

        self._hour += 1
        if self.media_interval_hours > 0:
            self.hours_since_media += 1

    def snapshot_scalars(self) -> dict:
        """Return scalars used by the timeseries writer (back-compat keys kept)."""
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
            "nodule_count": int(self._latest_nodule_count),
        }
