from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view

__all__ = ["FieldManager"]


@dataclass(slots=True)
class _Supplements:
    aa: float = 1.0
    bgp: float = 1.0
    mk4: float = 0.0


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

    def __init__(
        self,
        width: int,
        height: int,
        media_interval: int,
        supplements: dict,
        enable_nucleation: bool = True,
        enable_nodules: bool = True,
        s_thresh: float = 0.01,
        k_nuc: float = 0.08,
        k_ha: float = 0.04,
        nodule_window: int = 5,
        ca0: float = 1.0,
        pi0: float = 0.0,
        alp_to_pi_rate: float = 0.01,
        ca_per_acp: float = 3.0,
        pi_per_acp: float = 2.0,
        acp_to_ha_threshold: float = 0.20,
        collagen_nucleation_threshold: float = 1e-6,
        acp_per_ha_reaction: float = 3.0,
        ca_per_reaction: float = 1.0,
        ha_per_reaction: float = 2.0,
        h_plus_per_ha: float = 1.0,
        kw: float = 1.0,
        h_base: float = 1.0,
        oh_base: float = 1.0,
        aa_usage_per_collagen: float = 0.05,
        bgp_usage_per_pi: float = 3e-4,
        base_collagen_rate: float = 0.001,
        k_aa: float = 0.2,
    ):
        self.width = int(width)
        self.height = int(height)
        self.area = float(self.width * self.height)
        self._hour = 0
        self.media_interval_hours = max(0, int(media_interval))
        self.hours_since_media = 0
        self.enable_nucleation = bool(enable_nucleation)
        self.ca_supplied_total = 0.0
        self.pi_supplied_total = 0.0
        self.pi_generated_total = 0.0
        self.bgp_consumed_total = 0.0

        supp_kwargs = {k: float(v) for k, v in dict(supplements).items() if k in ("aa", "bgp", "mk4")}
        s = _Supplements(**supp_kwargs)
        self.supp = s
        self.aa_pool = max(0.0, s.aa)
        self.bgp_pool = max(0.0, s.bgp)
        self.mk4_level = max(0.0, s.mk4)
        self.CA0 = float(ca0)
        self.PI0 = float(pi0)
        self.ALP_TO_PI_RATE = float(alp_to_pi_rate)
        self.S_THRESH = float(s_thresh)
        self.K_NUC = float(k_nuc)  # per hour
        self.K_HA = float(k_ha)  # per hour
        self.CA_PER_ACP = float(ca_per_acp)
        self.PI_PER_ACP = float(pi_per_acp)
        self.ACP_TO_HA_THRESHOLD = float(acp_to_ha_threshold)
        self.COLLAGEN_NUCLEATION_THRESHOLD = float(collagen_nucleation_threshold)
        self.ACP_PER_HA_REACTION = float(acp_per_ha_reaction)
        self.CA_PER_REACTION = float(ca_per_reaction)
        self.HA_PER_REACTION = float(ha_per_reaction)
        self.H_PLUS_PER_HA = float(h_plus_per_ha)
        self.KW = float(kw)
        self.H_BASE = float(h_base)
        self.OH_BASE = float(oh_base)
        self.AA_USAGE_PER_COLLAGEN = float(aa_usage_per_collagen)
        self.BGP_USAGE_PER_PI = float(bgp_usage_per_pi)
        self.BASE_COLLAGEN_RATE = float(base_collagen_rate)
        self.K_AA = float(k_aa)

        self.ca = np.zeros((self.height, self.width), dtype=float)
        self.pi = np.zeros((self.height, self.width), dtype=float)
        self.collagen = np.zeros((self.height, self.width), dtype=float)
        self.alp = np.zeros((self.height, self.width), dtype=float)
        self.acp = np.zeros((self.height, self.width), dtype=float)
        self.ha = np.zeros((self.height, self.width), dtype=float)
        self.h_plus = np.full((self.height, self.width), self.H_BASE, dtype=float)
        self.oh_minus = np.full((self.height, self.width), self.OH_BASE, dtype=float)

        self.cell_ca = 0.0
        self.cell_pi = 0.0
        self.total_collagen = 0.0
        self.total_alp = 0.0
        self.ca_consumed_total = 0.0
        self.pi_consumed_total = 0.0

        self._ob_mask: Optional[np.ndarray] = None

        self._pending_secretion_ob_count: int = 0

        self.enable_nodules = bool(enable_nodules)
        self.nodule_window = int(max(1, nodule_window))
        self._nodule_counts: dict[int, int] = {}
        self._nodule_positions: dict[int, List[Tuple[float, float]]] = {}
        self._latest_nodule_count: int = 0

        self.refresh_media()

    def refresh_media(self) -> None:
        """Reset ionic fields to baseline; keep matrix/minerals."""
        self.ca.fill(self.CA0)
        self.pi.fill(self.PI0)
        self.ca_supplied_total += float(self.ca.sum())
        self.pi_supplied_total += float(self.pi.sum())
        self.cell_ca = 0.0
        self.cell_pi = 0.0
        self.h_plus.fill(self.H_BASE)
        self.oh_minus.fill(self.OH_BASE)
        self.aa_pool = max(0.0, self.supp.aa)
        self.bgp_pool = max(0.0, self.supp.bgp)
        self.hours_since_media = 0

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
        if self.mk4_level <= 0.0:
            f_mk4 = 1.0
        elif self.mk4_level < 1.0:
            f_mk4 = 1.3
        else:
            f_mk4 = 1.6
        amount = count * self.BASE_COLLAGEN_RATE * f_mk4 * f_aa
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

    def detect_nodules(
        self, day: int, ob_mask: np.ndarray
    ) -> Tuple[int, List[Tuple[float, float]]]:
        """
        Scan the grid with a window of size `nodule_window` and record nodules that meet:
          - every site is occupied by an osteoblast
          - each site has HA above the ACP→HA maturation threshold.
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
            hits = (ob_counts == required_ob) & (min_ha >= self.ACP_TO_HA_THRESHOLD)
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

    def step_hour(self) -> None:
        """
        Update ions from secreted ALP/collagen and perform nucleation/maturation.
        Media changes occur once the configured hour interval has elapsed, so
        snapshots capture the depleted state at the end of each day (24 hours).
        """
        if self.media_interval_hours > 0 and self.hours_since_media >= self.media_interval_hours:
            self.refresh_media()

        self._produce_pi_from_alp()

        self._apply_collagen_secretion()

        if self.enable_nucleation:
            if self.mk4_level <= 0.0:
                f_mk4 = 1.0
            elif self.mk4_level < 1.0:
                f_mk4 = 1.3
            else:
                f_mk4 = 1.6
            k_nuc_eff = self.K_NUC * f_mk4
            k_ha_eff = self.K_HA * f_mk4

            supersat = self.ca * self.pi
            if self._ob_mask is not None:
                region = self._ob_mask
            else:
                region = np.ones_like(self.ca, dtype=bool)

            eligible = region & (supersat >= self.S_THRESH)
            if eligible.any():
                eligible_idx = np.where(eligible)
                ca_avail = self.ca[eligible_idx]
                pi_avail = self.pi[eligible_idx]
                max_ca = ca_avail / self.CA_PER_ACP
                max_pi = pi_avail / self.PI_PER_ACP
                max_by_resources = np.minimum(max_ca, max_pi)
                formation = np.minimum(k_nuc_eff, max_by_resources)
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

            ha_candidates = self.acp >= self.ACP_TO_HA_THRESHOLD
            if self.COLLAGEN_NUCLEATION_THRESHOLD > 0.0:
                ha_candidates &= self.collagen >= self.COLLAGEN_NUCLEATION_THRESHOLD
            if ha_candidates.any():
                ha_idx = np.where(ha_candidates)
                available_acp = self.acp[ha_idx]
                acp_lumps = available_acp / self.ACP_PER_HA_REACTION
                possible_conversion = np.minimum(acp_lumps, k_ha_eff)
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

        self.oh_minus = np.clip(self.KW / (self.h_plus + 1e-12), 0.0, None)

        self.cell_ca = float(self.ca_consumed_total) / max(1.0, self.area)
        self.cell_pi = float(self.pi.mean())

        self._hour += 1
        if self.media_interval_hours > 0:
            self.hours_since_media += 1

    def snapshot_scalars(self) -> dict:
        """Return scalars used by the timeseries writer (back-compat keys kept)."""
        return {
            "cell_ca": float(self.cell_ca),
            "cell_pi": float(self.cell_pi),
            "mk4_level": float(getattr(self, "mk4_level", 0.0)),
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
            "acp_total": float(self.acp.sum()),
            "ha_total": float(self.ha.sum()),
            "h_plus_total": float(self.h_plus.sum()),
            "oh_minus_total": float(self.oh_minus.sum()),
            "ca_used_total": float(self.ca_consumed_total),
            "pi_used_total": float(self.pi_consumed_total),
            "nodule_count": int(self._latest_nodule_count),
        }
