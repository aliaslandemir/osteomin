from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np

__all__ = ["DatafitSchedule"]


@dataclass(slots=True)
class DatafitSchedule:
    """
    Lightweight loader that interpolates the bundled datafit.npz curves
    to integer hours and exposes per-hour increments for basal/osteo series.
    """

    path: Path
    total_hours: int
    time_hours: np.ndarray
    increments: Dict[str, np.ndarray]

    @classmethod
    def from_path(cls, path: Path, total_hours: int) -> "DatafitSchedule":
        data = np.load(path)
        required = {
            "time_hours",
            "alp_basal",
            "alp_osteo",
            "collagen_C1",
            "collagen_C2",
            "mineral_C1",
            "mineral_C2",
        }
        missing = sorted(required - set(data.files))
        if missing:
            raise ValueError(f"{path} is missing required keys: {', '.join(missing)}")

        hours = np.arange(max(0, int(total_hours)) + 1, dtype=float)
        src_t = np.asarray(data["time_hours"], dtype=float)

        def _interp(key: str) -> np.ndarray:
            series = np.asarray(data[key], dtype=float)
            return np.interp(hours, src_t, series)

        series = {
            "alp_basal": _interp("alp_basal"),
            "alp_osteo": _interp("alp_osteo"),
            "collagen_basal": _interp("collagen_C1"),
            "collagen_osteo": _interp("collagen_C2"),
            "mineral_basal": _interp("mineral_C1"),
            "mineral_osteo": _interp("mineral_C2"),
        }
        increments = {k: np.diff(v, prepend=v[0]) for k, v in series.items()}
        # ensure hour 0 contributes nothing (diff above may add initial offset)
        for v in increments.values():
            if v.size:
                v[0] = 0.0

        return cls(
            path=Path(path),
            total_hours=int(total_hours),
            time_hours=hours,
            increments=increments,
        )

    def hourly_increments(self, hour: int) -> Dict[str, float]:
        """Return the incremental contributions for the given 1-based hour index."""
        if hour <= 0:
            idx = 0
        else:
            idx = min(int(hour), int(self.total_hours))
        return {k: float(v[idx]) for k, v in self.increments.items()}
