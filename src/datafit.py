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
    def from_path(
        cls,
        path: Path,
        total_hours: int,
        *,
        condition: str | int | None = None,
    ) -> "DatafitSchedule":
        data = np.load(path)
        files = set(data.files)

        if "alp_preob_hourly" in files and "alp_ob_hourly" in files:
            hours = np.arange(max(0, int(total_hours)) + 1, dtype=float)
            src_t = np.asarray(data["time_hours"], dtype=float)

            if condition is None:
                cond_idx = 1
            elif isinstance(condition, int):
                cond_idx = int(condition)
            else:
                cond_labels = None
                if "conditions" in files:
                    cond_labels = [str(x) for x in np.asarray(data["conditions"]).tolist()]
                if cond_labels and condition in cond_labels:
                    cond_idx = cond_labels.index(condition)
                else:
                    raise ValueError(f"Unknown datafit condition '{condition}'")

            def _interp_hourly(arr: np.ndarray) -> np.ndarray:
                arr = np.asarray(arr, dtype=float)
                if arr.ndim != 2:
                    raise ValueError("hourly arrays must be [n_conditions, n_time]")
                if cond_idx < 0 or cond_idx >= arr.shape[0]:
                    raise ValueError(f"condition index {cond_idx} out of range")
                series = np.interp(hours, src_t, arr[cond_idx])
                return series

            alp_preob = _interp_hourly(data["alp_preob_hourly"])
            alp_ob = _interp_hourly(data["alp_ob_hourly"])
            col_preob = _interp_hourly(data["collagen_preob_hourly"])
            col_ob = _interp_hourly(data["collagen_ob_hourly"])
            min_preob = _interp_hourly(data["mineral_preob_hourly"])
            min_ob = _interp_hourly(data["mineral_ob_hourly"])

            series = {
                "alp_basal": alp_preob,
                "alp_osteo_ob": alp_ob,
                "collagen_basal": col_preob,
                "collagen_osteo_ob": col_ob,
                "mineral_basal": min_preob,
                "mineral_osteo_ob": min_ob,
            }
            increments = {k: np.asarray(v, dtype=float) for k, v in series.items()}

            return cls(
                path=Path(path),
                total_hours=int(total_hours),
                time_hours=hours,
                increments=increments,
            )

        required = {
            "time_hours",
            "alp_basal",
            "alp_osteo",
            "collagen_C1",
            "collagen_C2",
            "mineral_C1",
            "mineral_C2",
        }
        missing = sorted(required - files)
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
