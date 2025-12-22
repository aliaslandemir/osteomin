"""Helpers for converting between daily and hourly time scales."""

from __future__ import annotations

HOURS_PER_DAY: int = 24


def hours_from_days(days: float) -> int:
    """Return the nearest integer number of hours that correspond to the input days."""
    return int(round(days * HOURS_PER_DAY))


def per_day_rate_to_hourly(rate: float) -> float:
    """Convert a per-day rate to an equivalent hourly increment."""
    return float(rate) / HOURS_PER_DAY


def per_day_probability_to_hourly(prob: float) -> float:
    """Convert a per-day success probability into the equivalent hourly success probability."""
    p = float(prob)
    if p <= 0.0:
        return 0.0
    if p >= 1.0:
        return 1.0
    return 1.0 - (1.0 - p) ** (1.0 / HOURS_PER_DAY)
