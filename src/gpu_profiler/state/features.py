from __future__ import annotations

from math import sqrt


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return 0.0
    idx = (len(sorted_values) - 1) * p
    lo = int(idx)
    hi = min(lo + 1, len(sorted_values) - 1)
    frac = idx - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def std(values: list[float], mean: float) -> float:
    if len(values) < 2:
        return 0.0
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return sqrt(var)


def skew_ratio(by_rank: dict[str, float]) -> float | None:
    if not by_rank:
        return None
    vals = [v for v in by_rank.values() if v is not None]
    if not vals:
        return None
    low = min(vals)
    high = max(vals)
    if low == 0:
        return None
    return high / low
