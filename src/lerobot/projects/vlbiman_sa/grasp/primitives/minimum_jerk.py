from __future__ import annotations


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def minimum_jerk_profile(progress: float) -> float:
    s = _clamp01(progress)
    return 10.0 * (s**3) - 15.0 * (s**4) + 6.0 * (s**5)


def minimum_jerk_delta(start: float, goal: float, *, progress: float, next_progress: float) -> float:
    start_progress = minimum_jerk_profile(progress)
    end_progress = minimum_jerk_profile(next_progress)
    return (float(goal) - float(start)) * (end_progress - start_progress)


__all__ = [
    "minimum_jerk_delta",
    "minimum_jerk_profile",
]
