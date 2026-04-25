from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class IkSolutionSelectorConfig:
    preferred_joint_step_rad: float = 0.15
    max_joint_step_rad: float = 0.25


@dataclass(slots=True)
class JointTrajectoryMetrics:
    point_count: int
    max_joint_step_rad_inf: float
    mean_joint_step_rad_inf: float
    abrupt_step_count: int
    preferred_step_violations: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class IkSolutionSelector:
    def __init__(self, config: IkSolutionSelectorConfig | None = None):
        self.config = config or IkSolutionSelectorConfig()

    def select_nearest(
        self,
        candidate: np.ndarray,
        seed: np.ndarray,
        *,
        lower_bounds: np.ndarray,
        upper_bounds: np.ndarray,
    ) -> np.ndarray:
        normalized = np.asarray(candidate, dtype=float).copy()
        seed = np.asarray(seed, dtype=float)
        for idx in range(normalized.shape[0]):
            low = float(lower_bounds[idx])
            high = float(upper_bounds[idx])
            value = float(normalized[idx])
            best = value
            best_delta = abs(value - float(seed[idx]))
            for sign in (-1, 1):
                alternate = value + sign * 2.0 * np.pi
                if alternate < low or alternate > high:
                    continue
                delta = abs(alternate - float(seed[idx]))
                if delta < best_delta:
                    best = alternate
                    best_delta = delta
            normalized[idx] = best
        return np.minimum(np.maximum(normalized, lower_bounds), upper_bounds)

    def compute_metrics(self, joint_path: list[np.ndarray]) -> JointTrajectoryMetrics:
        if len(joint_path) <= 1:
            return JointTrajectoryMetrics(
                point_count=len(joint_path),
                max_joint_step_rad_inf=0.0,
                mean_joint_step_rad_inf=0.0,
                abrupt_step_count=0,
                preferred_step_violations=0,
            )

        deltas = [np.max(np.abs(curr - prev)) for prev, curr in zip(joint_path[:-1], joint_path[1:], strict=True)]
        abrupt_step_count = sum(delta > float(self.config.max_joint_step_rad) for delta in deltas)
        preferred_step_violations = sum(delta > float(self.config.preferred_joint_step_rad) for delta in deltas)
        return JointTrajectoryMetrics(
            point_count=len(joint_path),
            max_joint_step_rad_inf=float(max(deltas)),
            mean_joint_step_rad_inf=float(np.mean(deltas)),
            abrupt_step_count=int(abrupt_step_count),
            preferred_step_violations=int(preferred_step_violations),
        )


def densify_joint_transition(
    start: np.ndarray,
    end: np.ndarray,
    *,
    max_step_rad: float,
) -> list[np.ndarray]:
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    delta = end - start
    max_delta = float(np.max(np.abs(delta)))
    if max_delta <= max_step_rad + 1e-12:
        return [end]

    step_count = max(1, int(np.ceil(max_delta / max_step_rad)))
    return [start + delta * (index / step_count) for index in range(1, step_count + 1)]
