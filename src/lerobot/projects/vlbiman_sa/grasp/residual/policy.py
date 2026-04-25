from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import math
from typing import Any

from ..contracts import GraspAction, GraspState


def action_l2_norm(action: GraspAction) -> float:
    pose_sq_sum = sum(float(value) * float(value) for value in action.delta_pose_object)
    return math.sqrt(pose_sq_sum + float(action.delta_gripper) * float(action.delta_gripper))


@dataclass
class ResidualResult:
    action: GraspAction
    norm: float
    debug_terms: dict[str, Any] = field(default_factory=dict)


class ResidualPolicy(ABC):
    @abstractmethod
    def compute(self, state: GraspState, nominal_action: GraspAction) -> ResidualResult:
        raise NotImplementedError


__all__ = [
    "ResidualPolicy",
    "ResidualResult",
    "action_l2_norm",
]
