from __future__ import annotations

from ..contracts import GraspAction, GraspState
from .policy import ResidualPolicy, ResidualResult


class ZeroResidualPolicy(ResidualPolicy):
    def __init__(
        self,
        *,
        residual_enabled: bool = False,
        requested_policy: str = "zero",
        fallback_reason: str | None = None,
        fallback_detail: str | None = None,
    ) -> None:
        self.residual_enabled = bool(residual_enabled)
        self.requested_policy = str(requested_policy)
        self.fallback_reason = fallback_reason
        self.fallback_detail = fallback_detail

    def compute(self, state: GraspState, nominal_action: GraspAction) -> ResidualResult:
        action_debug_terms = {
            "policy": "zero",
            "residual_enabled": self.residual_enabled,
            "requested_policy": self.requested_policy,
            "is_residual_action": True,
        }
        if self.fallback_reason is not None:
            action_debug_terms["fallback_reason"] = self.fallback_reason
        if self.fallback_detail is not None:
            action_debug_terms["fallback_detail"] = self.fallback_detail

        result_debug_terms = {
            "policy": "zero",
            "residual_enabled": self.residual_enabled,
            "requested_policy": self.requested_policy,
            "nominal_action_present": nominal_action is not None,
        }
        if self.fallback_reason is not None:
            result_debug_terms["fallback_reason"] = self.fallback_reason
        if self.fallback_detail is not None:
            result_debug_terms["fallback_detail"] = self.fallback_detail

        return ResidualResult(
            action=GraspAction(
                delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
                delta_gripper=0.0,
                debug_terms=action_debug_terms,
            ),
            norm=0.0,
            debug_terms=result_debug_terms,
        )


__all__ = [
    "ZeroResidualPolicy",
]
