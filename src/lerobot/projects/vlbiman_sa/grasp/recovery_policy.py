from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .contracts import FRRGGraspConfig, GraspAction, GraspState


@dataclass
class RecoveryPolicyResult:
    proposal_type: str
    reason: str
    action: GraspAction
    debug_terms: dict[str, Any] = field(default_factory=dict)


def _backoff_action(state: GraspState, config: FRRGGraspConfig, reason: str) -> RecoveryPolicyResult:
    del state
    dz = -max(float(config.close_hold.preclose_distance_m), float(config.capture_build.target_depth_goal_m))
    return RecoveryPolicyResult(
        proposal_type="backoff",
        reason=reason,
        action=GraspAction(
            delta_pose_object=(0.0, 0.0, dz, 0.0, 0.0, 0.0),
            delta_gripper=0.0,
            debug_terms={
                "proposal_type": "backoff",
                "is_recovery_action": True,
                "failure_reason": reason,
            },
        ),
        debug_terms={"raw_dz": dz},
    )


def _half_open_action(state: GraspState, config: FRRGGraspConfig, reason: str) -> RecoveryPolicyResult:
    desired_open_width = float(config.handoff.handoff_open_width_m)
    delta_gripper = max(0.0, desired_open_width - float(state.gripper_width)) * 0.5
    return RecoveryPolicyResult(
        proposal_type="half_open",
        reason=reason,
        action=GraspAction(
            delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            delta_gripper=delta_gripper,
            debug_terms={
                "proposal_type": "half_open",
                "is_recovery_action": True,
                "failure_reason": reason,
            },
        ),
        debug_terms={
            "desired_open_width_m": desired_open_width,
            "raw_delta_gripper": delta_gripper,
        },
    )


def _recenter_action(state: GraspState, config: FRRGGraspConfig, reason: str) -> RecoveryPolicyResult:
    dx = float(config.capture_build.lat_gain) * float(state.e_lat) + float(config.capture_build.sym_gain) * float(state.e_sym)
    dyaw = float(config.capture_build.yaw_gain) * float(state.e_ang)
    return RecoveryPolicyResult(
        proposal_type="recenter",
        reason=reason,
        action=GraspAction(
            delta_pose_object=(dx, 0.0, 0.0, 0.0, 0.0, dyaw),
            delta_gripper=0.0,
            debug_terms={
                "proposal_type": "recenter",
                "is_recovery_action": True,
                "failure_reason": reason,
            },
        ),
        debug_terms={
            "raw_dx": dx,
            "raw_dyaw": dyaw,
        },
    )


def _abort_action(state: GraspState, config: FRRGGraspConfig, reason: str) -> RecoveryPolicyResult:
    del state
    del config
    return RecoveryPolicyResult(
        proposal_type="abort",
        reason=reason,
        action=GraspAction(
            delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            delta_gripper=0.0,
            stop=True,
            reason=reason,
            debug_terms={
                "proposal_type": "abort",
                "is_recovery_action": True,
                "failure_reason": reason,
            },
        ),
        debug_terms={},
    )


RECOVERY_PROPOSAL_MAP = {
    "capture_timeout": "backoff",
    "contact_not_detected": "half_open",
    "corridor_not_formed": "recenter",
    "large_drift": "recenter",
    "vision_lost": "abort",
    "object_jump": "abort",
    "non_finite_action": "abort",
    "invalid_phase": "abort",
    "max_retry_exceeded": "abort",
    "slip_detected": "abort",
    "unknown_state": "abort",
}


class RecoveryPolicy:
    def propose(self, state: GraspState, config: FRRGGraspConfig, reason: str) -> RecoveryPolicyResult:
        proposal_type = RECOVERY_PROPOSAL_MAP.get(reason, "abort")
        if proposal_type == "backoff":
            result = _backoff_action(state, config, reason)
        elif proposal_type == "half_open":
            result = _half_open_action(state, config, reason)
        elif proposal_type == "recenter":
            result = _recenter_action(state, config, reason)
        else:
            result = _abort_action(state, config, reason)
        result.debug_terms.update(
            {
                "proposal_type": result.proposal_type,
                "failure_reason": reason,
            }
        )
        return result


__all__ = [
    "RECOVERY_PROPOSAL_MAP",
    "RecoveryPolicy",
    "RecoveryPolicyResult",
]
