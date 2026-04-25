from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

from .contracts import FRRGGraspConfig, GraspAction, GraspState
from .residual.policy import action_l2_norm


def _serialize_action(action: GraspAction) -> dict[str, Any]:
    return {
        "delta_pose_object": list(action.delta_pose_object),
        "delta_gripper": float(action.delta_gripper),
        "stop": bool(action.stop),
        "reason": action.reason,
        "debug_terms": dict(action.debug_terms),
    }


def _clip(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def _has_non_finite_action(action: GraspAction) -> bool:
    return not all(math.isfinite(float(value)) for value in action.delta_pose_object) or not math.isfinite(
        float(action.delta_gripper)
    )


def _combined_action_payload(raw_action: GraspAction, residual_action: GraspAction) -> dict[str, Any]:
    combined_pose = [
        float(raw_action.delta_pose_object[idx]) + float(residual_action.delta_pose_object[idx]) for idx in range(6)
    ]
    combined_gripper = float(raw_action.delta_gripper) + float(residual_action.delta_gripper)
    return {
        "delta_pose_object": combined_pose,
        "delta_gripper": combined_gripper,
    }


def combine_actions(raw_action: GraspAction, residual_action: GraspAction) -> GraspAction:
    payload = _combined_action_payload(raw_action, residual_action)
    return GraspAction(
        delta_pose_object=tuple(payload["delta_pose_object"]),
        delta_gripper=float(payload["delta_gripper"]),
        debug_terms={
            "is_combined_action": True,
        },
    )


@dataclass
class SafetyLimitResult:
    raw_action: GraspAction
    residual_action: GraspAction
    safe_action: GraspAction
    limited: bool
    stop: bool
    reason: str | None
    raw_action_norm: float
    residual_norm: float
    safe_action_norm: float
    debug_terms: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw_action": _serialize_action(self.raw_action),
            "residual_action": _serialize_action(self.residual_action),
            "safe_action": _serialize_action(self.safe_action),
            "limited": self.limited,
            "stop": self.stop,
            "reason": self.reason,
            "raw_action_norm": self.raw_action_norm,
            "residual_norm": self.residual_norm,
            "safe_action_norm": self.safe_action_norm,
            "debug_terms": dict(self.debug_terms),
        }


def apply_safety_limits(
    state: GraspState,
    config: FRRGGraspConfig,
    raw_action: GraspAction,
    *,
    residual_action: GraspAction | None = None,
    object_jump_m: float = 0.0,
    invalid_phase: bool = False,
) -> SafetyLimitResult:
    residual = residual_action or GraspAction()
    raw_action_norm = action_l2_norm(raw_action)
    residual_norm = action_l2_norm(residual)
    non_finite_action = _has_non_finite_action(raw_action) or _has_non_finite_action(residual)
    combined_payload = _combined_action_payload(raw_action, residual)
    vision_hardstop = float(state.vision_conf) < float(config.safety.vision_hardstop_min)
    object_jump = float(object_jump_m) > float(config.safety.obj_jump_stop_m)
    hardstop_reason = None
    if non_finite_action:
        hardstop_reason = "non_finite_action"
    elif invalid_phase:
        hardstop_reason = "invalid_phase"
    elif object_jump:
        hardstop_reason = "object_jump"
    elif vision_hardstop:
        hardstop_reason = "vision_lost"

    debug_terms = {
        "vision_hardstop": vision_hardstop,
        "vision_conf": float(state.vision_conf),
        "vision_hardstop_min": float(config.safety.vision_hardstop_min),
        "object_jump_m": float(object_jump_m),
        "obj_jump_stop_m": float(config.safety.obj_jump_stop_m),
        "object_jump": object_jump,
        "non_finite_action": non_finite_action,
        "invalid_phase": bool(invalid_phase),
        "combined_action": combined_payload,
    }

    if hardstop_reason is not None:
        safe_action = GraspAction(
            delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
            delta_gripper=0.0,
            stop=True,
            reason=hardstop_reason,
            debug_terms={
                "safety_hardstop": True,
                "hardstop_reason": hardstop_reason,
            },
        )
        return SafetyLimitResult(
            raw_action=raw_action,
            residual_action=residual,
            safe_action=safe_action,
            limited=True,
            stop=True,
            reason=hardstop_reason,
            raw_action_norm=raw_action_norm,
            residual_norm=residual_norm,
            safe_action_norm=0.0,
            debug_terms=debug_terms,
        )

    combined_action = combine_actions(raw_action, residual)
    clipped_pose = (
        _clip(combined_action.delta_pose_object[0], config.safety.max_step_xyz_m[0]),
        _clip(combined_action.delta_pose_object[1], config.safety.max_step_xyz_m[1]),
        _clip(combined_action.delta_pose_object[2], config.safety.max_step_xyz_m[2]),
        _clip(combined_action.delta_pose_object[3], config.safety.max_step_rpy_rad[0]),
        _clip(combined_action.delta_pose_object[4], config.safety.max_step_rpy_rad[1]),
        _clip(combined_action.delta_pose_object[5], config.safety.max_step_rpy_rad[2]),
    )
    clipped_gripper = _clip(combined_action.delta_gripper, config.safety.max_gripper_delta_m)
    safe_action = GraspAction(
        delta_pose_object=clipped_pose,
        delta_gripper=clipped_gripper,
        debug_terms={
            "safety_hardstop": False,
            "limited_from_combined_action": True,
        },
    )
    limited = clipped_pose != combined_action.delta_pose_object or clipped_gripper != combined_action.delta_gripper
    return SafetyLimitResult(
        raw_action=raw_action,
        residual_action=residual,
        safe_action=safe_action,
        limited=limited,
        stop=False,
        reason=None,
        raw_action_norm=raw_action_norm,
        residual_norm=residual_norm,
        safe_action_norm=action_l2_norm(safe_action),
        debug_terms=debug_terms,
    )


__all__ = [
    "SafetyLimitResult",
    "apply_safety_limits",
    "combine_actions",
]
