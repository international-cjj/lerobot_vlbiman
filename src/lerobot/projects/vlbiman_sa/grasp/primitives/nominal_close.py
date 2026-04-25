from __future__ import annotations

from ..contracts import FRRGGraspConfig, GraspAction, GraspState
from .minimum_jerk import minimum_jerk_delta


def nominal_close_action(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    dt: float | None = None,
) -> GraspAction:
    step_dt = (1.0 / float(config.runtime.control_hz)) if dt is None else float(dt)
    if float(state.phase_elapsed_s) < float(config.close_hold.preclose_pause_s) and float(state.e_dep) > 0.0:
        progress = min(float(state.phase_elapsed_s) / max(float(config.close_hold.preclose_pause_s), 1e-9), 1.0)
        next_progress = min((float(state.phase_elapsed_s) + step_dt) / max(float(config.close_hold.preclose_pause_s), 1e-9), 1.0)
        raw_dz = minimum_jerk_delta(
            0.0,
            float(config.close_hold.preclose_distance_m),
            progress=progress,
            next_progress=next_progress,
        )
        dz = min(max(float(state.e_dep), 0.0), float(config.close_hold.preclose_distance_m), raw_dz)
        return GraspAction(
            delta_pose_object=(0.0, 0.0, dz, 0.0, 0.0, 0.0),
            delta_gripper=0.0,
            debug_terms={
                "primitive": "close",
                "close_mode": "preclose",
                "is_raw_action": True,
                "preclose_distance_m": float(config.close_hold.preclose_distance_m),
                "preclose_pause_s": float(config.close_hold.preclose_pause_s),
                "preclose_progress": progress,
                "preclose_next_progress": next_progress,
                "raw_dz": dz,
            },
        )

    raw_delta_gripper = -float(config.close_hold.close_speed_raw_per_s) * step_dt
    target_width = max(
        float(config.close_hold.close_width_target_m),
        float(state.gripper_width) + raw_delta_gripper,
    )
    delta_gripper = target_width - float(state.gripper_width)
    return GraspAction(
        delta_pose_object=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        delta_gripper=delta_gripper,
        debug_terms={
            "primitive": "close",
            "close_mode": "gripper_close",
            "is_raw_action": True,
            "close_speed_raw_per_s": float(config.close_hold.close_speed_raw_per_s),
            "dt": step_dt,
            "target_width_m": target_width,
            "raw_delta_gripper": delta_gripper,
        },
    )


__all__ = [
    "nominal_close_action",
]
