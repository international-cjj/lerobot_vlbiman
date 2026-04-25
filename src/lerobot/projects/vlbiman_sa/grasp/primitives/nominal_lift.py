from __future__ import annotations

from ..contracts import FRRGGraspConfig, GraspAction, GraspState
from .minimum_jerk import minimum_jerk_delta


def nominal_lift_action(
    state: GraspState,
    config: FRRGGraspConfig,
    *,
    dt: float | None = None,
) -> GraspAction:
    step_dt = (1.0 / float(config.runtime.control_hz)) if dt is None else float(dt)
    nominal_duration = max(
        float(config.lift_test.lift_height_m) / max(float(config.lift_test.lift_speed_mps), 1e-9),
        step_dt,
    )
    progress = min(float(state.phase_elapsed_s) / nominal_duration, 1.0)
    next_progress = min((float(state.phase_elapsed_s) + step_dt) / nominal_duration, 1.0)
    lifted_so_far = min(float(config.lift_test.lift_height_m), float(state.phase_elapsed_s) * float(config.lift_test.lift_speed_mps))
    remaining_height = max(0.0, float(config.lift_test.lift_height_m) - lifted_so_far)
    smooth_delta = minimum_jerk_delta(
        0.0,
        float(config.lift_test.lift_height_m),
        progress=progress,
        next_progress=next_progress,
    )
    linear_cap = float(config.lift_test.lift_speed_mps) * step_dt
    dz = min(max(smooth_delta, 0.0), linear_cap, remaining_height)
    return GraspAction(
        delta_pose_object=(0.0, dz, 0.0, 0.0, 0.0, 0.0),
        delta_gripper=0.0,
        debug_terms={
            "primitive": "lift",
            "is_raw_action": True,
            "dt": step_dt,
            "lift_progress": progress,
            "lift_next_progress": next_progress,
            "lifted_so_far_m": lifted_so_far,
            "remaining_height_m": remaining_height,
            "linear_cap_m": linear_cap,
            "raw_dy": dz,
        },
    )


__all__ = [
    "nominal_lift_action",
]
