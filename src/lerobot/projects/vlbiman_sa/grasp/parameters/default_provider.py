from __future__ import annotations

from dataclasses import dataclass, replace

from ..contracts import FRRGGraspConfig
from .theta_schema import DemoTheta, MAPPABLE_THETA_PARAMETER_NAMES, ThetaParameterSource


@dataclass(frozen=True)
class ThetaApplyResult:
    config: FRRGGraspConfig
    applied_overrides: dict[str, float]
    unused_parameters: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "applied_overrides": dict(self.applied_overrides),
            "unused_parameters": list(self.unused_parameters),
        }


def default_theta_sources_from_config(config: FRRGGraspConfig) -> dict[str, ThetaParameterSource]:
    return {
        "advance_speed_mps": ThetaParameterSource(
            name="advance_speed_mps",
            value=float(config.safety.max_step_xyz_m[2]) * float(config.runtime.control_hz),
            source="default_from_config",
            default_used=True,
            reason="derived_from_safety_linear_speed_cap",
            diagnostics={
                "max_step_xyz_m_z": float(config.safety.max_step_xyz_m[2]),
                "control_hz": float(config.runtime.control_hz),
            },
        ),
        "preclose_distance_m": ThetaParameterSource(
            name="preclose_distance_m",
            value=float(config.close_hold.preclose_distance_m),
            source="default_from_config",
            default_used=True,
            reason="copied_from_close_hold.preclose_distance_m",
        ),
        "close_speed_raw_per_s": ThetaParameterSource(
            name="close_speed_raw_per_s",
            value=float(config.close_hold.close_speed_raw_per_s),
            source="default_from_config",
            default_used=True,
            reason="copied_from_close_hold.close_speed_raw_per_s",
        ),
        "settle_time_s": ThetaParameterSource(
            name="settle_time_s",
            value=float(config.close_hold.settle_time_s),
            source="default_from_config",
            default_used=True,
            reason="copied_from_close_hold.settle_time_s",
        ),
        "lift_height_m": ThetaParameterSource(
            name="lift_height_m",
            value=float(config.lift_test.lift_height_m),
            source="default_from_config",
            default_used=True,
            reason="copied_from_lift_test.lift_height_m",
        ),
    }


def default_theta_from_config(config: FRRGGraspConfig) -> DemoTheta:
    defaults = default_theta_sources_from_config(config)
    return DemoTheta.from_mapping({name: defaults[name].value for name in defaults})


def apply_theta_overrides(config: FRRGGraspConfig, theta: DemoTheta) -> ThetaApplyResult:
    applied_overrides = {
        "close_hold.preclose_distance_m": float(theta.preclose_distance_m),
        "close_hold.close_speed_raw_per_s": float(theta.close_speed_raw_per_s),
        "close_hold.settle_time_s": float(theta.settle_time_s),
        "lift_test.lift_height_m": float(theta.lift_height_m),
    }
    updated_close_hold = replace(
        config.close_hold,
        preclose_distance_m=float(theta.preclose_distance_m),
        close_speed_raw_per_s=float(theta.close_speed_raw_per_s),
        settle_time_s=float(theta.settle_time_s),
    )
    updated_lift_test = replace(
        config.lift_test,
        lift_height_m=float(theta.lift_height_m),
    )
    updated_config = replace(
        config,
        close_hold=updated_close_hold,
        lift_test=updated_lift_test,
    )
    unused_parameters = [name for name in theta.to_dict() if name not in MAPPABLE_THETA_PARAMETER_NAMES]
    return ThetaApplyResult(
        config=updated_config,
        applied_overrides=applied_overrides,
        unused_parameters=unused_parameters,
    )


__all__ = [
    "ThetaApplyResult",
    "apply_theta_overrides",
    "default_theta_from_config",
    "default_theta_sources_from_config",
]
