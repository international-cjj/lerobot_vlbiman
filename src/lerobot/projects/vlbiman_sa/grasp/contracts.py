from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass
import math
from pathlib import Path
from typing import Any

import yaml


class ConfigError(ValueError):
    """Raised when the FRRG config is incomplete or invalid."""


def _require_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise ConfigError(f"{context} must be a mapping, got {type(payload).__name__}.")
    return payload


def _require_finite_number(section: str, key: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ConfigError(f"{section}.{key} must be a finite number, got {value!r}.")
    return float(value)


def _require_non_negative(section: str, key: str, value: Any) -> float:
    numeric = _require_finite_number(section, key, value)
    if numeric < 0.0:
        raise ConfigError(f"{section}.{key} must be >= 0, got {numeric}.")
    return numeric


def _require_positive(section: str, key: str, value: Any) -> float:
    numeric = _require_finite_number(section, key, value)
    if numeric <= 0.0:
        raise ConfigError(f"{section}.{key} must be > 0, got {numeric}.")
    return numeric


def _require_bool(section: str, key: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{section}.{key} must be a bool, got {type(value).__name__}.")
    return value


def _require_str(section: str, key: str, value: Any) -> str:
    if not isinstance(value, str) or not value:
        raise ConfigError(f"{section}.{key} must be a non-empty string.")
    return value


def _optional_path(section: str, key: str, value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{section}.{key} must be a non-empty string when provided.")
    return Path(value)


def _require_vector(
    section: str,
    key: str,
    value: Any,
    *,
    size: int,
    positive: bool = False,
    non_negative: bool = False,
) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ConfigError(f"{section}.{key} must be a list of {size} finite numbers.")

    result: list[float] = []
    for idx, item in enumerate(value):
        item_key = f"{key}[{idx}]"
        if positive:
            result.append(_require_positive(section, item_key, item))
        elif non_negative:
            result.append(_require_non_negative(section, item_key, item))
        else:
            result.append(_require_finite_number(section, item_key, item))
    return tuple(result)


def _vector3(name: str, values: tuple[float, ...]) -> tuple[float, float, float]:
    if len(values) != 3:
        raise ValueError(f"{name} must have exactly 3 values.")
    return (float(values[0]), float(values[1]), float(values[2]))


def _vector2(name: str, values: tuple[float, ...]) -> tuple[float, float]:
    if len(values) != 2:
        raise ValueError(f"{name} must have exactly 2 values.")
    return (float(values[0]), float(values[1]))


def _vector6(name: str, values: tuple[float, ...]) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ValueError(f"{name} must have exactly 6 values.")
    return (
        float(values[0]),
        float(values[1]),
        float(values[2]),
        float(values[3]),
        float(values[4]),
        float(values[5]),
    )


def _validate_finite_tuple(name: str, values: tuple[float, ...]) -> None:
    for value in values:
        if not math.isfinite(value):
            raise ValueError(f"{name} must contain only finite numbers.")


def _dataclass_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {field_info.name: _dataclass_to_dict(getattr(value, field_info.name)) for field_info in fields(value)}
    if isinstance(value, tuple):
        return [_dataclass_to_dict(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_to_dict(item) for item in value]
    if isinstance(value, dict):
        return {key: _dataclass_to_dict(item) for key, item in value.items()}
    if isinstance(value, Path):
        return str(value)
    return value


@dataclass(frozen=True)
class Pose6D:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]

    def __post_init__(self) -> None:
        xyz = _vector3("xyz", tuple(self.xyz))
        rpy = _vector3("rpy", tuple(self.rpy))
        _validate_finite_tuple("xyz", xyz)
        _validate_finite_tuple("rpy", rpy)
        object.__setattr__(self, "xyz", xyz)
        object.__setattr__(self, "rpy", rpy)

    @classmethod
    def zeros(cls) -> "Pose6D":
        return cls(xyz=(0.0, 0.0, 0.0), rpy=(0.0, 0.0, 0.0))


@dataclass
class GraspState:
    timestamp: float = 0.0
    phase: str = "HANDOFF"
    mode: str = "mock"
    retry_count: int = 0
    stable_count: int = 0
    phase_elapsed_s: float = 0.0
    ee_pose_base: Pose6D = field(default_factory=Pose6D.zeros)
    object_pose_base: Pose6D = field(default_factory=Pose6D.zeros)
    ee_pose_object: Pose6D = field(default_factory=Pose6D.zeros)
    gripper_width: float = 0.0
    gripper_cmd: float = 0.0
    gripper_current_proxy: float = 0.0
    vision_conf: float = 0.0
    target_visible: bool = False
    corridor_center_px: tuple[float, float] = (0.0, 0.0)
    object_center_px: tuple[float, float] = (0.0, 0.0)
    object_axis_angle: float = 0.0
    object_proj_width_px: float = 0.0
    object_proj_height_px: float = 0.0
    e_dep: float = 0.0
    e_lat: float = 0.0
    e_vert: float = 0.0
    e_ang: float = 0.0
    e_sym: float = 0.0
    occ_corridor: float = 0.0
    drift_obj: float = 0.0
    object_lift_m: float = 0.0
    capture_score: float = 0.0
    hold_score: float = 0.0
    lift_score: float = 0.0

    def __post_init__(self) -> None:
        self.corridor_center_px = _vector2("corridor_center_px", tuple(self.corridor_center_px))
        self.object_center_px = _vector2("object_center_px", tuple(self.object_center_px))


@dataclass
class GraspAction:
    delta_pose_object: tuple[float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    delta_gripper: float = 0.0
    stop: bool = False
    reason: str | None = None
    debug_terms: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta_pose_object = _vector6("delta_pose_object", tuple(self.delta_pose_object))
        _validate_finite_tuple("delta_pose_object", self.delta_pose_object)
        if not math.isfinite(float(self.delta_gripper)):
            raise ValueError("delta_gripper must be finite.")


@dataclass
class GuardResult:
    passed: bool
    score: float
    reason: str | None = None
    debug_terms: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not math.isfinite(float(self.score)):
            raise ValueError("score must be finite.")


@dataclass
class StepReport:
    step: str
    status: str
    completed_targets: list[str] = field(default_factory=list)
    changed_files: list[str] = field(default_factory=list)
    commands_run: list[str] = field(default_factory=list)
    acceptance: dict[str, Any] = field(default_factory=dict)
    hardware_called: bool = False
    camera_opened: bool = False
    mujoco_available: bool = False
    known_gaps: list[str] = field(default_factory=list)
    next_step_allowed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


@dataclass(frozen=True)
class RuntimeConfig:
    control_hz: float
    max_steps: int
    stable_window_frames: int
    max_retry_count: int
    default_input_mode: str


@dataclass(frozen=True)
class FeatureGeometryConfig:
    lateral_unit: str
    occ_definition: str
    angle_symmetry_weight_default: float


@dataclass(frozen=True)
class HandoffConfig:
    handoff_pos_tol_m: float
    handoff_yaw_tol_rad: float
    handoff_vis_min: float
    handoff_open_width_m: float


@dataclass(frozen=True)
class CaptureBuildConfig:
    mode: str
    solve_score_threshold: float
    close_score_threshold: float
    forward_enable_lat_tol_m: float
    forward_enable_ang_tol_rad: float
    forward_enable_occ_min: float
    target_depth_goal_m: float
    target_depth_max_m: float
    lat_gain: float
    sym_gain: float
    dep_gain: float
    vert_gain: float
    yaw_gain: float
    capture_timeout_s: float


@dataclass(frozen=True)
class CloseHoldConfig:
    preclose_distance_m: float
    preclose_pause_s: float
    close_speed_raw_per_s: float
    settle_time_s: float
    close_width_target_m: float
    contact_current_min: float
    contact_current_max: float
    hold_drift_max_m: float
    hold_score_threshold: float


@dataclass(frozen=True)
class LiftTestConfig:
    lift_height_m: float
    lift_speed_mps: float
    lift_hold_s: float
    slip_threshold_m: float
    lift_score_threshold: float


@dataclass(frozen=True)
class SafetyConfig:
    max_step_xyz_m: tuple[float, float, float]
    max_step_rpy_rad: tuple[float, float, float]
    max_gripper_delta_m: float
    vision_hardstop_min: float
    obj_jump_stop_m: float


@dataclass(frozen=True)
class ResidualConfig:
    enabled: bool
    policy: str
    checkpoint_path: Path | None = None


@dataclass(frozen=True)
class FRRGGraspConfig:
    runtime: RuntimeConfig
    feature_geometry: FeatureGeometryConfig
    handoff: HandoffConfig
    capture_build: CaptureBuildConfig
    close_hold: CloseHoldConfig
    lift_test: LiftTestConfig
    safety: SafetyConfig
    residual: ResidualConfig
    source_path: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        return _dataclass_to_dict(self)


def _section(payload: dict[str, Any], name: str) -> dict[str, Any]:
    if name not in payload:
        raise ConfigError(f"Missing required section: {name}")
    return _require_mapping(payload[name], context=name)


def _runtime_from_dict(payload: dict[str, Any]) -> RuntimeConfig:
    return RuntimeConfig(
        control_hz=_require_positive("runtime", "control_hz", payload.get("control_hz")),
        max_steps=int(_require_positive("runtime", "max_steps", payload.get("max_steps"))),
        stable_window_frames=int(
            _require_non_negative("runtime", "stable_window_frames", payload.get("stable_window_frames"))
        ),
        max_retry_count=int(_require_non_negative("runtime", "max_retry_count", payload.get("max_retry_count"))),
        default_input_mode=_require_str("runtime", "default_input_mode", payload.get("default_input_mode")),
    )


def _feature_geometry_from_dict(payload: dict[str, Any]) -> FeatureGeometryConfig:
    return FeatureGeometryConfig(
        lateral_unit=_require_str("feature_geometry", "lateral_unit", payload.get("lateral_unit")),
        occ_definition=_require_str("feature_geometry", "occ_definition", payload.get("occ_definition")),
        angle_symmetry_weight_default=_require_positive(
            "feature_geometry",
            "angle_symmetry_weight_default",
            payload.get("angle_symmetry_weight_default"),
        ),
    )


def _handoff_from_dict(payload: dict[str, Any]) -> HandoffConfig:
    return HandoffConfig(
        handoff_pos_tol_m=_require_positive("handoff", "handoff_pos_tol_m", payload.get("handoff_pos_tol_m")),
        handoff_yaw_tol_rad=_require_positive(
            "handoff",
            "handoff_yaw_tol_rad",
            payload.get("handoff_yaw_tol_rad"),
        ),
        handoff_vis_min=_require_non_negative("handoff", "handoff_vis_min", payload.get("handoff_vis_min")),
        handoff_open_width_m=_require_positive(
            "handoff",
            "handoff_open_width_m",
            payload.get("handoff_open_width_m"),
        ),
    )


def _capture_build_from_dict(payload: dict[str, Any]) -> CaptureBuildConfig:
    return CaptureBuildConfig(
        mode=_require_str("capture_build", "mode", payload.get("mode")),
        solve_score_threshold=_require_non_negative(
            "capture_build",
            "solve_score_threshold",
            payload.get("solve_score_threshold"),
        ),
        close_score_threshold=_require_non_negative(
            "capture_build",
            "close_score_threshold",
            payload.get("close_score_threshold"),
        ),
        forward_enable_lat_tol_m=_require_non_negative(
            "capture_build",
            "forward_enable_lat_tol_m",
            payload.get("forward_enable_lat_tol_m"),
        ),
        forward_enable_ang_tol_rad=_require_non_negative(
            "capture_build",
            "forward_enable_ang_tol_rad",
            payload.get("forward_enable_ang_tol_rad"),
        ),
        forward_enable_occ_min=_require_non_negative(
            "capture_build",
            "forward_enable_occ_min",
            payload.get("forward_enable_occ_min"),
        ),
        target_depth_goal_m=_require_non_negative(
            "capture_build",
            "target_depth_goal_m",
            payload.get("target_depth_goal_m"),
        ),
        target_depth_max_m=_require_non_negative(
            "capture_build",
            "target_depth_max_m",
            payload.get("target_depth_max_m"),
        ),
        lat_gain=_require_non_negative("capture_build", "lat_gain", payload.get("lat_gain")),
        sym_gain=_require_non_negative("capture_build", "sym_gain", payload.get("sym_gain")),
        dep_gain=_require_non_negative("capture_build", "dep_gain", payload.get("dep_gain")),
        vert_gain=_require_non_negative("capture_build", "vert_gain", payload.get("vert_gain")),
        yaw_gain=_require_non_negative("capture_build", "yaw_gain", payload.get("yaw_gain")),
        capture_timeout_s=_require_positive("capture_build", "capture_timeout_s", payload.get("capture_timeout_s")),
    )


def _close_hold_from_dict(payload: dict[str, Any]) -> CloseHoldConfig:
    return CloseHoldConfig(
        preclose_distance_m=_require_non_negative(
            "close_hold",
            "preclose_distance_m",
            payload.get("preclose_distance_m"),
        ),
        preclose_pause_s=_require_non_negative("close_hold", "preclose_pause_s", payload.get("preclose_pause_s")),
        close_speed_raw_per_s=_require_positive(
            "close_hold",
            "close_speed_raw_per_s",
            payload.get("close_speed_raw_per_s"),
        ),
        settle_time_s=_require_non_negative("close_hold", "settle_time_s", payload.get("settle_time_s")),
        close_width_target_m=_require_non_negative(
            "close_hold",
            "close_width_target_m",
            payload.get("close_width_target_m"),
        ),
        contact_current_min=_require_non_negative(
            "close_hold",
            "contact_current_min",
            payload.get("contact_current_min"),
        ),
        contact_current_max=_require_positive(
            "close_hold",
            "contact_current_max",
            payload.get("contact_current_max"),
        ),
        hold_drift_max_m=_require_non_negative("close_hold", "hold_drift_max_m", payload.get("hold_drift_max_m")),
        hold_score_threshold=_require_non_negative(
            "close_hold",
            "hold_score_threshold",
            payload.get("hold_score_threshold"),
        ),
    )


def _lift_test_from_dict(payload: dict[str, Any]) -> LiftTestConfig:
    return LiftTestConfig(
        lift_height_m=_require_non_negative("lift_test", "lift_height_m", payload.get("lift_height_m")),
        lift_speed_mps=_require_positive("lift_test", "lift_speed_mps", payload.get("lift_speed_mps")),
        lift_hold_s=_require_non_negative("lift_test", "lift_hold_s", payload.get("lift_hold_s")),
        slip_threshold_m=_require_non_negative("lift_test", "slip_threshold_m", payload.get("slip_threshold_m")),
        lift_score_threshold=_require_non_negative(
            "lift_test",
            "lift_score_threshold",
            payload.get("lift_score_threshold"),
        ),
    )


def _safety_from_dict(payload: dict[str, Any]) -> SafetyConfig:
    return SafetyConfig(
        max_step_xyz_m=_vector3(
            "safety.max_step_xyz_m",
            _require_vector("safety", "max_step_xyz_m", payload.get("max_step_xyz_m"), size=3, positive=True),
        ),
        max_step_rpy_rad=_vector3(
            "safety.max_step_rpy_rad",
            _require_vector(
                "safety",
                "max_step_rpy_rad",
                payload.get("max_step_rpy_rad"),
                size=3,
                non_negative=True,
            ),
        ),
        max_gripper_delta_m=_require_positive(
            "safety",
            "max_gripper_delta_m",
            payload.get("max_gripper_delta_m"),
        ),
        vision_hardstop_min=_require_non_negative(
            "safety",
            "vision_hardstop_min",
            payload.get("vision_hardstop_min"),
        ),
        obj_jump_stop_m=_require_positive("safety", "obj_jump_stop_m", payload.get("obj_jump_stop_m")),
    )


def _residual_from_dict(payload: dict[str, Any]) -> ResidualConfig:
    return ResidualConfig(
        enabled=_require_bool("residual", "enabled", payload.get("enabled")),
        policy=_require_str("residual", "policy", payload.get("policy")),
        checkpoint_path=_optional_path("residual", "checkpoint_path", payload.get("checkpoint_path")),
    )


def load_frrg_config(path: str | Path) -> FRRGGraspConfig:
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"FRRG config file not found: {config_path}")

    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    payload = _require_mapping(payload, context="frrg_config")

    config = FRRGGraspConfig(
        runtime=_runtime_from_dict(_section(payload, "runtime")),
        feature_geometry=_feature_geometry_from_dict(_section(payload, "feature_geometry")),
        handoff=_handoff_from_dict(_section(payload, "handoff")),
        capture_build=_capture_build_from_dict(_section(payload, "capture_build")),
        close_hold=_close_hold_from_dict(_section(payload, "close_hold")),
        lift_test=_lift_test_from_dict(_section(payload, "lift_test")),
        safety=_safety_from_dict(_section(payload, "safety")),
        residual=_residual_from_dict(_section(payload, "residual")),
        source_path=config_path,
    )

    if config.close_hold.contact_current_max < config.close_hold.contact_current_min:
        raise ConfigError("close_hold.contact_current_max must be >= close_hold.contact_current_min.")
    if config.capture_build.target_depth_max_m < config.capture_build.target_depth_goal_m:
        raise ConfigError("capture_build.target_depth_max_m must be >= capture_build.target_depth_goal_m.")
    if config.residual.policy not in {"zero", "bc"}:
        raise ConfigError("residual.policy must be one of {'zero', 'bc'}.")
    if config.residual.enabled and config.residual.policy == "zero":
        raise ConfigError("residual.enabled must stay false when residual.policy is 'zero'.")

    return config


__all__ = [
    "ConfigError",
    "FRRGGraspConfig",
    "GraspAction",
    "GraspState",
    "GuardResult",
    "Pose6D",
    "StepReport",
    "load_frrg_config",
]
