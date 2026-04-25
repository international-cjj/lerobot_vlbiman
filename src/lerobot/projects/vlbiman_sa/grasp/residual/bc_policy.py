from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Callable

import numpy as np

from ..contracts import FRRGGraspConfig, GraspAction, GraspState
from .policy import ResidualPolicy, ResidualResult, action_l2_norm
from .zero_policy import ZeroResidualPolicy


BC_FEATURE_NAMES = (
    "e_dep",
    "e_lat",
    "e_vert",
    "e_ang",
    "e_sym",
    "occ_corridor",
    "drift_obj",
    "gripper_width",
    "gripper_current_proxy",
    "vision_conf",
    "capture_score",
    "hold_score",
    "lift_score",
    "nominal_dx",
    "nominal_dy",
    "nominal_dz",
    "nominal_dyaw",
    "nominal_gripper",
)
BC_OUTPUT_DIM = 7


def _clip(value: float, limit: float) -> float:
    return max(-float(limit), min(float(limit), float(value)))


def residual_clip_limits_from_config(config: FRRGGraspConfig) -> tuple[tuple[float, float, float, float, float, float], float]:
    pose_limits = (
        float(config.safety.max_step_xyz_m[0]),
        float(config.safety.max_step_xyz_m[1]),
        float(config.safety.max_step_xyz_m[2]),
        float(config.safety.max_step_rpy_rad[0]),
        float(config.safety.max_step_rpy_rad[1]),
        float(config.safety.max_step_rpy_rad[2]),
    )
    return pose_limits, float(config.safety.max_gripper_delta_m)


def residual_action_to_vector(action: GraspAction) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(action.delta_pose_object[0]),
        float(action.delta_pose_object[1]),
        float(action.delta_pose_object[2]),
        float(action.delta_pose_object[3]),
        float(action.delta_pose_object[4]),
        float(action.delta_pose_object[5]),
        float(action.delta_gripper),
    )


def clip_residual_vector(
    values: tuple[float, float, float, float, float, float, float],
    *,
    clip_pose_limits: tuple[float, float, float, float, float, float],
    clip_gripper_limit: float,
) -> tuple[float, float, float, float, float, float, float]:
    return (
        _clip(values[0], clip_pose_limits[0]),
        _clip(values[1], clip_pose_limits[1]),
        _clip(values[2], clip_pose_limits[2]),
        _clip(values[3], clip_pose_limits[3]),
        _clip(values[4], clip_pose_limits[4]),
        _clip(values[5], clip_pose_limits[5]),
        _clip(values[6], clip_gripper_limit),
    )


def residual_vector_to_action(
    values: tuple[float, float, float, float, float, float, float],
    *,
    debug_terms: dict[str, Any] | None = None,
) -> GraspAction:
    return GraspAction(
        delta_pose_object=(
            float(values[0]),
            float(values[1]),
            float(values[2]),
            float(values[3]),
            float(values[4]),
            float(values[5]),
        ),
        delta_gripper=float(values[6]),
        debug_terms={} if debug_terms is None else dict(debug_terms),
    )


def extract_bc_feature_dict(state: GraspState, nominal_action: GraspAction) -> dict[str, float]:
    return {
        "e_dep": float(state.e_dep),
        "e_lat": float(state.e_lat),
        "e_vert": float(state.e_vert),
        "e_ang": float(state.e_ang),
        "e_sym": float(state.e_sym),
        "occ_corridor": float(state.occ_corridor),
        "drift_obj": float(state.drift_obj),
        "gripper_width": float(state.gripper_width),
        "gripper_current_proxy": float(state.gripper_current_proxy),
        "vision_conf": float(state.vision_conf),
        "capture_score": float(state.capture_score),
        "hold_score": float(state.hold_score),
        "lift_score": float(state.lift_score),
        "nominal_dx": float(nominal_action.delta_pose_object[0]),
        "nominal_dy": float(nominal_action.delta_pose_object[1]),
        "nominal_dz": float(nominal_action.delta_pose_object[2]),
        "nominal_dyaw": float(nominal_action.delta_pose_object[5]),
        "nominal_gripper": float(nominal_action.delta_gripper),
    }


def extract_bc_features(
    state: GraspState,
    nominal_action: GraspAction,
    *,
    feature_names: tuple[str, ...] = BC_FEATURE_NAMES,
) -> np.ndarray:
    feature_map = extract_bc_feature_dict(state, nominal_action)
    try:
        feature_values = [feature_map[name] for name in feature_names]
    except KeyError as exc:
        raise ValueError(f"Unsupported BC feature name: {exc.args[0]}") from exc
    feature_array = np.asarray(feature_values, dtype=float)
    if not np.all(np.isfinite(feature_array)):
        raise ValueError("BC feature vector must contain only finite values.")
    return feature_array


@dataclass(frozen=True)
class ResidualLabel:
    action: GraspAction
    raw_vector: tuple[float, float, float, float, float, float, float]
    clipped_vector: tuple[float, float, float, float, float, float, float]
    clip_applied: bool


def compute_demo_residual_label(
    demo_action: GraspAction,
    nominal_action: GraspAction,
    *,
    clip_pose_limits: tuple[float, float, float, float, float, float],
    clip_gripper_limit: float,
) -> ResidualLabel:
    raw_vector = tuple(
        float(demo_component) - float(nominal_component)
        for demo_component, nominal_component in zip(
            residual_action_to_vector(demo_action),
            residual_action_to_vector(nominal_action),
        )
    )
    clipped_vector = clip_residual_vector(
        raw_vector,
        clip_pose_limits=clip_pose_limits,
        clip_gripper_limit=clip_gripper_limit,
    )
    clip_applied = any(
        not math.isclose(float(raw), float(clipped), rel_tol=0.0, abs_tol=1e-12)
        for raw, clipped in zip(raw_vector, clipped_vector)
    )
    action = residual_vector_to_action(
        clipped_vector,
        debug_terms={
            "policy": "bc_label",
            "label_source": "demo_minus_nominal",
            "clip_applied": clip_applied,
            "is_residual_action": True,
        },
    )
    return ResidualLabel(
        action=action,
        raw_vector=raw_vector,
        clipped_vector=clipped_vector,
        clip_applied=clip_applied,
    )


def create_bc_checkpoint_payload(
    *,
    weights: np.ndarray,
    bias: np.ndarray,
    clip_pose_limits: tuple[float, float, float, float, float, float],
    clip_gripper_limit: float,
    feature_names: tuple[str, ...] = BC_FEATURE_NAMES,
    model_family: str = "linear",
) -> dict[str, Any]:
    if weights.shape != (BC_OUTPUT_DIM, len(feature_names)):
        raise ValueError(
            f"weights must have shape ({BC_OUTPUT_DIM}, {len(feature_names)}), got {tuple(weights.shape)}."
        )
    if bias.shape != (BC_OUTPUT_DIM,):
        raise ValueError(f"bias must have shape ({BC_OUTPUT_DIM},), got {tuple(bias.shape)}.")
    if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
        raise ValueError("BC checkpoint weights and bias must be finite.")
    return {
        "policy": "bc",
        "model_family": model_family,
        "feature_names": list(feature_names),
        "clip_pose_limits": [float(value) for value in clip_pose_limits],
        "clip_gripper_limit": float(clip_gripper_limit),
        "weights": weights.tolist(),
        "bias": bias.tolist(),
    }


def save_bc_checkpoint(
    path: str | Path,
    *,
    weights: np.ndarray,
    bias: np.ndarray,
    clip_pose_limits: tuple[float, float, float, float, float, float],
    clip_gripper_limit: float,
    feature_names: tuple[str, ...] = BC_FEATURE_NAMES,
    model_family: str = "linear",
) -> Path:
    checkpoint_path = Path(path)
    payload = create_bc_checkpoint_payload(
        weights=weights,
        bias=bias,
        clip_pose_limits=clip_pose_limits,
        clip_gripper_limit=clip_gripper_limit,
        feature_names=feature_names,
        model_family=model_family,
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return checkpoint_path


def _resolve_checkpoint_path(config: FRRGGraspConfig) -> Path | None:
    checkpoint_path = config.residual.checkpoint_path
    if checkpoint_path is None:
        return None
    if checkpoint_path.is_absolute():
        return checkpoint_path
    if config.source_path is not None:
        return (config.source_path.parent / checkpoint_path).resolve()
    return checkpoint_path.resolve()


def _load_checkpoint_payload(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"BC checkpoint must be a JSON object, got {type(payload).__name__}.")
    return payload


def _checkpoint_feature_names(payload: dict[str, Any]) -> tuple[str, ...]:
    feature_names_payload = payload.get("feature_names")
    if feature_names_payload is None:
        return BC_FEATURE_NAMES
    if not isinstance(feature_names_payload, list) or not feature_names_payload:
        raise ValueError("BC checkpoint feature_names must be a non-empty list.")
    feature_names = tuple(str(name) for name in feature_names_payload)
    for name in feature_names:
        if name not in BC_FEATURE_NAMES:
            raise ValueError(f"Unsupported BC feature name in checkpoint: {name}")
    return feature_names


def _finite_array(payload: Any, *, name: str, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(payload, dtype=float)
    if array.shape != expected_shape:
        raise ValueError(f"{name} must have shape {expected_shape}, got {tuple(array.shape)}.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


class BCResidualPolicy(ResidualPolicy):
    def __init__(
        self,
        *,
        checkpoint_path: Path,
        feature_names: tuple[str, ...],
        weights: np.ndarray,
        bias: np.ndarray,
        clip_pose_limits: tuple[float, float, float, float, float, float],
        clip_gripper_limit: float,
        model_family: str = "linear",
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.feature_names = tuple(feature_names)
        self.weights = np.asarray(weights, dtype=float)
        self.bias = np.asarray(bias, dtype=float)
        self.clip_pose_limits = tuple(float(value) for value in clip_pose_limits)
        self.clip_gripper_limit = float(clip_gripper_limit)
        self.model_family = str(model_family)

    @classmethod
    def from_checkpoint(cls, checkpoint_path: str | Path, config: FRRGGraspConfig) -> "BCResidualPolicy":
        resolved_path = Path(checkpoint_path).resolve()
        if not resolved_path.exists():
            raise FileNotFoundError(f"BC checkpoint not found: {resolved_path}")
        payload = _load_checkpoint_payload(resolved_path)
        feature_names = _checkpoint_feature_names(payload)
        weights = _finite_array(payload.get("weights"), name="weights", expected_shape=(BC_OUTPUT_DIM, len(feature_names)))
        bias = _finite_array(payload.get("bias"), name="bias", expected_shape=(BC_OUTPUT_DIM,))
        default_clip_pose_limits, default_clip_gripper_limit = residual_clip_limits_from_config(config)
        checkpoint_clip_pose = payload.get("clip_pose_limits", list(default_clip_pose_limits))
        checkpoint_clip_gripper = payload.get("clip_gripper_limit", default_clip_gripper_limit)
        clip_pose_limits_array = _finite_array(
            checkpoint_clip_pose,
            name="clip_pose_limits",
            expected_shape=(6,),
        )
        clip_gripper_limit = float(checkpoint_clip_gripper)
        if not math.isfinite(clip_gripper_limit) or clip_gripper_limit < 0.0:
            raise ValueError("clip_gripper_limit must be a finite non-negative number.")
        return cls(
            checkpoint_path=resolved_path,
            feature_names=feature_names,
            weights=weights,
            bias=bias,
            clip_pose_limits=tuple(float(value) for value in clip_pose_limits_array.tolist()),
            clip_gripper_limit=clip_gripper_limit,
            model_family=str(payload.get("model_family", "linear")),
        )

    def compute(self, state: GraspState, nominal_action: GraspAction) -> ResidualResult:
        feature_vector = extract_bc_features(state, nominal_action, feature_names=self.feature_names)
        raw_vector_array = (self.weights @ feature_vector) + self.bias
        raw_vector = tuple(float(value) for value in raw_vector_array.tolist())
        clipped_vector = clip_residual_vector(
            raw_vector,
            clip_pose_limits=self.clip_pose_limits,
            clip_gripper_limit=self.clip_gripper_limit,
        )
        clip_applied = any(
            not math.isclose(float(raw), float(clipped), rel_tol=0.0, abs_tol=1e-12)
            for raw, clipped in zip(raw_vector, clipped_vector)
        )
        raw_action = residual_vector_to_action(raw_vector)
        action = residual_vector_to_action(
            clipped_vector,
            debug_terms={
                "policy": "bc",
                "residual_enabled": True,
                "requested_policy": "bc",
                "model_family": self.model_family,
                "checkpoint_path": str(self.checkpoint_path),
                "clip_applied": clip_applied,
                "is_residual_action": True,
            },
        )
        return ResidualResult(
            action=action,
            norm=action_l2_norm(action),
            debug_terms={
                "policy": "bc",
                "residual_enabled": True,
                "requested_policy": "bc",
                "model_family": self.model_family,
                "checkpoint_path": str(self.checkpoint_path),
                "feature_dim": len(self.feature_names),
                "raw_output_norm": action_l2_norm(raw_action),
                "clip_applied": clip_applied,
            },
        )


def build_residual_policy(
    config: FRRGGraspConfig,
    *,
    checkpoint_loader: Callable[[Path, FRRGGraspConfig], ResidualPolicy] | None = None,
) -> ResidualPolicy:
    loader = checkpoint_loader or BCResidualPolicy.from_checkpoint
    if not config.residual.enabled:
        return ZeroResidualPolicy(
            residual_enabled=False,
            requested_policy=config.residual.policy,
        )

    if config.residual.policy == "zero":
        return ZeroResidualPolicy(
            residual_enabled=True,
            requested_policy="zero",
        )

    checkpoint_path = _resolve_checkpoint_path(config)
    if checkpoint_path is None:
        return ZeroResidualPolicy(
            residual_enabled=True,
            requested_policy="bc",
            fallback_reason="checkpoint_missing",
        )

    try:
        return loader(checkpoint_path, config)
    except Exception as exc:  # noqa: BLE001
        return ZeroResidualPolicy(
            residual_enabled=True,
            requested_policy="bc",
            fallback_reason="checkpoint_load_failed",
            fallback_detail=f"{type(exc).__name__}: {exc}",
        )


__all__ = [
    "BC_FEATURE_NAMES",
    "BCResidualPolicy",
    "ResidualLabel",
    "build_residual_policy",
    "clip_residual_vector",
    "compute_demo_residual_label",
    "create_bc_checkpoint_payload",
    "extract_bc_feature_dict",
    "extract_bc_features",
    "residual_action_to_vector",
    "residual_clip_limits_from_config",
    "residual_vector_to_action",
    "save_bc_checkpoint",
]
