from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import math

import yaml

from .detector import GroundingDINOConfig
from .rgb_servo_controller import RGBServoControllerConfig
from .sam2_live_tracker import SAM2LiveTrackerConfig
from .servo_safety import ServoSafetyConfig
from .sim_backend import SimBackendConfig
from .target_provider import TargetProviderConfig


class InvServoConfigError(ValueError):
    pass


def default_inv_rgb_servo_config_path() -> Path:
    return Path(__file__).resolve().parents[1] / "configs" / "inv_rgb_servo.yaml"


def _require_mapping(value: Any, *, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise InvServoConfigError(f"{name} must be a mapping, got {type(value).__name__}.")
    return value


def _finite_float(name: str, value: Any, *, positive: bool = False, non_negative: bool = False) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise InvServoConfigError(f"{name} must be a finite number, got {value!r}.")
    result = float(value)
    if positive and result <= 0.0:
        raise InvServoConfigError(f"{name} must be > 0, got {result}.")
    if non_negative and result < 0.0:
        raise InvServoConfigError(f"{name} must be >= 0, got {result}.")
    return result


def _positive_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value <= 0:
        raise InvServoConfigError(f"{name} must be a positive integer, got {value!r}.")
    return int(value)


def _non_negative_int(name: str, value: Any) -> int:
    if not isinstance(value, int) or value < 0:
        raise InvServoConfigError(f"{name} must be a non-negative integer, got {value!r}.")
    return int(value)


def _require_bool(name: str, value: Any) -> bool:
    if not isinstance(value, bool):
        raise InvServoConfigError(f"{name} must be a bool, got {type(value).__name__}.")
    return bool(value)


def _require_str(name: str, value: Any, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str) or (not allow_empty and not value.strip()):
        raise InvServoConfigError(f"{name} must be a non-empty string.")
    return value


def _optional_path(name: str, value: Any) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise InvServoConfigError(f"{name} must be null or a non-empty string.")
    return Path(value)


def _path(name: str, value: Any) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise InvServoConfigError(f"{name} must be a non-empty string.")
    return Path(value)


def _to_plain(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


@dataclass(slots=True)
class TargetConfig:
    phrase: str = "yellow ball"
    target_frame_path: Path | None = None
    target_mask_path: Path | None = None
    target_state_path: Path | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "TargetConfig":
        return cls(
            phrase=_require_str("target.phrase", payload.get("phrase", "yellow ball")),
            target_frame_path=_optional_path("target.target_frame_path", payload.get("target_frame_path")),
            target_mask_path=_optional_path("target.target_mask_path", payload.get("target_mask_path")),
            target_state_path=_optional_path("target.target_state_path", payload.get("target_state_path")),
        )

    def to_provider_config(self) -> TargetProviderConfig:
        return TargetProviderConfig(
            phrase=self.phrase,
            target_frame_path=self.target_frame_path,
            target_mask_path=self.target_mask_path,
            target_state_path=self.target_state_path,
        )


@dataclass(slots=True)
class DataConfig:
    original_flow_dir: Path
    camera: str = "wrist"
    camera_aliases: list[str] = field(
        default_factory=lambda: [
            "wrist",
            "wrist_rgb",
            "dabaidcw_rgb",
            "hand",
            "hand_camera",
            "wrist_camera",
        ]
    )
    groundingdino_check_frame: int = 75
    sam2_check_start_frame: int = 75
    sam2_check_end_frame: int = 100
    servo_validation_start_frame: int = 65
    servo_validation_target_frame: int = 100

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DataConfig":
        aliases = payload.get(
            "camera_aliases",
            ["wrist", "wrist_rgb", "dabaidcw_rgb", "hand", "hand_camera", "wrist_camera"],
        )
        if not isinstance(aliases, list) or not aliases:
            raise InvServoConfigError("data.camera_aliases must be a non-empty list.")
        return cls(
            original_flow_dir=_path("data.original_flow_dir", payload.get("original_flow_dir")),
            camera=_require_str("data.camera", payload.get("camera", "wrist")),
            camera_aliases=[_require_str("data.camera_aliases[]", item) for item in aliases],
            groundingdino_check_frame=_non_negative_int(
                "data.groundingdino_check_frame", payload.get("groundingdino_check_frame", 75)
            ),
            sam2_check_start_frame=_non_negative_int("data.sam2_check_start_frame", payload.get("sam2_check_start_frame", 75)),
            sam2_check_end_frame=_non_negative_int("data.sam2_check_end_frame", payload.get("sam2_check_end_frame", 100)),
            servo_validation_start_frame=_non_negative_int(
                "data.servo_validation_start_frame", payload.get("servo_validation_start_frame", 65)
            ),
            servo_validation_target_frame=_non_negative_int(
                "data.servo_validation_target_frame", payload.get("servo_validation_target_frame", 100)
            ),
        )

    def __post_init__(self) -> None:
        if self.sam2_check_end_frame < self.sam2_check_start_frame:
            raise InvServoConfigError("data.sam2_check_end_frame must be >= data.sam2_check_start_frame.")


@dataclass(slots=True)
class DetectorConfig:
    type: str = "groundingdino"
    repo_path: Path = Path("/home/cjj/ViT-VS/GD_DINOv2_Sim/GroundingDINO")
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    device: str = "cuda"
    max_reinit: int = 3
    validation_phrase: str = "yellow ball"

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DetectorConfig":
        detector_type = _require_str("detector.type", payload.get("type", "groundingdino"))
        if detector_type != "groundingdino":
            raise InvServoConfigError(f"detector.type must be 'groundingdino', got {detector_type!r}.")
        return cls(
            type=detector_type,
            repo_path=_path("detector.repo_path", payload.get("repo_path")),
            config_path=_optional_path("detector.config_path", payload.get("config_path")),
            checkpoint_path=_optional_path("detector.checkpoint_path", payload.get("checkpoint_path")),
            box_threshold=_finite_float("detector.box_threshold", payload.get("box_threshold", 0.30), non_negative=True),
            text_threshold=_finite_float("detector.text_threshold", payload.get("text_threshold", 0.25), non_negative=True),
            device=_require_str("detector.device", payload.get("device", "cuda")),
            max_reinit=_non_negative_int("detector.max_reinit", payload.get("max_reinit", 3)),
            validation_phrase=_require_str("detector.validation_phrase", payload.get("validation_phrase", "yellow ball")),
        )

    def to_detector_config(self) -> GroundingDINOConfig:
        return GroundingDINOConfig(
            repo_path=self.repo_path,
            config_path=self.config_path,
            checkpoint_path=self.checkpoint_path,
            box_threshold=self.box_threshold,
            text_threshold=self.text_threshold,
            device=self.device,
            validation_phrase=self.validation_phrase,
        )


@dataclass(slots=True)
class SAM2Config:
    model_id: str = "facebook/sam2-hiera-small"
    model_size: str = "tiny"
    config_file: str | None = None
    checkpoint_path: Path | None = None
    device: str = "auto"
    use_fp16: bool = True
    input_resize_width: int = 640
    min_fps: float = 10.0
    target_fps: float = 15.0
    max_update_ms: float = 100.0
    repo_path: Path | None = None
    independent_live_tracker: bool = True
    forbid_disk_frame_io: bool = True
    local_files_only: bool = True
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    mask_threshold: float = 0.0
    jpeg_quality: int = 95
    incremental_live_tracker: bool = True
    live_seed_with_previous_mask: bool = True
    validation_start_frame: int = 75
    validation_end_frame: int = 100
    validation_requires_all_masks: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SAM2Config":
        return cls(
            model_id=_require_str("sam2.model_id", payload.get("model_id", "facebook/sam2-hiera-small")),
            model_size=_require_str("sam2.model_size", payload.get("model_size", "tiny")),
            config_file=payload.get("config_file") if payload.get("config_file") is None else _require_str("sam2.config_file", payload.get("config_file")),
            checkpoint_path=_optional_path("sam2.checkpoint_path", payload.get("checkpoint_path")),
            device=_require_str("sam2.device", payload.get("device", "auto")),
            use_fp16=_require_bool("sam2.use_fp16", payload.get("use_fp16", True)),
            input_resize_width=_positive_int("sam2.input_resize_width", payload.get("input_resize_width", 640)),
            min_fps=_finite_float("sam2.min_fps", payload.get("min_fps", 10.0), positive=True),
            target_fps=_finite_float("sam2.target_fps", payload.get("target_fps", 15.0), positive=True),
            max_update_ms=_finite_float("sam2.max_update_ms", payload.get("max_update_ms", 100.0), positive=True),
            repo_path=_optional_path("sam2.repo_path", payload.get("repo_path")),
            independent_live_tracker=_require_bool(
                "sam2.independent_live_tracker", payload.get("independent_live_tracker", True)
            ),
            forbid_disk_frame_io=_require_bool("sam2.forbid_disk_frame_io", payload.get("forbid_disk_frame_io", True)),
            local_files_only=_require_bool("sam2.local_files_only", payload.get("local_files_only", True)),
            offload_video_to_cpu=_require_bool("sam2.offload_video_to_cpu", payload.get("offload_video_to_cpu", False)),
            offload_state_to_cpu=_require_bool("sam2.offload_state_to_cpu", payload.get("offload_state_to_cpu", False)),
            async_loading_frames=_require_bool("sam2.async_loading_frames", payload.get("async_loading_frames", False)),
            mask_threshold=_finite_float("sam2.mask_threshold", payload.get("mask_threshold", 0.0)),
            jpeg_quality=_positive_int("sam2.jpeg_quality", payload.get("jpeg_quality", 95)),
            incremental_live_tracker=_require_bool(
                "sam2.incremental_live_tracker", payload.get("incremental_live_tracker", True)
            ),
            live_seed_with_previous_mask=_require_bool(
                "sam2.live_seed_with_previous_mask", payload.get("live_seed_with_previous_mask", True)
            ),
            validation_start_frame=_non_negative_int("sam2.validation_start_frame", payload.get("validation_start_frame", 75)),
            validation_end_frame=_non_negative_int("sam2.validation_end_frame", payload.get("validation_end_frame", 100)),
            validation_requires_all_masks=_require_bool(
                "sam2.validation_requires_all_masks", payload.get("validation_requires_all_masks", True)
            ),
        )

    def __post_init__(self) -> None:
        if self.validation_end_frame < self.validation_start_frame:
            raise InvServoConfigError("sam2.validation_end_frame must be >= sam2.validation_start_frame.")

    def to_tracker_config(self) -> SAM2LiveTrackerConfig:
        return SAM2LiveTrackerConfig(
            model_id=self.model_id,
            model_size=self.model_size,
            config_file=self.config_file,
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_fp16=self.use_fp16,
            input_resize_width=self.input_resize_width,
            max_update_ms=self.max_update_ms,
            repo_path=self.repo_path,
            independent_live_tracker=self.independent_live_tracker,
            local_files_only=self.local_files_only,
            offload_video_to_cpu=self.offload_video_to_cpu,
            offload_state_to_cpu=self.offload_state_to_cpu,
            async_loading_frames=self.async_loading_frames,
            mask_threshold=self.mask_threshold,
            jpeg_quality=self.jpeg_quality,
            incremental_live_tracker=self.incremental_live_tracker,
            live_seed_with_previous_mask=self.live_seed_with_previous_mask,
        )


@dataclass(slots=True)
class ServoConfig:
    k_u: float = 0.015
    k_v: float = 0.015
    k_a: float = 0.010
    axis_sign_x: float = 1.0
    axis_sign_y: float = 1.0
    axis_sign_z: float = 1.0
    max_step_xy_m: float = 0.003
    max_step_z_m: float = 0.004
    center_tol_u: float = 0.035
    center_tol_v: float = 0.045
    area_tol: float = 0.12
    mask_iou_tol: float = 0.70
    stable_frames: int = 5
    max_steps: int = 120

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ServoConfig":
        return cls(
            k_u=_finite_float("servo.k_u", payload.get("k_u", 0.015), non_negative=True),
            k_v=_finite_float("servo.k_v", payload.get("k_v", 0.015), non_negative=True),
            k_a=_finite_float("servo.k_a", payload.get("k_a", 0.010), non_negative=True),
            axis_sign_x=_finite_float("servo.axis_sign_x", payload.get("axis_sign_x", 1.0)),
            axis_sign_y=_finite_float("servo.axis_sign_y", payload.get("axis_sign_y", 1.0)),
            axis_sign_z=_finite_float("servo.axis_sign_z", payload.get("axis_sign_z", 1.0)),
            max_step_xy_m=_finite_float("servo.max_step_xy_m", payload.get("max_step_xy_m", 0.003), positive=True),
            max_step_z_m=_finite_float("servo.max_step_z_m", payload.get("max_step_z_m", 0.004), positive=True),
            center_tol_u=_finite_float("servo.center_tol_u", payload.get("center_tol_u", 0.035), non_negative=True),
            center_tol_v=_finite_float("servo.center_tol_v", payload.get("center_tol_v", 0.045), non_negative=True),
            area_tol=_finite_float("servo.area_tol", payload.get("area_tol", 0.12), non_negative=True),
            mask_iou_tol=_finite_float("servo.mask_iou_tol", payload.get("mask_iou_tol", 0.70), non_negative=True),
            stable_frames=_positive_int("servo.stable_frames", payload.get("stable_frames", 5)),
            max_steps=_positive_int("servo.max_steps", payload.get("max_steps", 120)),
        )

    def to_controller_config(self) -> RGBServoControllerConfig:
        return RGBServoControllerConfig(
            k_u=self.k_u,
            k_v=self.k_v,
            k_a=self.k_a,
            axis_sign_x=self.axis_sign_x,
            axis_sign_y=self.axis_sign_y,
            axis_sign_z=self.axis_sign_z,
            max_step_xy_m=self.max_step_xy_m,
            max_step_z_m=self.max_step_z_m,
            center_tol_u=self.center_tol_u,
            center_tol_v=self.center_tol_v,
            area_tol=self.area_tol,
            mask_iou_tol=self.mask_iou_tol,
        )


@dataclass(slots=True)
class ServoValidationConfig:
    enabled: bool = True
    backend: str = "sim"
    start_frame: int = 65
    target_frame: int = 100
    phrase: str = "yellow ball"
    target_mask_path: Path = Path("outputs/vlbiman_sa/inv_rgb_servo/check_sam2_75_100/mask_000100.png")
    start_scene_source: str = "original_flow"
    scene_path: Path | None = None
    require_groundingdino: bool = True
    require_sam2: bool = True
    require_sim_execution: bool = True
    success_center_tol_u: float = 0.035
    success_center_tol_v: float = 0.045
    success_area_tol: float = 0.12
    success_mask_iou: float = 0.70
    stable_frames: int = 5
    max_steps: int = 120

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "ServoValidationConfig":
        return cls(
            enabled=_require_bool("servo_validation.enabled", payload.get("enabled", True)),
            backend=_require_str("servo_validation.backend", payload.get("backend", "sim")),
            start_frame=_non_negative_int("servo_validation.start_frame", payload.get("start_frame", 65)),
            target_frame=_non_negative_int("servo_validation.target_frame", payload.get("target_frame", 100)),
            phrase=_require_str("servo_validation.phrase", payload.get("phrase", "yellow ball")),
            target_mask_path=_path("servo_validation.target_mask_path", payload.get("target_mask_path")),
            start_scene_source=_require_str(
                "servo_validation.start_scene_source", payload.get("start_scene_source", "original_flow")
            ),
            scene_path=_optional_path("servo_validation.scene_path", payload.get("scene_path")),
            require_groundingdino=_require_bool(
                "servo_validation.require_groundingdino", payload.get("require_groundingdino", True)
            ),
            require_sam2=_require_bool("servo_validation.require_sam2", payload.get("require_sam2", True)),
            require_sim_execution=_require_bool(
                "servo_validation.require_sim_execution", payload.get("require_sim_execution", True)
            ),
            success_center_tol_u=_finite_float(
                "servo_validation.success_center_tol_u", payload.get("success_center_tol_u", 0.035), non_negative=True
            ),
            success_center_tol_v=_finite_float(
                "servo_validation.success_center_tol_v", payload.get("success_center_tol_v", 0.045), non_negative=True
            ),
            success_area_tol=_finite_float(
                "servo_validation.success_area_tol", payload.get("success_area_tol", 0.12), non_negative=True
            ),
            success_mask_iou=_finite_float(
                "servo_validation.success_mask_iou", payload.get("success_mask_iou", 0.70), non_negative=True
            ),
            stable_frames=_positive_int("servo_validation.stable_frames", payload.get("stable_frames", 5)),
            max_steps=_positive_int("servo_validation.max_steps", payload.get("max_steps", 120)),
        )

    def to_sim_backend_config(self) -> SimBackendConfig:
        return SimBackendConfig(dry_run=True, start_frame=self.start_frame, target_frame=self.target_frame)


@dataclass(slots=True)
class SafetyConfig:
    min_area_ratio: float = 0.002
    max_area_ratio: float = 0.60
    max_bbox_jump_ratio: float = 0.25
    max_area_jump_ratio: float = 0.40
    max_lost_frames: int = 3
    max_joint_step_rad: float = 0.08
    pause_on_low_fps: bool = True

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "SafetyConfig":
        return cls(
            min_area_ratio=_finite_float("safety.min_area_ratio", payload.get("min_area_ratio", 0.002), non_negative=True),
            max_area_ratio=_finite_float("safety.max_area_ratio", payload.get("max_area_ratio", 0.60), positive=True),
            max_bbox_jump_ratio=_finite_float(
                "safety.max_bbox_jump_ratio", payload.get("max_bbox_jump_ratio", 0.25), non_negative=True
            ),
            max_area_jump_ratio=_finite_float(
                "safety.max_area_jump_ratio", payload.get("max_area_jump_ratio", 0.40), non_negative=True
            ),
            max_lost_frames=_non_negative_int("safety.max_lost_frames", payload.get("max_lost_frames", 3)),
            max_joint_step_rad=_finite_float("safety.max_joint_step_rad", payload.get("max_joint_step_rad", 0.08), positive=True),
            pause_on_low_fps=_require_bool("safety.pause_on_low_fps", payload.get("pause_on_low_fps", True)),
        )

    def to_safety_config(self) -> ServoSafetyConfig:
        return ServoSafetyConfig(max_rotation_rad=self.max_joint_step_rad)


@dataclass(slots=True)
class BackendConfig:
    type: str = "robot"
    camera: str = "wrist"
    dry_run: bool = False

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "BackendConfig":
        return cls(
            type=_require_str("backend.type", payload.get("type", "robot")),
            camera=_require_str("backend.camera", payload.get("camera", "wrist")),
            dry_run=_require_bool("backend.dry_run", payload.get("dry_run", False)),
        )


@dataclass(slots=True)
class OutputConfig:
    output_dir: Path = Path("outputs/vlbiman_sa/inv_rgb_servo/default")
    save_overlay: bool = True
    save_every_n_frames: int = 5
    groundingdino_check_dir: Path = Path("outputs/vlbiman_sa/inv_rgb_servo/check_groundingdino_frame75")
    sam2_check_dir: Path = Path("outputs/vlbiman_sa/inv_rgb_servo/check_sam2_75_100")
    servo_check_dir: Path = Path("outputs/vlbiman_sa/inv_rgb_servo/check_servo_75_to_100")

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "OutputConfig":
        return cls(
            output_dir=_path("output.output_dir", payload.get("output_dir", "outputs/vlbiman_sa/inv_rgb_servo/default")),
            save_overlay=_require_bool("output.save_overlay", payload.get("save_overlay", True)),
            save_every_n_frames=_positive_int("output.save_every_n_frames", payload.get("save_every_n_frames", 5)),
            groundingdino_check_dir=_path(
                "output.groundingdino_check_dir",
                payload.get("groundingdino_check_dir", "outputs/vlbiman_sa/inv_rgb_servo/check_groundingdino_frame75"),
            ),
            sam2_check_dir=_path(
                "output.sam2_check_dir", payload.get("sam2_check_dir", "outputs/vlbiman_sa/inv_rgb_servo/check_sam2_75_100")
            ),
            servo_check_dir=_path(
                "output.servo_check_dir",
                payload.get("servo_check_dir", "outputs/vlbiman_sa/inv_rgb_servo/check_servo_75_to_100"),
            ),
        )


@dataclass(slots=True)
class InvRGBServoConfig:
    target: TargetConfig
    data: DataConfig
    detector: DetectorConfig
    sam2: SAM2Config
    servo: ServoConfig
    servo_validation: ServoValidationConfig
    safety: SafetyConfig
    backend: BackendConfig
    output: OutputConfig
    config_path: Path | None = None

    @classmethod
    def from_mapping(cls, payload: dict[str, Any], *, config_path: Path | None = None) -> "InvRGBServoConfig":
        data = _require_mapping(payload, name="inv_rgb_servo config")
        required_sections = [
            "target",
            "data",
            "detector",
            "sam2",
            "servo",
            "servo_validation",
            "safety",
            "backend",
            "output",
        ]
        missing = [section for section in required_sections if section not in data]
        if missing:
            raise InvServoConfigError(f"Missing config sections: {missing}")

        return cls(
            target=TargetConfig.from_mapping(_require_mapping(data["target"], name="target")),
            data=DataConfig.from_mapping(_require_mapping(data["data"], name="data")),
            detector=DetectorConfig.from_mapping(_require_mapping(data["detector"], name="detector")),
            sam2=SAM2Config.from_mapping(_require_mapping(data["sam2"], name="sam2")),
            servo=ServoConfig.from_mapping(_require_mapping(data["servo"], name="servo")),
            servo_validation=ServoValidationConfig.from_mapping(
                _require_mapping(data["servo_validation"], name="servo_validation")
            ),
            safety=SafetyConfig.from_mapping(_require_mapping(data["safety"], name="safety")),
            backend=BackendConfig.from_mapping(_require_mapping(data["backend"], name="backend")),
            output=OutputConfig.from_mapping(_require_mapping(data["output"], name="output")),
            config_path=config_path,
        )

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        return _to_plain(payload)

    def dry_run_summary(self, *, cli_dry_run: bool = False) -> dict[str, Any]:
        return {
            "config_path": None if self.config_path is None else str(self.config_path),
            "target_phrase": self.target.phrase,
            "original_flow_dir": str(self.data.original_flow_dir),
            "groundingdino_repo_path": str(self.detector.repo_path),
            "groundingdino_config_path": None if self.detector.config_path is None else str(self.detector.config_path),
            "groundingdino_checkpoint_path": None
            if self.detector.checkpoint_path is None
            else str(self.detector.checkpoint_path),
            "groundingdino_check_frame": self.data.groundingdino_check_frame,
            "sam2_validation_range": [self.sam2.validation_start_frame, self.sam2.validation_end_frame],
            "servo_validation_range": [self.servo_validation.start_frame, self.servo_validation.target_frame],
            "servo_validation_scene_path": None
            if self.servo_validation.scene_path is None
            else str(self.servo_validation.scene_path),
            "backend_type": self.backend.type,
            "dry_run": bool(cli_dry_run or self.backend.dry_run),
            "paths": {
                "original_flow_dir_exists": self.data.original_flow_dir.exists(),
                "groundingdino_repo_exists": self.detector.repo_path.exists(),
                "groundingdino_config_exists": None
                if self.detector.config_path is None
                else self.detector.config_path.exists(),
                "groundingdino_checkpoint_exists": None
                if self.detector.checkpoint_path is None
                else self.detector.checkpoint_path.exists(),
                "target_mask_path_exists": self.servo_validation.target_mask_path.exists(),
            },
        }

    def to_target_provider_config(self, *, prefer_servo_validation_target: bool = True) -> TargetProviderConfig:
        target_mask_path = self.target.target_mask_path
        target_frame_index = self.servo_validation.target_frame
        phrase = self.target.phrase
        if prefer_servo_validation_target:
            target_mask_path = target_mask_path or self.servo_validation.target_mask_path
            target_frame_index = self.servo_validation.target_frame
            phrase = self.servo_validation.phrase or phrase
        return TargetProviderConfig(
            phrase=phrase,
            target_frame_index=target_frame_index,
            target_frame_path=self.target.target_frame_path,
            target_mask_path=target_mask_path,
            target_state_path=self.target.target_state_path,
        )


def load_inv_rgb_servo_config(path: Path | str | None = None) -> InvRGBServoConfig:
    config_path = Path(path) if path is not None else default_inv_rgb_servo_config_path()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return InvRGBServoConfig.from_mapping(payload, config_path=config_path)
