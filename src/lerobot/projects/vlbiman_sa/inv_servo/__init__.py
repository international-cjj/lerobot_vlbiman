from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORTS = {
    "BackendConfig": ".config",
    "BackendObservation": ".execution_backend",
    "BackendStepResult": ".execution_backend",
    "DataConfig": ".config",
    "DetectionState": ".target_state",
    "DetectorConfig": ".config",
    "DetectorInitState": ".detector",
    "ExecutionBackend": ".execution_backend",
    "GroundingDINOConfig": ".detector",
    "GroundingDINODetector": ".detector",
    "InvGraspExecutor": ".inv_grasp_executor",
    "InvGraspExecutorConfig": ".inv_grasp_executor",
    "InvRGBServoConfig": ".config",
    "InvServoConfigError": ".config",
    "InvServoResult": ".target_state",
    "MaskState": ".target_state",
    "OutputConfig": ".config",
    "RGBServoController": ".rgb_servo_controller",
    "RGBServoControllerConfig": ".rgb_servo_controller",
    "RobotBackendConfig": ".robot_backend",
    "RobotExecutionBackend": ".robot_backend",
    "SAM2Config": ".config",
    "SAM2LiveTracker": ".sam2_live_tracker",
    "SAM2LiveTrackerConfig": ".sam2_live_tracker",
    "SafetyConfig": ".config",
    "ServoCommand": ".target_state",
    "ServoConfig": ".config",
    "ServoError": ".target_state",
    "ServoSafetyConfig": ".servo_safety",
    "ServoSafetyFilter": ".servo_safety",
    "ServoTarget": ".target_state",
    "ServoValidationConfig": ".config",
    "SimBackendConfig": ".sim_backend",
    "SimExecutionBackend": ".sim_backend",
    "TargetConfig": ".config",
    "TargetProvider": ".target_provider",
    "TargetProviderConfig": ".target_provider",
    "TargetState": ".target_state",
    "TraceLogger": ".trace_logger",
    "TraceLoggerConfig": ".trace_logger",
    "bbox_xyxy_from_mask": ".metrics",
    "default_inv_rgb_servo_config_path": ".config",
    "load_inv_rgb_servo_config": ".config",
    "mask_area_px": ".metrics",
    "mask_centroid_uv": ".metrics",
    "mask_iou": ".metrics",
    "mask_state_from_mask": ".metrics",
    "target_state_from_bbox": ".target_state",
    "target_state_from_mask": ".target_state",
}

__all__ = sorted(_EXPORTS)


def __getattr__(name: str) -> Any:
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(module_name, __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value
