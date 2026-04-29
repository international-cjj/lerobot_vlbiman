from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .execution_backend import ExecutionBackend
from .target_state import InvServoResult, ServoCommand


@dataclass(slots=True)
class RobotBackendConfig:
    enabled: bool = False
    require_explicit_enable: bool = True
    command_timeout_s: float = 1.0
    camera: str = "wrist"
    delta_frame: str = "tool"
    arm_velocity: float | None = None
    arm_smooth_factor: float | None = None
    camera_aliases: tuple[str, ...] = (
        "wrist",
        "wrist_rgb",
        "dabaidcw_rgb",
        "hand",
        "hand_camera",
        "wrist_camera",
    )


class RobotExecutionBackend(ExecutionBackend):
    name = "robot"

    def __init__(self, robot: Any | None = None, config: RobotBackendConfig | None = None):
        self.robot = robot
        self.config = config or RobotBackendConfig()
        self.frame_index = 0

    def reset_to_frame(self, frame_index: int) -> InvServoResult:
        return InvServoResult.failure(
            "robot_backend_cannot_reset_to_recorded_frame",
            {"backend": self.name, "frame_index": frame_index},
        )

    def get_observation(self) -> InvServoResult:
        return self.get_rgb_frame(self.config.camera)

    def get_rgb_frame(self, camera: str | None = None) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("robot_adapter_missing", {"backend": self.name})
        try:
            observation = self.robot.get_observation()
        except Exception as exc:
            return InvServoResult.failure(
                "robot_observation_failed",
                {"backend": self.name, "error": f"{type(exc).__name__}: {exc}"},
            )

        camera_key, image_rgb = self._resolve_rgb_image(observation, camera or self.config.camera)
        if camera_key is None or image_rgb is None:
            return InvServoResult.failure(
                "robot_rgb_frame_unavailable",
                {
                    "backend": self.name,
                    "camera": camera or self.config.camera,
                    "available_keys": sorted(str(key) for key in observation.keys()),
                },
            )

        robot_state = {
            str(key): value
            for key, value in observation.items()
            if key != camera_key and not self._looks_like_image(value)
        }
        self.frame_index += 1
        return InvServoResult.success(
            {
                "backend": self.name,
                "frame_index": self.frame_index,
                "camera": camera_key,
                "image_shape": list(image_rgb.shape),
                "image_rgb": image_rgb,
                "robot_state": robot_state,
                "depth": None,
            }
        )

    def execute_servo_command(self, command: ServoCommand) -> InvServoResult:
        if not self.config.enabled or self.config.require_explicit_enable:
            return InvServoResult.failure(
                "robot_backend_disabled",
                {"backend": self.name, "command": command.to_dict()},
            )
        if self.robot is None:
            return InvServoResult.failure("robot_adapter_missing", {"backend": self.name})
        action = {
            "delta_x": float(command.delta_xyz_m[0]),
            "delta_y": float(command.delta_xyz_m[1]),
            "delta_z": float(command.delta_xyz_m[2]),
            "delta_rx": float(command.delta_rpy_rad[0]),
            "delta_ry": float(command.delta_rpy_rad[1]),
            "delta_rz": float(command.delta_rpy_rad[2]),
            "delta_frame": str(self.config.delta_frame),
        }
        if command.gripper_position is not None:
            action["gripper.pos"] = float(command.gripper_position)
        if self.config.arm_velocity is not None:
            action["arm_velocity"] = float(self.config.arm_velocity)
        if self.config.arm_smooth_factor is not None:
            action["arm_smooth_factor"] = float(self.config.arm_smooth_factor)
        try:
            result = self.robot.send_action(action)
        except Exception as exc:
            return InvServoResult.failure(
                "robot_execute_failed",
                {
                    "backend": self.name,
                    "command": command.to_dict(),
                    "action": action,
                    "error": f"{type(exc).__name__}: {exc}",
                },
            )
        return InvServoResult.success(
            {
                "backend": self.name,
                "accepted": True,
                "command": command.to_dict(),
                "action": action,
                "robot_result": result,
            }
        )

    def close_gripper(self) -> InvServoResult:
        if self.robot is None:
            return InvServoResult.failure("robot_adapter_missing", {"backend": self.name})
        try:
            result = self.robot.send_action({"gripper.pos": -1.0})
        except Exception as exc:
            return InvServoResult.failure(
                "robot_close_gripper_failed",
                {"backend": self.name, "error": f"{type(exc).__name__}: {exc}"},
            )
        return InvServoResult.success({"backend": self.name, "closed": True, "robot_result": result})

    def _resolve_rgb_image(self, observation: dict[str, Any], preferred: str) -> tuple[str | None, np.ndarray | None]:
        candidates = [preferred, *self.config.camera_aliases]
        for value in list(candidates):
            if value.startswith("observation.images."):
                candidates.append(value.removeprefix("observation.images."))
            else:
                candidates.append(f"observation.images.{value}")
            if value.endswith("_camera"):
                candidates.append(value.removesuffix("_camera"))
            else:
                candidates.append(f"{value}_camera")

        for candidate in dict.fromkeys(str(item) for item in candidates if str(item).strip()):
            if candidate not in observation:
                continue
            image_rgb = self._coerce_rgb_image(observation[candidate])
            if image_rgb is not None:
                return candidate, image_rgb
        return None, None

    @staticmethod
    def _coerce_rgb_image(value: Any) -> np.ndarray | None:
        array = np.asarray(value)
        if array.ndim != 3 or array.shape[2] != 3:
            return None
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _looks_like_image(value: Any) -> bool:
        array = np.asarray(value)
        return array.ndim == 3 and array.shape[2] == 3
