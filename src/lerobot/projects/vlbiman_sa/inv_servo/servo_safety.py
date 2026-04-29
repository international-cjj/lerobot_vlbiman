from __future__ import annotations

from dataclasses import dataclass
import math

from .target_state import InvServoResult, ServoCommand


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


@dataclass(slots=True)
class ServoSafetyConfig:
    max_step_xy_m: float = 0.003
    max_step_z_m: float = 0.004
    max_rotation_rad: float = 0.04
    min_gripper_position: float = 0.0
    max_gripper_position: float = 1.0


class ServoSafetyFilter:
    def __init__(self, config: ServoSafetyConfig | None = None):
        self.config = config or ServoSafetyConfig()

    def filter_command(self, command: ServoCommand) -> InvServoResult:
        values = [*command.delta_xyz_m, *command.delta_rpy_rad]
        if command.gripper_position is not None:
            values.append(command.gripper_position)
        if not all(math.isfinite(float(value)) for value in values):
            return InvServoResult.failure("non_finite_command", {"command": command.to_dict()})

        safe_command = ServoCommand(
            delta_xyz_m=(
                _clamp(command.delta_xyz_m[0], self.config.max_step_xy_m),
                _clamp(command.delta_xyz_m[1], self.config.max_step_xy_m),
                _clamp(command.delta_xyz_m[2], self.config.max_step_z_m),
            ),
            delta_rpy_rad=tuple(_clamp(value, self.config.max_rotation_rad) for value in command.delta_rpy_rad),
            gripper_position=self._clamp_gripper(command.gripper_position),
            stop=command.stop,
            reason=command.reason,
            debug={**command.debug, "safety_filter": "step_clamp"},
        )
        limited = (
            safe_command.delta_xyz_m != command.delta_xyz_m
            or safe_command.delta_rpy_rad != command.delta_rpy_rad
            or safe_command.gripper_position != command.gripper_position
        )
        return InvServoResult.success(
            {"raw_command": command.to_dict(), "safe_command": safe_command.to_dict(), "limited": limited}
        )

    def _clamp_gripper(self, value: float | None) -> float | None:
        if value is None:
            return None
        return max(self.config.min_gripper_position, min(self.config.max_gripper_position, float(value)))
