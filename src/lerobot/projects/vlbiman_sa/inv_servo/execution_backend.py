from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .target_state import InvServoResult, ServoCommand


@dataclass(slots=True)
class BackendObservation:
    frame_index: int | None = None
    image: Any | None = None
    depth: Any | None = None
    robot_state: dict[str, Any] = field(default_factory=dict)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class BackendStepResult:
    accepted: bool
    command: ServoCommand
    observation: BackendObservation | None = None
    failure_reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_result(self) -> InvServoResult:
        state = {
            "accepted": self.accepted,
            "command": self.command.to_dict(),
            "observation": None if self.observation is None else self.observation.debug,
            "debug": dict(self.debug),
        }
        if not self.accepted:
            return InvServoResult.failure(self.failure_reason or "backend_step_rejected", state)
        return InvServoResult.success(state)


class ExecutionBackend:
    name = "base"

    def reset_to_frame(self, frame_index: int) -> InvServoResult:
        return InvServoResult.failure("backend_reset_not_implemented", {"backend": self.name, "frame_index": frame_index})

    def get_observation(self) -> InvServoResult:
        return InvServoResult.failure("backend_observation_not_implemented", {"backend": self.name})

    def execute_servo_command(self, command: ServoCommand) -> InvServoResult:
        return InvServoResult.failure(
            "backend_execute_not_implemented",
            {"backend": self.name, "command": command.to_dict()},
        )

    def close_gripper(self) -> InvServoResult:
        return InvServoResult.failure("backend_close_gripper_not_implemented", {"backend": self.name})
