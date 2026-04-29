from __future__ import annotations

from dataclasses import dataclass

from .execution_backend import ExecutionBackend
from .rgb_servo_controller import RGBServoController, RGBServoControllerConfig
from .servo_safety import ServoSafetyConfig, ServoSafetyFilter
from .sim_backend import SimExecutionBackend
from .target_state import InvServoResult, MaskState, ServoCommand, ServoTarget


@dataclass(slots=True)
class InvGraspExecutorConfig:
    max_steps: int = 120
    stable_frames: int = 5
    close_gripper_on_converged: bool = True


class InvGraspExecutor:
    def __init__(
        self,
        config: InvGraspExecutorConfig | None = None,
        controller: RGBServoController | None = None,
        safety_filter: ServoSafetyFilter | None = None,
        backend: ExecutionBackend | None = None,
    ):
        self.config = config or InvGraspExecutorConfig()
        self.controller = controller or RGBServoController(RGBServoControllerConfig())
        self.safety_filter = safety_filter or ServoSafetyFilter(ServoSafetyConfig())
        self.backend = backend or SimExecutionBackend()

    def run_step(self, current_mask: MaskState, target: ServoTarget, *, mask_iou: float | None = None) -> InvServoResult:
        command_result = self.controller.compute_command(current_mask, target, mask_iou=mask_iou)
        if not command_result.ok or command_result.state is None:
            return command_result

        command_state = command_result.state["command"]
        command = ServoCommand(
            delta_xyz_m=tuple(command_state["delta_xyz_m"]),
            delta_rpy_rad=tuple(command_state["delta_rpy_rad"]),
            gripper_position=command_state["gripper_position"],
            stop=command_state["stop"],
            reason=command_state["reason"],
            debug=command_state["debug"],
        )
        safe_result = self.safety_filter.filter_command(command)
        if not safe_result.ok or safe_result.state is None:
            return safe_result

        safe_command_state = safe_result.state["safe_command"]
        safe_command = ServoCommand(
            delta_xyz_m=tuple(safe_command_state["delta_xyz_m"]),
            delta_rpy_rad=tuple(safe_command_state["delta_rpy_rad"]),
            gripper_position=safe_command_state["gripper_position"],
            stop=safe_command_state["stop"],
            reason=safe_command_state["reason"],
            debug=safe_command_state["debug"],
        )
        backend_result = self.backend.execute_servo_command(safe_command)
        return InvServoResult.success(
            {
                "controller": command_result.state,
                "safety": safe_result.state,
                "backend": backend_result.to_dict(),
            }
        )

    def close_gripper(self) -> InvServoResult:
        if not self.config.close_gripper_on_converged:
            return InvServoResult.success({"closed": False, "reason": "close_gripper_disabled"})
        return self.backend.close_gripper()
