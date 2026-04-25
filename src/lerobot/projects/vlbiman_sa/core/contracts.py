from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol


@dataclass
class TaskGraspConfig:
    task_name: str = "single_arm_simple_grasp"
    fps: int = 10
    data_root: Path = Path("outputs/vlbiman_sa")
    transforms_path: Path = Path("src/lerobot/projects/vlbiman_sa/configs/transforms.yaml")
    handeye_result_path: Path = Path("outputs/vlbiman_sa/calib/handeye_result.json")
    recording_session_dir: Path | None = None
    skill_output_dir: Path | None = None
    skill_bank_path: Path | None = None
    vision_output_dir: Path | None = None
    pose_output_dir: Path | None = None
    trajectory_output_dir: Path | None = None
    live_result_path: Path | None = None
    intrinsics_path: Path = Path("src/lerobot/projects/vlbiman_sa/configs/camera_intrinsics.json")
    task_prompt: str = "orange"
    target_phrase: str = "orange"
    # Backward-compatible orientation controls used by existing task YAML files.
    align_target_orientation: bool = False
    primary_orientation_policy: str | None = None
    primary_reference_phrase: str | None = None
    primary_vision_dir_name: str = "t4_vision"
    secondary_target_phrase: str = "pink cup"
    secondary_align_target_orientation: bool = False
    secondary_orientation_policy: str | None = None
    secondary_reference_phrase: str | None = None
    secondary_vision_dir_name: str = "t4_vision_pink_cup"
    camera_serial_number: str | None = None
    robot_type: str = "cjjarm"
    dry_run: bool = True


class SupportsInitialize(Protocol):
    def initialize(self) -> None: ...


class SupportsStep(Protocol):
    def step(self, state: dict[str, Any]) -> dict[str, Any]: ...


class SupportsShutdown(Protocol):
    def shutdown(self) -> None: ...
