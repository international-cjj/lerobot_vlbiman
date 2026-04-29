from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


@dataclass(slots=True)
class RecorderConfig:
    task_name: str = "one_shot_demo"
    fps: int = 10
    control_rate_hz: float = 30.0
    duration_s: float = 60.0
    max_frames: int | None = None
    output_root: Path = Path("outputs/vlbiman_sa/recordings")
    run_name: str | None = None
    camera_timeout_ms: int = 500
    max_time_skew_ms: float = 50.0
    max_consecutive_failures: int = 5
    wait_for_start_space: bool = False
    overwrite: bool = False

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"fps must be positive, got {self.fps}.")
        if self.control_rate_hz <= 0:
            raise ValueError(f"control_rate_hz must be positive, got {self.control_rate_hz}.")
        if self.duration_s <= 0 and self.max_frames is None:
            raise ValueError("duration_s must be positive when max_frames is not set.")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError(f"max_frames must be positive, got {self.max_frames}.")
        if self.camera_timeout_ms <= 0:
            raise ValueError(f"camera_timeout_ms must be positive, got {self.camera_timeout_ms}.")
        if self.max_consecutive_failures < 0:
            raise ValueError(
                f"max_consecutive_failures must be non-negative, got {self.max_consecutive_failures}."
            )

    def resolve_frame_slots(self) -> int:
        if self.max_frames is not None:
            return int(self.max_frames)
        return max(1, int(math.ceil(float(self.duration_s) * float(self.fps))))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["output_root"] = str(self.output_root)
        payload["frame_slots"] = self.resolve_frame_slots()
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class FrameRecord:
    frame_index: int
    slot_index: int
    wall_time_ns: int
    relative_time_s: float
    scheduled_time_s: float
    capture_started_ns: int
    capture_ended_ns: int
    capture_latency_ms: float
    camera_timestamp_ns: int
    robot_timestamp_ns: int
    time_skew_ms: float
    action_timestamp_ns: int | None = None
    action_sent_timestamp_ns: int | None = None
    robot_observation: dict[str, float] = field(default_factory=dict)
    joint_positions: dict[str, float] = field(default_factory=dict)
    gripper_state: dict[str, float] = field(default_factory=dict)
    teleop_action: dict[str, float] = field(default_factory=dict)
    sent_action: dict[str, float] = field(default_factory=dict)
    ee_pose: list[float] | None = None
    color_path: str = ""
    depth_path: str = ""
    color_shape: list[int] = field(default_factory=list)
    depth_shape: list[int] = field(default_factory=list)
    camera_assets: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema_version"] = SCHEMA_VERSION
        return payload


@dataclass(slots=True)
class RecordingSummary:
    status: str
    target_frame_slots: int
    recorded_frames: int
    dropped_frames: int
    failed_frames: int
    achieved_fps: float
    average_time_skew_ms: float
    max_time_skew_ms: float
    started_at_ns: int
    ended_at_ns: int
    session_dir: Path
    metadata_path: Path
    manifest_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["session_dir"] = str(self.session_dir)
        payload["metadata_path"] = str(self.metadata_path)
        payload["manifest_path"] = str(self.manifest_path)
        payload["schema_version"] = SCHEMA_VERSION
        return payload
