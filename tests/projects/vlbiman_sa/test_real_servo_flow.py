from __future__ import annotations

import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

os.environ.setdefault("PYTHON_BIN", sys.executable)

from lerobot.projects.vlbiman_sa.app.override_t3_segments import _parse_segment
from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import _filter_skill_bank_segments
from lerobot.projects.vlbiman_sa.app.run_vlbiman_real_servo_flow import (
    _apply_servo_cli_overrides,
    _execute_servo_command_smooth,
    _find_servo_segment,
    _interpolated_servo_commands,
    _parse_frame_range_1based,
    _shape_servo_command,
    _split_pre_servo_points,
)
from lerobot.projects.vlbiman_sa.inv_servo.config import ServoConfig
from lerobot.projects.vlbiman_sa.inv_servo.robot_backend import RobotBackendConfig, RobotExecutionBackend
from lerobot.projects.vlbiman_sa.inv_servo.target_state import ServoCommand
from lerobot.projects.vlbiman_sa.skills import SkillBank, SkillSegment
from lerobot.projects.vlbiman_sa.trajectory.trajectory_composer import _is_visual_servo_segment


def _segment(segment_id: str, start: int, end: int, label: str, semantic_state: str) -> SkillSegment:
    return SkillSegment(
        segment_id=segment_id,
        start_frame=start,
        end_frame=end,
        start_time_s=float(start),
        end_time_s=float(end),
        representative_frame=(start + end) // 2,
        label=label,
        invariance="inv",
        confidence=1.0,
        frame_count=end - start + 1,
        joint_keys=[f"joint_{index}.pos" for index in range(1, 7)],
        metrics={"semantic_state": semantic_state},
    )


def _bank() -> SkillBank:
    return SkillBank(
        session_dir=Path("session"),
        output_dir=Path("session/analysis/t3_skill_bank"),
        frame_count=30,
        joint_keys=[f"joint_{index}.pos" for index in range(1, 7)],
        segments=[
            _segment("skill_000", 0, 9, "approach_redcan", "approach"),
            _segment("skill_001", 10, 14, "servo_redcan", "visual_servo"),
            _segment("skill_002", 15, 20, "grasp_redcan", "grasp"),
        ],
        summary={},
    )


def test_override_segment_accepts_visual_servo_target_metadata() -> None:
    spec = _parse_segment("51-55:servo_redcan:visual_servo:inv:redcan:55")

    assert spec.semantic_state == "visual_servo"
    assert spec.target_phrase == "redcan"
    assert spec.target_frame_1based == 55


def test_find_servo_segment_by_annotation_or_frame_range() -> None:
    bank = _bank()

    assert _is_visual_servo_segment(bank.segments[1])
    assert _find_servo_segment(bank, servo_segment_id=None, servo_frame_range=None).segment_id == "skill_001"
    assert _parse_frame_range_1based("11-12") == (10, 11)
    assert (
        _find_servo_segment(bank, servo_segment_id=None, servo_frame_range=_parse_frame_range_1based("11-12")).segment_id
        == "skill_001"
    )


def test_split_pre_servo_points_omits_servo_and_later_segments() -> None:
    points = [
        {"trajectory_index": 0, "segment_id": "skill_000", "joint_positions": [0.0] * 6},
        {"trajectory_index": 1, "segment_id": "skill_001", "joint_positions": [1.0] * 6},
        {"trajectory_index": 2, "segment_id": "skill_002", "joint_positions": [2.0] * 6},
    ]

    pre_points = _split_pre_servo_points(points, _bank(), "skill_001")

    assert [point["segment_id"] for point in pre_points] == ["skill_000"]


def test_servo_cli_overrides_include_proportional_gains() -> None:
    config = SimpleNamespace(servo=ServoConfig())
    args = SimpleNamespace(
        servo_k_u=0.021,
        servo_k_v=0.022,
        servo_k_a=0.018,
        servo_axis_sign_x=-1.0,
        servo_axis_sign_y=-1.0,
        servo_axis_sign_z=1.0,
        servo_max_step_xy_m=0.02,
        servo_max_step_z_m=0.03,
    )

    _apply_servo_cli_overrides(config, args)

    assert config.servo.k_u == 0.021
    assert config.servo.k_v == 0.022
    assert config.servo.k_a == 0.018
    assert config.servo.axis_sign_x == -1.0
    assert config.servo.max_step_z_m == 0.03


def test_interpolated_servo_commands_preserve_total_delta() -> None:
    command = ServoCommand(
        delta_xyz_m=(0.003, -0.006, 0.009),
        delta_rpy_rad=(0.01, -0.02, 0.03),
    )

    subcommands = _interpolated_servo_commands(command, duration_s=0.1, interpolation_hz=30.0, profile="smootherstep")

    assert len(subcommands) == 3
    assert np.allclose(np.sum([item.delta_xyz_m for item in subcommands], axis=0), command.delta_xyz_m)
    assert np.allclose(np.sum([item.delta_rpy_rad for item in subcommands], axis=0), command.delta_rpy_rad)
    assert subcommands[0].delta_xyz_m[2] < subcommands[1].delta_xyz_m[2]


def test_default_interpolated_servo_commands_use_constant_increments() -> None:
    command = ServoCommand(delta_xyz_m=(0.003, -0.006, 0.009))

    subcommands = _interpolated_servo_commands(command, duration_s=0.1, interpolation_hz=30.0)

    assert len(subcommands) == 3
    assert np.allclose([item.delta_xyz_m[2] for item in subcommands], [0.003, 0.003, 0.003])


def test_servo_command_shaping_filters_deadbands_and_limits_changes() -> None:
    previous = ServoCommand(delta_xyz_m=(0.0, 0.0010, 0.0020))
    command = ServoCommand(delta_xyz_m=(0.00005, 0.0020, 0.0060))

    shaped, shaping = _shape_servo_command(
        command,
        previous_command=previous,
        filter_alpha=0.5,
        deadband_xy_m=0.00008,
        deadband_z_m=0.00015,
        max_change_xy_m=0.00035,
        max_change_z_m=0.0015,
    )

    assert shaped.delta_xyz_m[0] == 0.0
    assert np.isclose(shaped.delta_xyz_m[1], 0.00135)
    assert np.isclose(shaped.delta_xyz_m[2], 0.0035)
    assert shaping["raw_delta_xyz_m"] == [0.00005, 0.002, 0.006]


def test_t6_suffix_filter_starts_after_servo_segment() -> None:
    filtered, summary = _filter_skill_bank_segments(
        _bank(),
        start_segment_id=None,
        start_after_segment_id="skill_001",
    )

    assert [segment.segment_id for segment in filtered.segments] == ["skill_002"]
    assert summary["mode"] == "suffix_after"
    assert summary["first_selected_segment_id"] == "skill_002"


class FakeRobot:
    def __init__(self) -> None:
        self.actions: list[dict[str, float]] = []

    def get_observation(self) -> dict[str, object]:
        return {"wrist": np.zeros((4, 4, 3), dtype=np.uint8)}

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.actions.append(action)
        return action


def test_smooth_servo_execution_sends_interpolated_substeps() -> None:
    robot = FakeRobot()
    backend = RobotExecutionBackend(
        robot,
        RobotBackendConfig(enabled=True, require_explicit_enable=False, camera="wrist"),
    )

    result, timing = _execute_servo_command_smooth(
        backend=backend,
        command=ServoCommand(delta_xyz_m=(0.003, -0.006, 0.009)),
        step_duration_s=0.001,
        interpolation_hz=3000.0,
        interpolation_profile="linear",
    )

    assert result.ok
    assert timing["substep_count"] == 3
    assert len(robot.actions) == 3
    assert np.isclose(sum(action["delta_x"] for action in robot.actions), 0.003)
    assert np.isclose(sum(action["delta_y"] for action in robot.actions), -0.006)
    assert np.isclose(sum(action["delta_z"] for action in robot.actions), 0.009)


def test_robot_backend_executes_pose_delta_command() -> None:
    robot = FakeRobot()
    backend = RobotExecutionBackend(
        robot,
        RobotBackendConfig(enabled=True, require_explicit_enable=False, camera="wrist"),
    )

    result = backend.execute_servo_command(
        ServoCommand(delta_xyz_m=(0.001, -0.002, 0.003), delta_rpy_rad=(0.01, 0.0, -0.01))
    )

    assert result.ok
    assert robot.actions == [
        {
            "delta_x": 0.001,
            "delta_y": -0.002,
            "delta_z": 0.003,
            "delta_rx": 0.01,
            "delta_ry": 0.0,
            "delta_rz": -0.01,
            "delta_frame": "tool",
        }
    ]


def test_robot_backend_can_override_servo_arm_velocity() -> None:
    robot = FakeRobot()
    backend = RobotExecutionBackend(
        robot,
        RobotBackendConfig(
            enabled=True,
            require_explicit_enable=False,
            camera="wrist",
            arm_velocity=5.4,
        ),
    )

    result = backend.execute_servo_command(ServoCommand(delta_xyz_m=(0.001, 0.0, 0.0)))

    assert result.ok
    assert robot.actions[-1]["arm_velocity"] == 5.4


def test_robot_backend_can_override_servo_arm_smooth_factor() -> None:
    robot = FakeRobot()
    backend = RobotExecutionBackend(
        robot,
        RobotBackendConfig(
            enabled=True,
            require_explicit_enable=False,
            camera="wrist",
            arm_smooth_factor=0.35,
        ),
    )

    result = backend.execute_servo_command(ServoCommand(delta_xyz_m=(0.001, 0.0, 0.0)))

    assert result.ok
    assert robot.actions[-1]["arm_smooth_factor"] == 0.35
