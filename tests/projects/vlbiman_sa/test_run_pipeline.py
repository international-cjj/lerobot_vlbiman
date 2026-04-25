from __future__ import annotations

from argparse import Namespace

from lerobot.projects.vlbiman_sa.app.run_pipeline import build_stage_command


def test_build_stage_command_record_applies_pipeline_and_cli_overrides():
    payload = {
        "hardware": {
            "camera_serial_number": "cam-from-hardware",
            "robot_serial_port": "/dev/ttyACM0",
            "teleop_port": "/dev/ttyUSB1",
        },
        "stages": {
            "record": {
                "config": "src/lerobot/projects/vlbiman_sa/configs/one_shot_record.yaml",
                "duration_s": 45.0,
                "teleop_port": "/dev/ttyUSB0",
                "no_teleop_calibrate": False,
            }
        },
    }
    args = Namespace(
        control_rate_hz=40.0,
        duration_s=30.0,
        max_frames=None,
        camera_serial_number=None,
        robot_serial_port="/dev/ttyACM1",
        teleop_port=None,
        no_teleop_calibrate=True,
        dry_run=False,
        log_level="INFO",
    )

    cmd = build_stage_command("record", payload, args)

    assert "--config" in cmd
    assert "--duration-s" in cmd
    assert "30.0" in cmd
    assert "--control-rate-hz" in cmd
    assert "40.0" in cmd
    assert "--camera-serial-number" in cmd
    assert "cam-from-hardware" in cmd
    assert "--robot-serial-port" in cmd
    assert "/dev/ttyACM1" in cmd
    assert "--teleop-port" in cmd
    assert "/dev/ttyUSB0" in cmd
    assert "--no-teleop-calibrate" in cmd


def test_build_stage_command_grasp_supports_dry_run_override():
    payload = {
        "stages": {
            "grasp": {
                "config": "src/lerobot/projects/vlbiman_sa/configs/task_grasp.yaml",
                "dry_run": False,
            }
        }
    }
    args = Namespace(
        control_rate_hz=None,
        duration_s=None,
        max_frames=None,
        camera_serial_number=None,
        robot_serial_port=None,
        teleop_port=None,
        no_teleop_calibrate=False,
        dry_run=True,
        log_level="DEBUG",
    )

    cmd = build_stage_command("grasp", payload, args)

    assert "--config" in cmd
    assert "--dry-run" in cmd
    assert "--log-level" in cmd
    assert "DEBUG" in cmd
