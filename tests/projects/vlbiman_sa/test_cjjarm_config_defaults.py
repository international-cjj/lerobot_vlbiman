from __future__ import annotations

from lerobot.cameras.configs import ColorMode
from lerobot_robot_cjjarm import CjjArmConfig, dabaidcw_rgb_camera_config


def test_gripper_trigger_threshold_default_is_half_previous_magnitude() -> None:
    config = CjjArmConfig()
    assert config.gripper_trigger_threshold == -0.15


def test_default_wrist_camera_is_dabaidcw_rgb_opencv_config() -> None:
    config = CjjArmConfig()
    wrist = config.cameras["wrist"]

    assert wrist.type == "opencv"
    assert wrist.width == 1920
    assert wrist.height == 1080
    assert wrist.fps == 30
    assert wrist.color_mode == ColorMode.RGB
    assert wrist.fourcc == "MJPG"


def test_dabaidcw_rgb_camera_config_allows_numeric_index_override() -> None:
    camera = dabaidcw_rgb_camera_config(index_or_path="1")

    assert camera.type == "opencv"
    assert camera.index_or_path == 1
    assert camera.color_mode == ColorMode.RGB
