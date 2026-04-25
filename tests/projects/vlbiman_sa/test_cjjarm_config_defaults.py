from __future__ import annotations

from lerobot_robot_cjjarm import CjjArmConfig


def test_gripper_trigger_threshold_default_is_half_previous_magnitude() -> None:
    config = CjjArmConfig()
    assert config.gripper_trigger_threshold == -0.15
