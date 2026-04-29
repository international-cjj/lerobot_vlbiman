from __future__ import annotations

import numpy as np

from lerobot_robot_cjjarm.kinematics import compose_pose_delta


def test_compose_pose_delta_can_translate_in_tool_frame() -> None:
    base_pose = np.array([0.4, 0.1, 0.2, 0.0, 0.0, np.pi / 2.0], dtype=float)
    delta_pose = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float)

    world_pose = compose_pose_delta(base_pose, delta_pose, translation_frame="world")
    tool_pose = compose_pose_delta(base_pose, delta_pose, translation_frame="tool")

    np.testing.assert_allclose(world_pose[:3], [0.41, 0.1, 0.2], atol=1e-9)
    np.testing.assert_allclose(tool_pose[:3], [0.4, 0.11, 0.2], atol=1e-9)
