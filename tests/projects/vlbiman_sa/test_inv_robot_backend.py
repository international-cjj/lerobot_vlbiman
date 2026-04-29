from __future__ import annotations

import numpy as np

from lerobot.projects.vlbiman_sa.inv_servo.robot_backend import RobotBackendConfig, RobotExecutionBackend


class FakeRobot:
    def __init__(self, image: np.ndarray):
        self.image = image

    def get_observation(self) -> dict[str, object]:
        return {
            "joint_1.pos": 0.1,
            "wrist": self.image,
            "depth": np.zeros(self.image.shape[:2], dtype=np.uint16),
        }


def test_robot_backend_reads_wrist_rgb_frame_from_memory() -> None:
    image = np.full((8, 12, 3), 127, dtype=np.uint8)
    backend = RobotExecutionBackend(FakeRobot(image), RobotBackendConfig(camera="wrist"))

    result = backend.get_rgb_frame()

    assert result.ok
    assert result.state is not None
    assert result.state["camera"] == "wrist"
    assert result.state["image_shape"] == [8, 12, 3]
    assert np.array_equal(result.state["image_rgb"], image)
    assert result.state["depth"] is None


def test_robot_backend_accepts_dabaidcw_rgb_alias() -> None:
    image = np.zeros((4, 6, 3), dtype=np.uint8)
    backend = RobotExecutionBackend(FakeRobot(image), RobotBackendConfig(camera="dabaidcw_rgb"))

    result = backend.get_observation()

    assert result.ok
    assert result.state is not None
    assert result.state["camera"] == "wrist"
