from __future__ import annotations

import numpy as np

from lerobot.projects.vlbiman_sa.geometry.frame_manager import FrameManager
from lerobot.projects.vlbiman_sa.geometry.transforms import invert_transform, make_transform


def test_frame_manager_can_chain_and_invert_transforms():
    manager = FrameManager()

    base_from_camera = make_transform(np.eye(3), np.array([1.0, 0.0, 0.0]))
    world_from_base = make_transform(np.eye(3), np.array([0.0, 2.0, 0.0]))
    manager.set_transform("base", "camera", base_from_camera)
    manager.set_transform("world", "base", world_from_base)

    world_from_camera = manager.get_transform("world", "camera")
    expected = world_from_base @ base_from_camera
    assert np.allclose(world_from_camera, expected)

    camera_from_world = manager.get_transform("camera", "world")
    assert np.allclose(camera_from_world, invert_transform(expected))


def test_frame_manager_transform_points():
    manager = FrameManager()
    base_from_camera = make_transform(np.eye(3), np.array([1.0, 2.0, 3.0]))
    manager.set_transform("base", "camera", base_from_camera)

    camera_points = np.array([[0.0, 0.0, 0.0], [1.0, -1.0, 0.5]])
    base_points = manager.transform_points("base", "camera", camera_points)
    assert np.allclose(base_points, np.array([[1.0, 2.0, 3.0], [2.0, 1.0, 3.5]]))

