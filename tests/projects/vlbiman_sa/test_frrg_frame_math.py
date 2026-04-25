from __future__ import annotations

import math

import numpy as np

from lerobot.projects.vlbiman_sa.grasp.contracts import Pose6D
from lerobot.projects.vlbiman_sa.grasp.frame_math import (
    compose_transform,
    invert_transform,
    matrix_to_pose6d,
    pose6d_to_matrix,
    wrap_to_pi,
)


def test_identity_pose_round_trip_is_stable():
    identity_pose = Pose6D.zeros()

    transform = pose6d_to_matrix(identity_pose)
    recovered = matrix_to_pose6d(transform)

    assert np.allclose(transform, np.eye(4), atol=1e-9)
    assert recovered == identity_pose


def test_pose_round_trip_preserves_translation_and_rpy():
    pose = Pose6D(xyz=(0.42, -0.13, 0.08), rpy=(0.3, -0.2, 0.5))

    recovered = matrix_to_pose6d(pose6d_to_matrix(pose))

    assert np.allclose(recovered.xyz, pose.xyz, atol=1e-9)
    assert np.allclose(recovered.rpy, pose.rpy, atol=1e-9)


def test_invert_transform_composes_back_to_identity():
    transform = pose6d_to_matrix(Pose6D(xyz=(0.25, 0.04, -0.11), rpy=(0.2, -0.35, 0.4)))

    product = compose_transform(transform, invert_transform(transform))

    assert np.allclose(product, np.eye(4), atol=1e-9)


def test_wrap_to_pi_stays_inside_closed_interval():
    angles = [0.0, math.pi, -math.pi, 3.0 * math.pi, -3.0 * math.pi, 2.5 * math.pi, -2.5 * math.pi]
    wrapped = [wrap_to_pi(angle) for angle in angles]

    assert all(-math.pi <= angle <= math.pi for angle in wrapped)
    assert math.isclose(wrapped[1], math.pi, abs_tol=1e-9)
    assert math.isclose(wrapped[2], -math.pi, abs_tol=1e-9)
    assert math.isclose(wrapped[3], math.pi, abs_tol=1e-9)
    assert math.isclose(wrapped[4], -math.pi, abs_tol=1e-9)


def test_matrix_to_pose6d_handles_gimbal_lock_deterministically():
    pose = Pose6D(xyz=(0.0, 0.0, 0.0), rpy=(0.6, math.pi / 2.0, -0.2))

    recovered = matrix_to_pose6d(pose6d_to_matrix(pose))

    assert math.isclose(recovered.rpy[0], 0.0, abs_tol=1e-9)
    assert math.isclose(recovered.rpy[1], math.pi / 2.0, abs_tol=1e-9)
    assert np.allclose(pose6d_to_matrix(recovered), pose6d_to_matrix(pose), atol=1e-9)
