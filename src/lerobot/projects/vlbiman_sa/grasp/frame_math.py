from __future__ import annotations

import math
from typing import Any

import numpy as np

from .contracts import Pose6D


def wrap_to_pi(angle_rad: float) -> float:
    """Wrap an angle to the closed interval [-pi, pi]."""
    wrapped = (float(angle_rad) + math.pi) % (2.0 * math.pi) - math.pi
    if math.isclose(wrapped, -math.pi, abs_tol=1e-12) and float(angle_rad) > 0.0:
        return math.pi
    return wrapped


def _coerce_pose6d(pose: Pose6D | dict[str, Any]) -> Pose6D:
    if isinstance(pose, Pose6D):
        return pose
    if isinstance(pose, dict):
        return Pose6D(xyz=tuple(pose["xyz"]), rpy=tuple(pose["rpy"]))
    raise TypeError(f"pose must be Pose6D or dict, got {type(pose).__name__}.")


def _ensure_homogeneous_matrix(transform: np.ndarray) -> np.ndarray:
    matrix = np.asarray(transform, dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"Expected a (4, 4) homogeneous transform, got {matrix.shape}.")
    if not np.isfinite(matrix).all():
        raise ValueError("Transform must contain only finite numbers.")
    if not np.allclose(matrix[3], np.array([0.0, 0.0, 0.0, 1.0], dtype=float), atol=1e-9):
        raise ValueError("Homogeneous transform last row must be [0, 0, 0, 1].")
    return matrix


def _rotation_from_rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def pose6d_to_matrix(pose: Pose6D | dict[str, Any]) -> np.ndarray:
    """Convert a Pose6D payload to a 4x4 homogeneous transform."""
    pose6d = _coerce_pose6d(pose)
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = _rotation_from_rpy(*pose6d.rpy)
    transform[:3, 3] = np.asarray(pose6d.xyz, dtype=float)
    return transform


def matrix_to_pose6d(transform: np.ndarray) -> Pose6D:
    """Convert a homogeneous transform to a Pose6D using ZYX yaw-pitch-roll convention."""
    matrix = _ensure_homogeneous_matrix(transform)
    rotation = matrix[:3, :3]
    translation = tuple(float(value) for value in matrix[:3, 3])

    pitch = math.atan2(-float(rotation[2, 0]), math.hypot(float(rotation[0, 0]), float(rotation[1, 0])))
    cos_pitch = math.cos(pitch)

    if abs(cos_pitch) > 1e-9:
        roll = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
        yaw = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    else:
        # In gimbal lock, yaw/roll are coupled. Fix roll=0 to keep the mapping deterministic.
        roll = 0.0
        yaw = math.atan2(-float(rotation[0, 1]), float(rotation[1, 1]))

    return Pose6D(
        xyz=translation,
        rpy=(wrap_to_pi(roll), wrap_to_pi(pitch), wrap_to_pi(yaw)),
    )


def invert_transform(transform: np.ndarray) -> np.ndarray:
    """Invert a homogeneous transform using the rigid-body block structure."""
    matrix = _ensure_homogeneous_matrix(transform)
    rotation = matrix[:3, :3]
    translation = matrix[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transform(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Compose two homogeneous transforms."""
    left_matrix = _ensure_homogeneous_matrix(left)
    right_matrix = _ensure_homogeneous_matrix(right)
    return left_matrix @ right_matrix


__all__ = [
    "compose_transform",
    "invert_transform",
    "matrix_to_pose6d",
    "pose6d_to_matrix",
    "wrap_to_pi",
]
