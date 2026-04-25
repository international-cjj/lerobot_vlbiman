from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def make_transform(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    rotation = np.asarray(rotation, dtype=float)
    translation = np.asarray(translation, dtype=float).reshape(3)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must be (3, 3), got {rotation.shape}")
    t = np.eye(4, dtype=float)
    t[:3, :3] = rotation
    t[:3, 3] = translation
    return t


def ensure_homogeneous_matrix(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"Expected (4, 4) homogeneous transform, got {transform.shape}")
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    transform = ensure_homogeneous_matrix(transform)
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    inverse = np.eye(4, dtype=float)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def compose_transform(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = ensure_homogeneous_matrix(left)
    right = ensure_homogeneous_matrix(right)
    return left @ right


def apply_transform_points(transform: np.ndarray, points_xyz: Iterable[Iterable[float]]) -> np.ndarray:
    transform = ensure_homogeneous_matrix(transform)
    points = np.asarray(points_xyz, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, 3)
    if points.shape[1] != 3:
        raise ValueError(f"Points must have shape (N, 3), got {points.shape}")
    hom = np.hstack([points, np.ones((points.shape[0], 1), dtype=float)])
    transformed = (transform @ hom.T).T
    return transformed[:, :3]


def translation_error_m(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = ensure_homogeneous_matrix(reference)
    estimate = ensure_homogeneous_matrix(estimate)
    return float(np.linalg.norm(reference[:3, 3] - estimate[:3, 3]))


def rotation_error_deg(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = ensure_homogeneous_matrix(reference)
    estimate = ensure_homogeneous_matrix(estimate)
    delta_r = reference[:3, :3].T @ estimate[:3, :3]
    trace = float(np.trace(delta_r))
    cos_theta = max(-1.0, min(1.0, 0.5 * (trace - 1.0)))
    theta = math.acos(cos_theta)
    return float(np.rad2deg(theta))

