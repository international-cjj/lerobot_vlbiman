from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import cv2
import numpy as np

from lerobot.projects.vlbiman_sa.geometry.transforms import invert_transform, make_transform

_METHOD_MAP: dict[str, int] = {
    "tsai": cv2.CALIB_HAND_EYE_TSAI,
    "park": cv2.CALIB_HAND_EYE_PARK,
    "horaud": cv2.CALIB_HAND_EYE_HORAUD,
    "andreff": cv2.CALIB_HAND_EYE_ANDREFF,
    "daniilidis": cv2.CALIB_HAND_EYE_DANIILIDIS,
}


@dataclass
class HandEyeResult:
    transform: np.ndarray
    target_frame: str
    source_frame: str
    setup: str
    method: str
    sample_count: int


def _normalize_matrices(matrices: Iterable[np.ndarray]) -> list[np.ndarray]:
    result = []
    for matrix in matrices:
        matrix = np.asarray(matrix, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"All transforms must be 4x4, got {matrix.shape}.")
        result.append(matrix)
    return result


def _split_rt(matrices: list[np.ndarray]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    rotations = [m[:3, :3].copy() for m in matrices]
    translations = [m[:3, 3].copy().reshape(3, 1) for m in matrices]
    return rotations, translations


def solve_hand_eye(
    base_from_gripper_seq: Iterable[np.ndarray],
    camera_from_target_seq: Iterable[np.ndarray],
    *,
    setup: str = "eye_to_hand",
    method: str = "tsai",
) -> HandEyeResult:
    base_from_gripper = _normalize_matrices(base_from_gripper_seq)
    camera_from_target = _normalize_matrices(camera_from_target_seq)
    if len(base_from_gripper) != len(camera_from_target):
        raise ValueError(
            f"Sample length mismatch: {len(base_from_gripper)} base_from_gripper vs "
            f"{len(camera_from_target)} camera_from_target."
        )
    if len(base_from_gripper) < 3:
        raise ValueError("At least 3 calibration samples are required.")

    method_key = method.lower()
    if method_key not in _METHOD_MAP:
        raise ValueError(f"Unsupported method '{method}'. Expected one of: {sorted(_METHOD_MAP)}.")
    method_code = _METHOD_MAP[method_key]

    setup_key = setup.lower()
    if setup_key not in {"eye_to_hand", "eye_in_hand"}:
        raise ValueError("setup must be either 'eye_to_hand' or 'eye_in_hand'.")

    r_target2cam, t_target2cam = _split_rt(camera_from_target)

    if setup_key == "eye_in_hand":
        r_gripper2base, t_gripper2base = _split_rt(base_from_gripper)
        r_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
            r_gripper2base,
            t_gripper2base,
            r_target2cam,
            t_target2cam,
            method=method_code,
        )
        transform = make_transform(r_cam2gripper, t_cam2gripper.reshape(3))
        return HandEyeResult(
            transform=transform,
            target_frame="gripper",
            source_frame="camera",
            setup=setup_key,
            method=method_key,
            sample_count=len(base_from_gripper),
        )

    # eye_to_hand: static camera, moving gripper. We invert gripper pose and use calibrateHandEye.
    gripper_from_base = [invert_transform(m) for m in base_from_gripper]
    r_base2gripper, t_base2gripper = _split_rt(gripper_from_base)
    r_cam2base, t_cam2base = cv2.calibrateHandEye(
        r_base2gripper,
        t_base2gripper,
        r_target2cam,
        t_target2cam,
        method=method_code,
    )
    transform = make_transform(r_cam2base, t_cam2base.reshape(3))
    return HandEyeResult(
        transform=transform,
        target_frame="base",
        source_frame="camera",
        setup=setup_key,
        method=method_key,
        sample_count=len(base_from_gripper),
    )

