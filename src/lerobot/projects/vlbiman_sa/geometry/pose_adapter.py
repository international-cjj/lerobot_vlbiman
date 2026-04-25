from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from lerobot.projects.vlbiman_sa.geometry.transforms import make_transform
from lerobot.projects.vlbiman_sa.vision.anchor_estimator import CameraIntrinsics
from lerobot.utils.rotation import Rotation

from .geometry_compensator import GeometryCompensation, GeometryCompensator, GeometryCompensatorConfig, GeometryObservation


@dataclass(slots=True)
class PoseAdapterConfig:
    reference_window_size: int = 4
    max_reference_frame_gap: int = 1
    min_reference_frames: int = 3
    z_rotation_frame: str = "world"
    align_target_orientation: bool = False
    orientation_policy: str | None = None


@dataclass(slots=True)
class DemoPoseFrame:
    frame_index: int
    stable: bool
    segment_id: str | None
    segment_label: str | None
    gripper_pose_matrix: list[list[float]]
    gripper_pose_6d: list[float]
    gripper_yaw_deg: float
    anchor_base_xyz_m: list[float]
    object_orientation_deg: float | None
    relative_xyz_m: list[float]
    depth_m: float | None
    major_axis_px: float | None
    minor_axis_px: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class TargetObservation:
    anchor_base_xyz_m: list[float]
    object_orientation_deg: float | None
    depth_m: float | None
    major_axis_px: float | None
    minor_axis_px: float | None


@dataclass(slots=True)
class PoseAdaptationResult:
    status: str
    alignment_mode: str
    orientation_policy: str
    reference_frame_indices: list[int]
    reference_segment_ids: list[str]
    reference_segment_labels: list[str]
    reference_selection: dict[str, Any]
    demo_anchor_base_xyz_m: list[float]
    target_anchor_base_xyz_m: list[float]
    reference_relative_xyz_m: list[float]
    delta_x_m: list[float]
    delta_theta_deg: float
    delta_h_m: float
    reference_gripper_pose_6d: list[float]
    reference_gripper_yaw_deg: float
    adapted_gripper_pose_6d: list[float]
    adapted_gripper_yaw_deg: float
    applied_yaw_delta_deg: float
    adapted_gripper_matrix: list[list[float]]
    geometry_compensation: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class PoseAdapter:
    def __init__(
        self,
        intrinsics: CameraIntrinsics,
        config: PoseAdapterConfig | None = None,
        geometry_config: GeometryCompensatorConfig | None = None,
    ):
        self.intrinsics = intrinsics
        self.config = config or PoseAdapterConfig()
        self.geometry_compensator = GeometryCompensator(intrinsics, geometry_config)

    def adapt(self, demo_frames: list[DemoPoseFrame], target: TargetObservation) -> PoseAdaptationResult:
        if not demo_frames:
            raise ValueError("No demo frames are available for pose adaptation.")

        selected_frames, selection_summary = self.select_reference_frames(demo_frames)
        if not selected_frames:
            raise ValueError("Failed to select reference frames for pose adaptation.")

        reference_relative = np.median(
            np.asarray([frame.relative_xyz_m for frame in selected_frames], dtype=float),
            axis=0,
        )
        reference_anchor = np.median(
            np.asarray([frame.anchor_base_xyz_m for frame in selected_frames], dtype=float),
            axis=0,
        )
        reference_pose = np.median(
            np.asarray([frame.gripper_pose_6d for frame in selected_frames], dtype=float),
            axis=0,
        )
        reference_rotation = np.asarray(selected_frames[len(selected_frames) // 2].gripper_pose_matrix, dtype=float)[:3, :3]
        reference_object_orientation_deg = _axial_mean_deg(
            [frame.object_orientation_deg for frame in selected_frames if frame.object_orientation_deg is not None]
        )
        orientation_policy = _resolve_orientation_policy(self.config)
        reference_yaw_deg = _matrix_world_yaw_deg(reference_rotation)

        target_anchor = np.asarray(target.anchor_base_xyz_m, dtype=float)
        delta_x = target_anchor - reference_anchor
        delta_theta_deg = 0.0
        if (
            orientation_policy == "target_yaw"
            and reference_object_orientation_deg is not None
            and target.object_orientation_deg is not None
        ):
            delta_theta_deg = _axial_delta_deg(target.object_orientation_deg, reference_object_orientation_deg)

        target_yaw_deg = _resolve_target_yaw_deg(
            orientation_policy=orientation_policy,
            reference_yaw_deg=reference_yaw_deg,
            target_object_delta_deg=delta_theta_deg,
        )
        target_rotation = _apply_world_yaw(reference_rotation, reference_yaw_deg, target_yaw_deg)
        applied_yaw_delta_deg = _wrap_deg(target_yaw_deg - reference_yaw_deg)
        compensation = self.geometry_compensator.compensate(
            GeometryObservation(
                depth_m=float(np.median([frame.depth_m for frame in selected_frames if frame.depth_m is not None]))
                if any(frame.depth_m is not None for frame in selected_frames)
                else None,
                major_axis_px=float(
                    np.median([frame.major_axis_px for frame in selected_frames if frame.major_axis_px is not None])
                )
                if any(frame.major_axis_px is not None for frame in selected_frames)
                else None,
                minor_axis_px=float(
                    np.median([frame.minor_axis_px for frame in selected_frames if frame.minor_axis_px is not None])
                )
                if any(frame.minor_axis_px is not None for frame in selected_frames)
                else None,
            ),
            GeometryObservation(
                depth_m=target.depth_m,
                major_axis_px=target.major_axis_px,
                minor_axis_px=target.minor_axis_px,
            ),
        )

        target_translation = target_anchor - reference_relative
        target_translation = target_translation + np.asarray([0.0, 0.0, compensation.delta_h_m], dtype=float)
        target_pose_matrix = make_transform(target_rotation, target_translation)
        target_pose_6d = np.concatenate(
            [target_translation, Rotation.from_matrix(target_rotation).as_rotvec()],
            axis=0,
        )

        return PoseAdaptationResult(
            status="ok",
            alignment_mode=_alignment_mode_for_policy(orientation_policy),
            orientation_policy=orientation_policy,
            reference_frame_indices=[frame.frame_index for frame in selected_frames],
            reference_segment_ids=[frame.segment_id for frame in selected_frames if frame.segment_id],
            reference_segment_labels=[frame.segment_label for frame in selected_frames if frame.segment_label],
            reference_selection=selection_summary,
            demo_anchor_base_xyz_m=reference_anchor.astype(float).tolist(),
            target_anchor_base_xyz_m=target_anchor.astype(float).tolist(),
            reference_relative_xyz_m=reference_relative.astype(float).tolist(),
            delta_x_m=delta_x.astype(float).tolist(),
            delta_theta_deg=float(delta_theta_deg),
            delta_h_m=float(compensation.delta_h_m),
            reference_gripper_pose_6d=reference_pose.astype(float).tolist(),
            reference_gripper_yaw_deg=float(reference_yaw_deg),
            adapted_gripper_pose_6d=target_pose_6d.astype(float).tolist(),
            adapted_gripper_yaw_deg=float(target_yaw_deg),
            applied_yaw_delta_deg=float(applied_yaw_delta_deg),
            adapted_gripper_matrix=target_pose_matrix.astype(float).tolist(),
            geometry_compensation=compensation.to_dict(),
        )

    def select_reference_frames(self, demo_frames: list[DemoPoseFrame]) -> tuple[list[DemoPoseFrame], dict[str, Any]]:
        valid_frames = [
            frame
            for frame in demo_frames
            if frame.anchor_base_xyz_m and frame.relative_xyz_m and frame.gripper_pose_6d and frame.gripper_pose_matrix
        ]
        if not valid_frames:
            return [], {"reason": "no_valid_demo_frames"}

        preferred = [frame for frame in valid_frames if frame.stable]
        if len(preferred) < self.config.min_reference_frames:
            preferred = valid_frames

        sequences = _split_contiguous(preferred, max_gap=self.config.max_reference_frame_gap)
        window_size = max(1, int(self.config.reference_window_size))
        best_frames: list[DemoPoseFrame] = []
        best_score: tuple[float, float, int] | None = None
        best_meta: dict[str, Any] = {}

        for sequence in sequences:
            if len(sequence) < self.config.min_reference_frames:
                continue
            windows = _sliding_windows(sequence, size=min(window_size, len(sequence)))
            for window in windows:
                relative = np.asarray([frame.relative_xyz_m for frame in window], dtype=float)
                distances = np.linalg.norm(relative, axis=1)
                span_mm = float(np.max(np.ptp(relative, axis=0)) * 1000.0)
                median_distance_mm = float(np.median(distances) * 1000.0)
                stable_count = sum(1 for frame in window if frame.stable)
                score = (median_distance_mm, span_mm, -stable_count)
                if best_score is None or score < best_score:
                    best_score = score
                    best_frames = list(window)
                    best_meta = {
                        "median_object_gripper_distance_mm": median_distance_mm,
                        "relative_span_mm": span_mm,
                        "stable_frame_count": stable_count,
                    }

        if not best_frames:
            fallback = sorted(valid_frames, key=lambda frame: np.linalg.norm(np.asarray(frame.relative_xyz_m, dtype=float)))
            best_frames = fallback[: max(self.config.min_reference_frames, 1)]
            relative = np.asarray([frame.relative_xyz_m for frame in best_frames], dtype=float)
            best_meta = {
                "median_object_gripper_distance_mm": float(np.median(np.linalg.norm(relative, axis=1)) * 1000.0),
                "relative_span_mm": float(np.max(np.ptp(relative, axis=0)) * 1000.0) if len(best_frames) > 1 else 0.0,
                "stable_frame_count": sum(1 for frame in best_frames if frame.stable),
            }

        selection_summary = {
            "window_size": len(best_frames),
            "source": "stable_windows" if any(frame.stable for frame in best_frames) else "distance_fallback",
            "frame_indices": [frame.frame_index for frame in best_frames],
            **best_meta,
        }
        return best_frames, selection_summary


def pose_matrix_to_pose6d(transform: np.ndarray) -> np.ndarray:
    transform = np.asarray(transform, dtype=float)
    rotation = Rotation.from_matrix(transform[:3, :3]).as_rotvec()
    return np.concatenate([transform[:3, 3], rotation], axis=0).astype(float)


def _sliding_windows(sequence: list[DemoPoseFrame], size: int) -> list[list[DemoPoseFrame]]:
    if size <= 0:
        return []
    if len(sequence) <= size:
        return [sequence]
    return [sequence[index : index + size] for index in range(0, len(sequence) - size + 1)]


def _split_contiguous(frames: list[DemoPoseFrame], max_gap: int) -> list[list[DemoPoseFrame]]:
    ordered = sorted(frames, key=lambda frame: frame.frame_index)
    if not ordered:
        return []
    groups: list[list[DemoPoseFrame]] = [[ordered[0]]]
    for frame in ordered[1:]:
        if frame.frame_index - groups[-1][-1].frame_index <= max_gap:
            groups[-1].append(frame)
        else:
            groups.append([frame])
    return groups


def _rotate_about_world_z(rotation: np.ndarray, delta_theta_deg: float) -> np.ndarray:
    delta_rad = math.radians(float(delta_theta_deg))
    cos_theta = math.cos(delta_rad)
    sin_theta = math.sin(delta_rad)
    delta_rotation = np.asarray(
        [
            [cos_theta, -sin_theta, 0.0],
            [sin_theta, cos_theta, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    return delta_rotation @ np.asarray(rotation, dtype=float)


def _matrix_world_yaw_deg(rotation: np.ndarray) -> float:
    rotation = np.asarray(rotation, dtype=float)
    return float(np.degrees(np.arctan2(rotation[1, 0], rotation[0, 0])))


def _apply_world_yaw(reference_rotation: np.ndarray, reference_yaw_deg: float, target_yaw_deg: float) -> np.ndarray:
    tilt_rotation = _rotate_about_world_z(reference_rotation, -reference_yaw_deg)
    return _rotate_about_world_z(tilt_rotation, target_yaw_deg)


def _normalize_orientation_policy(policy: str | None) -> str | None:
    if policy is None:
        return None
    normalized = str(policy).strip().lower().replace("-", "_")
    if not normalized:
        return None
    aliases = {
        "reference": "reference",
        "demo": "reference",
        "keep_demo": "reference",
        "position_only": "reference",
        "target_yaw": "target_yaw",
        "target": "target_yaw",
        "full": "target_yaw",
        "align_target_orientation": "target_yaw",
        "preserve_tilt": "preserve_tilt",
        "tilt_only": "preserve_tilt",
        "relax_yaw": "preserve_tilt",
        "yaw_relaxed": "preserve_tilt",
    }
    if normalized not in aliases:
        raise ValueError(
            "Unsupported orientation policy "
            f"{policy!r}. Expected one of: reference, target_yaw, preserve_tilt."
        )
    return aliases[normalized]


def _resolve_orientation_policy(config: PoseAdapterConfig) -> str:
    explicit = _normalize_orientation_policy(config.orientation_policy)
    if explicit is not None:
        return explicit
    return "target_yaw" if config.align_target_orientation else "reference"


def _resolve_target_yaw_deg(
    *,
    orientation_policy: str,
    reference_yaw_deg: float,
    target_object_delta_deg: float,
) -> float:
    if orientation_policy == "target_yaw":
        return _wrap_deg(reference_yaw_deg + target_object_delta_deg)
    if orientation_policy == "preserve_tilt":
        return 0.0
    return float(reference_yaw_deg)


def _alignment_mode_for_policy(orientation_policy: str) -> str:
    if orientation_policy == "target_yaw":
        return "position+orientation"
    if orientation_policy == "preserve_tilt":
        return "position+tilt_only"
    return "position_only"


def _wrap_deg(angle_deg: float) -> float:
    wrapped = (float(angle_deg) + 180.0) % 360.0 - 180.0
    return float(wrapped)


def _axial_mean_deg(angles_deg: list[float]) -> float | None:
    if not angles_deg:
        return None
    doubled = np.deg2rad(np.asarray(angles_deg, dtype=float) * 2.0)
    mean_vector = np.mean(np.exp(1j * doubled))
    if abs(mean_vector) < 1e-9:
        return None
    return float(np.rad2deg(np.angle(mean_vector)) / 2.0 % 180.0)


def _axial_delta_deg(target_deg: float, reference_deg: float) -> float:
    return float(_wrap_deg(2.0 * (float(target_deg) - float(reference_deg))) / 2.0)
