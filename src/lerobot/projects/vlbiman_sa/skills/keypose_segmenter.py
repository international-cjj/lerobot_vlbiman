from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord


@dataclass(slots=True)
class SegmenterConfig:
    min_segment_frames: int = 6
    max_segments: int = 12
    velocity_quantile: float = 0.68
    acceleration_quantile: float = 0.78
    transition_quantile: float = 0.76
    gripper_delta_threshold: float = 0.08
    merge_gap_frames: int = 3
    transition_smoothing_window: int = 5
    merge_same_label_segments: bool = True
    merge_short_bridge_segments: bool = True
    short_segment_merge_frames: int = 12
    mergeable_labels: tuple[str, ...] = ("transfer", "approach", "retreat", "stabilize")


@dataclass(slots=True)
class SegmenterFeatures:
    frame_indices: np.ndarray
    relative_time_s: np.ndarray
    joint_matrix: np.ndarray
    gripper_positions: np.ndarray
    velocity_norm: np.ndarray
    acceleration_norm: np.ndarray
    gripper_delta: np.ndarray
    transition_score: np.ndarray
    joint_keys: list[str] = field(default_factory=list)


class KeyposeSegmenter:
    def __init__(self, config: SegmenterConfig | None = None):
        self.config = config or SegmenterConfig()

    def extract_features(self, records: list[FrameRecord]) -> SegmenterFeatures:
        if len(records) < 2:
            raise ValueError("At least two frames are required for segmentation.")

        joint_keys = sorted(records[0].joint_positions.keys())
        if not joint_keys:
            raise ValueError("No joint positions found in recording metadata.")

        joint_matrix = np.asarray(
            [[float(record.joint_positions[key]) for key in joint_keys] for record in records],
            dtype=np.float64,
        )
        relative_time_s = np.asarray([float(record.relative_time_s) for record in records], dtype=np.float64)
        frame_indices = np.asarray([int(record.frame_index) for record in records], dtype=np.int64)
        gripper_positions = np.asarray(
            [
                float(
                    record.gripper_state.get(
                        "gripper.pos",
                        record.robot_observation.get("gripper.pos", 0.0),
                    )
                )
                for record in records
            ],
            dtype=np.float64,
        )

        velocity = np.gradient(joint_matrix, relative_time_s, axis=0, edge_order=1)
        acceleration = np.gradient(velocity, relative_time_s, axis=0, edge_order=1)
        velocity_norm = np.linalg.norm(velocity, axis=1)
        acceleration_norm = np.linalg.norm(acceleration, axis=1)
        gripper_delta = np.abs(np.gradient(gripper_positions, relative_time_s, edge_order=1))

        transition_score = (
            0.5 * self._normalize(velocity_norm)
            + 0.3 * self._normalize(acceleration_norm)
            + 0.2 * self._normalize(gripper_delta)
        )
        transition_score = self._smooth_signal(transition_score)

        return SegmenterFeatures(
            frame_indices=frame_indices,
            relative_time_s=relative_time_s,
            joint_matrix=joint_matrix,
            gripper_positions=gripper_positions,
            velocity_norm=velocity_norm,
            acceleration_norm=acceleration_norm,
            gripper_delta=gripper_delta,
            transition_score=transition_score,
            joint_keys=joint_keys,
        )

    def compute_boundaries(self, features: SegmenterFeatures) -> tuple[list[int], dict[int, list[str]]]:
        cfg = self.config
        frame_count = int(features.frame_indices.shape[0])
        boundary_map: dict[int, list[str]] = {0: ["start"], frame_count - 1: ["end"]}

        velocity_threshold = float(np.quantile(features.velocity_norm, cfg.velocity_quantile))
        acceleration_threshold = float(np.quantile(features.acceleration_norm, cfg.acceleration_quantile))
        transition_threshold = float(np.quantile(features.transition_score, cfg.transition_quantile))

        motion_mask = (
            (features.velocity_norm >= velocity_threshold)
            | (features.acceleration_norm >= acceleration_threshold)
            | (features.gripper_delta >= cfg.gripper_delta_threshold)
        )
        active_diff = np.diff(motion_mask.astype(np.int8), prepend=0)
        for index in np.flatnonzero(active_diff == 1):
            self._append_reason(boundary_map, int(index), "motion_start")
        for index in np.flatnonzero(active_diff == -1):
            self._append_reason(boundary_map, int(max(0, index - 1)), "motion_stop")

        transition_signal = features.transition_score
        for index in range(1, frame_count - 1):
            value = float(transition_signal[index])
            if value < transition_threshold:
                continue
            if value >= float(transition_signal[index - 1]) and value > float(transition_signal[index + 1]):
                self._append_reason(boundary_map, index, "transition_peak")
            if features.gripper_delta[index] >= cfg.gripper_delta_threshold:
                self._append_reason(boundary_map, index, "gripper_event")

        boundaries = self._merge_boundaries(sorted(boundary_map), frame_count)
        boundary_map = {index: boundary_map.get(index, ["merged_boundary"]) for index in boundaries}

        if len(boundaries) < 3:
            midpoint = frame_count // 2
            candidate = self._align_boundary(midpoint, frame_count)
            if candidate not in boundary_map:
                boundary_map[candidate] = ["fallback_midpoint"]
            boundaries = self._merge_boundaries(sorted(boundary_map), frame_count)

        while len(boundaries) - 1 > cfg.max_segments:
            removable = boundaries[1:-1]
            if not removable:
                break
            weakest = min(removable, key=lambda item: float(features.transition_score[item]))
            boundaries.remove(weakest)

        return boundaries, boundary_map

    def _merge_boundaries(self, boundaries: list[int], frame_count: int) -> list[int]:
        merged: list[int] = [0]
        min_gap = max(1, int(self.config.min_segment_frames))
        for candidate in boundaries[1:]:
            aligned = self._align_boundary(candidate, frame_count)
            if aligned <= merged[-1]:
                continue
            if aligned - merged[-1] < min_gap:
                continue
            merged.append(aligned)
        if merged[-1] != frame_count - 1:
            merged.append(frame_count - 1)
        return merged

    @staticmethod
    def _append_reason(boundary_map: dict[int, list[str]], index: int, reason: str) -> None:
        boundary_map.setdefault(index, [])
        if reason not in boundary_map[index]:
            boundary_map[index].append(reason)

    @staticmethod
    def _align_boundary(index: int, frame_count: int) -> int:
        return int(min(max(0, index), frame_count - 1))

    @staticmethod
    def _normalize(values: np.ndarray) -> np.ndarray:
        if values.size == 0:
            return values
        lower = float(np.min(values))
        upper = float(np.max(values))
        if upper - lower <= 1e-9:
            return np.zeros_like(values, dtype=np.float64)
        return (values - lower) / (upper - lower)

    def _smooth_signal(self, values: np.ndarray) -> np.ndarray:
        window = max(1, int(self.config.transition_smoothing_window))
        if window <= 1 or values.size <= 2:
            return values
        if window % 2 == 0:
            window += 1
        kernel = np.ones(window, dtype=np.float64) / float(window)
        return np.convolve(values, kernel, mode="same")
