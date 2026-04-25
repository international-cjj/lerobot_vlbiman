from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np


@dataclass(slots=True)
class MaskTrackerConfig:
    stability_window_size: int = 5
    position_variance_threshold_mm2: float = 100.0
    orientation_variance_threshold_deg2: float = 225.0


@dataclass(slots=True)
class TrackedMaskFrame:
    frame_index: int
    mask_area_px: int
    centroid_px: list[float] | None
    temporal_iou: float
    stable: bool
    position_std_mm: float | None
    position_variance_mm2: float | None
    orientation_std_deg: float | None
    orientation_variance_deg2: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MaskTracker:
    def __init__(self, config: MaskTrackerConfig | None = None):
        self.config = config or MaskTrackerConfig()

    def track(
        self,
        masks: list[tuple[int, np.ndarray]],
        anchors: dict[int, dict[str, Any]] | None = None,
    ) -> tuple[list[tuple[int, np.ndarray]], list[TrackedMaskFrame]]:
        tracked_meta: list[TrackedMaskFrame] = []
        previous_mask: np.ndarray | None = None
        anchors = anchors or {}

        for index, (frame_index, mask) in enumerate(masks):
            ys, xs = np.nonzero(mask > 0)
            centroid = [float(xs.mean()), float(ys.mean())] if xs.size else None
            temporal_iou = self._mask_iou(previous_mask, mask) if previous_mask is not None else 1.0
            stable, position_std_mm, position_variance_mm2, orientation_std_deg, orientation_variance_deg2 = (
                self._stability_for_index(index, masks, anchors)
            )
            tracked_meta.append(
                TrackedMaskFrame(
                    frame_index=frame_index,
                    mask_area_px=int(xs.size),
                    centroid_px=centroid,
                    temporal_iou=float(temporal_iou),
                    stable=stable,
                    position_std_mm=position_std_mm,
                    position_variance_mm2=position_variance_mm2,
                    orientation_std_deg=orientation_std_deg,
                    orientation_variance_deg2=orientation_variance_deg2,
                )
            )
            previous_mask = mask
        return masks, tracked_meta

    def _stability_for_index(
        self,
        current_index: int,
        masks: list[tuple[int, np.ndarray]],
        anchors: dict[int, dict[str, Any]],
    ) -> tuple[bool, float | None, float | None, float | None, float | None]:
        window_size = max(1, int(self.config.stability_window_size))
        if current_index + 1 < window_size:
            return False, None, None, None, None

        window = masks[current_index + 1 - window_size : current_index + 1]
        positions = []
        orientations = []
        for frame_index, _ in window:
            anchor = anchors.get(frame_index, {})
            xyz = anchor.get("camera_xyz_m")
            angle_deg = anchor.get("orientation_deg")
            if xyz is not None:
                positions.append(np.asarray(xyz, dtype=np.float64))
            if angle_deg is not None:
                orientations.append(float(angle_deg))

        if len(positions) != window_size or len(orientations) != window_size:
            return False, None, None, None, None

        stacked_positions = np.stack(positions, axis=0)
        position_variance_mm2 = float(np.max(np.var(stacked_positions, axis=0)) * 1_000_000.0)
        position_std_mm = float(np.sqrt(max(position_variance_mm2, 0.0)))
        orientation_variance_deg2, orientation_std_deg = self._axial_variance_deg2(orientations)
        stable = (
            position_variance_mm2 <= float(self.config.position_variance_threshold_mm2)
            and orientation_variance_deg2 is not None
            and orientation_variance_deg2 <= float(self.config.orientation_variance_threshold_deg2)
        )
        return stable, position_std_mm, position_variance_mm2, orientation_std_deg, orientation_variance_deg2

    @staticmethod
    def _axial_variance_deg2(values: list[float]) -> tuple[float | None, float | None]:
        if not values:
            return None, None
        radians = np.radians(np.asarray(values, dtype=np.float64))
        doubled = 2.0 * radians
        mean_sin = float(np.mean(np.sin(doubled)))
        mean_cos = float(np.mean(np.cos(doubled)))
        if np.hypot(mean_sin, mean_cos) <= 1e-9:
            return 8100.0, 90.0

        mean_angle_deg = float(np.degrees(np.arctan2(mean_sin, mean_cos) / 2.0))
        diffs_deg = np.asarray(
            [((value - mean_angle_deg + 90.0) % 180.0) - 90.0 for value in values],
            dtype=np.float64,
        )
        variance_deg2 = float(np.mean(np.square(diffs_deg)))
        return variance_deg2, float(np.sqrt(max(variance_deg2, 0.0)))

    @staticmethod
    def _mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float:
        if mask_a is None or mask_b is None:
            return 0.0
        inter = int(np.count_nonzero((mask_a > 0) & (mask_b > 0)))
        union = int(np.count_nonzero((mask_a > 0) | (mask_b > 0)))
        if union == 0:
            return 0.0
        return inter / union
