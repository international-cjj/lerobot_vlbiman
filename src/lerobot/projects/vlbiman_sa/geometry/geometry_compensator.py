from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from lerobot.projects.vlbiman_sa.vision.anchor_estimator import CameraIntrinsics


@dataclass(slots=True)
class GeometryObservation:
    depth_m: float | None
    major_axis_px: float | None
    minor_axis_px: float | None


@dataclass(slots=True)
class GeometryCompensatorConfig:
    size_metric: str = "mean_axis"
    height_gain: float = 0.5
    max_height_adjustment_m: float = 0.03


@dataclass(slots=True)
class GeometryCompensation:
    reference_size_m: float | None
    target_size_m: float | None
    delta_size_m: float
    delta_h_m: float
    size_ratio: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class GeometryCompensator:
    def __init__(self, intrinsics: CameraIntrinsics, config: GeometryCompensatorConfig | None = None):
        self.intrinsics = intrinsics
        self.config = config or GeometryCompensatorConfig()

    def compensate(
        self,
        reference: GeometryObservation,
        target: GeometryObservation,
    ) -> GeometryCompensation:
        reference_size_m = self.estimate_size_m(reference)
        target_size_m = self.estimate_size_m(target)
        if reference_size_m is None or target_size_m is None:
            return GeometryCompensation(
                reference_size_m=reference_size_m,
                target_size_m=target_size_m,
                delta_size_m=0.0,
                delta_h_m=0.0,
                size_ratio=None,
            )

        delta_size_m = float(target_size_m - reference_size_m)
        delta_h_m = float(self.config.height_gain * delta_size_m)
        max_adjustment = abs(float(self.config.max_height_adjustment_m))
        delta_h_m = max(-max_adjustment, min(max_adjustment, delta_h_m))
        return GeometryCompensation(
            reference_size_m=float(reference_size_m),
            target_size_m=float(target_size_m),
            delta_size_m=delta_size_m,
            delta_h_m=delta_h_m,
            size_ratio=float(target_size_m / reference_size_m) if abs(reference_size_m) > 1e-9 else None,
        )

    def estimate_size_m(self, observation: GeometryObservation) -> float | None:
        if observation.depth_m is None or observation.depth_m <= 0.0:
            return None
        axes = self._physical_axes_m(observation)
        if not axes:
            return None

        size_metric = self.config.size_metric.strip().lower()
        if size_metric == "minor_axis":
            return float(min(axes))
        if size_metric == "major_axis":
            return float(max(axes))
        return float(sum(axes) / len(axes))

    def _physical_axes_m(self, observation: GeometryObservation) -> list[float]:
        axes_px = [axis for axis in (observation.major_axis_px, observation.minor_axis_px) if axis is not None and axis > 0.0]
        if not axes_px:
            return []
        focal_mean = (abs(float(self.intrinsics.fx)) + abs(float(self.intrinsics.fy))) / 2.0
        if focal_mean <= 1e-9:
            return []
        return [float(axis_px * float(observation.depth_m) / focal_mean) for axis_px in axes_px]
