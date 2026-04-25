from __future__ import annotations

from typing import TYPE_CHECKING

from .anchor_estimator import AnchorEstimate, AnchorEstimator, AnchorEstimatorConfig, CameraIntrinsics
from .mask_tracker import MaskTracker, MaskTrackerConfig, TrackedMaskFrame
from .orientation_moments import OrientationEstimate, OrientationMomentsEstimator

if TYPE_CHECKING:
    from .vlm_segmentor import SegmentationFrameResult, VLMObjectSegmentor, VLMObjectSegmentorConfig

__all__ = [
    "AnchorEstimate",
    "AnchorEstimator",
    "AnchorEstimatorConfig",
    "CameraIntrinsics",
    "MaskTracker",
    "MaskTrackerConfig",
    "OrientationEstimate",
    "OrientationMomentsEstimator",
    "SegmentationFrameResult",
    "TrackedMaskFrame",
    "VLMObjectSegmentor",
    "VLMObjectSegmentorConfig",
]


def __getattr__(name: str):
    if name in {"SegmentationFrameResult", "VLMObjectSegmentor", "VLMObjectSegmentorConfig"}:
        from .vlm_segmentor import SegmentationFrameResult, VLMObjectSegmentor, VLMObjectSegmentorConfig

        globals().update(
            {
                "SegmentationFrameResult": SegmentationFrameResult,
                "VLMObjectSegmentor": VLMObjectSegmentor,
                "VLMObjectSegmentorConfig": VLMObjectSegmentorConfig,
            }
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
