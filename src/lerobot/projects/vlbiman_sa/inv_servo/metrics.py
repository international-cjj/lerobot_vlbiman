from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .target_state import MaskState


def binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {array.shape}.")
    return array > 0


def mask_area_px(mask: np.ndarray) -> int:
    return int(np.count_nonzero(binary_mask(mask)))


def mask_centroid_uv(mask: np.ndarray) -> tuple[float, float] | None:
    ys, xs = np.nonzero(binary_mask(mask))
    if xs.size == 0:
        return None
    return (float(xs.mean()), float(ys.mean()))


def bbox_xyxy_from_mask(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(binary_mask(mask))
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def mask_iou(mask_a: np.ndarray | None, mask_b: np.ndarray | None) -> float | None:
    if mask_a is None or mask_b is None:
        return None
    a = binary_mask(mask_a)
    b = binary_mask(mask_b)
    if a.shape != b.shape:
        raise ValueError(f"mask shapes must match, got {a.shape} and {b.shape}.")
    union = int(np.count_nonzero(a | b))
    if union == 0:
        return 0.0
    inter = int(np.count_nonzero(a & b))
    return float(inter / union)


def mask_state_from_mask(
    mask: np.ndarray,
    *,
    frame_index: int,
    source: str,
    mask_path: Path | None = None,
    debug: dict[str, Any] | None = None,
) -> MaskState:
    array = binary_mask(mask)
    return MaskState(
        frame_index=frame_index,
        image_size_hw=(int(array.shape[0]), int(array.shape[1])),
        mask_area_px=int(np.count_nonzero(array)),
        centroid_uv=mask_centroid_uv(array),
        bbox_xyxy=bbox_xyxy_from_mask(array),
        source=source,
        mask_path=mask_path,
        debug=debug or {},
    )
