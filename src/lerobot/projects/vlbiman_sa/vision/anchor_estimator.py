from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int | None = None
    height: int | None = None

    @classmethod
    def from_json(cls, path: Path) -> "CameraIntrinsics":
        payload = json.loads(path.read_text(encoding="utf-8"))
        meta = payload.get("metadata", {})
        camera_matrix = payload.get("camera_matrix", [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        fx = meta.get("fx", camera_matrix[0][0])
        fy = meta.get("fy", camera_matrix[1][1])
        cx = meta.get("cx", camera_matrix[0][2])
        cy = meta.get("cy", camera_matrix[1][2])
        return cls(
            fx=float(fx),
            fy=float(fy),
            cx=float(cx),
            cy=float(cy),
            width=int(meta["width"]) if meta.get("width") is not None else None,
            height=int(meta["height"]) if meta.get("height") is not None else None,
        )


@dataclass(slots=True)
class AnchorEstimatorConfig:
    depth_window: int = 7
    contact_strategy: str = "nearest_depth"


@dataclass(slots=True)
class AnchorEstimate:
    frame_index: int
    mask_area_px: int
    bbox_xyxy: list[int]
    centroid_px: list[float] | None
    contact_px: list[float] | None
    depth_m: float | None
    camera_xyz_m: list[float] | None
    orientation_deg: float | None
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class AnchorEstimator:
    def __init__(self, intrinsics: CameraIntrinsics, config: AnchorEstimatorConfig | None = None):
        self.intrinsics = intrinsics
        self.config = config or AnchorEstimatorConfig()

    def estimate(
        self,
        frame_index: int,
        mask: np.ndarray,
        depth_map: np.ndarray,
        orientation_deg: float | None,
        score: float,
    ) -> AnchorEstimate:
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return AnchorEstimate(
                frame_index=frame_index,
                mask_area_px=0,
                bbox_xyxy=[0, 0, 0, 0],
                centroid_px=None,
                contact_px=None,
                depth_m=None,
                camera_xyz_m=None,
                orientation_deg=orientation_deg,
                score=float(score),
            )

        x_min, x_max = int(xs.min()), int(xs.max())
        y_min, y_max = int(ys.min()), int(ys.max())
        centroid_x = float(xs.mean())
        centroid_y = float(ys.mean())
        contact_px = self._select_contact_point(xs, ys, depth_map)
        depth_m, camera_xyz = self._project_to_camera(contact_px, depth_map)
        return AnchorEstimate(
            frame_index=frame_index,
            mask_area_px=int(xs.size),
            bbox_xyxy=[x_min, y_min, x_max, y_max],
            centroid_px=[centroid_x, centroid_y],
            contact_px=list(contact_px) if contact_px is not None else None,
            depth_m=depth_m,
            camera_xyz_m=camera_xyz,
            orientation_deg=orientation_deg,
            score=float(score),
        )

    def _select_contact_point(
        self,
        xs: np.ndarray,
        ys: np.ndarray,
        depth_map: np.ndarray,
    ) -> tuple[float, float] | None:
        if self.config.contact_strategy == "centroid":
            return float(xs.mean()), float(ys.mean())

        depths = depth_map[ys, xs].astype(np.float32)
        valid = depths > 0
        if not np.any(valid):
            return float(xs.mean()), float(ys.mean())
        valid_xs = xs[valid]
        valid_ys = ys[valid]
        valid_depths = depths[valid]
        nearest_index = int(np.argmin(valid_depths))
        return float(valid_xs[nearest_index]), float(valid_ys[nearest_index])

    def _project_to_camera(
        self,
        contact_px: tuple[float, float] | None,
        depth_map: np.ndarray,
    ) -> tuple[float | None, list[float] | None]:
        if contact_px is None:
            return None, None

        x = int(round(contact_px[0]))
        y = int(round(contact_px[1]))
        window = max(1, int(self.config.depth_window // 2))
        y0 = max(0, y - window)
        y1 = min(depth_map.shape[0], y + window + 1)
        x0 = max(0, x - window)
        x1 = min(depth_map.shape[1], x + window + 1)
        local = depth_map[y0:y1, x0:x1].astype(np.float32)
        valid = local > 0
        if not np.any(valid):
            return None, None
        depth_m = float(np.median(local[valid]))
        if depth_m > 10.0:
            depth_m /= 1000.0
        scale_x = depth_map.shape[1] / self.intrinsics.width if self.intrinsics.width else 1.0
        scale_y = depth_map.shape[0] / self.intrinsics.height if self.intrinsics.height else 1.0
        fx = self.intrinsics.fx * scale_x
        fy = self.intrinsics.fy * scale_y
        cx = self.intrinsics.cx * scale_x
        cy = self.intrinsics.cy * scale_y
        camera_x = (contact_px[0] - cx) * depth_m / fx
        camera_y = (contact_px[1] - cy) * depth_m / fy
        return depth_m, [float(camera_x), float(camera_y), depth_m]
