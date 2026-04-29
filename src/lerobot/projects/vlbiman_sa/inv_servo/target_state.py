from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any
import math

import cv2
import numpy as np


def _to_plain(value: Any) -> Any:
    if is_dataclass(value):
        return {key: _to_plain(item) for key, item in asdict(value).items()}
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _to_plain(item) for key, item in value.items()}
    return value


def _finite_float(name: str, value: Any) -> float:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return float(value)


def _finite_tuple(name: str, values: tuple[Any, ...], size: int) -> tuple[float, ...]:
    if len(values) != size:
        raise ValueError(f"{name} must have exactly {size} values.")
    return tuple(_finite_float(f"{name}[{index}]", value) for index, value in enumerate(values))


def _optional_finite_tuple(name: str, values: tuple[Any, ...] | None, size: int) -> tuple[float, ...] | None:
    if values is None:
        return None
    return _finite_tuple(name, tuple(values), size)


def _binary_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim == 3:
        array = array[:, :, 0]
    if array.ndim != 2:
        raise ValueError(f"target mask must be 2D, got shape {array.shape}.")
    return array > 0


def _bbox_from_binary(mask: np.ndarray) -> tuple[float, float, float, float] | None:
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return (float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1))


def _center_from_bbox(bbox_xyxy: tuple[float, float, float, float]) -> tuple[float, float]:
    x0, y0, x1, y1 = bbox_xyxy
    return ((x0 + x1) * 0.5, (y0 + y1) * 0.5)


def _area_ratio(mask_area_px: int, image_size_hw: tuple[int, int]) -> float:
    height, width = image_size_hw
    return float(mask_area_px) / max(float(height * width), 1.0)


def _rgb_histogram_feature(rgb_frame: np.ndarray, mask: np.ndarray | None = None, *, bins: int = 8) -> list[float]:
    frame = np.asarray(rgb_frame)
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError(f"rgb_frame must have shape HxWx3, got {frame.shape}.")
    if bins <= 0:
        raise ValueError("bins must be positive.")

    if mask is not None:
        mask_binary = _binary_mask(mask)
        if mask_binary.shape != frame.shape[:2]:
            raise ValueError(f"mask shape {mask_binary.shape} does not match frame shape {frame.shape[:2]}.")
        pixels = frame[mask_binary]
    else:
        pixels = frame.reshape(-1, 3)

    if pixels.size == 0:
        return [0.0] * (bins * 3)

    features: list[float] = []
    for channel in range(3):
        hist, _ = np.histogram(pixels[:, channel], bins=bins, range=(0, 256))
        hist = hist.astype(np.float64)
        denom = float(hist.sum())
        if denom > 0.0:
            hist /= denom
        features.extend(float(value) for value in hist)
    return features


@dataclass(slots=True)
class InvServoResult:
    ok: bool
    state: dict[str, Any] | None = None
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if self.ok and self.failure_reason is not None:
            raise ValueError("successful InvServoResult must not carry failure_reason.")
        if not self.ok and not self.failure_reason:
            raise ValueError("failed InvServoResult must carry failure_reason.")

    @classmethod
    def success(cls, state: dict[str, Any] | None = None) -> "InvServoResult":
        return cls(ok=True, state=state or {}, failure_reason=None)

    @classmethod
    def failure(cls, reason: str, state: dict[str, Any] | None = None) -> "InvServoResult":
        return cls(ok=False, state=state, failure_reason=reason)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": bool(self.ok),
            "state": _to_plain(self.state),
            "failure_reason": self.failure_reason,
        }


@dataclass(slots=True)
class DetectionState:
    phrase: str
    score: float
    bbox_xyxy: tuple[float, float, float, float]
    frame_index: int | None = None
    label: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phrase:
            raise ValueError("phrase must be non-empty.")
        self.score = _finite_float("score", self.score)
        self.bbox_xyxy = _finite_tuple("bbox_xyxy", tuple(self.bbox_xyxy), 4)  # type: ignore[assignment]
        if self.frame_index is not None:
            self.frame_index = int(self.frame_index)

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(slots=True)
class MaskState:
    frame_index: int
    image_size_hw: tuple[int, int]
    mask_area_px: int
    centroid_uv: tuple[float, float] | None = None
    bbox_xyxy: tuple[float, float, float, float] | None = None
    source: str = "unknown"
    mask_path: Path | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.frame_index = int(self.frame_index)
        if len(self.image_size_hw) != 2:
            raise ValueError("image_size_hw must be (height, width).")
        height, width = int(self.image_size_hw[0]), int(self.image_size_hw[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size_hw values must be positive.")
        self.image_size_hw = (height, width)
        self.mask_area_px = int(self.mask_area_px)
        if self.mask_area_px < 0:
            raise ValueError("mask_area_px must be non-negative.")
        self.centroid_uv = _optional_finite_tuple("centroid_uv", self.centroid_uv, 2)  # type: ignore[assignment]
        self.bbox_xyxy = _optional_finite_tuple("bbox_xyxy", self.bbox_xyxy, 4)  # type: ignore[assignment]

    @property
    def visible(self) -> bool:
        return self.mask_area_px > 0 and self.centroid_uv is not None

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(slots=True)
class TargetState:
    target_center_uv: tuple[float, float]
    target_area_ratio: float
    target_bbox_xyxy: tuple[float, float, float, float]
    image_size_hw: tuple[int, int]
    frame_index: int | None = None
    target_mask: np.ndarray | None = None
    target_feature: list[float] | None = None
    source: str = "unknown"
    mask_path: Path | None = None
    frame_path: Path | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.target_center_uv = _finite_tuple("target_center_uv", tuple(self.target_center_uv), 2)  # type: ignore[assignment]
        self.target_area_ratio = _finite_float("target_area_ratio", self.target_area_ratio)
        if self.target_area_ratio < 0.0:
            raise ValueError("target_area_ratio must be non-negative.")
        self.target_bbox_xyxy = _finite_tuple("target_bbox_xyxy", tuple(self.target_bbox_xyxy), 4)  # type: ignore[assignment]
        if len(self.image_size_hw) != 2:
            raise ValueError("image_size_hw must be (height, width).")
        height, width = int(self.image_size_hw[0]), int(self.image_size_hw[1])
        if height <= 0 or width <= 0:
            raise ValueError("image_size_hw values must be positive.")
        self.image_size_hw = (height, width)
        if self.frame_index is not None:
            self.frame_index = int(self.frame_index)
        if self.target_mask is not None:
            mask = _binary_mask(self.target_mask).astype(np.uint8) * 255
            if mask.shape != self.image_size_hw:
                raise ValueError(f"target_mask shape {mask.shape} does not match image_size_hw {self.image_size_hw}.")
            self.target_mask = mask
        if self.target_feature is not None:
            self.target_feature = [_finite_float("target_feature[]", value) for value in self.target_feature]

    def to_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        payload = {
            "target_center_uv": list(self.target_center_uv),
            "target_area_ratio": float(self.target_area_ratio),
            "target_bbox_xyxy": list(self.target_bbox_xyxy),
            "target_mask": self.target_mask if include_mask else None,
            "target_mask_shape": None if self.target_mask is None else list(self.target_mask.shape),
            "target_feature": None if self.target_feature is None else list(self.target_feature),
            "image_size_hw": list(self.image_size_hw),
            "frame_index": self.frame_index,
            "source": self.source,
            "mask_path": None if self.mask_path is None else str(self.mask_path),
            "frame_path": None if self.frame_path is None else str(self.frame_path),
            "debug": dict(self.debug),
        }
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "TargetState":
        if not isinstance(payload, dict):
            raise ValueError("target state payload must be a mapping.")
        return cls(
            target_center_uv=tuple(payload["target_center_uv"]),
            target_area_ratio=payload["target_area_ratio"],
            target_bbox_xyxy=tuple(payload["target_bbox_xyxy"]),
            image_size_hw=tuple(payload["image_size_hw"]),
            frame_index=payload.get("frame_index"),
            target_mask=None,
            target_feature=payload.get("target_feature"),
            source=str(payload.get("source", "external_state")),
            mask_path=Path(payload["mask_path"]) if payload.get("mask_path") else None,
            frame_path=Path(payload["frame_path"]) if payload.get("frame_path") else None,
            debug=dict(payload.get("debug", {})),
        )


def target_state_from_mask(
    mask: np.ndarray,
    *,
    frame_index: int | None = None,
    rgb_frame: np.ndarray | None = None,
    source: str = "mask",
    mask_path: Path | None = None,
    frame_path: Path | None = None,
    feature_bins: int = 8,
) -> TargetState:
    binary = _binary_mask(mask)
    mask_area_px = int(np.count_nonzero(binary))
    if mask_area_px <= 0:
        raise ValueError("invalid_target_mask")

    bbox = _bbox_from_binary(binary)
    if bbox is None:
        raise ValueError("invalid_target_mask")
    ys, xs = np.nonzero(binary)
    image_size_hw = (int(binary.shape[0]), int(binary.shape[1]))
    feature = None
    if rgb_frame is not None:
        feature = _rgb_histogram_feature(rgb_frame, binary, bins=feature_bins)
    return TargetState(
        target_center_uv=(float(xs.mean()), float(ys.mean())),
        target_area_ratio=_area_ratio(mask_area_px, image_size_hw),
        target_bbox_xyxy=bbox,
        image_size_hw=image_size_hw,
        frame_index=frame_index,
        target_mask=binary.astype(np.uint8) * 255,
        target_feature=feature,
        source=source,
        mask_path=mask_path,
        frame_path=frame_path,
        debug={"mask_area_px": mask_area_px, "feature_bins": feature_bins if feature is not None else None},
    )


def target_state_from_bbox(
    bbox_xyxy: tuple[float, float, float, float] | list[float],
    *,
    image_size_hw: tuple[int, int] | list[int],
    frame_index: int | None = None,
    rgb_frame: np.ndarray | None = None,
    source: str = "bbox",
    frame_path: Path | None = None,
    feature_bins: int = 8,
) -> TargetState:
    bbox = _finite_tuple("bbox_xyxy", tuple(bbox_xyxy), 4)
    x0, y0, x1, y1 = bbox
    if x1 <= x0 or y1 <= y0:
        raise ValueError("invalid_target_state")
    height, width = int(image_size_hw[0]), int(image_size_hw[1])
    if height <= 0 or width <= 0:
        raise ValueError("invalid_target_state")
    area_ratio = float((x1 - x0) * (y1 - y0)) / max(float(height * width), 1.0)
    feature = _rgb_histogram_feature(rgb_frame, bins=feature_bins) if rgb_frame is not None else None
    return TargetState(
        target_center_uv=_center_from_bbox(bbox),  # type: ignore[arg-type]
        target_area_ratio=area_ratio,
        target_bbox_xyxy=bbox,  # type: ignore[arg-type]
        image_size_hw=(height, width),
        frame_index=frame_index,
        target_mask=None,
        target_feature=feature,
        source=source,
        frame_path=frame_path,
        debug={"feature_bins": feature_bins if feature is not None else None},
    )


@dataclass(slots=True)
class ServoTarget:
    phrase: str
    mask: MaskState
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.phrase:
            raise ValueError("phrase must be non-empty.")

    def to_dict(self) -> dict[str, Any]:
        return _to_plain(self)


@dataclass(slots=True)
class ServoError:
    du_norm: float
    dv_norm: float
    area_ratio_error: float
    mask_iou: float | None = None
    converged: bool = False
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.du_norm = _finite_float("du_norm", self.du_norm)
        self.dv_norm = _finite_float("dv_norm", self.dv_norm)
        self.area_ratio_error = _finite_float("area_ratio_error", self.area_ratio_error)
        if self.mask_iou is not None:
            self.mask_iou = _finite_float("mask_iou", self.mask_iou)

    def to_dict(self) -> dict[str, Any]:
        payload = _to_plain(self)
        e_u = -float(self.du_norm)
        e_v = -float(self.dv_norm)
        e_a = float(self.area_ratio_error)
        payload.update(
            {
                "e_u": e_u,
                "e_v": e_v,
                "e_a": e_a,
                "error_norm": float(math.sqrt(e_u * e_u + e_v * e_v + e_a * e_a)),
                "mask_iou_error": None if self.mask_iou is None else float(1.0 - self.mask_iou),
            }
        )
        return payload


@dataclass(slots=True)
class ServoCommand:
    delta_xyz_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    delta_rpy_rad: tuple[float, float, float] = (0.0, 0.0, 0.0)
    gripper_position: float | None = None
    stop: bool = False
    reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.delta_xyz_m = _finite_tuple("delta_xyz_m", tuple(self.delta_xyz_m), 3)  # type: ignore[assignment]
        self.delta_rpy_rad = _finite_tuple("delta_rpy_rad", tuple(self.delta_rpy_rad), 3)  # type: ignore[assignment]
        if self.gripper_position is not None:
            self.gripper_position = _finite_float("gripper_position", self.gripper_position)

    @classmethod
    def zero(cls, *, reason: str | None = None) -> "ServoCommand":
        return cls(stop=reason is not None, reason=reason)

    def to_dict(self) -> dict[str, Any]:
        payload = _to_plain(self)
        payload["delta_cam"] = list(self.delta_xyz_m)
        payload["delta_pose"] = {
            "translation": list(self.delta_xyz_m),
            "rotation_rpy": list(self.delta_rpy_rad),
        }
        return payload


def _build_self_test_mask() -> tuple[np.ndarray, np.ndarray]:
    rgb = np.zeros((64, 96, 3), dtype=np.uint8)
    rgb[:, :, 0] = 32
    rgb[:, :, 1] = 64
    rgb[:, :, 2] = 128
    mask = np.zeros((64, 96), dtype=np.uint8)
    cv2.circle(mask, (48, 32), 12, 255, -1)
    rgb[mask > 0] = (240, 220, 32)
    return rgb, mask


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Target state utilities for inv RGB visual servo.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def _run_self_test(config_path: Path | None = None) -> None:
    if config_path is not None:
        from lerobot.projects.vlbiman_sa.inv_servo.config import load_inv_rgb_servo_config

        cfg = load_inv_rgb_servo_config(config_path)
        if cfg.target.phrase != "yellow ball":
            raise RuntimeError(f"Unexpected target phrase: {cfg.target.phrase!r}")

    rgb, mask = _build_self_test_mask()
    state = target_state_from_mask(mask, frame_index=100, rgb_frame=rgb, source="self_test")
    bbox_state = target_state_from_bbox(
        state.target_bbox_xyxy,
        image_size_hw=state.image_size_hw,
        frame_index=100,
        rgb_frame=rgb,
        source="self_test_bbox",
    )
    if state.target_center_uv != (48.0, 32.0):
        raise RuntimeError(f"Unexpected target center: {state.target_center_uv}")
    if state.target_area_ratio <= 0.0:
        raise RuntimeError("Target area ratio must be positive.")
    if not state.target_feature or len(state.target_feature) != 24:
        raise RuntimeError("RGB histogram feature was not generated.")
    if bbox_state.target_area_ratio <= 0.0:
        raise RuntimeError("BBox target area ratio must be positive.")
    print("TARGET_STATE_INTERFACE_OK")
    print(json.dumps(state.to_dict(), ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.self_test:
        print("TARGET_STATE_NOOP")
        return
    _run_self_test(args.config)


if __name__ == "__main__":
    main(sys.argv[1:])
