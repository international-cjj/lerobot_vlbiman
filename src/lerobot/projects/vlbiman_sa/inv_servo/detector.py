from __future__ import annotations

from dataclasses import dataclass, field
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np

from .target_state import DetectionState, InvServoResult


for _offline_env_name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_offline_env_name, "1")


@dataclass(slots=True)
class GroundingDINOConfig:
    repo_path: Path = Path("/home/cjj/ViT-VS/GD_DINOv2_Sim/GroundingDINO")
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    box_threshold: float = 0.30
    text_threshold: float = 0.25
    device: str = "cuda"
    validation_phrase: str = "yellow ball"


@dataclass(slots=True)
class DetectorInitState:
    repo_path: Path
    package_importable: bool
    config_path: Path | None = None
    checkpoint_path: Path | None = None
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "repo_path": str(self.repo_path),
            "package_importable": bool(self.package_importable),
            "config_path": None if self.config_path is None else str(self.config_path),
            "checkpoint_path": None if self.checkpoint_path is None else str(self.checkpoint_path),
            "debug": dict(self.debug),
        }


class GroundingDINODetector:
    """GroundingDINO adapter with lazy model loading."""

    def __init__(self, config: GroundingDINOConfig | None = None):
        self.config = config or GroundingDINOConfig()
        self._model: Any | None = None

    def check_environment(self) -> InvServoResult:
        repo_path = Path(self.config.repo_path)
        if not repo_path.exists():
            return InvServoResult.failure(
                "groundingdino_repo_missing",
                {"repo_path": str(repo_path)},
            )

        config_path = self._resolve_config_path()
        checkpoint_path = self._resolve_checkpoint_path()
        package_importable = self._package_importable(repo_path)
        device_result = self._resolve_device_result()
        state = DetectorInitState(
            repo_path=repo_path,
            package_importable=package_importable,
            config_path=config_path,
            checkpoint_path=checkpoint_path,
            debug={
                "config_exists": config_path.exists(),
                "checkpoint_exists": checkpoint_path.exists(),
                "requested_device": self.config.device,
                "device": None if not device_result.ok else device_result.state["device"],
            },
        ).to_dict()
        if not package_importable:
            return InvServoResult.failure("groundingdino_package_not_importable", state)
        if not config_path.exists():
            return InvServoResult.failure("groundingdino_config_missing", state)
        if not checkpoint_path.exists():
            return InvServoResult.failure("groundingdino_checkpoint_missing", state)
        if not device_result.ok:
            state["debug"]["device_failure"] = device_result.to_dict()
            return InvServoResult.failure("groundingdino_device_unavailable", state)
        return InvServoResult.success(state)

    def detect(self, image: Any, *, phrase: str | None = None, frame_index: int | None = None) -> InvServoResult:
        target_phrase = phrase or self.config.validation_phrase
        if not target_phrase:
            return InvServoResult.failure("missing_target_phrase")
        image_rgb = self._coerce_rgb_image(image)
        if image_rgb is None:
            return InvServoResult.failure("invalid_rgb_frame")

        environment = self.check_environment()
        if not environment.ok:
            return environment

        model_result = self._ensure_model()
        if not model_result.ok:
            return model_result

        try:
            boxes, logits, phrases = self._predict(image_rgb, target_phrase)
        except Exception as exc:  # pragma: no cover - exercised by integration script.
            return InvServoResult.failure(
                "detector_model_error",
                {"error": f"{type(exc).__name__}: {exc}", "phrase": target_phrase, "frame_index": frame_index},
            )

        if len(boxes) == 0:
            return InvServoResult.failure(
                "no_detection",
                {"phrase": target_phrase, "frame_index": frame_index, "box_threshold": self.config.box_threshold},
            )

        best_index = int(np.argmax(np.asarray(logits, dtype=np.float64)))
        best_box = [float(value) for value in boxes[best_index]]
        best_score = float(logits[best_index])
        best_phrase = str(phrases[best_index]) if best_index < len(phrases) else target_phrase

        if best_score < float(self.config.box_threshold):
            return InvServoResult.failure(
                "low_score",
                {"score": best_score, "box_threshold": self.config.box_threshold, "phrase": best_phrase},
            )
        if not self._valid_bbox(best_box, image_rgb.shape):
            return InvServoResult.failure(
                "invalid_bbox",
                {"bbox_xyxy": best_box, "image_shape": list(image_rgb.shape), "phrase": best_phrase},
            )

        detection = DetectionState(
            phrase=target_phrase,
            score=best_score,
            bbox_xyxy=tuple(best_box),
            frame_index=frame_index,
            label=best_phrase,
            debug={
                "all_detections": [
                    {
                        "bbox_xyxy": [float(value) for value in box],
                        "score": float(score),
                        "phrase": str(det_phrase),
                    }
                    for box, score, det_phrase in zip(boxes, logits, phrases)
                ],
                "device": self.config.device,
                "resolved_device": self._resolve_device(),
                "box_threshold": self.config.box_threshold,
                "text_threshold": self.config.text_threshold,
            },
        )
        return InvServoResult.success(
            {
                "bbox_xyxy": list(detection.bbox_xyxy),
                "score": detection.score,
                "phrase": detection.phrase,
                "label": detection.label,
                "detection": detection.to_dict(),
            }
        )

    def _ensure_model(self) -> InvServoResult:
        if self._model is None:
            try:
                self._add_repo_to_path(Path(self.config.repo_path))
                from groundingdino.util.inference import load_model

                device = self._resolve_device()
                self._model = load_model(
                    str(self._resolve_config_path()),
                    str(self._resolve_checkpoint_path()),
                    device=device,
                )
            except Exception as exc:  # pragma: no cover - exercised by integration script.
                return InvServoResult.failure(
                    "detector_model_error",
                    {
                        "error": f"{type(exc).__name__}: {exc}",
                        "config_path": str(self._resolve_config_path()),
                        "checkpoint_path": str(self._resolve_checkpoint_path()),
                        "device": self.config.device,
                    },
                )
        return InvServoResult.success({"model_loaded": True, "device": self._resolve_device()})

    def _predict(self, image_rgb: np.ndarray, phrase: str) -> tuple[list[list[float]], list[float], list[str]]:
        self._add_repo_to_path(Path(self.config.repo_path))
        import torch
        from PIL import Image
        from torchvision.ops import box_convert

        import groundingdino.datasets.transforms as gd_transforms
        from groundingdino.util.inference import predict

        image_pil = Image.fromarray(image_rgb)
        transform = gd_transforms.Compose(
            [
                gd_transforms.RandomResize([800], max_size=1333),
                gd_transforms.ToTensor(),
                gd_transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image_tensor, _ = transform(image_pil, None)
        device = self._resolve_device()
        boxes, logits, phrases = predict(
            self._model,
            image_tensor,
            phrase,
            box_threshold=float(self.config.box_threshold),
            text_threshold=float(self.config.text_threshold),
            device=device,
        )
        height, width = image_rgb.shape[:2]
        if len(boxes) == 0:
            return [], [], []
        scale = torch.tensor([width, height, width, height], dtype=boxes.dtype)
        boxes_xyxy = box_convert(boxes=boxes * scale, in_fmt="cxcywh", out_fmt="xyxy")
        return boxes_xyxy.cpu().numpy().tolist(), logits.cpu().numpy().tolist(), list(phrases)

    @staticmethod
    def result_from_detection(detection: DetectionState) -> InvServoResult:
        return InvServoResult.success({"detection": detection.to_dict()})

    @staticmethod
    def _package_importable(repo_path: Path) -> bool:
        try:
            import groundingdino  # noqa: F401

            return True
        except ModuleNotFoundError:
            package_dir = repo_path / "groundingdino"
            return package_dir.exists() and (package_dir / "__init__.py").exists()

    def _resolve_config_path(self) -> Path:
        return Path(self.config.config_path or Path(self.config.repo_path) / "groundingdino/config/GroundingDINO_SwinT_OGC.py")

    def _resolve_checkpoint_path(self) -> Path:
        return Path(self.config.checkpoint_path or Path(self.config.repo_path).parent / "weights/groundingdino_swint_ogc.pth")

    def _resolve_device_result(self) -> InvServoResult:
        try:
            return InvServoResult.success({"device": self._resolve_device()})
        except RuntimeError as exc:
            return InvServoResult.failure("groundingdino_device_unavailable", {"error": str(exc)})

    def _resolve_device(self) -> str:
        requested = str(self.config.device or "auto").lower()
        if requested == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        if requested.startswith("cuda"):
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError(
                    f"GroundingDINO requested {self.config.device!r}, but torch.cuda.is_available() is false."
                )
        return str(self.config.device)

    @staticmethod
    def _add_repo_to_path(repo_path: Path) -> None:
        repo_str = str(repo_path)
        if repo_path.exists() and repo_str not in sys.path:
            sys.path.insert(0, repo_str)

    @staticmethod
    def _coerce_rgb_image(image: Any) -> np.ndarray | None:
        if image is None:
            return None
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            return None
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _valid_bbox(bbox_xyxy: list[float], image_shape: tuple[int, ...]) -> bool:
        if len(bbox_xyxy) != 4 or not all(np.isfinite(value) for value in bbox_xyxy):
            return False
        height, width = image_shape[:2]
        x0, y0, x1, y1 = bbox_xyxy
        return 0.0 <= x0 < x1 <= float(width) and 0.0 <= y0 < y1 <= float(height)
