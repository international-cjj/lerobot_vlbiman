from __future__ import annotations

import argparse
from contextlib import nullcontext
from dataclasses import dataclass
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from typing import Any, Iterable

import cv2
import numpy as np

from .target_state import DetectionState, InvServoResult, MaskState


DEFAULT_SAM2_REPO = Path("/home/cjj/lerobot_2026_1/third_party/sam2_official_20260324")


for _offline_env_name in ("HF_HUB_OFFLINE", "TRANSFORMERS_OFFLINE", "HF_DATASETS_OFFLINE"):
    os.environ.setdefault(_offline_env_name, "1")


@dataclass(slots=True)
class SAM2LiveTrackerConfig:
    model_id: str = "facebook/sam2-hiera-small"
    model_size: str = "small"
    config_file: str | None = None
    checkpoint_path: Path | None = None
    device: str = "auto"
    use_fp16: bool = True
    input_resize_width: int = 640
    max_update_ms: float = 100.0
    repo_path: Path | None = DEFAULT_SAM2_REPO
    independent_live_tracker: bool = True
    local_files_only: bool = True
    offload_video_to_cpu: bool = False
    offload_state_to_cpu: bool = False
    async_loading_frames: bool = False
    mask_threshold: float = 0.0
    jpeg_quality: int = 95
    incremental_live_tracker: bool = True
    live_seed_with_previous_mask: bool = True


@dataclass(slots=True)
class SAM2TrackedFrame:
    frame_index: int
    local_frame_index: int
    mask: np.ndarray
    mask_state: MaskState
    update_ms: float
    fps: float
    obj_ids: list[int]

    def to_dict(self, *, include_mask: bool = False) -> dict[str, Any]:
        return {
            "ok": self.mask_state.visible,
            "frame_index": self.frame_index,
            "local_frame_index": self.local_frame_index,
            "bbox_xyxy": None if self.mask_state.bbox_xyxy is None else list(self.mask_state.bbox_xyxy),
            "center_uv": None if self.mask_state.centroid_uv is None else list(self.mask_state.centroid_uv),
            "area_ratio": self._area_ratio(),
            "mask_area_px": self.mask_state.mask_area_px,
            "image_size_hw": list(self.mask_state.image_size_hw),
            "update_ms": float(self.update_ms),
            "fps": float(self.fps),
            "obj_ids": list(self.obj_ids),
            "failure_reason": None if self.mask_state.visible else "empty_sam2_mask",
            "mask": self.mask if include_mask else None,
        }

    def _area_ratio(self) -> float:
        height, width = self.mask_state.image_size_hw
        return float(self.mask_state.mask_area_px) / max(float(height * width), 1.0)


class SAM2LiveTracker:
    """Independent SAM2 video predictor adapter for RGB mask tracking."""

    def __init__(self, config: SAM2LiveTrackerConfig | None = None):
        self.config = config or SAM2LiveTrackerConfig()
        self._predictor: Any | None = None
        self._predictor_info: dict[str, Any] = {}
        self._initialized = False
        self._seed_bbox_xyxy: tuple[float, float, float, float] | None = None
        self._live_frames: list[np.ndarray] = []
        self._live_frame_indices: list[int] = []
        self._live_tempdir: tempfile.TemporaryDirectory[str] | None = None
        self._last_live_frame: np.ndarray | None = None
        self._last_live_frame_index: int | None = None
        self._last_live_mask: np.ndarray | None = None
        self._last_live_mask_state: MaskState | None = None
        self._last_live_bbox_xyxy: tuple[float, float, float, float] | None = None

    def check_environment(self) -> InvServoResult:
        try:
            self._ensure_sam2_on_path()
            from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES  # noqa: F401
            from sam2.sam2_video_predictor import SAM2VideoPredictor  # noqa: F401

            state: dict[str, Any] = {
                "sam2_package_importable": True,
                "repo_path": None if self.config.repo_path is None else str(self.config.repo_path),
                "repo_path_exists": None if self.config.repo_path is None else self.config.repo_path.exists(),
            }
            try:
                _, ckpt_path = self._resolve_model_files(load_hub=False)
                state["checkpoint_path"] = str(ckpt_path)
                state["checkpoint_exists"] = Path(ckpt_path).exists()
            except Exception as exc:  # keep import diagnostics useful even before weights are available
                state["checkpoint_error"] = str(exc)
            return InvServoResult.success(state)
        except ModuleNotFoundError as exc:
            state = {
                "sam2_package_importable": False,
                "error": str(exc),
                "repo_path": None if self.config.repo_path is None else str(self.config.repo_path),
                "repo_path_exists": None if self.config.repo_path is None else self.config.repo_path.exists(),
            }
            return InvServoResult.failure("sam2_package_not_importable", state)
        except Exception as exc:
            return InvServoResult.failure(
                "sam2_environment_check_failed",
                {
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                    "repo_path": None if self.config.repo_path is None else str(self.config.repo_path),
                },
            )

    def initialize(self, rgb_frame: Any, bbox_xyxy: DetectionState | Iterable[float]) -> dict[str, Any]:
        if rgb_frame is None:
            return self._failure_payload("frame_missing")
        try:
            bbox = self._coerce_bbox(bbox_xyxy)
            frame_index = int(bbox_xyxy.frame_index) if isinstance(bbox_xyxy, DetectionState) and bbox_xyxy.frame_index is not None else 0
            self.reset()
            self._seed_bbox_xyxy = bbox
            self._live_frames = [self._coerce_rgb_frame(rgb_frame)]
            self._live_frame_indices = [frame_index]
            tracked = self.track_sequence(
                self._live_frames,
                frame_indices=self._live_frame_indices,
                init_bbox_xyxy=bbox,
                work_dir=self._live_work_dir(),
            )
            self._store_live_seed(self._live_frames[0], frame_index, tracked[0])
            self._initialized = True
            return self._payload_from_tracked_frame(tracked[0])
        except Exception as exc:
            return self._failure_payload("sam2_init_failed", error=exc)

    def update(self, rgb_frame: Any, *, frame_index: int | None = None) -> dict[str, Any]:
        if rgb_frame is None:
            return self._failure_payload("frame_missing")
        if self._seed_bbox_xyxy is None:
            return self._failure_payload("sam2_tracker_not_initialized")
        try:
            next_index = int(frame_index) if frame_index is not None else self._live_frame_indices[-1] + 1
            frame = self._coerce_rgb_frame(rgb_frame)
            if self.config.incremental_live_tracker and self._last_live_frame is not None:
                tracked_frame = self._track_incremental_live_frame(
                    previous_frame=self._last_live_frame,
                    current_frame=frame,
                    previous_frame_index=int(self._last_live_frame_index or 0),
                    current_frame_index=next_index,
                    previous_mask=self._last_live_mask,
                    previous_bbox_xyxy=self._last_live_bbox_xyxy or self._seed_bbox_xyxy,
                    work_dir=self._live_work_dir(),
                )
                self._store_live_seed(frame, next_index, tracked_frame)
                self._live_frames = [frame]
                self._live_frame_indices = [next_index]
                self._initialized = True
                payload = self._payload_from_tracked_frame(tracked_frame)
                payload["tracking_mode"] = "incremental_pair"
                return payload
            self._live_frames.append(frame)
            self._live_frame_indices.append(next_index)
            tracked = self.track_sequence(
                self._live_frames,
                frame_indices=self._live_frame_indices,
                init_bbox_xyxy=self._seed_bbox_xyxy,
                work_dir=self._live_work_dir(),
            )
            self._store_live_seed(frame, next_index, tracked[-1])
            self._initialized = True
            return self._payload_from_tracked_frame(tracked[-1])
        except Exception as exc:
            return self._failure_payload("sam2_update_failed", error=exc)

    def reset(self) -> None:
        self._initialized = False
        self._seed_bbox_xyxy = None
        self._live_frames = []
        self._live_frame_indices = []
        self._last_live_frame = None
        self._last_live_frame_index = None
        self._last_live_mask = None
        self._last_live_mask_state = None
        self._last_live_bbox_xyxy = None
        if self._live_tempdir is not None:
            self._live_tempdir.cleanup()
            self._live_tempdir = None

    def track(self, frame: Any, *, frame_index: int) -> InvServoResult:
        payload = self.update(frame, frame_index=frame_index)
        if payload["ok"]:
            serializable = dict(payload)
            serializable["mask"] = None
            return InvServoResult.success(serializable)
        return InvServoResult.failure(str(payload["failure_reason"]), {"frame_index": frame_index})

    def track_sequence(
        self,
        frames: Iterable[np.ndarray],
        *,
        frame_indices: Iterable[int] | None = None,
        init_bbox_xyxy: DetectionState | Iterable[float],
        work_dir: Path | None = None,
    ) -> list[SAM2TrackedFrame]:
        rgb_frames = [self._coerce_rgb_frame(frame) for frame in frames]
        if not rgb_frames:
            raise ValueError("frames must not be empty.")
        indices = list(range(len(rgb_frames))) if frame_indices is None else [int(index) for index in frame_indices]
        if len(indices) != len(rgb_frames):
            raise ValueError("frame_indices length must match frames length.")
        self._validate_frame_sizes(rgb_frames)
        bbox = self._coerce_bbox(init_bbox_xyxy)

        predictor = self.load_predictor()
        result_by_local_index: dict[int, SAM2TrackedFrame] = {}
        temp_ctx: tempfile.TemporaryDirectory[str] | nullcontext[str]
        temp_ctx = tempfile.TemporaryDirectory(prefix="sam2_frames_") if work_dir is None else nullcontext(str(work_dir))
        with temp_ctx as frame_dir_str:
            frame_dir = Path(frame_dir_str)
            self._write_jpeg_sequence(rgb_frames, frame_dir)
            import torch

            with torch.inference_mode(), self._autocast_context():
                inference_state = predictor.init_state(
                    video_path=str(frame_dir),
                    offload_video_to_cpu=self.config.offload_video_to_cpu,
                    offload_state_to_cpu=self.config.offload_state_to_cpu,
                    async_loading_frames=self.config.async_loading_frames,
                )

                start = time.perf_counter()
                local_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                    inference_state=inference_state,
                    frame_idx=0,
                    obj_id=1,
                    box=np.asarray(bbox, dtype=np.float32),
                )
                seed_update_ms = (time.perf_counter() - start) * 1000.0
                result_by_local_index[int(local_idx)] = self._tracked_frame_from_logits(
                    mask_logits=mask_logits,
                    frame_index=indices[int(local_idx)],
                    local_frame_index=int(local_idx),
                    update_ms=seed_update_ms,
                    obj_ids=obj_ids,
                )

                iterator = predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(rgb_frames),
                    reverse=False,
                )
                while True:
                    start = time.perf_counter()
                    try:
                        local_idx, obj_ids, mask_logits = next(iterator)
                    except StopIteration:
                        break
                    update_ms = (time.perf_counter() - start) * 1000.0
                    result_by_local_index[int(local_idx)] = self._tracked_frame_from_logits(
                        mask_logits=mask_logits,
                        frame_index=indices[int(local_idx)],
                        local_frame_index=int(local_idx),
                        update_ms=update_ms,
                        obj_ids=obj_ids,
                    )

        missing = [index for index in range(len(rgb_frames)) if index not in result_by_local_index]
        if missing:
            raise RuntimeError(f"SAM2 did not return masks for local frames: {missing}")
        return [result_by_local_index[index] for index in range(len(rgb_frames))]

    def _track_incremental_live_frame(
        self,
        *,
        previous_frame: np.ndarray,
        current_frame: np.ndarray,
        previous_frame_index: int,
        current_frame_index: int,
        previous_mask: np.ndarray | None,
        previous_bbox_xyxy: tuple[float, float, float, float] | None,
        work_dir: Path | None,
    ) -> SAM2TrackedFrame:
        rgb_frames = [self._coerce_rgb_frame(previous_frame), self._coerce_rgb_frame(current_frame)]
        self._validate_frame_sizes(rgb_frames)
        if previous_bbox_xyxy is None and previous_mask is None:
            raise RuntimeError("incremental live tracking requires a previous mask or bbox seed.")

        predictor = self.load_predictor()
        result_by_local_index: dict[int, SAM2TrackedFrame] = {}
        temp_ctx: tempfile.TemporaryDirectory[str] | nullcontext[str]
        temp_ctx = tempfile.TemporaryDirectory(prefix="sam2_frames_") if work_dir is None else nullcontext(str(work_dir))
        with temp_ctx as frame_dir_str:
            frame_dir = Path(frame_dir_str)
            self._write_jpeg_sequence(rgb_frames, frame_dir)
            import torch

            with torch.inference_mode(), self._autocast_context():
                inference_state = predictor.init_state(
                    video_path=str(frame_dir),
                    offload_video_to_cpu=self.config.offload_video_to_cpu,
                    offload_state_to_cpu=self.config.offload_state_to_cpu,
                    async_loading_frames=self.config.async_loading_frames,
                )

                seed_start = time.perf_counter()
                seed_mode = "bbox"
                seed_mask = self._usable_seed_mask(previous_mask)
                if self.config.live_seed_with_previous_mask and seed_mask is not None and hasattr(predictor, "add_new_mask"):
                    local_idx, obj_ids, mask_logits = predictor.add_new_mask(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        mask=seed_mask,
                    )
                    seed_mode = "mask"
                else:
                    if previous_bbox_xyxy is None:
                        raise RuntimeError("previous bbox seed is unavailable and mask seeding is disabled or unsupported.")
                    local_idx, obj_ids, mask_logits = predictor.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=0,
                        obj_id=1,
                        box=np.asarray(previous_bbox_xyxy, dtype=np.float32),
                    )
                seed_update_ms = (time.perf_counter() - seed_start) * 1000.0
                result_by_local_index[int(local_idx)] = self._tracked_frame_from_logits(
                    mask_logits=mask_logits,
                    frame_index=previous_frame_index,
                    local_frame_index=int(local_idx),
                    update_ms=seed_update_ms,
                    obj_ids=obj_ids,
                )

                iterator = predictor.propagate_in_video(
                    inference_state,
                    start_frame_idx=0,
                    max_frame_num_to_track=2,
                    reverse=False,
                )
                while True:
                    start = time.perf_counter()
                    try:
                        local_idx, obj_ids, mask_logits = next(iterator)
                    except StopIteration:
                        break
                    update_ms = (time.perf_counter() - start) * 1000.0
                    result_by_local_index[int(local_idx)] = self._tracked_frame_from_logits(
                        mask_logits=mask_logits,
                        frame_index=previous_frame_index if int(local_idx) == 0 else current_frame_index,
                        local_frame_index=int(local_idx),
                        update_ms=update_ms,
                        obj_ids=obj_ids,
                    )

        if 1 not in result_by_local_index:
            raise RuntimeError("SAM2 incremental live tracking did not return the current frame mask.")
        tracked_frame = result_by_local_index[1]
        tracked_frame.mask_state.debug.update(
            {
                "tracking_mode": "incremental_pair",
                "seed_mode": seed_mode,
                "seed_frame_index": previous_frame_index,
                "current_frame_index": current_frame_index,
                "live_history_frames": 2,
            }
        )
        return tracked_frame

    def _store_live_seed(self, frame: np.ndarray, frame_index: int, tracked_frame: SAM2TrackedFrame) -> None:
        self._last_live_frame = self._coerce_rgb_frame(frame)
        self._last_live_frame_index = int(frame_index)
        self._last_live_mask_state = tracked_frame.mask_state
        if tracked_frame.mask_state.visible:
            self._last_live_mask = np.asarray(tracked_frame.mask, dtype=np.uint8)
            self._last_live_bbox_xyxy = tracked_frame.mask_state.bbox_xyxy
        elif self._last_live_bbox_xyxy is None:
            self._last_live_bbox_xyxy = self._seed_bbox_xyxy

    @staticmethod
    def _usable_seed_mask(mask: np.ndarray | None) -> np.ndarray | None:
        if mask is None:
            return None
        seed_mask = np.asarray(mask)
        if seed_mask.ndim == 3:
            seed_mask = seed_mask[:, :, 0]
        if seed_mask.ndim != 2:
            return None
        seed_mask = seed_mask > 0
        if not bool(np.any(seed_mask)):
            return None
        return np.ascontiguousarray(seed_mask)

    def load_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        self._ensure_sam2_on_path()
        import torch
        from sam2.build_sam import build_sam2_video_predictor

        config_file, ckpt_path = self._resolve_model_files(load_hub=True)
        device = self._select_device(torch)
        predictor = build_sam2_video_predictor(config_file=config_file, ckpt_path=str(ckpt_path), device=device)
        self._predictor = predictor
        self._predictor_info = {
            "model_id": self.config.model_id,
            "config_file": config_file,
            "checkpoint_path": str(ckpt_path),
            "device": device,
            "repo_path": None if self.config.repo_path is None else str(self.config.repo_path),
            "local_files_only": self.config.local_files_only,
        }
        return predictor

    @property
    def predictor_info(self) -> dict[str, Any]:
        return dict(self._predictor_info)

    @staticmethod
    def result_from_mask(mask_state: MaskState) -> InvServoResult:
        return InvServoResult.success({"mask": mask_state.to_dict()})

    def _ensure_sam2_on_path(self) -> None:
        if self.config.repo_path is None:
            return
        repo_path = Path(self.config.repo_path).expanduser()
        if not repo_path.exists():
            raise FileNotFoundError(f"SAM2 repo path not found: {repo_path}")
        repo_path_str = str(repo_path)
        if repo_path_str not in sys.path:
            sys.path.insert(0, repo_path_str)

    def _resolve_model_files(self, *, load_hub: bool) -> tuple[str, Path]:
        from sam2.build_sam import HF_MODEL_ID_TO_FILENAMES

        if self.config.model_id not in HF_MODEL_ID_TO_FILENAMES:
            raise ValueError(f"Unsupported SAM2 model_id: {self.config.model_id!r}")
        default_config_file, checkpoint_name = HF_MODEL_ID_TO_FILENAMES[self.config.model_id]
        config_file = self.config.config_file or default_config_file

        if self.config.checkpoint_path is not None:
            checkpoint_path = Path(self.config.checkpoint_path).expanduser()
        else:
            if not load_hub:
                checkpoint_path = self._cached_hf_checkpoint_path(checkpoint_name)
            else:
                from huggingface_hub import hf_hub_download

                checkpoint_path = Path(
                    hf_hub_download(
                        repo_id=self.config.model_id,
                        filename=checkpoint_name,
                        local_files_only=self.config.local_files_only,
                    )
                )
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"SAM2 checkpoint not found: {checkpoint_path}")
        return config_file, checkpoint_path

    def _cached_hf_checkpoint_path(self, checkpoint_name: str) -> Path:
        cache_root = Path.home() / ".cache" / "huggingface" / "hub"
        model_dir = cache_root / f"models--{self.config.model_id.replace('/', '--')}"
        refs_main = model_dir / "refs" / "main"
        if refs_main.exists():
            snapshot = model_dir / "snapshots" / refs_main.read_text(encoding="utf-8").strip()
            candidate = snapshot / checkpoint_name
            if candidate.exists():
                return candidate
        for candidate in model_dir.glob(f"snapshots/*/{checkpoint_name}"):
            if candidate.exists():
                return candidate
        return model_dir / "snapshots" / "missing" / checkpoint_name

    def _select_device(self, torch_module: Any) -> str:
        requested = str(self.config.device or "auto").lower()
        if requested == "auto":
            return "cuda" if torch_module.cuda.is_available() else "cpu"
        if requested.startswith("cuda") and not torch_module.cuda.is_available():
            raise RuntimeError(f"SAM2 requested {self.config.device!r}, but torch.cuda.is_available() is false.")
        return str(self.config.device)

    def _autocast_context(self) -> Any:
        if not self.config.use_fp16:
            return nullcontext()
        device = self._predictor_info.get("device", self.config.device)
        if str(device).startswith("cuda"):
            import torch

            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        return nullcontext()

    def _live_work_dir(self) -> Path:
        if self._live_tempdir is None:
            self._live_tempdir = tempfile.TemporaryDirectory(prefix="sam2_live_frames_")
        return Path(self._live_tempdir.name)

    def _write_jpeg_sequence(self, rgb_frames: list[np.ndarray], frame_dir: Path) -> None:
        frame_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frame_dir.glob("*.jpg"):
            old_frame.unlink()
        params = [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)]
        for local_index, rgb_frame in enumerate(rgb_frames):
            bgr_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            output_path = frame_dir / f"{local_index:05d}.jpg"
            if not cv2.imwrite(str(output_path), bgr_frame, params):
                raise RuntimeError(f"Failed to write SAM2 input frame: {output_path}")

    def _tracked_frame_from_logits(
        self,
        *,
        mask_logits: Any,
        frame_index: int,
        local_frame_index: int,
        update_ms: float,
        obj_ids: Iterable[int],
    ) -> SAM2TrackedFrame:
        mask = self._mask_from_logits(mask_logits)
        mask_state = self._mask_state_from_binary(
            mask,
            frame_index=frame_index,
            local_frame_index=local_frame_index,
            update_ms=update_ms,
        )
        fps = 1000.0 / update_ms if update_ms > 0.0 else 0.0
        return SAM2TrackedFrame(
            frame_index=frame_index,
            local_frame_index=local_frame_index,
            mask=mask,
            mask_state=mask_state,
            update_ms=update_ms,
            fps=fps,
            obj_ids=[int(obj_id) for obj_id in obj_ids],
        )

    def _mask_from_logits(self, mask_logits: Any) -> np.ndarray:
        tensor = mask_logits
        while getattr(tensor, "ndim", 0) > 2:
            tensor = tensor[0]
        mask = (tensor > float(self.config.mask_threshold)).detach().cpu().numpy()
        return mask.astype(np.uint8) * 255

    @staticmethod
    def _mask_state_from_binary(
        mask: np.ndarray,
        *,
        frame_index: int,
        local_frame_index: int,
        update_ms: float,
    ) -> MaskState:
        binary = np.asarray(mask) > 0
        height, width = binary.shape
        area = int(np.count_nonzero(binary))
        if area <= 0:
            return MaskState(
                frame_index=frame_index,
                image_size_hw=(height, width),
                mask_area_px=0,
                source="sam2",
                debug={"local_frame_index": local_frame_index, "update_ms": update_ms},
            )
        ys, xs = np.nonzero(binary)
        return MaskState(
            frame_index=frame_index,
            image_size_hw=(height, width),
            mask_area_px=area,
            centroid_uv=(float(xs.mean()), float(ys.mean())),
            bbox_xyxy=(float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)),
            source="sam2",
            debug={"local_frame_index": local_frame_index, "update_ms": update_ms},
        )

    @staticmethod
    def _coerce_rgb_frame(frame: Any) -> np.ndarray:
        array = np.asarray(frame)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError(f"rgb_frame must have shape HxWx3, got {array.shape}.")
        if array.dtype != np.uint8:
            array = np.clip(array, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(array)

    @staticmethod
    def _validate_frame_sizes(frames: list[np.ndarray]) -> None:
        first_shape = frames[0].shape
        for index, frame in enumerate(frames):
            if frame.shape != first_shape:
                raise ValueError(f"frame {index} shape {frame.shape} does not match first frame {first_shape}.")

    @staticmethod
    def _coerce_bbox(value: DetectionState | Iterable[float]) -> tuple[float, float, float, float]:
        if isinstance(value, DetectionState):
            raw = value.bbox_xyxy
        else:
            raw = tuple(value)
        if len(raw) != 4:
            raise ValueError("bbox_xyxy must contain four values.")
        x0, y0, x1, y1 = (float(item) for item in raw)
        if x1 <= x0 or y1 <= y0:
            raise ValueError(f"Invalid bbox_xyxy: {(x0, y0, x1, y1)}")
        return (x0, y0, x1, y1)

    @staticmethod
    def _payload_from_tracked_frame(frame: SAM2TrackedFrame) -> dict[str, Any]:
        payload = frame.to_dict(include_mask=True)
        payload["mask_state"] = frame.mask_state.to_dict()
        return payload

    @staticmethod
    def _failure_payload(reason: str, *, error: Exception | None = None) -> dict[str, Any]:
        return {
            "ok": False,
            "mask": None,
            "bbox_xyxy": None,
            "center_uv": None,
            "area_ratio": None,
            "fps": 0.0,
            "update_ms": 0.0,
            "failure_reason": reason if error is None else f"{reason}: {type(error).__name__}: {error}",
        }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM2 live tracker utilities.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--sam2-repo", type=Path, default=None)
    return parser.parse_args(argv)


def _run_self_test(config_path: Path | None, sam2_repo: Path | None) -> None:
    tracker_config = SAM2LiveTrackerConfig()
    if config_path is not None:
        from lerobot.projects.vlbiman_sa.inv_servo.config import load_inv_rgb_servo_config

        cfg = load_inv_rgb_servo_config(config_path)
        tracker_config = cfg.sam2.to_tracker_config()
    if sam2_repo is not None:
        tracker_config.repo_path = sam2_repo

    tracker = SAM2LiveTracker(tracker_config)
    environment = tracker.check_environment()
    if not environment.ok:
        raise RuntimeError(json.dumps(environment.to_dict(), ensure_ascii=False))

    rgb = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.circle(rgb, (160, 120), 34, (230, 210, 30), -1)
    tracked = tracker.track_sequence([rgb], frame_indices=[0], init_bbox_xyxy=(126, 86, 194, 154))
    if len(tracked) != 1 or not tracked[0].mask_state.visible:
        raise RuntimeError("SAM2 self-test returned an empty mask.")

    payload = {
        "ok": True,
        "environment": environment.to_dict(),
        "predictor": tracker.predictor_info,
        "mask": tracked[0].to_dict(include_mask=False),
    }
    print("SAM2_LIVE_TRACKER_INTERFACE_OK")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.self_test:
        print("SAM2_LIVE_TRACKER_NOOP")
        return
    _run_self_test(args.config, args.sam2_repo)


if __name__ == "__main__":
    main(sys.argv[1:])
