from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from PIL import Image
from sam2.sam2_video_predictor import SAM2VideoPredictor
from transformers import AutoModelForCausalLM, AutoProcessor

from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord


@dataclass(slots=True)
class VLMObjectSegmentorConfig:
    florence_model_id: str = "microsoft/Florence-2-base"
    sam2_model_id: str = "facebook/sam2-hiera-small"
    task_prompt: str = "<CAPTION_TO_PHRASE_GROUNDING>"
    local_files_only: bool = True
    max_new_tokens: int = 256
    num_beams: int = 3
    seed_search_stride: int = 20
    max_seed_frames: int = 8
    mask_threshold: float = 0.0
    jpeg_quality: int = 95
    offload_video_to_cpu: bool = True
    offload_state_to_cpu: bool = True


@dataclass(slots=True)
class FlorenceDetection:
    frame_index: int
    local_frame_index: int
    label: str
    phrase: str
    bbox_xyxy: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentationFrameResult:
    frame_index: int
    local_frame_index: int
    score: float
    bbox_xyxy: list[int]
    component_count: int
    mask_area_px: int
    debug: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentationRunResult:
    masks: list[tuple[int, np.ndarray]]
    frame_results: list[SegmentationFrameResult]
    seed_detection: FlorenceDetection
    video_frame_dir: Path | None
    detection_log_path: Path | None


class VLMObjectSegmentor:
    def __init__(self, config: VLMObjectSegmentorConfig | None = None):
        self.config = config or VLMObjectSegmentorConfig()
        self._predictor: SAM2VideoPredictor | None = None
        self._processor: Any | None = None
        self._florence_model: Any | None = None

    def segment_video(
        self,
        session_dir: Path,
        records: list[FrameRecord],
        frame_indices: list[int],
        output_dir: Path,
        target_phrase: str,
        keep_artifacts: bool = True,
    ) -> SegmentationRunResult:
        output_dir.mkdir(parents=True, exist_ok=True)
        video_frame_dir = output_dir / "sam2_video_frames"
        video_frame_dir.mkdir(parents=True, exist_ok=True)
        mapping = self._prepare_jpeg_frames(session_dir, records, frame_indices, video_frame_dir)

        seed_detection = self._detect_seed_box(session_dir, records, frame_indices, target_phrase)
        detection_log_path = output_dir / "florence_seed_detection.json" if keep_artifacts else None
        if detection_log_path is not None:
            detection_log_path.write_text(
                json.dumps(seed_detection.to_dict(), indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

        predictor = self._get_predictor()
        local_frame_index = seed_detection.local_frame_index
        with torch.inference_mode(), self._autocast():
            inference_state = predictor.init_state(
                video_path=str(video_frame_dir),
                offload_video_to_cpu=self.config.offload_video_to_cpu,
                offload_state_to_cpu=self.config.offload_state_to_cpu,
            )
            results_by_local_idx: dict[int, np.ndarray] = {}
            _, _, initial_masks = predictor.add_new_points_or_box(
                inference_state,
                frame_idx=local_frame_index,
                obj_id=1,
                box=seed_detection.bbox_xyxy,
            )
            results_by_local_idx[local_frame_index] = np.asarray(
                initial_masks[0].detach().cpu().numpy()
            ).squeeze()
            for tracked_local_index, _, video_res_masks in predictor.propagate_in_video(inference_state):
                results_by_local_idx[int(tracked_local_index)] = np.asarray(
                    video_res_masks[0].detach().cpu().numpy()
                ).squeeze()

        masks: list[tuple[int, np.ndarray]] = []
        frame_results: list[SegmentationFrameResult] = []
        for local_idx, original_idx in enumerate(frame_indices):
            logits = results_by_local_idx.get(local_idx)
            if logits is None:
                mask = np.zeros_like(load_frame_assets(session_dir, records[original_idx])[1], dtype=np.uint8)
            else:
                logits = np.asarray(logits).squeeze()
                if logits.ndim != 2:
                    raise ValueError(
                        f"Expected a 2D SAM2 mask logit map, got shape {tuple(logits.shape)} for frame {original_idx}."
                    )
                mask = (logits > self.config.mask_threshold).astype(np.uint8) * 255
            bbox = self._bbox_from_mask(mask)
            masks.append((original_idx, mask))
            frame_results.append(
                SegmentationFrameResult(
                    frame_index=original_idx,
                    local_frame_index=local_idx,
                    score=1.0 if int(np.count_nonzero(mask)) > 0 else 0.0,
                    bbox_xyxy=bbox,
                    component_count=1 if int(np.count_nonzero(mask)) > 0 else 0,
                    mask_area_px=int(np.count_nonzero(mask)),
                    debug={
                        "seed_frame_index": seed_detection.frame_index,
                        "seed_local_frame_index": seed_detection.local_frame_index,
                        "seed_bbox_xyxy": seed_detection.bbox_xyxy,
                        "source": "florence2_plus_sam2",
                        "video_frame_path": str(mapping[local_idx]) if keep_artifacts else None,
                    },
                )
            )

        return SegmentationRunResult(
            masks=masks,
            frame_results=frame_results,
            seed_detection=seed_detection,
            video_frame_dir=video_frame_dir if keep_artifacts else None,
            detection_log_path=detection_log_path,
        )

    def _prepare_jpeg_frames(
        self,
        session_dir: Path,
        records: list[FrameRecord],
        frame_indices: list[int],
        output_dir: Path,
    ) -> dict[int, Path]:
        for stale_path in output_dir.glob("*.jpg"):
            stale_path.unlink()
        mapping: dict[int, Path] = {}
        for local_idx, original_idx in enumerate(frame_indices):
            jpg_path = output_dir / f"{local_idx:05d}.jpg"
            color_rgb, _ = load_frame_assets(session_dir, records[original_idx])
            color_bgr = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(jpg_path), color_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.config.jpeg_quality)])
            mapping[local_idx] = jpg_path
        return mapping

    def _detect_seed_box(
        self,
        session_dir: Path,
        records: list[FrameRecord],
        frame_indices: list[int],
        target_phrase: str,
    ) -> FlorenceDetection:
        processor, model = self._get_florence_processor_model()

        candidate_offsets = list(range(0, len(frame_indices), max(1, self.config.seed_search_stride)))
        if 0 not in candidate_offsets:
            candidate_offsets.insert(0, 0)
        candidate_offsets = candidate_offsets[: max(1, self.config.max_seed_frames)]

        detection: FlorenceDetection | None = None
        with torch.inference_mode(), self._autocast():
            for local_idx in candidate_offsets:
                original_idx = frame_indices[local_idx]
                color_rgb, _ = load_frame_assets(session_dir, records[original_idx])
                image = Image.fromarray(color_rgb)
                prompt = self.config.task_prompt + target_phrase
                inputs = processor(text=prompt, images=image, return_tensors="pt")
                inputs = {
                    key: value.to(self._device)
                    if hasattr(value, "to")
                    else value
                    for key, value in inputs.items()
                }
                if "pixel_values" in inputs:
                    inputs["pixel_values"] = inputs["pixel_values"].to(
                        self._device,
                        torch.float16 if torch.cuda.is_available() else torch.float32,
                    )
                generated_ids = model.generate(
                    **inputs,
                    max_new_tokens=self.config.max_new_tokens,
                    num_beams=self.config.num_beams,
                    do_sample=False,
                )
                generated_text = processor.batch_decode(generated_ids, skip_special_tokens=False)[0]
                parsed = processor.post_process_generation(
                    generated_text,
                    task=self.config.task_prompt,
                    image_size=image.size,
                )
                candidate = self._pick_detection(
                    parsed.get(self.config.task_prompt, {}),
                    target_phrase=target_phrase,
                    frame_index=original_idx,
                    local_frame_index=local_idx,
                )
                if candidate is not None:
                    detection = candidate
                    break

        if detection is None:
            raise RuntimeError(f"Florence-2 could not detect target phrase: {target_phrase!r}")
        return detection

    def _get_predictor(self) -> SAM2VideoPredictor:
        if self._predictor is None:
            self._predictor = SAM2VideoPredictor.from_pretrained(
                self.config.sam2_model_id,
                local_files_only=self.config.local_files_only,
            )
        return self._predictor

    def _get_florence_processor_model(self) -> tuple[Any, Any]:
        if self._processor is None:
            self._processor = AutoProcessor.from_pretrained(
                self.config.florence_model_id,
                trust_remote_code=True,
                local_files_only=self.config.local_files_only,
            )
        if self._florence_model is None:
            self._florence_model = AutoModelForCausalLM.from_pretrained(
                self.config.florence_model_id,
                trust_remote_code=True,
                local_files_only=self.config.local_files_only,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            ).to(self._device)
            self._florence_model.eval()
        return self._processor, self._florence_model

    @staticmethod
    def _pick_detection(
        payload: dict[str, Any],
        target_phrase: str,
        frame_index: int,
        local_frame_index: int,
    ) -> FlorenceDetection | None:
        bboxes = list(payload.get("bboxes", []))
        labels = list(payload.get("labels", []))
        target_phrase_lower = target_phrase.lower().strip()

        best_index = None
        best_area = -1.0
        for index, (bbox, label) in enumerate(zip(bboxes, labels)):
            label_lower = str(label).lower().strip()
            if target_phrase_lower not in label_lower and label_lower not in target_phrase_lower:
                continue
            x0, y0, x1, y1 = [float(item) for item in bbox]
            area = max(0.0, x1 - x0) * max(0.0, y1 - y0)
            if area > best_area:
                best_area = area
                best_index = index

        if best_index is None and bboxes:
            best_index = int(
                np.argmax(
                    [
                        max(0.0, float(bbox[2]) - float(bbox[0]))
                        * max(0.0, float(bbox[3]) - float(bbox[1]))
                        for bbox in bboxes
                    ]
                )
            )

        if best_index is None:
            return None

        bbox = [float(item) for item in bboxes[best_index]]
        label = str(labels[best_index]) if labels else target_phrase
        return FlorenceDetection(
            frame_index=frame_index,
            local_frame_index=local_frame_index,
            label=label,
            phrase=target_phrase,
            bbox_xyxy=bbox,
        )

    @staticmethod
    def _bbox_from_mask(mask: np.ndarray) -> list[int]:
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0:
            return [0, 0, 0, 0]
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @property
    def _device(self) -> torch.device:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _autocast():
        if torch.cuda.is_available():
            return torch.autocast("cuda", dtype=torch.float16)
        return nullcontext()
