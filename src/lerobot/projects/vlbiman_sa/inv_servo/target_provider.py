from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .metrics import mask_state_from_mask
from .target_state import (
    InvServoResult,
    MaskState,
    ServoTarget,
    TargetState,
    target_state_from_bbox,
    target_state_from_mask,
)


@dataclass(slots=True)
class TargetProviderConfig:
    phrase: str = "yellow ball"
    target_frame_index: int = 100
    target_frame_path: Path | None = None
    target_mask_path: Path | None = None
    target_state_path: Path | None = None
    feature_bins: int = 8


class TargetProvider:
    def __init__(self, config: TargetProviderConfig | None = None):
        self.config = config or TargetProviderConfig()

    def target_phrase(self) -> str:
        return self.config.phrase

    def from_mask_state(self, mask_state: MaskState, *, phrase: str | None = None) -> InvServoResult:
        target = ServoTarget(phrase=phrase or self.config.phrase, mask=mask_state)
        return InvServoResult.success({"target": target.to_dict()})

    def load(self, request: dict[str, Any] | None = None) -> dict[str, Any]:
        request = dict(request or {})
        phrase = str(request.get("target_phrase") or request.get("phrase") or self.config.phrase)
        if not phrase:
            return self._failure("missing_target_request")

        frame_index = int(request.get("target_frame_index", self.config.target_frame_index))
        target_frame = self._coerce_frame(request.get("target_frame"))
        target_mask = self._coerce_mask(request.get("target_mask"))

        target_frame_path = self._request_path(
            request,
            keys=("target_frame_path", "frame_path"),
            fallback=self.config.target_frame_path,
        )
        target_mask_path = self._request_path(
            request,
            keys=("target_mask_path", "mask_path"),
            fallback=self.config.target_mask_path,
        )
        target_state_path = self._request_path(
            request,
            keys=("target_state_path", "state_path"),
            fallback=self.config.target_state_path,
        )

        if target_frame is None and target_frame_path is not None:
            frame_result = self._read_rgb_frame(target_frame_path)
            if not frame_result["ok"]:
                return self._failure(frame_result["failure_reason"], target_phrase=phrase, target_frame_path=target_frame_path)
            target_frame = frame_result["frame"]

        if target_mask is None and target_mask_path is not None:
            mask_result = self._read_mask(target_mask_path)
            if not mask_result["ok"]:
                return self._failure(mask_result["failure_reason"], target_phrase=phrase, target_mask_path=target_mask_path)
            target_mask = mask_result["mask"]

        if target_mask is not None:
            try:
                target_state = target_state_from_mask(
                    target_mask,
                    frame_index=frame_index,
                    rgb_frame=target_frame,
                    source="target_mask",
                    mask_path=target_mask_path,
                    frame_path=target_frame_path,
                    feature_bins=self.config.feature_bins,
                )
            except (TypeError, ValueError) as exc:
                return self._failure("invalid_target_mask", target_phrase=phrase, debug={"error": str(exc)})
            return self._success(
                target_phrase=phrase,
                target_frame=target_frame,
                target_mask=target_mask,
                target_state=target_state,
            )

        external_state = request.get("target_state")
        if external_state is None and target_state_path is not None:
            state_result = self._read_target_state(target_state_path)
            if not state_result["ok"]:
                return self._failure(
                    state_result["failure_reason"],
                    target_phrase=phrase,
                    target_state_path=target_state_path,
                )
            external_state = state_result["target_state"]

        if external_state is not None:
            try:
                target_state = (
                    external_state
                    if isinstance(external_state, TargetState)
                    else TargetState.from_dict(dict(external_state))
                )
            except (KeyError, TypeError, ValueError) as exc:
                return self._failure("invalid_target_state", target_phrase=phrase, debug={"error": str(exc)})
            return self._success(
                target_phrase=phrase,
                target_frame=target_frame,
                target_mask=target_state.target_mask,
                target_state=target_state,
            )

        bbox = request.get("bbox_xyxy") or request.get("target_bbox_xyxy")
        image_size_hw = request.get("image_size_hw")
        if bbox is not None:
            if image_size_hw is None and target_frame is not None:
                image_size_hw = tuple(np.asarray(target_frame).shape[:2])
            if image_size_hw is None:
                return self._failure("invalid_target_state", target_phrase=phrase, debug={"error": "image_size_hw missing"})
            try:
                target_state = target_state_from_bbox(
                    bbox,
                    image_size_hw=image_size_hw,
                    frame_index=frame_index,
                    rgb_frame=target_frame,
                    source="target_bbox",
                    frame_path=target_frame_path,
                    feature_bins=self.config.feature_bins,
                )
            except (TypeError, ValueError) as exc:
                return self._failure("invalid_target_state", target_phrase=phrase, debug={"error": str(exc)})
            return self._success(
                target_phrase=phrase,
                target_frame=target_frame,
                target_mask=None,
                target_state=target_state,
            )

        if target_state_path is not None:
            return self._failure("target_state_not_found", target_phrase=phrase, target_state_path=target_state_path)
        if target_mask_path is not None:
            return self._failure("target_mask_not_found", target_phrase=phrase, target_mask_path=target_mask_path)
        if target_frame_path is not None:
            return self._failure("target_frame_not_found", target_phrase=phrase, target_frame_path=target_frame_path)
        return self._failure("missing_target_request", target_phrase=phrase)

    def load_target_mask(self, mask_path: Path | None = None) -> InvServoResult:
        raw_path = mask_path or self.config.target_mask_path
        if raw_path is None:
            return InvServoResult.failure("target_mask_not_found")
        resolved_path = Path(raw_path)
        if not resolved_path.exists():
            return InvServoResult.failure("target_mask_not_found", {"target_mask_path": str(resolved_path)})

        mask = cv2.imread(str(resolved_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return InvServoResult.failure("invalid_target_mask", {"target_mask_path": str(resolved_path)})

        mask_state = mask_state_from_mask(
            mask,
            frame_index=self.config.target_frame_index,
            source="target_mask_file",
            mask_path=resolved_path,
        )
        return self.from_mask_state(mask_state)

    def provide(self, current_state: dict[str, Any] | None = None) -> InvServoResult:
        result = self.load(current_state)
        if not result["ok"]:
            return InvServoResult.failure(str(result["failure_reason"]), self._jsonable_payload(result))
        state = self._jsonable_payload(result)
        return InvServoResult.success(state)

    @staticmethod
    def _request_path(request: dict[str, Any], *, keys: tuple[str, ...], fallback: Path | None) -> Path | None:
        for key in keys:
            value = request.get(key)
            if value is not None:
                return Path(value)
        return fallback

    @staticmethod
    def _coerce_frame(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value)

    @staticmethod
    def _coerce_mask(value: Any) -> np.ndarray | None:
        if value is None:
            return None
        return np.asarray(value)

    @staticmethod
    def _read_rgb_frame(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"ok": False, "failure_reason": "target_frame_not_found"}
        frame_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            return {"ok": False, "failure_reason": "target_frame_not_found"}
        return {"ok": True, "frame": cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB), "failure_reason": None}

    @staticmethod
    def _read_mask(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"ok": False, "failure_reason": "target_mask_not_found"}
        mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return {"ok": False, "failure_reason": "invalid_target_mask"}
        return {"ok": True, "mask": mask, "failure_reason": None}

    @staticmethod
    def _read_target_state(path: Path) -> dict[str, Any]:
        if not path.exists():
            return {"ok": False, "failure_reason": "target_state_not_found"}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return {"ok": False, "failure_reason": "invalid_target_state"}
        return {"ok": True, "target_state": payload, "failure_reason": None}

    @staticmethod
    def _success(
        *,
        target_phrase: str,
        target_frame: np.ndarray | None,
        target_mask: np.ndarray | None,
        target_state: TargetState,
    ) -> dict[str, Any]:
        return {
            "ok": True,
            "target_phrase": target_phrase,
            "target_frame": target_frame,
            "target_mask": target_mask,
            "target_state": target_state.to_dict(),
            "failure_reason": None,
        }

    @staticmethod
    def _failure(
        failure_reason: str | None,
        *,
        target_phrase: str | None = None,
        target_frame_path: Path | None = None,
        target_mask_path: Path | None = None,
        target_state_path: Path | None = None,
        debug: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "ok": False,
            "target_phrase": target_phrase,
            "target_frame": None,
            "target_mask": None,
            "target_state": None,
            "failure_reason": failure_reason or "missing_target_request",
            "debug": {
                "target_frame_path": None if target_frame_path is None else str(target_frame_path),
                "target_mask_path": None if target_mask_path is None else str(target_mask_path),
                "target_state_path": None if target_state_path is None else str(target_state_path),
                **(debug or {}),
            },
        }

    @staticmethod
    def _jsonable_payload(payload: dict[str, Any]) -> dict[str, Any]:
        result = dict(payload)
        frame = result.get("target_frame")
        mask = result.get("target_mask")
        result["target_frame"] = None if frame is None else {"shape": list(np.asarray(frame).shape)}
        result["target_mask"] = None if mask is None else {"shape": list(np.asarray(mask).shape)}
        return result
