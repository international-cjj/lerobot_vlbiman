from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path
import sys

import cv2
import numpy as np

from .target_state import InvServoResult, MaskState, ServoCommand, ServoError, ServoTarget


def _clamp(value: float, limit: float) -> float:
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


@dataclass(slots=True)
class RGBServoControllerConfig:
    k_u: float = 0.015
    k_v: float = 0.015
    k_a: float = 0.010
    axis_sign_x: float = 1.0
    axis_sign_y: float = 1.0
    axis_sign_z: float = 1.0
    max_step_xy_m: float = 0.003
    max_step_z_m: float = 0.004
    center_tol_u: float = 0.035
    center_tol_v: float = 0.045
    area_tol: float = 0.12
    mask_iou_tol: float = 0.70


class RGBServoController:
    def __init__(self, config: RGBServoControllerConfig | None = None):
        self.config = config or RGBServoControllerConfig()

    def compute_error(
        self,
        current_mask: MaskState,
        target: ServoTarget,
        *,
        mask_iou: float | None = None,
    ) -> InvServoResult:
        if not current_mask.visible:
            return InvServoResult.failure("current_mask_not_visible", {"current_mask": current_mask.to_dict()})
        if not target.mask.visible:
            return InvServoResult.failure("target_mask_not_visible", {"target": target.to_dict()})

        height, width = current_mask.image_size_hw
        target_height, target_width = target.mask.image_size_hw
        if (height, width) != (target_height, target_width):
            return InvServoResult.failure(
                "mask_image_size_mismatch",
                {"current_size_hw": [height, width], "target_size_hw": [target_height, target_width]},
            )

        current_u, current_v = current_mask.centroid_uv or (0.0, 0.0)
        target_u, target_v = target.mask.centroid_uv or (0.0, 0.0)
        e_u = (current_u - target_u) / max(float(width), 1.0)
        e_v = (current_v - target_v) / max(float(height), 1.0)
        du_norm = -e_u
        dv_norm = -e_v

        eps = 1e-6
        current_area_ratio = float(current_mask.mask_area_px) / max(float(height * width), 1.0)
        target_area_ratio = float(target.mask.mask_area_px) / max(float(height * width), 1.0)
        area_ratio_error = math.log((target_area_ratio + eps) / (current_area_ratio + eps))
        converged = (
            abs(du_norm) <= self.config.center_tol_u
            and abs(dv_norm) <= self.config.center_tol_v
            and abs(area_ratio_error) <= self.config.area_tol
            and (mask_iou is None or mask_iou >= self.config.mask_iou_tol)
        )
        error = ServoError(
            du_norm=du_norm,
            dv_norm=dv_norm,
            area_ratio_error=area_ratio_error,
            mask_iou=mask_iou,
            converged=converged,
            debug={
                "current_centroid_uv": list(current_mask.centroid_uv or (0.0, 0.0)),
                "target_centroid_uv": list(target.mask.centroid_uv or (0.0, 0.0)),
                "current_area_px": current_mask.mask_area_px,
                "target_area_px": target.mask.mask_area_px,
                "current_area_ratio": current_area_ratio,
                "target_area_ratio": target_area_ratio,
            },
        )
        return InvServoResult.success({"error": error.to_dict()})

    def compute_command(
        self,
        current_mask: MaskState,
        target: ServoTarget,
        *,
        mask_iou: float | None = None,
    ) -> InvServoResult:
        error_result = self.compute_error(current_mask, target, mask_iou=mask_iou)
        if not error_result.ok or error_result.state is None:
            return error_result

        error_state = error_result.state["error"]
        error = ServoError(
            du_norm=error_state["du_norm"],
            dv_norm=error_state["dv_norm"],
            area_ratio_error=error_state["area_ratio_error"],
            mask_iou=error_state["mask_iou"],
            converged=error_state["converged"],
            debug=error_state["debug"],
        )
        if error.converged:
            command = ServoCommand.zero(reason="target_aligned")
        else:
            raw_delta_xyz_m = (
                _clamp(self.config.k_u * error.du_norm, self.config.max_step_xy_m),
                _clamp(self.config.k_v * error.dv_norm, self.config.max_step_xy_m),
                _clamp(self.config.k_a * error.area_ratio_error, self.config.max_step_z_m),
            )
            signed_delta_xyz_m = (
                self.config.axis_sign_x * raw_delta_xyz_m[0],
                self.config.axis_sign_y * raw_delta_xyz_m[1],
                self.config.axis_sign_z * raw_delta_xyz_m[2],
            )
            command = ServoCommand(
                delta_xyz_m=signed_delta_xyz_m,
                delta_rpy_rad=(0.0, 0.0, 0.0),
                debug={
                    "controller": "centroid_area_p",
                    "rotation_policy": "zero_rpy_placeholder",
                    "axis_sign": [self.config.axis_sign_x, self.config.axis_sign_y, self.config.axis_sign_z],
                    "raw_delta_xyz_m": list(raw_delta_xyz_m),
                },
            )

        if not all(math.isfinite(value) for value in command.delta_xyz_m):
            return InvServoResult.failure("non_finite_servo_command", {"command": command.to_dict()})
        return InvServoResult.success({"error": error.to_dict(), "command": command.to_dict()})


def _self_test_mask(center_u: int, center_v: int, radius: int = 16) -> np.ndarray:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.circle(mask, (int(center_u), int(center_v)), int(radius), 255, -1)
    return mask


def _mask_state(mask: np.ndarray, *, frame_index: int, source: str) -> MaskState:
    from .metrics import mask_state_from_mask

    return mask_state_from_mask(mask, frame_index=frame_index, source=source)


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RGB mask visual-servo controller utilities.")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--config", type=Path, default=None)
    return parser.parse_args(argv)


def _run_self_test(config_path: Path | None = None) -> None:
    controller_config = RGBServoControllerConfig()
    if config_path is not None:
        from lerobot.projects.vlbiman_sa.inv_servo.config import load_inv_rgb_servo_config

        controller_config = load_inv_rgb_servo_config(config_path).servo.to_controller_config()

    controller = RGBServoController(controller_config)
    current_mask = _self_test_mask(70, 60, 14)
    target_mask = _self_test_mask(82, 60, 18)
    current_state = _mask_state(current_mask, frame_index=75, source="self_test_current")
    target_state = _mask_state(target_mask, frame_index=100, source="self_test_target")
    target = ServoTarget(phrase="yellow ball", mask=target_state)

    from .metrics import mask_iou

    command_result = controller.compute_command(current_state, target, mask_iou=mask_iou(current_mask, target_mask))
    if not command_result.ok or command_result.state is None:
        raise RuntimeError(json.dumps(command_result.to_dict(), ensure_ascii=False))

    command = command_result.state["command"]
    error = command_result.state["error"]
    if len(command["delta_cam"]) != 3:
        raise RuntimeError("delta_cam must contain three values.")
    if command["delta_pose"]["translation"] != command["delta_cam"]:
        raise RuntimeError("delta_pose.translation must mirror delta_cam.")
    if len(command["delta_pose"]["rotation_rpy"]) != 3:
        raise RuntimeError("delta_pose.rotation_rpy must contain three values.")
    if not all(math.isfinite(float(value)) for value in command["delta_cam"]):
        raise RuntimeError("delta_cam contains non-finite values.")
    for key in ("e_u", "e_v", "e_a", "error_norm", "mask_iou"):
        if key not in error:
            raise RuntimeError(f"missing error field: {key}")

    aligned_result = controller.compute_command(target_state, target, mask_iou=1.0)
    if not aligned_result.ok or aligned_result.state is None:
        raise RuntimeError(json.dumps(aligned_result.to_dict(), ensure_ascii=False))
    if not aligned_result.state["error"]["converged"]:
        raise RuntimeError("aligned mask should be converged.")

    print("SERVO_CONTROLLER_OK")
    print(json.dumps(command_result.state, ensure_ascii=False, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if not args.self_test:
        print("SERVO_CONTROLLER_NOOP")
        return
    _run_self_test(args.config)


if __name__ == "__main__":
    main(sys.argv[1:])
