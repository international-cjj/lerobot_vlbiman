from __future__ import annotations

from lerobot.projects.vlbiman_sa.inv_servo.rgb_servo_controller import RGBServoController, RGBServoControllerConfig
from lerobot.projects.vlbiman_sa.inv_servo.target_state import MaskState, ServoTarget


def test_real_axis_signs_flip_u_and_keep_approach_command() -> None:
    controller = RGBServoController(
        RGBServoControllerConfig(
            axis_sign_x=-1.0,
            axis_sign_y=-1.0,
            axis_sign_z=1.0,
        )
    )
    target = ServoTarget(
        phrase="redcan",
        mask=MaskState(
            frame_index=50,
            image_size_hw=(1080, 1920),
            mask_area_px=143_521,
            centroid_uv=(1029.35, 804.92),
        ),
    )
    current = MaskState(
        frame_index=1,
        image_size_hw=(1080, 1920),
        mask_area_px=52_679,
        centroid_uv=(1044.20, 524.90),
    )

    result = controller.compute_command(current, target)

    assert result.ok
    assert result.state is not None
    command = result.state["command"]
    assert command["debug"]["axis_sign"] == [-1.0, -1.0, 1.0]
    assert command["debug"]["raw_delta_xyz_m"][0] < 0.0
    assert command["delta_cam"][0] > 0.0
    assert command["delta_cam"][1] < 0.0
    assert command["delta_cam"][2] > 0.0


def test_default_v_tolerance_accepts_small_real_residual() -> None:
    controller = RGBServoController(RGBServoControllerConfig())
    target = ServoTarget(
        phrase="redcan",
        mask=MaskState(
            frame_index=50,
            image_size_hw=(1000, 1000),
            mask_area_px=10_000,
            centroid_uv=(500.0, 500.0),
        ),
    )
    current = MaskState(
        frame_index=1,
        image_size_hw=(1000, 1000),
        mask_area_px=10_000,
        centroid_uv=(500.0, 540.0),
    )

    result = controller.compute_error(current, target, mask_iou=0.71)

    assert result.ok
    assert result.state is not None
    assert result.state["error"]["dv_norm"] == -0.04
    assert result.state["error"]["converged"]
