#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


def _maybe_reexec_in_repo_venv() -> None:
    if os.environ.get("PYTHON_BIN") or os.environ.get("CONDA_PREFIX") or os.environ.get("VIRTUAL_ENV"):
        return
    repo_root = Path(__file__).resolve().parents[5]
    default_conda_python = Path.home() / "miniconda3" / "envs" / "lerobot" / "bin" / "python"
    repo_python = default_conda_python if default_conda_python.exists() else Path(sys.executable)
    if not repo_python.exists():
        return
    if Path(sys.executable).resolve() == repo_python.resolve():
        return
    if os.environ.get("VLBIMAN_REEXEC") == "1":
        return
    env = os.environ.copy()
    env["VLBIMAN_REEXEC"] = "1"
    os.execve(str(repo_python), [str(repo_python), __file__, *sys.argv[1:]], env)


def _bootstrap_pythonpath() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    for candidate in (repo_root / "src", repo_root, repo_root / "lerobot_robot_cjjarm"):
        candidate_str = str(candidate)
        if candidate.exists() and candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_pythonpath()

from lerobot.projects.vlbiman_sa.runtime_env import prepare_mujoco_runtime, require_mujoco_viewer_backend

prepare_mujoco_runtime(argv=sys.argv[1:])

import mujoco
import mujoco.viewer

from lerobot.projects.vlbiman_sa.app.run_mujoco_dual_camera_scene import (
    _default_base_mjcf_path,
    _default_handeye_path,
    _default_output_root,
    _default_urdf_path,
    _optional_xyz_override,
    _write_json,
)
from lerobot.projects.vlbiman_sa.sim import (
    DEFAULT_SCENE_PRESET_NAME,
    DEFAULT_EXTERNAL_CAMERA_NAME,
    DEFAULT_WRIST_CAMERA_NAME,
    DualCameraSceneConfig,
    TargetSphereConfig,
    build_dual_camera_scene,
    load_base_from_camera_transform,
    load_wrist_camera_mount_pose,
    scene_preset_names,
    scene_preset_objects,
    scene_preset_summary,
)


VIEW_MODE_OVERVIEW = "overview"
VIEW_MODE_EXTERNAL = "external"
VIEW_MODE_WRIST = "wrist"
GLFW_KEY_ESCAPE = 256
GLFW_KEY_1 = 49
GLFW_KEY_2 = 50
GLFW_KEY_3 = 51
GLFW_KEY_KP_1 = 321
GLFW_KEY_KP_2 = 322
GLFW_KEY_KP_3 = 323


def _build_viewer_overlay_texts(*, overlay_line: str, scene_name: str) -> list[tuple[int, int, str, str]]:
    font = int(mujoco.mjtFontScale.mjFONTSCALE_150)
    gridpos = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
    return [
        (font, gridpos, "view", overlay_line),
        (font, gridpos, "keys", "1 external  2 wrist  3 overview/free  Esc exit"),
        (font, gridpos, "scene", scene_name),
    ]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open the pure MuJoCo dual-camera scene in an interactive viewer. "
            "Press 1 for the external Gemini camera, 2 for the wrist camera, 3 for overview, Esc to exit."
        )
    )
    parser.add_argument("--base-mjcf", type=Path, default=_default_base_mjcf_path())
    parser.add_argument("--urdf-path", type=Path, default=_default_urdf_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--run-id", type=str, default="viewer_latest")
    parser.add_argument("--target-x", type=float, default=0.45)
    parser.add_argument("--target-y", type=float, default=0.0)
    parser.add_argument("--target-z", type=float, default=0.08)
    parser.add_argument("--target-radius-m", type=float, default=0.022)
    parser.add_argument("--target-mass-kg", type=float, default=0.12)
    parser.add_argument(
        "--plate-x", type=float, default=None, help="Override green plate center X in world meters."
    )
    parser.add_argument(
        "--plate-y", type=float, default=None, help="Override green plate center Y in world meters."
    )
    parser.add_argument(
        "--plate-z", type=float, default=None, help="Override green plate center Z in world meters."
    )
    parser.add_argument(
        "--ball-x", type=float, default=None, help="Override yellow ball center X in world meters."
    )
    parser.add_argument(
        "--ball-y", type=float, default=None, help="Override yellow ball center Y in world meters."
    )
    parser.add_argument(
        "--ball-z", type=float, default=None, help="Override yellow ball center Z in world meters."
    )
    parser.add_argument(
        "--scene-preset",
        type=str,
        choices=scene_preset_names(),
        default=DEFAULT_SCENE_PRESET_NAME,
        help="Optional physical-object preset to append to the canonical scene.",
    )
    parser.add_argument("--settle-steps", type=int, default=300)
    parser.add_argument("--camera-fovy-deg", type=float, default=58.0)
    parser.add_argument("--display", type=str, default=None)
    parser.add_argument("--show-left-ui", action="store_true")
    parser.add_argument("--show-right-ui", action="store_true")
    parser.add_argument("--write-scene-only", action="store_true")
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _apply_overview_camera(handle: mujoco.viewer.Handle) -> None:
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    handle.cam.azimuth = 148.0
    handle.cam.elevation = -20.0
    handle.cam.distance = 1.75
    handle.cam.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)


def _apply_fixed_camera(handle: mujoco.viewer.Handle, *, model: mujoco.MjModel, camera_name: str) -> None:
    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in MuJoCo model.")
    handle.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
    handle.cam.fixedcamid = camera_id


def _set_view_mode(
    handle: mujoco.viewer.Handle,
    *,
    model: mujoco.MjModel,
    view_mode: str,
    external_camera_name: str,
    wrist_camera_name: str,
) -> str:
    if view_mode == VIEW_MODE_EXTERNAL:
        _apply_fixed_camera(handle, model=model, camera_name=external_camera_name)
        return f"view={external_camera_name} [1]"
    if view_mode == VIEW_MODE_WRIST:
        _apply_fixed_camera(handle, model=model, camera_name=wrist_camera_name)
        return f"view={wrist_camera_name} [2]"
    _apply_overview_camera(handle)
    return "view=overview [3]"


def _scene_output_dir(output_root: Path, run_id: str) -> Path:
    return output_root.resolve() / run_id


def _write_scene(
    *,
    base_mjcf_path: Path,
    urdf_path: Path,
    handeye_result_path: Path,
    output_root: Path,
    run_id: str,
    target_position_xyz_m: tuple[float, float, float],
    target_radius_m: float,
    target_mass_kg: float,
    green_plate_position_xyz_m: tuple[float | None, float | None, float | None] | None,
    yellow_ball_position_xyz_m: tuple[float | None, float | None, float | None] | None,
    scene_preset: str,
    camera_fovy_deg: float,
) -> tuple[Path, dict[str, Any]]:
    output_dir = _scene_output_dir(output_root, run_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    external_camera_transform = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)
    scene_config = DualCameraSceneConfig(
        camera_fovy_deg=float(camera_fovy_deg),
        include_target_cube=str(scene_preset).strip().lower() == DEFAULT_SCENE_PRESET_NAME,
        target=TargetSphereConfig(
            position_xyz_m=tuple(float(value) for value in target_position_xyz_m),
            radius_m=float(target_radius_m),
            mass_kg=float(target_mass_kg),
            rgba=(1.0, 1.0, 0.0, 0.0),
        ),
        objects=scene_preset_objects(
            scene_preset,
            center_xy_m=(float(target_position_xyz_m[0]), float(target_position_xyz_m[1])),
            green_plate_position_xyz_m=green_plate_position_xyz_m,
            yellow_ball_position_xyz_m=yellow_ball_position_xyz_m,
        ),
    )
    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=external_camera_transform,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=scene_config,
    )

    scene_path = output_dir / "scene" / "dual_camera_target_scene.mjcf"
    scene_path.parent.mkdir(parents=True, exist_ok=True)
    scene_path.write_text(artifacts.xml_text, encoding="utf-8")
    summary = {
        "status": "scene_written",
        "scene_path": str(scene_path.resolve()),
        "base_mjcf_path": str(base_mjcf_path.resolve()),
        "urdf_path": str(urdf_path.resolve()),
        "handeye_result_path": str(handeye_result_path.resolve()),
        "external_camera_name": scene_config.external_camera_name,
        "wrist_camera_name": scene_config.wrist_camera_name,
        "target_position_xyz_m": [float(value) for value in target_position_xyz_m],
        "target_radius_m": float(target_radius_m),
        "target_mass_kg": float(target_mass_kg),
        "scene_preset": scene_preset_summary(scene_preset),
        "objects": {
            scene_object.object_key: {
                "body_name": scene_object.body_name,
                "geom_name": scene_object.geom_name,
                "shape": scene_object.shape,
                "initial_position_xyz_m": [float(value) for value in scene_object.position_xyz_m],
                "size_xyz_m": [float(value) for value in scene_object.size_xyz_m],
                "container_rim_height_m": float(scene_object.container_rim_height_m),
                "container_rim_thickness_m": float(scene_object.container_rim_thickness_m),
                "container_rim_segments": int(scene_object.container_rim_segments),
            }
            for scene_object in scene_config.objects
        },
        "hardware_called": False,
        "camera_opened": False,
        "mujoco_available": True,
    }
    _write_json(output_dir / "viewer_scene_summary.json", summary)
    return scene_path, summary


def run_dual_camera_viewer(
    *,
    base_mjcf_path: Path,
    urdf_path: Path,
    handeye_result_path: Path,
    output_root: Path,
    run_id: str,
    target_position_xyz_m: tuple[float, float, float],
    target_radius_m: float,
    target_mass_kg: float,
    green_plate_position_xyz_m: tuple[float | None, float | None, float | None] | None,
    yellow_ball_position_xyz_m: tuple[float | None, float | None, float | None] | None,
    scene_preset: str,
    settle_steps: int,
    camera_fovy_deg: float,
    show_left_ui: bool,
    show_right_ui: bool,
    write_scene_only: bool,
) -> tuple[Path, dict[str, Any]]:
    scene_path, summary = _write_scene(
        base_mjcf_path=base_mjcf_path,
        urdf_path=urdf_path,
        handeye_result_path=handeye_result_path,
        output_root=output_root,
        run_id=run_id,
        target_position_xyz_m=target_position_xyz_m,
        target_radius_m=target_radius_m,
        target_mass_kg=target_mass_kg,
        green_plate_position_xyz_m=green_plate_position_xyz_m,
        yellow_ball_position_xyz_m=yellow_ball_position_xyz_m,
        scene_preset=scene_preset,
        camera_fovy_deg=camera_fovy_deg,
    )
    if write_scene_only:
        return scene_path, summary

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for _ in range(max(int(settle_steps), 0)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    keyboard_state = {
        "exit_requested": False,
        "view_mode": VIEW_MODE_OVERVIEW,
    }

    def _key_callback(keycode: int) -> None:
        if keycode == GLFW_KEY_ESCAPE:
            keyboard_state["exit_requested"] = True
        elif keycode in {GLFW_KEY_1, GLFW_KEY_KP_1}:
            keyboard_state["view_mode"] = VIEW_MODE_EXTERNAL
        elif keycode in {GLFW_KEY_2, GLFW_KEY_KP_2}:
            keyboard_state["view_mode"] = VIEW_MODE_WRIST
        elif keycode in {GLFW_KEY_3, GLFW_KEY_KP_3}:
            keyboard_state["view_mode"] = VIEW_MODE_OVERVIEW

    require_mujoco_viewer_backend()
    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=_key_callback,
        show_left_ui=bool(show_left_ui),
        show_right_ui=bool(show_right_ui),
    )
    try:
        active_view_mode: str | None = None
        overlay_line = ""
        while viewer.is_running():
            if keyboard_state["exit_requested"]:
                break
            requested_view_mode = str(keyboard_state["view_mode"])
            with viewer.lock():
                if requested_view_mode != active_view_mode:
                    overlay_line = _set_view_mode(
                        viewer,
                        model=model,
                        view_mode=requested_view_mode,
                        external_camera_name=DEFAULT_EXTERNAL_CAMERA_NAME,
                        wrist_camera_name=DEFAULT_WRIST_CAMERA_NAME,
                    )
                    active_view_mode = requested_view_mode
                viewer.set_texts(_build_viewer_overlay_texts(overlay_line=overlay_line, scene_name=scene_path.name))
            viewer.sync()
            time.sleep(1.0 / 60.0)
    finally:
        try:
            with viewer.lock():
                viewer.clear_texts()
        except Exception:
            logging.exception("Failed to clear viewer texts.")
        viewer.close()
    return scene_path, summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    if args.display is not None:
        os.environ["DISPLAY"] = args.display
    scene_path, _ = run_dual_camera_viewer(
        base_mjcf_path=args.base_mjcf,
        urdf_path=args.urdf_path,
        handeye_result_path=args.handeye_result,
        output_root=args.output_root,
        run_id=args.run_id,
        target_position_xyz_m=(float(args.target_x), float(args.target_y), float(args.target_z)),
        target_radius_m=float(args.target_radius_m),
        target_mass_kg=float(args.target_mass_kg),
        green_plate_position_xyz_m=_optional_xyz_override(args.plate_x, args.plate_y, args.plate_z),
        yellow_ball_position_xyz_m=_optional_xyz_override(args.ball_x, args.ball_y, args.ball_z),
        scene_preset=str(args.scene_preset),
        settle_steps=int(args.settle_steps),
        camera_fovy_deg=float(args.camera_fovy_deg),
        show_left_ui=bool(args.show_left_ui),
        show_right_ui=bool(args.show_right_ui),
        write_scene_only=bool(args.write_scene_only),
    )
    logging.info("Dual-camera viewer scene ready at %s", scene_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
