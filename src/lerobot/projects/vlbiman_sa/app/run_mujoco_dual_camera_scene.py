#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import cv2
import numpy as np

os.environ.setdefault("MUJOCO_GL", "egl")


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

import mujoco

from lerobot.projects.vlbiman_sa.sim import (
    DEFAULT_SCENE_PRESET_NAME,
    DEFAULT_EXTERNAL_CAMERA_NAME,
    DEFAULT_TARGET_BODY_NAME,
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
from lerobot.utils.rotation import Rotation


def _default_base_mjcf_path() -> Path:
    return (
        REPO_ROOT
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )


def _default_urdf_path() -> Path:
    return (
        REPO_ROOT
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )


def _default_handeye_path() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "mujoco_dual_camera_scene"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a pure MuJoCo dual-camera scene using the static Gemini hand-eye transform, "
            "the CJJ arm wrist-camera mount from URDF, and a virtual graspable spherical target."
        )
    )
    parser.add_argument("--base-mjcf", type=Path, default=_default_base_mjcf_path())
    parser.add_argument("--urdf-path", type=Path, default=_default_urdf_path())
    parser.add_argument("--handeye-result", type=Path, default=_default_handeye_path())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--run-id", type=str, default="latest")
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
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument("--camera-fovy-deg", type=float, default=58.0)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args(argv)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _optional_xyz_override(
    x: float | None,
    y: float | None,
    z: float | None,
) -> tuple[float | None, float | None, float | None] | None:
    if x is None and y is None and z is None:
        return None
    return (x, y, z)


def _save_rgb(path: Path, frame_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame_bgr = cv2.cvtColor(np.asarray(frame_rgb, dtype=np.uint8), cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), frame_bgr)


def _render_named_camera(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    camera_name: str,
    width: int,
    height: int,
) -> np.ndarray:
    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
    if camera_id < 0:
        raise ValueError(f"Camera '{camera_name}' not found in MuJoCo scene.")
    renderer = mujoco.Renderer(model, int(height), int(width))
    try:
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
        camera.fixedcamid = camera_id
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8)
    finally:
        renderer.close()


def _render_overview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    renderer = mujoco.Renderer(model, int(height), int(width))
    try:
        camera = mujoco.MjvCamera()
        camera.azimuth = 148.0
        camera.elevation = -20.0
        camera.distance = 1.75
        camera.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera=camera)
        return np.asarray(renderer.render(), dtype=np.uint8)
    finally:
        renderer.close()


def _body_pose_dict(model: mujoco.MjModel, data: mujoco.MjData, body_name: str) -> dict[str, Any]:
    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name))
    if body_id < 0:
        raise ValueError(f"Body '{body_name}' not found in MuJoCo scene.")
    xyz = np.asarray(data.xpos[body_id], dtype=float).reshape(3)
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3)
    quat_xyzw = Rotation.from_matrix(rotation).as_quat()
    return {
        "body_name": body_name,
        "xyz_m": [float(value) for value in xyz],
        "quat_xyzw": [float(value) for value in quat_xyzw],
        "rotation_matrix": [[float(value) for value in row] for row in rotation],
    }


def run_dual_camera_scene(
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
    render_width: int,
    render_height: int,
    camera_fovy_deg: float,
) -> tuple[Path, dict[str, Any]]:
    output_dir = output_root.resolve() / run_id
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

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for _ in range(max(int(settle_steps), 0)):
        mujoco.mj_step(model, data)
    mujoco.mj_forward(model, data)

    external_rgb = _render_named_camera(
        model,
        data,
        camera_name=scene_config.external_camera_name,
        width=int(render_width),
        height=int(render_height),
    )
    wrist_rgb = _render_named_camera(
        model,
        data,
        camera_name=scene_config.wrist_camera_name,
        width=int(render_width),
        height=int(render_height),
    )
    overview_rgb = _render_overview(model, data, width=int(render_width), height=int(render_height))

    external_image_path = output_dir / f"{scene_config.external_camera_name}.png"
    wrist_image_path = output_dir / f"{scene_config.wrist_camera_name}.png"
    overview_image_path = output_dir / "overview.png"
    _save_rgb(external_image_path, external_rgb)
    _save_rgb(wrist_image_path, wrist_rgb)
    _save_rgb(overview_image_path, overview_rgb)

    summary = {
        "status": "ok",
        "scene_path": str(scene_path.resolve()),
        "base_mjcf_path": str(base_mjcf_path.resolve()),
        "urdf_path": str(urdf_path.resolve()),
        "handeye_result_path": str(handeye_result_path.resolve()),
        "settle_steps": int(settle_steps),
        "scene_preset": scene_preset_summary(scene_preset),
        "render_width": int(render_width),
        "render_height": int(render_height),
        "external_camera": {
            "camera_name": scene_config.external_camera_name,
            "body_name": scene_config.external_camera_body_name,
            "image_path": str(external_image_path.resolve()),
            "transform_base": artifacts.external_camera_transform_base.tolist(),
            "body_pose_world": _body_pose_dict(model, data, scene_config.external_camera_body_name),
        },
        "wrist_camera": {
            "camera_name": scene_config.wrist_camera_name,
            "body_name": scene_config.wrist_camera_body_name,
            "parent_body_name": scene_config.wrist_camera_parent_body,
            "image_path": str(wrist_image_path.resolve()),
            "transform_parent": artifacts.wrist_camera_transform_parent.tolist(),
            "body_pose_world": _body_pose_dict(model, data, scene_config.wrist_camera_body_name),
        },
        "virtual_target": {
            "body_name": scene_config.target.body_name,
            "geom_name": scene_config.target.geom_name,
            "initial_position_xyz_m": [float(value) for value in target_position_xyz_m],
            "radius_m": float(target_radius_m),
            "mass_kg": float(target_mass_kg),
            "body_pose_world": _body_pose_dict(model, data, scene_config.target.body_name),
        },
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
                "body_pose_world": _body_pose_dict(model, data, scene_object.body_name),
            }
            for scene_object in scene_config.objects
        },
        "overview_image_path": str(overview_image_path.resolve()),
        "mujoco_available": True,
        "hardware_called": False,
        "camera_opened": False,
    }
    summary_path = output_dir / "summary.json"
    _write_json(summary_path, summary)
    return summary_path, summary


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    summary_path, _ = run_dual_camera_scene(
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
        render_width=int(args.render_width),
        render_height=int(args.render_height),
        camera_fovy_deg=float(args.camera_fovy_deg),
    )
    logging.info("Pure MuJoCo dual-camera scene summary written to %s", summary_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
