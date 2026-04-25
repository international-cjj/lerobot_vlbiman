from __future__ import annotations

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lerobot.projects.vlbiman_sa.sim import (
    DEFAULT_EXTERNAL_CAMERA_NAME,
    DEFAULT_SCENE_PRESET_NAME,
    DEFAULT_TABLE_GEOM_NAME,
    DEFAULT_TARGET_BODY_NAME,
    DEFAULT_TARGET_CUBE_BODY_NAME,
    DEFAULT_TARGET_CUBE_GEOM_NAME,
    DEFAULT_WRIST_CAMERA_NAME,
    PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M,
    PROJECT_GREEN_CONTAINER_RIM_SEGMENTS,
    PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M,
    PROJECT_GREEN_PLATE_BODY_NAME,
    PROJECT_GREEN_PLATE_DIAMETER_M,
    PROJECT_GREEN_PLATE_GEOM_NAME,
    PROJECT_SCENE_PRESET_NAME,
    PROJECT_YELLOW_BALL_BASE_DIAMETER_M,
    PROJECT_YELLOW_BALL_BODY_NAME,
    PROJECT_YELLOW_BALL_DIAMETER_SCALE,
    PROJECT_YELLOW_BALL_DIAMETER_M,
    PROJECT_YELLOW_BALL_FRICTION,
    PROJECT_YELLOW_BALL_GEOM_NAME,
    PROJECT_YELLOW_BALL_RIGHT_OFFSET_M,
    DualCameraSceneConfig,
    ScenePrimitiveObjectConfig,
    TargetSphereConfig,
    build_dual_camera_scene,
    load_base_from_camera_transform,
    load_wrist_camera_mount_pose,
    scene_preset_objects,
    scene_preset_summary,
    scene_object_body_name,
)
from lerobot.projects.vlbiman_sa.sim.mujoco_dual_camera_scene import PROJECT_GRIPPER_STROKE_M
from lerobot.utils.rotation import Rotation


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def test_load_wrist_camera_mount_pose_from_urdf() -> None:
    repo_root = _repo_root()
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    xyz, rpy = load_wrist_camera_mount_pose(urdf_path)

    np.testing.assert_allclose(xyz, np.asarray([0.0471, 0.0, 0.063314], dtype=float), atol=1e-9)
    np.testing.assert_allclose(rpy, np.asarray([0.0, 0.2618, 0.0], dtype=float), atol=1e-9)


def test_build_dual_camera_scene_contains_named_cameras_and_virtual_target() -> None:
    repo_root = _repo_root()
    base_mjcf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    handeye_result_path = repo_root / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"

    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)
    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
    )

    root = ET.fromstring(artifacts.xml_text)
    worldbody = root.find("worldbody")
    assert worldbody is not None

    external_camera = root.find(f".//camera[@name='{DEFAULT_EXTERNAL_CAMERA_NAME}']")
    wrist_camera = root.find(f".//camera[@name='{DEFAULT_WRIST_CAMERA_NAME}']")
    target_body = root.find(f".//body[@name='{DEFAULT_TARGET_BODY_NAME}']")
    target_geom = root.find(f".//geom[@name='virtual_target_geom']")
    table_geom = root.find(f".//geom[@name='{DEFAULT_TABLE_GEOM_NAME}']")
    target_cube_body = root.find(f".//body[@name='{DEFAULT_TARGET_CUBE_BODY_NAME}']")
    target_cube_geom = root.find(f".//geom[@name='{DEFAULT_TARGET_CUBE_GEOM_NAME}']")

    assert external_camera is not None
    assert wrist_camera is not None
    assert target_body is not None
    assert target_geom is not None
    assert table_geom is not None
    assert target_cube_body is not None
    assert target_cube_geom is not None
    assert float(target_geom.attrib["size"]) > 0.0
    assert np.asarray(artifacts.external_camera_transform_base, dtype=float).shape == (4, 4)
    assert np.asarray(artifacts.wrist_camera_transform_parent, dtype=float).shape == (4, 4)


def test_build_dual_camera_scene_encodes_camera_frames_with_expected_axes_and_quat_order() -> None:
    repo_root = _repo_root()
    base_mjcf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    handeye_result_path = repo_root / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"

    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)
    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
    )

    root = ET.fromstring(artifacts.xml_text)
    external_camera = root.find(f".//camera[@name='{DEFAULT_EXTERNAL_CAMERA_NAME}']")
    wrist_camera = root.find(f".//camera[@name='{DEFAULT_WRIST_CAMERA_NAME}']")
    external_body = root.find(".//body[@name='front_camera_body']")
    wrist_body = root.find(".//body[@name='wrist_camera_body']")

    assert external_camera is not None
    assert wrist_camera is not None
    assert external_body is not None
    assert wrist_body is not None

    assert external_camera.attrib["xyaxes"] == "1 0 0 0 -1 0"
    assert wrist_camera.attrib["xyaxes"] == "0 -1 0 0 0 1"

    external_quat_xyzw = Rotation.from_matrix(np.asarray(base_from_camera[:3, :3], dtype=float)).as_quat()
    external_quat_wxyz = np.asarray(
        [external_quat_xyzw[3], external_quat_xyzw[0], external_quat_xyzw[1], external_quat_xyzw[2]],
        dtype=float,
    )
    actual_external_quat = np.fromstring(external_body.attrib["quat"], sep=" ", dtype=float)
    np.testing.assert_allclose(actual_external_quat, external_quat_wxyz, atol=1e-9)


def test_build_dual_camera_scene_accepts_custom_target_config() -> None:
    repo_root = _repo_root()
    base_mjcf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    handeye_result_path = repo_root / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"
    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)

    custom_target = TargetSphereConfig(position_xyz_m=(0.41, -0.07, 0.12), radius_m=0.03, mass_kg=0.08)
    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=None,
    )
    default_root = ET.fromstring(artifacts.xml_text)
    default_target_body = default_root.find(f".//body[@name='{DEFAULT_TARGET_BODY_NAME}']")
    assert default_target_body is not None

    custom_artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=DualCameraSceneConfig(camera_fovy_deg=artifacts.config.camera_fovy_deg, target=custom_target),
    )
    custom_root = ET.fromstring(custom_artifacts.xml_text)
    custom_target_body = custom_root.find(f".//body[@name='{DEFAULT_TARGET_BODY_NAME}']")
    custom_target_geom = custom_root.find(".//geom[@name='virtual_target_geom']")
    assert custom_target_body is not None
    assert custom_target_geom is not None
    assert custom_target_body.attrib["pos"] == "0.41 -0.07 0.12"
    assert custom_target_geom.attrib["size"] == "0.03"


def test_build_dual_camera_scene_can_embed_multiple_physical_objects() -> None:
    repo_root = _repo_root()
    base_mjcf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    handeye_result_path = repo_root / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"
    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)

    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=DualCameraSceneConfig(
            objects=(
                ScenePrimitiveObjectConfig(
                    object_key="pink_cup",
                    shape="cylinder",
                    position_xyz_m=(0.33, 0.18, 0.06),
                    size_xyz_m=(0.035, 0.05, 0.0),
                ),
                ScenePrimitiveObjectConfig(
                    object_key="blue_box",
                    shape="box",
                    position_xyz_m=(0.28, -0.12, 0.05),
                    size_xyz_m=(0.04, 0.03, 0.025),
                ),
            ),
        ),
    )

    root = ET.fromstring(artifacts.xml_text)
    cup_body = root.find(f".//body[@name='{scene_object_body_name('pink_cup')}']")
    box_body = root.find(f".//body[@name='{scene_object_body_name('blue_box')}']")
    assert cup_body is not None
    assert box_body is not None


def test_project_scene_preset_objects_match_required_dimensions() -> None:
    objects = scene_preset_objects(PROJECT_SCENE_PRESET_NAME, center_xy_m=(0.45, 0.0))
    assert len(objects) == 2

    green_plate, yellow_ball = objects
    assert green_plate.body_name == PROJECT_GREEN_PLATE_BODY_NAME
    assert green_plate.geom_name == PROJECT_GREEN_PLATE_GEOM_NAME
    assert green_plate.shape == "container"
    assert green_plate.freejoint is False
    assert np.isclose(2.0 * float(green_plate.size_xyz_m[0]), PROJECT_GREEN_PLATE_DIAMETER_M)
    assert np.isclose(green_plate.container_rim_height_m, PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M)
    assert np.isclose(green_plate.container_rim_thickness_m, PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M)
    assert green_plate.container_rim_segments == PROJECT_GREEN_CONTAINER_RIM_SEGMENTS
    assert np.allclose(green_plate.position_xyz_m[:2], (0.45, 0.0))

    assert yellow_ball.body_name == PROJECT_YELLOW_BALL_BODY_NAME
    assert yellow_ball.geom_name == PROJECT_YELLOW_BALL_GEOM_NAME
    assert yellow_ball.shape == "sphere"
    assert yellow_ball.freejoint is True
    assert np.isclose(2.0 * float(yellow_ball.size_xyz_m[0]), PROJECT_YELLOW_BALL_DIAMETER_M)
    assert np.allclose(yellow_ball.position_xyz_m[:2], (0.45, -PROJECT_YELLOW_BALL_RIGHT_OFFSET_M))
    assert np.isclose(PROJECT_YELLOW_BALL_DIAMETER_M, PROJECT_YELLOW_BALL_BASE_DIAMETER_M * 1.5)
    assert np.isclose(PROJECT_YELLOW_BALL_DIAMETER_SCALE, 1.5)
    assert np.isclose(PROJECT_GREEN_PLATE_DIAMETER_M, PROJECT_GRIPPER_STROKE_M)
    assert np.allclose(yellow_ball.friction, PROJECT_YELLOW_BALL_FRICTION)

    summary = scene_preset_summary(PROJECT_SCENE_PRESET_NAME)
    assert summary["preset_name"] == PROJECT_SCENE_PRESET_NAME
    assert np.isclose(float(summary["yellow_ball_base_diameter_m"]), PROJECT_YELLOW_BALL_BASE_DIAMETER_M)
    assert np.isclose(float(summary["yellow_ball_diameter_scale"]), PROJECT_YELLOW_BALL_DIAMETER_SCALE)
    assert np.isclose(float(summary["yellow_ball_diameter_m"]), PROJECT_YELLOW_BALL_DIAMETER_M)
    assert np.isclose(float(summary["yellow_ball_right_offset_m"]), PROJECT_YELLOW_BALL_RIGHT_OFFSET_M)
    assert np.isclose(float(summary["green_plate_diameter_m"]), PROJECT_GREEN_PLATE_DIAMETER_M)
    assert np.isclose(float(summary["green_container_rim_height_m"]), PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M)
    assert np.isclose(float(summary["green_container_rim_thickness_m"]), PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M)
    assert int(summary["green_container_rim_segments"]) == PROJECT_GREEN_CONTAINER_RIM_SEGMENTS


def test_project_scene_preset_objects_accept_independent_position_overrides() -> None:
    objects = scene_preset_objects(
        PROJECT_SCENE_PRESET_NAME,
        center_xy_m=(0.45, 0.0),
        green_plate_position_xyz_m=(0.36, 0.08, None),
        yellow_ball_position_xyz_m=(0.52, -0.11, 0.09),
    )
    green_plate, yellow_ball = objects

    assert np.allclose(green_plate.position_xyz_m[:2], (0.36, 0.08))
    assert np.isclose(green_plate.position_xyz_m[2], 0.041)
    assert np.allclose(yellow_ball.position_xyz_m, (0.52, -0.11, 0.09))


def test_build_dual_camera_scene_can_embed_project_scene_preset() -> None:
    repo_root = _repo_root()
    base_mjcf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower-home.mjcf"
    )
    urdf_path = (
        repo_root
        / "lerobot_robot_cjjarm"
        / "lerobot_robot_cjjarm"
        / "cjjarm_urdf"
        / "TRLC-DK1-Follower.urdf"
    )
    handeye_result_path = repo_root / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json"
    base_from_camera = load_base_from_camera_transform(handeye_result_path)
    wrist_xyz, wrist_rpy = load_wrist_camera_mount_pose(urdf_path)

    artifacts = build_dual_camera_scene(
        base_mjcf_path=base_mjcf_path,
        base_from_external_camera=base_from_camera,
        wrist_camera_xyz_m=wrist_xyz,
        wrist_camera_rpy_rad=wrist_rpy,
        config=DualCameraSceneConfig(
            include_target_cube=False,
            target=TargetSphereConfig(rgba=(1.0, 1.0, 0.0, 0.0)),
            objects=scene_preset_objects(PROJECT_SCENE_PRESET_NAME, center_xy_m=(0.45, 0.0)),
        ),
    )

    root = ET.fromstring(artifacts.xml_text)
    green_plate_body = root.find(f".//body[@name='{PROJECT_GREEN_PLATE_BODY_NAME}']")
    yellow_ball_body = root.find(f".//body[@name='{PROJECT_YELLOW_BALL_BODY_NAME}']")
    target_cube_body = root.find(f".//body[@name='{DEFAULT_TARGET_CUBE_BODY_NAME}']")
    green_plate_freejoint = root.find(f".//body[@name='{PROJECT_GREEN_PLATE_BODY_NAME}']/freejoint")
    yellow_ball_freejoint = root.find(f".//body[@name='{PROJECT_YELLOW_BALL_BODY_NAME}']/freejoint")
    green_plate_base_geom = root.find(f".//geom[@name='{PROJECT_GREEN_PLATE_GEOM_NAME}']")
    green_plate_rim_geoms = root.findall(f".//body[@name='{PROJECT_GREEN_PLATE_BODY_NAME}']/geom")
    assert green_plate_body is not None
    assert yellow_ball_body is not None
    assert target_cube_body is None
    assert green_plate_freejoint is None
    assert yellow_ball_freejoint is not None
    assert green_plate_base_geom is not None
    assert green_plate_base_geom.attrib["type"] == "cylinder"
    assert len([geom for geom in green_plate_rim_geoms if "_rim_" in geom.attrib.get("name", "")]) == PROJECT_GREEN_CONTAINER_RIM_SEGMENTS


def test_default_scene_preset_summary_is_default_only() -> None:
    assert scene_preset_objects(DEFAULT_SCENE_PRESET_NAME) == ()
    assert scene_preset_summary(DEFAULT_SCENE_PRESET_NAME) == {"preset_name": DEFAULT_SCENE_PRESET_NAME}
