from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from lerobot.utils.rotation import Rotation


DEFAULT_EXTERNAL_CAMERA_NAME = "front_camera"
DEFAULT_WRIST_CAMERA_NAME = "wrist_camera"
DEFAULT_TARGET_BODY_NAME = "virtual_target_body"
DEFAULT_TARGET_GEOM_NAME = "virtual_target_geom"
DEFAULT_TARGET_FREEJOINT_NAME = f"{DEFAULT_TARGET_BODY_NAME}_free"
DEFAULT_TABLE_GEOM_NAME = "table_geom"
DEFAULT_TARGET_CUBE_BODY_NAME = "target_cube"
DEFAULT_TARGET_CUBE_GEOM_NAME = "target_cube_geom"
DEFAULT_TARGET_CUBE_FREEJOINT_NAME = f"{DEFAULT_TARGET_CUBE_BODY_NAME}_free"
DEFAULT_SCENE_PRESET_NAME = "default"
PROJECT_SCENE_PRESET_NAME = "green_plate_yellow_ball"
PROJECT_GREEN_PLATE_BODY_NAME = "project_green_plate_body"
PROJECT_GREEN_PLATE_GEOM_NAME = "project_green_plate_geom"
PROJECT_YELLOW_BALL_BODY_NAME = "project_yellow_ball_body"
PROJECT_YELLOW_BALL_GEOM_NAME = "project_yellow_ball_geom"
PROJECT_GRIPPER_STROKE_M = 0.046
PROJECT_YELLOW_BALL_BASE_DIAMETER_M = PROJECT_GRIPPER_STROKE_M * 0.5
PROJECT_YELLOW_BALL_DIAMETER_SCALE = 1.5
PROJECT_YELLOW_BALL_DIAMETER_M = PROJECT_YELLOW_BALL_BASE_DIAMETER_M * PROJECT_YELLOW_BALL_DIAMETER_SCALE
PROJECT_YELLOW_BALL_RADIUS_M = PROJECT_YELLOW_BALL_DIAMETER_M * 0.5
PROJECT_YELLOW_BALL_RIGHT_OFFSET_M = 0.20
PROJECT_YELLOW_BALL_MASS_KG = 0.045
PROJECT_YELLOW_BALL_FRICTION = (3.0, 0.12, 0.01)
PROJECT_YELLOW_BALL_SOLREF = (0.002, 1.0)
PROJECT_YELLOW_BALL_SOLIMP = (0.98, 0.999, 0.0005)
PROJECT_GREEN_PLATE_DIAMETER_M = 0.06
PROJECT_GREEN_PLATE_RADIUS_M = PROJECT_GREEN_PLATE_DIAMETER_M * 0.5
PROJECT_GREEN_PLATE_HALF_HEIGHT_M = 0.001
PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M = 0.02
PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M = 0.004
PROJECT_GREEN_CONTAINER_RIM_SEGMENTS = 32

_EXTERNAL_CAMERA_XYAXES_IN_SENSOR_FRAME = np.asarray(
    [
        [1.0, 0.0, 0.0],  # image right
        [0.0, -1.0, 0.0],  # image up
    ],
    dtype=float,
)

_WRIST_CAMERA_XYAXES_IN_SENSOR_FRAME = np.asarray(
    [
        [0.0, -1.0, 0.0],  # image right
        [0.0, 0.0, 1.0],  # image up
    ],
    dtype=float,
)


@dataclass(frozen=True)
class TargetSphereConfig:
    position_xyz_m: tuple[float, float, float] = (0.45, 0.0, 0.08)
    radius_m: float = 0.022
    mass_kg: float = 0.12
    rgba: tuple[float, float, float, float] = (0.93, 0.36, 0.08, 0.35)
    friction: tuple[float, float, float] = (1.4, 0.03, 0.001)
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.93, 0.98, 0.001)
    body_name: str = DEFAULT_TARGET_BODY_NAME
    geom_name: str = DEFAULT_TARGET_GEOM_NAME


@dataclass(frozen=True)
class TableConfig:
    geom_name: str = DEFAULT_TABLE_GEOM_NAME
    position_xyz_m: tuple[float, float, float] = (0.42, 0.0, 0.02)
    size_xyz_m: tuple[float, float, float] = (0.28, 0.36, 0.02)
    rgba: tuple[float, float, float, float] = (0.56, 0.47, 0.38, 1.0)
    friction: tuple[float, float, float] = (1.4, 0.05, 0.002)


@dataclass(frozen=True)
class ScenePrimitiveObjectConfig:
    object_key: str
    shape: str = "sphere"
    position_xyz_m: tuple[float, float, float] = (0.45, 0.0, 0.08)
    size_xyz_m: tuple[float, float, float] = (0.022, 0.0, 0.0)
    mass_kg: float = 0.12
    rgba: tuple[float, float, float, float] = (0.75, 0.75, 0.75, 1.0)
    friction: tuple[float, float, float] = (1.4, 0.03, 0.001)
    solref: tuple[float, float] = (0.005, 1.0)
    solimp: tuple[float, float, float] = (0.93, 0.98, 0.001)
    body_name: str = ""
    geom_name: str = ""
    freejoint: bool = True
    container_rim_height_m: float = 0.0
    container_rim_thickness_m: float = 0.0
    container_rim_segments: int = 0


@dataclass(frozen=True)
class DualCameraSceneConfig:
    external_camera_name: str = DEFAULT_EXTERNAL_CAMERA_NAME
    external_camera_body_name: str = "front_camera_body"
    wrist_camera_name: str = DEFAULT_WRIST_CAMERA_NAME
    wrist_camera_body_name: str = "wrist_camera_body"
    wrist_camera_parent_body: str = "link6-7"
    table: TableConfig = TableConfig()
    target: TargetSphereConfig = TargetSphereConfig()
    target_cube: ScenePrimitiveObjectConfig = ScenePrimitiveObjectConfig(
        object_key="target_cube",
        shape="box",
        position_xyz_m=(0.45, 0.0, 0.0605),
        size_xyz_m=(0.020, 0.020, 0.020),
        mass_kg=0.08,
        rgba=(0.21, 0.62, 0.92, 1.0),
        friction=(1.6, 0.08, 0.003),
        solref=(0.003, 1.0),
        solimp=(0.96, 0.995, 0.001),
        body_name=DEFAULT_TARGET_CUBE_BODY_NAME,
        geom_name=DEFAULT_TARGET_CUBE_GEOM_NAME,
    )
    include_target_cube: bool = True
    objects: tuple[ScenePrimitiveObjectConfig, ...] = ()
    camera_fovy_deg: float = 58.0
    add_camera_markers: bool = True


@dataclass(frozen=True)
class DualCameraSceneArtifacts:
    xml_text: str
    external_camera_transform_base: np.ndarray
    wrist_camera_transform_parent: np.ndarray
    config: DualCameraSceneConfig


def _format_vec(values: np.ndarray | tuple[float, ...]) -> str:
    vector = np.asarray(values, dtype=float).reshape(-1)
    return " ".join(f"{float(value):.9g}" for value in vector)


def _transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    roll = float(rpy[0])
    pitch = float(rpy[1])
    yaw = float(rpy[2])
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation = np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )
    transform = np.eye(4, dtype=float)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(xyz, dtype=float).reshape(3)
    return transform


def _quat_wxyz_from_transform(transform: np.ndarray) -> np.ndarray:
    quat_xyzw = Rotation.from_matrix(np.asarray(transform, dtype=float)[:3, :3]).as_quat()
    return np.asarray([quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]], dtype=float)


def _load_json_object(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object at {path}, got {type(payload).__name__}.")
    return payload


def load_base_from_camera_transform(handeye_result_path: Path) -> np.ndarray:
    payload = _load_json_object(Path(handeye_result_path))
    transform = np.asarray(payload.get("base_from_camera"), dtype=float)
    if transform.shape != (4, 4):
        raise ValueError(f"base_from_camera in {handeye_result_path} must be a 4x4 transform.")
    return transform


def load_wrist_camera_mount_pose(urdf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    root = ET.fromstring(Path(urdf_path).read_text(encoding="utf-8"))
    for joint in root.findall("joint"):
        if joint.attrib.get("name") != "camera_joint":
            continue
        origin = joint.find("origin")
        if origin is None:
            raise ValueError(f"camera_joint in {urdf_path} is missing <origin>.")
        xyz_text = origin.attrib.get("xyz", "0 0 0")
        rpy_text = origin.attrib.get("rpy", "0 0 0")
        xyz = np.fromstring(xyz_text, sep=" ", dtype=float)
        rpy = np.fromstring(rpy_text, sep=" ", dtype=float)
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(f"camera_joint origin in {urdf_path} must have xyz/rpy with 3 values.")
        return xyz, rpy
    raise ValueError(f"camera_joint not found in {urdf_path}.")


def _append_marker_geom(parent: ET.Element, *, name: str, rgba: tuple[float, float, float, float]) -> None:
    ET.SubElement(
        parent,
        "geom",
        attrib={
            "name": name,
            "type": "box",
            "size": "0.01 0.01 0.006",
            "rgba": _format_vec(rgba),
            "contype": "0",
            "conaffinity": "0",
            "mass": "0.0001",
        },
    )


def _append_camera_sensor_body(
    parent: ET.Element,
    *,
    body_name: str,
    camera_name: str,
    transform: np.ndarray,
    camera_xyaxes_sensor_frame: np.ndarray,
    camera_fovy_deg: float,
    add_marker: bool,
    marker_rgba: tuple[float, float, float, float],
) -> ET.Element:
    sensor_body = ET.SubElement(
        parent,
        "body",
        attrib={
            "name": body_name,
            "pos": _format_vec(np.asarray(transform[:3, 3], dtype=float)),
            "quat": _format_vec(_quat_wxyz_from_transform(transform)),
        },
    )
    ET.SubElement(
        sensor_body,
        "camera",
        attrib={
            "name": camera_name,
            "pos": "0 0 0",
            "xyaxes": _format_vec(np.asarray(camera_xyaxes_sensor_frame, dtype=float).reshape(-1)),
            "fovy": f"{float(camera_fovy_deg):.6g}",
        },
    )
    if add_marker:
        _append_marker_geom(sensor_body, name=f"{body_name}_marker", rgba=marker_rgba)
    return sensor_body


def _append_virtual_target(worldbody: ET.Element, target: TargetSphereConfig) -> None:
    body = ET.SubElement(
        worldbody,
        "body",
        attrib={
            "name": target.body_name,
            "pos": _format_vec(target.position_xyz_m),
        },
    )
    ET.SubElement(body, "freejoint", attrib={"name": f"{target.body_name}_free"})
    ET.SubElement(
        body,
        "geom",
        attrib={
            "name": target.geom_name,
            "type": "sphere",
            "size": f"{float(target.radius_m):.9g}",
            "mass": f"{float(target.mass_kg):.9g}",
            "rgba": _format_vec(target.rgba),
            "friction": _format_vec(target.friction),
            "solref": _format_vec(target.solref),
            "solimp": _format_vec(target.solimp),
            "contype": "0",
            "conaffinity": "0",
        },
    )
    ET.SubElement(
        body,
        "site",
        attrib={
            "name": f"{target.body_name}_center",
            "type": "sphere",
            "size": f"{min(float(target.radius_m) * 0.15, 0.004):.9g}",
            "rgba": "0.1 0.9 0.2 1.0",
        },
    )


def _slugify_object_key(object_key: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(object_key).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    normalized = normalized.strip("_")
    return normalized or "object"


def scene_object_body_name(object_key: str) -> str:
    return f"scene_object_{_slugify_object_key(object_key)}_body"


def scene_object_geom_name(object_key: str) -> str:
    return f"scene_object_{_slugify_object_key(object_key)}_geom"


def scene_object_freejoint_name(object_key: str) -> str:
    return f"{scene_object_body_name(object_key)}_free"


def scene_preset_names() -> tuple[str, ...]:
    return (DEFAULT_SCENE_PRESET_NAME, PROJECT_SCENE_PRESET_NAME)


def _resolve_position_override(
    default_xyz_m: tuple[float, float, float],
    override_xyz_m: tuple[float | None, float | None, float | None] | None,
) -> tuple[float, float, float]:
    if override_xyz_m is None:
        return tuple(float(value) for value in default_xyz_m)
    if len(override_xyz_m) != 3:
        raise ValueError("Position override must contain exactly 3 values.")
    return tuple(
        float(default_value) if override_value is None else float(override_value)
        for default_value, override_value in zip(default_xyz_m, override_xyz_m, strict=True)
    )


def scene_preset_objects(
    preset_name: str,
    *,
    center_xy_m: tuple[float, float] = (0.45, 0.0),
    table_top_z_m: float | None = None,
    green_plate_position_xyz_m: tuple[float | None, float | None, float | None] | None = None,
    yellow_ball_position_xyz_m: tuple[float | None, float | None, float | None] | None = None,
) -> tuple[ScenePrimitiveObjectConfig, ...]:
    normalized = str(preset_name or DEFAULT_SCENE_PRESET_NAME).strip().lower()
    if normalized in {"", DEFAULT_SCENE_PRESET_NAME, "none"}:
        return ()
    if normalized != PROJECT_SCENE_PRESET_NAME:
        raise ValueError(
            f"Unsupported scene preset '{preset_name}'. Expected one of {scene_preset_names()}."
        )

    table_cfg = TableConfig()
    table_top_z = (
        float(table_top_z_m)
        if table_top_z_m is not None
        else float(table_cfg.position_xyz_m[2] + table_cfg.size_xyz_m[2])
    )
    center_x = float(center_xy_m[0])
    center_y = float(center_xy_m[1])
    yellow_ball_y = center_y - PROJECT_YELLOW_BALL_RIGHT_OFFSET_M
    plate_center_z = table_top_z + PROJECT_GREEN_PLATE_HALF_HEIGHT_M
    ball_center_z = table_top_z + 2.0 * PROJECT_GREEN_PLATE_HALF_HEIGHT_M + PROJECT_YELLOW_BALL_RADIUS_M
    green_plate_position_xyz = _resolve_position_override(
        (center_x, center_y, plate_center_z),
        green_plate_position_xyz_m,
    )
    yellow_ball_position_xyz = _resolve_position_override(
        (center_x, yellow_ball_y, ball_center_z),
        yellow_ball_position_xyz_m,
    )

    return (
        ScenePrimitiveObjectConfig(
            object_key="green_plate",
            shape="container",
            position_xyz_m=green_plate_position_xyz,
            size_xyz_m=(PROJECT_GREEN_PLATE_RADIUS_M, PROJECT_GREEN_PLATE_HALF_HEIGHT_M, 0.0),
            mass_kg=0.0,
            rgba=(0.18, 0.72, 0.28, 1.0),
            friction=(1.6, 0.08, 0.003),
            solref=(0.003, 1.0),
            solimp=(0.96, 0.995, 0.001),
            body_name=PROJECT_GREEN_PLATE_BODY_NAME,
            geom_name=PROJECT_GREEN_PLATE_GEOM_NAME,
            freejoint=False,
            container_rim_height_m=PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M,
            container_rim_thickness_m=PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M,
            container_rim_segments=PROJECT_GREEN_CONTAINER_RIM_SEGMENTS,
        ),
        ScenePrimitiveObjectConfig(
            object_key="yellow_ball",
            shape="sphere",
            position_xyz_m=yellow_ball_position_xyz,
            size_xyz_m=(PROJECT_YELLOW_BALL_RADIUS_M, 0.0, 0.0),
            mass_kg=PROJECT_YELLOW_BALL_MASS_KG,
            rgba=(0.98, 0.88, 0.16, 1.0),
            friction=PROJECT_YELLOW_BALL_FRICTION,
            solref=PROJECT_YELLOW_BALL_SOLREF,
            solimp=PROJECT_YELLOW_BALL_SOLIMP,
            body_name=PROJECT_YELLOW_BALL_BODY_NAME,
            geom_name=PROJECT_YELLOW_BALL_GEOM_NAME,
            freejoint=True,
        ),
    )


def scene_preset_summary(preset_name: str) -> dict[str, float | str]:
    normalized = str(preset_name or DEFAULT_SCENE_PRESET_NAME).strip().lower()
    if normalized in {"", DEFAULT_SCENE_PRESET_NAME, "none"}:
        return {"preset_name": DEFAULT_SCENE_PRESET_NAME}
    if normalized != PROJECT_SCENE_PRESET_NAME:
        raise ValueError(
            f"Unsupported scene preset '{preset_name}'. Expected one of {scene_preset_names()}."
        )
    return {
        "preset_name": PROJECT_SCENE_PRESET_NAME,
        "gripper_stroke_m": PROJECT_GRIPPER_STROKE_M,
        "yellow_ball_base_diameter_m": PROJECT_YELLOW_BALL_BASE_DIAMETER_M,
        "yellow_ball_diameter_scale": PROJECT_YELLOW_BALL_DIAMETER_SCALE,
        "yellow_ball_diameter_m": PROJECT_YELLOW_BALL_DIAMETER_M,
        "yellow_ball_radius_m": PROJECT_YELLOW_BALL_RADIUS_M,
        "yellow_ball_right_offset_m": PROJECT_YELLOW_BALL_RIGHT_OFFSET_M,
        "yellow_ball_mass_kg": PROJECT_YELLOW_BALL_MASS_KG,
        "yellow_ball_friction": PROJECT_YELLOW_BALL_FRICTION,
        "green_plate_diameter_m": PROJECT_GREEN_PLATE_DIAMETER_M,
        "green_plate_radius_m": PROJECT_GREEN_PLATE_RADIUS_M,
        "green_plate_half_height_m": PROJECT_GREEN_PLATE_HALF_HEIGHT_M,
        "green_container_rim_height_m": PROJECT_GREEN_CONTAINER_RIM_HEIGHT_M,
        "green_container_rim_thickness_m": PROJECT_GREEN_CONTAINER_RIM_THICKNESS_M,
        "green_container_rim_segments": PROJECT_GREEN_CONTAINER_RIM_SEGMENTS,
    }


def _append_table_geom(worldbody: ET.Element, table: TableConfig) -> None:
    ET.SubElement(
        worldbody,
        "geom",
        attrib={
            "name": table.geom_name,
            "type": "box",
            "pos": _format_vec(table.position_xyz_m),
            "size": _format_vec(table.size_xyz_m),
            "rgba": _format_vec(table.rgba),
            "friction": _format_vec(table.friction),
        },
    )


def _append_primitive_object(worldbody: ET.Element, obj: ScenePrimitiveObjectConfig) -> None:
    shape = str(obj.shape).strip().lower()
    if shape not in {"sphere", "cylinder", "box", "container"}:
        raise ValueError(f"Unsupported MuJoCo primitive object shape: {obj.shape}")
    body_name = str(obj.body_name).strip() or scene_object_body_name(obj.object_key)
    geom_name = str(obj.geom_name).strip() or scene_object_geom_name(obj.object_key)
    size = np.asarray(obj.size_xyz_m, dtype=float).reshape(3)
    body = ET.SubElement(
        worldbody,
        "body",
        attrib={
            "name": body_name,
            "pos": _format_vec(obj.position_xyz_m),
        },
    )
    if bool(obj.freejoint):
        ET.SubElement(body, "freejoint", attrib={"name": f"{body_name}_free"})
    if shape == "container":
        base_radius = float(size[0])
        base_half_height = float(size[1])
        rim_height = float(obj.container_rim_height_m)
        rim_thickness = float(obj.container_rim_thickness_m)
        rim_segments = max(int(obj.container_rim_segments), 8)
        if base_radius <= 0.0 or base_half_height <= 0.0 or rim_height <= 0.0 or rim_thickness <= 0.0:
            raise ValueError("Container objects require positive radius, base height, rim height, and rim thickness.")
        ET.SubElement(
            body,
            "geom",
            attrib={
                "name": geom_name,
                "type": "cylinder",
                "size": _format_vec((base_radius, base_half_height, 0.0)),
                "mass": f"{float(obj.mass_kg):.9g}",
                "rgba": _format_vec(obj.rgba),
                "friction": _format_vec(obj.friction),
                "solref": _format_vec(obj.solref),
                "solimp": _format_vec(obj.solimp),
            },
        )
        rim_radius = base_radius - 0.5 * rim_thickness
        segment_half_length = np.pi * max(rim_radius, rim_thickness) / rim_segments
        rim_center_z = base_half_height + 0.5 * rim_height
        for segment_index in range(rim_segments):
            theta = 2.0 * np.pi * segment_index / rim_segments
            x = rim_radius * np.cos(theta)
            y = rim_radius * np.sin(theta)
            yaw = theta - 0.5 * np.pi
            yaw_quat = (float(np.cos(0.5 * yaw)), 0.0, 0.0, float(np.sin(0.5 * yaw)))
            ET.SubElement(
                body,
                "geom",
                attrib={
                    "name": f"{geom_name}_rim_{segment_index:02d}",
                    "type": "box",
                    "pos": _format_vec((x, y, rim_center_z)),
                    "quat": _format_vec(yaw_quat),
                    "size": _format_vec((segment_half_length, 0.5 * rim_thickness, 0.5 * rim_height)),
                    "mass": "0",
                    "rgba": _format_vec(obj.rgba),
                    "friction": _format_vec(obj.friction),
                    "solref": _format_vec(obj.solref),
                    "solimp": _format_vec(obj.solimp),
                },
            )
    else:
        ET.SubElement(
            body,
            "geom",
            attrib={
                "name": geom_name,
                "type": shape,
                "size": _format_vec(size),
                "mass": f"{float(obj.mass_kg):.9g}",
                "rgba": _format_vec(obj.rgba),
                "friction": _format_vec(obj.friction),
                "solref": _format_vec(obj.solref),
                "solimp": _format_vec(obj.solimp),
            },
        )
    ET.SubElement(
        body,
        "site",
        attrib={
            "name": f"{body_name}_center",
            "type": "sphere",
            "size": f"{min(max(float(np.max(size)), 0.004) * 0.15, 0.008):.9g}",
            "rgba": "0.1 0.9 0.2 1.0",
        },
    )


def build_dual_camera_scene(
    *,
    base_mjcf_path: Path,
    base_from_external_camera: np.ndarray,
    wrist_camera_xyz_m: np.ndarray,
    wrist_camera_rpy_rad: np.ndarray,
    config: DualCameraSceneConfig | None = None,
) -> DualCameraSceneArtifacts:
    scene_config = config or DualCameraSceneConfig()
    resolved_base_mjcf_path = Path(base_mjcf_path).resolve()
    root = ET.fromstring(resolved_base_mjcf_path.read_text(encoding="utf-8"))
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError(f"{base_mjcf_path} does not contain a <worldbody>.")
    wrist_parent = root.find(f".//body[@name='{scene_config.wrist_camera_parent_body}']")
    if wrist_parent is None:
        raise ValueError(f"Body '{scene_config.wrist_camera_parent_body}' not found in {base_mjcf_path}.")

    external_transform = np.asarray(base_from_external_camera, dtype=float)
    if external_transform.shape != (4, 4):
        raise ValueError("base_from_external_camera must be a 4x4 transform.")
    wrist_transform = _transform_from_xyz_rpy(
        np.asarray(wrist_camera_xyz_m, dtype=float).reshape(3),
        np.asarray(wrist_camera_rpy_rad, dtype=float).reshape(3),
    )

    asset_root = resolved_base_mjcf_path.parent
    for element in root.iter():
        file_value = element.attrib.get("file")
        if not file_value:
            continue
        resolved_file = Path(file_value)
        if not resolved_file.is_absolute():
            resolved_file = (asset_root / resolved_file).resolve()
        element.set("file", str(resolved_file))

    _append_camera_sensor_body(
        worldbody,
        body_name=scene_config.external_camera_body_name,
        camera_name=scene_config.external_camera_name,
        transform=external_transform,
        camera_xyaxes_sensor_frame=_EXTERNAL_CAMERA_XYAXES_IN_SENSOR_FRAME,
        camera_fovy_deg=scene_config.camera_fovy_deg,
        add_marker=scene_config.add_camera_markers,
        marker_rgba=(0.12, 0.72, 0.98, 0.85),
    )
    _append_camera_sensor_body(
        wrist_parent,
        body_name=scene_config.wrist_camera_body_name,
        camera_name=scene_config.wrist_camera_name,
        transform=wrist_transform,
        camera_xyaxes_sensor_frame=_WRIST_CAMERA_XYAXES_IN_SENSOR_FRAME,
        camera_fovy_deg=scene_config.camera_fovy_deg,
        add_marker=scene_config.add_camera_markers,
        marker_rgba=(0.98, 0.76, 0.14, 0.85),
    )
    _append_table_geom(worldbody, scene_config.table)
    _append_virtual_target(worldbody, scene_config.target)
    if bool(scene_config.include_target_cube):
        _append_primitive_object(worldbody, scene_config.target_cube)
    for obj in scene_config.objects:
        _append_primitive_object(worldbody, obj)

    xml_text = ET.tostring(root, encoding="unicode")
    return DualCameraSceneArtifacts(
        xml_text=xml_text,
        external_camera_transform_base=external_transform,
        wrist_camera_transform_parent=wrist_transform,
        config=scene_config,
    )
