from __future__ import annotations

import json
import sys
import traceback
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
PLUGIN_ROOTS = [
    REPO_ROOT / "lerobot_robot_cjjarm",
    REPO_ROOT / "lerobot_teleoperator_zhonglin",
]
for path in [REPO_ROOT, REPO_ROOT / "src", *PLUGIN_ROOTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


try:
    import mujoco
except Exception:  # pragma: no cover - handled in output
    mujoco = None


@dataclass
class Inventory:
    urdf: list[str]
    xacro: list[str]
    xml: list[str]
    mjcf: list[str]
    mesh: list[str]
    sim_python: list[str]
    entrypoints: list[str]


def _rel(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _parse_float(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return float(text)
    except Exception:
        return None


def _parse_float_list(text: str | None) -> list[float] | None:
    if text is None:
        return None
    values = np.fromstring(text, sep=" ", dtype=float)
    if values.size == 0:
        return None
    return [float(v) for v in values.tolist()]


def _resolve_asset(base_dir: Path, filename: str | None) -> tuple[str | None, bool | None]:
    if not filename:
        return None, None
    candidate = Path(filename)
    if not candidate.is_absolute():
        candidate = (base_dir / candidate).resolve()
    return str(candidate), candidate.exists()


def inventory_files(root: Path) -> Inventory:
    urdf = sorted(_rel(path) for path in root.rglob("*.urdf"))
    xacro = sorted(_rel(path) for path in root.rglob("*.xacro"))
    xml = sorted(_rel(path) for path in root.rglob("*.xml"))
    mjcf = sorted(_rel(path) for path in root.rglob("*.mjcf"))
    mesh_exts = {".stl", ".dae", ".obj"}
    mesh = sorted(_rel(path) for path in root.rglob("*") if path.suffix.lower() in mesh_exts)

    sim_keywords = ("mujoco", "gymnasium", "lerobot", "env", "simulation", "sim", "robot")
    sim_python: list[str] = []
    for path in root.rglob("*.py"):
        rel = _rel(path)
        lowered = rel.lower()
        if any(keyword in lowered for keyword in sim_keywords):
            sim_python.append(rel)
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="ignore").lower()
        except Exception:
            continue
        if any(keyword in text for keyword in sim_keywords):
            sim_python.append(rel)
    sim_python = sorted(set(sim_python))

    entrypoint_keywords = (
        "main.py",
        "env.py",
        "train.py",
        "test.py",
        "simulate.py",
        "mujoco_env.py",
    )
    entrypoints = sorted(
        _rel(path)
        for path in root.rglob("*.py")
        if any(token in path.name.lower() for token in entrypoint_keywords)
        or path.name.startswith("run_")
        or "teleop" in path.name.lower()
    )

    return Inventory(
        urdf=urdf,
        xacro=xacro,
        xml=xml,
        mjcf=mjcf,
        mesh=mesh,
        sim_python=sim_python,
        entrypoints=entrypoints,
    )


def _geometry_dict(geom: ET.Element | None, *, base_dir: Path) -> dict[str, Any] | None:
    if geom is None:
        return None
    for tag in ("mesh", "box", "cylinder", "sphere", "capsule"):
        node = geom.find(tag)
        if node is None:
            continue
        result: dict[str, Any] = {"type": tag}
        if tag == "mesh":
            filename = node.attrib.get("filename")
            resolved, exists = _resolve_asset(base_dir, filename)
            result["filename"] = filename
            result["resolved_path"] = resolved
            result["path_exists"] = exists
            result["scale"] = _parse_float_list(node.attrib.get("scale"))
        elif tag == "box":
            result["size"] = _parse_float_list(node.attrib.get("size"))
        elif tag == "cylinder":
            result["radius"] = _parse_float(node.attrib.get("radius"))
            result["length"] = _parse_float(node.attrib.get("length"))
        elif tag == "sphere":
            result["radius"] = _parse_float(node.attrib.get("radius"))
        elif tag == "capsule":
            result["radius"] = _parse_float(node.attrib.get("radius"))
            result["length"] = _parse_float(node.attrib.get("length"))
        return result
    return {"type": "unknown"}


def parse_urdf(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    base_dir = path.parent

    links: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    all_link_names: set[str] = set()
    child_links: set[str] = set()

    for link in root.findall("link"):
        name = link.attrib["name"]
        all_link_names.add(name)
        visuals = []
        for visual in link.findall("visual"):
            visuals.append(_geometry_dict(visual.find("geometry"), base_dir=base_dir))
        collisions = []
        for collision in link.findall("collision"):
            collisions.append(_geometry_dict(collision.find("geometry"), base_dir=base_dir))

        inertial = link.find("inertial")
        mass_value = None
        inertia_dict = None
        if inertial is not None:
            mass_node = inertial.find("mass")
            inertia_node = inertial.find("inertia")
            mass_value = _parse_float(mass_node.attrib.get("value") if mass_node is not None else None)
            if inertia_node is not None:
                inertia_dict = {
                    key: _parse_float(inertia_node.attrib.get(key))
                    for key in ("ixx", "ixy", "ixz", "iyy", "iyz", "izz")
                }

        links.append(
            {
                "name": name,
                "has_visual": bool(visuals),
                "has_collision": bool(collisions),
                "has_inertial": inertial is not None,
                "visuals": visuals,
                "collisions": collisions,
                "mass": mass_value,
                "inertia": inertia_dict,
                "collision_types": sorted({item["type"] for item in collisions if item}),
                "visual_mesh_paths_exist": all(
                    item.get("path_exists", True) for item in visuals if item and item.get("type") == "mesh"
                ),
                "collision_mesh_paths_exist": all(
                    item.get("path_exists", True) for item in collisions if item and item.get("type") == "mesh"
                ),
                "stl_mesh_without_scale": any(
                    item
                    and item.get("type") == "mesh"
                    and str(item.get("filename", "")).lower().endswith(".stl")
                    and item.get("scale") is None
                    for item in visuals + collisions
                ),
            }
        )

    for joint in root.findall("joint"):
        parent = joint.find("parent")
        child = joint.find("child")
        child_link = child.attrib.get("link") if child is not None else None
        if child_link:
            child_links.add(child_link)
        limit = joint.find("limit")
        dynamics = joint.find("dynamics")
        mimic = joint.find("mimic")
        joints.append(
            {
                "name": joint.attrib.get("name"),
                "type": joint.attrib.get("type"),
                "parent_link": parent.attrib.get("link") if parent is not None else None,
                "child_link": child_link,
                "axis": _parse_float_list(joint.find("axis").attrib.get("xyz"))
                if joint.find("axis") is not None
                else None,
                "limit": {
                    "lower": _parse_float(limit.attrib.get("lower")) if limit is not None else None,
                    "upper": _parse_float(limit.attrib.get("upper")) if limit is not None else None,
                    "effort": _parse_float(limit.attrib.get("effort")) if limit is not None else None,
                    "velocity": _parse_float(limit.attrib.get("velocity")) if limit is not None else None,
                },
                "dynamics": {
                    "damping": _parse_float(dynamics.attrib.get("damping")) if dynamics is not None else None,
                    "friction": _parse_float(dynamics.attrib.get("friction")) if dynamics is not None else None,
                },
                "mimic": mimic.attrib if mimic is not None else None,
                "is_gripper_joint": "gripper" in (joint.attrib.get("name", "").lower())
                or "finger" in (child_link or "").lower(),
            }
        )

    root_links = sorted(all_link_names - child_links)
    return {
        "path": _rel(path),
        "root_links": root_links,
        "links": links,
        "joints": joints,
        "summary": {
            "link_count": len(links),
            "joint_count": len(joints),
            "links_missing_collision": [link["name"] for link in links if not link["has_collision"]],
            "links_missing_inertial": [link["name"] for link in links if not link["has_inertial"]],
            "links_with_mesh_collision": [
                link["name"] for link in links if any(item and item.get("type") == "mesh" for item in link["collisions"])
            ],
            "links_with_primitive_collision": [
                link["name"]
                for link in links
                if any(item and item.get("type") in {"box", "cylinder", "sphere", "capsule"} for item in link["collisions"])
            ],
            "root_link_has_inertial": any(
                link["name"] in root_links and link["has_inertial"] for link in links
            ),
            "gripper_joints": [joint["name"] for joint in joints if joint["is_gripper_joint"]],
        },
    }


def parse_mjcf(path: Path) -> dict[str, Any]:
    root = ET.fromstring(path.read_text(encoding="utf-8"))
    base_dir = path.parent
    worldbody = root.find("worldbody")
    option = root.find("option")
    default_geom = root.find("./default/geom")
    asset_meshes = []
    asset_by_name: dict[str, dict[str, Any]] = {}
    for mesh in root.findall("./asset/mesh"):
        resolved, exists = _resolve_asset(base_dir, mesh.attrib.get("file"))
        item = {
            "name": mesh.attrib.get("name"),
            "file": mesh.attrib.get("file"),
            "resolved_path": resolved,
            "path_exists": exists,
            "scale": _parse_float_list(mesh.attrib.get("scale")),
        }
        asset_meshes.append(item)
        if item["name"]:
            asset_by_name[item["name"]] = item

    bodies: list[dict[str, Any]] = []
    joints: list[dict[str, Any]] = []
    geoms: list[dict[str, Any]] = []
    cameras: list[dict[str, Any]] = []
    lights: list[dict[str, Any]] = []

    def walk_body(body: ET.Element, parent: str | None, depth: int) -> None:
        body_name = body.attrib.get("name") or f"<unnamed_body_{len(bodies)}>"
        bodies.append(
            {
                "name": body_name,
                "parent": parent,
                "depth": depth,
                "pos": _parse_float_list(body.attrib.get("pos")),
                "quat": _parse_float_list(body.attrib.get("quat")),
            }
        )
        for joint in body.findall("joint"):
            joints.append(
                {
                    "name": joint.attrib.get("name"),
                    "type": joint.attrib.get("type", "hinge"),
                    "body": body_name,
                    "axis": _parse_float_list(joint.attrib.get("axis")),
                    "range": _parse_float_list(joint.attrib.get("range")),
                    "actuatorfrcrange": _parse_float_list(joint.attrib.get("actuatorfrcrange")),
                }
            )
        for freejoint in body.findall("freejoint"):
            joints.append(
                {
                    "name": freejoint.attrib.get("name"),
                    "type": "free",
                    "body": body_name,
                    "axis": None,
                    "range": None,
                    "actuatorfrcrange": None,
                }
            )
        for geom in body.findall("geom"):
            mesh_name = geom.attrib.get("mesh")
            geoms.append(
                {
                    "name": geom.attrib.get("name"),
                    "body": body_name,
                    "type": geom.attrib.get("type", "sphere" if mesh_name is None else "mesh"),
                    "size": _parse_float_list(geom.attrib.get("size")),
                    "mesh": mesh_name,
                    "mesh_file_exists": asset_by_name.get(mesh_name, {}).get("path_exists") if mesh_name else None,
                    "rgba": _parse_float_list(geom.attrib.get("rgba")),
                    "mass": _parse_float(geom.attrib.get("mass")),
                    "density": _parse_float(geom.attrib.get("density")),
                    "contype": geom.attrib.get("contype"),
                    "conaffinity": geom.attrib.get("conaffinity"),
                    "friction": _parse_float_list(geom.attrib.get("friction")),
                }
            )
        for camera in body.findall("camera"):
            cameras.append(
                {
                    "name": camera.attrib.get("name"),
                    "body": body_name,
                    "pos": _parse_float_list(camera.attrib.get("pos")),
                    "quat": _parse_float_list(camera.attrib.get("quat")),
                    "xyaxes": _parse_float_list(camera.attrib.get("xyaxes")),
                    "fovy": _parse_float(camera.attrib.get("fovy")),
                }
            )
        for light in body.findall("light"):
            lights.append(
                {
                    "name": light.attrib.get("name"),
                    "body": body_name,
                    "pos": _parse_float_list(light.attrib.get("pos")),
                    "dir": _parse_float_list(light.attrib.get("dir")),
                }
            )
        for child in body.findall("body"):
            walk_body(child, body_name, depth + 1)

    if worldbody is not None:
        for geom in worldbody.findall("geom"):
            mesh_name = geom.attrib.get("mesh")
            geoms.append(
                {
                    "name": geom.attrib.get("name"),
                    "body": "<worldbody>",
                    "type": geom.attrib.get("type", "sphere" if mesh_name is None else "mesh"),
                    "size": _parse_float_list(geom.attrib.get("size")),
                    "mesh": mesh_name,
                    "mesh_file_exists": asset_by_name.get(mesh_name, {}).get("path_exists") if mesh_name else None,
                    "rgba": _parse_float_list(geom.attrib.get("rgba")),
                    "mass": _parse_float(geom.attrib.get("mass")),
                    "density": _parse_float(geom.attrib.get("density")),
                    "contype": geom.attrib.get("contype"),
                    "conaffinity": geom.attrib.get("conaffinity"),
                    "friction": _parse_float_list(geom.attrib.get("friction")),
                }
            )
        for camera in worldbody.findall("camera"):
            cameras.append(
                {
                    "name": camera.attrib.get("name"),
                    "body": "<worldbody>",
                    "pos": _parse_float_list(camera.attrib.get("pos")),
                    "quat": _parse_float_list(camera.attrib.get("quat")),
                    "xyaxes": _parse_float_list(camera.attrib.get("xyaxes")),
                    "fovy": _parse_float(camera.attrib.get("fovy")),
                }
            )
        for light in worldbody.findall("light"):
            lights.append(
                {
                    "name": light.attrib.get("name"),
                    "body": "<worldbody>",
                    "pos": _parse_float_list(light.attrib.get("pos")),
                    "dir": _parse_float_list(light.attrib.get("dir")),
                }
            )
        for body in worldbody.findall("body"):
            walk_body(body, "<worldbody>", 0)

    actuators = []
    actuator_root = root.find("actuator")
    if actuator_root is not None:
        for actuator in list(actuator_root):
            actuators.append(
                {
                    "type": actuator.tag,
                    "name": actuator.attrib.get("name"),
                    "joint": actuator.attrib.get("joint"),
                    "tendon": actuator.attrib.get("tendon"),
                    "gear": _parse_float_list(actuator.attrib.get("gear")),
                    "ctrlrange": _parse_float_list(actuator.attrib.get("ctrlrange")),
                    "forcerange": _parse_float_list(actuator.attrib.get("forcerange")),
                }
            )

    return {
        "path": _rel(path),
        "option": option.attrib if option is not None else {},
        "default_geom": default_geom.attrib if default_geom is not None else {},
        "worldbody_bodies": bodies,
        "joints": joints,
        "geoms": geoms,
        "asset_meshes": asset_meshes,
        "actuators": actuators,
        "cameras": cameras,
        "lights": lights,
        "summary": {
            "body_count": len(bodies),
            "joint_count": len(joints),
            "geom_count": len(geoms),
            "actuator_count": len(actuators),
            "camera_count": len(cameras),
            "light_count": len(lights),
            "collision_disabled_geoms": [
                geom["name"]
                for geom in geoms
                if geom.get("contype") == "0" and geom.get("conaffinity") == "0"
            ],
            "mesh_geoms": [geom["name"] for geom in geoms if geom.get("type") == "mesh"],
            "primitive_geoms": [
                geom["name"]
                for geom in geoms
                if geom.get("type") in {"box", "cylinder", "sphere", "capsule", "ellipsoid"}
            ],
            "mesh_assets_missing": [mesh["name"] for mesh in asset_meshes if mesh["path_exists"] is False],
            "freejoints": [joint["name"] for joint in joints if joint["type"] == "free"],
            "interaction_object_names": [
                name
                for name in [body["name"] for body in bodies] + [geom.get("name") for geom in geoms if geom.get("name")]
                if any(token in (name or "").lower() for token in ("table", "object", "cube", "box", "target", "cup", "floor"))
            ],
        },
    }


def _model_names(model: Any, mjt_obj: Any, count: int) -> list[str]:
    result = []
    for index in range(int(count)):
        name = mujoco.mj_id2name(model, mjt_obj, index)
        result.append(name or f"<unnamed_{index}>")
    return result


def smoke_load_model(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": _rel(path), "status": "unavailable"}
    if mujoco is None:
        result["error"] = "mujoco import failed"
        return result
    try:
        model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)
        initial_body_xpos = np.asarray(data.xpos, dtype=float).copy() if model.nbody > 0 else np.zeros((0, 3))
        tracked_joint_qpos_initial = {}
        for joint_name in ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6", "gripper_left", "gripper_right"):
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            if joint_id >= 0:
                tracked_joint_qpos_initial[joint_name] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        max_ncon = int(data.ncon)
        for _ in range(100):
            mujoco.mj_step(model, data)
            max_ncon = max(max_ncon, int(data.ncon))
        final_body_xpos = np.asarray(data.xpos, dtype=float).copy() if model.nbody > 0 else np.zeros((0, 3))
        tracked_joint_qpos_final = {}
        for joint_name in tracked_joint_qpos_initial:
            joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name))
            tracked_joint_qpos_final[joint_name] = float(data.qpos[int(model.jnt_qposadr[joint_id])])
        body_displacement = (
            float(np.max(np.linalg.norm(final_body_xpos - initial_body_xpos, axis=1)))
            if final_body_xpos.size and initial_body_xpos.size
            else 0.0
        )
        result.update(
            {
                "status": "ok",
                "nbody": int(model.nbody),
                "njnt": int(model.njnt),
                "ngeom": int(model.ngeom),
                "nu": int(model.nu),
                "ncam": int(model.ncam),
                "body_names": _model_names(model, mujoco.mjtObj.mjOBJ_BODY, model.nbody),
                "joint_names": _model_names(model, mujoco.mjtObj.mjOBJ_JOINT, model.njnt),
                "geom_names": _model_names(model, mujoco.mjtObj.mjOBJ_GEOM, model.ngeom),
                "actuator_names": _model_names(model, mujoco.mjtObj.mjOBJ_ACTUATOR, model.nu),
                "camera_names": _model_names(model, mujoco.mjtObj.mjOBJ_CAMERA, model.ncam),
                "qpos_finite": bool(np.isfinite(data.qpos).all()),
                "qvel_finite": bool(np.isfinite(data.qvel).all()),
                "qpos_abs_max": float(np.max(np.abs(data.qpos))) if data.qpos.size else 0.0,
                "qvel_abs_max": float(np.max(np.abs(data.qvel))) if data.qvel.size else 0.0,
                "max_ncon_during_100_steps": max_ncon,
                "max_body_displacement_m": body_displacement,
                "tracked_joint_qpos_initial": tracked_joint_qpos_initial,
                "tracked_joint_qpos_final": tracked_joint_qpos_final,
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return result


def _rewrite_all_file_paths(xml_root: ET.Element, *, base_dir: Path) -> None:
    for element in xml_root.iter():
        file_attr = element.attrib.get("file")
        if not file_attr:
            continue
        resolved = Path(file_attr)
        if not resolved.is_absolute():
            resolved = (base_dir / resolved).resolve()
        element.set("file", str(resolved))


def cube_contact_test(base_mjcf_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"path": _rel(base_mjcf_path), "status": "unavailable"}
    if mujoco is None:
        result["error"] = "mujoco import failed"
        return result
    try:
        baseline_model = mujoco.MjModel.from_xml_path(str(base_mjcf_path.resolve()))
        baseline_data = mujoco.MjData(baseline_model)
        mujoco.mj_forward(baseline_model, baseline_data)

        target_xy = np.array([0.0, 0.0], dtype=float)
        target_z = 0.35
        body_names = _model_names(baseline_model, mujoco.mjtObj.mjOBJ_BODY, baseline_model.nbody)
        candidate_names = [name for name in body_names if name in {"finger_left", "finger_right", "link6-7"}]
        if "finger_left" in candidate_names and "finger_right" in candidate_names:
            left_id = int(mujoco.mj_name2id(baseline_model, mujoco.mjtObj.mjOBJ_BODY, "finger_left"))
            right_id = int(mujoco.mj_name2id(baseline_model, mujoco.mjtObj.mjOBJ_BODY, "finger_right"))
            target_xy = 0.5 * (
                np.asarray(baseline_data.xpos[left_id], dtype=float)[:2]
                + np.asarray(baseline_data.xpos[right_id], dtype=float)[:2]
            )
            target_z = float(
                max(
                    float(baseline_data.xpos[left_id][2]),
                    float(baseline_data.xpos[right_id][2]),
                )
                + 0.12
            )
        elif "link6-7" in candidate_names:
            link_id = int(mujoco.mj_name2id(baseline_model, mujoco.mjtObj.mjOBJ_BODY, "link6-7"))
            target_xy = np.asarray(baseline_data.xpos[link_id], dtype=float)[:2]
            target_z = float(baseline_data.xpos[link_id][2] + 0.15)

        root = ET.fromstring(base_mjcf_path.read_text(encoding="utf-8"))
        _rewrite_all_file_paths(root, base_dir=base_mjcf_path.parent)
        worldbody = root.find("worldbody")
        if worldbody is None:
            raise ValueError(f"{base_mjcf_path} does not contain worldbody")
        cube_body = ET.SubElement(
            worldbody,
            "body",
            attrib={
                "name": "audit_cube_body",
                "pos": f"{target_xy[0]:.9g} {target_xy[1]:.9g} {target_z:.9g}",
            },
        )
        ET.SubElement(cube_body, "freejoint", attrib={"name": "audit_cube_free"})
        ET.SubElement(
            cube_body,
            "geom",
            attrib={
                "name": "audit_cube_geom",
                "type": "box",
                "size": "0.025 0.025 0.025",
                "mass": "0.08",
                "rgba": "0.2 0.8 1 1",
                "friction": "1.2 0.03 0.001",
            },
        )
        model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        cube_geom_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "audit_cube_geom"))
        qpos_addr = None
        joint_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "audit_cube_free"))
        if joint_id >= 0:
            qpos_addr = int(model.jnt_qposadr[joint_id])

        def describe_geom(geom_id: int) -> str:
            geom_name = model.geom(geom_id).name
            if geom_name:
                return geom_name
            body_name = model.body(int(model.geom_bodyid[geom_id])).name or f"<body_{int(model.geom_bodyid[geom_id])}>"
            return f"{body_name}:geom_{geom_id}"

        contact_pairs: set[tuple[str, str]] = set()
        cube_contact_steps = 0
        max_ncon = 0
        for _ in range(200):
            mujoco.mj_step(model, data)
            max_ncon = max(max_ncon, int(data.ncon))
            cube_touched = False
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                if int(contact.geom1) == cube_geom_id:
                    other = int(contact.geom2)
                elif int(contact.geom2) == cube_geom_id:
                    other = int(contact.geom1)
                else:
                    continue
                cube_touched = True
                name1 = describe_geom(cube_geom_id)
                name2 = describe_geom(other)
                pair = tuple(sorted((name1, name2)))
                contact_pairs.add(pair)
            if cube_touched:
                cube_contact_steps += 1

        cube_qpos = (
            np.asarray(data.qpos[qpos_addr : qpos_addr + 7], dtype=float).tolist() if qpos_addr is not None else None
        )
        final_cube_center_z = float(data.qpos[qpos_addr + 2]) if qpos_addr is not None else None
        result.update(
            {
                "status": "ok",
                "max_ncon": max_ncon,
                "cube_contact_steps": cube_contact_steps,
                "unique_contact_pairs": [list(pair) for pair in sorted(contact_pairs)],
                "final_cube_qpos": cube_qpos,
                "final_cube_center_z_m": final_cube_center_z,
                "cube_passed_below_floor_top": bool(final_cube_center_z is not None and final_cube_center_z < -0.005),
                "robot_contact_pairs": [
                    list(pair)
                    for pair in sorted(contact_pairs)
                    if all("floor" not in item for item in pair)
                ],
            }
        )
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return result


def cjjarm_sim_runtime_probe() -> dict[str, Any]:
    result: dict[str, Any] = {"status": "unavailable"}
    if mujoco is None:
        result["error"] = "mujoco import failed"
        return result
    try:
        from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig

        default_cfg = CjjArmSimConfig(use_viewer=False, scene_settle_steps=0, render_width=64, render_height=64)
        legacy_cfg = CjjArmSimConfig(
            use_viewer=False,
            scene_profile="legacy",
            scene_settle_steps=0,
            render_width=64,
            render_height=64,
        )
        default_sim = CjjArmSim(default_cfg)
        legacy_sim = CjjArmSim(legacy_cfg)

        default_sim.connect()
        legacy_sim.connect()
        try:
            left_joint_id = int(mujoco.mj_name2id(default_sim.model, mujoco.mjtObj.mjOBJ_JOINT, "gripper_left"))
            left_before = float(default_sim.data.qpos[default_sim.model.jnt_qposadr[left_joint_id]])
            default_sim._set_virtual_target_position(np.asarray([0.20, 0.0, 0.159971], dtype=float), settle_steps=0)
            contact_counts = []
            for _ in range(8):
                default_sim.send_action({"gripper.pos": -1.0})
                contact_counts.append(int(default_sim.data.ncon))
            left_after = float(default_sim.data.qpos[default_sim.model.jnt_qposadr[left_joint_id]])

            result.update(
                {
                    "status": "ok",
                    "default_scene_profile": default_cfg.scene_profile,
                    "default_model": {
                        "nu": int(default_sim.model.nu),
                        "ncam": int(default_sim.model.ncam),
                        "ngeom": int(default_sim.model.ngeom),
                        "nbody": int(default_sim.model.nbody),
                        "has_virtual_target_freejoint": int(
                            mujoco.mj_name2id(default_sim.model, mujoco.mjtObj.mjOBJ_JOINT, "virtual_target_body_free")
                        )
                        >= 0,
                        "camera_names": _model_names(
                            default_sim.model, mujoco.mjtObj.mjOBJ_CAMERA, default_sim.model.ncam
                        ),
                    },
                    "legacy_model": {
                        "scene_profile": legacy_cfg.scene_profile,
                        "nu": int(legacy_sim.model.nu),
                        "ncam": int(legacy_sim.model.ncam),
                        "ngeom": int(legacy_sim.model.ngeom),
                        "nbody": int(legacy_sim.model.nbody),
                        "camera_names": _model_names(
                            legacy_sim.model, mujoco.mjtObj.mjOBJ_CAMERA, legacy_sim.model.ncam
                        ),
                    },
                    "gripper_qpos_change_without_actuator": {
                        "left_before": left_before,
                        "left_after": left_after,
                        "changed": not np.isclose(left_before, left_after),
                        "contact_counts_during_close": contact_counts,
                    },
                    "has_bus_attribute_default": hasattr(default_sim, "bus"),
                    "has_bus_attribute_legacy": hasattr(legacy_sim, "bus"),
                }
            )
        finally:
            default_sim.disconnect()
            legacy_sim.disconnect()
        return result
    except Exception as exc:
        result.update(
            {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
        )
        return result


def robot_env_probe() -> dict[str, Any]:
    result: dict[str, Any] = {
        "source_summary": {
            "env_class": "RobotEnv",
            "reset_returns": ["observation", "info"],
            "step_returns": ["observation", "reward", "terminated", "truncated", "info"],
            "observation_source_keys": ["agent_pos", "pixels", "<joint>.pos"],
            "action_space_shape_declared": [3],
            "action_space_shape_with_gripper": [4],
        }
    }
    try:
        from lerobot.envs.configs import HILSerlRobotEnvConfig
        from lerobot.rl.gym_manipulator import RobotEnv
        from lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig

        sim = CjjArmSim(CjjArmSimConfig(use_viewer=False, scene_settle_steps=0, render_width=64, render_height=64))
        try:
            env = RobotEnv(robot=sim, use_gripper=True, display_cameras=False, reset_pose=None, reset_time_s=0.0)
            env.close()
            result["instantiation"] = {"status": "ok"}
        except Exception as exc:
            result["instantiation"] = {
                "status": "error",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }
            try:
                sim.disconnect()
            except Exception:
                pass

        result["env_config_class"] = asdict(HILSerlRobotEnvConfig())
        return result
    except Exception as exc:
        return {
            "status": "error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }


def determine_load_chain() -> dict[str, Any]:
    return {
        "cjjarm_sim_config_defaults": {
            "urdf_path": _rel(
                REPO_ROOT
                / "lerobot_robot_cjjarm"
                / "lerobot_robot_cjjarm"
                / "cjjarm_urdf"
                / "TRLC-DK1-Follower.urdf"
            ),
            "scene_base_mjcf_path": _rel(
                REPO_ROOT
                / "lerobot_robot_cjjarm"
                / "lerobot_robot_cjjarm"
                / "cjjarm_urdf"
                / "TRLC-DK1-Follower-home.mjcf"
            ),
            "scene_profile_default": "dual_camera_target",
        },
        "actual_runtime_chain": [
            "scripts/teleop_cjjarm_keyboard.py -> CjjArmSimConfig -> CjjArmSim",
            "scripts/teleop_cjjarm_zhonglin.py -> CjjArmSimConfig -> CjjArmSim",
            "run_visual_closed_loop_validation.py -> build_dual_camera_scene(...) -> MjModel.from_xml_string(...)",
            "play_mujoco_trajectory.py -> build_dual_camera_scene(...) or raw MJCF in legacy mode",
            "run_frrg_mujoco_validate.py -> CjjArmSim(default config)",
        ],
        "key_observation": (
            "default cjjarm_sim runtime loads a generated MJCF scene built from the base home MJCF plus camera "
            "and target bodies; URDF is still used for kinematics and wrist camera mount extraction; legacy cjjarm_sim "
            "falls back to loading the URDF directly."
        ),
    }


def main() -> int:
    inventory = inventory_files(REPO_ROOT)
    source_urdfs = [
        REPO_ROOT / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "TRLC-DK1-Follower.urdf",
        REPO_ROOT / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "TRLC-DK1-Follower-home.urdf",
    ]
    source_mjcfs = [
        REPO_ROOT / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "TRLC-DK1-Follower.mjcf",
        REPO_ROOT / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "TRLC-DK1-Follower-home.mjcf",
    ]

    payload = {
        "inventory": asdict(inventory),
        "load_chain": determine_load_chain(),
        "urdf": {str(_rel(path)): parse_urdf(path) for path in source_urdfs if path.exists()},
        "mjcf": {str(_rel(path)): parse_mjcf(path) for path in source_mjcfs if path.exists()},
        "mujoco_smoke": {
            str(_rel(path)): smoke_load_model(path) for path in [*source_urdfs, *source_mjcfs] if path.exists()
        },
        "cube_contact_test": cube_contact_test(source_mjcfs[-1]),
        "cjjarm_sim_runtime": cjjarm_sim_runtime_probe(),
        "robot_env_probe": robot_env_probe(),
    }

    output_path = REPO_ROOT / "tools" / "audit" / "simulation_audit_data.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
