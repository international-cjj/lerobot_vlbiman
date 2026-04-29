from __future__ import annotations

import math
from typing import Any

import numpy as np


def _as_rotation_matrix(value: np.ndarray) -> np.ndarray:
    matrix = np.asarray(value, dtype=float).reshape(3, 3)
    return matrix.copy()


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    matrix = _as_rotation_matrix(rotation)
    sy = math.sqrt(float(matrix[0, 0] * matrix[0, 0] + matrix[1, 0] * matrix[1, 0]))
    singular = sy < 1e-9
    if not singular:
        roll = math.atan2(float(matrix[2, 1]), float(matrix[2, 2]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = math.atan2(float(matrix[1, 0]), float(matrix[0, 0]))
    else:
        roll = math.atan2(float(-matrix[1, 2]), float(matrix[1, 1]))
        pitch = math.atan2(float(-matrix[2, 0]), sy)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=float)


def rpy_to_rotation_matrix(rpy: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    roll, pitch, yaw = [float(value) for value in np.asarray(rpy, dtype=float).reshape(3)]
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.asarray(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ],
        dtype=float,
    )


def rotation_matrix_to_quat_wxyz(rotation: np.ndarray) -> np.ndarray:
    matrix = _as_rotation_matrix(rotation)
    trace = float(np.trace(matrix))
    quat = np.zeros(4, dtype=float)
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quat[0] = 0.25 * scale
        quat[1] = (matrix[2, 1] - matrix[1, 2]) / scale
        quat[2] = (matrix[0, 2] - matrix[2, 0]) / scale
        quat[3] = (matrix[1, 0] - matrix[0, 1]) / scale
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat[0] = (matrix[2, 1] - matrix[1, 2]) / scale
            quat[1] = 0.25 * scale
            quat[2] = (matrix[0, 1] + matrix[1, 0]) / scale
            quat[3] = (matrix[0, 2] + matrix[2, 0]) / scale
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat[0] = (matrix[0, 2] - matrix[2, 0]) / scale
            quat[1] = (matrix[0, 1] + matrix[1, 0]) / scale
            quat[2] = 0.25 * scale
            quat[3] = (matrix[1, 2] + matrix[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat[0] = (matrix[1, 0] - matrix[0, 1]) / scale
            quat[1] = (matrix[0, 2] + matrix[2, 0]) / scale
            quat[2] = (matrix[1, 2] + matrix[2, 1]) / scale
            quat[3] = 0.25 * scale
    norm = float(np.linalg.norm(quat))
    return quat / norm if norm > 0.0 else np.asarray([1.0, 0.0, 0.0, 0.0], dtype=float)


def rotation_matrix_to_rotvec(rotation: np.ndarray) -> np.ndarray:
    matrix = _as_rotation_matrix(rotation)
    cos_theta = float((np.trace(matrix) - 1.0) * 0.5)
    theta = math.acos(float(np.clip(cos_theta, -1.0, 1.0)))
    if theta < 1e-9:
        return 0.5 * np.asarray(
            [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
            dtype=float,
        )
    denom = 2.0 * math.sin(theta)
    axis = np.asarray(
        [matrix[2, 1] - matrix[1, 2], matrix[0, 2] - matrix[2, 0], matrix[1, 0] - matrix[0, 1]],
        dtype=float,
    ) / denom
    return axis * theta


def rotvec_to_rotation_matrix(rotvec: np.ndarray | list[float] | tuple[float, float, float]) -> np.ndarray:
    vector = np.asarray(rotvec, dtype=float).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta < 1e-12:
        skew = np.asarray(
            [[0.0, -vector[2], vector[1]], [vector[2], 0.0, -vector[0]], [-vector[1], vector[0], 0.0]],
            dtype=float,
        )
        return np.eye(3, dtype=float) + skew
    axis = vector / theta
    x, y, z = [float(value) for value in axis]
    skew = np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
    return np.eye(3, dtype=float) + math.sin(theta) * skew + (1.0 - math.cos(theta)) * (skew @ skew)


def pose_from_body(model: Any, data: Any, body_name: str) -> dict[str, np.ndarray]:
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(body_name)))
    if body_id < 0:
        raise ValueError(f"MuJoCo body not found: {body_name!r}")
    rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
    return {
        "name": str(body_name),
        "position": np.asarray(data.xpos[body_id], dtype=float).copy(),
        "rotation_matrix": rotation,
        "quat": rotation_matrix_to_quat_wxyz(rotation),
        "rpy": rotation_matrix_to_rpy(rotation),
    }


def pose_from_camera(model: Any, data: Any, camera_name: str) -> dict[str, np.ndarray]:
    import mujoco

    camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, str(camera_name)))
    if camera_id < 0:
        raise ValueError(f"MuJoCo camera not found: {camera_name!r}")
    rotation = np.asarray(data.cam_xmat[camera_id], dtype=float).reshape(3, 3).copy()
    return {
        "name": str(camera_name),
        "position": np.asarray(data.cam_xpos[camera_id], dtype=float).copy(),
        "rotation_matrix": rotation,
        "quat": rotation_matrix_to_quat_wxyz(rotation),
        "rpy": rotation_matrix_to_rpy(rotation),
    }


def pose_to_plain(pose: dict[str, Any]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pose.items():
        if isinstance(value, np.ndarray):
            payload[key] = value.astype(float).tolist()
        elif key == "rotation_matrix":
            payload[key] = np.asarray(value, dtype=float).reshape(3, 3).tolist()
        else:
            payload[key] = value
    return payload


def delta_pose_to_vector(delta_pose: dict[str, Any]) -> np.ndarray:
    translation = np.asarray(delta_pose.get("translation", delta_pose.get("delta_xyz_m", (0.0, 0.0, 0.0))), dtype=float).reshape(3)
    rotation = np.asarray(delta_pose.get("rotation_rpy", delta_pose.get("delta_rpy_rad", (0.0, 0.0, 0.0))), dtype=float).reshape(3)
    return np.concatenate([translation, rotation]).astype(float)


def compose_target_pose(
    current_ee_pose: dict[str, Any],
    delta_pose: dict[str, Any],
    *,
    translation_frame: str,
    camera_pose: dict[str, Any] | None = None,
) -> dict[str, Any]:
    translation_frame = str(translation_frame or "camera").lower()
    delta = delta_pose_to_vector(delta_pose)
    current_position = np.asarray(current_ee_pose["position"], dtype=float).reshape(3)
    current_rotation = np.asarray(current_ee_pose["rotation_matrix"], dtype=float).reshape(3, 3)

    if translation_frame in {"world", "base"}:
        world_translation = delta[:3]
        frame_debug = {"translation_frame": "world"}
    elif translation_frame in {"ee", "tool", "end_effector"}:
        world_translation = current_rotation @ delta[:3]
        frame_debug = {"translation_frame": "ee"}
    elif translation_frame in {"camera", "camera_image", "image_camera"}:
        if camera_pose is None:
            raise ValueError("camera_pose is required when translation_frame='camera'.")
        camera_rotation = np.asarray(camera_pose["rotation_matrix"], dtype=float).reshape(3, 3)
        # Controller deltas are visual-servo correction commands. Empirically for the
        # wrist camera, reducing pixel error requires the opposite lateral image
        # motion, while positive z still means moving along the optical axis.
        # MuJoCo camera axes follow the OpenGL convention: +x right, +y up, -z forward.
        mujoco_camera_delta = np.asarray([-delta[0], delta[1], -delta[2]], dtype=float)
        world_translation = camera_rotation @ mujoco_camera_delta
        frame_debug = {
            "translation_frame": "camera",
            "camera_name": camera_pose.get("name"),
            "camera_convention": "controller_correction=[-right,+up,forward], mujoco=[right,up,back]",
            "mujoco_camera_delta": mujoco_camera_delta.tolist(),
        }
    else:
        raise ValueError(f"Unsupported delta translation frame: {translation_frame!r}")

    world_rotvec = delta[3:]
    if translation_frame in {"camera", "camera_image", "image_camera"} and camera_pose is not None:
        camera_rotation = np.asarray(camera_pose["rotation_matrix"], dtype=float).reshape(3, 3)
        world_rotvec = camera_rotation @ np.asarray([-delta[3], delta[4], -delta[5]], dtype=float)
    elif translation_frame in {"ee", "tool", "end_effector"}:
        world_rotvec = current_rotation @ delta[3:]
    target_rotation = rotvec_to_rotation_matrix(world_rotvec) @ current_rotation

    target = {
        "name": current_ee_pose.get("name", "target_ee"),
        "position": current_position + world_translation,
        "rotation_matrix": target_rotation,
        "quat": rotation_matrix_to_quat_wxyz(target_rotation),
        "rpy": rotation_matrix_to_rpy(target_rotation),
        "debug": {
            **frame_debug,
            "input_delta_pose": {
                "translation": delta[:3].tolist(),
                "rotation_rpy": delta[3:].tolist(),
            },
            "world_translation": world_translation.tolist(),
            "world_rotation_rpy": world_rotvec.tolist(),
        },
    }
    return target


def solve_ik(
    *,
    model: Any,
    data: Any,
    target_pose: dict[str, Any],
    ee_body_name: str,
    q_init: np.ndarray,
    arm_qpos_addr: list[int],
    arm_dof_addr: list[int],
    joint_limits: np.ndarray,
    joint_directions: np.ndarray,
    max_iters: int = 80,
    pos_tol: float = 0.002,
    rot_tol: float = 0.08,
    damping: float = 1e-4,
    step_scale: float = 0.65,
    max_step_rad: float = 0.08,
    position_weight: float = 1.0,
    rotation_weight: float = 0.05,
) -> dict[str, Any]:
    import mujoco

    body_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, str(ee_body_name)))
    if body_id < 0:
        return {"ok": False, "failure_reason": "ee_body_not_found", "ee_body_name": str(ee_body_name)}

    qpos_snapshot = np.asarray(data.qpos, dtype=float).copy()
    qvel_snapshot = np.asarray(data.qvel, dtype=float).copy()
    ctrl_snapshot = np.asarray(data.ctrl, dtype=float).copy()

    q = np.asarray(q_init, dtype=float).reshape(len(arm_qpos_addr)).copy()
    limits = np.asarray(joint_limits, dtype=float).reshape(len(arm_qpos_addr), 2)
    directions = np.asarray(joint_directions, dtype=float).reshape(len(arm_qpos_addr))
    target_position = np.asarray(target_pose["position"], dtype=float).reshape(3)
    target_rotation = np.asarray(target_pose["rotation_matrix"], dtype=float).reshape(3, 3)
    jacp = np.zeros((3, int(model.nv)), dtype=float)
    jacr = np.zeros((3, int(model.nv)), dtype=float)

    best_q = q.copy()
    best_pos_err = float("inf")
    best_rot_err = float("inf")
    failure_reason = "ik_max_iters_exceeded"
    iterations = 0

    try:
        for iteration in range(max(int(max_iters), 1)):
            iterations = iteration + 1
            data.qpos[np.asarray(arm_qpos_addr, dtype=int)] = q * directions
            data.qvel[np.asarray(arm_dof_addr, dtype=int)] = 0.0
            mujoco.mj_forward(model, data)

            current_position = np.asarray(data.xpos[body_id], dtype=float).copy()
            current_rotation = np.asarray(data.xmat[body_id], dtype=float).reshape(3, 3).copy()
            pos_err = target_position - current_position
            rot_err = rotation_matrix_to_rotvec(target_rotation @ current_rotation.T)
            pos_norm = float(np.linalg.norm(pos_err))
            rot_norm = float(np.linalg.norm(rot_err))
            if pos_norm + rot_norm < best_pos_err + best_rot_err:
                best_q = q.copy()
                best_pos_err = pos_norm
                best_rot_err = rot_norm
            if pos_norm <= float(pos_tol) and rot_norm <= float(rot_tol):
                failure_reason = None
                break

            mujoco.mj_jacBody(model, data, jacp, jacr, body_id)
            columns = np.asarray(arm_dof_addr, dtype=int)
            j_pos = jacp[:, columns] * directions.reshape(1, -1)
            j_rot = jacr[:, columns] * directions.reshape(1, -1)
            j = np.vstack([j_pos * float(position_weight), j_rot * float(rotation_weight)])
            err = np.concatenate([pos_err * float(position_weight), rot_err * float(rotation_weight)])
            lhs = j @ j.T + float(damping) * np.eye(6, dtype=float)
            dq = j.T @ np.linalg.solve(lhs, err)
            dq = np.asarray(dq, dtype=float) * float(step_scale)
            max_step = abs(float(max_step_rad))
            if max_step > 0.0:
                dq = np.clip(dq, -max_step, max_step)
            q = np.clip(q + dq, limits[:, 0], limits[:, 1])

        ok = failure_reason is None
        result_q = q.copy() if ok else best_q.copy()
        data.qpos[np.asarray(arm_qpos_addr, dtype=int)] = result_q * directions
        data.qvel[np.asarray(arm_dof_addr, dtype=int)] = 0.0
        mujoco.mj_forward(model, data)
        achieved_pose = pose_from_body(model, data, ee_body_name)
        achieved_pos_err = target_position - np.asarray(achieved_pose["position"], dtype=float)
        achieved_rot_err = rotation_matrix_to_rotvec(target_rotation @ np.asarray(achieved_pose["rotation_matrix"], dtype=float).reshape(3, 3).T)
        pos_norm = float(np.linalg.norm(achieved_pos_err))
        rot_norm = float(np.linalg.norm(achieved_rot_err))
        if not ok and pos_norm <= float(pos_tol) and rot_norm <= float(rot_tol):
            ok = True
            failure_reason = None
        return {
            "ok": bool(ok),
            "failure_reason": failure_reason,
            "target_q": result_q.astype(float).tolist(),
            "iterations": int(iterations),
            "position_error": achieved_pos_err.astype(float).tolist(),
            "rotation_error": achieved_rot_err.astype(float).tolist(),
            "position_error_norm": pos_norm,
            "rotation_error_norm": rot_norm,
            "achieved_pose": pose_to_plain(achieved_pose),
            "limits_applied": True,
            "method": "mujoco_body_jacobian_damped_least_squares",
        }
    except Exception as exc:
        return {
            "ok": False,
            "failure_reason": f"ik_exception:{type(exc).__name__}",
            "error": str(exc),
        }
    finally:
        data.qpos[:] = qpos_snapshot
        data.qvel[:] = qvel_snapshot
        data.ctrl[:] = ctrl_snapshot
        mujoco.mj_forward(model, data)
