#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import select
import sys
import termios
import time
import tty
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from lerobot.projects.vlbiman_sa.calib.handeye_solver import solve_hand_eye
    from lerobot.projects.vlbiman_sa.geometry.frame_manager import FrameManager
    from lerobot.projects.vlbiman_sa.geometry.transforms import (
        invert_transform,
        make_transform,
        rotation_error_deg,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[5]
    src_root = repo_root / "src"
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from lerobot.projects.vlbiman_sa.calib.handeye_solver import solve_hand_eye
    from lerobot.projects.vlbiman_sa.geometry.frame_manager import FrameManager
    from lerobot.projects.vlbiman_sa.geometry.transforms import (
        invert_transform,
        make_transform,
        rotation_error_deg,
    )

logger = logging.getLogger(__name__)


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    extra_paths = [
        repo_root / "src",
        repo_root,
        repo_root / "lerobot_robot_cjjarm",
        repo_root / "lerobot_camera_gemini335l",
    ]
    for path in extra_paths:
        if path.exists() and str(path) not in sys.path:
            sys.path.insert(0, str(path))
    return repo_root


@dataclass
class BoardPose:
    camera_from_target: np.ndarray
    reprojection_error_px: float
    corners: np.ndarray


@dataclass
class IkPyFallbackState:
    chain: Any
    arm_joint_names: list[str]
    link_indices: list[int]
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray


class OpenCVCameraNode:
    def __init__(self, index_or_path: str | int, width: int, height: int, fps: int):
        self.index_or_path = int(index_or_path) if str(index_or_path).isdigit() else index_or_path
        self.width = width
        self.height = height
        self.fps = fps
        self.cap: cv2.VideoCapture | None = None

    def connect(self) -> None:
        self.cap = cv2.VideoCapture(self.index_or_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open OpenCV camera: {self.index_or_path}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.fps)

    def read_rgb(self) -> np.ndarray:
        if self.cap is None:
            raise RuntimeError("OpenCV camera is not connected.")
        ok, frame_bgr = self.cap.read()
        if not ok:
            raise RuntimeError("Failed to read frame from OpenCV camera.")
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    def disconnect(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None


def _build_gemini_camera(camera_cfg: dict[str, Any]) -> Any:
    from lerobot.cameras.configs import ColorMode
    from lerobot_camera_gemini335l.config_gemini335l import Gemini335LCameraConfig
    from lerobot_camera_gemini335l.gemini335l import Gemini335LCamera

    color_mode = camera_cfg.get("color_mode", "rgb").lower()
    cfg = Gemini335LCameraConfig(
        serial_number_or_name=camera_cfg.get("serial_number_or_name"),
        width=int(camera_cfg["width"]),
        height=int(camera_cfg["height"]),
        fps=int(camera_cfg["fps"]),
        color_mode=ColorMode.RGB if color_mode == "rgb" else ColorMode.BGR,
        use_depth=bool(camera_cfg.get("align_depth_to_color", False)),
        align_depth_to_color=bool(camera_cfg.get("align_depth_to_color", False)),
        align_mode=str(camera_cfg.get("align_mode", "sw")),
        profile_selection_strategy=str(camera_cfg.get("profile_selection_strategy", "closest")),
        depth_work_mode=(
            str(camera_cfg["depth_work_mode"])
            if camera_cfg.get("depth_work_mode") is not None
            else None
        ),
        disp_search_range_mode=(
            int(camera_cfg["disp_search_range_mode"])
            if camera_cfg.get("disp_search_range_mode") is not None
            else None
        ),
        disp_search_offset=(
            int(camera_cfg["disp_search_offset"])
            if camera_cfg.get("disp_search_offset") is not None
            else None
        ),
    )
    return Gemini335LCamera(cfg)


def _build_camera_node(cfg: dict[str, Any]) -> Any:
    backend = str(cfg.get("backend", "gemini335l")).lower()
    if backend == "gemini335l":
        return _build_gemini_camera(cfg)
    if backend == "opencv":
        return OpenCVCameraNode(
            index_or_path=cfg.get("opencv_index_or_path", 0),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=int(cfg["fps"]),
        )
    raise ValueError(f"Unsupported camera backend: {backend}")


def _load_robot_node(cfg: dict[str, Any]) -> Any:
    from lerobot_robot_cjjarm.cjjarm_robot import CjjArm
    from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig

    robot_cfg = CjjArmConfig()
    if cfg.get("serial_port"):
        robot_cfg.serial_port = str(cfg["serial_port"])
    if bool(cfg.get("disable_robot_cameras", True)):
        robot_cfg.cameras = {}
    return CjjArm(robot_cfg)


def _delta_pose_from_key(key: str, t_step: float, r_step: float) -> np.ndarray | None:
    delta_map = {
        "w": np.array([+t_step, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        "s": np.array([-t_step, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=float),
        "a": np.array([0.0, +t_step, 0.0, 0.0, 0.0, 0.0], dtype=float),
        "d": np.array([0.0, -t_step, 0.0, 0.0, 0.0, 0.0], dtype=float),
        "r": np.array([0.0, 0.0, +t_step, 0.0, 0.0, 0.0], dtype=float),
        "f": np.array([0.0, 0.0, -t_step, 0.0, 0.0, 0.0], dtype=float),
        "u": np.array([0.0, 0.0, 0.0, +r_step, 0.0, 0.0], dtype=float),
        "o": np.array([0.0, 0.0, 0.0, -r_step, 0.0, 0.0], dtype=float),
        "i": np.array([0.0, 0.0, 0.0, 0.0, +r_step, 0.0], dtype=float),
        "k": np.array([0.0, 0.0, 0.0, 0.0, -r_step, 0.0], dtype=float),
        "j": np.array([0.0, 0.0, 0.0, 0.0, 0.0, +r_step], dtype=float),
        "l": np.array([0.0, 0.0, 0.0, 0.0, 0.0, -r_step], dtype=float),
    }
    return delta_map.get(key.lower())


def _delta_action_dict(delta_pose: np.ndarray) -> dict[str, float]:
    return {
        "delta_x": float(delta_pose[0]),
        "delta_y": float(delta_pose[1]),
        "delta_z": float(delta_pose[2]),
        "delta_rx": float(delta_pose[3]),
        "delta_ry": float(delta_pose[4]),
        "delta_rz": float(delta_pose[5]),
    }


def _compose_transform_delta(base_from_tool: np.ndarray, delta_pose: np.ndarray) -> np.ndarray:
    out = np.asarray(base_from_tool, dtype=float).copy()
    out[:3, 3] += np.asarray(delta_pose[:3], dtype=float)
    if np.linalg.norm(delta_pose[3:]) < 1e-12:
        return out
    delta_rot, _ = cv2.Rodrigues(np.asarray(delta_pose[3:], dtype=float).reshape(3, 1))
    out[:3, :3] = delta_rot @ out[:3, :3]
    return out


def _build_ikpy_fallback_state(robot: Any) -> IkPyFallbackState:
    from ikpy.chain import Chain

    arm_joint_names = list(robot.config.urdf_joint_map.keys())
    urdf_joint_names = [robot.config.urdf_joint_map[name] for name in arm_joint_names]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        probe_chain = Chain.from_urdf_file(str(robot.config.urdf_path))
    index_by_name = {link.name: idx for idx, link in enumerate(probe_chain.links)}

    missing = [name for name in urdf_joint_names if name not in index_by_name]
    if missing:
        raise RuntimeError(f"IKPy cannot find URDF joints: {missing}")

    link_indices = [index_by_name[name] for name in urdf_joint_names]
    active_mask = [False] * len(probe_chain.links)
    for idx in link_indices:
        active_mask[idx] = True

    chain = Chain.from_urdf_file(str(robot.config.urdf_path), active_links_mask=active_mask)

    lower_bounds: list[float] = []
    upper_bounds: list[float] = []
    for link in chain.links:
        lower, upper = link.bounds
        lower_bounds.append(float(-np.inf if lower is None else lower))
        upper_bounds.append(float(np.inf if upper is None else upper))

    return IkPyFallbackState(
        chain=chain,
        arm_joint_names=arm_joint_names,
        link_indices=link_indices,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        upper_bounds=np.asarray(upper_bounds, dtype=float),
    )


def _full_q_from_observation(obs: dict[str, Any], state: IkPyFallbackState) -> np.ndarray:
    full_q = np.zeros(len(state.chain.links), dtype=float)
    for joint_name, idx in zip(state.arm_joint_names, state.link_indices, strict=True):
        full_q[idx] = float(obs[f"{joint_name}.pos"])
    return full_q


def _select_min_delta_solution(
    candidate: np.ndarray,
    seed: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
) -> np.ndarray:
    normalized = np.asarray(candidate, dtype=float).copy()
    for idx in range(normalized.shape[0]):
        low = float(lower_bounds[idx])
        high = float(upper_bounds[idx])
        v0 = float(normalized[idx])
        best = v0
        best_delta = abs(v0 - float(seed[idx]))
        for k in (-1, 1):
            alt = v0 + k * 2.0 * np.pi
            if alt < low or alt > high:
                continue
            alt_delta = abs(alt - float(seed[idx]))
            if alt_delta < best_delta:
                best = alt
                best_delta = alt_delta
        normalized[idx] = best
    clipped = np.minimum(np.maximum(normalized, lower_bounds), upper_bounds)
    return clipped


def _move_robot_once_with_ikpy(
    robot: Any,
    delta_pose: np.ndarray,
    state: IkPyFallbackState,
) -> None:
    obs = robot.get_observation()
    current_q = _full_q_from_observation(obs, state)
    current_tf = state.chain.forward_kinematics(current_q)
    target_tf = _compose_transform_delta(current_tf, delta_pose)

    raw_solution = state.chain.inverse_kinematics_frame(
        target_tf,
        initial_position=current_q,
        orientation_mode="all",
        regularization_parameter=1e-4,
        optimizer="least_squares",
    )
    raw_solution = np.asarray(raw_solution, dtype=float)
    solution = _select_min_delta_solution(
        raw_solution,
        seed=current_q,
        lower_bounds=state.lower_bounds,
        upper_bounds=state.upper_bounds,
    )

    joint_action = {
        f"{joint_name}.pos": float(solution[idx])
        for joint_name, idx in zip(state.arm_joint_names, state.link_indices, strict=True)
    }
    robot.send_action(joint_action)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Config file must be a mapping: {path}")
    return payload


def _load_intrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    camera_matrix = np.asarray(payload["camera_matrix"], dtype=float)
    dist_coeffs = np.asarray(payload["dist_coeffs"], dtype=float).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3, got {camera_matrix.shape}")
    return camera_matrix, dist_coeffs


def _build_board_object_points(cols: int, rows: int, square_size_m: float) -> np.ndarray:
    grid = np.zeros((rows * cols, 3), dtype=np.float32)
    grid[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * float(square_size_m)
    return grid


def _estimate_board_pose(
    rgb_frame: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    board_cfg: dict[str, Any],
) -> BoardPose | None:
    cols = int(board_cfg["cols"])
    rows = int(board_cfg["rows"])
    square = float(board_cfg["square_size_m"])
    refine = bool(board_cfg.get("refine_corners", True))
    detector = str(board_cfg.get("detector", "sb_fallback")).lower()

    gray = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2GRAY)

    found = False
    corners = None
    pattern = (cols, rows)

    if detector in ("sb", "sb_fallback") and hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = int(
            board_cfg.get(
                "sb_flags",
                int(cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY),
            )
        )
        found, sb_corners = cv2.findChessboardCornersSB(gray, pattern, flags=sb_flags)
        if found:
            corners = sb_corners.reshape(-1, 1, 2).astype(np.float32)

    if (not found) and detector in ("classic", "sb_fallback"):
        classic_flags = int(
            board_cfg.get(
                "classic_flags",
                int(cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE),
            )
        )
        found, corners = cv2.findChessboardCorners(gray, pattern, flags=classic_flags)

    if not found:
        return None

    if refine and detector != "sb":
        term = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), term)

    object_points = _build_board_object_points(cols, rows, square)
    ok, rvec, tvec = cv2.solvePnP(object_points, corners, camera_matrix, dist_coeffs)
    if not ok:
        return None

    rot, _ = cv2.Rodrigues(rvec)
    camera_from_target = make_transform(rot, tvec.reshape(3))

    projected, _ = cv2.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    reproj_error = float(np.mean(np.linalg.norm(projected.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)))

    return BoardPose(
        camera_from_target=camera_from_target,
        reprojection_error_px=reproj_error,
        corners=corners,
    )


def _read_base_from_gripper(robot: Any, ikpy_state: IkPyFallbackState | None = None) -> np.ndarray:
    obs = robot.get_observation()
    if robot.kinematics is not None:
        joint_names = list(robot.config.urdf_joint_map.keys())
        joint_positions = np.array([obs[f"{name}.pos"] for name in joint_names], dtype=float)
        pose_6d = robot.kinematics.compute_fk(joint_positions)
        rot, _ = cv2.Rodrigues(pose_6d[3:].reshape(3, 1))
        return make_transform(rot, pose_6d[:3])

    if ikpy_state is None:
        raise RuntimeError("Robot kinematics unavailable and IKPy fallback is not initialized.")
    full_q = _full_q_from_observation(obs, ikpy_state)
    return np.asarray(ikpy_state.chain.forward_kinematics(full_q), dtype=float)


def _matrix_to_list(matrix: np.ndarray) -> list[list[float]]:
    return np.asarray(matrix, dtype=float).tolist()


def _save_samples(path: Path, base_from_gripper: list[np.ndarray], camera_from_target: list[np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "base_from_gripper": [_matrix_to_list(m) for m in base_from_gripper],
        "camera_from_target": [_matrix_to_list(m) for m in camera_from_target],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _rotation_average(rotations: list[np.ndarray]) -> np.ndarray:
    m = np.zeros((3, 3), dtype=float)
    for rot in rotations:
        m += rot
    u, _, vt = np.linalg.svd(m)
    rot = u @ vt
    if np.linalg.det(rot) < 0:
        u[:, -1] *= -1
        rot = u @ vt
    return rot


def _evaluate_eye_to_hand(
    base_from_camera: np.ndarray,
    base_from_gripper: list[np.ndarray],
    camera_from_target: list[np.ndarray],
) -> dict[str, float]:
    gripper_from_target_list = [
        invert_transform(bfg) @ base_from_camera @ cft
        for bfg, cft in zip(base_from_gripper, camera_from_target, strict=True)
    ]
    mean_t = np.mean([m[:3, 3] for m in gripper_from_target_list], axis=0)
    mean_r = _rotation_average([m[:3, :3] for m in gripper_from_target_list])
    mean_tf = make_transform(mean_r, mean_t)

    trans_errors_mm = [
        float(np.linalg.norm(m[:3, 3] - mean_t) * 1000.0)
        for m in gripper_from_target_list
    ]
    rot_errors_deg = [rotation_error_deg(mean_tf, m) for m in gripper_from_target_list]

    return {
        "translation_mean_mm": float(np.mean(trans_errors_mm)),
        "translation_std_mm": float(np.std(trans_errors_mm)),
        "translation_max_mm": float(np.max(trans_errors_mm)),
        "rotation_mean_deg": float(np.mean(rot_errors_deg)),
        "rotation_std_deg": float(np.std(rot_errors_deg)),
        "rotation_max_deg": float(np.max(rot_errors_deg)),
    }


def _print_controls() -> None:
    print("\nKeyboard control:")
    print("  w/s: +x/-x, a/d: +y/-y, r/f: +z/-z")
    print("  u/o: +rx/-rx, i/k: +ry/-ry, j/l: +rz/-rz")
    print("  [Space]: capture one sample pair")
    print("  [Enter]: finish collection")


@contextmanager
def _raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def _read_key(timeout_s: float) -> str | None:
    ready, _, _ = select.select([sys.stdin], [], [], timeout_s)
    if not ready:
        return None
    key = sys.stdin.read(1)
    return key


def _move_robot_once(
    robot: Any,
    key: str,
    t_step: float,
    r_step: float,
    ikpy_state: IkPyFallbackState | None,
) -> bool:
    delta_pose = _delta_pose_from_key(key, t_step=t_step, r_step=r_step)
    if delta_pose is None:
        return False
    if robot.kinematics is not None:
        robot.send_action(_delta_action_dict(delta_pose))
    else:
        if ikpy_state is None:
            raise RuntimeError("Robot kinematics unavailable and IKPy fallback is not initialized.")
        _move_robot_once_with_ikpy(robot, delta_pose=delta_pose, state=ikpy_state)
    return True


def run(config_path: Path) -> int:
    cfg = _load_yaml(config_path)
    camera = None
    robot = None
    ikpy_state: IkPyFallbackState | None = None
    try:
        logger.info("=== Phase 1: hardware init ===")
        camera = _build_camera_node(cfg["camera"])
        robot = _load_robot_node(cfg["robot"])

        camera.connect()
        rgb = camera.read() if hasattr(camera, "read") else camera.read_rgb()
        logger.info("Camera frame received: shape=%s", tuple(rgb.shape))

        robot.connect()
        if robot.kinematics is None:
            ikpy_state = _build_ikpy_fallback_state(robot)
            logger.info("Robot kinematics backend: IKPy fallback (pinocchio unavailable).")
        else:
            logger.info("Robot kinematics backend: pinocchio.")
        base_from_gripper = _read_base_from_gripper(robot, ikpy_state=ikpy_state)
        logger.info("Robot pose read success. base_from_gripper:\n%s", base_from_gripper)

        logger.info("=== Phase 2: config precheck ===")
        mode = str(cfg.get("mode", "eye_to_hand")).lower()
        if mode != "eye_to_hand":
            raise ValueError(f"Current auto-flow expects eye_to_hand. Got mode='{mode}'.")
        logger.info("Calibration mode: %s", mode)

        intrinsics_path = Path(cfg["intrinsics"]["path"])
        camera_matrix, dist_coeffs = _load_intrinsics(intrinsics_path)
        logger.info("Loaded intrinsics from %s", intrinsics_path)
        logger.info("Sample schema keys: base_from_gripper, camera_from_target")

        logger.info("=== Phase 3: vision smoke test + user confirmation ===")
        pose = _estimate_board_pose(rgb, camera_matrix, dist_coeffs, cfg["board"])
        if pose is None:
            raise RuntimeError("Cannot detect chessboard on current frame. Fix board visibility and retry.")
        logger.info("Single-shot camera_from_target:\n%s", pose.camera_from_target)
        logger.info("Single-shot reprojection_error_px=%.3f", pose.reprojection_error_px)
        logger.info("Current base_from_gripper:\n%s", base_from_gripper)
        confirm = input("Confirm board detection is correct? [y/N]: ").strip().lower()
        if confirm != "y":
            logger.warning("User aborted after vision smoke test.")
            return 2

        logger.info("=== Phase 4: keyboard IK move + trigger sampling ===")
        _print_controls()
        sampling_cfg = cfg["sampling"]
        min_samples = int(sampling_cfg["min_samples"])
        max_samples = int(sampling_cfg["max_samples"])
        loop_hz = float(sampling_cfg["loop_hz"])
        t_step = float(sampling_cfg["translation_step_m"])
        r_step = float(sampling_cfg["rotation_step_rad"])
        samples_base_from_gripper: list[np.ndarray] = []
        samples_camera_from_target: list[np.ndarray] = []
        debug_dir = Path(cfg["output"]["debug_dir"])
        save_debug_images = bool(cfg["output"].get("save_debug_images", False))

        with _raw_terminal_mode():
            while True:
                key = _read_key(timeout_s=1.0 / max(loop_hz, 1e-3))
                if key is None:
                    continue
                if key == "\n":
                    logger.info("Enter pressed. Stop sampling loop.")
                    break
                if key == " ":
                    frame = camera.read() if hasattr(camera, "read") else camera.read_rgb()
                    board_pose = _estimate_board_pose(frame, camera_matrix, dist_coeffs, cfg["board"])
                    if board_pose is None:
                        logger.warning("Capture skipped: chessboard not detected.")
                        continue
                    robot_pose = _read_base_from_gripper(robot, ikpy_state=ikpy_state)
                    samples_base_from_gripper.append(robot_pose)
                    samples_camera_from_target.append(board_pose.camera_from_target)
                    idx = len(samples_base_from_gripper)
                    logger.info(
                        "Captured sample #%d | reprojection_error_px=%.3f",
                        idx,
                        board_pose.reprojection_error_px,
                    )
                    if save_debug_images:
                        debug_dir.mkdir(parents=True, exist_ok=True)
                        preview = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        cv2.drawChessboardCorners(
                            preview,
                            (int(cfg["board"]["cols"]), int(cfg["board"]["rows"])),
                            board_pose.corners,
                            True,
                        )
                        cv2.imwrite(str(debug_dir / f"sample_{idx:03d}.png"), preview)
                    if idx >= max_samples:
                        logger.info("Reached max_samples=%d, ending collection.", max_samples)
                        break
                    continue
                _move_robot_once(
                    robot,
                    key,
                    t_step=t_step,
                    r_step=r_step,
                    ikpy_state=ikpy_state,
                )

        samples_path = Path(cfg["output"]["samples_json"])
        _save_samples(samples_path, samples_base_from_gripper, samples_camera_from_target)
        logger.info("Saved samples to %s", samples_path)

        if len(samples_base_from_gripper) < min_samples:
            raise RuntimeError(
                f"Not enough samples. required={min_samples}, got={len(samples_base_from_gripper)}."
            )

        logger.info("=== Phase 5: solve hand-eye matrix ===")
        result = solve_hand_eye(
            samples_base_from_gripper,
            samples_camera_from_target,
            setup=mode,
            method="tsai",
        )
        base_from_camera = result.transform
        logger.info("Solved base_from_camera:\n%s", base_from_camera)

        logger.info("=== Phase 6: error validation + write config ===")
        metrics = _evaluate_eye_to_hand(
            base_from_camera=base_from_camera,
            base_from_gripper=samples_base_from_gripper,
            camera_from_target=samples_camera_from_target,
        )
        logger.info("Validation metrics: %s", metrics)

        max_t = float(cfg["validation"]["max_translation_mm"])
        max_r = float(cfg["validation"]["max_rotation_deg"])
        passed = metrics["translation_max_mm"] <= max_t and metrics["rotation_max_deg"] <= max_r

        result_path = Path(cfg["output"]["result_json"])
        result_path.parent.mkdir(parents=True, exist_ok=True)
        result_payload = {
            "passed": passed,
            "mode": mode,
            "sample_count": len(samples_base_from_gripper),
            "base_from_camera": _matrix_to_list(base_from_camera),
            "metrics": metrics,
            "thresholds": {"max_translation_mm": max_t, "max_rotation_deg": max_r},
        }
        result_path.write_text(json.dumps(result_payload, indent=2), encoding="utf-8")
        logger.info("Saved solve result to %s", result_path)

        if not passed:
            raise RuntimeError(
                "Calibration did not pass thresholds. "
                f"translation_max_mm={metrics['translation_max_mm']:.3f}, "
                f"rotation_max_deg={metrics['rotation_max_deg']:.3f}."
            )

        transforms_path = Path(cfg["output"]["transforms_yaml"])
        manager = FrameManager.from_yaml(transforms_path)
        manager.set_transform("base", "camera", base_from_camera)
        manager.to_yaml(transforms_path)
        logger.info("Updated transforms YAML: %s", transforms_path)
        return 0
    finally:
        try:
            if robot is not None:
                robot.disconnect()
        except Exception as exc:
            logger.warning("Robot disconnect warning: %s", exc)
        try:
            if camera is not None:
                camera.disconnect()
        except Exception as exc:
            logger.warning("Camera disconnect warning: %s", exc)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Full automatic eye-to-hand calibration flow.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("src/lerobot/projects/vlbiman_sa/configs/handeye_auto.yaml"),
    )
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def main() -> int:
    _bootstrap_paths()
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    config_path = args.config
    if config_path.is_dir():
        config_path = config_path / "handeye_auto.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    return run(config_path)


if __name__ == "__main__":
    raise SystemExit(main())
