#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import multiprocessing as mp
import os
import queue
import select
import shutil
import subprocess
import sys
import threading
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    import termios
    import tty
except ImportError:  # pragma: no cover - Windows fallback.
    termios = None
    tty = None


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


def _bootstrap_paths() -> Path:
    repo_root = Path(__file__).resolve().parents[5]
    desired_paths = [
        repo_root / "src",
        repo_root / "lerobot_robot_cjjarm",
        repo_root / "lerobot_teleoperator_zhonglin",
        repo_root,
    ]
    for path in reversed(desired_paths):
        if not path.exists():
            continue
        path_str = str(path)
        while path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)
    return repo_root


_maybe_reexec_in_repo_venv()
REPO_ROOT = _bootstrap_paths()

from lerobot.projects.vlbiman_sa.app.run_pose_adaptation import build_pose_pipeline_config, run_pose_adaptation_pipeline
from lerobot.projects.vlbiman_sa.app.run_trajectory_generation import (
    TrajectoryPipelineConfig,
    run_trajectory_generation_pipeline,
)
from lerobot.projects.vlbiman_sa.core.contracts import TaskGraspConfig
from lerobot.projects.vlbiman_sa.demo.io import (
    append_frame_metadata,
    create_session_dir,
    save_frame_assets,
    save_named_camera_assets,
    write_manifest,
)
from lerobot.projects.vlbiman_sa.demo.rgbd_recorder import _extract_ee_pose, _extract_gripper_state, _extract_joint_positions, _extract_numeric_observation
from lerobot.projects.vlbiman_sa.demo.schema import FrameRecord, RecorderConfig, RecordingSummary
from lerobot.projects.vlbiman_sa.sim import (
    PROJECT_GREEN_PLATE_BODY_NAME,
    PROJECT_SCENE_PRESET_NAME,
    PROJECT_YELLOW_BALL_BODY_NAME,
    PROJECT_YELLOW_BALL_RADIUS_M,
    PROJECT_YELLOW_BALL_RIGHT_OFFSET_M,
    scene_preset_names,
    scene_preset_summary,
)
from lerobot.projects.vlbiman_sa.vision import CameraIntrinsics
from lerobot.projects.vlbiman_sa.skills import InvarianceClassifierConfig, SegmenterConfig, build_skill_bank
from lerobot_robot_cjjarm.cjjarm_sim import CjjArmSim
from lerobot_robot_cjjarm.config_cjjarm_sim import CjjArmSimConfig


def _default_task_config_path() -> Path:
    return REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "configs" / "task_grasp.yaml"


def _default_output_root() -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "original_flow"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the original non-FRRG VLBiMan flow with selectable execution backend.")
    parser.add_argument("--source", choices=("sim", "sim-session", "real-session"), default="sim")
    parser.add_argument("--task-config", type=Path, default=_default_task_config_path())
    parser.add_argument("--session-dir", type=Path, default=None, help="Required for --source sim-session or real-session.")
    parser.add_argument("--live-result-path", type=Path, default=None, help="Required for sim-session/real-session T5 unless config already points to one.")
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--target-phrase", type=str, default="yellow ball")
    parser.add_argument("--aux-target-phrase", action="append", default=[])
    parser.add_argument("--scene-preset", choices=scene_preset_names(), default=PROJECT_SCENE_PRESET_NAME)
    parser.add_argument(
        "--scene-run-id",
        type=str,
        default=None,
        help="Bind sim recording/execution to outputs/vlbiman_sa/mujoco_dual_camera_scene/<run-id>/summary.json.",
    )
    parser.add_argument("--scene-summary-path", type=Path, default=None, help="Bind sim recording/execution to a scene summary.json.")
    parser.add_argument("--scene-path", type=Path, default=None, help="Bind sim recording/execution directly to a scene MJCF.")
    parser.add_argument("--sim-target-x", type=float, default=0.45)
    parser.add_argument("--sim-target-y", type=float, default=0.0)
    parser.add_argument("--sim-target-z", type=float, default=0.08)
    parser.add_argument("--sim-fps", type=int, default=10)
    parser.add_argument("--sim-frames", type=int, default=300)
    parser.add_argument("--render-width", type=int, default=640)
    parser.add_argument("--render-height", type=int, default=480)
    parser.add_argument(
        "--sim-teleop",
        choices=("plan", "zhongling"),
        default="plan",
        help="Action source for --source sim recording. 'plan' replays the built-in demo; 'zhongling' reads the Zhongling controller.",
    )
    parser.add_argument("--teleop-port", type=str, default=None, help="Zhongling controller serial port for --sim-teleop zhongling.")
    parser.add_argument("--no-teleop-calibrate", action="store_true", help="Skip Zhongling zero-position calibration on connect.")
    parser.add_argument(
        "--record-viewer",
        action="store_true",
        help="Open a MuJoCo viewer during --source sim recording. Keys: 1 front, 2 wrist, 3 free, Space/Enter/Esc record controls.",
    )
    parser.add_argument(
        "--manual-record-control",
        action="store_true",
        help="For --source sim recording: Space starts/stops each attempt, Space re-records, Enter saves.",
    )
    parser.add_argument(
        "--stop-after",
        choices=("record", "skill_build", "trajectory", "execute"),
        default="execute",
        help="Stop after a pipeline boundary. 'record' is valid for --source sim.",
    )
    parser.add_argument("--skip-skill-build", action="store_true", help="Reuse an existing analysis/t3_skill_bank/skill_bank.json.")
    parser.add_argument("--skip-real-t4", action="store_true", help="For real-session, reuse existing T4 outputs instead of running VLM.")
    parser.add_argument("--execute-backend", choices=("none", "sim", "robot"), default="none")
    parser.add_argument("--open-viewer", action="store_true", help="For sim execution, open MuJoCo viewer instead of diagnostics only.")
    parser.add_argument("--display", type=str, default=None)
    parser.add_argument("--robot-serial-port", type=str, default=None)
    parser.add_argument("--dry-run-robot", action="store_true")
    parser.add_argument("--log-level", default="INFO")
    return parser.parse_args()


def _timestamp_run_id(prefix: str) -> str:
    return f"{prefix}_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _load_task_config(path: Path) -> TaskGraspConfig:
    import yaml

    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Task config must be a mapping: {path}")
    for key in (
        "data_root",
        "transforms_path",
        "handeye_result_path",
        "recording_session_dir",
        "skill_output_dir",
        "skill_bank_path",
        "vision_output_dir",
        "pose_output_dir",
        "trajectory_output_dir",
        "live_result_path",
        "intrinsics_path",
    ):
        if key in payload and payload[key] is not None:
            payload[key] = Path(payload[key])
    return TaskGraspConfig(**payload)


def _normalize_object_key(value: str) -> str:
    normalized = "".join(char.lower() if char.isalnum() else "_" for char in str(value).strip())
    while "__" in normalized:
        normalized = normalized.replace("__", "_")
    return normalized.strip("_") or "object"


def _object_key_for_phrase(phrase: str) -> str:
    key = _normalize_object_key(phrase)
    if key in {"orange", "yellow", "yellow_ball", "ball"}:
        return "yellow_ball"
    if key in {"plate", "green_plate"}:
        return "green_plate"
    if key in {"pink_cup", "cup"}:
        return "pink_cup"
    return key


def _scene_summary_path_for_run_id(run_id: str) -> Path:
    return REPO_ROOT / "outputs" / "vlbiman_sa" / "mujoco_dual_camera_scene" / str(run_id) / "summary.json"


def _resolve_maybe_relative_path(path: Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else (REPO_ROOT / path).resolve()


def _parse_xyz_attr(value: str | None) -> np.ndarray | None:
    if value is None:
        return None
    vector = np.fromstring(value, sep=" ", dtype=float)
    if vector.shape != (3,):
        return None
    return vector


def _object_positions_from_scene_xml(scene_path: Path) -> dict[str, np.ndarray]:
    root = ET.fromstring(Path(scene_path).read_text(encoding="utf-8"))
    body_names = {
        "yellow_ball": PROJECT_YELLOW_BALL_BODY_NAME,
        "orange": PROJECT_YELLOW_BALL_BODY_NAME,
        "green_plate": PROJECT_GREEN_PLATE_BODY_NAME,
    }
    positions: dict[str, np.ndarray] = {}
    for object_key, body_name in body_names.items():
        body = root.find(f".//body[@name='{body_name}']")
        if body is None:
            continue
        xyz = _parse_xyz_attr(body.attrib.get("pos"))
        if xyz is not None:
            positions[object_key] = xyz
    return positions


def _resolve_bound_scene(args: argparse.Namespace) -> dict[str, Any]:
    summary_path = args.scene_summary_path
    if summary_path is None and args.scene_run_id:
        summary_path = _scene_summary_path_for_run_id(args.scene_run_id)
    scene_path = args.scene_path
    summary_payload: dict[str, Any] = {}
    object_positions: dict[str, np.ndarray] = {}

    if summary_path is not None:
        summary_path = _resolve_maybe_relative_path(Path(summary_path))
        summary_payload = _load_json(summary_path)
        if scene_path is None and summary_payload.get("scene_path"):
            scene_path = Path(str(summary_payload["scene_path"]))
        for object_key, payload in dict(summary_payload.get("objects", {})).items():
            if not isinstance(payload, dict):
                continue
            xyz = payload.get("initial_position_xyz_m")
            if xyz is None and isinstance(payload.get("body_pose_world"), dict):
                xyz = payload["body_pose_world"].get("position_xyz_m")
            if xyz is None:
                continue
            vector = np.asarray(xyz, dtype=float).reshape(3)
            normalized_key = _object_key_for_phrase(str(object_key))
            object_positions[normalized_key] = vector
            if normalized_key == "yellow_ball":
                object_positions["orange"] = vector

    if scene_path is not None:
        scene_path = _resolve_maybe_relative_path(Path(scene_path))
        if not scene_path.exists():
            raise FileNotFoundError(f"Bound scene file not found: {scene_path}")
        xml_positions = _object_positions_from_scene_xml(scene_path)
        object_positions.update(xml_positions)

    if summary_path is not None and not summary_path.exists():
        raise FileNotFoundError(f"Bound scene summary not found: {summary_path}")

    return {
        "summary_path": summary_path,
        "scene_path": scene_path,
        "object_positions": object_positions,
        "summary": summary_payload,
    }


def _sim_object_positions(args: argparse.Namespace) -> dict[str, np.ndarray]:
    x = float(args.sim_target_x)
    y = float(args.sim_target_y)
    yellow_ball_y = y - PROJECT_YELLOW_BALL_RIGHT_OFFSET_M
    table_top_z = 0.04
    yellow_ball_z = table_top_z + 0.008 + PROJECT_YELLOW_BALL_RADIUS_M
    positions = {
        "yellow_ball": np.asarray([x, yellow_ball_y, yellow_ball_z], dtype=float),
        "orange": np.asarray([x, yellow_ball_y, yellow_ball_z], dtype=float),
        "green_plate": np.asarray([x, y, table_top_z + 0.004], dtype=float),
        "pink_cup": np.asarray([x, y + 0.12, table_top_z + 0.05], dtype=float),
    }
    bound_scene = getattr(args, "_bound_scene", {}) or {}
    for object_key, position in dict(bound_scene.get("object_positions", {})).items():
        positions[str(object_key)] = np.asarray(position, dtype=float).reshape(3)
    return positions


def _joint_plan(frame_count: int) -> list[tuple[np.ndarray, float]]:
    home = np.asarray([0.0, -0.52, 0.78, 0.0, 0.58, 0.0], dtype=float)
    approach = np.asarray([0.18, -0.68, 0.92, -0.12, 0.74, 0.18], dtype=float)
    grasp = np.asarray([0.24, -0.82, 1.08, -0.18, 0.86, 0.22], dtype=float)
    lift = np.asarray([0.18, -0.60, 0.78, -0.08, 0.68, 0.18], dtype=float)
    place = np.asarray([-0.12, -0.66, 0.88, 0.16, 0.80, -0.16], dtype=float)
    waypoints = [
        (home, 0.001),
        (approach, 0.001),
        (grasp, 0.001),
        (grasp, -0.040),
        (lift, -0.040),
        (place, -0.040),
        (home, -0.040),
    ]
    out: list[tuple[np.ndarray, float]] = []
    for index in range(max(int(frame_count), 2)):
        u = index / max(frame_count - 1, 1)
        scaled = u * (len(waypoints) - 1)
        left = min(int(np.floor(scaled)), len(waypoints) - 2)
        alpha = float(scaled - left)
        q0, g0 = waypoints[left]
        q1, g1 = waypoints[left + 1]
        out.append(((1.0 - alpha) * q0 + alpha * q1, (1.0 - alpha) * g0 + alpha * g1))
    return out


def _joint_action(joint_positions: np.ndarray, gripper_pos: float) -> dict[str, float]:
    return {
        "joint_1.pos": float(joint_positions[0]),
        "joint_2.pos": float(joint_positions[1]),
        "joint_3.pos": float(joint_positions[2]),
        "joint_4.pos": float(joint_positions[3]),
        "joint_5.pos": float(joint_positions[4]),
        "joint_6.pos": float(joint_positions[5]),
        "gripper.pos": float(gripper_pos),
    }


def _build_zhongling_teleop(args: argparse.Namespace) -> Any:
    from lerobot_teleoperator_zhonglin.config_zhonglin import ZhonglinTeleopConfig
    from lerobot_teleoperator_zhonglin.zhonglin_teleop import ZhonglinTeleop

    cfg = ZhonglinTeleopConfig(port=str(args.teleop_port)) if args.teleop_port else ZhonglinTeleopConfig()
    return ZhonglinTeleop(cfg)


class _TeleopActionSampler:
    def __init__(self, teleop: Any, *, sample_period_s: float) -> None:
        self.teleop = teleop
        self.sample_period_s = max(float(sample_period_s), 0.001)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._first_sample_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._latest_action = {str(name): 0.0 for name in dict(getattr(teleop, "action_features", {}) or {})}
        self._sample_count = 0
        self._last_error_log_s = 0.0

    @property
    def sample_count(self) -> int:
        with self._lock:
            return int(self._sample_count)

    def connect(self, *, calibrate: bool) -> None:
        self.teleop.connect(calibrate=calibrate)

    def start(self) -> None:
        if self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="zhongling-action-sampler", daemon=True)
        self._thread.start()

    def wait_for_first_sample(self, timeout_s: float) -> bool:
        return self._first_sample_event.wait(timeout=max(float(timeout_s), 0.0))

    def get_action(self) -> dict[str, float]:
        with self._lock:
            return dict(self._latest_action)

    def disconnect(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None
        disconnect = getattr(self.teleop, "disconnect", None)
        if disconnect is not None:
            disconnect()

    def _run(self) -> None:
        while not self._stop_event.is_set():
            loop_started_s = time.perf_counter()
            try:
                action = dict(self.teleop.get_action())
            except Exception as exc:
                now_s = time.perf_counter()
                if now_s - self._last_error_log_s >= 1.0:
                    logging.warning("Zhongling teleop read failed; keeping latest action: %s", exc)
                    self._last_error_log_s = now_s
                action = {}
            if action:
                with self._lock:
                    self._latest_action.update({str(key): float(value) for key, value in action.items()})
                    self._sample_count += 1
                self._first_sample_event.set()
            elapsed_s = time.perf_counter() - loop_started_s
            self._stop_event.wait(timeout=max(self.sample_period_s - elapsed_s, 0.001))


def _sim_record_action(
    *,
    args: argparse.Namespace,
    frame_index: int,
    plan: list[tuple[np.ndarray, float]],
    teleop: _TeleopActionSampler | None,
) -> dict[str, float]:
    if teleop is not None:
        return teleop.get_action()
    joint_positions, gripper_pos = plan[frame_index]
    return _joint_action(joint_positions, gripper_pos)


def _record_viewer_process_main(
    *,
    scene_path: str,
    command_queue: Any,
    key_queue: Any,
) -> None:
    import mujoco
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    state: dict[str, Any] = {"view_mode": "free", "exit_requested": False}

    def _send_key(key: str) -> None:
        try:
            key_queue.put_nowait(key)
        except queue.Full:
            pass

    def _key_callback(keycode: int) -> None:
        if keycode in {_SimRecordViewer._GLFW_KEY_1, _SimRecordViewer._GLFW_KEY_KP_1}:
            state["view_mode"] = "front"
            return
        if keycode in {_SimRecordViewer._GLFW_KEY_2, _SimRecordViewer._GLFW_KEY_KP_2}:
            state["view_mode"] = "wrist"
            return
        if keycode in {_SimRecordViewer._GLFW_KEY_3, _SimRecordViewer._GLFW_KEY_KP_3}:
            state["view_mode"] = "free"
            return
        if keycode == _SimRecordViewer._GLFW_KEY_SPACE:
            _send_key("space")
            return
        if keycode in {_SimRecordViewer._GLFW_KEY_ENTER, _SimRecordViewer._GLFW_KEY_KP_ENTER}:
            _send_key("enter")
            return
        if keycode == _SimRecordViewer._GLFW_KEY_ESCAPE:
            _send_key("esc")
            state["exit_requested"] = True

    def _apply_fixed_camera(viewer: Any, camera_name: str) -> None:
        camera_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name))
        if camera_id < 0:
            return
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = camera_id

    def _set_view_mode(viewer: Any, view_mode: str) -> str:
        if view_mode == "front":
            _apply_fixed_camera(viewer, "front_camera")
            return "view=front_camera [1]"
        if view_mode == "wrist":
            _apply_fixed_camera(viewer, "wrist_camera")
            return "view=wrist_camera [2]"
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        viewer.cam.azimuth = 148.0
        viewer.cam.elevation = -20.0
        viewer.cam.distance = 1.75
        viewer.cam.lookat[:] = np.asarray([0.0, 0.0, 0.24], dtype=float)
        return "view=free [3]"

    def _overlay_texts(overlay_line: str) -> list[tuple[int, int, str, str]]:
        font = int(mujoco.mjtFontScale.mjFONTSCALE_150)
        gridpos = int(mujoco.mjtGridPos.mjGRID_TOPLEFT)
        return [
            (font, gridpos, "view", overlay_line),
            (font, gridpos, "keys", "1 front  2 wrist  3 free  Space start/stop  Esc quit"),
        ]

    viewer = mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=_key_callback,
        show_left_ui=False,
        show_right_ui=False,
    )
    active_view_mode: str | None = None
    overlay_line = "view=free [3]"
    latest_payload: dict[str, Any] | None = None
    try:
        while viewer.is_running() and not bool(state["exit_requested"]):
            while True:
                try:
                    command = command_queue.get_nowait()
                except queue.Empty:
                    break
                if not isinstance(command, dict):
                    continue
                if command.get("type") == "close":
                    state["exit_requested"] = True
                    break
                if command.get("type") == "state":
                    latest_payload = command

            with viewer.lock():
                if latest_payload is not None:
                    qpos = latest_payload.get("qpos")
                    if qpos is not None:
                        data.qpos[:] = np.asarray(qpos, dtype=float).reshape(model.nq)
                    ctrl = latest_payload.get("ctrl")
                    if ctrl is not None and model.nu > 0:
                        data.ctrl[:] = np.asarray(ctrl, dtype=float).reshape(model.nu)
                    mujoco.mj_forward(model, data)
                    latest_payload = None

                requested_view_mode = str(state["view_mode"])
                if requested_view_mode != active_view_mode:
                    overlay_line = _set_view_mode(viewer, requested_view_mode)
                    active_view_mode = requested_view_mode
                viewer.set_texts(_overlay_texts(overlay_line))
            viewer.sync()
            time.sleep(1.0 / 60.0)
    finally:
        try:
            with viewer.lock():
                viewer.clear_texts()
        except Exception:
            pass
        viewer.close()


class _SimRecordViewer:
    _GLFW_KEY_ESCAPE = 256
    _GLFW_KEY_ENTER = 257
    _GLFW_KEY_KP_ENTER = 335
    _GLFW_KEY_SPACE = 32
    _GLFW_KEY_1 = 49
    _GLFW_KEY_2 = 50
    _GLFW_KEY_3 = 51
    _GLFW_KEY_KP_1 = 321
    _GLFW_KEY_KP_2 = 322
    _GLFW_KEY_KP_3 = 323

    def __init__(self, sim: CjjArmSim, *, enabled: bool) -> None:
        self.sim = sim
        self.enabled = bool(enabled)
        self._lock = threading.Lock()
        self._manual_keys: list[str] = []
        self._ctx: Any | None = None
        self._command_queue: Any | None = None
        self._key_queue: Any | None = None
        self._process: Any | None = None

    def start(self) -> None:
        if not self.enabled:
            return
        from lerobot.projects.vlbiman_sa.runtime_env import require_mujoco_viewer_backend

        require_mujoco_viewer_backend()
        scene_path = getattr(self.sim, "_load_source_path", None) or self.sim.generated_scene_path
        if scene_path is None:
            raise RuntimeError("Record viewer requires a concrete MuJoCo scene path.")

        self._ctx = mp.get_context("spawn")
        self._command_queue = self._ctx.Queue(maxsize=2)
        self._key_queue = self._ctx.Queue(maxsize=16)
        self._process = self._ctx.Process(
            target=_record_viewer_process_main,
            kwargs={
                "scene_path": str(scene_path),
                "command_queue": self._command_queue,
                "key_queue": self._key_queue,
            },
            daemon=True,
        )
        self._process.start()
        self.sync()

    def close(self) -> None:
        if self._process is None:
            return
        self._put_command({"type": "close"})
        self._process.join(timeout=1.0)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=1.0)
        self._process = None
        self._command_queue = None
        self._key_queue = None

    def pop_key(self) -> str | None:
        self._poll_child_keys()
        with self._lock:
            if not self._manual_keys:
                return None
            return self._manual_keys.pop(0)

    def sim_data_lock(self) -> Any:
        return contextlib.nullcontext()

    def sync(self) -> None:
        if self._process is None:
            return
        self._poll_child_keys()
        if not self._process.is_alive():
            logging.warning("MuJoCo record viewer process exited.")
            self._process = None
            return
        payload = {
            "type": "state",
            "qpos": np.asarray(self.sim.data.qpos, dtype=float).tolist(),
            "ctrl": np.asarray(self.sim.data.ctrl, dtype=float).tolist() if self.sim.model.nu > 0 else [],
        }
        self._put_command(payload)

    def _put_command(self, payload: dict[str, Any]) -> None:
        if self._command_queue is None:
            return
        try:
            self._command_queue.put_nowait(payload)
        except queue.Full:
            try:
                self._command_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._command_queue.put_nowait(payload)
            except queue.Full:
                pass

    def _poll_child_keys(self) -> None:
        if self._key_queue is None:
            return
        while True:
            try:
                key = self._key_queue.get_nowait()
            except queue.Empty:
                break
            with self._lock:
                self._manual_keys.append(str(key))

class _TerminalKeyReader:
    def __init__(self) -> None:
        self._old_settings: list[Any] | None = None

    def __enter__(self) -> "_TerminalKeyReader":
        if not sys.stdin.isatty() or termios is None or tty is None:
            raise RuntimeError("--manual-record-control requires an interactive terminal.")
        self._old_settings = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self._old_settings is not None and termios is not None:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, self._old_settings)

    def read_key(self) -> str | None:
        ready, _, _ = select.select([sys.stdin], [], [], 0.0)
        if not ready:
            return None
        char = sys.stdin.read(1)
        if char == " ":
            return "space"
        if char in {"\n", "\r"}:
            return "enter"
        if char in {"\x7f", "\b"}:
            return "backspace"
        if char == "\x1b":
            return "esc"
        return char

    def wait_for_key(self, allowed: set[str]) -> str:
        while True:
            key = self.read_key()
            if key in allowed:
                return key
            time.sleep(0.05)

    def drain(self) -> None:
        while self.read_key() is not None:
            pass


def _read_record_key(key_reader: _TerminalKeyReader | None, viewer: _SimRecordViewer | None) -> str | None:
    key = key_reader.read_key() if key_reader is not None else None
    return key or (viewer.pop_key() if viewer is not None else None)


def _wait_for_record_key(
    key_reader: _TerminalKeyReader | None,
    viewer: _SimRecordViewer | None,
    allowed: set[str],
) -> str:
    while True:
        key = _read_record_key(key_reader, viewer)
        if key in allowed:
            return key
        if viewer is not None:
            viewer.sync()
        time.sleep(0.05)


def _drain_record_keys(key_reader: _TerminalKeyReader | None, viewer: _SimRecordViewer | None) -> None:
    if key_reader is not None:
        key_reader.drain()
    if viewer is not None:
        while viewer.pop_key() is not None:
            pass


def _prepare_sim_session_dir(recordings_dir: Path, session_name: str) -> Path:
    session_dir = recordings_dir / session_name
    if session_dir.exists():
        shutil.rmtree(session_dir)
    return create_session_dir(recordings_dir, run_name=session_name, overwrite=True)


def _prewarm_recording_renderer(sim: CjjArmSim) -> None:
    logging.info("Prewarming MuJoCo offscreen camera renderer before opening record viewer...")
    sim.render_cameras()
    logging.info("MuJoCo offscreen camera renderer is ready.")


def _record_sim_session_to_dir(
    args: argparse.Namespace,
    *,
    session_dir: Path,
    key_reader: _TerminalKeyReader | None = None,
    realtime: bool = False,
    wait_for_start: bool = False,
) -> Path:
    metadata_path = session_dir / "metadata.jsonl"
    manifest_path = session_dir / "manifest.json"
    frame_limit = int(args.sim_frames)
    cfg = RecorderConfig(
        task_name=f"sim_{_normalize_object_key(args.target_phrase)}",
        fps=int(args.sim_fps),
        max_frames=frame_limit,
        output_root=session_dir.parent,
        run_name=session_dir.name,
        overwrite=True,
    )
    sim = CjjArmSim(
        CjjArmSimConfig(
            id=f"vlbiman_original_{session_dir.parent.name}",
            mujoco_model_path=str((getattr(args, "_bound_scene", {}) or {}).get("scene_path") or ""),
            use_viewer=False,
            render_width=int(args.render_width),
            render_height=int(args.render_height),
            scene_settle_steps=0,
            scene_preset=str(args.scene_preset),
            scene_target_x=float(args.sim_target_x),
            scene_target_y=float(args.sim_target_y),
            scene_target_z=float(args.sim_target_z),
        )
    )
    started_at_ns = time.time_ns()
    recorded_frames = 0
    stopped_by_key = False
    period_s = 1.0 / max(int(args.sim_fps), 1)
    teleop = (
        _TeleopActionSampler(_build_zhongling_teleop(args), sample_period_s=period_s)
        if args.sim_teleop == "zhongling"
        else None
    )
    viewer: _SimRecordViewer | None = None
    sim.connect()
    try:
        if args.record_viewer:
            _prewarm_recording_renderer(sim)
            viewer = _SimRecordViewer(sim, enabled=True)
            viewer.start()
        if teleop is not None:
            print(
                "[manual record] connecting Zhongling controller and calibrating zero position...",
                flush=True,
            )
            teleop.connect(calibrate=not bool(args.no_teleop_calibrate))
            teleop.start()
            if teleop.wait_for_first_sample(timeout_s=1.0):
                print("[manual record] Zhongling sampling is live.", flush=True)
            else:
                print(
                    "[manual record] Zhongling sampling has not produced data yet; "
                    "recording will keep the latest/zero action until data arrives.",
                    flush=True,
                )
        if wait_for_start:
            print(
                "[manual record] viewer ready. Press Space in the terminal or MuJoCo viewer to start; Esc to quit.",
                flush=True,
            )
            key = _wait_for_record_key(key_reader, viewer, {"space", "esc"})
            if key == "esc":
                raise KeyboardInterrupt
            _drain_record_keys(key_reader, viewer)
            print(
                "[manual record] recording... press Space to stop. "
                f"Auto-stop at {int(args.sim_frames)} frames.",
                flush=True,
            )
        plan = _joint_plan(frame_limit)
        start_mono = time.perf_counter()
        last_progress_mono = 0.0
        for frame_index, (joint_positions, gripper_pos) in enumerate(plan):
            if realtime:
                target_mono = start_mono + frame_index * period_s
                while True:
                    if key_reader is not None or viewer is not None:
                        key = _read_record_key(key_reader, viewer)
                        if key == "space":
                            stopped_by_key = True
                            break
                        if key == "esc":
                            raise KeyboardInterrupt
                    if viewer is not None:
                        viewer.sync()
                    sleep_s = target_mono - time.perf_counter()
                    if sleep_s <= 0:
                        break
                    time.sleep(min(sleep_s, 0.02))
                if stopped_by_key:
                    break
            elif key_reader is not None or viewer is not None:
                key = _read_record_key(key_reader, viewer)
                if key == "space":
                    stopped_by_key = True
                    break
                if key == "esc":
                    raise KeyboardInterrupt

            action = _sim_record_action(
                args=args,
                frame_index=frame_index,
                plan=plan,
                teleop=teleop,
            )
            debug_frame = recorded_frames < 3
            if debug_frame:
                logging.info(
                    "Recording frame %d: sending action with keys=%s",
                    recorded_frames + 1,
                    sorted(action.keys()),
                )
            data_lock = viewer.sim_data_lock() if viewer is not None else contextlib.nullcontext()
            with data_lock:
                sent_action = sim.send_action(action)
                if debug_frame:
                    logging.info("Recording frame %d: action applied.", recorded_frames + 1)
                if debug_frame:
                    logging.info("Recording frame %d: rendering camera observations.", recorded_frames + 1)
                obs = sim.get_observation()
                if debug_frame:
                    logging.info("Recording frame %d: camera observations rendered.", recorded_frames + 1)
            if viewer is not None:
                if debug_frame:
                    logging.info("Recording frame %d: queueing viewer update.", recorded_frames + 1)
                viewer.sync()
                if debug_frame:
                    logging.info("Recording frame %d: viewer update queued.", recorded_frames + 1)
            numeric_observation = _extract_numeric_observation(obs)
            if "gripper.pos" in sent_action:
                numeric_observation["gripper.pos"] = float(sent_action["gripper.pos"])
            joint_state = _extract_joint_positions(numeric_observation)
            gripper_state = _extract_gripper_state(numeric_observation)
            front_rgb = np.asarray(obs[sim.config.front_camera_observation_key], dtype=np.uint8)
            wrist_rgb = np.asarray(obs[sim.config.wrist_camera_observation_key], dtype=np.uint8)
            front_depth_mm = np.full(front_rgb.shape[:2], 850, dtype=np.uint16)
            wrist_depth_mm = np.full(wrist_rgb.shape[:2], 850, dtype=np.uint16)
            now_ns = time.time_ns()
            frame = FrameRecord(
                frame_index=frame_index,
                slot_index=frame_index,
                wall_time_ns=now_ns,
                relative_time_s=frame_index * period_s,
                scheduled_time_s=frame_index * period_s,
                capture_started_ns=now_ns,
                capture_ended_ns=now_ns,
                capture_latency_ms=0.0,
                camera_timestamp_ns=now_ns,
                robot_timestamp_ns=now_ns,
                time_skew_ms=0.0,
                action_timestamp_ns=now_ns,
                action_sent_timestamp_ns=now_ns,
                robot_observation=numeric_observation,
                joint_positions=joint_state,
                gripper_state=gripper_state,
                teleop_action=action,
                sent_action=_extract_numeric_observation(sent_action),
                ee_pose=_extract_ee_pose(sim, numeric_observation),
            )
            frame = save_frame_assets(session_dir, frame, color_rgb=front_rgb, depth_map=front_depth_mm)
            frame = save_named_camera_assets(
                session_dir,
                frame,
                camera_name="front_camera",
                color_rgb=front_rgb,
                depth_map=front_depth_mm,
            )
            frame = save_named_camera_assets(
                session_dir,
                frame,
                camera_name="wrist_camera",
                color_rgb=wrist_rgb,
                depth_map=wrist_depth_mm,
            )
            append_frame_metadata(metadata_path, frame)
            recorded_frames += 1
            now_mono = time.perf_counter()
            if args.manual_record_control and (recorded_frames == 1 or now_mono - last_progress_mono >= 1.0):
                suffix = f"; Zhongling samples={teleop.sample_count}" if teleop is not None else ""
                print(
                    f"[manual record] recorded {recorded_frames}/{frame_limit} frames{suffix}; press Space to stop.",
                    flush=True,
                )
                last_progress_mono = now_mono
            if key_reader is not None or viewer is not None:
                key = _read_record_key(key_reader, viewer)
                if key == "space":
                    stopped_by_key = True
                    break
                if key == "esc":
                    raise KeyboardInterrupt
    finally:
        if viewer is not None:
            viewer.close()
        if teleop is not None:
            teleop.disconnect()
        sim.disconnect()

    ended_at_ns = time.time_ns()
    summary = RecordingSummary(
        status="completed" if recorded_frames > 0 else "failed",
        target_frame_slots=frame_limit,
        recorded_frames=recorded_frames,
        dropped_frames=0,
        failed_frames=0,
        achieved_fps=float(args.sim_fps),
        average_time_skew_ms=0.0,
        max_time_skew_ms=0.0,
        started_at_ns=started_at_ns,
        ended_at_ns=ended_at_ns,
        session_dir=session_dir,
        metadata_path=metadata_path,
        manifest_path=manifest_path,
    )
    write_manifest(
        manifest_path,
        cfg,
        summary,
        extra={
            "source": "mujoco_sim",
            "target_phrase": args.target_phrase,
            "aux_target_phrases": list(args.aux_target_phrase or []),
            "scene_preset": scene_preset_summary(args.scene_preset),
            "cameras": {
                "primary_camera": "front_camera",
                "recorded_cameras": ["front_camera", "wrist_camera"],
                "primary_alias_paths": {"rgb": "rgb", "depth": "depth"},
                "camera_asset_root": "cameras",
            },
            "manual_record_control": bool(args.manual_record_control),
            "sim_teleop": str(args.sim_teleop),
            "teleop_port": str(args.teleop_port) if args.teleop_port else None,
            "bound_scene": {
                "summary_path": str((getattr(args, "_bound_scene", {}) or {}).get("summary_path") or ""),
                "scene_path": str((getattr(args, "_bound_scene", {}) or {}).get("scene_path") or ""),
                "object_positions": {
                    str(key): [float(value) for value in np.asarray(position, dtype=float).reshape(3)]
                    for key, position in dict((getattr(args, "_bound_scene", {}) or {}).get("object_positions", {})).items()
                },
            },
            "stopped_by_key": bool(stopped_by_key),
        },
    )
    return session_dir


def _record_sim_session_manual(args: argparse.Namespace, run_dir: Path) -> Path:
    recordings_dir = run_dir / "recordings"
    final_session_name = "sim_one_shot"
    attempt_index = 1
    auto_start = False
    with _TerminalKeyReader() as key_reader:
        while True:
            attempt_session_name = f"{final_session_name}_attempt_{attempt_index:02d}"
            if auto_start:
                print(f"\n[manual record] attempt {attempt_index}: recording started.", flush=True)
            elif args.record_viewer:
                print(
                    f"\n[manual record] attempt {attempt_index}: opening MuJoCo viewer before recording starts.",
                    flush=True,
                )
            else:
                print(
                    f"\n[manual record] attempt {attempt_index}: press Space to start recording, Esc to quit.",
                    flush=True,
                )
                key = key_reader.wait_for_key({"space", "esc"})
                if key == "esc":
                    raise KeyboardInterrupt
            key_reader.drain()

            attempt_dir = _prepare_sim_session_dir(recordings_dir, attempt_session_name)
            if args.record_viewer and not auto_start:
                print(
                    "[manual record] use 1/2/3 in the MuJoCo viewer to switch views; "
                    "press Space to start.",
                    flush=True,
                )
            else:
                print(
                    "[manual record] recording... press Space to stop. "
                    f"Auto-stop at {int(args.sim_frames)} frames.",
                    flush=True,
                )
            _record_sim_session_to_dir(
                args,
                session_dir=attempt_dir,
                key_reader=key_reader,
                realtime=True,
                wait_for_start=bool(args.record_viewer and not auto_start),
            )
            key_reader.drain()
            summary = _load_json(attempt_dir / "manifest.json").get("summary", {})
            recorded_frames = int(summary.get("recorded_frames", 0))
            print(
                f"[manual record] stopped. frames={recorded_frames}. "
                "Press Enter to save, Space to re-record, Esc to quit.",
                flush=True,
            )
            allowed_keys = {"space", "esc"} if recorded_frames <= 0 else {"enter", "space", "esc"}
            if recorded_frames <= 0:
                print("[manual record] no frames recorded; press Space to re-record or Esc to quit.", flush=True)
            key = key_reader.wait_for_key(allowed_keys)
            if key == "enter":
                final_dir = recordings_dir / final_session_name
                if final_dir.exists():
                    shutil.rmtree(final_dir)
                attempt_dir.rename(final_dir)
                print(f"[manual record] saved: {final_dir}", flush=True)
                return final_dir
            if key == "esc":
                shutil.rmtree(attempt_dir, ignore_errors=True)
                raise KeyboardInterrupt
            shutil.rmtree(attempt_dir, ignore_errors=True)
            attempt_index += 1
            auto_start = True


def _record_sim_session(args: argparse.Namespace, run_dir: Path) -> Path:
    if args.manual_record_control:
        return _record_sim_session_manual(args, run_dir)
    session_dir = _prepare_sim_session_dir(run_dir / "recordings", "sim_one_shot")
    return _record_sim_session_to_dir(args, session_dir=session_dir)


def _base_from_camera(path: Path) -> np.ndarray:
    payload = _load_json(path)
    matrix = np.asarray(payload.get("base_from_camera"), dtype=float)
    if matrix.shape != (4, 4):
        raise ValueError(f"base_from_camera in {path} must be 4x4.")
    return matrix


def _camera_xyz_from_base(base_from_camera: np.ndarray, base_xyz: np.ndarray) -> np.ndarray:
    camera_from_base = np.linalg.inv(np.asarray(base_from_camera, dtype=float))
    point = np.concatenate([np.asarray(base_xyz, dtype=float).reshape(3), [1.0]])
    return (camera_from_base @ point)[:3]


def _project_camera_xyz(camera_xyz: np.ndarray, intrinsics: CameraIntrinsics, shape_hw: tuple[int, int]) -> tuple[float, float]:
    height, width = shape_hw
    scale_x = width / intrinsics.width if intrinsics.width else 1.0
    scale_y = height / intrinsics.height if intrinsics.height else 1.0
    fx = intrinsics.fx * scale_x
    fy = intrinsics.fy * scale_y
    cx = intrinsics.cx * scale_x
    cy = intrinsics.cy * scale_y
    z = max(float(camera_xyz[2]), 1e-6)
    u = float(camera_xyz[0]) * fx / z + cx
    v = float(camera_xyz[1]) * fy / z + cy
    return float(np.clip(u, 8, width - 9)), float(np.clip(v, 8, height - 9))


def _oracle_anchor(
    *,
    frame_index: int,
    object_key: str,
    phrase: str,
    base_xyz: np.ndarray,
    base_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    camera_xyz = _camera_xyz_from_base(base_from_camera, base_xyz)
    u, v = _project_camera_xyz(camera_xyz, intrinsics, image_shape)
    radius = 16 if object_key != "green_plate" else 24
    x0 = int(max(0, round(u - radius)))
    y0 = int(max(0, round(v - radius)))
    x1 = int(min(image_shape[1] - 1, round(u + radius)))
    y1 = int(min(image_shape[0] - 1, round(v + radius)))
    mask_area = max(1, (x1 - x0 + 1) * (y1 - y0 + 1))
    orientation = {
        "angle_rad": 0.0,
        "angle_deg": 0.0,
        "centroid_px": [u, v],
        "major_axis_px": float(radius * 2),
        "minor_axis_px": float(radius * 2),
        "covariance": [[float(radius * radius), 0.0], [0.0, float(radius * radius)]],
    }
    return {
        "frame_index": int(frame_index),
        "mask_area_px": int(mask_area),
        "bbox_xyxy": [x0, y0, x1, y1],
        "centroid_px": [u, v],
        "contact_px": [u, v],
        "depth_m": float(max(camera_xyz[2], 0.001)),
        "camera_xyz_m": [float(value) for value in camera_xyz],
        "orientation_deg": 0.0,
        "score": 1.0,
        "orientation": orientation,
    }


def _write_sim_t4_for_object(
    *,
    session_dir: Path,
    output_dir: Path,
    phrase: str,
    base_xyz: np.ndarray,
    base_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> dict[str, Any]:
    from lerobot.projects.vlbiman_sa.demo.io import load_frame_assets, load_frame_records

    records = load_frame_records(session_dir)
    if not records:
        raise ValueError(f"No records found in {session_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "masks").mkdir(parents=True, exist_ok=True)
    (output_dir / "overlays").mkdir(parents=True, exist_ok=True)
    object_key = _object_key_for_phrase(phrase)
    anchors: list[dict[str, Any]] = []
    frames: list[dict[str, Any]] = []
    for local_index, record in enumerate(records):
        color_rgb, _ = load_frame_assets(session_dir, record)
        anchor = _oracle_anchor(
            frame_index=int(record.frame_index),
            object_key=object_key,
            phrase=phrase,
            base_xyz=base_xyz,
            base_from_camera=base_from_camera,
            intrinsics=intrinsics,
            image_shape=color_rgb.shape[:2],
        )
        anchors.append(anchor)
        x0, y0, x1, y1 = anchor["bbox_xyxy"]
        mask = np.zeros(color_rgb.shape[:2], dtype=np.uint8)
        mask[y0 : y1 + 1, x0 : x1 + 1] = 255
        cv2.imwrite(str(output_dir / "masks" / f"frame_{record.frame_index:06d}.png"), mask)
        overlay = cv2.cvtColor(color_rgb, cv2.COLOR_RGB2BGR)
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2.putText(overlay, phrase, (max(4, x0), max(18, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.imwrite(str(output_dir / "overlays" / f"frame_{record.frame_index:06d}.png"), overlay)
        frames.append(
            {
                "frame_index": int(record.frame_index),
                "local_frame_index": int(local_index),
                "score": 1.0,
                "bbox_xyxy": anchor["bbox_xyxy"],
                "component_count": 1,
                "mask_area_px": anchor["mask_area_px"],
                "debug": {
                    "source": "mujoco_oracle",
                    "target_phrase": phrase,
                    "object_key": object_key,
                },
                "tracking": {
                    "frame_index": int(record.frame_index),
                    "mask_area_px": anchor["mask_area_px"],
                    "centroid_px": anchor["centroid_px"],
                    "temporal_iou": 1.0,
                    "stable": local_index >= 4,
                    "position_std_mm": 0.0 if local_index >= 4 else None,
                    "position_variance_mm2": 0.0 if local_index >= 4 else None,
                    "orientation_std_deg": 0.0 if local_index >= 4 else None,
                    "orientation_variance_deg2": 0.0 if local_index >= 4 else None,
                },
                "anchor": anchor,
            }
        )
    summary = {
        "status": "ok",
        "mode": "mujoco_oracle",
        "target_phrase": phrase,
        "object_key": object_key,
        "frame_count": len(records),
        "mask_valid_ratio": 1.0,
        "anchor_valid_ratio": 1.0,
        "mean_temporal_iou": 1.0,
        "stable_ratio": max(0.0, (len(records) - 4) / max(len(records), 1)),
        "first_stable_frame": 4 if len(records) > 4 else 0,
        "min_position_variance_mm2": 0.0,
        "min_orientation_variance_deg2": 0.0,
        "seed_detection": {"label": phrase, "phrase": phrase, "frame_index": 0, "bbox_xyxy": anchors[0]["bbox_xyxy"]},
    }
    _save_json(output_dir / "anchors.json", anchors)
    _save_json(output_dir / "frames.json", frames)
    _save_json(output_dir / "summary.json", summary)
    _save_json(output_dir / "self_check.json", {"status": "pass", "pipeline": "mujoco_oracle"})
    return summary


def _write_sim_live_result(
    *,
    path: Path,
    phrases: list[str],
    object_positions: dict[str, np.ndarray],
    base_from_camera: np.ndarray,
    intrinsics: CameraIntrinsics,
) -> Path:
    objects: dict[str, Any] = {}
    for phrase in phrases:
        object_key = _object_key_for_phrase(phrase)
        base_xyz = object_positions.get(object_key)
        if base_xyz is None:
            raise KeyError(f"No simulated object position for phrase '{phrase}' (object_key={object_key}).")
        anchor = _oracle_anchor(
            frame_index=0,
            object_key=object_key,
            phrase=phrase,
            base_xyz=base_xyz,
            base_from_camera=base_from_camera,
            intrinsics=intrinsics,
            image_shape=(int(intrinsics.height or 480), int(intrinsics.width or 640)),
        )
        objects[object_key] = {
            "status": "ok",
            "target_phrase": phrase,
            "seed_detection": {"label": phrase, "phrase": phrase, "frame_index": 0, "bbox_xyxy": anchor["bbox_xyxy"]},
            "segmentation": {"frame_index": 0, "score": 1.0, "bbox_xyxy": anchor["bbox_xyxy"], "component_count": 1, "mask_area_px": anchor["mask_area_px"]},
            "anchor": anchor,
            "orientation": anchor["orientation"],
            "base_xyz_m": [float(value) for value in base_xyz],
        }
    primary_key = _object_key_for_phrase(phrases[0])
    primary = objects[primary_key]
    payload = {
        "status": "ok",
        "mode": "mujoco_oracle",
        "target_phrase": phrases[0],
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "base_xyz_m": primary["base_xyz_m"],
        "anchor": primary["anchor"],
        "orientation": primary["orientation"],
        "objects": objects,
    }
    _save_json(path, payload)
    return path


def _run_analysis(
    *,
    args: argparse.Namespace,
    task_config: TaskGraspConfig,
    session_dir: Path,
    live_result_path: Path,
    run_dir: Path,
) -> tuple[Path | None, Path]:
    analysis_dir = session_dir / "analysis"
    skill_output_dir = analysis_dir / "t3_skill_bank"
    if args.skip_skill_build:
        skill_bank_path = skill_output_dir / "skill_bank.json"
        if not skill_bank_path.exists():
            raise FileNotFoundError(f"Cannot reuse T3; missing skill bank: {skill_bank_path}")
    else:
        skill_result = build_skill_bank(
            session_dir=session_dir,
            output_dir=skill_output_dir,
            segmenter_config=SegmenterConfig(),
            classifier_config=(
                InvarianceClassifierConfig(var_score_threshold=0.35)
                if args.source in {"sim", "sim-session"}
                else InvarianceClassifierConfig()
            ),
        )
        skill_bank_path = skill_result.skill_bank_path
    if args.stop_after == "skill_build":
        flow_summary = {
            "status": "ok",
            "source": args.source,
            "stopped_after": "skill_build",
            "target_phrase": args.target_phrase,
            "aux_target_phrases": list(args.aux_target_phrase or []),
            "session_dir": str(session_dir),
            "live_result_path": str(live_result_path),
            "skill_bank_path": str(skill_bank_path),
            "run_dir": str(run_dir),
        }
        _save_json(run_dir / "flow_summary.json", flow_summary)
        return None, run_dir / "flow_summary.json"
    if args.source == "real-session" and not args.skip_real_t4:
        from lerobot.projects.vlbiman_sa.app.run_visual_analysis import VisionConfig, run_visual_analysis_pipeline

        vision_payload = {
            "segmentor": {},
            "tracker": {},
            "anchor": {},
        }
        run_visual_analysis_pipeline(
            VisionConfig(
                session_dir=session_dir,
                skill_bank_path=skill_bank_path,
                output_dir=analysis_dir / "t4_vision",
                intrinsics_path=task_config.intrinsics_path,
                target_phrase=args.target_phrase,
                task_prompt=args.target_phrase,
                frame_stride=1,
                use_var_segments_only=False,
            ),
            vision_payload,
        )
        for phrase in args.aux_target_phrase or []:
            run_visual_analysis_pipeline(
                VisionConfig(
                    session_dir=session_dir,
                    skill_bank_path=skill_bank_path,
                    output_dir=analysis_dir / f"t4_vision_{_normalize_object_key(phrase)}",
                    intrinsics_path=task_config.intrinsics_path,
                    target_phrase=phrase,
                    task_prompt=phrase,
                    frame_stride=1,
                    use_var_segments_only=False,
                ),
                vision_payload,
            )

    task_config.recording_session_dir = session_dir
    task_config.skill_bank_path = skill_bank_path
    task_config.skill_output_dir = skill_output_dir
    task_config.vision_output_dir = analysis_dir / "t4_vision"
    task_config.pose_output_dir = analysis_dir / "t5_pose"
    task_config.trajectory_output_dir = analysis_dir / "t6_trajectory"
    task_config.live_result_path = live_result_path
    task_config.target_phrase = args.target_phrase
    task_config.task_prompt = args.target_phrase
    task_config.primary_reference_phrase = args.target_phrase
    if args.aux_target_phrase:
        task_config.secondary_target_phrase = str(args.aux_target_phrase[0])
        task_config.secondary_reference_phrase = str(args.aux_target_phrase[0])
        task_config.secondary_vision_dir_name = f"t4_vision_{_normalize_object_key(args.aux_target_phrase[0])}"

    pose_summary = run_pose_adaptation_pipeline(
        build_pose_pipeline_config(
            task_config=task_config,
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=analysis_dir / "t5_pose",
            live_result_path=live_result_path,
            auxiliary_target_phrases=list(args.aux_target_phrase or []),
            allow_configured_secondary_fallback=bool(args.aux_target_phrase),
        )
    )
    if pose_summary.get("status") != "ok":
        raise RuntimeError(f"T5 pose adaptation failed: {pose_summary}")
    trajectory_summary = run_trajectory_generation_pipeline(
        TrajectoryPipelineConfig(
            session_dir=session_dir,
            analysis_dir=analysis_dir,
            output_dir=analysis_dir / "t6_trajectory",
            skill_bank_path=skill_bank_path,
            adapted_pose_path=analysis_dir / "t5_pose" / "adapted_pose.json",
        )
    )
    trajectory_points_path = Path(str(trajectory_summary["trajectory_points_path"]))
    flow_summary = {
        "status": "ok",
        "source": args.source,
        "stopped_after": args.stop_after,
        "target_phrase": args.target_phrase,
        "aux_target_phrases": list(args.aux_target_phrase or []),
        "session_dir": str(session_dir),
        "live_result_path": str(live_result_path),
        "skill_bank_path": str(skill_bank_path),
        "pose_summary_path": str(analysis_dir / "t5_pose" / "summary.json"),
        "trajectory_summary_path": str(analysis_dir / "t6_trajectory" / "summary.json"),
        "trajectory_points_path": str(trajectory_points_path),
        "run_dir": str(run_dir),
    }
    _save_json(run_dir / "flow_summary.json", flow_summary)
    return trajectory_points_path, run_dir / "flow_summary.json"


def _run_execution(args: argparse.Namespace, trajectory_points_path: Path) -> int:
    if args.execute_backend == "none":
        return 0
    if args.execute_backend == "sim":
        cmd = [
            sys.executable,
            str(REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "app" / "play_mujoco_trajectory.py"),
            "--trajectory-points",
            str(trajectory_points_path),
            "--scene-preset",
            str(args.scene_preset),
            "--target-phrase",
            str(args.target_phrase),
        ]
        for phrase in args.aux_target_phrase or []:
            cmd.extend(["--aux-target-phrase", str(phrase)])
        if args.display:
            cmd.extend(["--display", str(args.display)])
        if not args.open_viewer:
            cmd.append("--preview-diagnostics-only")
        return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode

    cmd = [
        sys.executable,
        str(REPO_ROOT / "src" / "lerobot" / "projects" / "vlbiman_sa" / "app" / "run_execute_t6_on_robot.py"),
        "--trajectory-points",
        str(trajectory_points_path),
        "--target-phrase",
        str(args.target_phrase),
    ]
    for phrase in args.aux_target_phrase or []:
        cmd.extend(["--aux-target-phrase", str(phrase)])
    if args.robot_serial_port:
        cmd.extend(["--robot-serial-port", str(args.robot_serial_port)])
    if args.dry_run_robot:
        cmd.append("--dry-run")
    return subprocess.run(cmd, cwd=REPO_ROOT, check=False).returncode


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)
    run_id = args.run_id or _timestamp_run_id(f"{args.source}_{_normalize_object_key(args.target_phrase)}")
    run_dir = (args.output_root / run_id).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    task_config = _load_task_config(args.task_config)
    args._bound_scene = _resolve_bound_scene(args)
    if args.source == "sim" and args._bound_scene.get("scene_path") is not None:
        logging.info("Bound sim recording to scene: %s", args._bound_scene["scene_path"])

    if args.source == "sim":
        session_dir = _record_sim_session(args, run_dir)
        object_positions = _sim_object_positions(args)
        base_from_camera = _base_from_camera(task_config.handeye_result_path)
        intrinsics = CameraIntrinsics.from_json(task_config.intrinsics_path)
        phrases = [args.target_phrase, *list(args.aux_target_phrase or [])]
        live_result_path = run_dir / "live_result.json"
        _write_sim_live_result(
            path=live_result_path,
            phrases=phrases,
            object_positions=object_positions,
            base_from_camera=base_from_camera,
            intrinsics=intrinsics,
        )
        for index, phrase in enumerate(phrases):
            output_dir = session_dir / "analysis" / ("t4_vision" if index == 0 else f"t4_vision_{_normalize_object_key(phrase)}")
            object_key = _object_key_for_phrase(phrase)
            _write_sim_t4_for_object(
                session_dir=session_dir,
                output_dir=output_dir,
                phrase=phrase,
                base_xyz=object_positions[object_key],
                base_from_camera=base_from_camera,
                intrinsics=intrinsics,
            )
        if args.stop_after == "record":
            flow_summary = {
                "status": "ok",
                "source": args.source,
                "stopped_after": "record",
                "target_phrase": args.target_phrase,
                "aux_target_phrases": list(args.aux_target_phrase or []),
                "session_dir": str(session_dir),
                "live_result_path": str(live_result_path),
                "run_dir": str(run_dir),
                "t4_dirs": [
                    str(session_dir / "analysis" / ("t4_vision" if index == 0 else f"t4_vision_{_normalize_object_key(phrase)}"))
                    for index, phrase in enumerate(phrases)
                ],
            }
            _save_json(run_dir / "flow_summary.json", flow_summary)
            logging.info("Sim recording/session summary: %s", run_dir / "flow_summary.json")
            return 0
    else:
        if args.stop_after == "record":
            raise ValueError("--stop-after record is only supported with --source sim; use run_one_shot_record.py for real recording.")
        if args.session_dir is None:
            raise ValueError(f"--session-dir is required with --source {args.source}.")
        session_dir = args.session_dir
        live_result_path = args.live_result_path or task_config.live_result_path
        if live_result_path is None or not Path(live_result_path).exists():
            raise FileNotFoundError(
                f"Provide --live-result-path for {args.source}, or set an existing path in task config."
            )
        live_result_path = Path(live_result_path)

    trajectory_points_path, flow_summary_path = _run_analysis(
        args=args,
        task_config=task_config,
        session_dir=session_dir,
        live_result_path=live_result_path,
        run_dir=run_dir,
    )
    logging.info("Original VLBiMan flow summary: %s", flow_summary_path)
    if trajectory_points_path is None:
        logging.info("Pipeline stopped before T6 trajectory generation.")
        return 0
    logging.info("Trajectory points: %s", trajectory_points_path)
    if args.stop_after == "trajectory":
        return 0
    return _run_execution(args, trajectory_points_path)


if __name__ == "__main__":
    raise SystemExit(main())
