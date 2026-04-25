from __future__ import annotations

import logging
import threading
import time
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .io import append_frame_metadata, create_session_dir, save_frame_assets, write_manifest
from .schema import FrameRecord, RecorderConfig, RecordingSummary

logger = logging.getLogger(__name__)


class SupportsRGBDRead(Protocol):
    def read_rgbd(self, timeout_ms: int = 200) -> tuple[np.ndarray, np.ndarray]: ...


class SupportsObservation(Protocol):
    def get_observation(self) -> dict[str, Any]: ...


class SupportsSendAction(Protocol):
    def send_action(self, action: dict[str, Any]) -> dict[str, Any]: ...


class SupportsTeleopAction(Protocol):
    def get_action(self) -> dict[str, Any]: ...


@dataclass(slots=True)
class _ActionSnapshot:
    teleop_action: dict[str, float]
    sent_action: dict[str, float]
    action_timestamp_ns: int | None
    action_sent_timestamp_ns: int | None


class _TeleopControlLoop:
    def __init__(
        self,
        *,
        teleop: SupportsTeleopAction,
        robot: SupportsObservation,
        robot_lock: threading.Lock,
        rate_hz: float,
    ) -> None:
        self.teleop = teleop
        self.robot = robot
        self.robot_lock = robot_lock
        self.rate_hz = rate_hz
        self._snapshot = _ActionSnapshot({}, {}, None, None)
        self._snapshot_lock = threading.Lock()
        self._first_action_event = threading.Event()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, name="vlbiman-teleop-control", daemon=True)
        self._thread.start()

    def wait_until_ready(self, timeout_s: float) -> bool:
        return self._first_action_event.wait(timeout=timeout_s)

    def snapshot(self) -> _ActionSnapshot:
        if self._error is not None:
            raise RuntimeError("Teleop control loop failed.") from self._error
        with self._snapshot_lock:
            return _ActionSnapshot(
                teleop_action=dict(self._snapshot.teleop_action),
                sent_action=dict(self._snapshot.sent_action),
                action_timestamp_ns=self._snapshot.action_timestamp_ns,
                action_sent_timestamp_ns=self._snapshot.action_sent_timestamp_ns,
            )

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def _run(self) -> None:
        robot_send_action = getattr(self.robot, "send_action", None)
        if robot_send_action is None:
            self._error = AttributeError(f"{self.robot} does not implement send_action(), but teleop is enabled.")
            self._first_action_event.set()
            return

        period_s = 1.0 / max(self.rate_hz, 1e-3)
        next_time = time.perf_counter()
        while not self._stop_event.is_set():
            try:
                action_raw = self.teleop.get_action()
                action_timestamp_ns = time.time_ns()
                teleop_action = _extract_numeric_observation(action_raw)
                with self.robot_lock:
                    sent_action_raw = robot_send_action(dict(action_raw))
                action_sent_timestamp_ns = time.time_ns()
                sent_action = _extract_numeric_observation(sent_action_raw)
                with self._snapshot_lock:
                    self._snapshot = _ActionSnapshot(
                        teleop_action=teleop_action,
                        sent_action=sent_action,
                        action_timestamp_ns=action_timestamp_ns,
                        action_sent_timestamp_ns=action_sent_timestamp_ns,
                    )
                self._first_action_event.set()
            except Exception as exc:
                self._error = exc
                self._first_action_event.set()
                logger.exception("Teleop control loop failed.")
                return

            next_time += period_s
            sleep_s = next_time - time.perf_counter()
            if sleep_s > 0:
                self._stop_event.wait(timeout=sleep_s)


class RGBDRecorder:
    def __init__(
        self,
        config: RecorderConfig,
        robot: SupportsObservation,
        camera: SupportsRGBDRead,
        teleop: SupportsTeleopAction | None = None,
        *,
        manifest_extra: dict[str, Any] | None = None,
    ) -> None:
        self.config = config
        self.robot = robot
        self.camera = camera
        self.teleop = teleop
        self.manifest_extra = manifest_extra or {}

    def record(self) -> RecordingSummary:
        self.config.validate()
        target_slots = self.config.resolve_frame_slots()
        session_dir = create_session_dir(
            output_root=self.config.output_root,
            run_name=self.config.run_name,
            overwrite=self.config.overwrite,
        )
        metadata_path = session_dir / "metadata.jsonl"
        manifest_path = session_dir / "manifest.json"
        period_s = 1.0 / float(self.config.fps)

        recorded_frames = 0
        dropped_frames = 0
        failed_frames = 0
        skew_samples: list[float] = []
        first_capture_started_ns: int | None = None
        last_capture_ended_ns: int | None = None
        started_at_ns = time.time_ns()
        interrupted = False
        robot_lock = threading.Lock()

        with ExitStack() as stack:
            if _connect_if_needed(self.camera):
                stack.callback(_disconnect_quietly, self.camera)
            if _connect_if_needed(self.robot):
                stack.callback(_disconnect_quietly, self.robot)
            if self.teleop is not None and _connect_if_needed(self.teleop):
                stack.callback(_disconnect_quietly, self.teleop)

            control_loop: _TeleopControlLoop | None = None
            if self.teleop is not None and self.config.control_rate_hz > float(self.config.fps):
                control_loop = _TeleopControlLoop(
                    teleop=self.teleop,
                    robot=self.robot,
                    robot_lock=robot_lock,
                    rate_hz=self.config.control_rate_hz,
                )
                control_loop.start()
                stack.callback(control_loop.stop)
                if not control_loop.wait_until_ready(timeout_s=max(period_s, 0.25)):
                    logger.warning(
                        "Teleop control loop did not produce an action within %.3f s.",
                        max(period_s, 0.25),
                    )

            start_mono = time.perf_counter()
            slot_index = 0
            consecutive_failures = 0

            try:
                while slot_index < target_slots:
                    scheduled_mono = start_mono + slot_index * period_s
                    now_mono = time.perf_counter()
                    if scheduled_mono > now_mono:
                        time.sleep(scheduled_mono - now_mono)

                    capture_started_ns = time.time_ns()
                    capture_start_mono = time.perf_counter()
                    try:
                        with robot_lock:
                            robot_observation = self.robot.get_observation()
                        robot_timestamp_ns = time.time_ns()
                        color_rgb, depth_map = self.camera.read_rgbd(timeout_ms=self.config.camera_timeout_ms)
                        camera_timestamp_ns = time.time_ns()
                        if control_loop is not None:
                            action_snapshot = control_loop.snapshot()
                            teleop_action = action_snapshot.teleop_action
                            sent_action = action_snapshot.sent_action
                            action_timestamp_ns = action_snapshot.action_timestamp_ns
                            action_sent_timestamp_ns = action_snapshot.action_sent_timestamp_ns
                        else:
                            teleop_action, sent_action, action_timestamp_ns, action_sent_timestamp_ns = (
                                self._get_and_send_action(robot_lock)
                            )
                    except Exception:
                        failed_frames += 1
                        consecutive_failures += 1
                        logger.exception("Failed to record frame for slot %s.", slot_index)
                        if consecutive_failures > self.config.max_consecutive_failures:
                            raise RuntimeError(
                                f"Exceeded max_consecutive_failures={self.config.max_consecutive_failures}."
                            )
                        slot_index += 1
                        continue

                    consecutive_failures = 0
                    capture_ended_ns = time.time_ns()
                    capture_end_mono = time.perf_counter()
                    if first_capture_started_ns is None:
                        first_capture_started_ns = capture_started_ns
                    last_capture_ended_ns = capture_ended_ns

                    numeric_observation = _extract_numeric_observation(robot_observation)
                    joint_positions = _extract_joint_positions(numeric_observation)
                    gripper_state = _extract_gripper_state(numeric_observation)
                    ee_pose = _extract_ee_pose(self.robot, numeric_observation)

                    frame = FrameRecord(
                        frame_index=recorded_frames,
                        slot_index=slot_index,
                        wall_time_ns=capture_ended_ns,
                        relative_time_s=(capture_started_ns - started_at_ns) / 1e9,
                        scheduled_time_s=slot_index * period_s,
                        capture_started_ns=capture_started_ns,
                        capture_ended_ns=capture_ended_ns,
                        capture_latency_ms=(capture_ended_ns - capture_started_ns) / 1e6,
                        camera_timestamp_ns=camera_timestamp_ns,
                        robot_timestamp_ns=robot_timestamp_ns,
                        action_timestamp_ns=action_timestamp_ns,
                        action_sent_timestamp_ns=action_sent_timestamp_ns,
                        time_skew_ms=abs(robot_timestamp_ns - camera_timestamp_ns) / 1e6,
                        robot_observation=numeric_observation,
                        joint_positions=joint_positions,
                        gripper_state=gripper_state,
                        teleop_action=teleop_action,
                        sent_action=sent_action,
                        ee_pose=ee_pose,
                    )
                    frame = save_frame_assets(session_dir, frame, color_rgb=color_rgb, depth_map=depth_map)
                    append_frame_metadata(metadata_path, frame)

                    recorded_frames += 1
                    skew_samples.append(frame.time_skew_ms)
                    if frame.time_skew_ms > self.config.max_time_skew_ms:
                        logger.warning(
                            "Frame %s exceeded max_time_skew_ms: %.2f > %.2f",
                            frame.frame_index,
                            frame.time_skew_ms,
                            self.config.max_time_skew_ms,
                        )

                    slot_index += 1
                    late_by_s = capture_end_mono - (start_mono + slot_index * period_s)
                    if late_by_s > 0:
                        skipped_slots = int(late_by_s // period_s)
                        if skipped_slots > 0:
                            dropped_frames += skipped_slots
                            slot_index = min(target_slots, slot_index + skipped_slots)

                    if recorded_frames == 1 or recorded_frames % 10 == 0:
                        logger.info(
                            "Recorded %s/%s frames (dropped=%s, failed=%s).",
                            recorded_frames,
                            target_slots,
                            dropped_frames,
                            failed_frames,
                        )
            except KeyboardInterrupt:
                interrupted = True
                logger.info("Interrupted by user. Finalizing partial recording.")

        ended_at_ns = time.time_ns()
        achieved_fps = 0.0
        if recorded_frames > 0 and first_capture_started_ns is not None and last_capture_ended_ns is not None:
            duration_s = max((last_capture_ended_ns - first_capture_started_ns) / 1e9, 1e-9)
            achieved_fps = recorded_frames / duration_s

        summary = RecordingSummary(
            status=(
                "interrupted"
                if interrupted
                else ("completed" if recorded_frames > 0 else "failed")
            ),
            target_frame_slots=target_slots,
            recorded_frames=recorded_frames,
            dropped_frames=dropped_frames,
            failed_frames=failed_frames,
            achieved_fps=achieved_fps,
            average_time_skew_ms=float(np.mean(skew_samples)) if skew_samples else 0.0,
            max_time_skew_ms=float(np.max(skew_samples)) if skew_samples else 0.0,
            started_at_ns=started_at_ns,
            ended_at_ns=ended_at_ns,
            session_dir=session_dir,
            metadata_path=metadata_path,
            manifest_path=manifest_path,
        )
        write_manifest(manifest_path, self.config, summary, extra=self.manifest_extra)
        return summary

    def _get_and_send_action(
        self,
        robot_lock: threading.Lock,
    ) -> tuple[dict[str, float], dict[str, float], int | None, int | None]:
        if self.teleop is None:
            return {}, {}, None, None

        action_raw = self.teleop.get_action()
        action_timestamp_ns = time.time_ns()
        teleop_action = _extract_numeric_observation(action_raw)

        robot_send_action = getattr(self.robot, "send_action", None)
        if robot_send_action is None:
            raise AttributeError(f"{self.robot} does not implement send_action(), but teleop is enabled.")
        with robot_lock:
            sent_action_raw = robot_send_action(dict(action_raw))
        action_sent_timestamp_ns = time.time_ns()
        sent_action = _extract_numeric_observation(sent_action_raw)
        return teleop_action, sent_action, action_timestamp_ns, action_sent_timestamp_ns


def _connect_if_needed(node: Any) -> bool:
    is_connected = getattr(node, "is_connected", None)
    if isinstance(is_connected, bool) and is_connected:
        return False
    connect = getattr(node, "connect", None)
    if connect is None:
        return False
    connect()
    return True


def _disconnect_quietly(node: Any) -> None:
    disconnect = getattr(node, "disconnect", None)
    if disconnect is None:
        return
    try:
        disconnect()
    except Exception:
        logger.exception("Failed to disconnect %s cleanly.", node)


def _extract_numeric_observation(observation: dict[str, Any]) -> dict[str, float]:
    numeric: dict[str, float] = {}
    for key, value in observation.items():
        if isinstance(value, (bool, str, bytes)):
            continue
        if np.isscalar(value):
            numeric[str(key)] = float(value)
    return numeric


def _extract_joint_positions(observation: dict[str, float]) -> dict[str, float]:
    return {
        key: value
        for key, value in sorted(observation.items())
        if key.endswith(".pos") and not key.startswith("gripper.")
    }


def _extract_gripper_state(observation: dict[str, float]) -> dict[str, float]:
    return {key: value for key, value in sorted(observation.items()) if key.startswith("gripper.")}


def _extract_ee_pose(robot: Any, observation: dict[str, float]) -> list[float] | None:
    pose_keys = ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]
    if all(key in observation for key in pose_keys):
        return [float(observation[key]) for key in pose_keys]

    simple_pose_keys = ["x", "y", "z", "rx", "ry", "rz"]
    if all(key in observation for key in simple_pose_keys):
        return [float(observation[key]) for key in simple_pose_keys]

    kinematics = getattr(robot, "kinematics", None)
    config = getattr(robot, "config", None)
    urdf_joint_map = getattr(config, "urdf_joint_map", None)
    if kinematics is None or urdf_joint_map is None:
        return None

    joint_names = list(urdf_joint_map.keys())
    required_keys = [f"{joint_name}.pos" for joint_name in joint_names]
    if not all(key in observation for key in required_keys):
        return None

    joint_positions = np.array([observation[key] for key in required_keys], dtype=float)
    pose = np.asarray(kinematics.compute_fk(joint_positions), dtype=float).reshape(-1)
    if pose.shape[0] < 6:
        return None
    return pose[:6].astype(float).tolist()
