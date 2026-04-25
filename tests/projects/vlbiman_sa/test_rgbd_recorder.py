from __future__ import annotations

import time
from types import SimpleNamespace

import numpy as np

from lerobot.projects.vlbiman_sa.demo.io import iter_recorded_frames, load_manifest
from lerobot.projects.vlbiman_sa.demo.rgbd_recorder import RGBDRecorder
from lerobot.projects.vlbiman_sa.demo.schema import RecorderConfig


class FakeKinematics:
    def compute_fk(self, joint_positions: np.ndarray) -> np.ndarray:
        return np.array(
            [
                joint_positions[0],
                joint_positions[1],
                joint_positions[2],
                0.1,
                0.2,
                0.3,
            ],
            dtype=float,
        )


class FakeRobot:
    def __init__(self) -> None:
        self.is_connected = False
        self.kinematics = FakeKinematics()
        self.config = SimpleNamespace(
            urdf_joint_map={
                "joint_1": "joint1",
                "joint_2": "joint2",
                "joint_3": "joint3",
            }
        )
        self._step = 0
        self.send_count = 0

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_observation(self) -> dict[str, float]:
        value = float(self._step)
        self._step += 1
        return {
            "joint_1.pos": 0.1 + value,
            "joint_2.pos": 0.2 + value,
            "joint_3.pos": 0.3 + value,
            "joint_1.vel": 1.0,
            "joint_2.vel": 2.0,
            "joint_3.vel": 3.0,
            "gripper.pos": 0.5,
            "gripper.torque": 0.05,
        }

    def send_action(self, action: dict[str, float]) -> dict[str, float]:
        self.send_count += 1
        return dict(action)


class FakeCamera:
    def __init__(self, delay_s: float = 0.0) -> None:
        self.is_connected = False
        self._step = 0
        self.delay_s = delay_s

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def read_rgbd(self, timeout_ms: int = 200) -> tuple[np.ndarray, np.ndarray]:
        if self.delay_s > 0:
            time.sleep(self.delay_s)
        value = self._step
        self._step += 1
        color = np.full((4, 5, 3), value, dtype=np.uint8)
        depth = np.full((4, 5), value, dtype=np.uint16)
        return color, depth


class FakeTeleop:
    def __init__(self) -> None:
        self.is_connected = False
        self._step = 0

    def connect(self) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False

    def get_action(self) -> dict[str, float]:
        value = float(self._step)
        self._step += 1
        return {
            "joint_1.pos": value + 1.0,
            "joint_2.pos": value + 2.0,
            "joint_3.pos": value + 3.0,
            "gripper.pos": 0.25,
        }


def test_rgbd_recorder_persists_and_replays_recording(tmp_path):
    config = RecorderConfig(
        fps=100,
        max_frames=3,
        output_root=tmp_path,
        run_name="demo_case",
    )
    recorder = RGBDRecorder(
        config,
        robot=FakeRobot(),
        camera=FakeCamera(),
        teleop=FakeTeleop(),
        manifest_extra={"source": "test"},
    )

    summary = recorder.record()

    assert summary.status == "completed"
    assert summary.recorded_frames == 3
    assert summary.failed_frames == 0
    assert summary.session_dir == tmp_path / "demo_case"

    manifest = load_manifest(summary.manifest_path)
    assert manifest["config"]["frame_slots"] == 3
    assert manifest["extra"]["source"] == "test"

    replayed = list(iter_recorded_frames(summary.session_dir))
    assert len(replayed) == 3

    first_frame, first_rgb, first_depth = replayed[0]
    assert first_frame.color_path == "rgb/frame_000000.png"
    assert first_frame.depth_path == "depth/frame_000000.npy"
    assert first_frame.ee_pose == [0.1, 0.2, 0.3, 0.1, 0.2, 0.3]
    assert first_frame.teleop_action == {
        "joint_1.pos": 1.0,
        "joint_2.pos": 2.0,
        "joint_3.pos": 3.0,
        "gripper.pos": 0.25,
    }
    assert first_frame.sent_action == first_frame.teleop_action
    assert np.all(first_rgb == 0)
    assert np.all(first_depth == 0)

    last_frame, last_rgb, last_depth = replayed[-1]
    assert last_frame.joint_positions["joint_3.pos"] == 2.3
    assert np.all(last_rgb == 2)
    assert np.all(last_depth == 2)


def test_rgbd_recorder_runs_teleop_control_faster_than_recording(tmp_path):
    robot = FakeRobot()
    config = RecorderConfig(
        fps=10,
        control_rate_hz=40,
        max_frames=3,
        output_root=tmp_path,
        run_name="control_rate_case",
    )
    recorder = RGBDRecorder(
        config,
        robot=robot,
        camera=FakeCamera(delay_s=0.05),
        teleop=FakeTeleop(),
    )

    summary = recorder.record()

    assert summary.status == "completed"
    assert summary.recorded_frames == 3
    assert robot.send_count > summary.recorded_frames
