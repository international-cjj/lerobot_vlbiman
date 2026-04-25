from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from lerobot_teleoperator_zhonglin import ZhonglinTeleop, ZhonglinTeleopConfig


def _make_connected_teleop() -> ZhonglinTeleop:
    teleop = ZhonglinTeleop(ZhonglinTeleopConfig(port="/dev/null"))
    teleop.ser = SimpleNamespace(is_open=True)
    teleop._connected = True
    teleop._calibrated = True
    teleop.zero_angles = {servo_id: 0.0 for servo_id in teleop.valid_servos}
    return teleop


def test_joint1_deadband_holds_small_readback_noise(monkeypatch) -> None:
    teleop = _make_connected_teleop()

    def _fake_read_servo_angle_deg(servo_id: int) -> float:
        if servo_id == 0:
            return 0.5
        return 0.0

    monkeypatch.setattr(teleop, "_read_servo_angle_deg", _fake_read_servo_angle_deg)
    action = teleop.get_action()

    assert action["joint_1.pos"] == 0.0


def test_joint1_updates_when_motion_exceeds_deadband(monkeypatch) -> None:
    teleop = _make_connected_teleop()

    def _fake_read_servo_angle_deg(servo_id: int) -> float:
        if servo_id == 0:
            return 5.0
        return 0.0

    monkeypatch.setattr(teleop, "_read_servo_angle_deg", _fake_read_servo_angle_deg)
    action = teleop.get_action()

    assert action["joint_1.pos"] == np.deg2rad(5.0)
