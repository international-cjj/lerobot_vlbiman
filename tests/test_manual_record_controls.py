from __future__ import annotations

from types import SimpleNamespace

from lerobot.scripts.lerobot_record import (
    _wait_for_manual_episode_save_decision,
    _wait_for_manual_episode_start,
)
from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardPoseTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardPoseTeleop
from lerobot.teleoperators.keyboard import teleop_keyboard as teleop_keyboard_module
from lerobot.utils.control_utils import (
    DISCARD_EPISODE_EVENT,
    EPISODE_TOGGLE_EVENT,
    SAVE_EPISODE_EVENT,
)


def _manual_events() -> dict[str, bool]:
    return {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
        EPISODE_TOGGLE_EVENT: False,
        SAVE_EPISODE_EVENT: False,
        DISCARD_EPISODE_EVENT: False,
    }


def test_wait_for_manual_episode_start_consumes_space_toggle() -> None:
    events = _manual_events()
    events[EPISODE_TOGGLE_EVENT] = True

    assert _wait_for_manual_episode_start(events, poll_interval_s=0.0) is True
    assert events[EPISODE_TOGGLE_EVENT] is False


def test_wait_for_manual_episode_save_decision_accepts_enter() -> None:
    events = _manual_events()
    events[SAVE_EPISODE_EVENT] = True

    assert _wait_for_manual_episode_save_decision(events, poll_interval_s=0.0) == "save"
    assert events[SAVE_EPISODE_EVENT] is False


def test_wait_for_manual_episode_save_decision_accepts_backspace() -> None:
    events = _manual_events()
    events[DISCARD_EPISODE_EVENT] = True

    assert _wait_for_manual_episode_save_decision(events, poll_interval_s=0.0) == "discard"
    assert events[DISCARD_EPISODE_EVENT] is False


def test_keyboard_pose_uses_left_shift_for_positive_z(monkeypatch) -> None:
    fake_keyboard = SimpleNamespace(
        Key=SimpleNamespace(
            up="up",
            down="down",
            left="left",
            right="right",
            shift="shift",
            shift_r="shift_r",
            ctrl_l="ctrl_l",
            ctrl_r="ctrl_r",
            space="space",
        )
    )
    monkeypatch.setattr(teleop_keyboard_module, "keyboard", fake_keyboard)
    monkeypatch.setattr(teleop_keyboard_module, "PYNPUT_AVAILABLE", True)
    monkeypatch.setattr(KeyboardPoseTeleop, "is_connected", property(lambda self: True))

    teleop = KeyboardPoseTeleop(KeyboardPoseTeleopConfig())
    teleop.current_pressed = {fake_keyboard.Key.shift: True}
    action = teleop.get_action()
    assert action["delta_z"] == teleop.config.pos_step

    teleop.current_pressed = {fake_keyboard.Key.space: True}
    action = teleop.get_action()
    assert action["delta_z"] == 0.0
