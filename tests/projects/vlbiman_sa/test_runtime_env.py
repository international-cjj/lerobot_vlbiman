from __future__ import annotations

import os

import pytest

from lerobot.projects.vlbiman_sa import runtime_env


def test_prepare_mujoco_runtime_uses_cli_display_and_glfw(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(runtime_env, "_can_open_x_display", lambda display: display == ":9")

    backend = runtime_env.prepare_mujoco_runtime(argv=["--display", ":9"])

    assert backend == "glfw"
    assert os.environ["DISPLAY"] == ":9"
    assert os.environ["MUJOCO_GL"] == "glfw"


def test_prepare_mujoco_runtime_prefers_egl_for_headless(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":9")
    monkeypatch.delenv("MUJOCO_GL", raising=False)
    monkeypatch.setattr(runtime_env, "_can_open_x_display", lambda display: True)

    backend = runtime_env.prepare_mujoco_runtime(argv=["--headless", "--display", ":9"])

    assert backend == "egl"
    assert os.environ["MUJOCO_GL"] == "egl"


def test_require_mujoco_viewer_backend_rejects_non_glfw_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":1")
    monkeypatch.setenv("MUJOCO_GL", "egl")
    monkeypatch.setattr(runtime_env, "_can_open_x_display", lambda display: True)

    with pytest.raises(RuntimeError, match="MUJOCO_GL=glfw"):
        runtime_env.require_mujoco_viewer_backend()


def test_require_mujoco_viewer_backend_rejects_unreachable_display(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("DISPLAY", ":404")
    monkeypatch.setenv("MUJOCO_GL", "glfw")
    monkeypatch.setattr(runtime_env, "_can_open_x_display", lambda display: False)

    with pytest.raises(RuntimeError, match="reachable X display"):
        runtime_env.require_mujoco_viewer_backend()
