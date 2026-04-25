from __future__ import annotations

import ctypes
import ctypes.util
import os
import site
import sys
from pathlib import Path
from typing import Sequence


def strip_user_site_packages() -> None:
    os.environ.setdefault("PYTHONNOUSERSITE", "1")
    candidate_paths = _user_site_paths()
    if not candidate_paths:
        return

    filtered_paths: list[str] = []
    for entry in sys.path:
        if not entry:
            filtered_paths.append(entry)
            continue
        try:
            resolved = Path(entry).expanduser().resolve()
        except OSError:
            filtered_paths.append(entry)
            continue
        if any(resolved == candidate or candidate in resolved.parents for candidate in candidate_paths):
            continue
        filtered_paths.append(entry)
    sys.path[:] = filtered_paths


def _user_site_paths() -> set[Path]:
    raw_paths = site.getusersitepackages()
    if isinstance(raw_paths, str):
        raw_paths = [raw_paths]

    candidates: set[Path] = set()
    for raw_path in raw_paths:
        if not raw_path:
            continue
        try:
            candidates.add(Path(raw_path).expanduser().resolve())
        except OSError:
            continue
    return candidates


def apply_display_from_argv(
    argv: Sequence[str] | None = None,
    *,
    option_names: Sequence[str] = ("--display",),
) -> str | None:
    argv = list(sys.argv[1:] if argv is None else argv)
    for index, token in enumerate(argv):
        for option_name in option_names:
            prefix = f"{option_name}="
            if token == option_name and index + 1 < len(argv):
                display = str(argv[index + 1]).strip()
                if display:
                    os.environ["DISPLAY"] = display
                    return display
            if token.startswith(prefix):
                display = str(token[len(prefix) :]).strip()
                if display:
                    os.environ["DISPLAY"] = display
                    return display
    return None


def cli_has_flag(argv: Sequence[str] | None = None, *, flag_names: Sequence[str]) -> bool:
    argv = set(sys.argv[1:] if argv is None else argv)
    return any(flag_name in argv for flag_name in flag_names)


def configure_mujoco_gl_backend(*, prefer_headless: bool = False) -> str:
    backend = str(os.environ.get("MUJOCO_GL", "")).strip().lower()
    if backend:
        return backend

    if prefer_headless:
        backend = "egl"
    elif _can_open_x_display(os.environ.get("DISPLAY")):
        backend = "glfw"
    else:
        backend = "egl"
    os.environ["MUJOCO_GL"] = backend
    return backend


def prepare_mujoco_runtime(
    argv: Sequence[str] | None = None,
    *,
    headless_flag_names: Sequence[str] = ("--headless",),
    display_option_names: Sequence[str] = ("--display",),
) -> str:
    apply_display_from_argv(argv, option_names=display_option_names)
    prefer_headless = cli_has_flag(argv, flag_names=headless_flag_names)
    return configure_mujoco_gl_backend(prefer_headless=prefer_headless)


def require_mujoco_viewer_backend() -> None:
    backend = configure_mujoco_gl_backend(prefer_headless=False)
    if backend != "glfw":
        raise RuntimeError(
            "Interactive MuJoCo viewer requires MUJOCO_GL=glfw. "
            f"Current backend is {backend!r}. Set DISPLAY to a live X server or run in headless mode."
        )

    display = str(os.environ.get("DISPLAY", "")).strip()
    if not _can_open_x_display(display):
        raise RuntimeError(
            "Interactive MuJoCo viewer requires a reachable X display. "
            f"Current DISPLAY is {display or '<unset>'!r}."
        )


def _can_open_x_display(display: str | None) -> bool:
    display = str(display or "").strip()
    if not display:
        return False

    library_name = ctypes.util.find_library("X11")
    if not library_name:
        return True
    try:
        x11 = ctypes.CDLL(library_name)
    except OSError:
        return True

    x11.XOpenDisplay.argtypes = [ctypes.c_char_p]
    x11.XOpenDisplay.restype = ctypes.c_void_p
    x11.XCloseDisplay.argtypes = [ctypes.c_void_p]
    x11.XCloseDisplay.restype = ctypes.c_int

    handle = x11.XOpenDisplay(display.encode("utf-8"))
    if not handle:
        return False
    x11.XCloseDisplay(handle)
    return True
