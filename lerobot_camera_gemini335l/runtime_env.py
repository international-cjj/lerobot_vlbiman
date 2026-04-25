from __future__ import annotations

import sys
from importlib.util import find_spec
from pathlib import Path


def get_python_executable() -> str:
    """Return the active Python executable for copy-paste friendly commands."""
    return sys.executable


def find_udev_install_script() -> str | None:
    """Locate Orbbec's udev installation script from installed pyorbbecsdk package."""
    spec = find_spec("pyorbbecsdk")
    if spec is None or spec.origin is None:
        spec = find_spec("pyorbbecsdk2")
    if spec is None or spec.origin is None:
        return None

    package_dir = Path(spec.origin).resolve().parent
    candidate = package_dir / "shared" / "install_udev_rules.sh"
    if candidate.exists():
        return str(candidate)
    return None


def build_diag_command_hint() -> str:
    python_bin = get_python_executable()
    return (
        f"PYTHONPATH=src:. {python_bin} "
        "-m lerobot_camera_gemini335l.diagnose_depth_startup "
        "--serial-number-or-name <YOUR_CAMERA_SERIAL> "
        "--width 640 --height 400 --fps 30 "
        "--color-stream-format MJPG --depth-stream-format Y16 "
        "--align-mode sw --align-depth-to-color"
    )
