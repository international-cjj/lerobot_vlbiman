#!/usr/bin/env python

from __future__ import annotations

import argparse
import logging
import sys
import time
from contextlib import contextmanager
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_ROOTS = [
    REPO_ROOT / "lerobot_robot_cjjarm",
]
for path in [REPO_ROOT, REPO_ROOT / "src", *PLUGIN_ROOTS]:
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from lerobot.teleoperators.keyboard.configuration_keyboard import KeyboardPoseTeleopConfig
from lerobot.teleoperators.keyboard.teleop_keyboard import KeyboardPoseTeleop
from lerobot_robot_cjjarm.cjjarm_robot import CjjArm
from lerobot_robot_cjjarm.config_cjjarm import CjjArmConfig


@contextmanager
def _terminal_keyboard_mode():
    if sys.platform == "win32" or not sys.stdin.isatty():
        yield
        return

    import termios

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~(termios.ECHO | termios.ICANON)
    new_settings[6][termios.VMIN] = 1
    new_settings[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, new_settings)
    try:
        yield
    finally:
        termios.tcflush(fd, termios.TCIFLUSH)
        termios.tcsetattr(fd, termios.TCSANOW, old_settings)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Teleoperate CJJ arm in 6DoF using keyboard.")
    parser.add_argument("--robot-type", default="cjjarm", choices=["cjjarm", "cjjarm_sim"])
    parser.add_argument("--robot-port", default=None, help="Serial port for the CJJ arm (e.g. /dev/ttyACM0).")
    parser.add_argument(
        "--viewer",
        action="store_true",
        help="Enable MuJoCo viewer window when using --robot-type cjjarm_sim.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Disable MuJoCo viewer window (headless mode) when using --robot-type cjjarm_sim.",
    )
    parser.add_argument("--rate-hz", type=float, default=30.0, help="Teleop loop frequency.")
    parser.add_argument("--pos-step", type=float, default=0.01, help="Delta translation per tick (meters).")
    parser.add_argument("--rot-step", type=float, default=0.05, help="Delta rotation per tick (radians).")
    parser.add_argument("--no-gripper", action="store_true", help="Disable gripper commands.")
    parser.add_argument("--use-cameras", action="store_true", help="Enable cameras defined in the robot config.")
    parser.add_argument(
        "--target-xyz",
        type=float,
        nargs=3,
        default=(0.45, 0.0, 0.08),
        metavar=("X", "Y", "Z"),
        help="Collision-test object position in the unified MuJoCo scene.",
    )
    parser.add_argument("--target-radius-m", type=float, default=0.022, help="Collision-test object radius in meters.")
    parser.add_argument("--target-mass-kg", type=float, default=0.12, help="Collision-test object mass in kg.")
    parser.add_argument(
        "--legacy-scene",
        action="store_true",
        help="Use the old arm-only MuJoCo model instead of the dual-camera collision scene.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=0.0,
        help="Optional max run duration in seconds. 0 means run until Ctrl+C.",
    )
    parser.add_argument("--log-level", default="INFO", help="Logging level.")
    return parser.parse_args()


def _build_robot(args: argparse.Namespace):
    if args.robot_type == "cjjarm_sim":
        from lerobot_robot_cjjarm.cjjarm_sim import CjjArmSim
        from lerobot_robot_cjjarm.config_cjjarm_sim import CjjArmSimConfig
        cfg = CjjArmSimConfig()
        # Default to showing the viewer in sim mode unless explicitly disabled.
        cfg.use_viewer = bool(args.viewer) or (not args.no_viewer)
        cfg.scene_profile = "legacy" if bool(args.legacy_scene) else "dual_camera_target"
        cfg.legacy_raw_urdf_enabled = bool(args.legacy_scene)
        cfg.scene_target_x = float(args.target_xyz[0])
        cfg.scene_target_y = float(args.target_xyz[1])
        cfg.scene_target_z = float(args.target_xyz[2])
        cfg.scene_target_radius_m = float(args.target_radius_m)
        cfg.scene_target_mass_kg = float(args.target_mass_kg)
        return CjjArmSim(cfg)

    cfg = CjjArmConfig()
    if args.robot_port:
        cfg.serial_port = args.robot_port
    if not args.use_cameras:
        cfg.cameras = {}
    return CjjArm(cfg)


def main() -> None:
    args = _parse_args()
    # Force logging setup so startup status is visible even if other modules configured logging first.
    logging.basicConfig(level=getattr(logging, args.log_level.upper(), logging.INFO), force=True)

    robot = _build_robot(args)
    teleop_cfg = KeyboardPoseTeleopConfig(
        pos_step=args.pos_step,
        rot_step=args.rot_step,
        use_gripper=not args.no_gripper,
    )
    teleop = KeyboardPoseTeleop(teleop_cfg)

    logger = logging.getLogger(__name__)
    try:
        teleop.connect()
        if not teleop.is_connected:
            raise RuntimeError("Keyboard teleop is not available (pynput missing or no DISPLAY).")
        if args.robot_type == "cjjarm":
            logger.info("Connecting robot on %s", getattr(robot.config, "serial_port", ""))
        else:
            logger.info("Connecting robot (sim)")
        robot.connect()
    except Exception:
        if teleop.is_connected:
            teleop.disconnect()
        raise

    period = 1.0 / max(args.rate_hz, 1e-3)
    next_time = time.perf_counter()
    start_time = next_time
    logger.info("Teleop loop running at %.1f Hz. Press Ctrl+C to stop.", args.rate_hz)
    if args.duration_s > 0:
        logger.info("Auto-stop enabled: %.2f seconds.", args.duration_s)

    try:
        with _terminal_keyboard_mode():
            while True:
                if args.duration_s > 0 and (time.perf_counter() - start_time) >= args.duration_s:
                    logger.info("Reached --duration-s limit; stopping teleop loop.")
                    break
                action = teleop.get_action()
                robot.send_action(action)
                next_time += period
                sleep_s = next_time - time.perf_counter()
                if sleep_s > 0:
                    time.sleep(sleep_s)
    except KeyboardInterrupt:
        logger.info("Stopping teleop loop.")
    finally:
        try:
            if teleop.is_connected:
                teleop.disconnect()
        finally:
            robot.disconnect()


if __name__ == "__main__":
    main()
