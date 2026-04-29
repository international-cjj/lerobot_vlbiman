from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent / "lerobot_robot_cjjarm"
__path__ = [str(_PACKAGE_DIR)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

from .config_cjjarm import (
    CjjArmConfig,
    dabaidcw_rgb_camera_config,
    default_cjjarm_cameras_config,
    wrist_rgb_camera_config,
)
from .cjjarm_robot import CjjArm

try:
    from .cjjarm_sim import CjjArmSim
    from .config_cjjarm_sim import CjjArmSimConfig
except ImportError:
    CjjArmSim = None
    CjjArmSimConfig = None


def _alias_legacy_nested_modules() -> None:
    legacy_prefix = f"{__name__}.lerobot_robot_cjjarm"
    sys.modules.setdefault(legacy_prefix, sys.modules[__name__])
    for suffix in (
        "DM_CAN",
        "cjjarm_robot",
        "cjjarm_sim",
        "config_cjjarm",
        "config_cjjarm_sim",
        "kinematics",
    ):
        module = sys.modules.get(f"{__name__}.{suffix}")
        if module is not None:
            sys.modules.setdefault(f"{legacy_prefix}.{suffix}", module)


_alias_legacy_nested_modules()

__all__ = [
    "CjjArm",
    "CjjArmConfig",
    "dabaidcw_rgb_camera_config",
    "default_cjjarm_cameras_config",
    "wrist_rgb_camera_config",
]

if CjjArmSim is not None and CjjArmSimConfig is not None:
    __all__ += ["CjjArmSim", "CjjArmSimConfig"]
