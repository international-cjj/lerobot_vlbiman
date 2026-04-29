import sys

from .config_cjjarm import (
    CjjArmConfig,
    dabaidcw_rgb_camera_config,
    default_cjjarm_cameras_config,
    wrist_rgb_camera_config,
)
from .cjjarm_robot import CjjArm

__all__ = [
    "CjjArm",
    "CjjArmConfig",
    "dabaidcw_rgb_camera_config",
    "default_cjjarm_cameras_config",
    "wrist_rgb_camera_config",
]

try:
    from .config_cjjarm_sim import CjjArmSimConfig
    from .cjjarm_sim import CjjArmSim
except ImportError:
    CjjArmSimConfig = None
    CjjArmSim = None
else:
    __all__ += ["CjjArmSim", "CjjArmSimConfig"]


def _alias_legacy_nested_modules() -> None:
    if __name__ != "lerobot_robot_cjjarm":
        return
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
