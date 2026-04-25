from .lerobot_robot_cjjarm import CjjArm, CjjArmConfig

__all__ = ["CjjArm", "CjjArmConfig"]

try:
    from .lerobot_robot_cjjarm import CjjArmSim, CjjArmSimConfig
except ImportError:
    CjjArmSim = None
    CjjArmSimConfig = None
else:
    __all__ += ["CjjArmSim", "CjjArmSimConfig"]
