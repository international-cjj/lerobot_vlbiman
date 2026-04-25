from .config_cjjarm import CjjArmConfig
from .cjjarm_robot import CjjArm

__all__ = ["CjjArm", "CjjArmConfig"]

try:
    from .config_cjjarm_sim import CjjArmSimConfig
    from .cjjarm_sim import CjjArmSim
except ImportError:
    CjjArmSimConfig = None
    CjjArmSim = None
else:
    __all__ += ["CjjArmSim", "CjjArmSimConfig"]
