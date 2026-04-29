# lerobot_teleoperator_zhonglin/__init__.py
import sys

from .config_zhonglin import ZhonglinTeleopConfig
from .zhonglin_teleop import ZhonglinTeleop

__all__ = ["ZhonglinTeleop", "ZhonglinTeleopConfig"]


def _alias_legacy_nested_modules() -> None:
    if __name__ != "lerobot_teleoperator_zhonglin":
        return
    legacy_prefix = f"{__name__}.lerobot_teleoperator_zhonglin"
    sys.modules.setdefault(legacy_prefix, sys.modules[__name__])
    for suffix in ("config_zhonglin", "zhonglin_teleop"):
        module = sys.modules.get(f"{__name__}.{suffix}")
        if module is not None:
            sys.modules.setdefault(f"{legacy_prefix}.{suffix}", module)


_alias_legacy_nested_modules()
