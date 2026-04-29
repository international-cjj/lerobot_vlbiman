from pathlib import Path
import sys

_PACKAGE_DIR = Path(__file__).resolve().parent / "lerobot_teleoperator_zhonglin"
__path__ = [str(_PACKAGE_DIR)]
if __spec__ is not None:
    __spec__.submodule_search_locations = __path__

from .config_zhonglin import ZhonglinTeleopConfig
from .zhonglin_teleop import ZhonglinTeleop


def _alias_legacy_nested_modules() -> None:
    legacy_prefix = f"{__name__}.lerobot_teleoperator_zhonglin"
    sys.modules.setdefault(legacy_prefix, sys.modules[__name__])
    for suffix in ("config_zhonglin", "zhonglin_teleop"):
        module = sys.modules.get(f"{__name__}.{suffix}")
        if module is not None:
            sys.modules.setdefault(f"{legacy_prefix}.{suffix}", module)


_alias_legacy_nested_modules()

__all__ = ["ZhonglinTeleop", "ZhonglinTeleopConfig"]
