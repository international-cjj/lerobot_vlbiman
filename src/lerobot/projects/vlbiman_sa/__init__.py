"""VLBiMan single-arm project package."""

from __future__ import annotations

from typing import Any

__all__ = ["GraspOrchestrator", "TaskGraspConfig"]


def __getattr__(name: str) -> Any:
    if name in {"GraspOrchestrator", "TaskGraspConfig"}:
        from .core.grasp_orchestrator import GraspOrchestrator
        from .core.contracts import TaskGraspConfig

        exports = {
            "GraspOrchestrator": GraspOrchestrator,
            "TaskGraspConfig": TaskGraspConfig,
        }
        return exports[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
