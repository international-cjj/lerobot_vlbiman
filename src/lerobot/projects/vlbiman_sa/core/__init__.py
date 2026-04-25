from .contracts import TaskGraspConfig

__all__ = ["GraspOrchestrator", "TaskGraspConfig"]


def __getattr__(name: str):
    if name == "GraspOrchestrator":
        from .grasp_orchestrator import GraspOrchestrator

        globals()["GraspOrchestrator"] = GraspOrchestrator
        return GraspOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
