from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from pathlib import Path
from typing import Any


THETA_PARAMETER_NAMES = (
    "advance_speed_mps",
    "preclose_distance_m",
    "close_speed_raw_per_s",
    "settle_time_s",
    "lift_height_m",
)

MAPPABLE_THETA_PARAMETER_NAMES = (
    "preclose_distance_m",
    "close_speed_raw_per_s",
    "settle_time_s",
    "lift_height_m",
)

_POSITIVE_PARAMETER_NAMES = frozenset({"advance_speed_mps", "close_speed_raw_per_s"})
_NON_NEGATIVE_PARAMETER_NAMES = frozenset({"preclose_distance_m", "settle_time_s", "lift_height_m"})


def _coerce_theta_value(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric, got {type(value).__name__}.")
    numeric = float(value)
    if not math.isfinite(numeric):
        raise ValueError(f"{name} must be finite, got {value!r}.")
    if name in _POSITIVE_PARAMETER_NAMES and numeric <= 0.0:
        raise ValueError(f"{name} must be > 0, got {numeric}.")
    if name in _NON_NEGATIVE_PARAMETER_NAMES and numeric < 0.0:
        raise ValueError(f"{name} must be >= 0, got {numeric}.")
    return numeric


@dataclass(frozen=True)
class ThetaParameterSource:
    name: str
    value: float
    source: str
    default_used: bool = False
    reason: str | None = None
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "value", _coerce_theta_value(self.name, self.value))
        if not isinstance(self.source, str) or not self.source:
            raise ValueError("source must be a non-empty string.")
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": self.source,
            "default_used": self.default_used,
            "reason": self.reason,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class DemoTheta:
    advance_speed_mps: float
    preclose_distance_m: float
    close_speed_raw_per_s: float
    settle_time_s: float
    lift_height_m: float

    def __post_init__(self) -> None:
        for name in THETA_PARAMETER_NAMES:
            object.__setattr__(self, name, _coerce_theta_value(name, getattr(self, name)))

    def to_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in THETA_PARAMETER_NAMES}

    @classmethod
    def from_mapping(cls, payload: dict[str, Any]) -> "DemoTheta":
        if not isinstance(payload, dict):
            raise ValueError(f"theta payload must be a mapping, got {type(payload).__name__}.")
        values = {}
        for name in THETA_PARAMETER_NAMES:
            if name not in payload:
                raise ValueError(f"Missing theta parameter: {name}")
            values[name] = _coerce_theta_value(name, payload[name])
        return cls(**values)


def load_theta_samples(path: str | Path) -> DemoTheta:
    theta_path = Path(path).resolve()
    if not theta_path.exists():
        raise FileNotFoundError(f"Theta sample file not found: {theta_path}")
    payload = json.loads(theta_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"theta sample file must contain a JSON object, got {type(payload).__name__}.")
    if isinstance(payload.get("theta"), dict):
        payload = payload["theta"]
    return DemoTheta.from_mapping(payload)


__all__ = [
    "DemoTheta",
    "MAPPABLE_THETA_PARAMETER_NAMES",
    "THETA_PARAMETER_NAMES",
    "ThetaParameterSource",
    "load_theta_samples",
]
