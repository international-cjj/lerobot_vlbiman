from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any


@dataclass(slots=True)
class TraceLoggerConfig:
    output_dir: Path
    filename: str = "trace.jsonl"


class TraceLogger:
    def __init__(self, config: TraceLoggerConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.config.output_dir / self.config.filename

    def write_event(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {"event_type": event_type, "payload": payload}
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    def read_events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            return []
        return [json.loads(line) for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]
