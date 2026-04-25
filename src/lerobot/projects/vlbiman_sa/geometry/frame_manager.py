from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import yaml

from .transforms import apply_transform_points, compose_transform, invert_transform


def _edge_key(target: str, source: str) -> str:
    return f"{target}__from__{source}"


def _parse_edge_key(edge_key: str) -> tuple[str, str]:
    parts = edge_key.split("__from__")
    if len(parts) != 2:
        raise ValueError(f"Invalid transform key '{edge_key}'. Expected '<target>__from__<source>'.")
    return parts[0], parts[1]


@dataclass
class FrameManager:
    transforms: dict[tuple[str, str], np.ndarray] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "FrameManager":
        yaml_path = Path(yaml_path)
        if not yaml_path.exists():
            return cls()
        payload = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
        transform_entries = payload.get("transforms", {})
        transforms: dict[tuple[str, str], np.ndarray] = {}
        for raw_key, raw_matrix in transform_entries.items():
            target, source = _parse_edge_key(raw_key)
            matrix = np.asarray(raw_matrix, dtype=float)
            if matrix.shape != (4, 4):
                raise ValueError(f"Transform '{raw_key}' must be 4x4, got {matrix.shape}.")
            transforms[(target, source)] = matrix
        return cls(transforms=transforms)

    def to_yaml(self, yaml_path: str | Path) -> None:
        yaml_path = Path(yaml_path)
        yaml_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "transforms": {
                _edge_key(target, source): matrix.tolist()
                for (target, source), matrix in sorted(self.transforms.items())
            }
        }
        yaml_path.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    def set_transform(self, target: str, source: str, transform: np.ndarray) -> None:
        matrix = np.asarray(transform, dtype=float)
        if matrix.shape != (4, 4):
            raise ValueError(f"Transform must be 4x4, got {matrix.shape}.")
        self.transforms[(target, source)] = matrix

    def known_frames(self) -> set[str]:
        frames: set[str] = set()
        for target, source in self.transforms:
            frames.add(target)
            frames.add(source)
        return frames

    def get_transform(self, target: str, source: str) -> np.ndarray:
        if target == source:
            return np.eye(4, dtype=float)
        direct = self.transforms.get((target, source))
        if direct is not None:
            return direct.copy()

        path = self._find_path(target=target, source=source)
        if path is None:
            raise KeyError(f"Cannot find transform from '{source}' to '{target}'.")

        transform = np.eye(4, dtype=float)
        for hop_target, hop_source in path:
            edge = self.transforms.get((hop_target, hop_source))
            if edge is not None:
                step = edge
            else:
                inverse_edge = self.transforms.get((hop_source, hop_target))
                if inverse_edge is None:
                    raise KeyError(f"Missing edge for hop {hop_target} <- {hop_source}.")
                step = invert_transform(inverse_edge)
            transform = compose_transform(transform, step)
        return transform

    def transform_points(self, target: str, source: str, points_xyz: np.ndarray) -> np.ndarray:
        transform = self.get_transform(target=target, source=source)
        return apply_transform_points(transform, points_xyz)

    def transform_pose(self, target: str, source: str, pose: np.ndarray) -> np.ndarray:
        transform = self.get_transform(target=target, source=source)
        pose = np.asarray(pose, dtype=float)
        if pose.shape != (4, 4):
            raise ValueError(f"Pose must be 4x4, got {pose.shape}")
        return compose_transform(transform, pose)

    def _find_path(self, target: str, source: str) -> list[tuple[str, str]] | None:
        queue: deque[tuple[str, list[tuple[str, str]]]] = deque([(source, [])])
        visited = {source}
        neighbors = self._build_neighbors()

        while queue:
            current, path = queue.popleft()
            if current == target:
                return path
            for nxt in neighbors.get(current, set()):
                if nxt in visited:
                    continue
                visited.add(nxt)
                queue.append((nxt, path + [(nxt, current)]))
        return None

    def _build_neighbors(self) -> dict[str, set[str]]:
        neighbors: dict[str, set[str]] = {}
        for target, source in self.transforms:
            neighbors.setdefault(source, set()).add(target)
            neighbors.setdefault(target, set()).add(source)
        return neighbors

