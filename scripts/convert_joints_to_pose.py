#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from lerobot.datasets.compute_stats import aggregate_stats, get_feature_stats
from lerobot.utils.constants import HF_LEROBOT_HOME
from lerobot_robot_cjjarm.kinematics import CjjArmKinematics


DEFAULT_URDF_JOINT_MAP = {
    "joint_1": "joint1",
    "joint_2": "joint2",
    "joint_3": "joint3",
    "joint_4": "joint4",
    "joint_5": "joint5",
    "joint_6": "joint6",
}
DEFAULT_POSE_NAMES = ["ee.x", "ee.y", "ee.z", "ee.rx", "ee.ry", "ee.rz"]


def _load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert joint-space actions to EE pose actions.")
    parser.add_argument("--repo-id", required=True, help="Input dataset repo id (local dataset name).")
    parser.add_argument("--root", default=None, help="Root directory containing datasets.")
    parser.add_argument("--output-repo-id", default=None, help="Output dataset repo id.")
    parser.add_argument("--output-root", default=None, help="Root directory for output dataset.")
    parser.add_argument("--urdf-path", default=None, help="URDF path for FK.")
    parser.add_argument("--end-effector-frame", default="tool0", help="End-effector frame name.")
    parser.add_argument(
        "--urdf-joint-map",
        default=None,
        help="JSON dict mapping dataset joint names to URDF joint names.",
    )
    parser.add_argument(
        "--pose-names",
        default=None,
        help="JSON list for pose action names, default: ee.x, ee.y, ee.z, ee.rx, ee.ry, ee.rz.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    root = Path(args.root) if args.root else HF_LEROBOT_HOME
    input_dir = root / args.repo_id
    if not input_dir.exists():
        raise FileNotFoundError(f"Input dataset not found: {input_dir}")

    output_root = Path(args.output_root) if args.output_root else root
    output_repo_id = args.output_repo_id or f"{args.repo_id}_ee"
    output_dir = output_root / output_repo_id
    if output_dir.exists():
        raise FileExistsError(f"Output dataset already exists: {output_dir}")

    shutil.copytree(input_dir, output_dir)

    info_path = output_dir / "meta" / "info.json"
    info = _load_json(info_path)
    if "action" not in info["features"]:
        raise ValueError("Dataset has no 'action' feature to convert.")

    urdf_joint_map = DEFAULT_URDF_JOINT_MAP
    if args.urdf_joint_map:
        urdf_joint_map = json.loads(args.urdf_joint_map)

    pose_names = DEFAULT_POSE_NAMES
    if args.pose_names:
        pose_names = json.loads(args.pose_names)

    action_names = info["features"]["action"]["names"]
    joint_action_names = [f"{name}.pos" for name in urdf_joint_map.keys()]
    joint_indices = []
    for name in joint_action_names:
        if name not in action_names:
            raise ValueError(f"Joint action '{name}' not found in dataset action names: {action_names}")
        joint_indices.append(action_names.index(name))

    urdf_path = args.urdf_path or str(
        (Path(__file__).resolve().parent.parent / "lerobot_robot_cjjarm" / "lerobot_robot_cjjarm" / "cjjarm_urdf" / "TRLC-DK1-Follower.urdf")
    )
    kinematics = CjjArmKinematics(
        urdf_path=urdf_path,
        end_effector_frame=args.end_effector_frame,
        joint_names=list(urdf_joint_map.values()),
    )

    data_dir = output_dir / "data"
    data_paths = sorted(data_dir.glob("**/*.parquet"))
    if not data_paths:
        raise FileNotFoundError(f"No data parquet files found under {data_dir}")

    per_file_stats = []
    per_episode_stats: dict[int, dict[str, np.ndarray]] = {}

    for data_path in data_paths:
        df = pd.read_parquet(data_path)
        if "action" not in df.columns:
            continue

        action_array = np.stack(df["action"].to_numpy())
        arm_actions = action_array[:, joint_indices]
        pose_actions = np.array([kinematics.compute_fk(row) for row in arm_actions], dtype=float)
        df["action"] = list(pose_actions)
        df.to_parquet(data_path, index=False)

        per_file_stats.append({"action": get_feature_stats(pose_actions, axis=0, keepdims=False)})

        for ep_idx, ep_df in df.groupby("episode_index"):
            ep_actions = np.stack(ep_df["action"].to_numpy())
            per_episode_stats[int(ep_idx)] = get_feature_stats(ep_actions, axis=0, keepdims=False)

    if per_file_stats:
        agg_stats = aggregate_stats(per_file_stats)
        stats_path = output_dir / "meta" / "stats.json"
        if stats_path.exists():
            stats = _load_json(stats_path)
        else:
            stats = {}
        stats["action"] = {k: v.tolist() for k, v in agg_stats["action"].items()}
        _write_json(stats_path, stats)

    episodes_dir = output_dir / "meta" / "episodes"
    for ep_path in sorted(episodes_dir.glob("**/*.parquet")):
        ep_df = pd.read_parquet(ep_path)
        if "episode_index" not in ep_df.columns:
            continue
        for row_idx, row in ep_df.iterrows():
            ep_idx = int(row["episode_index"])
            if ep_idx not in per_episode_stats:
                continue
            for stat_key, value in per_episode_stats[ep_idx].items():
                col = f"stats/action/{stat_key}"
                ep_df.at[row_idx, col] = value.tolist()
        ep_df.to_parquet(ep_path, index=False)

    info["features"]["action"]["shape"] = [6]
    info["features"]["action"]["names"] = pose_names
    _write_json(info_path, info)

    print(f"Converted dataset saved to {output_dir}")


if __name__ == "__main__":
    main()
