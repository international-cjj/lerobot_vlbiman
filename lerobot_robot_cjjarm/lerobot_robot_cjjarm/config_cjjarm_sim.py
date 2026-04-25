from dataclasses import dataclass, field
from pathlib import Path

from lerobot.robots.config import RobotConfig


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _default_urdf_path() -> str:
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / "cjjarm_urdf" / "TRLC-DK1-Follower.urdf")


def _default_base_mjcf_path() -> str:
    base_dir = Path(__file__).resolve().parent
    return str(base_dir / "cjjarm_urdf" / "TRLC-DK1-Follower-home.mjcf")


def _default_handeye_result_path() -> str:
    return str(_repo_root() / "outputs" / "vlbiman_sa" / "calib" / "handeye_result.json")


def _default_generated_scene_dir() -> str:
    return str(_repo_root() / "outputs" / "vlbiman_sa" / "cjjarm_sim")


@RobotConfig.register_subclass("cjjarm_sim")
@dataclass
class CjjArmSimConfig(RobotConfig):
    urdf_path: str = field(default_factory=_default_urdf_path)
    mujoco_model_path: str = ""
    end_effector_frame: str = "tool0"

    joint_action_order: list[str] = field(
        default_factory=lambda: [
            "joint_1",
            "joint_2",
            "joint_3",
            "joint_4",
            "joint_5",
            "joint_6",
        ]
    )
    urdf_joint_map: dict[str, str] = field(
        default_factory=lambda: {
            "joint_1": "joint1",
            "joint_2": "joint2",
            "joint_3": "joint3",
            "joint_4": "joint4",
            "joint_5": "joint5",
            "joint_6": "joint6",
        }
    )
    joint_directions: dict[str, float] = field(
        default_factory=lambda: {
            "joint_1": 1.0,
            "joint_2": 1.0,
            # Keep joint_3 positive in sim so raising motion is available from home pose.
            "joint_3": 1.0,
            "joint_4": 1.0,
            "joint_5": 1.0,
            "joint_6": 1.0,
            "gripper": 1.0,
        }
    )

    render_width: int = 480
    render_height: int = 480
    max_delta_per_step: float = 0.1
    action_substeps: int = 20
    gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
    use_viewer: bool = False
    render_camera_name: str = "front_camera"
    front_camera_observation_key: str = "observation.images.front"
    wrist_camera_observation_key: str = "observation.images.wrist"

    scene_profile: str = "dual_camera_target"
    scene_preset: str = "default"
    scene_base_mjcf_path: str = field(default_factory=_default_base_mjcf_path)
    scene_handeye_result_path: str = field(default_factory=_default_handeye_result_path)
    scene_generated_mjcf_dir: str = field(default_factory=_default_generated_scene_dir)
    scene_camera_fovy_deg: float = 58.0
    scene_settle_steps: int = 120
    scene_target_x: float = 0.45
    scene_target_y: float = 0.0
    scene_target_z: float = 0.08
    scene_target_radius_m: float = 0.022
    scene_target_mass_kg: float = 0.12
    legacy_raw_urdf_enabled: bool = False

    use_gripper: bool = True
    gripper_open_pos: float = 0.001
    gripper_closed_pos: float = -0.045
    gripper_min_pos: float = -0.045
    gripper_max_pos: float = 0.001
    gripper_max_delta_per_step: float = 0.008
