# simulation_audit_report

## 1. 当前工程加载链路总结

结论：当前项目不是“只加载 URDF”或“只加载 MJCF”。

- `cjjarm_sim` 默认配置把场景基座设为 [config_cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/config_cjjarm_sim.py#L74) 中的 `scene_profile="dual_camera_target"`，基础模型来自 [config_cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/config_cjjarm_sim.py#L75) 指向的 `TRLC-DK1-Follower-home.mjcf`，同时还保留 [config_cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/config_cjjarm_sim.py#L28) 的 `TRLC-DK1-Follower.urdf`。
- `CjjArmSim._load_model_from_config()` 在默认 `dual_camera_target` 下会读取 base MJCF、手眼结果和 URDF 里的 `camera_joint`，再调用 `build_dual_camera_scene(...)` 生成一份新的 MJCF 字符串并 `MjModel.from_xml_string(...)` 加载；只有显式指定模型路径，或者不走 `dual_camera_target` 时，才会退回到直接 `MjModel.from_xml_path(urdf_path)`。依据：[cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L108)。
- `run_visual_closed_loop_validation.py` 和 `play_mujoco_trajectory.py` 的实际链路与上面一致：`legacy_scene` 直接读 base MJCF；默认则是 `build_dual_camera_scene(...) -> MjModel.from_xml_string(...)`。依据：[run_visual_closed_loop_validation.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py#L269)，[play_mujoco_trajectory.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py#L352)。
- `CjjArmSim` 本身不是 actuator 驱动仿真，而是直接改 `data.qpos` 再 `mj_step` 的运动学式控制。依据：[cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L404)。

因此，当前工程的真实仿真加载链路是：

1. 交互/验证入口脚本
2. `CjjArmSimConfig`
3. `CjjArmSim`
4. `TRLC-DK1-Follower-home.mjcf` 作为 base scene
5. `TRLC-DK1-Follower.urdf` 仅用于运动学与腕部相机安装位姿提取
6. `build_dual_camera_scene(...)` 生成运行时 MJCF
7. MuJoCo `MjModel/MjData`

补充：`legacy` 模式不是加载 `TRLC-DK1-Follower-home.mjcf`，而是直接回退到 `TRLC-DK1-Follower.urdf`。依据：[cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L164)。

当前可以判断的主要仿真启动命令如下。它们来自 [遥控脚本.md](/home/cjj/lerobot_vlbiman/运行脚本/遥控脚本.md#L62)、[遥控脚本.md](/home/cjj/lerobot_vlbiman/运行脚本/遥控脚本.md#L73)、[遥控脚本.md](/home/cjj/lerobot_vlbiman/运行脚本/遥控脚本.md#L193) 和 [遥控脚本.md](/home/cjj/lerobot_vlbiman/运行脚本/遥控脚本.md#L149)：

```bash
PYTHONPATH=src:. "$PY" scripts/teleop_cjjarm_keyboard.py --robot-type cjjarm_sim

PYTHONPATH=src:. "$PY" scripts/teleop_cjjarm_zhonglin.py --robot-type cjjarm_sim --teleop-port /dev/ttyUSB0

HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISPLAY=:1 "$PY" src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py --task-config <task.yaml> --display :1 --aux-target-phrase "pink cup"

DISPLAY=:1 "$PY" src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py --demo-session-dir <demo_dir> --orange-base-xyz <x y z> --pink-cup-base-xyz <x y z> --display :1
```

## 2. 文件结构检查

### 2.1 资产清单

- URDF:
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf`
- Xacro:
  - _None found_
- XML:
  - _None found in source tree_
- 源 MJCF:
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf`
- 生成 MJCF:
  - `outputs/vlbiman_sa/mujoco_dual_camera_scene/.../dual_camera_target_scene.mjcf`
  - `outputs/vlbiman_sa/visual_pickorange_true_closed_loop/.../pickorange_true_closed_loop_scene.mjcf`
- Mesh（`.stl/.dae/.obj`）:
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/base_link.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link1-2.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link2-3.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link3-4.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link4-5.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link5-6.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/link6-7.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/finger_left.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/finger_right.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/base_link.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link1-2.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link2-3.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link3-4.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link4-5.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link5-6.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/link6-7.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/finger_left.stl`
  - `lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/meshes/collision/finger_right.stl`

### 2.2 关键 Python 文件

完整关键词命中清单有 `506` 个 Python 文件，已经保存到 [tools/audit/simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。本次实际审计的主文件是：

- [lerobot_robot_cjjarm/lerobot_robot_cjjarm/config_cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/config_cjjarm_sim.py)
- [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py)
- [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_robot.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_robot.py)
- [src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py)
- [src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py)
- [src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py)
- [src/lerobot/projects/vlbiman_sa/app/run_frrg_mujoco_validate.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/run_frrg_mujoco_validate.py)
- [src/lerobot/rl/gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py)
- [src/lerobot/envs/configs.py](/home/cjj/lerobot_vlbiman/src/lerobot/envs/configs.py)
- [src/lerobot/envs/factory.py](/home/cjj/lerobot_vlbiman/src/lerobot/envs/factory.py)
- [src/lerobot/envs/utils.py](/home/cjj/lerobot_vlbiman/src/lerobot/envs/utils.py)
- [scripts/teleop_cjjarm_keyboard.py](/home/cjj/lerobot_vlbiman/scripts/teleop_cjjarm_keyboard.py)
- [scripts/teleop_cjjarm_zhonglin.py](/home/cjj/lerobot_vlbiman/scripts/teleop_cjjarm_zhonglin.py)

## 3. 发现的 URDF 问题

### 3.1 `TRLC-DK1-Follower.urdf`

来源：[TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf)

链接状态表：

| link name | visual | collision | inertial | visual mesh ok | collision mesh ok | collision kind | mass | inertia status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| base_link | yes | yes | yes | yes | yes | mesh | 0.4455 | present |
| link1-2 | yes | yes | yes | yes | yes | mesh | 0.0886 | present |
| link2-3 | yes | yes | yes | yes | yes | mesh | 1.0327 | present |
| link3-4 | yes | yes | yes | yes | yes | mesh | 0.6073 | present |
| link4-5 | yes | yes | yes | yes | yes | mesh | 0.4232 | present |
| link5-6 | yes | yes | yes | yes | yes | mesh | 0.4214 | present |
| link6-7 | yes | yes | yes | yes | yes | mesh | 0.8506 | present |
| finger_left | yes | yes | yes | yes | yes | mesh | 0.0500 | present |
| finger_right | yes | yes | yes | yes | yes | mesh | 0.0500 | present |
| tool0 | no | no | no | - | - | - | - | missing |
| camera | no | no | no | - | - | - | - | missing |

关节状态表：

| joint name | type | parent | child | axis | limit lower/upper/effort/velocity | dynamics damping/friction | mimic | gripper? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| joint1 | revolute | base_link | link1-2 | 0.0, 0.0, 1.0 | -2.0944..2.0944; eff=100.0000; vel=5.4000 | missing | no | no |
| joint2 | revolute | link1-2 | link2-3 | 0.0, -1.0, 0.0 | -3.1416..0.0100; eff=100.0000; vel=5.4000 | missing | no | no |
| joint3 | revolute | link2-3 | link3-4 | 0.0, -1.0, 0.0 | -0.0100..4.7124; eff=100.0000; vel=5.4000 | missing | no | no |
| joint4 | revolute | link3-4 | link4-5 | 0.0, -1.0, 0.0 | -1.9500..1.5708; eff=20.0000; vel=5.4000 | missing | no | no |
| joint5 | revolute | link4-5 | link5-6 | 0.0, 0.0, -1.0 | -1.5708..1.5708; eff=20.0000; vel=5.4000 | missing | no | no |
| joint6 | revolute | link5-6 | link6-7 | 1.0, 0.0, 0.0 | -3.1416..3.1416; eff=20.0000; vel=5.4000 | missing | no | no |
| gripper_left | prismatic | link6-7 | finger_left | 0.0, 1.0, 0.0 | -0.0450..0.0010; eff=20.0000; vel=0.1800 | missing | no | yes |
| gripper_right | prismatic | link6-7 | finger_right | 0.0, -1.0, 0.0 | -0.0450..0.0010; eff=20.0000; vel=0.1800 | missing | no | yes |
| gripper_tool0 | fixed | link6-7 | tool0 | - | missing | missing | no | yes |
| camera_joint | fixed | link6-7 | camera | - | missing | missing | no | no |

重点结论：

- 根链接是 `base_link`，而且根链接本身带 inertial。依据：[TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L18)。这类结构对 KDL/某些 URDF->MuJoCo 转换路径并不理想。
- 机器人 9 个有碰撞的 link 全部使用 mesh collision，没有任何 box/cylinder/sphere 简化碰撞体。依据：[TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L34) 到 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L374)。
- 夹爪 finger 虽然有独立 collision，但仍然是 STL mesh，缺少面向抓取接触的内侧 box pad。依据：`finger_left` 在 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L324)，`finger_right` 在 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L367)。
- 所有关节都定义了 limit，但没有 `dynamics damping/friction`。例如 `joint1` 在 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L70)，`gripper_left` 在 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L334)。这会让无控制或弱控制时的仿真更容易塌落、振荡或打滑。
- `tool0` 和 `camera` 只有 link/frame，没有 collision 和 inertial。依据：[TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L395)。
- 所有 visual/collision mesh 都没有 `scale`。例如 `base_link` visual/collision 分别在 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L30) 和 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L39)。这不等于一定有单位错误，但如果 STL 是毫米导出，这里没有任何 `0.001` 补偿。
- 夹爪结构是双 prismatic joint，不是 mimic，不是 tendon。依据：[TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L334) 和 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L377)。

### 3.2 `TRLC-DK1-Follower-home.urdf`

来源：[TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf)

链接状态表：

| link name | visual | collision | inertial | visual mesh ok | collision mesh ok | collision kind | mass | inertia status |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| world | no | no | no | - | - | - | - | missing |
| base_link | yes | yes | yes | yes | yes | mesh | 0.4455 | present |
| link1-2 | yes | yes | yes | yes | yes | mesh | 0.0886 | present |
| link2-3 | yes | yes | yes | yes | yes | mesh | 1.0327 | present |
| link3-4 | yes | yes | yes | yes | yes | mesh | 0.6073 | present |
| link4-5 | yes | yes | yes | yes | yes | mesh | 0.4232 | present |
| link5-6 | yes | yes | yes | yes | yes | mesh | 0.4214 | present |
| link6-7 | yes | yes | yes | yes | yes | mesh | 0.8506 | present |
| finger_left | yes | yes | yes | yes | yes | mesh | 0.0500 | present |
| finger_right | yes | yes | yes | yes | yes | mesh | 0.0500 | present |
| tool0 | no | no | no | - | - | - | - | missing |
| camera | no | no | no | - | - | - | - | missing |
| floor | yes | yes | no | - | - | box | - | missing |
| wall_back | yes | no | no | - | - | - | - | missing |
| wall_side | yes | no | no | - | - | - | - | missing |
| rug | yes | no | no | - | - | - | - | missing |
| table_top | yes | yes | no | - | - | box | - | missing |
| table_leg_fl | yes | no | no | - | - | - | - | missing |
| table_leg_fr | yes | no | no | - | - | - | - | missing |
| table_leg_bl | yes | no | no | - | - | - | - | missing |
| table_leg_br | yes | no | no | - | - | - | - | missing |
| cabinet | yes | no | no | - | - | - | - | missing |
| box_object | yes | yes | no | - | - | box | - | missing |
| block_object | yes | yes | no | - | - | box | - | missing |

关节状态表：

| joint name | type | parent | child | axis | limit lower/upper/effort/velocity | dynamics damping/friction | mimic | gripper? |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| world_to_base | fixed | world | base_link | - | missing | missing | no | no |
| joint1 | revolute | base_link | link1-2 | 0.0, 0.0, 1.0 | -2.0944..2.0944; eff=100.0000; vel=5.4000 | missing | no | no |
| joint2 | revolute | link1-2 | link2-3 | 0.0, -1.0, 0.0 | -3.1416..0.0100; eff=100.0000; vel=5.4000 | missing | no | no |
| joint3 | revolute | link2-3 | link3-4 | 0.0, -1.0, 0.0 | -0.0100..4.7124; eff=100.0000; vel=5.4000 | missing | no | no |
| joint4 | revolute | link3-4 | link4-5 | 0.0, -1.0, 0.0 | -1.9500..1.5708; eff=20.0000; vel=5.4000 | missing | no | no |
| joint5 | revolute | link4-5 | link5-6 | 0.0, 0.0, -1.0 | -1.5708..1.5708; eff=20.0000; vel=5.4000 | missing | no | no |
| joint6 | revolute | link5-6 | link6-7 | 1.0, 0.0, 0.0 | -3.1416..3.1416; eff=20.0000; vel=5.4000 | missing | no | no |
| gripper_left | prismatic | link6-7 | finger_left | 0.0, 1.0, 0.0 | -0.0450..0.0010; eff=20.0000; vel=0.1800 | missing | no | yes |
| gripper_right | prismatic | link6-7 | finger_right | 0.0, -1.0, 0.0 | -0.0450..0.0010; eff=20.0000; vel=0.1800 | missing | no | yes |
| gripper_tool0 | fixed | link6-7 | tool0 | - | missing | missing | no | yes |
| camera_joint | fixed | link6-7 | camera | - | missing | missing | no | no |
| world_to_floor | fixed | world | floor | - | missing | missing | no | no |
| world_to_wall_back | fixed | world | wall_back | - | missing | missing | no | no |
| world_to_wall_side | fixed | world | wall_side | - | missing | missing | no | no |
| world_to_rug | fixed | world | rug | - | missing | missing | no | no |
| world_to_table_top | fixed | world | table_top | - | missing | missing | no | no |
| world_to_table_leg_fl | fixed | world | table_leg_fl | - | missing | missing | no | no |
| world_to_table_leg_fr | fixed | world | table_leg_fr | - | missing | missing | no | no |
| world_to_table_leg_bl | fixed | world | table_leg_bl | - | missing | missing | no | no |
| world_to_table_leg_br | fixed | world | table_leg_br | - | missing | missing | no | no |
| world_to_cabinet | fixed | world | cabinet | - | missing | missing | no | no |
| world_to_box | fixed | world | box_object | - | missing | missing | no | no |
| world_to_block | fixed | world | block_object | - | missing | missing | no | no |

重点结论：

- `home.urdf` 通过 `world_to_base` 固定了基座，因此 root link 变成 `world`，规避了 plain URDF 的 root inertial 问题。依据：[TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf#L4)。
- 但这个 home 场景里的房间/家具大多只是 visual，不是完整物理体。`wall_back`、`wall_side`、`rug`、四条桌腿、`cabinet` 都没有 collision；`floor`、`table_top`、`box_object`、`block_object` 虽然有 collision，但全部没有 inertial。依据：`floor` 在 [TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf#L427)，`wall_back` 在 [TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf#L450)，`table_top` 在 [TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf#L501)，`cabinet` 在 [TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf#L592)。
- 直接把 `home.urdf` 当 MuJoCo 场景源并不可靠。实际只读加载测试里，MuJoCo 只生成了 9 个 body 和 13 个 geom，body 名称只有 robot link，没有这些场景物体；这说明你在 URDF 里加的“房间/桌子”不一定会按预期进入 MuJoCo 物理世界。依据：本次加载测试结果保存在 [simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。

## 4. 发现的 MJCF 问题

### 4.1 `TRLC-DK1-Follower.mjcf`

来源：[TRLC-DK1-Follower.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf)

`worldbody`/body 层级：

| body | parent | depth | pos |
| --- | --- | --- | --- |
| link1-2 | <worldbody> | 0 | 0.0, 0.0, 0.0615 |
| link2-3 | link1-2 | 1 | 0.02, 0.0, 0.0385 |
| link3-4 | link2-3 | 2 | -0.264, 0.0, 0.0 |
| link4-5 | link3-4 | 3 | 0.244004, 0.0, 0.059971 |
| link5-6 | link4-5 | 4 | 0.061868, 0.0, 0.0425 |
| link6-7 | link5-6 | 5 | 0.032, 0.0, -0.0425 |
| finger_left | link6-7 | 6 | - |
| finger_right | link6-7 | 6 | - |

joint 列表：

| joint | type | body | axis | range | actuatorfrcrange |
| --- | --- | --- | --- | --- | --- |
| joint1 | hinge | link1-2 | 0.0, 0.0, 1.0 | -2.0944, 2.0944 | -100.0, 100.0 |
| joint2 | hinge | link2-3 | 0.0, -1.0, 0.0 | -3.14159, 0.01 | -100.0, 100.0 |
| joint3 | hinge | link3-4 | 0.0, -1.0, 0.0 | -0.01, 4.71239 | -100.0, 100.0 |
| joint4 | hinge | link4-5 | 0.0, -1.0, 0.0 | -1.95, 1.5708 | -20.0, 20.0 |
| joint5 | hinge | link5-6 | 0.0, 0.0, -1.0 | -1.5708, 1.5708 | -20.0, 20.0 |
| joint6 | hinge | link6-7 | 1.0, 0.0, 0.0 | -3.14159, 3.14159 | -20.0, 20.0 |
| gripper_left | slide | finger_left | 0.0, 1.0, 0.0 | -0.045, 0.001 | -20.0, 20.0 |
| gripper_right | slide | finger_right | 0.0, -1.0, 0.0 | -0.045, 0.001 | -20.0, 20.0 |

geom 列表：

| geom | body | type | size | mesh | mass/density | contype/conaffinity | friction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| - | <worldbody> | mesh | - | base_link | mass=-; density=- | - / - | - |
| - | link1-2 | mesh | - | link1-2 | mass=-; density=- | - / - | - |
| - | link2-3 | mesh | - | link2-3 | mass=-; density=- | - / - | - |
| - | link3-4 | mesh | - | link3-4 | mass=-; density=- | - / - | - |
| - | link4-5 | mesh | - | link4-5 | mass=-; density=- | - / - | - |
| - | link5-6 | mesh | - | link5-6 | mass=-; density=- | - / - | - |
| - | link6-7 | mesh | - | link6-7 | mass=-; density=- | - / - | - |
| - | finger_left | mesh | - | finger_left | mass=-; density=- | - / - | - |
| - | finger_right | mesh | - | finger_right | mass=-; density=- | - / - | - |

mesh asset：

| mesh asset | file | path exists | scale |
| --- | --- | --- | --- |
| base_link | base_link.stl | yes | - |
| link1-2 | link1-2.stl | yes | - |
| link2-3 | link2-3.stl | yes | - |
| link3-4 | link3-4.stl | yes | - |
| link4-5 | link4-5.stl | yes | - |
| link5-6 | link5-6.stl | yes | - |
| link6-7 | link6-7.stl | yes | - |
| finger_left | finger_left.stl | yes | - |
| finger_right | finger_right.stl | yes | - |

camera：

_None_

light：

_None_

重点结论：

- 没有 `<actuator>` 段，`nu=0`。静态解析和 MuJoCo 只读加载都一致。依据：源文件 [TRLC-DK1-Follower.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf) 只有 59 行，且本次 `MjModel` 加载结果见 [simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。
- 没有 `<camera>`，`ncam=0`。这意味着 raw MJCF 本体不能直接给 LeRobot 提供 MuJoCo RGB/depth 相机源。
- 没有显式桌面/目标物体，`interaction_object_names` 为空。它只是机器人本体。
- 机器人所有 geom 仍然是 mesh geom，没有简化接触几何。

### 4.2 `TRLC-DK1-Follower-home.mjcf`

来源：[TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf)

`worldbody`/body 层级：

| body | parent | depth | pos |
| --- | --- | --- | --- |
| arm_base | <worldbody> | 0 | 0.0, 0.0, 0.0 |
| link1-2 | arm_base | 1 | 0.0, 0.0, 0.0615 |
| link2-3 | link1-2 | 2 | 0.02, 0.0, 0.0385 |
| link3-4 | link2-3 | 3 | -0.264, 0.0, 0.0 |
| link4-5 | link3-4 | 4 | 0.244004, 0.0, 0.059971 |
| link5-6 | link4-5 | 5 | 0.061868, 0.0, 0.0425 |
| link6-7 | link5-6 | 6 | 0.032, 0.0, -0.0425 |
| finger_left | link6-7 | 7 | - |
| finger_right | link6-7 | 7 | - |

joint 列表：

| joint | type | body | axis | range | actuatorfrcrange |
| --- | --- | --- | --- | --- | --- |
| joint1 | hinge | link1-2 | 0.0, 0.0, 1.0 | -2.0944, 2.0944 | -100.0, 100.0 |
| joint2 | hinge | link2-3 | 0.0, -1.0, 0.0 | -3.14159, 0.01 | -100.0, 100.0 |
| joint3 | hinge | link3-4 | 0.0, -1.0, 0.0 | -0.01, 4.71239 | -100.0, 100.0 |
| joint4 | hinge | link4-5 | 0.0, -1.0, 0.0 | -1.95, 1.5708 | -20.0, 20.0 |
| joint5 | hinge | link5-6 | 0.0, 0.0, -1.0 | -1.5708, 1.5708 | -20.0, 20.0 |
| joint6 | hinge | link6-7 | 1.0, 0.0, 0.0 | -3.14159, 3.14159 | -20.0, 20.0 |
| gripper_left | slide | finger_left | 0.0, 1.0, 0.0 | -0.045, 0.001 | -20.0, 20.0 |
| gripper_right | slide | finger_right | 0.0, -1.0, 0.0 | -0.045, 0.001 | -20.0, 20.0 |

geom 列表：

| geom | body | type | size | mesh | mass/density | contype/conaffinity | friction |
| --- | --- | --- | --- | --- | --- | --- | --- |
| floor | <worldbody> | box | 2.0, 2.0, 0.02 | - | mass=-; density=- | - / - | - |
| - | arm_base | mesh | - | base_link | mass=-; density=- | - / - | - |
| - | link1-2 | mesh | - | link1-2 | mass=-; density=- | - / - | - |
| - | link2-3 | mesh | - | link2-3 | mass=-; density=- | - / - | - |
| - | link3-4 | mesh | - | link3-4 | mass=-; density=- | - / - | - |
| - | link4-5 | mesh | - | link4-5 | mass=-; density=- | - / - | - |
| - | link5-6 | mesh | - | link5-6 | mass=-; density=- | - / - | - |
| - | link6-7 | mesh | - | link6-7 | mass=-; density=- | - / - | - |
| - | finger_left | mesh | - | finger_left | mass=-; density=- | - / - | - |
| - | finger_right | mesh | - | finger_right | mass=-; density=- | - / - | - |

mesh asset：

| mesh asset | file | path exists | scale |
| --- | --- | --- | --- |
| base_link | base_link.stl | yes | - |
| link1-2 | link1-2.stl | yes | - |
| link2-3 | link2-3.stl | yes | - |
| link3-4 | link3-4.stl | yes | - |
| link4-5 | link4-5.stl | yes | - |
| link5-6 | link5-6.stl | yes | - |
| link6-7 | link6-7.stl | yes | - |
| finger_left | finger_left.stl | yes | - |
| finger_right | finger_right.stl | yes | - |

camera：

_None_

light：

| light | body | pos | dir |
| --- | --- | --- | --- |
| key_light | <worldbody> | 1.5, -1.0, 2.5 | -0.5, 0.3, -1.0 |
| fill_light | <worldbody> | -1.0, 1.0, 1.8 | 0.5, -0.2, -1.0 |

option/default：

- `timestep=0.002`, `gravity=0 0 -9.81`, `integrator=RK4`, `iterations=50`。依据：[TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L3)。
- default geom friction/contact 参数：`friction="1.0 0.005 0.0001"`, `solref="0.02 1"`, `solimp="0.9 0.95 0.001"`, `condim="3"`。依据：[TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L4)。

重点结论：

- 仍然没有 `<actuator>`，`nu=0`，因此 raw MuJoCo 模型没有任何 joint control channel。
- 仍然没有 `<camera>`，只有 light。双相机来自运行时 scene builder，不在 base MJCF 本体里。依据：[mujoco_dual_camera_scene.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py#L156)。
- 只有 `floor` 这个交互物体；没有桌子、没有目标 cube、没有 grasp object。依据：[TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L20) 到 [TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L25)，以及本次结构解析结果。
- mesh asset 文件路径当前能解析成功，是因为 `cjjarm_urdf/` 根目录下有一层指向 `meshes/collision/*.stl` 的符号链接；scene builder 还会把所有 `file=` 重写成绝对路径。依据：[mujoco_dual_camera_scene.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py#L314)。

## 5. 发现的 MuJoCo 加载问题

MuJoCo 100 步空仿真结果：

| model | status | nbody | njnt | ngeom | nu | ncam | max body disp (m) | joint2@100 | qvel max | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf | ok | 9 | 8 | 9 | 0 | 0 | 0.1751 | -0.6900 | 7.0587 | - |
| lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf | ok | 9 | 8 | 13 | 0 | 0 | 0.1751 | -0.6900 | 7.0587 | - |
| lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf | ok | 9 | 8 | 9 | 0 | 0 | 0.1751 | -0.6900 | 7.0587 | - |
| lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf | ok | 10 | 8 | 10 | 0 | 0 | 0.1768 | -0.6830 | 7.0595 | - |

重点结论：

- 四个源模型都能被 MuJoCo 读入，但四个模型全部 `nu=0`、`ncam=0`。这说明“能加载”不等于“能控制”或“能渲染相机观测”。
- 四个模型在 100 步无控制空仿真后，`joint2` 都从 `0` 漂到了约 `-0.69 rad`，`max_body_displacement_m` 约 `0.175 m`。这意味着模型在重力下自行塌落，而不是稳稳停在一个 home pose。依据：本次加载测试结果和关节漂移记录在 [simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。
- `qpos/qvel` 全程有限，没有 NaN 或爆炸；问题不是数值发散，而是“无 actuator + 无稳定持位控制 + 无足够阻尼”导致的重力塌落。

## 6. 发现的 collision / contact 问题

cube 接触测试结果：

- 基础场景：`TRLC-DK1-Follower-home.mjcf`
- 方式：在 `link6-7/finger` 附近上方生成一个 `0.05m` 边长 cube，自由下落 200 步
- 结果：
  - `max_ncon = 7`
  - `cube_contact_steps = 101`
  - `unique_contact_pairs = [['audit_cube_geom', 'floor'], ['audit_cube_geom', 'link4-5:geom_5'], ['audit_cube_geom', 'link6-7:geom_7']]`
  - `final_cube_center_z_m = 0.006240`
  - `cube_passed_below_floor_top = False`

结论：

- MuJoCo 的接触求解器不是完全失效。cube 确实与 `floor`、`link4-5`、`link6-7` 发生了接触，因此“完全没有 collision/contact”不是当前主因。
- 但机器人碰撞体仍然是 mesh geom，接触对上也表现为 link-level mesh 碰撞，不是为 grasp/contact 调过的简化 pad/cage 结构。这对稳定抓取非常不利。
- 默认双相机场景 builder 新增的相机 marker 会显式设置 `contype=0` / `conaffinity=0`，它们是纯可视化，不参与物理接触；这部分是正确的。依据：[mujoco_dual_camera_scene.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py#L140)。

## 7. 发现的 gripper 问题

现状：

- gripper joint：`gripper_left`、`gripper_right`
- raw MJCF actuator：_None_
- 结构类型：双 slide/prismatic joint，不是 mimic，不是 tendon。依据：[TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L54) 与 [TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf#L59)。

运行时探测结果：

- `CjjArmSim` 默认场景：`nu=0`, `ncam=2`, `virtual_target_body_free` 存在
- `legacy` 场景：`nu=0`, `ncam=0`
- 在没有 actuator 的前提下，连续发送 `{"gripper.pos": -1.0}`，`gripper_left` qpos 从 `0.0` 变到 `-0.019612`，并持续有接触计数。这说明 gripper 运动不是通过 MuJoCo actuator/ctrl 完成，而是由 `CjjArmSim` 直接写 `qpos` 完成。依据：[cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L420)。

结论：

- 你的 gripper 当前确实“会动”，但这不是物理控制链路，而是脚本直接改状态量。
- raw MuJoCo 模型不存在 `gripper actuator`、不存在 `ctrlrange`，所以也谈不上检查 action 输入与 actuator `ctrlrange` 是否匹配；这条控制链本身就没建出来。
- 现有 `gripper.pos` 还是离散阈值逻辑，不是连续开合目标。真机版本依据 [cjjarm_robot.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_robot.py#L246)，仿真版本依据 [cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L489)。

## 8. 发现的 LeRobot env 问题

自定义环境入口是 [gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L120) 的 `RobotEnv`。

源代码行为：

- `reset()` 返回 `(obs, info)`。依据：[gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L221)。
- `step(action)` 返回 `(obs, reward, terminated, truncated, info)`。依据：[gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L252)。
- observation 结构来自 `_get_observation()`，核心键是 `agent_pos`、`pixels` 和各个 `<joint>.pos`。依据：[gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L166)。
- observation space 只有 `observation.state` 和 `observation.images.*` 的等价结构，没有额外的 `observation.env_state`。
- action space 在源码里被硬编码为 3 维；开 gripper 时变成 4 维。依据：[gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L203)。
- 但 `step()` 却按 `self.robot.bus.motors.keys()` 枚举所有关节去取 action。依据：[gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L254)。

问题：

- `RobotEnv` 假设 robot 插件实现了 `self.robot.bus.motors`。但 `CjjArm` 实际存的是 `self.motors`，不是 `self.bus`。依据：[cjjarm_robot.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_robot.py#L50)。
- `CjjArmSim` 也没有 `bus` 属性；本次运行时探测结果是 `has_bus_attribute_default = false`。依据：[simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。
- 因此 `gym_manipulator` 和当前 CJJ robot/sim 插件接口是不兼容的。即使不考虑别的问题，这条 env 封装也不能直接用来训练/评测 `cjjarm_sim`。
- 此外，本次导入 `gym_manipulator` 时还直接失败在 `torchvision::nms` 缺失，说明当前 `lerobot` 训练/数据处理环境本身还有一层 PyTorch/TorchVision 依赖冲突。依据：[simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。

## 9. 最可能导致“仿真环境效果很差”的前三个根因

1. **没有 actuator 控制链，只有直接写 `qpos` 的伪控制。**
   - 证据：四个源模型全部 `nu=0`；`CjjArmSim` 用 `data.qpos[...] = ...` 直接驱动关节和夹爪。见 [cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py#L404)。
2. **机器人接触几何过于“CAD mesh 化”，尤其夹爪没有内侧简化接触面。**
   - 证据：robot link collision 全是 mesh；finger collision 也是 mesh STL，没有 box pad。见 [TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf#L324)。
3. **LeRobot env / 训练封装和 robot 插件接口不一致，而且当前环境还有 torchvision 依赖冲突。**
   - 证据：`RobotEnv` 读 `robot.bus.motors`，但 CJJ 插件没有 `bus`；导入 `gym_manipulator` 时还报 `operator torchvision::nms does not exist`。见 [gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py#L153) 和 [simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)。

## 10. 建议的修复优先级

### 必须立刻修

- 给 `TRLC-DK1-Follower.mjcf` / `TRLC-DK1-Follower-home.mjcf` 增加真实 actuator，并把 `CjjArmSim` 从直接写 `qpos` 改成 `data.ctrl` 驱动。
- 明确只保留一条 canonical 仿真链路。
  - 建议保留 “base MJCF + 运行时 scene builder + MuJoCo cameras”。
  - 不建议再让 `legacy` 模式直接回退到 raw URDF 作为主仿真路径。
- 重做机器人，尤其夹爪的 collision 模型。
  - 至少给 finger 增加简化的内侧 box/capsule 接触面。
  - 给主要 link 增加简化碰撞体，避免全靠 mesh 接触。
- 修复 `gym_manipulator` 与 `cjjarm/cjjarm_sim` 的接口。
  - 要么让 robot 插件实现 `bus` 兼容层。
  - 要么改 `RobotEnv` 不再假设 `robot.bus.motors`。
  - 同时把 action 维度和真实 joint 数量对齐。

### 可以后面修

- 为 scene object/桌面/容器系统性加入 inertial、friction、named geoms、named cameras。
- 给 MJCF 加相机本体定义，减少对运行时拼接的依赖。
- 给关节补充 damping/friction，更稳定地抑制塌落和微振荡。
- 统一 geom 命名，避免大量 `<unnamed_X>`，便于 contact/debug。

### 暂时不用管

- `tool0` 和 `camera` frame 本身没有 inertial/collision，只要它们保持 frame 语义，不是当前主因。
- `home.urdf` 里大量 purely-visual 房间道具，如果最终 canonical 仿真不走 URDF scene，这部分不是优先级最高的问题。
- `outputs/` 目录下历史生成的 MJCF/summary 不需要立刻清理。

## 11. 下一步应该修改哪些文件（本轮未修改）

必须优先动这些文件：

1. [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.mjcf)
2. [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.mjcf)
3. [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower.urdf)
4. [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_urdf/TRLC-DK1-Follower-home.urdf)
5. [lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py](/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm/lerobot_robot_cjjarm/cjjarm_sim.py)
6. [src/lerobot/rl/gym_manipulator.py](/home/cjj/lerobot_vlbiman/src/lerobot/rl/gym_manipulator.py)

按需要再动这些文件：

1. [src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/sim/mujoco_dual_camera_scene.py)
2. [src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py)
3. [src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py)
4. [src/lerobot/projects/vlbiman_sa/app/run_frrg_mujoco_validate.py](/home/cjj/lerobot_vlbiman/src/lerobot/projects/vlbiman_sa/app/run_frrg_mujoco_validate.py)

## 附：辅助证据

- 审计原始数据：[tools/audit/simulation_audit_data.json](/home/cjj/lerobot_vlbiman/tools/audit/simulation_audit_data.json)
- 审计脚本：[tools/audit/run_simulation_audit.py](/home/cjj/lerobot_vlbiman/tools/audit/run_simulation_audit.py)
- 本报告生成脚本：[tools/audit/render_simulation_audit_report.py](/home/cjj/lerobot_vlbiman/tools/audit/render_simulation_audit_report.py)
