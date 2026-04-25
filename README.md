# VLBiman Single-Arm Adaptation

基于 Hugging Face LeRobot 的 VLBiman 单臂复现与硬件适配工程。当前仓库围绕 CJJ Arm、Gemini 335L RGBD 相机、Zhonglin 遥操作器和 MuJoCo 仿真环境，搭建了一套从 one-shot 示教采集到视觉定位、几何适配、轨迹生成、末端抓取闭环验证的 Python 工作流。

本仓库重点不是 LeRobot 上游通用能力展示，而是记录 VLBiman 在本地机械臂平台上的适配工作：硬件插件、数据采集链路、单臂抓取任务流程、仿真校验和 FRRG 末端闭环模块。

## 当前能力

- CJJ Arm 机械臂驱动：串口自动探测、DM 电机映射、夹爪控制、URDF/IK 配置和 LeRobot `RobotConfig` 注册。
- Gemini 335L 相机插件：RGBD 读取、深度对齐、profile 探测、无 GUI 健康检查和实时预览脚本。
- Zhonglin 遥操作器：串口输入读取、零位标定、动作映射，并支持真实机械臂和仿真机械臂两种后端。
- VLBiman 单臂流程：one-shot RGBD 示教采集、技能切段、var/inv 片段识别、Florence-2 + SAM2 目标分割、锚点估计、位姿适配、渐进 IK 轨迹生成。
- MuJoCo 验证：基于 CJJ Arm MJCF/URDF 资产生成双相机场景，支持离线轨迹回放、目标物位置更新和视觉闭环验证。
- FRRG 末端闭环：针对粗接近后的最后几厘米抓取，提供 `HANDOFF -> CAPTURE_BUILD -> CLOSE_HOLD -> LIFT_TEST` 状态机、guard、score、安全限幅、zero residual 和 benchmark fixture。

## 工程结构

```text
.
├── lerobot_robot_cjjarm/                 # CJJ Arm 真实机器人与仿真机器人适配
├── lerobot_camera_gemini335l/            # Orbbec Gemini 335L RGBD 相机插件
├── lerobot_teleoperator_zhonglin/        # Zhonglin 遥操作器适配
├── scripts/                              # 常用遥操作、相机检查、VLBiman 运行脚本
├── src/lerobot/projects/vlbiman_sa/
│   ├── app/                              # 流程入口：采集、分析、规划、验证、FRRG
│   ├── calib/                            # 手眼标定与相机内参工具
│   ├── configs/                          # pipeline、record、vision、grasp、FRRG 配置
│   ├── core/                             # 顶层任务编排与任务合同
│   ├── demo/                             # RGBD 示教数据 schema、IO、recorder
│   ├── geometry/                         # 坐标变换、frame manager、位姿适配
│   ├── grasp/                            # FRRG 末端抓取闭环
│   ├── sim/                              # MuJoCo 双相机场景构建
│   ├── skills/                           # 关键点切段、技能库、var/inv 判别
│   ├── trajectory/                       # 渐进 IK 与轨迹拼接
│   └── vision/                           # VLM 分割、mask 追踪、锚点与朝向估计
└── tests/projects/vlbiman_sa/            # 单臂适配与 FRRG 的回归测试
```

## 端到端流程

```text
Task text + one-shot RGBD demo + live RGBD
  -> RGBDRecorder
  -> KeyposeSegmenter + InvarianceClassifier
  -> VLMObjectSegmentor + MaskTracker + AnchorEstimator
  -> PoseAdapter + GeometryCompensator
  -> TrajectoryComposer + ProgressiveIKPlanner
  -> MuJoCo replay / real robot execution / FRRG final grasp loop
```

`src/lerobot/projects/vlbiman_sa/configs/pipeline.yaml` 是顶层流程配置。默认开启 `record` 阶段，其他阶段可按需开启，或用 `--stages` 指定。

## 快速开始

建议使用 Python 3.10 环境。先安装 LeRobot 基础工程和本仓库的三个本地插件：

```bash
python -m pip install -e .
python -m pip install -e ./lerobot_robot_cjjarm
python -m pip install -e ./lerobot_camera_gemini335l
python -m pip install -e ./lerobot_teleoperator_zhonglin
```

相机链路检查：

```bash
./scripts/gemini335l/run_stream_viewer.sh --list
./scripts/gemini335l/quick_check.sh <camera_serial_number>
```

键盘遥操作仿真机械臂：

```bash
PYTHONPATH=src:. python scripts/teleop_cjjarm_keyboard.py --robot-type cjjarm_sim
```

Zhonglin 遥操作仿真机械臂：

```bash
PYTHONPATH=src:. python scripts/teleop_cjjarm_zhonglin.py \
  --robot-type cjjarm_sim \
  --teleop-port /dev/ttyUSB0
```

查看 VLBiman 流程计划，不实际执行硬件：

```bash
PYTHONPATH=src:. python src/lerobot/projects/vlbiman_sa/app/run_pipeline.py \
  --config src/lerobot/projects/vlbiman_sa/configs/pipeline.yaml \
  --plan-only
```

采集 one-shot RGBD 示教：

```bash
PYTHONPATH=src:. python src/lerobot/projects/vlbiman_sa/app/run_one_shot_record.py \
  --config src/lerobot/projects/vlbiman_sa/configs/one_shot_record.yaml \
  --camera-serial-number <camera_serial_number> \
  --robot-serial-port /dev/ttyACM0 \
  --teleop-port /dev/ttyUSB0
```

运行 FRRG mock benchmark：

```bash
PYTHONPATH=src:. python src/lerobot/projects/vlbiman_sa/app/run_frrg_benchmark.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --fixtures tests/fixtures/vlbiman_sa \
  --output-dir outputs/vlbiman_sa/frrg/benchmark
```

## 关键配置

- `configs/one_shot_record.yaml`：RGBD 采集、机器人串口、Zhonglin 遥操作器和录制参数。
- `configs/task_grasp.yaml`：单臂抓取任务、示教 session、T3/T4/T5/T6 输出路径和目标文本。
- `configs/frrg_grasp.yaml`：FRRG 状态机、抓取特征、闭合判断、lift test、安全限幅和 residual 策略。
- `configs/handeye_auto.yaml`：手眼标定自动采样与求解参数。
- `configs/vision_analysis.yaml`：Florence-2、SAM2、mask tracker 和锚点估计参数。

运行产物默认写入 `outputs/vlbiman_sa/`。该目录用于保存示教数据、视觉分析结果、轨迹、benchmark 和报告，通常不提交到 Git。

## 验证范围

当前测试覆盖集中在 VLBiman 单臂适配和 FRRG 末端闭环：

```bash
PYTHONPATH=src:.:lerobot_robot_cjjarm:lerobot_camera_gemini335l:lerobot_teleoperator_zhonglin \
  pytest tests/projects/vlbiman_sa tests/test_manual_record_controls.py
```

已有测试包括：

- CJJ Arm 配置、MuJoCo 环境、双相机场景和接触抓取验证。
- RGBD recorder schema/IO、运行环境 bootstrap、pipeline command 构造。
- frame manager、pose/trajectory 关键逻辑和 IK 方案选择。
- FRRG config、observer、feature geometry、scores、phase guards、state machine、safety limiter、recovery、benchmark、zero residual 和 controller。

## 适配边界

本工程保留 LeRobot 上游代码作为基础框架，但展示重点是 `vlbiman_sa` 任务和本地硬件插件。真实硬件执行前需要确认：

- Gemini 335L 序列号、分辨率、FPS、深度对齐模式可稳定读帧。
- CJJ Arm 串口、关节方向、夹爪闭合方向和限位配置与当前设备一致。
- Zhonglin 遥操作器完成零位标定，动作映射符合当前任务方向。
- `outputs/vlbiman_sa/calib/handeye_result.json` 与当前相机安装位姿一致。
- 在仿真或 dry-run 中验证过轨迹连续性、安全限幅和失败退出路径。

## 状态说明

这是一个面向 VLBiman 单臂抓取复现的适配型工程仓库。当前版本已经完成核心 Python 模块、硬件插件入口、MuJoCo 验证链路和 FRRG dry-run/fixture 验证；后续工作主要围绕真实硬件长时间稳定性、更多物体类别的视觉鲁棒性、真实抓取统计评测和 residual 策略训练展开。
