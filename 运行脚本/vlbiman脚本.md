## 使用前提

先准备运行环境：

```bash
cd /home/cjj/lerobot_vlbiman
source /home/cjj/miniconda3/etc/profile.d/conda.sh
conda activate lerobot
export PY="$(command -v python)"
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman:/home/cjj/lerobot_vlbiman/lerobot_robot_cjjarm
```

如果 `"$PY"` 为空：

```bash
export PY=/home/cjj/miniconda3/envs/lerobot/bin/python
```

## 统一流程

现在真机和纯仿真都统一走 VLBiMan 原版非 FRRG 流程：

```text
one-shot session -> T3 切分 -> T4 目标定位 -> T5 位姿适配 -> T6 轨迹生成 -> 执行端
```

执行端只切换机械臂控制部分：

- 真机仿真验证：视觉仍来自真机 RGBD / 真机录制数据，只有 T6 机械臂执行接 MuJoCo
- 真机执行：视觉来自真机，T6 机械臂执行接真机
- 纯仿真：场景、视觉真值、录制、T6 执行都来自 MuJoCo

物体指定统一用：

- `--target-phrase`：主目标
- `--aux-target-phrase`：辅助目标，可重复

当前项目仿真场景 preset：

```text
green_plate_yellow_ball
```

包含：

- `project_green_plate_geom`：绿色容器，底部尺寸保持原 preset，边沿高度 `5 cm`
- `project_yellow_ball_geom`：黄色球体，直径为原始尺寸的 `1.5` 倍，默认在 front camera 画面右侧偏移绿色盘中心 0.20 m；可用场景搭建命令独立改变球体和盘子位置；摩擦已提高，便于夹爪夹持
- `front_camera`：外部相机
- `wrist_camera`：腕部相机

## 一、真机部分

真机按“录制 -> 切分 -> 仿真验证 -> 真机执行”走。仿真验证不会发真机动作；视觉和 live 目标定位仍然使用真机数据。

### 1. 录制

```bash
RUN_NAME="one_shot_real_$(date -u +%Y%m%dT%H%M%SZ)"
SESSION_DIR="outputs/vlbiman_sa/recordings/$RUN_NAME"

"$PY" src/lerobot/projects/vlbiman_sa/app/run_one_shot_record.py \
  --config src/lerobot/projects/vlbiman_sa/configs/one_shot_record.yaml \
  --run-name "$RUN_NAME" \
  --output-root outputs/vlbiman_sa/recordings \
  --robot-serial-port /dev/ttyACM0 \
  --teleop-port /dev/ttyUSB0 \
  --duration-s 60 \
  --fps 10 \
  --control-rate-hz 30 \
  --log-level INFO
```

检查录制目录：

```bash
echo "$SESSION_DIR"
ls "$SESSION_DIR"/metadata.jsonl "$SESSION_DIR"/rgb "$SESSION_DIR"/depth
```

### 2. 切分

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_skill_build.py \
  --config src/lerobot/projects/vlbiman_sa/configs/skill_build.yaml \
  --session-dir "$SESSION_DIR" \
  --output-dir "$SESSION_DIR/analysis/t3_skill_bank" \
  --log-level INFO
```

可选：导出 T3 帧标签检查结果。

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/export_t3_frame_labels.py \
  --session-dir "$SESSION_DIR" \
  --skill-bank-path "$SESSION_DIR/analysis/t3_skill_bank/skill_bank.json" \
  --output-dir "$SESSION_DIR/analysis/t3_frame_dataset"
```

### 3. 仿真验证

先准备真机视觉的 live 目标定位结果。已有结果时直接复用：

```bash
LIVE_RESULT="outputs/vlbiman_sa/live_orange_pose/latest_result.json"
```

如果要现场用真机 RGBD 重新定位主目标和辅助目标，用这个命令生成 `latest_result.json`。这一步使用真机相机，MuJoCo 只用于后面的机械臂预演。

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISPLAY=:1 \
"$PY" src/lerobot/projects/vlbiman_sa/app/run_visual_closed_loop_validation.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp.yaml \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --scene-preset green_plate_yellow_ball \
  --capture-mode single_shot \
  --max-cycles 1 \
  --max-runtime-s 30 \
  --display :1 \
  --log-level INFO

LIVE_RESULT="outputs/vlbiman_sa/live_orange_pose/latest_result.json"
```

然后跑原版 T4/T5/T6，并把执行端接到 MuJoCo：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source real-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --skip-skill-build \
  --execute-backend sim \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```

如果 T4 已经跑过并确认 `analysis/t4_vision*` 可用，可以复用 T4：

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source real-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend sim \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```

打开 viewer 看机械臂动作：

```bash
DISPLAY=:1 "$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source real-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend sim \
  --open-viewer \
  --display :1 \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```

### 4. 真机执行

先做执行 dry-run：

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source real-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend robot \
  --robot-serial-port /dev/ttyACM0 \
  --dry-run-robot \
  --log-level INFO
```

确认 dry-run 后正式发真机：

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source real-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "orange" \
  --aux-target-phrase "pink cup" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend robot \
  --robot-serial-port /dev/ttyACM0 \
  --log-level INFO
```

## 二、纯仿真部分

纯仿真按“场景搭建 -> 查看场景 -> 录制 -> 切分 -> 执行”走。这里使用 MuJoCo oracle 写出 T4 兼容结果，但 T3/T5/T6 仍然是 VLBiMan 原版流程。录制时会同时保存 `front_camera` 和 `wrist_camera`；原版 VLBiMan 读取的 `rgb/`、`depth/` 仍然作为 `front_camera` 的主相机兼容入口。

### 1. 场景搭建

```bash
SCENE_RUN_ID="pure_sim_plate_ball"
PLATE_X=0.45
PLATE_Y=0.3
PLATE_Z=0.041
BALL_X=0.45
BALL_Y=0.0
BALL_Z=0.05925

"$PY" src/lerobot/projects/vlbiman_sa/app/run_mujoco_dual_camera_scene.py \
  --run-id "$SCENE_RUN_ID" \
  --scene-preset green_plate_yellow_ball \
  --target-x 0.45 \
  --target-y 0.00 \
  --plate-x "$PLATE_X" \
  --plate-y "$PLATE_Y" \
  --plate-z "$PLATE_Z" \
  --ball-x "$BALL_X" \
  --ball-y "$BALL_Y" \
  --ball-z "$BALL_Z" \
  --settle-steps 80 \
  --render-width 640 \
  --render-height 480
```

位置单位都是米，坐标是 MuJoCo 世界坐标。`--target-x/--target-y/--target-z` 仍只控制隐藏的 virtual target；可见的绿色盘子和黄色球体分别由 `--plate-*`、`--ball-*` 控制。只想改平面位置时，可以只传 `--plate-x/--plate-y` 或 `--ball-x/--ball-y`，不传的轴会沿用默认高度。

输出：

- `outputs/vlbiman_sa/mujoco_dual_camera_scene/pure_sim_plate_ball/summary.json`
- `outputs/vlbiman_sa/mujoco_dual_camera_scene/pure_sim_plate_ball/front_camera.png`
- `outputs/vlbiman_sa/mujoco_dual_camera_scene/pure_sim_plate_ball/wrist_camera.png`
- `outputs/vlbiman_sa/mujoco_dual_camera_scene/pure_sim_plate_ball/overview.png`

### 2. 查看场景

继续使用上一步的 `PLATE_*` / `BALL_*` 位置变量；如果新开终端，需要先重新设置这几个变量。

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_mujoco_dual_camera_viewer.py \
  --run-id pure_sim_plate_ball_viewer \
  --scene-preset green_plate_yellow_ball \
  --plate-x "$PLATE_X" \
  --plate-y "$PLATE_Y" \
  --plate-z "$PLATE_Z" \
  --ball-x "$BALL_X" \
  --ball-y "$BALL_Y" \
  --ball-z "$BALL_Z" \
  --settle-steps 300
```

按键：

- `1`：切到 `front_camera`
- `2`：切到 `wrist_camera`
- `3`：切到 `overview` / 自由视角
- `Esc`：退出

只写场景、不打开 GUI：

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_mujoco_dual_camera_viewer.py \
  --run-id pure_sim_plate_ball_scene_only \
  --scene-preset green_plate_yellow_ball \
  --plate-x "$PLATE_X" \
  --plate-y "$PLATE_Y" \
  --plate-z "$PLATE_Z" \
  --ball-x "$BALL_X" \
  --ball-y "$BALL_Y" \
  --ball-z "$BALL_Z" \
  --write-scene-only
```

### 3. 录制

这里生成的是 `run_skill_build.py` 可直接读取的 VLBiMan 原生 session。录制命令通过 `--scene-run-id "$SCENE_RUN_ID"` 绑定到上一步搭建的场景；viewer 和录制数据都使用同一个 `dual_camera_target_scene.mjcf`。

```bash
SCENE_RUN_ID="${SCENE_RUN_ID:-pure_sim_plate_ball}"
RUN_ID="pure_sim_original_$(date -u +%Y%m%dT%H%M%SZ)"
FLOW_DIR="outputs/vlbiman_sa/original_flow/$RUN_ID"
SESSION_DIR="$FLOW_DIR/recordings/sim_one_shot"
LIVE_RESULT="$FLOW_DIR/live_result.json"

"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source sim \
  --run-id "$RUN_ID" \
  --target-phrase "yellow ball" \
  --aux-target-phrase "green plate" \
  --scene-preset green_plate_yellow_ball \
  --scene-run-id "$SCENE_RUN_ID" \
  --sim-frames 300 \
  --sim-fps 10 \
  --sim-teleop zhongling \
  --teleop-port /dev/ttyUSB0 \
  --render-width 640 \
  --render-height 480 \
  --manual-record-control \
  --record-viewer \
  --stop-after record \
  --log-level INFO
```

按键：

- 运行命令后，先把 Zhongling 控制器放到本次录制的零位；默认会在连接时把当前控制器姿态标定为零位
- 看到 `Zhongling sampling is live.` 后再按 `Space` 开始录制；控制器读取在后台线程运行，不会阻塞 viewer 或空格停止
- 录制 viewer 在独立进程中运行，主录制进程只向 viewer 推送仿真状态；切换视角不会阻塞双相机数据落盘
- MuJoCo viewer 只控制显示视角，不改变录制数据；`rgb/depth` 仍保存 `front_camera`，`cameras/front_camera` 和 `cameras/wrist_camera` 会同时落盘
- `1`：viewer 切到 `front_camera`
- `2`：viewer 切到 `wrist_camera`
- `3`：viewer 切到自由视角，之后可用鼠标调整视角
- `Space`：开始录制，可以在终端或 viewer 中按
- `Space`：结束当前录制，可以在终端或 viewer 中按
- 当前录制结束后 viewer 会关闭，保存/重录在终端继续按键
- 结束后如果不满意，按 `Space` 直接重新录制
- 结束后如果满意，按 `Enter` 保存为 `"$SESSION_DIR"`
- `Esc`：退出，不保存当前尝试

检查输出：

```bash
ls "$SESSION_DIR"/metadata.jsonl "$SESSION_DIR"/rgb "$SESSION_DIR"/depth
ls "$SESSION_DIR"/cameras/front_camera/rgb "$SESSION_DIR"/cameras/wrist_camera/rgb
cat "$FLOW_DIR/flow_summary.json"
```


### 4. 播放

```bash
  PY=${PYTHON_BIN:-$HOME/miniconda3/envs/lerobot/bin/python}
  SESSION=outputs/vlbiman_sa/original_flow/pure_sim_original_20260427T072327Z/recordings/sim_one_shot
  SCENE=outputs/vlbiman_sa/mujoco_dual_camera_scene/pure_sim_plate_ball/scene/dual_camera_target_scene.mjcf

  "$PY" src/lerobot/projects/vlbiman_sa/app/play_mujoco_trajectory.py \
    --demo-session-dir "$SESSION" \
    --model-path "$SCENE" \
    --legacy-scene \
    --target-phrase "yellow ball" \
    --step-duration-s 0.1 \
    --physics-replay \
    --physics-action-source sent \
    --physics-substeps 20 \
    --once

```


PATH=/home/cjj/miniconda3/envs/lerobot/bin:$PATH python src/lerobot/projects/vlbiman_sa/app/run_inv_rgb_servo_replay_viewer.py \
    --source-run-dir outputs/vlbiman_sa/original_flow/pure_sim_original_20260427T072327Z \
    --camera wrist \
    --servo-start-frame 75 \
    --target-frame 100 \
    --target-phrase "yellow ball" \
    --config src/lerobot/projects/vlbiman_sa/configs/inv_rgb_servo.yaml \
    --output-dir outputs/vlbiman_sa/inv_rgb_servo/yellow_ball_replay_viewer \
    --view free


### 5. 切分

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source sim-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "yellow ball" \
  --aux-target-phrase "green plate" \
  --stop-after skill_build \
  --execute-backend none \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```

### 5. 执行

复用上一步的仿真 session、T3 和 MuJoCo oracle T4，继续跑 T5/T6，并在 MuJoCo 执行：

```bash
"$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source sim-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "yellow ball" \
  --aux-target-phrase "green plate" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend sim \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```

打开 viewer 看动作：

```bash
DISPLAY=:1 "$PY" src/lerobot/projects/vlbiman_sa/app/run_vlbiman_original_flow.py \
  --source sim-session \
  --session-dir "$SESSION_DIR" \
  --live-result-path "$LIVE_RESULT" \
  --target-phrase "yellow ball" \
  --aux-target-phrase "green plate" \
  --skip-skill-build \
  --skip-real-t4 \
  --execute-backend sim \
  --open-viewer \
  --display :1 \
  --scene-preset green_plate_yellow_ball \
  --log-level INFO
```



  PYTHONNOUSERSITE=1 /home/cjj/miniconda3/envs/lerobot/bin/python \
    src/lerobot/projects/vlbiman_sa/app/run_vlbiman_real_servo_flow.py \
    --session-dir outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01 \
    --live-result-path outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01/analysis/live_result_from_recorded_frame50.json \
    --task-config outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01/analysis/task_grasp_redcan_real.yaml\
    --target-phrase redcan \
    --servo-segment skill_001 \
    --servo-target-frame 50 \
    --target-mask-path outputs/vlbiman_sa/inv_rgb_servo/check_redcan_real_20260428T125653Z_01_sam2_frame50/mask_000050.png \
    --camera wrist \
    --max-servo-steps 50 \
    --stable-servo-frames 2 \
    --servo-axis-sign-x -1 \
    --servo-axis-sign-y -1 \
    --servo-axis-sign-z 1 \
    --servo-max-step-xy-m 0.02 \
    --servo-max-step-z-m 0.02 \
    --save-overlay-every 1 \
    --step-duration-s 0.10
/

  PYTHONNOUSERSITE=1 /home/cjj/miniconda3/envs/lerobot/bin/python \
      src/lerobot/projects/vlbiman_sa/app/run_vlbiman_real_servo_flow.py \
      --session-dir outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01 \
      --capture-live-result \
      --task-config outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01/analysis/task_grasp_redcan_real.yaml \
      --target-phrase redcan \
      --servo-segment skill_001 \
      --servo-target-frame 50 \
      --target-mask-path outputs/vlbiman_sa/inv_rgb_servo/check_redcan_real_20260428T125653Z_01_sam2_frame50/mask_000050.png \
      --camera wrist \
      --max-servo-steps 50 \
      --stable-servo-frames 2 \
      --servo-axis-sign-x -1 \
      --servo-axis-sign-y -1 \
      --servo-axis-sign-z 1 \
      --servo-max-step-xy-m 0.02 \
      --servo-max-step-z-m 0.02 \
      --servo-interpolation-hz 30 \
      --save-overlay-every 5 \
      --step-duration-s 0.10 \
      --force-t6




  PYTHONNOUSERSITE=1 /home/cjj/miniconda3/envs/lerobot/bin/python \
      src/lerobot/projects/vlbiman_sa/app/run_vlbiman_real_servo_flow.py \
      --session-dir outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01 \
      --capture-live-result \
      --task-config outputs/vlbiman_sa/recordings/one_shot_real_20260428T125653Z_01/analysis/task_grasp_redcan_real.yaml \
      --target-phrase redcan \
      --servo-segment skill_001 \
      --servo-target-frame 50 \
      --target-mask-path outputs/vlbiman_sa/inv_rgb_servo/check_redcan_real_20260428T125653Z_01_sam2_frame50/mask_000050.png \
      --camera wrist \
      --max-servo-steps 70 \
      --stable-servo-frames 2 \
      --servo-axis-sign-x -1 \
      --servo-axis-sign-y -1 \
      --servo-axis-sign-z 1 \
      --servo-max-step-xy-m 0.02 \
      --servo-max-step-z-m 0.02 \
      --servo-step-duration-s 0.10 \
      --servo-interpolation-hz 20 \
      --servo-interpolation-profile linear \
      --servo-arm-velocity 2.0 \
      --servo-arm-smooth-factor 0.5 \
      --servo-command-filter-alpha 0.5 \
      --servo-command-deadband-xy-m 0.00008 \
      --servo-command-deadband-z-m 0.00015 \
      --servo-command-max-change-xy-m 0.00035 \
      --servo-command-max-change-z-m 0.003 \
      --save-overlay-every 5 \
      --step-duration-s 0.05 \
      --force-t6 \
      --servo-k-u 0.03 \
      --servo-k-v 0.03 \
      --servo-k-a 0.02 







