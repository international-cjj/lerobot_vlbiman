## 使用目的

这个脚本复用当前 `vlbiman` 的仿真闭环场景，不改场景本身，只把 `pickorange` 阶段原先的重放动作替换成 FRRG 当前方案，再把后续 `transfer / place / retreat` 接回原流程继续执行。

当前替换的是 `T6` 轨迹里的抓取段：

- `segment_id = skill_001`
- `segment_label = gripper_close`

对应运行脚本：

- `src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_frrg_validation.py`

## 使用前提

```bash
cd /home/cjj/lerobot_vlbiman
conda activate lerobot
```

和现有 `vlbiman` 脚本保持一致，运行时显式指定：

```bash
PYTHONNOUSERSITE=1
PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman
```

## 推荐命令：Headless 仿真

这是当前实际跑通的一条命令，适合先验证 FRRG 是否成功接管 `pickorange`。

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_frrg_validation.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --reuse-live-result outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json \
  --target-phrase orange \
  --aux-target-phrase "pink cup" \
  --headless \
  --log-level INFO
```

说明：

- `--reuse-live-result` 复用现成视觉结果，不重新采集。
- `--headless` 走离屏渲染。
- `MUJOCO_GL=egl` 是当前无桌面窗口时的必要设置。

## 推荐命令：带界面仿真

如果你要直接看 MuJoCo 窗口，可以运行：

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISPLAY=:1 \
PYTHONNOUSERSITE=1 PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_frrg_validation.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --reuse-live-result outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json \
  --target-phrase orange \
  --aux-target-phrase "pink cup" \
  --display :1 \
  --log-level INFO
```

如果你用 `xvfb-run` 启动，不要额外手动覆盖 `--display`。

## 当前已验证结果

最新实际跑通目录：

- `outputs/vlbiman_sa/visual_pickorange_frrg/pickorange_frrg_20260422T075135Z`

关键结果：

- `frrg_status = success`
- `frrg_final_phase = SUCCESS`
- `frrg_steps_run = 19`
- `pick_segment_id = skill_001`
- `pick_segment_label = gripper_close`
- `original_pick_point_count = 12`
- `frrg_point_count = 19`
- `suffix_bridge_decision.bridge_injected = true`

说明：

- 原先 `pickorange` 的抓取重放段已经被 FRRG 替换。
- FRRG 成功执行后，流程继续接回原来的 `transfer / place / retreat`。

## 重点输出文件

运行完成后重点看这些文件：

- 总结：
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/summary.json`
- FRRG 控制器结果：
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/analysis/frrg_pickorange/summary.json`
- 拼接后的执行轨迹：
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/execution/spliced_points.json`
- FRRG handoff 输入摘要：
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/analysis/coarse_summary.json`
- Headless 预览图：
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/preview/frame_000.png`
  - `outputs/vlbiman_sa/visual_pickorange_frrg/<run_dir>/preview/frame_last.png`
