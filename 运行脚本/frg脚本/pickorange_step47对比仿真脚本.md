## 使用目的

这条脚本用于做同起点 A/B 对比：

- 从 `trajectory_index = 0` 开始
- 先正常播放前 `40` 步
- 到第 `40` 步后，直接跳过 `41-67` 这些“抓取前但两边都不变”的 `approach` 帧
- A 分支停留在第 `40` 步的当前位置，直接开始原始 `gripper_close`
- B 分支从第 `40` 步的当前位置起步，使用 FRRG 的增量抓取动作逼近目标
- 两个分支都只执行抓取段，抓完就停止，不再接后面的 `pink cup` 接近、搬运和放置段
- FRRG 分支的抓取播放速度放慢为原来的 `1/3`

输出内容包括：

- 两条分支各自的视频
- 两条分支各自的 `summary.json`
- 总对比汇总 `summary.json`

对应脚本：

- `src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_branch_compare.py`

## 使用前提

```bash
cd /home/cjj/lerobot_vlbiman
conda activate lerobot
```

## MuJoCo 界面观看脚本

下面两条才是直接打开 MuJoCo 仿真界面观看的命令。

注意：

- 窗口模式一次只看一个分支，所以这里分别给出原始 replay 和 FRRG 两条命令。
- 窗口模式不要加 `--headless`。
- 这里固定从 `trajectory_index = 0` 开始，并在 `40` 之后直接跳到抓取入口。
- 这里固定只看到抓取结束，不接回后续 `pink cup` 相关流程。

### 原始 replay 分支

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISPLAY=:1 \
PYTHONNOUSERSITE=1 PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_branch_compare.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --reuse-live-result outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json \
  --target-phrase orange \
  --aux-target-phrase "pink cup" \
  --start-index 0 \
  --jump-to-pick-after-index 40 \
  --stop-after-pick \
  --branch original_replay \
  --display :1 \
  --final-hold-s 1.0 \
  --log-level INFO
```

### FRRG 替换分支

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DISPLAY=:1 \
PYTHONNOUSERSITE=1 PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_branch_compare.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --reuse-live-result outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json \
  --target-phrase orange \
  --aux-target-phrase "pink cup" \
  --start-index 0 \
  --jump-to-pick-after-index 40 \
  --stop-after-pick \
  --frrg-playback-repeat 3 \
  --branch frrg_replacement \
  --display :1 \
  --final-hold-s 1.0 \
  --log-level INFO
```

说明：

- `--branch original_replay` 看的是原始 `vlbiman` 重放抓取段。
- `--branch frrg_replacement` 看的是 `pickorange` 被 FRRG 替换后的抓取段。
- `--jump-to-pick-after-index 40` 的意思是：
  - 播放 `0-40`
  - 直接跳过 `41-67`
  - 从抓取入口开始看差异
- `--stop-after-pick` 的意思是：
  - 只保留 `0-40 + grasp`
  - 抓取结束就停止
  - 不接回后续 `pink cup` 的接近、搬运、放置、退回段
- `--frrg-playback-repeat 3` 的意思是：
  - FRRG 每个抓取动作点重复播放 `3` 次
  - 等效为 FRRG 抓取段放慢到原来的 `1/3`
- 如果你的窗口环境不是 `:1`，把命令里的 `DISPLAY=:1` 和 `--display :1` 一起改掉。

## Headless 导出命令

如果你要的是导出视频和汇总，而不是直接看 MuJoCo 窗口，用下面这条：

```bash
MUJOCO_GL=egl HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
PYTHONNOUSERSITE=1 PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python3 src/lerobot/projects/vlbiman_sa/app/run_visual_pickorange_branch_compare.py \
  --task-config src/lerobot/projects/vlbiman_sa/configs/task_grasp_one_shot_full_20260411T061326.yaml \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --reuse-live-result outputs/vlbiman_sa/live_orange_pose/live_scene_20260420T133513Z.json \
  --target-phrase orange \
  --aux-target-phrase "pink cup" \
  --start-index 0 \
  --jump-to-pick-after-index 40 \
  --stop-after-pick \
  --frrg-playback-repeat 3 \
  --headless \
  --log-level INFO
```

## 本次实际运行结果

实际跑通目录：

- `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z`

重点文件：

- 总对比汇总：
  - `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z/summary.json`
- 原始 replay 分支视频：
  - `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z/branches/original_replay/validation.mp4`
- FRRG 替换分支视频：
  - `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z/branches/frrg_replacement/validation.mp4`
- 原始 replay 分支总结：
  - `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z/branches/original_replay/summary.json`
- FRRG 替换分支总结：
  - `outputs/vlbiman_sa/visual_pickorange_compare/pickorange_compare_20260422T112807Z/branches/frrg_replacement/summary.json`

## 当前结果摘要

- 起点：`trajectory_index = 0`
- 跳转点：`trajectory_index = 40`
- 被跳过的未变段：`41-67`，共 `27` 个 pre-pick `approach` 点
- `stop_after_pick = true`
- `frrg_playback_repeat = 3`，等效 `frrg_playback_speed_ratio = 0.3333333333333333`
- 原始 `pickorange` 段：
  - `12` 个点
  - `source = hold_arm_from_current`
  - `joint_span_rad_inf = 0.0`
- FRRG 替换段：
  - `19` 个点
  - `joint_span_rad_inf = 0.07818440443516517`
  - `frrg_status = success`
  - `frrg_final_phase = SUCCESS`
  - 从第 `40` 步当前位姿起步，没有再跳回原始 pregrasp 位姿
  - GUI 播放时按 `3` 倍重复后，表现为：
    - `handoff = 3` 点
    - `capture_build = 18` 点
    - `close_hold = 18` 点
    - `lift_test = 18` 点

整条分支对比：

- 原始 replay：`53` 个点，末端平移路径长度 `218.51 mm`
- FRRG 替换：`98` 个点，末端平移路径长度 `237.38 mm`

这说明当前这版对比已经是：

- `0-40` 正常播放
- 原始分支是在第 `40` 步位置原地闭夹，然后停止
- FRRG 分支是在第 `40` 步位置切入，执行 FRRG 版本 grasp，然后停止
- 后面不再接近 `pink cup`
- FRRG 抓取段比原始 replay 多出额外的增量动作，因此能更直观看到“发现偏差后继续逼近并完成 grasp”的差异
