# FRRG 接入 VLBiMan 的纯 Python 分步任务计划书

> 本计划书是后续实现 FRRG 的唯一执行顺序依据。
>
> 当前仓库是 LeRobot / VLBiMan 的纯 Python 工程环境，不使用 ROS、MoveIt、catkin、URDF、SRDF、rospy、moveit_commander、ROS topic 或任何 ROS 运行时。
>
> 项目中存在 `cjjarm` 机械臂和 `Gemini335` 摄像头相关硬件能力，但本计划书的执行阶段禁止调用硬件。验证优先使用纯代码 dry-run、离线 replay、mock fixture；如项目已有 MuJoCo 环境或后续补充 MuJoCo 资产，可增加 MuJoCo 仿真验证，但仍不得调用真实 `cjjarm` 或 `Gemini335` 设备。

---

## 0. 强制约束

### 0.1 环境约束

本任务只在当前 `lerobot_vlbiman` 仓库内实现纯 Python 逻辑：

- 允许使用 Python 标准库、项目已有 Python 模块、NumPy、PyTorch、pytest、yaml/json 等普通 Python 依赖。
- 允许读取 mock JSON、示教数据、现有 VLBiMan 中间结果和离线日志。
- 允许生成 dry-run action、报告文件、benchmark 文件、MuJoCo 仿真输入文件和测试 fixture。
- 允许保留 `cjjarm`、`Gemini335` 的 adapter 接口定义，但首版不得实例化、连接、打开、轮询或发送命令到真实硬件。
- 不允许引入 ROS、MoveIt、catkin、rospy、moveit_commander、geometry_msgs、sensor_msgs、tf、tf2_ros、rosbag、roslaunch、rosrun。
- 不允许在计划或代码中要求启动外部机器人中间件。
- 不允许把硬件调用作为任何步骤的验收条件。

### 0.2 执行顺序约束

后续所有实现必须严格按本计划书的步骤顺序执行。

禁止行为：

- 禁止跳过前一步验收直接实现后一步。
- 禁止在基础 dry-run 未通过前实现学习 residual。
- 禁止在 mock 数据闭环未通过前接入真实采集数据。
- 禁止在本计划首版执行中调用 `cjjarm` 或 `Gemini335`。
- 禁止为了通过测试删除 safety、guard 或失败原因检查。
- 禁止用未确认字段补猜接口。

每一步必须产生：

```text
outputs/vlbiman_sa/frrg/reports/step_<N>_<name>.json
```

报告必须包含：

```json
{
  "step": "step_XX",
  "status": "passed_or_failed",
  "completed_targets": [],
  "changed_files": [],
  "commands_run": [],
  "acceptance": {},
  "hardware_called": false,
  "camera_opened": false,
  "mujoco_available": false,
  "known_gaps": [],
  "next_step_allowed": true
}
```

只有当 `next_step_allowed: true` 时，才允许进入下一步。

### 0.3 首版范围

首版只交付：

```text
纯 Python FRRG dry-run 闭环
  -> mock / 文件输入
  -> 可选 MuJoCo 仿真输入输出
  -> 状态观测
  -> 特征计算
  -> 状态机
  -> guard
  -> 名义控制
  -> safety limiter
  -> zero residual
  -> 报告与 benchmark
```

首版不交付：

- ROS 接入。
- MoveIt 规划。
- 真实 `cjjarm` 命令下发。
- 真实 `Gemini335` 图像采集。
- 任何真实硬件驱动调用。
- 端到端学习控制。
- 未经验证的 residual 在线控制。

### 0.4 硬件边界与验证优先级

本项目允许存在以下硬件相关代码或配置：

- `cjjarm`：机械臂硬件能力或已有接口。
- `Gemini335`：相机硬件能力或已有接口。

但在本计划的 Step 00 到 Step 15 中，这些硬件只能作为“未来 adapter 名称”或“离线数据来源说明”出现，不能被实际调用。

验证优先级固定为：

1. 纯代码单元测试：验证数学、状态机、guard、safety、报告格式。
2. mock fixture dry-run：验证完整 FRRG 闭环。
3. 离线 replay：读取已有日志或示教文件，不访问设备。
4. MuJoCo 仿真：如果项目已有 MuJoCo 依赖和资产，则用仿真验证 action proposal 的几何合理性。
5. 真实硬件：不属于本计划首版执行范围。后续若要接入，必须另写硬件安全计划。

如果 MuJoCo 不存在或资产不完整，不阻塞首版完成；必须回退到纯代码和 mock/replay 验证。

---

## 1. 任务定位

FRRG 的目标是在 VLBiMan 粗接近之后，用纯 Python 闭环逻辑替代原先“末段轨迹重放即认为成功”的做法。

原有逻辑可以抽象为：

```text
one-shot demo
  -> skill bank
  -> vision anchor
  -> pose adaptation
  -> coarse approach output
  -> fixed replay
```

FRRG 接入后的纯 Python 链路：

```text
one-shot demo
  -> skill bank
  -> vision anchor
  -> pose adaptation
  -> coarse approach output
  -> FRRG pure Python closed loop dry-run
  -> action proposal / report / benchmark
```

FRRG 首版只负责“最后几厘米抓取闭环”的计算逻辑：

1. `HANDOFF`：确认粗接近输出可被 FRRG 接管。
2. `CAPTURE_BUILD`：建立目标进入两指走廊的闭合构型。
3. `CLOSE_HOLD`：闭合并做接触代理与稳持判断。
4. `LIFT_TEST`：用小提起逻辑判断是否可能滑脱。
5. `SUCCESS / FAILURE`：输出可复核报告。

---

## 2. 成功定义

FRRG 首版完成时必须满足：

1. 不依赖 ROS 或任何外部机器人中间件。
2. 不调用 `cjjarm` 或 `Gemini335` 时可以完整运行 dry-run。
3. 可以从 mock state 运行 `HANDOFF -> CAPTURE_BUILD -> CLOSE_HOLD -> LIFT_TEST -> SUCCESS`。
4. 可以从 mock failure case 得到明确 `failure_reason`。
5. 所有动作 proposal 都经过 `SafetyLimiter`。
6. residual 默认是 `zero_policy`，输出恒为 0。
7. 每个阶段切换都记录 `score`、`reason`、`debug_terms`。
8. 每一步实现都有独立测试和验收报告。
9. 后续接入真实数据时只替换输入 adapter，不改控制器核心。
10. MuJoCo 验证如果可用，必须作为额外验证层；如果不可用，报告中必须明确 `mujoco_available: false`。

---

## 3. 纯 Python 架构

### 3.1 模块目录

目标目录结构：

```text
src/lerobot/projects/vlbiman_sa/
  configs/
    frrg_grasp.yaml

  grasp/
    contracts.py
    observer.py
    frame_math.py
    feature_geometry.py
    scores.py
    state_machine.py
    phase_guards.py
    safety_limits.py
    recovery_policy.py
    closed_loop_controller.py
    command_adapter.py
    mode_selector.py
    grasp_solver.py
    task_constraints.py

  grasp/parameters/
    theta_schema.py
    default_provider.py
    demo_extractor.py
    regression_provider.py

  grasp/primitives/
    nominal_capture.py
    nominal_close.py
    nominal_lift.py
    minimum_jerk.py

  grasp/residual/
    policy.py
    zero_policy.py
    bc_policy.py
    train_bc_residual.py

  app/
    run_frrg_grasp_dryrun.py
    run_frrg_from_coarse.py
    run_frrg_extract_demo.py
    run_frrg_benchmark.py
    run_frrg_mujoco_validate.py        # 可选，仅在 MuJoCo 可用时实现

  sim/
    mujoco_adapter.py                  # 可选，只接收/输出文件，不调用真实硬件
```

测试目录：

```text
tests/projects/vlbiman_sa/
  test_frrg_contracts.py
  test_frrg_config.py
  test_frrg_frame_math.py
  test_frrg_feature_geometry.py
  test_frrg_scores.py
  test_frrg_state_machine.py
  test_frrg_phase_guards.py
  test_frrg_nominal_primitives.py
  test_frrg_safety_limits.py
  test_frrg_recovery_policy.py
  test_frrg_controller.py
  test_frrg_coarse_handoff.py
  test_frrg_demo_extractor.py

tests/fixtures/vlbiman_sa/
  frrg_handoff_ready.json
  frrg_capture_ready.json
  frrg_nominal_success.json
  frrg_capture_timeout.json
  frrg_vision_lost.json
  frrg_slip_detected.json
  coarse_handoff_summary.json
```

输出目录：

```text
outputs/vlbiman_sa/frrg/
  dryrun/
  reports/
  demo_params/
  benchmarks/
  failure_cases/
```

### 3.2 模块职责

`contracts.py`：

- 定义数据结构。
- 不依赖任何外部系统。
- 所有模块都只通过 contracts 传递状态和动作。

`frame_math.py`：

- 实现 4x4 齐次矩阵、位姿组合、逆变换、角度 wrap。
- 只用 NumPy 或纯 Python。
- 不使用任何外部坐标服务。

`observer.py`：

- 从 mock JSON、粗接近输出、示教日志构造 `GraspState`。
- 不产生控制动作。

`feature_geometry.py`：

- 计算局部误差和视觉几何特征。
- 不读取配置文件，不负责状态迁移。

`scores.py`：

- 计算 capture、hold、lift 的分数。
- 输出分数分项，便于调试。

`state_machine.py`：

- 只维护合法状态迁移。
- 不包含控制律。

`phase_guards.py`：

- 判断阶段切换和失败条件。
- 每个 guard 返回 `GuardResult`。

`nominal_capture.py`、`nominal_close.py`、`nominal_lift.py`：

- 输出名义动作 proposal。
- 不做安全裁剪。

`safety_limits.py`：

- 所有动作 proposal 的唯一出口。
- 负责限幅、硬停、异常动作拒绝。

`recovery_policy.py`：

- 将失败原因映射到恢复动作。
- 首版只输出 dry-run proposal。

`closed_loop_controller.py`：

- 串联 observer、guard、nominal、residual、safety、report。
- 不直接读取文件，由 app 层注入输入。

`command_adapter.py`：

- 首版只把动作 proposal 转成 JSON 日志。
- 不实现设备命令。
- 不导入或调用 `cjjarm`。
- 不导入或调用 `Gemini335`。

`sim/mujoco_adapter.py`：

- 可选模块。
- 只在 MuJoCo 已安装且项目已有模型资产时实现。
- 输入为 FRRG safe action 与仿真初始状态文件。
- 输出为仿真 rollout summary。
- 不作为 Step 00 到 Step 13 的前置条件。

---

## 4. 核心数据结构

### 4.1 Pose6D

```python
Pose6D = {
  "xyz": [float, float, float],
  "rpy": [float, float, float],
}
```

### 4.2 GraspState

```python
GraspState = {
  "timestamp": float,
  "phase": str,
  "mode": str,
  "retry_count": int,
  "stable_count": int,
  "phase_elapsed_s": float,

  "ee_pose_base": Pose6D,
  "object_pose_base": Pose6D,
  "ee_pose_object": Pose6D,

  "gripper_width": float,
  "gripper_cmd": float,
  "gripper_current_proxy": float,

  "vision_conf": float,
  "target_visible": bool,
  "corridor_center_px": [float, float],
  "object_center_px": [float, float],
  "object_axis_angle": float,
  "object_proj_width_px": float,
  "object_proj_height_px": float,

  "e_dep": float,
  "e_lat": float,
  "e_vert": float,
  "e_ang": float,
  "e_sym": float,
  "occ_corridor": float,
  "drift_obj": float,

  "capture_score": float,
  "hold_score": float,
  "lift_score": float,
}
```

### 4.3 GraspAction

```python
GraspAction = {
  "delta_pose_object": [dx, dy, dz, droll, dpitch, dyaw],
  "delta_gripper": float,
  "stop": bool,
  "reason": str | None,
  "debug_terms": dict,
}
```

首版只允许 `dx`、`dy`、`dz`、`dyaw`、`delta_gripper` 非零。

### 4.4 GuardResult

```python
GuardResult = {
  "passed": bool,
  "score": float,
  "reason": str | None,
  "debug_terms": dict,
}
```

### 4.5 StepReport

每个计划步骤必须输出：

```python
StepReport = {
  "step": str,
  "status": str,
  "completed_targets": list[str],
  "changed_files": list[str],
  "commands_run": list[str],
  "acceptance": dict,
  "known_gaps": list[str],
  "next_step_allowed": bool,
}
```

---

## 5. 核心算法规格

### 5.1 坐标与位姿

使用纯 Python 数学计算坐标关系：

\[
T^o_e=(T^b_o)^{-1}T^b_e
\]

其中：

- `base`：基准坐标系，只是数据中的参考系名称。
- `object`：目标局部坐标系。
- `end_effector`：末端执行器坐标系。
- `gripper`：夹爪语义坐标系。

夹爪语义轴约定：

- \(x_g\)：左右横向，两指闭合方向。
- \(y_g\)：竖向。
- \(z_g\)：前进或入爪方向。

### 5.2 局部误差

横向误差：

\[
e_{lat}^{px}=u_o-u_c
\]

如有深度和内参：

\[
e_{lat}=\frac{z_o}{f_x}e_{lat}^{px}
\]

没有可靠尺度时，必须标记单位为 `px`，不得把像素误差当成米制动作。

深度误差：

\[
e_{dep}=z^*-z^o_e
\]

竖向误差：

\[
e_{vert}=y^*-y^o_e
\]

首版没有可靠竖向估计时：

```text
e_vert = 0
dy = 0
debug_terms["e_vert_disabled"] = 1
```

左右对称误差：

\[
e_{sym}=\frac{d_L-d_R}{d_L+d_R+\epsilon}
\]

角度误差：

\[
e_{ang}=\mathrm{wrapToPi}(\alpha_o-\alpha_g)
\]

走廊占据率：

\[
occ_{corridor}=\frac{|M_o\cap C_t|}{|M_o|+\epsilon}
\]

漂移量：

\[
drift_{obj}(t)=
\left\|
(c_o(t)-c_c(t))-(c_o(t-1)-c_c(t-1))
\right\|_2
\]

### 5.3 状态机

固定状态机：

```text
HANDOFF
  -> CAPTURE_BUILD
    -> CLOSE_HOLD
      -> LIFT_TEST
        -> SUCCESS

任一执行阶段：
  -> RECOVERY
  -> FAILURE
```

合法阶段外的任何状态进入 `FAILURE`，原因是 `unknown_state`。

### 5.4 Capture Score

\[
score_{cap}=
\omega_1\exp(-e_{lat}^2/\sigma_{lat}^2)
+\omega_2\exp(-e_{ang}^2/\sigma_{ang}^2)
+\omega_3occ_{corridor}
+\omega_4\exp(-e_{sym}^2/\sigma_{sym}^2)
+\omega_5q_{vis}
\]

### 5.5 Forward Gate

\[
g_{fwd}=
\mathbb{1}[|e_{lat}|<\tau_{lat}^{fwd}]
\cdot
\mathbb{1}[|e_{ang}|<\tau_{ang}^{fwd}]
\cdot
\mathbb{1}[occ_{corridor}>\tau_{occ}^{fwd}]
\cdot
\mathbb{1}[q_{vis}>\tau_{vis}^{fwd}]
\]

### 5.6 Capture 控制律

\[
\Delta x=\mathrm{sat}(k_{lat}e_{lat}+k_{sym}e_{sym},\Delta x_{max})
\]

\[
\Delta y=\mathrm{sat}(k_{vert}e_{vert},\Delta y_{max})
\]

\[
\Delta z=\mathrm{sat}(g_{fwd}k_{dep}e_{dep},\Delta z_{max})
\]

\[
\Delta\psi=\mathrm{sat}(k_{ang}e_{ang},\Delta\psi_{max})
\]

`CAPTURE_BUILD` 阶段 `delta_gripper=0`。

### 5.7 Close Hold

闭合更新：

\[
g_{t+1}=\max(g_{min},g_t-v_{close}\Delta t)
\]

接触代理：

\[
\chi_{contact}=
\mathbb{1}[w_g<w_{open}^{cmd}-\tau_w]
\cdot
\mathbb{1}[I_g\in[I_{min},I_{max}]]
\]

没有电流代理时，必须在报告中标记：

```text
contact_current_unavailable: true
```

稳持评分：

\[
score_{hold}=
\eta_1\chi_{contact}
+\eta_2\exp(-drift_{obj}^2/\sigma_{drift}^2)
+\eta_3\mathbb{1}[w_g\in[w_{min}^{hold},w_{max}^{hold}]]
+\eta_4q_{vis}
\]

### 5.8 Lift Test

小提起 proposal：

\[
\Delta z_{lift}(t)=\min(v_{lift}\Delta t,h_{remain})
\]

滑移评分：

\[
score_{lift}=
\zeta_1\exp(-drift_{obj}^2/\sigma_{slip}^2)
+\zeta_2q_{vis}
+\zeta_3\mathbb{1}[\chi_{contact}=1]
\]

### 5.9 Safety Limiter

所有动作必须经过：

\[
u_{safe}=\Pi_{safe}(u_{nom}+u_{res})
\]

硬限制：

```text
|delta_x| <= max_step_xyz_m[0]
|delta_y| <= max_step_xyz_m[1]
|delta_z| <= max_step_xyz_m[2]
|delta_yaw| <= max_step_rpy_rad[2]
|delta_gripper| <= max_gripper_delta_m
```

硬停条件：

```text
vision_conf < vision_hardstop_min
object_jump > obj_jump_stop_m
non_finite_action == true
invalid_phase == true
```

硬停时：

```text
stop = true
delta_pose_object = [0, 0, 0, 0, 0, 0]
delta_gripper = 0
reason = explicit_failure_reason
```

### 5.10 Residual

首版只启用：

```text
residual_policy = zero
u_res = 0
```

BC residual 只保留接口，不进入首版执行闭环。

---

## 6. 配置规格

目标文件：

```text
src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml
```

建议内容：

```yaml
runtime:
  control_hz: 20
  max_steps: 100
  stable_window_frames: 5
  max_retry_count: 1
  default_input_mode: mock

feature_geometry:
  lateral_unit: m
  occ_definition: object_ratio
  angle_symmetry_weight_default: 1.0

handoff:
  handoff_pos_tol_m: 0.015
  handoff_yaw_tol_rad: 0.18
  handoff_vis_min: 0.45
  handoff_open_width_m: 0.06

capture_build:
  mode: enveloping
  solve_score_threshold: 0.55
  close_score_threshold: 0.72
  forward_enable_lat_tol_m: 0.004
  forward_enable_ang_tol_rad: 0.12
  forward_enable_occ_min: 0.55
  target_depth_goal_m: 0.012
  target_depth_max_m: 0.028
  lat_gain: 0.8
  sym_gain: 0.4
  dep_gain: 0.5
  vert_gain: 0.0
  yaw_gain: 0.6
  capture_timeout_s: 8.0

close_hold:
  preclose_distance_m: 0.006
  preclose_pause_s: 0.15
  close_speed_raw_per_s: 0.25
  settle_time_s: 0.25
  close_width_target_m: 0.0
  contact_current_min: 0.12
  contact_current_max: 0.85
  hold_drift_max_m: 0.003
  hold_score_threshold: 0.70

lift_test:
  lift_height_m: 0.015
  lift_speed_mps: 0.01
  lift_hold_s: 0.25
  slip_threshold_m: 0.004
  lift_score_threshold: 0.74

safety:
  max_step_xyz_m: [0.003, 0.003, 0.003]
  max_step_rpy_rad: [0.0, 0.0, 0.06]
  max_gripper_delta_m: 0.002
  vision_hardstop_min: 0.15
  obj_jump_stop_m: 0.02

residual:
  enabled: false
  policy: zero
```

配置验收必须检查：

- 字段完整。
- 单位明确。
- 所有数值有限。
- safety 上限为正。
- residual 默认关闭。

---

## 7. 严格分步实施计划

本节所有步骤按固定顺序执行，共 16 步。后续实现、验收和汇报都必须引用这里的步骤编号。

### 7.0 步骤总览

1. 第 00 步（01/16）：文档与环境确认
2. 第 01 步（02/16）：配置与 contracts 骨架
3. 第 02 步（03/16）：mock 输入与 dry-run 入口
4. 第 03 步（04/16）：纯 Python 位姿数学
5. 第 04 步（05/16）：Observer 与 GraspState 构造
6. 第 05 步（06/16）：Feature Geometry
7. 第 06 步（07/16）：Scores 与 Phase Guards
8. 第 07 步（08/16）：固定状态机
9. 第 08 步（09/16）：名义控制原语
10. 第 09 步（10/16）：Safety Limiter 与 Zero Residual
11. 第 10 步（11/16）：Closed Loop Controller
12. 第 11 步（12/16）：Recovery Policy 与错误报告
13. 第 12 步（13/16）：粗接近输出接入
14. 第 13 步（14/16）：示教参数提取
15. 第 14 步（15/16）：Benchmark 与仿真 / 纯代码验证
16. 第 15 步（16/16）：BC Residual 接口占位

---

### 第 00 步（01/16）：文档与环境确认

完成目标：

确认当前任务只面向纯 Python 执行路径，记录 `cjjarm` 和 `Gemini335` 是否存在，但不调用硬件，并建立后续步骤报告规范。

实施方案：

1. 检查仓库中是否存在 `src/lerobot/projects/vlbiman_sa`。
2. 检查 Python 导入路径。
3. 检查是否已有 FRRG 相关文件，避免误覆盖用户已有实现。
4. 扫描项目中 `cjjarm`、`Gemini335`、`gemini335` 相关文件，只记录路径，不 import、不实例化、不打开设备。
5. 检查 MuJoCo 是否可导入，只记录 `mujoco_available`，不得因此阻塞首版。
6. 建立 `outputs/vlbiman_sa/frrg/reports/`。
7. 输出 `step_00_environment.json`。

验收标准：

```bash
python -c "import sys; print(sys.version)"
```

```bash
test -d src/lerobot/projects/vlbiman_sa
```

```bash
rg -n "rospy|moveit|catkin|roslaunch|rosrun|geometry_msgs|sensor_msgs|tf2_ros" \
  src/lerobot/projects/vlbiman_sa tests || true
```

```bash
rg -n "cjjarm|Gemini335|gemini335" src tests || true
```

```bash
python -c "import importlib.util; print(importlib.util.find_spec('mujoco') is not None)"
```

通过条件：

- 项目目录存在。
- Python 命令可运行。
- 报告文件存在。
- 若发现 ROS 相关引用，必须记录为 `existing_ros_reference`，不得在 FRRG 新代码中继续引入。
- 若发现 `cjjarm` / `Gemini335` 相关引用，必须记录为 `existing_hardware_reference`，不得在 FRRG 执行步骤中调用。
- `mujoco_available` 只影响 Step 14 是否启用仿真分支，不影响 Step 01 到 Step 13。

下一步准入：

`step_00_environment.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 01 步（02/16）：配置与 contracts 骨架

完成目标：

建立 FRRG 的配置文件和核心数据结构，使后续所有模块有统一输入输出。

实施方案：

1. 新建 `configs/frrg_grasp.yaml`。
2. 新建 `grasp/contracts.py`。
3. 定义 `Pose6D`、`GraspState`、`GraspAction`、`GuardResult`、`StepReport`。
4. 实现配置加载函数和基础范围检查。
5. 编写 contracts 与 config 单元测试。
6. 输出 `step_01_contracts_config.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_contracts.py \
       tests/projects/vlbiman_sa/test_frrg_config.py -q
```

通过条件：

- 配置可以被加载。
- 缺失必需字段会报明确错误。
- 非法 safety 数值会报明确错误。
- contracts 不导入任何禁止依赖。

下一步准入：

`step_01_contracts_config.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 02 步（03/16）：mock 输入与 dry-run 入口

完成目标：

在不调用 `cjjarm`、不打开 `Gemini335`、不依赖任何真实数据源的情况下，可以运行 FRRG app 并输出 summary。

实施方案：

1. 新建 `app/run_frrg_grasp_dryrun.py`。
2. 新建最小 mock fixture：`frrg_handoff_ready.json`。
3. 实现读取配置、读取 mock state、生成 run 目录。
4. 先不做控制逻辑，只输出输入摘要和空 `phase_trace`。
5. 输出 `summary.json` 和 `step_02_dryrun_entry.json`。

验收标准：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --mock-state tests/fixtures/vlbiman_sa/frrg_handoff_ready.json \
  --max-steps 3
```

通过条件：

- 退出码为 0。
- 生成 `outputs/vlbiman_sa/frrg/dryrun/<run_id>/summary.json`。
- summary 包含 `status`、`config_path`、`mock_state_path`、`phase_trace`。
- 不要求任何外部服务。
- summary 必须包含 `hardware_called: false`。

下一步准入：

`step_02_dryrun_entry.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 03 步（04/16）：纯 Python 位姿数学

完成目标：

建立坐标变换所需的纯 Python 数学工具。

实施方案：

1. 新建 `grasp/frame_math.py`。
2. 实现 `pose6d_to_matrix`。
3. 实现 `matrix_to_pose6d`。
4. 实现 `invert_transform`。
5. 实现 `compose_transform`。
6. 实现 `wrap_to_pi`。
7. 编写确定性单元测试。
8. 输出 `step_03_frame_math.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_frame_math.py -q
```

通过条件：

- 单位矩阵往返正确。
- 位姿逆变换相乘接近单位阵。
- `wrap_to_pi` 输出在 `[-pi, pi]`。
- 不依赖任何禁止模块。

下一步准入：

`step_03_frame_math.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 04 步（05/16）：Observer 与 GraspState 构造

完成目标：

从 mock JSON 或离线 replay 字段构造完整 `GraspState`，不得从真实摄像头采集。

实施方案：

1. 新建 `grasp/observer.py`。
2. 从 mock JSON 读取末端位姿、目标位姿、夹爪状态、视觉字段。
3. 调用 `frame_math.py` 计算 `ee_pose_object`。
4. 对缺失字段使用显式默认值，并记录 `missing_fields`。
5. 更新 dry-run app，使每步都能得到 `GraspState`。
6. 如果字段来自 Gemini335 录制数据，只允许作为离线文件读取，不允许打开摄像头。
7. 输出 `step_04_observer.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_observer.py -q
```

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --mock-state tests/fixtures/vlbiman_sa/frrg_handoff_ready.json \
  --max-steps 1
```

通过条件：

- `GraspState` 字段完整。
- `ee_pose_object` 有有限数值。
- 缺失字段不会静默失败。
- summary 中记录最后一帧状态摘要。
- 报告中 `camera_opened` 必须为 `false`。

下一步准入：

`step_04_observer.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 05 步（06/16）：Feature Geometry

完成目标：

实现 FRRG 所需局部误差和几何特征。

实施方案：

1. 新建 `grasp/feature_geometry.py`。
2. 实现 `compute_lateral_error`。
3. 实现 `compute_depth_error`。
4. 实现 `compute_vertical_error`。
5. 实现 `compute_symmetry_error`。
6. 实现 `compute_angle_error`。
7. 实现 `compute_corridor_occupancy`。
8. 实现 `compute_object_drift`。
9. 将结果写回 `GraspState`。
10. 输出 `step_05_feature_geometry.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_feature_geometry.py -q
```

通过条件：

- 所有误差字段都是有限值。
- 像素单位和米单位不会混用。
- 对称误差在边界样例中符号正确。
- 角度误差跨 `pi` 边界时正确 wrap。

下一步准入：

`step_05_feature_geometry.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 06 步（07/16）：Scores 与 Phase Guards

完成目标：

实现 capture、hold、lift 三类评分和所有阶段 guard。

实施方案：

1. 新建 `grasp/scores.py`。
2. 实现 `compute_capture_score`。
3. 实现 `compute_hold_score`。
4. 实现 `compute_lift_score`。
5. 新建 `grasp/phase_guards.py`。
6. 实现 `handoff_guard`。
7. 实现 `capture_to_close_guard`。
8. 实现 `close_to_lift_guard`。
9. 实现 `lift_to_success_guard`。
10. 实现失败 guard。
11. 所有 guard 返回 `GuardResult`。
12. 输出 `step_06_scores_guards.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_scores.py \
       tests/projects/vlbiman_sa/test_frrg_phase_guards.py -q
```

通过条件：

- 每个 guard 都有 passed、score、reason、debug_terms。
- `vision_lost`、`capture_timeout`、`slip_detected` 可以被明确识别。
- 正例 fixture 能通过对应 guard。
- 反例 fixture 给出明确失败原因。

下一步准入：

`step_06_scores_guards.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 07 步（08/16）：固定状态机

完成目标：

实现合法状态迁移，禁止非法跳转。

实施方案：

1. 新建 `grasp/state_machine.py`。
2. 定义阶段枚举或常量。
3. 定义合法转移表。
4. 实现 `next_phase(current_phase, guard_results)`。
5. 实现 `phase_trace` 记录结构。
6. 编写成功链路和失败链路测试。
7. 输出 `step_07_state_machine.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_state_machine.py -q
```

通过条件：

- `HANDOFF -> CAPTURE_BUILD -> CLOSE_HOLD -> LIFT_TEST -> SUCCESS` 可走通。
- 任一阶段可以进入 `FAILURE`。
- recovery 次数超过上限进入 `FAILURE`。
- 非法跳转被拒绝并记录原因。

下一步准入：

`step_07_state_machine.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 08 步（09/16）：名义控制原语

完成目标：

实现 capture、close、lift 的名义动作 proposal。

实施方案：

1. 新建 `grasp/primitives/minimum_jerk.py`。
2. 新建 `grasp/primitives/nominal_capture.py`。
3. 新建 `grasp/primitives/nominal_close.py`。
4. 新建 `grasp/primitives/nominal_lift.py`。
5. capture 实现横向、深度、角度闭环。
6. close 实现夹爪宽度变化 proposal。
7. lift 实现小提起 proposal。
8. 单元测试验证动作方向和边界。
9. 输出 `step_08_nominal_primitives.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_nominal_primitives.py -q
```

通过条件：

- capture 在 forward gate 关闭时 `dz == 0`。
- capture 不改变夹爪。
- close 只改变夹爪或 preclose 小位移。
- lift 只产生受限小提起 proposal。
- 所有 proposal 未经 safety 前可以超过边界，但必须标记为 raw action。

下一步准入：

`step_08_nominal_primitives.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 09 步（10/16）：Safety Limiter 与 Zero Residual

完成目标：

保证所有动作 proposal 都经过统一 safety，且 residual 默认恒为 0。

实施方案：

1. 新建 `grasp/safety_limits.py`。
2. 新建 `grasp/residual/policy.py`。
3. 新建 `grasp/residual/zero_policy.py`。
4. 实现平移、旋转、夹爪限幅。
5. 实现视觉置信度硬停。
6. 实现非有限值硬停。
7. 实现 object jump 硬停。
8. 实现 residual 输出统计。
9. 输出 `step_09_safety_zero_residual.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_safety_limits.py \
       tests/projects/vlbiman_sa/test_frrg_zero_residual.py -q
```

通过条件：

- 超限动作被裁剪。
- 硬停条件下动作全为 0。
- residual norm 始终为 0。
- safety 输出包含 `raw_action` 和 `safe_action` 对比。

下一步准入：

`step_09_safety_zero_residual.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 10 步（11/16）：Closed Loop Controller

完成目标：

串联 observer、features、scores、guards、state machine、nominal、residual、safety，形成完整 dry-run 闭环。

实施方案：

1. 新建 `grasp/closed_loop_controller.py`。
2. 实现 `controller.step(state)`。
3. 实现 `controller.run(max_steps)`。
4. 每一步记录 state、guard、raw action、safe action、phase transition。
5. 更新 `run_frrg_grasp_dryrun.py` 使用 controller。
6. 用 success fixture 跑到 `SUCCESS`。
7. 用 failure fixtures 跑到明确 `FAILURE`。
8. 输出 `step_10_closed_loop_controller.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_controller.py -q
```

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --mock-state tests/fixtures/vlbiman_sa/frrg_nominal_success.json \
  --max-steps 80
```

通过条件：

- success fixture 到达 `SUCCESS`。
- failure fixture 给出明确失败原因。
- 所有 step 都有 safe action。
- summary 中有完整 `phase_trace`。
- `max_residual_norm == 0.0`。

下一步准入：

`step_10_closed_loop_controller.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 11 步（12/16）：Recovery Policy 与错误报告

完成目标：

实现可复核的恢复策略和失败报告。

实施方案：

1. 新建 `grasp/recovery_policy.py`。
2. 定义 failure reason 到 recovery action 的映射。
3. 支持 `backoff`、`half_open`、`recenter`、`abort` 四种 proposal。
4. 在 controller 中加入 retry count。
5. 输出 failure report。
6. 输出 `step_11_recovery_reports.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_recovery_policy.py -q
```

通过条件：

- `capture_timeout` 先 recovery，超过重试上限 failure。
- `vision_lost` 硬停并给出明确原因。
- `object_jump` 硬停并给出明确原因。
- failure report 包含最后状态、失败原因、safe action、phase trace。

下一步准入：

`step_11_recovery_reports.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 12 步（13/16）：粗接近输出接入

完成目标：

从现有 VLBiMan 粗接近输出文件或 Python 对象构造 FRRG handoff，不引入任何外部中间件。

实施方案：

1. 新建 `app/run_frrg_from_coarse.py`。
2. 定义 `coarse_handoff_summary.json` fixture。
3. 明确 handoff 需要字段：目标位姿、末端预抓位姿、夹爪初值、视觉摘要。
4. 将粗接近输出转换为 `GraspState` 初始输入。
5. 缺失字段必须报告 `missing_coarse_field`。
6. 不允许猜测字段含义。
7. 输出 `step_12_coarse_handoff.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_coarse_handoff.py -q
```

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_coarse.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --coarse-summary tests/fixtures/vlbiman_sa/coarse_handoff_summary.json \
  --max-steps 80
```

通过条件：

- 可从 coarse summary 启动 FRRG。
- summary 包含 `handoff_source: coarse_python`。
- 字段缺失时明确失败。
- 不出现任何禁止依赖。

下一步准入：

`step_12_coarse_handoff.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 13 步（14/16）：示教参数提取

完成目标：

从 one-shot 示教或已有技能片段中提取 `theta`，用于替代纯默认参数。

实施方案：

1. 新建 `grasp/parameters/theta_schema.py`。
2. 新建 `grasp/parameters/default_provider.py`。
3. 新建 `grasp/parameters/demo_extractor.py`。
4. 新建 `app/run_frrg_extract_demo.py`。
5. 从示教片段提取 `advance_speed_mps`。
6. 从示教片段提取 `preclose_distance_m`。
7. 从示教片段提取 `close_speed_raw_per_s`。
8. 从示教片段提取 `settle_time_s`。
9. 从示教片段提取 `lift_height_m`。
10. 对无法提取的字段使用默认值并记录原因。
11. 输出 `step_13_demo_theta.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_demo_extractor.py -q
```

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_extract_demo.py \
  --session-dir <one_shot_session_dir> \
  --output-dir outputs/vlbiman_sa/frrg/demo_params/<session_name>
```

通过条件：

- 生成 `theta_samples.json`。
- 生成 `extraction_report.json`。
- 每个参数有来源或默认原因。
- theta 可以被 dry-run app 加载。

下一步准入：

`step_13_demo_theta.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 14 步（15/16）：Benchmark 与仿真 / 纯代码验证

完成目标：

建立离线评测入口，对 mock / replay 数据运行 FRRG 并统计指标；如果 MuJoCo 可用，再增加仿真验证。无论是否启用 MuJoCo，都不得调用真实 `cjjarm` 或 `Gemini335`。

实施方案：

1. 新建 `app/run_frrg_benchmark.py`。
2. 支持输入多个 fixture 或 replay session。
3. 统计每个 episode 的 phase trace。
4. 统计成功率、失败原因、平均步数、最大动作幅度。
5. 生成 `metrics.json`、`episodes.jsonl`、`report.md`。
6. 检查 `mujoco_available`。
7. 如果 MuJoCo 不可用，报告中写入 `mujoco_available: false`，使用纯代码 benchmark 作为验收依据。
8. 如果 MuJoCo 可用且已有模型资产，则可新增 `app/run_frrg_mujoco_validate.py` 和 `sim/mujoco_adapter.py`，将 FRRG safe action 转成仿真 rollout，不接触真实硬件。
9. MuJoCo 仿真输出 `mujoco_rollouts.jsonl` 和 `mujoco_report.md`。
10. 输出 `step_14_benchmark.json`。

验收标准：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_benchmark.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --fixtures tests/fixtures/vlbiman_sa \
  --output-dir outputs/vlbiman_sa/frrg/benchmarks/mock_suite
```

通过条件：

- 生成 `metrics.json`。
- 生成 `episodes.jsonl`。
- 生成 `report.md`。
- 指标包含 success rate、failure reason top list、max action step。
- `hardware_called` 必须为 `false`。
- 如果 `mujoco_available: false`，纯代码 benchmark 通过即可进入下一步。
- 如果 `mujoco_available: true` 且提供模型资产，MuJoCo rollout 不得出现非有限状态或越界 action。

下一步准入：

`step_14_benchmark.json` 中 `next_step_allowed` 必须为 `true`。

---

### 第 15 步（16/16）：BC Residual 接口占位

完成目标：

只建立 residual 学习接口，不启用在线 residual。

实施方案：

1. 新建 `grasp/residual/bc_policy.py`。
2. 新建 `grasp/residual/train_bc_residual.py`。
3. 定义 residual label：

\[
u_{res}^{demo}=u_{demo}-u_{nom}
\]

4. 实现 label clipping。
5. 实现模型加载失败时自动回退 zero policy。
6. 配置中保持 `residual.enabled: false`。
7. 输出 `step_15_residual_interface.json`。

验收标准：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_zero_residual.py -q
```

通过条件：

- 默认 residual 仍为 0。
- residual disabled 时不加载模型。
- residual 接口不影响 Step 10 的 dry-run 成功。
- 任何 residual 输出仍必须经过 safety。

下一步准入：

该步骤是首版最后一步。完成后才允许讨论是否启用学习 residual。

---

## 8. 标准失败原因

必须统一使用以下失败原因：

```text
config_invalid
mock_state_invalid
missing_input_field
missing_coarse_field
frame_math_error
feature_invalid
vision_lost
capture_timeout
corridor_not_formed
contact_not_detected
large_drift
slip_detected
object_jump
non_finite_action
invalid_phase
hardware_call_attempted
camera_open_attempted
missing_sim_asset
max_retry_exceeded
unknown_state
```

`unknown_state` 只能作为兜底。只要出现，必须补测试或补明确原因。

---

## 9. 输出报告规范

每次 dry-run 必须输出：

```text
summary.json
phase_trace.jsonl
actions.jsonl
states.jsonl
guards.jsonl
```

`summary.json` 至少包含：

```json
{
  "status": "success_or_failure",
  "final_phase": "SUCCESS_or_FAILURE",
  "failure_reason": null,
  "max_steps": 80,
  "steps_run": 0,
  "phase_trace": [],
  "max_raw_action_norm": 0.0,
  "max_safe_action_norm": 0.0,
  "max_residual_norm": 0.0,
  "all_actions_limited": true,
  "input_mode": "mock_or_coarse_python_or_replay_or_mujoco",
  "hardware_called": false,
  "camera_opened": false,
  "mujoco_available": false,
  "mujoco_validation_status": "not_run_or_passed_or_failed"
}
```

每个 action 记录必须包含：

```json
{
  "step": 0,
  "phase": "CAPTURE_BUILD",
  "raw_action": {},
  "residual_action": {},
  "safe_action": {},
  "limited": true,
  "stop": false,
  "reason": null
}
```

---

## 10. 最终交付清单

首版最终必须交付：

1. `计划书/FRRG计划书.md`。
2. `configs/frrg_grasp.yaml`。
3. `grasp/contracts.py`。
4. `grasp/frame_math.py`。
5. `grasp/observer.py`。
6. `grasp/feature_geometry.py`。
7. `grasp/scores.py`。
8. `grasp/phase_guards.py`。
9. `grasp/state_machine.py`。
10. `grasp/primitives/minimum_jerk.py`。
11. `grasp/primitives/nominal_capture.py`。
12. `grasp/primitives/nominal_close.py`。
13. `grasp/primitives/nominal_lift.py`。
14. `grasp/safety_limits.py`。
15. `grasp/recovery_policy.py`。
16. `grasp/residual/zero_policy.py`。
17. `grasp/closed_loop_controller.py`。
18. `app/run_frrg_grasp_dryrun.py`。
19. `app/run_frrg_from_coarse.py`。
20. `app/run_frrg_extract_demo.py`。
21. `app/run_frrg_benchmark.py`。
22. 可选：`app/run_frrg_mujoco_validate.py`，仅在 MuJoCo 可用且有模型资产时交付。
23. 可选：`sim/mujoco_adapter.py`，仅在 MuJoCo 可用且有模型资产时交付。
24. 对应 pytest 测试。
25. 对应 mock fixtures。
26. 每一步 `step_XX_*.json` 验收报告。
27. 报告中明确 `hardware_called: false` 和 `camera_opened: false`。

---

## 11. 后续行动起点

后续真正开始写代码时，必须从 `Step 00` 开始。

第一项实现任务不是 controller，也不是 residual，而是：

```text
Step 00：文档与环境确认
```

只有 `Step 00` 通过，才允许执行：

```text
Step 01：配置与 contracts 骨架
```

本计划书中的步骤顺序即后续工作顺序，不再按旧版 A-H 或 ROS/设备接入流程执行。
