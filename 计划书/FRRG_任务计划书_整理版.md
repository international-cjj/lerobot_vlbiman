# FRRG 接入 VLBiMan 的底层抓取闭环任务计划书（整理补全版）

## 一、任务目的

### 1.1 任务定位

本任务的目标，是在现有 `vlbiman_sa` 单臂执行链路中，新增一层 **粗接近之后的底层抓取闭环执行端**，用于替代原流程中对抓取末段与接触段的 `inv` 不变轨迹重放。

也就是说，本任务**不重复建设**以下能力：

- one-shot 示教采集
- 技能切分
- 视觉锚点估计
- 位姿适配
- 粗接近轨迹生成

这些能力继续由 VLBiMan 上层负责。FRRG 只接管**粗接近已经到位后的最后几厘米抓取执行**，包括：

1. 交接确认
2. 入爪构型建立
3. 闭爪接触确认
4. 稳持确认
5. 小提起验证
6. 失败恢复与终止上报

因此，系统职责链路从原来的：

```text
one-shot demo
  -> T3 skill_bank
  -> T4 vision anchor
  -> T5 pose adaptation
  -> T6 trajectory generation
  -> T7 robot execution
```

调整为：

```text
one-shot demo
  -> T3 skill_bank
  -> T4 vision anchor
  -> T5 pose adaptation
  -> T6 coarse approach trajectory
  -> FRRG grasp closed loop
  -> T7 robot execution / report
```

### 1.2 核心任务目标

本任务最终要把“抓取成功”的判断依据，从：

- 轨迹放完了

改成：

- 目标进入两指有效通道
- 夹爪闭合后出现接触
- 稳持阶段目标相对夹爪不明显漂移
- 小提起后目标未滑脱

因此，FRRG 首版的目标不是“智能程度最高”，而是“执行端逻辑最清晰、最可验收、最安全”。

### 1.3 首版工程原则

首版严格采用：

**固定流程 + 示教参数化 + 局部残差修正**

含义如下：

1. 主流程由固定状态机维护，不交给学习模型决定。
2. 阶段参数可以从示教中提取，但安全边界不学习。
3. 末段短程动作可由参数化模板或 DMP 表达。
4. 残差只允许在安全限幅内做很小修正。
5. 所有失败必须可归因，不能只给出 `unknown_error`。

### 1.4 成功定义

FRRG 首版完成，必须同时满足以下 9 项：

1. 粗接近后的抓取不再依赖原 `inv` 末段重放。
2. `HANDOFF -> CAPTURE_BUILD -> CLOSE_HOLD -> LIFT_TEST` 状态机可 dry-run 完整跑通。
3. 所有动作都经过安全限幅。
4. 每个失败都有明确原因与报告文件。
5. 可以从 T6 输出直接启动 FRRG。
6. 可以从 one-shot 示教提取 `theta` 阶段参数。
7. 真机 dry-run 能完整读状态但不发动作。
8. 真机可按 `--stop-after-phase` 分阶段验收。
9. 小规模评测可与原 `inv` 重放基线对比。

---

## 二、任务具体实施办法

### 2.1 环境搭建

#### 2.1.1 系统分层

本任务建议采用三层运行结构：

```text
进程 A：ROS + MoveIt + 机械臂驱动
进程 B：vlbiman_sa + FRRG 主控制器
进程 C：视觉模型/残差模型（可选）
```

原因：

- ROS / MoveIt 负责系统控制与规划。
- FRRG 负责抓取执行逻辑与状态机。
- 视觉模型与学习模型独立进程运行，避免依赖冲突。

#### 2.1.2 ROS / MoveIt 层

保持当前 ROS Noetic + catkin + MoveIt 体系，不建议为了接入 FRRG 重构底层。

最少要求：

- 机械臂 URDF / SRDF 可正常加载
- MoveIt planning group 可用
- 夹爪控制接口可调用
- 手眼相机 TF 链路可读
- 现有 T5 / T6 粗接近流程可运行

#### 2.1.3 Python 算法层

FRRG 接入 `lerobot_vlbiman` 项目内，独立维护配置、主控制器、观察器、状态机与验收脚本。

建议最少环境能力：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

以及项目导入检查：

```bash
PYTHONPATH=/home/cjj/lerobot_vlbiman/src:/home/cjj/lerobot_vlbiman \
python -c "import lerobot; import src.lerobot.projects.vlbiman_sa"
```

### 2.2 代码架构

建议目录如下：

```text
src/lerobot/projects/vlbiman_sa/
  configs/
    frrg_grasp.yaml

  core/
    contracts.py
    grasp_orchestrator.py

  grasp/
    contracts.py
    state_machine.py
    observer.py
    phase_guards.py
    safety_limits.py
    recovery_policy.py
    closed_loop_controller.py
    command_adapter.py

  grasp/parameters/
    theta_schema.py
    default_provider.py
    demo_extractor.py
    regression_provider.py

  grasp/primitives/
    nominal_capture.py
    nominal_close.py
    nominal_lift.py
    dmp.py

  grasp/residual/
    policy.py
    zero_policy.py
    bc_policy.py
    train_bc_residual.py

  app/
    run_frrg_grasp_dryrun.py
    run_frrg_from_t6.py
    run_frrg_extract_demo.py
    run_frrg_benchmark.py
```

输出目录统一为：

```text
outputs/vlbiman_sa/frrg/
  dryrun/
  demo_params/
  robot_runs/
  reports/
```

### 2.3 在线运行架构

FRRG 在线数据流固定为：

```text
T6 粗接近执行到预抓壳层 T_pre
  -> FRRGHandoffState
  -> FrameManager
  -> GraspStateObserver
  -> PhaseParameterProvider
  -> NominalPrimitiveRunner
  -> ResidualPolicy
  -> PhaseGuardEvaluator
  -> SafetyLimiter
  -> GraspClosedLoopController
  -> RobotCommandAdapter
  -> 机器人执行 / dry-run 日志输出
```

### 2.4 状态机设计

状态机必须固定为：

```text
HANDOFF
  -> CAPTURE_BUILD
    -> CLOSE_HOLD
      -> LIFT_TEST
        -> SUCCESS

任一阶段：
  -> RECOVERY
  -> FAILURE
```

各阶段含义：

- `HANDOFF`：确认粗接近已到位、目标可见、夹爪已张开。
- `CAPTURE_BUILD`：最后几厘米微推进与微对位，直到目标进入两指通道。
- `CLOSE_HOLD`：闭爪，判断是否真的夹住而不是空夹。
- `LIFT_TEST`：小幅提起，确认目标没有滑脱。
- `RECOVERY`：微退、半开夹爪、重置局部判定，再尝试一次。
- `FAILURE`：终止动作，上报失败原因。

### 2.5 数学建模

这部分为本任务必须补齐的核心内容。

#### 2.5.1 坐标系定义

定义 5 个坐标系：

- `B`：机械臂基座坐标系
- `O`：目标物体局部坐标系
- `G`：抓取走廊坐标系
- `E`：末端执行器坐标系
- `C`：手眼相机坐标系

每周期先计算：

\[
{}^{O}T_{E}(t)=({}^{B}T_{O}(t))^{-1}{}^{B}T_{E}(t)
\]

这样 FRRG 的末段控制在“末端相对物体”的坐标下进行，而不是直接在 base 坐标下盲动。

#### 2.5.2 状态向量

每周期维护：

\[
s_t=[{}^{O}T_E(t), q_t, g_t, I_t, \phi_t, r_t, z_t^{vis}]
\]

其中：

- \(q_t\)：关节角
- \(g_t\)：夹爪开口
- \(I_t\)：夹爪电流或接触替代量
- \(\phi_t\)：当前阶段
- \(r_t\)：重试次数
- \(z_t^{vis}\)：视觉观测特征

#### 2.5.3 局部误差定义

FRRG 首版必须显式维护以下误差：

\[
e_t=
\begin{bmatrix}
e_{dep} \\
e_{lat} \\
e_{vert} \\
e_{ang} \\
e_{sym} \\
e_{occ}
\end{bmatrix}
\]

定义如下：

1. 深度误差

\[
e_{dep}=x^{G}_{ref}-x^{G}_{tip}
\]

2. 横向误差

\[
e_{lat}=y^{G}_{obj\_center}-y^{G}_{tip}
\]

3. 竖向误差

\[
e_{vert}=z^{G}_{ref}-z^{G}_{tip}
\]

4. 朝向误差

\[
e_{ang}=\mathrm{wrap}(\psi_{obj}-\psi_{gripper})
\]

5. 左右对称误差

\[
e_{sym}=\frac{d_L-d_R}{d_L+d_R+\varepsilon}
\]

6. 通道占据率

\[
e_{occ}=\frac{|M_{obj}\cap M_{corridor}|}{|M_{corridor}|}
\]

其中：

- \(d_L,d_R\) 为目标左右边缘到夹爪中线的距离
- \(M_{obj}\) 为目标掩膜
- \(M_{corridor}\) 为两指之间的有效通道掩膜

#### 2.5.4 滑移检测量

稳持与小提起阶段必须计算目标相对夹爪的漂移量：

\[
e_{drift}(t)=\left\|{}^{E}p_{obj}(t)-{}^{E}p_{obj}(t-\Delta t)\right\|_2
\]

注意：这里必须是“目标相对夹爪”的漂移，而不能只看“目标相对相机”的漂移。

#### 2.5.5 名义控制律

##### Capture-Build 阶段

\[
u^{cap}_{nom}(t)=
\begin{bmatrix}
\Delta x \\
\Delta y \\
\Delta z \\
\Delta \psi \\
\Delta g
\end{bmatrix}
=
\begin{bmatrix}
\mathrm{sat}(k_x e_{dep}+v_f\Delta t,d_{x,max}) \\
\mathrm{sat}(k_y e_{lat},d_{y,max}) \\
\mathrm{sat}(k_z e_{vert},d_{z,max}) \\
\mathrm{sat}(k_\psi e_{ang},\psi_{max}) \\
0
\end{bmatrix}
\]

含义：

- 沿推进方向固定小步前进
- 用横向误差修正左右偏差
- 用角度误差修正偏航
- 夹爪在该阶段不闭合

##### Close-Hold 阶段

夹爪闭合更新为：

\[
g_{t+1}=\max(g_{min}, g_t-v_{close}\Delta t)
\]

当满足下列条件，判定为“已接触”：

\[
g_t\in[g^c_{min},g^c_{max}]\land I_t\in[I_{min},I_{max}]\land e_{drift}(t)<\tau_{drift,close}
\]

##### Lift-Test 阶段

\[
\Delta z_{lift}(t)=\mathrm{sat}(h_{lift}-h_{acc}(t),d_{z,max})
\]

若在时间窗 \(T_{lift}\) 内有：

\[
\max_{t\in T_{lift}} e_{drift}(t)<\tau_{slip}
\]

则通过小提起验证。

#### 2.5.6 阶段切换判据

##### HANDOFF -> CAPTURE_BUILD

\[
\|{}^Bp_E-{}^Bp_{pre}\|_2<\tau_{pre}
\land vis\_ok=1
\land g_t>g_{open,min}
\]

##### CAPTURE_BUILD -> CLOSE_HOLD

\[
|e_{lat}|<\tau_{lat}
\land |e_{ang}|<\tau_{ang}
\land e_{occ}>\tau_{occ}
\land |e_{dep}|<\tau_{dep}
\]

并连续满足 \(K_{stable}\) 帧：

\[
K_{stable}=\left\lceil\frac{T_{stable}}{\Delta t}\right\rceil
\]

##### CLOSE_HOLD -> LIFT_TEST

\[
contact\_ok=1\land e_{drift}<\tau_{drift,close}
\]

##### LIFT_TEST -> SUCCESS

\[
\max_{t\in T_{lift}} e_{drift}(t)<\tau_{slip}\land vis\_ok=1
\]

#### 2.5.7 安全限幅

所有输出必须经过统一安全投影：

\[
u_{safe}(t)=\Pi_{\mathcal U}(u_{nom}(t)+u_{res}(t))
\]

其中安全集合为：

\[
\mathcal U=\{u:|\Delta p_i|\le d_{i,max}, |\Delta r_i|\le r_{i,max}, |\Delta q_j|\le q_{j,max}, |\Delta g|\le g_{max}\}
\]

并设置硬停止条件：

\[
vis\_conf<\tau_{vis}
\lor ik\_fail=1
\lor jump\_obj>\tau_{jump}
\Rightarrow u_{safe}=0
\]

#### 2.5.8 示教参数提取

从示教段 \(\mathcal D=\{p_t,R_t,g_t\}_{t=t_a}^{t_b}\) 提取：

\[
pregrasp\_offset=\|p_{handoff}-p_{capture\_start}\|_2
\]

\[
advance\_speed=\mathrm{median}\left(\frac{\|p_{t+1}-p_t\|_2}{\Delta t}\right)
\]

\[
preclose\_distance=\langle p_{close\_trigger}-p_{corridor\_entry},\hat x_g\rangle
\]

\[
close\_speed=\mathrm{median}\left(-\frac{g_{t+1}-g_t}{\Delta t}\right)
\]

\[
lift\_height=z_{peak}-z_{lift\_start}
\]

\[
settle\_time=t_{stable}-t_{close\_end}
\]

\[
guard\_stable\_frames=\left\lceil\frac{T_{stable}}{\Delta t}\right\rceil
\]

#### 2.5.9 DMP 与残差接口

若后续启用 DMP，统一接口为：

\[
\tau \dot z=\alpha_z(\beta_z(g-y)-z)+f(x)
\]

\[
\tau \dot y=z
\]

\[
\tau \dot x=-\alpha_x x
\]

残差策略仅允许输出小修正：

\[
u_{res}(t)=\pi_\phi(s_t),\quad \|u_{res}(t)\|_\infty\le \rho
\]

BC 标签定义为：

\[
\hat u_{res}^{demo}(t)=\mathrm{clip}(u_{demo}(t)-u_{nom}(t),\rho)
\]

训练损失：

\[
\mathcal L_{BC}=\sum_t\|\pi_\phi(s_t)-\hat u_{res}^{demo}(t)\|_2^2
\]

### 2.6 配置参数 `theta`

首版 `theta` 至少包含：

```text
theta = {
  pregrasp_offset_m,
  corridor_half_width_m,
  advance_speed_mps,
  angular_speed_radps,
  preclose_distance_m,
  preclose_pause_s,
  close_speed_raw_per_s,
  settle_time_s,
  lift_height_m,
  slip_threshold_m,
  guard_stable_frames,
  max_retry_count
}
```

并要求在配置加载时做范围检查。

建议硬限制：

```text
advance_speed_mps <= 0.03
lift_height_m <= 0.04
|delta_xy| 单步 <= 0.004m
|delta_z| 单步 <= 0.003m
|delta_yaw| 单步 <= 5deg
```

### 2.7 基础实施顺序

实施顺序固定如下：

1. 先完成模块骨架和 dry-run。
2. 再完成 observer、状态机、guard。
3. 再完成名义控制器、安全限幅、恢复策略。
4. 再接 T6 handoff。
5. 再做示教参数提取。
6. 再补零残差接口。
7. 最后才做真机分阶段验证和 benchmark。

---

## 三、任务分解

下面将原来的 T1-T12 收束为 8 个工程阶段。每个阶段都必须包括：任务目标、验收标准、指令输出、阶段结果汇报。

### 阶段 A：环境冻结与基线验证

#### 任务目标

冻结当前可运行环境，确认 FRRG 接入前，原有 VLBiMan 粗接近链路可正常运行。

#### 验收标准

执行：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

以及原有 dry-run 主入口，要求：

- Python 导入正常
- 现有 pipeline 能运行
- 有日志输出

#### 指令输出

```text
status: success / failure
env_name:
python_version:
ros_workspace:
vlbiman_branch:
```

#### 阶段结果汇报

```text
阶段：A 环境冻结与基线验证
完成内容：
- 环境信息采集
- 原 pipeline dry-run 验证
验收结果：
- torch 导入是否通过
- 原 dry-run 是否通过
风险与遗留：
- 版本冲突点
- 待确认硬件接口
```

### 阶段 B：FRRG 骨架与配置建立

#### 任务目标

建立 `frrg_grasp.yaml`、`contracts.py`、`run_frrg_grasp_dryrun.py`，保证无硬件也能构造 FRRG 控制链路。

#### 验收标准

执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --max-steps 3 \
  --dry-run
```

要求：

- 退出码为 0
- 生成 `summary.json`
- 至少包含 `status`、`dry_run`、`phase_trace`、`config_path`

#### 指令输出

```text
status: ok
dry_run: true
config_loaded: true
phase_trace: []
```

#### 阶段结果汇报

必须汇报：

- 新增文件列表
- 配置字段列表
- 默认安全限幅值
- 是否改动现有 T3-T6 代码

### 阶段 C：状态观测与坐标系接入

#### 任务目标

完成 `observer.py`，使 FRRG 每周期都能得到完整 `GraspState`。

#### 验收标准

执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --mock-state tests/fixtures/vlbiman_sa/frrg_capture_ready.json \
  --max-steps 5 \
  --dry-run
```

要求最后一帧包含：

- `ee_pose_object`
- `e_dep`
- `e_lat`
- `e_vert`
- `e_ang`
- `e_sym`
- `e_occ`
- `e_drift`

#### 指令输出

```text
observer_ok: true
state_fields_complete: true
frame_chain: camera->base->object->tool
```

#### 阶段结果汇报

必须汇报：

- `GraspState` 字段定义
- 坐标链路
- mock 样本最后一帧误差数值

### 阶段 D：状态机与判定逻辑

#### 任务目标

实现固定状态机与所有阶段 guard。

#### 验收标准

执行：

```bash
pytest tests/projects/vlbiman_sa/test_frrg_state_machine.py \
       tests/projects/vlbiman_sa/test_frrg_phase_guards.py -q
```

测试必须覆盖：

- 正常成功链路
- `capture_timeout -> RECOVERY`
- `vision_lost -> FAILURE`
- `slip_detected -> FAILURE`
- `max_retry_exceeded -> FAILURE`

#### 指令输出

```text
phase_machine_ok: true
legal_transition_only: true
failure_reasons:
- vision_lost
- capture_timeout
- corridor_not_formed
- contact_not_detected
- slip_detected
- ik_fail
- object_jump
- max_retry_exceeded
```

#### 阶段结果汇报

必须给出：

- 状态转移表
- 每个失败原因的恢复策略
- 正反例测试覆盖数量

### 阶段 E：名义控制、安全限幅、恢复策略

#### 任务目标

实现 `CAPTURE_BUILD / CLOSE_HOLD / LIFT_TEST` 的名义动作、安全限幅与恢复策略。

#### 验收标准

执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_grasp_dryrun.py \
  --config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --mock-state tests/fixtures/vlbiman_sa/frrg_nominal_success.json \
  --max-steps 80 \
  --dry-run
```

要求：

- `phase_trace` 走到 SUCCESS
- 所有动作都满足限幅
- 恢复测试中必须能进入 `RECOVERY`

#### 指令输出

```text
nominal_controller_ok: true
all_actions_limited: true
recovery_enabled: true
```

#### 阶段结果汇报

必须汇报：

- 最大单步平移
- 最大单步旋转
- 最大单步夹爪变化
- 恢复动作最大幅度
- 是否存在未经过 limiter 的动作出口

### 阶段 F：与 T6 粗接近接入

#### 任务目标

从现有 T6 输出启动 FRRG，并正式替代原末段 `inv` 重放作为抓取成功判断。

#### 验收标准

执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_t6.py \
  --task-config <task_config.yaml> \
  --frrg-config src/lerobot/projects/vlbiman_sa/configs/frrg_grasp.yaml \
  --trajectory-points <trajectory_points.json> \
  --trajectory-summary <summary.json> \
  --dry-run
```

要求输出：

- `handoff_source: t6`
- `handoff_ok: true`
- `final_status`
- `phase_trace`

#### 指令输出

```text
handoff_source: t6
handoff_ok: true
inv_replay_replaced: true
```

#### 阶段结果汇报

必须说明：

- handoff 用了哪些字段
- 是否已经不再把原 `inv` 末段作为成功依据
- 是否影响原 pipeline 回归测试

### 阶段 G：示教参数提取与零残差接口

#### 任务目标

从 one-shot 示教中提取 `theta`，并启用 `zero_policy` 跑通完整 FRRG。

#### 验收标准

执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_extract_demo.py \
  --session-dir <one_shot_session_dir> \
  --skill-bank-path <skill_bank.json> \
  --output-dir outputs/vlbiman_sa/frrg/demo_params/<session_name>
```

要求生成：

- `theta_samples.json`
- `extraction_report.json`

再执行 dry-run，要求报告中出现：

```text
residual_policy: zero
max_residual_norm: 0.0
```

#### 指令输出

```text
demo_extractor_ok: true
theta_samples_generated: true
residual_policy: zero
```

#### 阶段结果汇报

必须说明：

- 每个参数来源于哪一段示教帧
- 哪些参数用默认值补齐
- `theta` 样例内容

### 阶段 H：真机分阶段验收与小规模评测

#### 任务目标

先做真机 dry-run，再按阶段放行动作，最后做小规模 benchmark。

#### 验收标准

依次执行：

```bash
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_t6.py ... --stop-after-phase HANDOFF
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_t6.py ... --stop-after-phase CAPTURE_BUILD
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_t6.py ... --stop-after-phase CLOSE_HOLD
python src/lerobot/projects/vlbiman_sa/app/run_frrg_from_t6.py ... --stop-after-phase LIFT_TEST
```

每阶段要求：

- 无越界动作
- 失败时立即停机
- 有报告文件
- 人工可复核

最后执行 benchmark，生成：

- `metrics.json`
- `episodes.jsonl`
- `failure_cases/`
- `report.md`

#### 指令输出

```text
robot_dryrun_ok: true
stop_after_phase_ok: true
benchmark_ok: true
capture_success_rate:
close_hold_success_rate:
lift_test_success_rate:
```

#### 阶段结果汇报

必须汇报：

- capture 成功率
- close-hold 成功率
- lift-test 成功率
- 平均重试次数
- 失败原因 Top 3
- 相比原 `inv` 重放的改进或退化

---

## 四、错误审查流程

本流程的核心原则是：

**越不触动底层系统，优先级越高；真正修改底层控制与硬件参数的方法，必须放在最低优先级。**

### 审查级别 0：文档与接口审查

只检查：

- 配置字段是否完整
- `theta` 是否有单位、上下限
- `GraspState / GraspAction / GraspReport` 是否闭合
- 失败原因集合是否标准化

特点：

- 不连硬件
- 不发动作
- 不改代码逻辑

### 审查级别 1：离线回放审查

只读历史日志、mock 状态、示教数据，检查：

- 坐标系是否连贯
- 局部误差是否数值稳定
- guard 是否会误切换
- 名义控制是否抖动
- 示教参数提取是否异常

如果这一级未通过，禁止进入真机阶段。

### 审查级别 2：真机 dry-run 审查

连真机，但不发动作，只运行：

- 传感器读取
- T6 handoff 读取
- FRRG 全链路计算
- 动作建议日志输出

要求：

- 能持续读取状态
- 能生成阶段判定
- 能生成建议动作
- 但绝不下发动作

### 审查级别 3：影子模式审查

机械臂仍按旧流程或人工控制，FRRG 只在后台计算：

- 它本来想怎么动
- 它会在什么时刻触发恢复
- 它会在什么时刻判失败

若影子模式下 FRRG 的判断明显不合理，则退回级别 1 修正。

### 审查级别 4：分阶段微动作审查

只允许按以下顺序逐段放开：

1. `HANDOFF`
2. `CAPTURE_BUILD`
3. `CLOSE_HOLD`
4. `LIFT_TEST`

要求：

- 每段都可用 `--stop-after-phase` 截断
- 每段都有人在场复核
- 任意异常都可急停

### 审查级别 5：全链路小样本启用

只在以下条件下允许完整放开 FRRG：

- 单一物体
- 固定光照
- 固定桌面
- 低速
- 小位移
- 有人工急停

一旦出现以下任一情况，必须立即停机并回退：

- `vision_lost`
- `ik_fail`
- `object_jump`
- `unknown_state`

### 审查级别 6：底层系统修改（最低优先级）

只有在前面 0-5 级都证明：问题不是 FRRG 上层逻辑，而是底层能力边界时，才允许触动下列内容：

- MoveIt 控制器配置
- 关节限位
- 机械臂驱动
- 固件
- 夹爪阈值
- 机械结构参数
- 手眼标定重做

执行这一级前，必须提交完整证据链：

```text
现象
-> 运行日志
-> 状态量记录
-> guard 判定结果
-> limiter 输出结果
-> 人工复盘结论
-> 为什么确定不是上层逻辑问题
```

没有这条证据链，不允许改底层。

### 错误审查输出模板

每次审查固定按以下格式输出：

```text
审查级别：
问题类型：
触发阶段：
现象描述：
关联日志文件：
关联状态快照：
是否触发安全限幅：
是否触发恢复：
当前结论：
下一步动作：
是否允许升级到下一审查级别：
```

---

## 附：最终交付清单

本任务最终必须交付：

1. 整理后的任务计划书
2. `frrg_grasp.yaml`
3. `contracts.py / observer.py / state_machine.py / phase_guards.py`
4. `nominal_capture.py / nominal_close.py / nominal_lift.py`
5. `safety_limits.py / recovery_policy.py`
6. `run_frrg_grasp_dryrun.py`
7. `run_frrg_from_t6.py`
8. `run_frrg_extract_demo.py`
9. `run_frrg_benchmark.py`
10. dry-run / 真机 / benchmark 三类报告模板

## 附：本版最关键的补全内容

相较原始计划书，本版已明确补全：

1. 坐标系定义
2. 状态向量定义
3. 六类局部误差公式
4. 滑移检测公式
5. 三段名义控制律
6. 四段阶段切换判据
7. 统一安全限幅公式
8. 示教参数提取公式
9. DMP 与残差接口形式
10. 分级错误审查流程

