以下是为您整理好的 Markdown 格式文档：

›

### 论文算法（按步骤，单臂化后）

1.  **输入定义**：任务文本 `T`、one-shot 演示 `D={(O_t,A_t)}`、新场景 `S_new`，输出新动作轨迹 `Ã_new`（对应 `manu.tex:101`）。
2.  **one-shot 采集**：按 10Hz 记录 RGB(D) + 末端位姿 + 夹爪状态（论文描述在 `manu.tex:461`）。
3.  **时空分割**：用速度/加速度突变、夹爪开闭切换检测关键点，切成片段 `M_i`（`manu.tex:110`）。
4.  **原子技能判别**：按 `bind(o,r,t)` 判断片段是“需适应 var”还是“可复用 inv”（`manu.tex:117`）。
5.  **场景语义定位**：由任务文本生成提示词，VLM 分割目标物体 mask（论文例子 Florence-2 + SAM2，见 `manu.tex:132`）。
6.  **锚点几何计算**：从 mask 求代表点（质心/接触点），结合深度反投影得到 3D 锚点，计算 `Δx`。
7.  **朝向估计**：对方向敏感物体，使用图像矩主轴法求 `Δθ`（算法在 `manu.tex:465`）。
8.  **类别差异补偿**：由点云高度/宽度差计算 `Δh` 等几何补偿（`manu.tex:149`）。
9.  **轨迹构型**：把 var 片段做位姿迁移，把 inv 片段直接复用并拼接（`manu.tex:154`）。
10. **渐进 IK 优化**：用插值子目标逐步 IK 收敛，避免一次性求解不稳（`manu.tex:157`）。
11. **动态补偿**：抓取接近阶段加 `δ_base`, `δ_z`，减少早碰撞（`manu.tex:165`）。
12. **抗干扰闭环**：抓取前持续追踪目标，滑窗稳定后才抓；失败则重估计重试（`manu.tex:506`）。

-----

### 当前项目复现可行性评估

可直接复用的基础能力较强：

  * 机械臂控制与 IK 已有 `cjjarm_robot.py:19`、`kinematics.py:153`
  * Gemini335L 的 RGBD 驱动已具备 `gemini335l.py:222`
  * 第三方插件自动注册链路也在 `import_utils.py:133`

### 结论

单臂“最简单抓取”复现概率高（建议按阶段推进，预计可实现论文核心思路约 70%-80%）；双臂协同、复杂任务组合、跨体迁移部分不纳入本次范围。

-----

### 系统处理流程

```text
Task Text + One-shot Demo + Online RGBD
  -> GraspOrchestrator
    -> DemoRecorder / DemoParser
    -> Segmenter + InvarianceClassifier
    -> VLMObjectSegmentor + AnchorEstimator
    -> PoseAdapter (Δx, Δθ, Δh)
    -> TrajectoryComposer + ProgressiveIKPlanner
    -> ClosedLoopExecutor (stability gate + retry)
    -> Evaluator (success rate / latency / robustness)
  -> CjjArm Driver + Gemini335L Driver
```

-----

### 任务拆分（每项含文件、调用、功能、验收）

| 任务 | 要建立/修改的文件 | 调用关系 | 实现功能 | 验收标准 |
| :--- | :--- | :--- | :--- | :--- |
| **T1 工程骨架与统一配置** | 新建：`task_grasp.yaml`, `run_grasp_online.py`, `grasp_orchestrator.py`, `contracts.py` | `run_grasp_online.py` -\> `grasp_orchestrator.py` -\> 各子模块接口 | 建立单臂抓取项目入口、配置加载、模块依赖注入、统一日志与运行模式（dry-run/real） | `--dry-run` 可完整走通流程并退出码 0；配置缺失时给出明确错误 |
| **T1.5 手眼标定与坐标系管理** | 新建：`run_handeye_calib.py`, `handeye_solver.py`, `frame_manager.py`, `transforms.py`, `transforms.yaml` | `run_handeye_calib.py` -\> `handeye_solver.py` -\> `transforms.yaml`；<br>在线时 各模块 -\> `frame_manager.py` -\> `transforms.py` | 统一管理 camera/base/tool/world/object 坐标系，提供点与位姿的正逆向变换和批量变换 | 重投影误差 \< 2px；pose 回环误差平移 \< 1mm、旋转 \< 0.5°；1000 点随机双向转换稳定 |
| **T2 RGBD+机器人状态同步采集** | 新建：`run_one_shot_record.py`, `rgbd_recorder.py`, `schema.py`, `io.py`；<br>复用：`cjjarm_robot.py`, `gemini335l.py` | `run_one_shot_record.py` -\> `rgbd_recorder.py` -\> `CjjArm.get_observation` + `Gemini.read_rgbd` | 采集 one-shot 演示（RGB、Depth、关节、夹爪、EE pose、时间戳）并落盘 | 连续 60s 录制，10Hz 丢帧率 \< 5%，时间对齐误差 \< 50ms，数据可重放 |
| **T3 演示分解与技能库存储** | 新建：`run_skill_build.py`, `keypose_segmenter.py`, `invariance_classifier.py`, `skill_bank.py` | `run_skill_build.py` -\> `keypose_segmenter.py` -\> `invariance_classifier.py` -\> `skill_bank.py` | 用速度/加速度/夹爪状态切段，标注 var/inv，形成可复用技能片段库 | 抽检 20 条片段，边界可接受率 \> 90%；技能库可被后续模块读取 |
| **T4 视觉语言分割与锚点估计** | 新建：`vlm_segmentor.py`, `mask_tracker.py`, `anchor_estimator.py`, `orientation_moments.py` | orchestrator -\> `vlm_segmentor.py` -\> `mask_tracker.py` -\> `anchor_estimator.py`/`orientation_moments.py` | 1. Florence-2+SAM2 提取 2D 掩码 $\mathbf{M}_k^{\text{2D}}$；<br>2. 滑动窗口检测 $\mathbf{p}_t$ 与 $\theta_t$ 方差以判定目标稳定；<br>3. 计算掩码质心 $(\bar{x}, \bar{y})$ 并结合内参反投影为 3D 锚点 $\mathbf{p}^{\text{new}}$；<br>4. 基于二阶中心矩提取协方差矩阵特征向量，计算主轴角度 $\theta$。 | 30 张标注图 mask IoU \> 0.75；连续帧位姿追踪无高频抖动；方向敏感物体朝向误差 \< 15°。 |
| **T5 几何适配（Δx/Δθ/Δh）** | 新建：`pose_adapter.py`, `geometry_compensator.py`；<br>复用：`frame_manager.py` | `anchor_estimator.py` + `orientation_moments.py` -\> `pose_adapter.py` -\> `geometry_compensator.py` | 计算并应用位移/旋转/尺寸补偿，输出新场景抓取目标位姿 | 预抓取目标点误差 \< 2cm；高度补偿后碰桌/过高失败率显著下降 |
| **T6 轨迹构型与渐进IK** | 新建：`trajectory_composer.py`, `progressive_ik.py`, `ik_solution_selector.py`；<br>复用：`kinematics.py` | `skill_bank.py` + `pose_adapter.py` -\> `trajectory_composer.py` -\> `progressive_ik.py` -\> `ik_solution_selector.py` -\> `robot.send_action` | 拼接 var+inv 轨迹，插值点逐步 IK；在多解中按最小 Δq 选解，保证关节连续 | IK 成功率 \> 95%；max(||Δq\_t||∞) \< 0.15rad；任一关节单步跳变 \< 0.25rad；无姿态突变 |
| **T7 抗干扰闭环执行** | 新建：`stability_gate.py`, `closed_loop_executor.py`, `retry_policy.py`, `safety_limits.py` | `closed_loop_executor.py` -\> `stability_gate.py` -\> plan/act -\> `retry_policy.py` | 抓取前稳定窗检测、抓取失败重试、扰动下安全执行 | 动态扰动场景成功率 \>= 65%（13/20）；重试次数可控且无越界动作 |
| **T8 评测与复现报告** | 新建：`benchmark_grasp.py`, `metrics.py`, `report_generator.py`, `vlbiman_sa_reproduction.md` | `benchmark_grasp.py` -\> `closed_loop_executor.py` -\> `metrics.py` -\> `report_generator.py` | 固化评测协议与自动报告，输出同物体新位置/新实例/扰动三组指标 | 每组 20 次统计可复现；报告自动生成并可追溯配置与版本 |



