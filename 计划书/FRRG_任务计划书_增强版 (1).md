# FRRG 底层抓取闭环任务计划书（增强版：补齐公式、算法接口、训练目标）

> 本增强版不是推翻原草案，而是在原草案的基础上补齐 Codex 可直接实现的算法层定义：
> 1. 统一符号与坐标系；
> 2. 阶段状态与控制输入的数学形式；
> 3. 第二阶段“最后到哪里闭合”的闭合解求解器；
> 4. 各阶段 guard 的明确公式；
> 5. 安全限幅、恢复、示教参数提取、参数回归、BC 残差的训练目标。

---

## 0. 对原草案的诊断

原草案已经把工程骨架写清楚了：固定状态机、theta、dry-run、回退、测试入口、残差接口都具备了雏形。
但要让 Codex 可以直接按图施工，还缺下面六类“必须落成公式”的内容：

1. **状态量到底怎么计算**：`e_lat/e_dep/e_sym/e_ang/occ_corridor/drift_obj` 目前只有名称，没有可直接编码的公式。
2. **第二阶段到底输出什么**：现在写的是“微对位 + 推进”，但还没有定义名义控制律、前进门控和闭合触发分数。
3. **不同物体如何统一**：刚体、非凸、小物体、需要附带预动作的物体，不能只靠“中心进入走廊”这一条规则。
4. **guard 的阈值如何组织**：目前有条件描述，但缺少统一的阶段评分/逻辑组合形式。
5. **示教如何影响白盒参数**：还没有把示教提取目标写成可执行损失函数和字段映射。
6. **残差如何训练**：目前只有接口，没有 `u_demo - u_nom` 的明确定义与样本组织方式。

本增强版主要补这六部分。

---

## 1. 总体设计原则（增强版）

FRRG 仍然坚持原草案的总原则：

- 主流程固定，不让学习模块决定大阶段迁移；
- 安全边界、最大动作、最大重试次数全部白盒显式写死；
- 学习只放在三处：
  1. `theta` 阶段参数生成；
  2. 第二阶段候选闭合解评分；
  3. 小幅残差补偿。

因此完整控制结构写成：

\[
\pi_{\text{FRRG}} = \Pi_{\text{safe}}\Big( u_{\text{nom}}(s_t, \theta, m_t) + u_{\text{res}}(s_t) \Big)
\]

其中：

- \(s_t\)：当前控制周期的抓取状态；
- \(\theta\)：阶段参数；
- \(m_t\)：第二阶段当前采用的闭合模式；
- \(u_{\text{nom}}\)：白盒名义动作；
- \(u_{\text{res}}\)：局部残差动作；
- \(\Pi_{\text{safe}}\)：安全投影/限幅器。

这一定义与“白盒主干 + 参数化原语 + 受限残差”的路线一致，适合先工程落地，再逐步引入学习。

---

## 2. 统一符号与坐标系

### 2.1 坐标系定义

- 基坐标系：\(\mathcal{F}_b\)
- 相机坐标系：\(\mathcal{F}_c\)
- 末端工具坐标系：\(\mathcal{F}_e\)
- 夹爪局部坐标系：\(\mathcal{F}_g\)
- 目标局部坐标系：\(\mathcal{F}_o\)

其中，夹爪局部坐标系采用如下约定：

- \(x_g\)：左右横向（两指闭合方向）
- \(y_g\)：竖向或与桌面法向相近方向
- \(z_g\)：前进/入爪方向

若现有机器人末端坐标定义不同，必须在 `command_adapter.py` 中统一映射到这个语义坐标系，再写控制器。

### 2.2 位姿表示

采用 SE(3) 位姿：

\[
T^b_e = \begin{bmatrix} R^b_e & p^b_e \\ 0 & 1 \end{bmatrix},
\quad
T^b_o = \begin{bmatrix} R^b_o & p^b_o \\ 0 & 1 \end{bmatrix}
\]

末端在目标局部系下的位姿：

\[
T^o_e = (T^b_o)^{-1} T^b_e
\]

其平移项记为：

\[
p^o_e = [x^o_e, y^o_e, z^o_e]^\top
\]

旋转误差优先使用 yaw 主误差（因为首版关注入爪对向方向）：

\[
e_{\text{yaw}} = \text{wrapToPi}(\psi_e - \psi_o^{\text{grasp}})
\]

若后续加入完整 6D 抓取，再扩展到 axis-angle：

\[
e_R = \log\big((R^b_o R^{o,\text{goal}}_e)^\top R^b_e\big)
\]

---

## 3. 统一状态定义

原草案中的 `s_t` 需要正式化为：

\[
s_t = (o_t, x_t, g_t, q_t, \phi_t, r_t)
\]

其中：

- \(o_t\)：视觉观测与目标局部特征；
- \(x_t\)：末端与目标相对位姿；
- \(g_t\)：夹爪状态；
- \(q_t\)：当前阶段参数；
- \(\phi_t\)：当前 phase 与 phase 内计数器；
- \(r_t\)：恢复状态与重试次数。

建议代码中落实为：

```python
GraspState = {
  "timestamp": float,
  "phase": str,
  "mode": str,
  "retry_count": int,
  "stable_count": int,
  "ee_pose_base": Pose6D,
  "object_pose_base": Pose6D,
  "ee_pose_object": Pose6D,
  "gripper_width": float,
  "gripper_cmd": float,
  "gripper_current_proxy": float,
  "vision_conf": float,
  "target_visible": bool,
  "corridor_mask_ratio": float,
  "corridor_center_px": [float, float],
  "object_center_px": [float, float],
  "object_axis_angle": float,
  "object_proj_width_px": float,
  "object_proj_height_px": float,
  "e_lat": float,
  "e_dep": float,
  "e_sym": float,
  "e_ang": float,
  "occ_corridor": float,
  "drift_obj": float,
  "capture_score": float,
  "hold_score": float,
  "slip_score": float,
}
```

---

## 4. 第二阶段的核心升级：把“入爪构型建立”改成“闭合解求解器”

这是增强版最重要的改动。

### 4.1 为什么必须改

原草案里 `CAPTURE_BUILD` 的目标写成“直到目标进入有效两指通道”，这对圆球、盒子、普通凸物体是成立的；
但对以下对象不够：

- 非凸物体：正确抓位可能在局部结构，不在整体中心附近；
- 很小的物体：直接中心对中可能会被推飞；
- 需要附带动作/任务约束：不是只要抓稳，还要方便后续操作；
- 贴桌/薄片类：可能需要先形成可抓边。

因此第二阶段真正输出的，不应只是“推进一点再闭”，而应是：

\[
\gamma^* = (m^*, g^*, a^*_{\text{pre}}, q^*)
\]

其中：

- \(m^*\)：选择的闭合模式；
- \(g^*\)：目标闭合位姿；
- \(a^*_{\text{pre}}\)：是否需要预动作；
- \(q^*\)：该模式下的阶段参数子集。

### 4.2 闭合模式集合

首版建议支持四个模式：

\[
\mathcal{M} = \{m_{\text{env}}, m_{\text{local}}, m_{\text{assist}}, m_{\text{preact}}\}
\]

1. **包围闭合** \(m_{\text{env}}\)
   - 适合球、果实、杯子、盒子、鸭子这类普通凸体。
2. **局部结构闭合** \(m_{\text{local}}\)
   - 适合非凸、带把手、带局部凹槽、局部突出区域的目标。
3. **环境辅助闭合** \(m_{\text{assist}}\)
   - 适合贴桌小物体、薄片、边缘抓取。
4. **预动作后闭合** \(m_{\text{preact}}\)
   - 当前没有稳定闭合解时，先执行一个小推/小拨/小滑动作，再回到第二阶段重新求解。

首版实现可以不训练模式分类器，而采用**规则优先 + 可配置目标标签**：

```yaml
object_grasp_mode:
  orange: enveloping
  duck: enveloping
  mug_handle: local_feature
  key: assist_edge
  towel: preaction_or_edge
```

后续再把模式选择替换成学习评分。

### 4.3 闭合候选定义

对任一模式 \(m\)，生成候选集合：

\[
\mathcal{G}_m = \{ g_i = (p_i, R_i, w_i, d_i, a_i^{\text{pre}}) \}_{i=1}^{N_m}
\]

其中：

- \(p_i\)：闭合中心点；
- \(R_i\)：闭合时末端姿态；
- \(w_i\)：预张开宽度；
- \(d_i\)：闭合前允许推进深度；
- \(a_i^{\text{pre}}\)：必要的预动作。

### 4.4 候选评分函数

每个候选定义统一评分：

\[
S(g_i \mid s_t, c_{task}) =
\lambda_{vis} S_{vis}
+ \lambda_{geom} S_{geom}
+ \lambda_{task} S_{task}
+ \lambda_{reach} S_{reach}
- \lambda_{col} S_{col}
- \lambda_{risk} S_{risk}
\]

含义：

- \(S_{vis}\)：视觉可确认性；
- \(S_{geom}\)：局部闭合几何质量；
- \(S_{task}\)：是否满足任务/后续动作约束；
- \(S_{reach}\)：机器人可达性；
- \(S_{col}\)：碰撞/贴桌风险；
- \(S_{risk}\)：闭合失败风险。

首版实现建议：

- `S_vis` 用可见度与稳定帧计算；
- `S_geom` 用走廊对中、角度一致性、宽度余量；
- `S_task` 用目标允许抓取区域与禁抓区；
- `S_reach` 用 IK 可行 + 离奇异位姿距离；
- `S_col` 用桌面碰撞距离、邻物体距离；
- `S_risk` 用视觉不确定性、历史失败先验。

最终选择：

\[
g^* = \arg\max_{g_i \in \cup_m \mathcal{G}_m} S(g_i)
\]

若最大分仍低于阈值：

\[
\max_i S(g_i) < \tau_{\text{grasp-solve}}
\Rightarrow a^*_{\text{pre}} \neq \varnothing
\]

即触发预动作分支。

---

## 5. 视觉特征与局部误差的明确公式

原草案里最需要补齐的是这部分。

### 5.1 两指走廊定义

在图像平面中，利用当前末端位姿和夹爪几何模型，投影得到两指内侧边界：

- 左内边界：\(\ell_L\)
- 右内边界：\(\ell_R\)

两边界之间的区域定义为走廊掩码：

\[
\mathcal{C}_t = \{ u \in \Omega \mid u \text{ 位于 } \ell_L, \ell_R \text{ 之间} \}
\]

在代码里，首版完全可以用图像多边形掩码实现。

### 5.2 目标中心误差

假设目标掩码为 \(M_o\)，其像素重心为：

\[
c_o = (u_o, v_o)
\]

走廊中心线重心为：

\[
c_c = (u_c, v_c)
\]

定义横向误差：

\[
e_{lat}^{px} = u_o - u_c
\]

若已知深度 \(z_o\) 和相机内参 \(f_x\)，换算为米：

\[
e_{lat} = \frac{z_o}{f_x} \cdot e_{lat}^{px}
\]

### 5.3 深度误差

若可获得目标局部系下期望闭合中心点 \(p^{o,*}_e = [x^*, y^*, z^*]^\top\)，则推进方向误差定义为：

\[
e_{dep} = z^* - z^o_e
\]

这里 \(z^o_e\) 是当前末端在目标局部坐标系下的推进轴位置。

### 5.4 左右对称误差

设目标掩码与左侧边界最短距离为 \(d_L\)，与右侧边界最短距离为 \(d_R\)，定义：

\[
e_{sym} = d_L - d_R
\]

若 \(e_{sym} > 0\)，说明目标更靠右；若 \(e_{sym} < 0\)，说明目标更靠左。

### 5.5 角度误差

由掩码主轴或局部关键点得到目标主方向 \(\alpha_o\)，由夹爪闭合方向得到参考角 \(\alpha_g\)，定义：

\[
e_{ang} = \mathrm{wrapToPi}(\alpha_o - \alpha_g)
\]

对近似旋转对称物体，引入对称权重 \(\rho_{symm} \in [0,1]\)：

\[
\tilde e_{ang} = \rho_{symm} \cdot e_{ang}
\]

对于橘子、苹果等近球体，可设 \(\rho_{symm} \approx 0\)，避免无意义角度控制。

### 5.6 走廊占据率

目标与走廊的相交占比定义为：

\[
occ_{corridor} = \frac{|M_o \cap \mathcal{C}_t|}{|M_o| + \epsilon}
\]

其中 \(|\cdot|\) 表示掩码像素面积。

这个量用来衡量“目标有多少比例已经进入可闭合通道”。

### 5.7 连续漂移量

定义目标中心相对夹爪中心的连续变化：

\[
drift_{obj}(t) = \left\| (c_o(t) - c_c(t)) - (c_o(t-1) - c_c(t-1)) \right\|_2
\]

如需米制，可乘相机尺度系数。

### 5.8 视觉置信度

建议引入统一视觉置信度：

\[
q_{vis} = w_1 q_{det} + w_2 q_{seg} + w_3 q_{track} - w_4 q_{occ}
\]

其中：
- \(q_{det}\)：检测框或短语 grounding 置信度；
- \(q_{seg}\)：分割质量置信度；
- \(q_{track}\)：连续跟踪稳定度；
- \(q_{occ}\)：遮挡惩罚。

首版没有完整多模块时，可直接先设：

\[
q_{vis} = \mathbb{1}[\text{target visible}] \cdot \min(1, occ_{corridor} + 0.5)
\]

---

## 6. 阶段参数 theta（增强版字段）

原草案的 `theta` 太少，不足以支撑第二阶段的统一求解。建议扩成：

```text
theta = {
  # 通用
  control_hz,
  stable_window_frames,
  max_retry_count,

  # HANDOFF
  handoff_pos_tol_m,
  handoff_yaw_tol_rad,
  handoff_vis_min,
  handoff_open_width_m,

  # CAPTURE_BUILD 通用
  mode,
  corridor_half_width_m,
  corridor_margin_m,
  target_depth_goal_m,
  target_depth_max_m,
  lat_gain,
  dep_gain,
  yaw_gain,
  sym_gain,
  advance_speed_mps,
  lateral_speed_mps,
  angular_speed_radps,
  forward_enable_lat_tol_m,
  forward_enable_ang_tol_rad,
  forward_enable_occ_min,
  close_score_threshold,
  solve_score_threshold,

  # 模式特有
  local_region_id,
  allow_env_assist,
  allow_preaction,
  preaction_type,
  preaction_dx_m,
  preaction_dyaw_rad,

  # CLOSE_HOLD
  preclose_distance_m,
  preclose_pause_s,
  close_speed_raw_per_s,
  close_width_target_m,
  contact_current_min,
  contact_current_max,
  hold_drift_max_m,
  settle_time_s,
  hold_score_threshold,

  # LIFT_TEST
  lift_height_m,
  lift_speed_mps,
  lift_hold_s,
  slip_threshold_m,
  lift_score_threshold,

  # safety
  max_step_xyz_m,
  max_step_rpy_rad,
  max_joint_delta,
  max_gripper_delta_m,
}
```

---

## 7. 每个阶段的控制律与 guard 公式

## 7.1 HANDOFF

### 目标
确认 T6 已把末端送到可接管的预抓壳层。

### guard

\[
G_{handoff} = 
\mathbb{1}[\|p_e^o - p_{pre}^o\| < \tau_p]
\land \mathbb{1}[|e_{yaw}| < \tau_\psi]
\land \mathbb{1}[q_{vis} > \tau_{vis}]
\land \mathbb{1}[w_g > w_{open}^{min}]
\]

若成立，则进入 `CAPTURE_BUILD`。

### 动作

HANDOFF 首版通常不输出运动，只允许：
- 若夹爪未达到开度，补开到 `handoff_open_width_m`；
- 若末端轻微偏离预抓壳层，可执行极小修正：

\[
\Delta p_{handoff} = -K_h (p_e^o - p_{pre}^o)
\]

并经过安全裁剪。

---

## 7.2 CAPTURE_BUILD（最关键）

### 目标
在当前模式下，把末端送到“允许闭合”的局部构型，而不是盲目前进。

### 7.2.1 构型评分

定义闭合准备分数：

\[
score_{cap} =
\omega_1 \exp\left(-\frac{e_{lat}^2}{\sigma_{lat}^2}\right)
+ \omega_2 \exp\left(-\frac{\tilde e_{ang}^2}{\sigma_{ang}^2}\right)
+ \omega_3 occ_{corridor}
+ \omega_4 \exp\left(-\frac{e_{sym}^2}{\sigma_{sym}^2}\right)
+ \omega_5 q_{vis}
\]

其中 \(\sum_i \omega_i = 1\)。

### 7.2.2 前进门控

前进不是始终开放，而是门控的：

\[
g_{fwd} =
\mathbb{1}[|e_{lat}| < \tau_{lat}^{fwd}]
\cdot
\mathbb{1}[|\tilde e_{ang}| < \tau_{ang}^{fwd}]
\cdot
\mathbb{1}[occ_{corridor} > \tau_{occ}^{fwd}]
\cdot
\mathbb{1}[q_{vis} > \tau_{vis}^{fwd}]
\]

### 7.2.3 名义控制律

在目标局部系下定义名义增量：

\[
\Delta x_{nom}^{o} =
\begin{bmatrix}
\Delta x \\
\Delta y \\
\Delta z \\
\Delta \psi
\end{bmatrix}
=
\begin{bmatrix}
\mathrm{sat}(k_{lat} e_{lat} + k_{sym} e_{sym}, \Delta x_{max}) \\
\mathrm{sat}(k_{vert} e_{vert}, \Delta y_{max}) \\
\mathrm{sat}(g_{fwd} \cdot k_{dep} e_{dep}, \Delta z_{max}) \\
\mathrm{sat}(k_{ang} \tilde e_{ang}, \Delta \psi_{max})
\end{bmatrix}
\]

若首版没有 `e_vert`，直接设 \(\Delta y = 0\)。

注意：
- 只有 \(g_{fwd}=1\) 时允许沿 \(z\) 前进；
- 若推进中某一帧失去门控条件，则 \(\Delta z\) 立即归零。

### 7.2.4 闭合触发条件

不是单个阈值，而是“分数 + 连续稳定”：

\[
G_{cap\rightarrow close} =
\mathbb{1}[score_{cap} > \tau_{cap}]
\land
\mathbb{1}[N_{stable} \ge N_{stable}^{cap}]
\land
\mathbb{1}[z_e^o \ge z_{goal}^{close}]
\]

### 7.2.5 失败条件

\[
F_{cap} =
\mathbb{1}[q_{vis} < \tau_{vis}^{stop}]
\lor
\mathbb{1}[|e_{lat}| > \tau_{lat}^{fail}]
\lor
\mathbb{1}[|\tilde e_{ang}| > \tau_{ang}^{fail}]
\lor
\mathbb{1}[z_e^o > z_{max}^{fail}]
\lor
\mathbb{1}[t_{phase} > T_{cap}^{max}]
\]

若 \(F_{cap}=1\)，进入 `RECOVERY`。

---

## 7.3 CLOSE_HOLD

### 目标
在当前允许闭合构型上完成预闭合、正式闭合、短时稳定确认。

### 7.3.1 三段式动作

1. **preclose**：先推进到闭合前距离
2. **close**：按速度闭合
3. **settle**：静置并观测

### 7.3.2 接触替代量

因为首版不依赖触觉，用夹爪宽度与电流替代量构造接触代理：

\[
\chi_{contact} =
\mathbb{1}[w_g < w_{open}^{cmd} - \tau_w]
\cdot
\mathbb{1}[I_g \in [I_{min}, I_{max}]]
\]

这里：
- \(w_g\)：当前夹爪宽度；
- \(I_g\)：电流替代量或电机负载代理。

### 7.3.3 稳持评分

\[
score_{hold} =
\eta_1 \chi_{contact}
+ \eta_2 \exp\left(-\frac{drift_{obj}^2}{\sigma_{drift}^2}\right)
+ \eta_3 \mathbb{1}[w_g \in [w_{min}^{hold}, w_{max}^{hold}]]
+ \eta_4 q_{vis}
\]

### 7.3.4 成功条件

\[
G_{close\rightarrow lift} =
\mathbb{1}[score_{hold} > \tau_{hold}]
\land
\mathbb{1}[t_{settle} > T_{settle}]
\]

### 7.3.5 失败条件

\[
F_{close} =
\mathbb{1}[q_{vis} < \tau_{vis}^{stop}]
\lor
\mathbb{1}[w_g < w_{min}^{crush}] 
\lor
\mathbb{1}[I_g > I_{max}^{hard}]
\lor
\mathbb{1}[drift_{obj} > \tau_{drift}^{fail}]
\]

成立则进入 `RECOVERY`。

---

## 7.4 LIFT_TEST

### 目标
小提起验证是否真正抓住，而不是刚好卡住或虚持。

### 名义动作

\[
\Delta p_{lift}^o = [0, 0, 0]^\top \text{ in object frame}
\quad\text{or}\quad
\Delta p_{lift}^b = [0, 0, h_{lift}]^\top \text{ in base frame}
\]

首版建议在基坐标系沿竖直向上提起：

\[
\Delta z_{lift}(t) = \min(v_{lift} \Delta t, h_{remain})
\]

### 滑移评分

\[
score_{lift} =
\zeta_1 \exp\left(-\frac{drift_{obj}^2}{\sigma_{slip}^2}\right)
+ \zeta_2 q_{vis}
+ \zeta_3 \mathbb{1}[\chi_{contact}=1]
\]

### 成功条件

\[
G_{lift\rightarrow success} =
\mathbb{1}[score_{lift} > \tau_{lift}]
\land
\mathbb{1}[h_{done} \ge h_{lift}^{target}]
\land
\mathbb{1}[drift_{obj} < \tau_{slip}]
\]

### 失败条件

\[
F_{lift} =
\mathbb{1}[drift_{obj} > \tau_{slip}^{fail}]
\lor
\mathbb{1}[q_{vis} < \tau_{vis}^{stop}]
\lor
\mathbb{1}[\chi_{contact}=0]
\]

成立则进入 `RECOVERY` 或直接 `FAILURE`。

---

## 8. 安全限幅器的明确形式

安全限幅器必须对所有动作统一处理：

\[
u_t^{safe} = \Pi_{safe}(u_t)
\]

其中：

\[
\Pi_{safe}(u_t) =
\mathrm{clip}_{\Delta p}(\mathrm{clip}_{\Delta R}(\mathrm{clip}_{g}(u_t)))
\]

建议按顺序实现：

### 8.1 末端位姿限幅

\[
\Delta p^{safe} =
\mathrm{clip}(\Delta p, -\Delta p_{max}, \Delta p_{max})
\]

\[
\Delta r^{safe} =
\mathrm{clip}(\Delta r, -\Delta r_{max}, \Delta r_{max})
\]

### 8.2 夹爪限幅

\[
\Delta g^{safe} = \mathrm{clip}(\Delta g, -\Delta g_{max}, \Delta g_{max})
\]

### 8.3 视觉丢失硬停

\[
q_{vis} < \tau_{vis}^{hardstop} \Rightarrow u_t^{safe}=0
\]

### 8.4 IK 失败硬停

若 `command_adapter.py` 无法把目标位姿转成可执行关节动作：

\[
\text{IKFail} \Rightarrow u_t^{safe}=0,\ \text{phase}=FAILURE
\]

### 8.5 跳变硬停

若目标局部位姿帧间跳变过大：

\[
\|p_o(t)-p_o(t-1)\| > \tau_{obj-jump}
\Rightarrow u_t^{safe}=0
\]

---

## 9. 恢复策略的明确形式

恢复策略只允许非常有限的动作原语：

\[
a_{rec} \in \{\text{backoff},\text{half-open},\text{recenter},\text{abort}\}
\]

### 9.1 backoff

沿进入方向反向微退：

\[
\Delta z_{backoff} = -d_{backoff}
\]

### 9.2 half-open

\[
g \leftarrow g_{half-open}
\]

### 9.3 recenter

清空稳定计数，回到 `CAPTURE_BUILD` 初始子状态。

### 9.4 恢复决策

\[
\text{if } retry\_count < N_{max}:
\quad phase \leftarrow RECOVERY \rightarrow CAPTURE\_BUILD
\]

\[
\text{else } phase \leftarrow FAILURE
\]

建议保留恢复原因字典：

```python
RECOVERY_REASON_TO_ACTION = {
  "vision_lost": ["stop", "backoff", "abort_or_retry"],
  "capture_timeout": ["backoff", "half_open", "retry"],
  "large_drift": ["half_open", "backoff", "retry"],
  "slip_detected": ["down_or_backoff", "half_open", "retry"],
}
```

---

## 10. phase-2 / phase-4 名义原语的具体表示

原草案允许 DMP 或轻量轨迹模板。为了让首版更容易实现，建议：

- **首版实现：最小 jerk 模板 + 比例误差闭环**
- **接口保留：DMP 兼容**

### 10.1 最小 jerk 标量轨迹

对任一标量轨迹分量 \(y\)：

\[
y(\tau)=y_0 + (y_g-y_0)(10\tau^3 - 15\tau^4 + 6\tau^5),\quad \tau \in [0,1]
\]

它适合 `Lift-Test` 的小幅上提，也适合 `Capture-Build` 的前进标量基轨迹。

### 10.2 DMP 兼容接口

若后续换 DMP，接口保持：

\[
\tau \dot v = K(g-y) - Dv + f(s),
\quad
\tau \dot y = v
\]

但首版先不要让 Codex 直接实现完整 DMP 库，优先把接口留好。

---

## 11. 示教参数提取：从演示到 theta 的明确公式

这是原草案第二个最关键缺口。

## 11.1 先切阶段

从 one-shot 示教或已有分段结果中取末段片段：

\[
\mathcal{D}^{demo} = \{(o_t, x_t, g_t, u_t)\}_{t=1}^{T}
\]

分成：

- `handoff seg`
- `capture seg`
- `close seg`
- `lift seg`

### 11.2 参数提取公式

#### 11.2.1 pregrasp_offset

\[
pregrasp\_offset_m = \|p_{handoff,start}^o - p_{capture,start}^o\|
\]

#### 11.2.2 advance_speed

\[
advance\_speed\_mps = \mathrm{median}_{t \in capture}\left( \frac{\Delta z_t}{\Delta t} \right)
\]

#### 11.2.3 lateral gain 建议值

由 capture 段拟合：

\[
\Delta x_t^{demo} \approx k_{lat} e_{lat,t} + k_{sym} e_{sym,t}
\]

做最小二乘：

\[
\hat{\beta} = [\hat k_{lat}, \hat k_{sym}]^\top = (X^\top X + \lambda I)^{-1} X^\top Y
\]

其中：

\[
X = \begin{bmatrix} e_{lat,1} & e_{sym,1} \\\vdots & \vdots \\\ e_{lat,n} & e_{sym,n} \end{bmatrix},
\quad
Y = \begin{bmatrix} \Delta x_1^{demo} \\\vdots \\\ \Delta x_n^{demo} \end{bmatrix}
\]

#### 11.2.4 yaw gain 建议值

\[
\Delta \psi_t^{demo} \approx k_{ang} e_{ang,t}
\Rightarrow
\hat k_{ang} = \frac{\sum_t e_{ang,t} \Delta\psi_t^{demo}}{\sum_t e_{ang,t}^2 + \epsilon}
\]

#### 11.2.5 preclose_distance

定义闭合开始帧 \(t_c\)，则：

\[
preclose\_distance_m = z_e^o(t_c) - z_{goal}^{close}
\]

#### 11.2.6 close_speed

\[
close\_speed = \mathrm{median}_{t \in close}\left( \frac{|g_{t+1}-g_t|}{\Delta t} \right)
\]

#### 11.2.7 settle_time

从闭合结束到稳持确认的持续时间：

\[
settle\_time_s = t_{hold\_ready} - t_{close\_done}
\]

#### 11.2.8 lift_height

\[
lift\_height_m = z_e^b(t_{lift,end}) - z_e^b(t_{lift,start})
\]

#### 11.2.9 stable frames 建议值

若控制频率为 \(f_c\)，可从示教中稳定窗口时间 \(T_{stable}^{demo}\) 计算：

\[
N_{stable}^{demo} = \lceil f_c T_{stable}^{demo} \rceil
\]

---

## 12. 阶段参数生成器：低维回归器的明确训练目标

对于对象相关参数生成器，不要学策略，而学：

\[
\hat \theta = f_\varphi(\xi)
\]

其中 \(\xi\) 是对象条件特征，例如：

- 目标类别 one-hot / embedding
- 目标尺寸：宽、高、厚
- 目标主方向
- 目标局部可见度
- 是否近似对称
- 任务标签
- 当前相对位姿初值

### 12.1 训练损失

\[
\mathcal{L}_{\theta} =
\sum_j w_j \|\hat\theta_j - \theta_j^{demo}\|_2^2
+ \lambda_{range} \mathcal{L}_{range}
\]

其中：
- \(\theta_j^{demo}\)：示教提取参数；
- \(\mathcal{L}_{range}\)：超出物理范围的惩罚。

### 12.2 范围投影

推理时必须二次投影：

\[
\hat \theta^{safe} = \Pi_{\theta\_range}(\hat \theta)
\]

即使回归器输出异常，也不得绕过物理上下限。

---

## 13. BC 残差策略：明确的数据组织与训练目标

### 13.1 残差定义

对每一帧，先计算名义控制器输出：

\[
u_{nom,t} = u_{nom}(s_t, \theta)
\]

再从示教动作得到总动作：

\[
u_{demo,t}
\]

定义残差标签：

\[
u_{res,t}^{demo} = u_{demo,t} - u_{nom,t}
\]

### 13.2 残差裁剪标签

训练前先把标签裁剪到安全盒内：

\[
\tilde u_{res,t}^{demo} = \mathrm{clip}(u_{res,t}^{demo}, -b_{res}, b_{res})
\]

例如：

- \(|dx|, |dy| \le 4\text{mm}\)
- \(|dz| \le 3\text{mm}\)
- \(|d\psi| \le 5^\circ\)

### 13.3 BC 损失

\[
\hat u_{res,t} = \pi_\omega(s_t)
\]

\[
\mathcal{L}_{BC} =
\sum_t \|\hat u_{res,t} - \tilde u_{res,t}^{demo}\|_2^2
+ \lambda_{smooth} \sum_t \|\hat u_{res,t} - \hat u_{res,t-1}\|_2^2
\]

其中平滑项防止残差抖动。

### 13.4 在线执行

\[
u_t = \Pi_{safe}\big(u_{nom,t} + \hat u_{res,t}\big)
\]

若残差策略未启用，则：

\[
\hat u_{res,t}=0
\]

---

## 14. 多物体鲁棒抓取：如何在计划书里落成工程可实现结构

前面的对话已经确认，仅靠“中心进入走廊”不足以覆盖多类对象，因此计划书中必须新增“模式相关闭合解求解”。

建议新增文件：

```text
grasp/mode_selector.py          # 规则/学习的模式选择
grasp/grasp_solver.py           # 候选生成与评分
grasp/task_constraints.py       # 允许抓/禁抓/后续操作约束
```

### 14.1 mode_selector.py

输入：
- object label
- object geometric tag
- task tag
- current scene context

输出：
- `mode`
- `mode_confidence`

首版规则版本：

```python
def select_mode(obj_tag, task_tag):
    if obj_tag in {"orange", "apple", "duck", "cup", "box"}:
        return "enveloping"
    if obj_tag in {"mug_handle", "scissors", "tool_nonconvex"}:
        return "local_feature"
    if obj_tag in {"coin", "key", "flat_small"}:
        return "assist_edge"
    if obj_tag in {"towel", "cloth", "deformable_flat"}:
        return "preaction_or_edge"
    return "enveloping"
```

### 14.2 grasp_solver.py

输入：
- 当前 `GraspState`
- `mode`
- task constraints

输出：
- `best_candidate`
- `solve_score`
- `need_preaction`

### 14.3 task_constraints.py

需要支持：
- 禁抓区域 mask
- 允许抓区域 mask
- 抓后操作方向约束
- 抓后保持姿态约束

这对“带其他动作或约束”的目标非常重要。

---

## 15. 对原文件结构的补充建议

在原计划的文件架构基础上，建议新增：

```text
grasp/
  mode_selector.py                  # 闭合模式选择器
  grasp_solver.py                   # 第二阶段候选求解器
  task_constraints.py               # 任务与禁抓区域约束
  feature_geometry.py               # 走廊、中心、边界、误差计算
  scores.py                         # capture/hold/lift 评分公式
```

其中：
- `feature_geometry.py` 负责 `e_lat/e_dep/e_sym/e_ang/occ_corridor/drift_obj`
- `scores.py` 负责 `score_cap/score_hold/score_lift`

这样 Codex 不会把误差计算、阶段控制和评分逻辑混在一个文件里。

---

## 16. 对原 T5/T7/T9/T10 的直接增强内容

### 16.1 T5 guard 增强

原计划里 T5 只写了条件句，增强后要求：

- 每个 guard 实现为单独函数；
- 每个 guard 同时返回：
  - `passed: bool`
  - `score: float`
  - `reason: Optional[str]`
  - `debug_terms: Dict[str, float]`

例如：

```python
{
  "passed": True,
  "score": 0.83,
  "reason": None,
  "debug_terms": {
    "e_lat": 0.0021,
    "e_ang": 0.031,
    "occ_corridor": 0.78,
    "stable_count": 6,
  }
}
```

### 16.2 T7 名义动作增强

原计划里 T7 需要新增：
- `compute_capture_score()`
- `compute_forward_gate()`
- `compute_nominal_capture_delta()`
- `compute_hold_score()`
- `compute_lift_score()`

### 16.3 T9 提取器增强

不仅提取 `theta` 的标量，还要输出：
- `capture_regression_fit.json`
- `residual_samples.jsonl`
- `mode_labels.json`

### 16.4 T10 残差接口增强

增加 `ResidualPolicyOutput`：

```python
{
  "delta_pose_object": [dx, dy, dz, dyaw],
  "delta_gripper": float,
  "raw_norm": float,
  "clipped_norm": float,
  "is_clipped": bool,
}
```

---

## 17. 建议新增的 YAML 配置段

```yaml
capture_build:
  mode: enveloping
  solve_score_threshold: 0.55
  close_score_threshold: 0.72
  forward_enable_lat_tol_m: 0.004
  forward_enable_ang_tol_rad: 0.12
  forward_enable_occ_min: 0.55
  corridor_half_width_m: 0.018
  corridor_margin_m: 0.003
  target_depth_goal_m: 0.012
  target_depth_max_m: 0.028
  lat_gain: 0.8
  sym_gain: 0.4
  dep_gain: 0.5
  yaw_gain: 0.6
  stable_window_frames: 5

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
```

以上数值只是首版保守默认值，必须在真机前再二次核验。

---

## 18. Codex 实现时必须遵守的工程约束

1. **先实现零学习闭环**：mode 用规则，theta 用手写默认值，residual=0。
2. **误差计算独立文件**：不要把视觉特征解析散落在 controller 内。
3. **控制器只吃统一 `GraspState`**：控制器不能直接到处读 ROS topic / camera 结果。
4. **所有动作只允许一个出口**：必须统一经过 `SafetyLimiter`。
5. **所有阶段切换必须落盘**：转移前状态、原因、分数、关键误差都要写日志。
6. **真机默认 dry-run**：只有显式 `--execute` 才允许真发动作。
7. **残差和参数学习都必须可拔插**：禁用时系统仍能工作。

---

## 19. 对原草案的直接补丁结论

如果你想让 Codex 能“看文档就开工”，那原草案至少要补进下面这些硬内容：

### 必补 1：状态和误差公式
- `e_lat/e_dep/e_sym/e_ang/occ_corridor/drift_obj` 明确化。

### 必补 2：第二阶段控制律
- `score_cap`
- `forward_gate`
- `Δx_nom^o`
- `close trigger`

### 必补 3：多模式闭合解求解
- `mode_selector`
- `grasp_solver`
- `task_constraints`

### 必补 4：示教提取目标
- `theta` 提取公式
- `k_lat/k_ang` 等拟合公式
- `residual label` 构造公式

### 必补 5：残差训练目标
- `u_res_demo = u_demo - u_nom`
- BC 损失
- 在线安全裁剪

### 必补 6：统一评分输出
- capture/hold/lift 三阶段都输出 `score + reason + debug_terms`

---

## 20. 建议你现在就替换进原计划书的新增章节标题

建议在原草案中新增以下章节：

- **4.3A 坐标系与位姿符号**
- **4.3B 手眼局部误差计算公式**
- **4.3C 第二阶段闭合解求解器**
- **4.4A 扩展版 theta 字段**
- **4.5A 安全投影公式**
- **5.1 mode_selector / grasp_solver / scores 文件职责**
- **T5A guard 返回结构**
- **T7A 名义控制律公式**
- **T9A 示教参数提取公式**
- **T10A BC 残差数据与损失**

这样文档结构仍然延续原草案，但已经足够让 Codex 直接拆任务实现。

---

## 21. 最后的落地建议

如果你准备立刻让 Codex 开干，建议把任务分成四轮，而不是一次全做：

### 第一轮
只做：
- 误差计算
- 状态机
- guard
- 零残差
- safety
- dry-run

### 第二轮
补：
- 第二阶段评分
- 门控前进
- 闭合触发
- hold/lift 分数

### 第三轮
补：
- demo extractor
- theta 回归器
- mode selector（先规则版）

### 第四轮
补：
- BC residual
- 候选求解器重排序
- 多对象模式扩展

这样风险最低，也最符合你原草案里“先工程闭环，再参数化，再残差”的顺序。
