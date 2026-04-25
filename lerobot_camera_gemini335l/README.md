# lerobot_camera_gemini335l

Gemini 335L 的 LeRobot 相机插件（基于 Orbbec 官方 Python SDK）。

## 快速开始（4 条命令）

1）在当前激活环境安装依赖（`.venv` 或 conda 都可）：

```bash
python -m pip install pyorbbecsdk2
python -m pip install -e ./lerobot_camera_gemini335l
```

2）列出设备：

```bash
./scripts/gemini335l/run_stream_viewer.sh --list
```

3）一键健康检查（检测设备 + 启动诊断 + 连续读帧）：

```bash
./scripts/gemini335l/quick_check.sh <你的相机序列号>
```

4）实时预览（`q` / `Esc` 退出）：

```bash
./scripts/gemini335l/run_stream_viewer.sh \
  --serial-number-or-name <你的相机序列号> \
  --stream rgbd \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest
```

## 为什么这套命令稳定

- 启动脚本会自动识别当前 Python 环境，不再依赖硬编码 conda 路径。
- `quick_check.sh` 会覆盖完整链路：
  - SDK 导入与设备枚举
  - RGB / RGBD 启动探测
  - 无界面连续读帧（验证可被算法调用）
- 默认参数使用兼容性更高的组合（`align-mode=sw` + `closest`）。

## 最小命令集

### 1）检测设备

```bash
./scripts/gemini335l/run_stream_viewer.sh --list
```

### 2）查看流配置（分辨率 / FPS / 格式）

```bash
./scripts/gemini335l/run_stream_viewer.sh \
  --list-profiles \
  --serial-number-or-name <你的相机序列号>
```

### 3）启动诊断并输出推荐命令

```bash
./scripts/gemini335l/diagnose_depth_startup.sh \
  --serial-number-or-name <你的相机序列号>
```

### 4）实时深度查看

```bash
./scripts/gemini335l/run_depth_viewer.sh \
  --serial-number-or-name <你的相机序列号> \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --show-color
```

### 5）算法调用验证（无 GUI）

```bash
./scripts/gemini335l/run_stream_viewer.sh \
  --serial-number-or-name <你的相机序列号> \
  --stream rgbd \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --headless --frame-limit 120
```

如果第 5 条通过，说明驱动和相机数据链路可被其他系统或算法稳定调用。

## 给算法/系统的 Python 调用示例

```python
from lerobot_camera_gemini335l.config_gemini335l import Gemini335LCameraConfig
from lerobot_camera_gemini335l.gemini335l import Gemini335LCamera

cfg = Gemini335LCameraConfig(
    serial_number_or_name="<你的相机序列号>",
    width=640,
    height=400,
    fps=15,
    use_depth=True,
    align_depth_to_color=True,
    align_mode="sw",
    profile_selection_strategy="closest",
)

cam = Gemini335LCamera(cfg)
cam.connect()
try:
    color, depth = cam.read_rgbd(timeout_ms=700)
    print(color.shape, depth.shape, depth.dtype)
finally:
    cam.disconnect()
```

## 可调参数总览（重点）

下面把常用脚本的参数按用途整理出来，便于快速调优。

### `run_depth_viewer.sh` / `depth_viewer`

基础连接参数：
- `--serial-number-or-name`：序列号、设备名或 UID。
- `--width` / `--height` / `--fps`：请求分辨率与帧率。
- `--color-stream-format`：彩色格式（如 `MJPG`、`RGB`）。
- `--depth-stream-format`：深度格式（如 `Y16`、`Z16`）。

对齐和启动稳定性参数：
- `--align-depth-to-color`：开启深度对齐到彩色。
- `--align-mode {hw,sw}`：
  - `hw`：硬件对齐，性能高，但兼容性依赖设备/模式。
  - `sw`：软件对齐，稳定性更高，推荐排障首选。
- `--profile-selection-strategy {exact,closest}`：
  - `exact`：严格匹配请求模式。
  - `closest`：自动选择最接近可用模式，推荐排障时使用。

深度质量与近距离调参：
- `--depth-work-mode`：深度算法模式（如 `Default`、`Hand`、`High Accuracy`）。
- `--disp-search-range-mode`：视差搜索范围模式（整数）。
- `--disp-search-offset`：视差搜索偏移（整数）。

显示与输出参数：
- `--show-color`：并排显示彩色图。
- `--min-depth-mm` / `--max-depth-mm`：深度可视化裁剪范围。
- `--colormap {bone,jet,turbo}`：深度伪彩色风格。
- `--timeout-ms`：单帧等待超时。
- `--frame-limit`：自动处理 N 帧后退出（调试很有用）。
- `--headless`：无 GUI 模式（用于算法调用/服务器环境）。
- `--output-dir`：按 `s` 保存快照时的输出目录。

### `run_stream_viewer.sh` / `stream_viewer`

除上述大多数参数外，新增：
- `--stream {rgb,depth,rgbd}`：选择显示流类型。
- `--list`：列出设备。
- `--list-profiles`：列出当前设备全部可用流配置。

### `diagnose_depth_startup.sh` / `diagnose_depth_startup`

用于自动探测“哪套参数最可能成功启动”：
- `--read-count`：每次探测读帧次数。
- `--timeout-ms`：探测读帧超时。
- `--probe-all-cameras/--no-probe-all-cameras`：是否继续探测其他相机。
- `--stop-on-first-success/--no-stop-on-first-success`：找到成功配置后是否立刻停止。
- `--output-json`：诊断报告输出路径。

### `smoke_test`

用于一次性验通：
- `--use-depth`：启用深度流。
- `--preview`：弹窗预览。
- `--read-count`：读取帧数后落盘。
- `--output-dir`：保存 `color.png`、`depth.npy`、`depth_preview.png`。

## 调参建议（按问题定位）

### 场景 1：高分辨率失败（你现在这个情况）

你当前命令 `1280x720@30 + --align-mode hw` 的失败信息是：
- `Current stream profile is not support hardware d2c process`

这类问题优先调整：
1. 把 `--align-mode hw` 改为 `--align-mode sw`。
2. 保留 `--profile-selection-strategy closest`。
3. 如仍不稳，降到 `640x400@15`。

推荐替代命令：

```bash
./scripts/gemini335l/run_depth_viewer.sh \
  --serial-number-or-name CP3F5420000Z \
  --width 1280 --height 720 --fps 30 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --show-color
```

### 场景 2：要优先保证算法可调用（不关心弹窗）

直接无界面运行：

```bash
./scripts/gemini335l/run_stream_viewer.sh \
  --serial-number-or-name CP3F5420000Z \
  --stream rgbd \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --headless --frame-limit 120
```

### 场景 3：需要尽量近距离测量

建议顺序：
1. 先用 `--depth-work-mode Hand`。
2. 再尝试 `--disp-search-range-mode` 与 `--disp-search-offset`。
3. 每次只改一个参数，记录结果。

注意：模式名包含空格时必须加引号，例如：

```bash
--depth-work-mode "High Accuracy"
```

## 最近工作距离调参（已实测）

设备：`CP3F5420000Z`，对齐：`--align-mode sw --align-depth-to-color`，同一环境下采样验证。

实测可用参数：
- `--depth-work-mode Default`：可用
- `--depth-work-mode Hand`：可用
- `--depth-work-mode "High Accuracy"`：可用（注意引号）

实测不可用参数（当前固件/SDK 组合）：
- `--disp-search-range-mode 1`：设备拒绝设置（errorCode: 2）
- `--disp-search-offset`：依赖上面的 range mode，当前也不建议使用

你可以直接用下面两套“已验证可跑”的近距配置：

```bash
# 方案 A：通用稳定（推荐先用）
./scripts/gemini335l/run_depth_viewer.sh \
  --serial-number-or-name CP3F5420000Z \
  --width 640 --height 360 --fps 30 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --depth-work-mode Hand \
  --show-color
```

```bash
# 方案 B：高精度模式（模式名有空格，必须加引号）
./scripts/gemini335l/run_depth_viewer.sh \
  --serial-number-or-name CP3F5420000Z \
  --width 640 --height 360 --fps 30 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --depth-work-mode "High Accuracy" \
  --show-color
```

## 常见故障与彻底修复

### 1）`ImportError: pyorbbecsdk`

在运行脚本的同一个环境里执行：

```bash
python -m pip install pyorbbecsdk2
```

说明：插件已兼容 `pyorbbecsdk` / `pyorbbecsdk2` 两种模块名。

### 2）`openUsbDevice failed` / `Permission denied`

按程序输出执行 Orbbec udev 安装脚本，然后拔插相机重试。

### 3）`uvc_stream_open_ctrl failed`

先跑诊断：

```bash
./scripts/gemini335l/diagnose_depth_startup.sh --serial-number-or-name <SERIAL>
```

优先使用低压力配置：`640x400@15` + `--align-mode sw`。

### 3.1）`Current stream profile is not support hardware d2c process`

这是硬件对齐（`hw`）与当前分辨率/帧率组合不兼容。

解决：
- 改用 `--align-mode sw`。
- 或降低分辨率/帧率（优先 `640x400@15`）。
- 建议保留 `--profile-selection-strategy closest`。

### 4）OpenCV 窗口报错（你这次遇到的）

报错特征：
- `cvNamedWindow`
- `The function is not implemented`

根因：环境安装了 `opencv-python-headless`（无 GUI 版本）。

彻底修复（在当前环境执行）：

```bash
python -m pip uninstall -y opencv-python-headless
python -m pip install opencv-python==4.12.0.88
```

修复后验证：

```bash
./scripts/gemini335l/run_depth_viewer.sh \
  --serial-number-or-name <你的相机序列号> \
  --width 640 --height 400 --fps 15 \
  --align-mode sw --align-depth-to-color \
  --profile-selection-strategy closest \
  --show-color --frame-limit 5
```

如果你只是做算法读帧、不需要窗口显示，可直接加 `--headless`。

## 辅助脚本

- `scripts/gemini335l/quick_check.sh`
- `scripts/gemini335l/diagnose_depth_startup.sh`
- `scripts/gemini335l/run_depth_viewer.sh`
- `scripts/gemini335l/run_stream_viewer.sh`
