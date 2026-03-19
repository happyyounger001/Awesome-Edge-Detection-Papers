# Fencing Lunge AI Trainer

这是一个面向 Windows 的 Python GUI 原型，用于分析儿童花剑弓步训练视频。

## 功能概览

本项目提供一个桌面界面，支持：

- 导入训练视频。
- 播放视频并叠加 MediaPipe Pose 人体骨架。
- 手动标记 `go` 时刻。
- 自动分析：
  - 手脚顺序
  - 弓步稳定性（含手 / 肘 / 肩晃动）
  - 是否抬大腿
  - 是否歪头
  - 是否踢小腿
- 输出敏捷性时间：
  - `go → 出手`
  - `go → 弓步完成`
  - `出手 → 弓步完成`
- 生成中文图文报告窗口。
- 导出叠加视频、关键帧截图、CSV 指标和图表。
- 支持视频加载 / 分析 / 报告加载的圆形百分比进度弹窗。
- 支持 0.5x / 1.0x / 1.5x / 2.0x 播放倍速与重播。

## 技术栈

- Python 3.10+
- GUI: PySide6
- 视频处理: OpenCV
- 姿态识别: MediaPipe Pose
- 数据处理: NumPy / Pandas
- 图表: Matplotlib
- 中文字幕绘制: Pillow
- 配置: YAML

## 目录结构

```text
project/
├── ui/
├── pose/
├── analysis/
├── rules/
├── report/
├── config/
├── main.py
```

## Windows 安装

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## 运行方式

```powershell
python main.py
```

## 使用流程

1. 点击“导入视频”。
2. 在主界面播放/暂停视频并拖动进度条。
3. 在需要时点击“手动标记 go”。
4. 点击“开始分析”。
5. 使用倍速按钮和进度条进行人工复核。
6. 点击“查看分析报告”，打开中文独立报告窗口。


## 能生成 EXE 吗？

可以。项目已经补充了 **PyInstaller 打包脚本** 和 `.spec` 文件，可在 Windows 上生成桌面版 EXE。

### 方式 1：使用打包脚本（推荐）

```powershell
python build_exe.py
```

生成目录：

```text
dist/
  FencingLungeAITrainer/
    FencingLungeAITrainer.exe
```

如果你想生成单文件版本：

```powershell
python build_exe.py --onefile
```

### 方式 2：直接使用 spec 文件

```powershell
pyinstaller FencingLungeAITrainer.spec
```

### 打包说明

- 打包时会自动把 `config.yaml` 一起带入 EXE 输出目录。
- 脚本已经包含 `mediapipe`、`matplotlib`、`pandas` 的收集参数，减少 Windows 下缺模块问题。
- 对于 MediaPipe + OpenCV + PySide6 组合，**优先推荐 one-folder 版本**，稳定性通常高于 one-file。
- 首次打包前请先确认：
  - `pip install -r requirements.txt`
  - `python main.py` 可以正常启动

## 输出目录

分析完成后会在 `outputs/<视频名>/` 生成：

```text
overlay.mp4
report_zh.html
metrics.csv
figures/
  time_compare.png
  stability_curve.png
frames/
  出手瞬间.png
  弓步完成瞬间.png
  预警触发瞬间.png
```

## 参数调节

界面右侧新增“当前参数设置”面板，可直接查看并调整：

- **弓步稳定性**：多少秒内保持不晃动才算达标。
- **抬大腿角度阈值**：膝-髋连线与地面夹角达到多少度时触发“抬大腿”判定。
- **头部偏斜阈值**：头部线条偏离水平超过多少度触发“歪头提醒”。
- **膝关节点需低于髋关节点**：作为抬大腿判定的附加条件，可直接勾选启用。

点击“应用参数”后，实时分析区、视频字幕预警、正式分析结果会使用同一套参数。

## 规则说明

- **手脚顺序**：当前产品逻辑保持为“先手后脚”判定为达标，异常字幕显示为“先脚后手预警”。
- **弓步计数**：严格按状态机执行，只有 `IDLE -> LUNGE_OUT -> LUNGE_RETURN -> IDLE` 完整结束后，已完成弓步数才会 +1。
- **弓步稳定性**：弓步峰值后在设定秒数内，手 / 肘 / 肩晃动都不超过阈值才算达标。
- **抬大腿**：当“膝关节点低于髋关节点”且“膝-髋连线与地面夹角 >= 阈值”时触发预警。
- **头部姿态**：左右眼连线偏离水平超过阈值时触发“歪头提醒”。
- **踢小腿**：踝相对膝的前摆速度超过阈值判定为有踢小腿。

所有阈值都可在 `config.yaml` 中修改。

## 最低可运行版本能力

- GUI 可打开视频。
- 骨架叠加与中文字幕不乱码。
- 支持手动 `go` 标记。
- 计算三个关键时间。
- 完成四类规则判断。
- 实时分析区显示“当前第 X 个弓步 / 已完成弓步数 / 当前状态 / 是否得分 / 细项判定 / 状态机”。
- 参数面板支持修改弓步稳定性、抬大腿阈值、头部偏斜阈值。
- 视频加载 / 分析 / 报告加载都有圆形进度弹窗。
- 生成中文报告窗口。

## 测试

```bash
pytest -q
python -m compileall ui pose analysis rules report config main.py tests
```

## 限制说明

- 推荐固定侧面机位，确保全身入镜。
- 当前为 2D 姿态估计，不追踪剑尖。
- 光照差、遮挡、摄像机移动会影响结果。
- 当前 `go` 至少支持手动标记，自动音频检测未作为本版 MVP 强制项。
