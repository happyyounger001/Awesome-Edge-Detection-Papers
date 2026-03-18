# Fencing Lunge AI Trainer

这是一个面向 Windows 的 Python GUI 原型，用于分析儿童花剑弓步训练视频。

## 功能概览

本项目提供一个桌面界面，支持：

- 导入训练视频。
- 播放视频并叠加 MediaPipe Pose 人体骨架。
- 手动标记 `go` 时刻。
- 自动分析：
  - 手脚顺序
  - 弓步稳定性
  - 是否抬大腿
  - 是否踢小腿
- 输出敏捷性时间：
  - `go → 出手`
  - `go → 弓步完成`
  - `出手 → 弓步完成`
- 生成中文图文报告窗口。
- 导出叠加视频、关键帧截图、CSV 指标和图表。

## 技术栈

- Python 3.10+
- GUI: PySide6
- 视频处理: OpenCV
- 姿态识别: MediaPipe Pose
- 数据处理: NumPy / Pandas
- 图表: Matplotlib
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
5. 点击“查看分析报告”，打开中文独立报告窗口。

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

## 规则说明

- **手脚顺序**：持剑手手腕速度首次超过阈值，与前脚速度首次超过阈值比较。
- **弓步稳定性**：弓步峰值后身体速度稳定所需时间是否超过 1 秒。
- **抬大腿**：肩-髋-膝夹角超过阈值时触发预警。
- **踢小腿**：踝相对膝的前摆速度超过阈值判定为有踢小腿。

所有阈值都可在 `config.yaml` 中修改。

## 最低可运行版本能力

- GUI 可打开视频。
- 骨架叠加。
- 支持手动 `go` 标记。
- 计算三个关键时间。
- 完成四类规则判断。
- 在界面中显示 👍 / 🚨。
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
