from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config import AppConfig
from rules.evaluator import SequenceEvaluation


class ReportGenerator:
    def __init__(self, config: AppConfig):
        self.config = config

    def generate(
        self,
        video_path: Path,
        output_dir: Path,
        evaluation: SequenceEvaluation,
        keyframes: dict[str, Path],
    ) -> tuple[Path, dict[str, Path]]:
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)

        timing_chart = self._build_timing_chart(figures_dir / "time_compare.png", evaluation)
        stability_chart = self._build_stability_chart(figures_dir / "stability_curve.png", evaluation.metrics)
        charts = {"timing": timing_chart, "stability": stability_chart}

        html_path = output_dir / "report_zh.html"
        html_path.write_text(self._render_html(video_path, evaluation, charts, keyframes), encoding="utf-8")
        csv_path = output_dir / "metrics.csv"
        evaluation.metrics.to_csv(csv_path, index=False, encoding="utf-8-sig")
        return html_path, charts

    def _build_timing_chart(self, path: Path, evaluation: SequenceEvaluation) -> Path:
        labels = ["go→出手", "go→弓步", "出手→弓步"]
        values = [
            evaluation.timing.go_to_hand or 0.0,
            evaluation.timing.go_to_finish or 0.0,
            evaluation.timing.hand_to_finish or 0.0,
        ]
        plt.figure(figsize=(6, 4))
        plt.bar(labels, values, color=["#4caf50", "#2196f3", "#ff9800"])
        plt.ylabel("秒")
        plt.title("时间对比图")
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def _build_stability_chart(self, path: Path, metrics: pd.DataFrame) -> Path:
        plt.figure(figsize=(7, 4))
        plt.plot(metrics["time"], metrics["body_speed"], label="身体速度")
        plt.plot(metrics["time"], metrics["foot_speed"], label="脚部速度", alpha=0.75)
        plt.xlabel("时间（秒）")
        plt.ylabel("归一化速度")
        plt.title("稳定性变化图")
        plt.legend()
        plt.tight_layout()
        plt.savefig(path)
        plt.close()
        return path

    def _render_html(
        self,
        video_path: Path,
        evaluation: SequenceEvaluation,
        charts: dict[str, Path],
        keyframes: dict[str, Path],
    ) -> str:
        def fmt(value: float | None) -> str:
            return "未检测" if value is None else f"{value:.3f} 秒"

        created_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        quality = evaluation.quality
        detected_go = "是" if evaluation.detected_go or evaluation.timing.go_time is not None else "否"
        keyframe_blocks = "".join(
            f'<div class="frame-card"><h4>{title}</h4><img src="{path.name}" alt="{title}"></div>'
            for title, path in keyframes.items()
        )
        explanation_blocks = "".join(f"<li>{line}</li>" for line in quality.explanations)
        return f"""
<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <title>击剑弓步AI分析报告</title>
  <style>
    body {{ font-family: 'Microsoft YaHei', sans-serif; margin: 24px; color: #222; }}
    h1, h2 {{ color: #114b8b; }}
    table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
    th, td {{ border: 1px solid #d0d7de; padding: 8px 10px; text-align: left; }}
    th {{ background: #eef5ff; }}
    .tag-good {{ color: #14833b; font-weight: bold; }}
    .tag-warn {{ color: #c62828; font-weight: bold; }}
    .charts, .frames {{ display: flex; gap: 18px; flex-wrap: wrap; }}
    .chart-card, .frame-card {{ width: 48%; min-width: 320px; }}
    img {{ width: 100%; border: 1px solid #d0d7de; border-radius: 8px; }}
    .summary {{ background: #f7fbff; padding: 12px 16px; border-radius: 8px; margin-bottom: 20px; }}
  </style>
</head>
<body>
  <h1>击剑弓步AI分析报告</h1>
    <div class="summary">
    <p><strong>视频名称：</strong>{video_path.name}</p>
    <p><strong>分析时间：</strong>{created_at}</p>
    <p><strong>是否检测到 go：</strong>{detected_go}</p>
    <p><strong>完整弓步数：</strong>{evaluation.completed_lunges}</p>
    <p><strong>总体评价：</strong><span class="{'tag-good' if quality.overall_status == '👍' else 'tag-warn'}">{quality.overall_status}</span></p>
    <p><strong>实时状态：</strong>{quality.status_text}</p>
    <p><strong>本次最需要改进：</strong>{quality.top_issue or '动作整体达标'}</p>
    <p><strong>训练建议：</strong>{quality.top_advice or '继续保持当前节奏和稳定性'}</p>
    <p><strong>质量评分：</strong>总分 {quality.overall_score:.2f} / 姿态 {quality.posture_score:.2f} / 时序 {quality.timing_score:.2f} / 稳定 {quality.stability_score:.2f} / 协调 {quality.coordination_score:.2f}</p>
    <p><strong>当前参数：</strong>稳定性 {self.config.stability_seconds_threshold:.2f} 秒；上肢晃动阈值 {self.config.upper_body_motion_threshold:.3f}；抬大腿角度阈值 {self.config.thigh_raise_angle_threshold:.1f} 度；头部偏斜阈值 {self.config.head_tilt_angle_threshold:.1f} 度；膝低于髋 = {self.config.require_knee_below_hip_for_thigh_raise}</p>
  </div>

  <h2>时间分析表</h2>
  <table>
    <tr><th>项目</th><th>时间</th></tr>
    <tr><td>go→出手</td><td>{fmt(evaluation.timing.go_to_hand)}</td></tr>
    <tr><td>go→弓步</td><td>{fmt(evaluation.timing.go_to_finish)}</td></tr>
    <tr><td>出手→弓步</td><td>{fmt(evaluation.timing.hand_to_finish)}</td></tr>
  </table>

  <h2>动作质量表</h2>
  <table>
    <tr><th>项目</th><th>结果</th></tr>
    <tr><td>手脚顺序</td><td>{quality.hand_foot_order} {quality.hand_foot_status}</td></tr>
    <tr><td>弓步稳定</td><td>{quality.stability} {quality.stability_status}</td></tr>
    <tr><td>抬大腿</td><td>{quality.thigh_raise} {quality.thigh_raise_status}</td></tr>
    <tr><td>头部姿态</td><td>{quality.head_tilt} {quality.head_tilt_status}</td></tr>
    <tr><td>踢小腿</td><td>{quality.calf_kick} {quality.calf_kick_status}</td></tr>
    <tr><td>总体评价</td><td>{quality.overall_status}</td></tr>
  </table>

  <h2>图表分析</h2>
  <div class="charts">
    <div class="chart-card"><h4>时间对比图</h4><img src="figures/{charts['timing'].name}" alt="时间对比图"></div>
    <div class="chart-card"><h4>稳定性变化图</h4><img src="figures/{charts['stability'].name}" alt="稳定性变化图"></div>
  </div>

  <h2>图文解释</h2>
  <ul>
    {explanation_blocks}
  </ul>

  <h2>训练建议</h2>
  <table>
    <tr><th>字段</th><th>内容</th></tr>
    <tr><td>第 X 个弓步</td><td>第 {evaluation.frame_assessments[-1].current_lunge_index if evaluation.frame_assessments else 1} 个弓步</td></tr>
    <tr><td>当前状态</td><td>{quality.status_text}</td></tr>
    <tr><td>最关键问题</td><td>{quality.top_issue or '动作整体达标'}</td></tr>
    <tr><td>建议</td><td>{quality.top_advice or '继续保持当前节奏和稳定性'}</td></tr>
  </table>

  <h2>偏差说明</h2>
  <ul>
    {"".join(f"<li>{item}</li>" for item in (quality.deviations or ["当前动作与标准区间接近"]))}
  </ul>

  <h2>关键帧截图</h2>
  <div class="frames">
    {keyframe_blocks}
  </div>
</body>
</html>
"""
