from __future__ import annotations

from pathlib import Path

from analysis.models import AnalysisResult
from config import AppConfig
from pose.estimator import PoseEstimator
from report.generator import ReportGenerator
from rules.evaluator import evaluate_sequence
from ui.video_overlay import render_overlay_video


class AnalysisEngine:
    def __init__(self, config: AppConfig):
        self.config = config
        self.pose_estimator = PoseEstimator(config)
        self.report_generator = ReportGenerator(config)

    def close(self) -> None:
        self.pose_estimator.close()

    def analyze_video(
        self,
        video_path: str | Path,
        output_dir: str | Path,
        manual_go_time: float | None,
        progress_callback=None,
    ) -> AnalysisResult:
        video_path = Path(video_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        if progress_callback is not None:
            progress_callback(5)
        pose_sequence = self.pose_estimator.extract(
            video_path,
            progress_callback=(lambda percent: progress_callback(5 + int(percent * 0.35))) if progress_callback else None,
        )
        if progress_callback is not None:
            progress_callback(45)
        evaluation = evaluate_sequence(pose_sequence, self.config, manual_go_time)

        if progress_callback is not None:
            progress_callback(70)
        overlay_video, keyframes = render_overlay_video(
            video_path=video_path,
            output_dir=output_dir,
            pose_sequence=pose_sequence,
            evaluation=evaluation,
        )
        if progress_callback is not None:
            progress_callback(90)
        report_html, charts = self.report_generator.generate(
            video_path=video_path,
            output_dir=output_dir,
            evaluation=evaluation,
            keyframes=keyframes,
        )
        if progress_callback is not None:
            progress_callback(100)

        return AnalysisResult(
            video_path=video_path,
            output_dir=output_dir,
            timing=evaluation.timing,
            quality=evaluation.quality,
            frame_assessments=evaluation.frame_assessments,
            report_html=report_html,
            overlay_video=overlay_video,
            keyframes=keyframes,
            detected_go=evaluation.detected_go,
            charts=charts,
        )
