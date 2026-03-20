from __future__ import annotations

from pathlib import Path
from typing import Any

from analysis.models import AnalysisResult
from config import AppConfig
from feature_extractor import extract_video_features
from pose.estimator import PoseEstimator
from quality_scorer import build_training_feedback, score_features
from report.generator import ReportGenerator
from rules.evaluator import evaluate_sequence
from ui.video_overlay import render_overlay_video


class AnalysisEngine:
    def __init__(self, config: AppConfig, quality_standard: dict[str, Any] | None = None):
        self.config = config
        self.quality_standard = quality_standard
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
        feature_summary = extract_video_features(pose_sequence, self.config, str(video_path), category="training_video")
        if self.quality_standard is not None:
            scores, explanations = score_features(feature_summary, self.quality_standard)
            feedback = build_training_feedback(feature_summary, self.quality_standard)
            evaluation.quality.posture_score = scores.get("posture_score", 0.0)
            evaluation.quality.timing_score = scores.get("timing_score", 0.0)
            evaluation.quality.stability_score = scores.get("stability_score", 0.0)
            evaluation.quality.coordination_score = scores.get("coordination_score", 0.0)
            evaluation.quality.overall_score = scores.get("overall_score", 0.0)
            evaluation.quality.deviations = explanations
            evaluation.quality.status_text = feedback["status_text"]
            evaluation.quality.top_issue = feedback["top_issue"]
            evaluation.quality.top_advice = feedback["top_advice"]
            for assessment in evaluation.frame_assessments:
                assessment.current_text = feedback["status_text"]
                assessment.score_text = "得分" if feedback["status_text"] == "很棒，得分！" else "未得分"
                assessment.top_issue = feedback["top_issue"]
                assessment.coaching_advice = feedback["top_advice"]

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
