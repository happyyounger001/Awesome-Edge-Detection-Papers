from __future__ import annotations

from typing import Any, Callable

from core.models import AnalysisResult, FramePacket, ResolvedStandardProfile
from core.phase_scorer import PhaseScorer


class AnalyzerEngine:
    def __init__(
        self,
        profile: ResolvedStandardProfile,
        feature_extractor,
        phase_scorer: PhaseScorer,
        smoother,
        lunge_sm,
        pose_infer: Callable[[Any], Any] | None = None,
    ):
        self.profile = profile
        self.feature_extractor = feature_extractor
        self.phase_scorer = phase_scorer
        self.smoother = smoother
        self.lunge_sm = lunge_sm
        self.pose_infer = pose_infer or (lambda frame: None)

    def analyze(self, frame_packet: FramePacket) -> AnalysisResult:
        pose_data = self.pose_infer(frame_packet.frame)
        features = self.feature_extractor.extract(
            frame_packet.frame,
            pose_data,
            frame_packet.timestamp_ms,
            frame_packet.frame_index,
        )
        scores = self.phase_scorer.score(features)
        smoothed = self.smoother.update(scores)
        lunge_state = self.lunge_sm.update(smoothed)
        phase_cn = self.profile.phase_definitions_map.get(smoothed.best_phase_key)

        return AnalysisResult(
            timestamp_ms=frame_packet.timestamp_ms,
            frame_index=frame_packet.frame_index,
            current_phase_key=smoothed.best_phase_key,
            current_phase_cn=phase_cn.display_name_cn if phase_cn else smoothed.best_phase_key,
            phase_scores=smoothed.phase_scores,
            lunge_count=lunge_state.lunge_count,
            lunge_state=lunge_state.cycle_state.value,
            metrics=features.feature_values,
            warnings=[],
        )
