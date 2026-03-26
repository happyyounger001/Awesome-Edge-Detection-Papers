from __future__ import annotations

from math import exp

from core.models import FrameFeatures, PhaseScoreSnapshot, ResolvedStandardProfile


def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + exp(-x))


class PhaseScorer:
    def __init__(self, profile: ResolvedStandardProfile):
        self.profile = profile
        self._rules = {rule.phase_key: rule.feature_weights for rule in profile.scoring_rules}

    def score(self, features: FrameFeatures) -> PhaseScoreSnapshot:
        scores: dict[str, float] = {}
        for phase in self.profile.phase_definitions:
            weights = self._rules.get(phase.phase_key, {})
            raw = 0.0
            for key, w in weights.items():
                raw += features.feature_values.get(key, 0.0) * w
            scores[phase.phase_key] = max(0.0, min(1.0, _sigmoid(raw)))
        best = max(scores.items(), key=lambda kv: kv[1])[0] if scores else ""
        return PhaseScoreSnapshot(
            timestamp_ms=features.timestamp_ms,
            frame_index=features.frame_index,
            phase_scores=scores,
            best_phase_key=best,
        )
