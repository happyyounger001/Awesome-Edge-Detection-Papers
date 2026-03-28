from __future__ import annotations

from collections import defaultdict

from core.models import PhaseScoreSnapshot, ResolvedStandardProfile


class TemporalSmoother:
    def __init__(self, profile: ResolvedStandardProfile):
        self.profile = profile
        self.current_phase = profile.phase_definitions[0].phase_key if profile.phase_definitions else ""
        self._hold = defaultdict(int)

    def update(self, score_snapshot: PhaseScoreSnapshot) -> PhaseScoreSnapshot:
        phase_map = self.profile.phase_definitions_map
        cur_def = phase_map.get(self.current_phase)
        current_score = score_snapshot.phase_scores.get(self.current_phase, 0.0)
        if cur_def and current_score < cur_def.exit_threshold:
            candidate = self.current_phase
            candidate_score = current_score
            for phase in self.profile.phase_definitions:
                score = score_snapshot.phase_scores.get(phase.phase_key, 0.0)
                if score >= phase.enter_threshold and score > candidate_score:
                    candidate = phase.phase_key
                    candidate_score = score
            if candidate != self.current_phase:
                self._hold[candidate] += 1
                need = phase_map[candidate].min_hold_frames
                if self._hold[candidate] >= need:
                    self.current_phase = candidate
                    self._hold.clear()
            else:
                self._hold.clear()
        return PhaseScoreSnapshot(
            timestamp_ms=score_snapshot.timestamp_ms,
            frame_index=score_snapshot.frame_index,
            phase_scores=score_snapshot.phase_scores,
            best_phase_key=self.current_phase,
        )
