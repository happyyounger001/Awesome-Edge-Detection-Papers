from __future__ import annotations

from core.models import LungeCycleState, LungeStateOutput, PhaseScoreSnapshot, ResolvedStandardProfile


class LungeStateMachine:
    def __init__(self, profile: ResolvedStandardProfile):
        self.profile = profile
        self.state = LungeCycleState.READY
        self.lunge_count = 0
        self.cycle_start_ms: int | None = None
        self.completion_timestamp_ms: int | None = None
        self.reached = False

    def update(self, snap: PhaseScoreSnapshot) -> LungeStateOutput:
        phase = snap.best_phase_key
        if self.state == LungeCycleState.READY:
            if phase == "STARTED":
                self.state = LungeCycleState.STARTED
                self.cycle_start_ms = snap.timestamp_ms
                self.reached = False

        elif self.state == LungeCycleState.STARTED:
            if self._timeout(snap):
                self._abort()
            elif phase == "EXTENDING":
                self.state = LungeCycleState.EXTENDING

        elif self.state == LungeCycleState.EXTENDING:
            if self._timeout(snap):
                self._abort()
            elif phase == "REACHED":
                self.reached = True
                self.state = LungeCycleState.REACHED

        elif self.state == LungeCycleState.REACHED:
            if self._timeout(snap):
                self._abort()
            elif phase == "RECOVERING":
                self.state = LungeCycleState.RECOVERING

        elif self.state == LungeCycleState.RECOVERING:
            if self._timeout(snap):
                self._abort()
            elif phase == "READY" and self.reached:
                self.lunge_count += 1
                self.completion_timestamp_ms = snap.timestamp_ms
                self.state = LungeCycleState.COMPLETED_LOCK
                return self._build_output(snap, True)

        elif self.state == LungeCycleState.COMPLETED_LOCK:
            if phase == "READY":
                self._reset_cycle()
                self.state = LungeCycleState.READY

        elif self.state == LungeCycleState.ABORTED:
            if phase == "READY":
                self._reset_cycle()
                self.state = LungeCycleState.READY

        return self._build_output(snap, False)

    def _timeout(self, snap: PhaseScoreSnapshot) -> bool:
        if self.cycle_start_ms is None:
            return False
        phase_map = self.profile.phase_definitions_map
        timeout = phase_map.get(self.state.value).timeout_ms if self.state.value in phase_map else 1200
        return snap.timestamp_ms - self.cycle_start_ms > timeout

    def _abort(self) -> None:
        self.state = LungeCycleState.ABORTED
        self.cycle_start_ms = None
        self.reached = False

    def _reset_cycle(self) -> None:
        self.cycle_start_ms = None
        self.reached = False

    def _build_output(self, snap: PhaseScoreSnapshot, just_completed: bool) -> LungeStateOutput:
        return LungeStateOutput(
            cycle_state=self.state,
            lunge_count=self.lunge_count,
            just_completed=just_completed,
            completion_timestamp_ms=self.completion_timestamp_ms if just_completed else None,
            debug_info={"phase": snap.best_phase_key, "frame": snap.frame_index},
        )
