from __future__ import annotations

from core.lunge_state_machine import LungeStateMachine
from core.models import FeatureDefinition, PhaseDefinition, PhaseScoreSnapshot, ResolvedStandardProfile, ScoringRule, TransitionRule


def _profile():
    return ResolvedStandardProfile(
        lesson_id="L1",
        action_type="lunge",
        version="v1",
        feature_definitions=[FeatureDefinition("f", "f", "f")],
        phase_definitions=[
            PhaseDefinition("READY", "准备", "", 0.8, 0.6, 1, 5000),
            PhaseDefinition("STARTED", "启动", "", 0.8, 0.6, 1, 5000),
            PhaseDefinition("EXTENDING", "推进", "", 0.8, 0.6, 1, 5000),
            PhaseDefinition("REACHED", "到位", "", 0.8, 0.6, 1, 5000),
            PhaseDefinition("RECOVERING", "回收", "", 0.8, 0.6, 1, 5000),
        ],
        transition_rules=[TransitionRule("READY", "STARTED")],
        scoring_rules=[ScoringRule("READY", {"f": 1.0})],
        ui_labels={},
    )


def _snap(i, phase):
    return PhaseScoreSnapshot(timestamp_ms=i * 100, frame_index=i, phase_scores={phase: 1.0}, best_phase_key=phase)


def test_complete_lunge_cycle_counts_once():
    sm = LungeStateMachine(_profile())
    phases = ["STARTED", "EXTENDING", "REACHED", "RECOVERING", "READY", "READY"]
    out = None
    for i, p in enumerate(phases, start=1):
        out = sm.update(_snap(i, p))
    assert out is not None
    assert sm.lunge_count == 1


def test_missing_started_never_backfill():
    sm = LungeStateMachine(_profile())
    for i, p in enumerate(["REACHED", "RECOVERING", "READY"], start=1):
        sm.update(_snap(i, p))
    assert sm.lunge_count == 0


def test_two_independent_lunges_count_two():
    sm = LungeStateMachine(_profile())
    seq = ["STARTED", "EXTENDING", "REACHED", "RECOVERING", "READY", "READY"] * 2
    for i, p in enumerate(seq, start=1):
        sm.update(_snap(i, p))
    assert sm.lunge_count == 2


def test_timeout_aborts_and_rearm_from_ready():
    profile = _profile()
    profile.phase_definitions[1].timeout_ms = 100
    sm = LungeStateMachine(profile)
    sm.update(_snap(1, "STARTED"))
    sm.update(_snap(5, "STARTED"))
    assert sm.state.value == "ABORTED"
    sm.update(_snap(6, "READY"))
    assert sm.state.value == "READY"
