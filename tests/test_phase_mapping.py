from __future__ import annotations

from core.models import (
    FeatureDefinition,
    PhaseDefinition,
    ResolvedStandardProfile,
    ScoringRule,
    TransitionRule,
)


def _profile():
    return ResolvedStandardProfile(
        lesson_id="L1",
        action_type="lunge",
        version="v1",
        feature_definitions=[FeatureDefinition("f", "f", "f")],
        phase_definitions=[
            PhaseDefinition("READY", "准备位稳定", "desc", 0.8, 0.6, 1, 1200),
            PhaseDefinition("RECOVERING", "回收复原中", "desc", 0.8, 0.6, 1, 1200),
        ],
        transition_rules=[TransitionRule("READY", "STARTED")],
        scoring_rules=[ScoringRule("READY", {"f": 1.0})],
        ui_labels={"READY": "准备位稳定", "RECOVERING": "回收复原中"},
    )


def test_all_phases_have_cn_mapping():
    profile = _profile()
    for phase in profile.phase_definitions:
        assert phase.display_name_cn
        assert profile.ui_labels.get(phase.phase_key)
