from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from core.models import (
    FeatureDefinition,
    PhaseDefinition,
    ResolvedStandardProfile,
    ScoringRule,
    TransitionRule,
)


class StandardProfileError(ValueError):
    pass


class StandardLoader:
    def __init__(self, source_path: str | Path):
        self.source_path = Path(source_path)

    def load_profile(self, lesson_id: str, action_type: str) -> ResolvedStandardProfile:
        if not self.source_path.exists():
            raise StandardProfileError(f"standard source not found: {self.source_path}")
        data = json.loads(self.source_path.read_text(encoding="utf-8"))
        required = ["lesson_id", "action_type", "version", "feature_definitions", "phase_definitions", "transition_rules", "scoring_rules", "ui_labels"]
        missing = [k for k in required if k not in data]
        if missing:
            raise StandardProfileError(f"当前学习标准未加载完成，缺失字段: {', '.join(missing)}")
        if data["lesson_id"] != lesson_id or data["action_type"] != action_type:
            raise StandardProfileError("profile does not match requested lesson/action")

        profile = ResolvedStandardProfile(
            lesson_id=data["lesson_id"],
            action_type=data["action_type"],
            version=data["version"],
            feature_definitions=[FeatureDefinition(**f) for f in data["feature_definitions"]],
            phase_definitions=[PhaseDefinition(**p) for p in data["phase_definitions"]],
            transition_rules=[TransitionRule(**t) for t in data["transition_rules"]],
            scoring_rules=[ScoringRule(**s) for s in data["scoring_rules"]],
            ui_labels=dict(data["ui_labels"]),
        )
        for phase in profile.phase_definitions:
            if phase.enter_threshold <= phase.exit_threshold:
                raise StandardProfileError(f"invalid hysteresis for phase {phase.phase_key}")
        return profile

    @staticmethod
    def dump_profile(profile: ResolvedStandardProfile, output_path: str | Path) -> None:
        data = asdict(profile)
        Path(output_path).write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
