from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional


@dataclass
class FeatureDefinition:
    feature_key: str
    display_name_cn: str
    description_cn: str
    params: Dict[str, float] = field(default_factory=dict)


@dataclass
class PhaseDefinition:
    phase_key: str
    display_name_cn: str
    description_cn: str
    enter_threshold: float
    exit_threshold: float
    min_hold_frames: int
    timeout_ms: int


@dataclass
class TransitionRule:
    from_phase: str
    to_phase: str
    required: bool = True


@dataclass
class ScoringRule:
    phase_key: str
    feature_weights: Dict[str, float]


@dataclass
class ResolvedStandardProfile:
    lesson_id: str
    action_type: str
    version: str
    feature_definitions: List[FeatureDefinition]
    phase_definitions: List[PhaseDefinition]
    transition_rules: List[TransitionRule]
    scoring_rules: List[ScoringRule]
    ui_labels: Dict[str, str]

    @property
    def phase_definitions_map(self) -> Dict[str, PhaseDefinition]:
        return {item.phase_key: item for item in self.phase_definitions}


@dataclass
class FrameFeatures:
    timestamp_ms: int
    frame_index: int
    feature_values: Dict[str, float]


@dataclass
class PhaseScoreSnapshot:
    timestamp_ms: int
    frame_index: int
    phase_scores: Dict[str, float]
    best_phase_key: str


@dataclass
class AnalysisResult:
    timestamp_ms: int
    frame_index: int
    current_phase_key: str
    current_phase_cn: str
    phase_scores: Dict[str, float]
    lunge_count: int
    lunge_state: str
    metrics: Dict[str, float]
    warnings: List[str]


@dataclass
class FramePacket:
    frame_index: int
    timestamp_ms: int
    frame: object


class LungeCycleState(str, Enum):
    READY = "READY"
    STARTED = "STARTED"
    EXTENDING = "EXTENDING"
    REACHED = "REACHED"
    RECOVERING = "RECOVERING"
    COMPLETED_LOCK = "COMPLETED_LOCK"
    ABORTED = "ABORTED"


@dataclass
class LungeStateOutput:
    cycle_state: LungeCycleState
    lunge_count: int
    just_completed: bool
    completion_timestamp_ms: Optional[int]
    debug_info: dict
