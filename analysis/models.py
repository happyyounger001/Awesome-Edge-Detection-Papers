from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TimingResult:
    go_time: float | None
    hand_start_time: float | None
    foot_start_time: float | None
    lunge_finish_time: float | None

    @property
    def go_to_hand(self) -> float | None:
        if self.go_time is None or self.hand_start_time is None:
            return None
        return self.hand_start_time - self.go_time

    @property
    def go_to_finish(self) -> float | None:
        if self.go_time is None or self.lunge_finish_time is None:
            return None
        return self.lunge_finish_time - self.go_time

    @property
    def hand_to_finish(self) -> float | None:
        if self.hand_start_time is None or self.lunge_finish_time is None:
            return None
        return self.lunge_finish_time - self.hand_start_time


@dataclass
class LungeQuality:
    hand_foot_order: str
    hand_foot_status: str
    stability: str
    stability_status: str
    thigh_raise: str
    thigh_raise_status: str
    head_tilt: str
    head_tilt_status: str
    calf_kick: str
    calf_kick_status: str
    overall_status: str
    posture_score: float = 0.0
    timing_score: float = 0.0
    stability_score: float = 0.0
    coordination_score: float = 0.0
    overall_score: float = 0.0
    status_text: str = "加油，还能更好！"
    top_issue: str = ""
    top_advice: str = ""
    explanations: list[str] = field(default_factory=list)
    deviations: list[str] = field(default_factory=list)


@dataclass
class FrameAssessment:
    frame_index: int
    timestamp: float
    current_lunge_index: int
    completed_lunge_count: int
    lunge_state: str
    current_text: str
    current_icon: str
    score_text: str
    top_issue: str
    coaching_advice: str
    hand_foot_order: str
    stability: str
    stability_details: str
    thigh_raise: str
    head_tilt: str
    calf_kick: str
    show_thigh_warning: bool
    show_order_warning: bool
    show_head_tilt_warning: bool


@dataclass
class AnalysisResult:
    video_path: Path
    output_dir: Path
    timing: TimingResult
    quality: LungeQuality
    frame_assessments: list[FrameAssessment]
    report_html: Path
    overlay_video: Path
    keyframes: dict[str, Path]
    detected_go: bool
    charts: dict[str, Path]
