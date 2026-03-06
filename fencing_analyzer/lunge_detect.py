from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from .config import AnalyzerConfig
from .pose import PoseSequence


@dataclass
class LungeEvent:
    lunge_id: int
    start_frame: int
    end_frame: int

    @property
    def duration_frames(self) -> int:
        return self.end_frame - self.start_frame + 1


def _select_front_back_ankles(pose: PoseSequence) -> tuple[np.ndarray, np.ndarray]:
    left = pose.x["left_ankle"]
    right = pose.x["right_ankle"]
    # side view assumption: larger temporal mean x is front side
    if float(np.mean(left)) > float(np.mean(right)):
        return left, right
    return right, left


def detect_lunges(pose: PoseSequence, cfg: AnalyzerConfig) -> List[LungeEvent]:
    front, back = _select_front_back_ankles(pose)
    dt = 1.0 / pose.fps
    vel = np.maximum(np.abs(np.gradient(front, dt)), np.abs(np.gradient(back, dt)))

    min_len = max(int(cfg.min_lunge_duration * pose.fps), 1)
    min_gap = int(cfg.min_gap_between_lunges * pose.fps)

    events: List[LungeEvent] = []
    active = False
    start = 0
    settle = 0

    for i, v in enumerate(vel):
        if not active and v >= cfg.v_start_thr:
            active = True
            start = i
            settle = 0
            continue

        if active:
            if v <= cfg.v_end_thr:
                settle += 1
            else:
                settle = 0

            if settle >= cfg.settle_frames:
                end = i - cfg.settle_frames
                if end - start + 1 >= min_len:
                    if not events or start - events[-1].end_frame >= min_gap:
                        events.append(LungeEvent(len(events) + 1, start, end))
                active = False
                settle = 0

    if active:
        end = len(vel) - 1
        if end - start + 1 >= min_len and (not events or start - events[-1].end_frame >= min_gap):
            events.append(LungeEvent(len(events) + 1, start, end))

    return events
