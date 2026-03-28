from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PoseFrame:
    index: int
    timestamp: float
    points: dict[str, tuple[float, float]]
    visibility: dict[str, float]


@dataclass
class PoseSequence:
    fps: float
    width: int
    height: int
    frames: list[PoseFrame]
    arrays: dict[str, np.ndarray]

    @property
    def frame_count(self) -> int:
        return len(self.frames)
