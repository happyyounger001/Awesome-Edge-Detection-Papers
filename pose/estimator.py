from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from config import AppConfig
from pose.models import PoseFrame, PoseSequence

KEYPOINTS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
}

SKELETON = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
    ("left_shoulder", "left_wrist"),
    ("right_shoulder", "right_wrist"),
]


class PoseEstimator:
    def __init__(self, config: AppConfig):
        self.config = config
        self._pose = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self._pose.close()

    def extract(self, video_path: str | Path) -> PoseSequence:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        raw_xy = {name: [] for name in KEYPOINTS}
        raw_vis = {name: [] for name in KEYPOINTS}
        frames: list[PoseFrame] = []
        frame_index = 0

        while True:
            ok, frame = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = self._pose.process(rgb)

            points: dict[str, tuple[float, float]] = {}
            visibilities: dict[str, float] = {}
            if result.pose_landmarks is None:
                for name in KEYPOINTS:
                    raw_xy[name].append((np.nan, np.nan))
                    raw_vis[name].append(0.0)
                    points[name] = (0.0, 0.0)
                    visibilities[name] = 0.0
            else:
                landmarks = result.pose_landmarks.landmark
                for name, idx in KEYPOINTS.items():
                    lm = landmarks[idx]
                    raw_xy[name].append((lm.x, lm.y))
                    raw_vis[name].append(lm.visibility)
                    points[name] = (lm.x, lm.y)
                    visibilities[name] = lm.visibility

            frames.append(PoseFrame(frame_index, frame_index / fps, points, visibilities))
            frame_index += 1

        cap.release()
        arrays = self._smooth_arrays(raw_xy, raw_vis)
        for frame in frames:
            for name in KEYPOINTS:
                frame.points[name] = (
                    float(arrays[name][frame.index, 0]),
                    float(arrays[name][frame.index, 1]),
                )
                frame.visibility[name] = float(raw_vis[name][frame.index])

        return PoseSequence(fps=fps, width=width, height=height, frames=frames, arrays=arrays)

    def _smooth_arrays(self, raw_xy: dict[str, list[tuple[float, float]]], raw_vis: dict[str, list[float]]) -> dict[str, np.ndarray]:
        output: dict[str, np.ndarray] = {}
        alpha = self.config.smoothing_alpha
        for name, samples in raw_xy.items():
            arr = np.array(samples, dtype=float)
            vis = np.array(raw_vis[name], dtype=float)
            for axis in range(2):
                column = arr[:, axis]
                mask = np.isnan(column) | (vis < self.config.visibility_threshold)
                if np.all(mask):
                    column[:] = 0.0
                else:
                    idx = np.arange(len(column))
                    column[mask] = np.interp(idx[mask], idx[~mask], column[~mask])
                for i in range(1, len(column)):
                    column[i] = alpha * column[i] + (1 - alpha) * column[i - 1]
                arr[:, axis] = column
            output[name] = arr
        return output
