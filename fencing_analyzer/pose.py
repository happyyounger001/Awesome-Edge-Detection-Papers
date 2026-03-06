from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import cv2
import numpy as np

from .utils import ema_smooth

LANDMARKS = {
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_wrist": 15,
    "right_wrist": 16,
}


@dataclass
class PoseSequence:
    fps: float
    width: int
    height: int
    frames: int
    x: Dict[str, np.ndarray]
    y: Dict[str, np.ndarray]
    vis: Dict[str, np.ndarray]



def extract_pose_sequence(video_path: str | Path, ema_alpha: float = 0.35) -> PoseSequence:
    try:
        import mediapipe as mp
    except ImportError as exc:
        raise RuntimeError("mediapipe is required for pose extraction") from exc

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    per_key_x: Dict[str, List[float]] = {k: [] for k in LANDMARKS}
    per_key_y: Dict[str, List[float]] = {k: [] for k in LANDMARKS}
    per_key_vis: Dict[str, List[float]] = {k: [] for k in LANDMARKS}

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = pose.process(rgb)
        if result.pose_landmarks is None:
            for name in LANDMARKS:
                per_key_x[name].append(np.nan)
                per_key_y[name].append(np.nan)
                per_key_vis[name].append(0.0)
            continue
        lm = result.pose_landmarks.landmark
        for name, idx in LANDMARKS.items():
            per_key_x[name].append(lm[idx].x)
            per_key_y[name].append(lm[idx].y)
            per_key_vis[name].append(lm[idx].visibility)

    cap.release()
    pose.close()

    x = {k: np.array(v, dtype=float) for k, v in per_key_x.items()}
    y = {k: np.array(v, dtype=float) for k, v in per_key_y.items()}
    vis = {k: np.array(v, dtype=float) for k, v in per_key_vis.items()}

    for name in LANDMARKS:
        for arr in (x[name], y[name]):
            mask = np.isnan(arr)
            if np.all(mask):
                arr[:] = 0.0
            else:
                idx = np.arange(len(arr))
                arr[mask] = np.interp(idx[mask], idx[~mask], arr[~mask])
            arr[:] = ema_smooth(arr, ema_alpha)

    frames = len(next(iter(x.values()))) if x else 0
    return PoseSequence(fps=fps, width=width, height=height, frames=frames, x=x, y=y, vis=vis)
