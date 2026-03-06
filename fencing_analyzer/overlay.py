from __future__ import annotations

from pathlib import Path
from typing import List

import cv2

from .lunge_detect import LungeEvent
from .metrics import LungeMetrics
from .pose import LANDMARKS, PoseSequence

CONNECTIONS = [
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


def render_overlay(
    video_path: str | Path,
    out_path: str | Path,
    pose: PoseSequence,
    events: List[LungeEvent],
    metrics: List[LungeMetrics],
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, pose.fps, (pose.width, pose.height))

    event_map = {(e.start_frame, e.end_frame): e.lunge_id for e in events}
    metric_by_id = {m.lunge_id: m for m in metrics}

    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok or frame_idx >= pose.frames:
            break

        points = {}
        for name in LANDMARKS:
            x = int(pose.x[name][frame_idx] * pose.width)
            y = int(pose.y[name][frame_idx] * pose.height)
            points[name] = (x, y)
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)

        for a, b in CONNECTIONS:
            cv2.line(frame, points[a], points[b], (255, 200, 0), 2)

        active_id = None
        for (s, e), lid in event_map.items():
            if s <= frame_idx <= e:
                active_id = lid
                break

        cv2.putText(frame, f"Lunge count: {len(events)}", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        if active_id:
            m = metric_by_id[active_id]
            txt = f"Active #{active_id} knee_min={m.knee_angle_min:.1f} trunk_max={m.trunk_lean_max:.1f}"
            cv2.putText(frame, txt, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()
