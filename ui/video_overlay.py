from __future__ import annotations

from pathlib import Path

import cv2

from config import AppConfig
from pose.estimator import KEYPOINTS, SKELETON
from pose.models import PoseSequence
from rules.evaluator import SequenceEvaluation


def render_overlay_video(
    video_path: Path,
    output_dir: Path,
    pose_sequence: PoseSequence,
    evaluation: SequenceEvaluation,
    config: AppConfig,
) -> tuple[Path, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay.mp4"

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        pose_sequence.fps,
        (pose_sequence.width, pose_sequence.height),
    )

    keyframe_indices = {
        "出手瞬间": evaluation.hand_frame,
        "弓步完成瞬间": evaluation.finish_frame,
        "预警触发瞬间": evaluation.warning_frame,
    }
    saved_keyframes: dict[str, Path] = {}
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame_index >= pose_sequence.frame_count:
            break
        assessment = evaluation.frame_assessments[min(frame_index, len(evaluation.frame_assessments) - 1)]
        annotated = draw_pose_overlay(frame, pose_sequence, assessment.frame_index, assessment)
        writer.write(annotated)

        for title, idx in keyframe_indices.items():
            if idx is not None and frame_index == idx and title not in saved_keyframes:
                keyframe_path = frames_dir / f"{title}.png"
                cv2.imwrite(str(keyframe_path), annotated)
                saved_keyframes[title] = keyframe_path
        frame_index += 1

    cap.release()
    writer.release()
    return overlay_path, saved_keyframes


def draw_pose_overlay(frame, pose_sequence: PoseSequence, frame_index: int, assessment) -> any:
    canvas = frame.copy()
    points: dict[str, tuple[int, int]] = {}
    for name in KEYPOINTS:
        x = int(pose_sequence.arrays[name][frame_index, 0] * pose_sequence.width)
        y = int(pose_sequence.arrays[name][frame_index, 1] * pose_sequence.height)
        points[name] = (x, y)
        cv2.circle(canvas, (x, y), 4, (0, 255, 0), -1)

    for start, end in SKELETON:
        cv2.line(canvas, points[start], points[end], (255, 215, 0), 2)

    cv2.putText(canvas, f"Time: {assessment.timestamp:.2f}s", (20, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(canvas, f"AI: {assessment.current_text} {assessment.current_icon}", (20, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 255), 2)
    cv2.putText(canvas, f"手脚顺序: {assessment.hand_foot_order}", (20, 96), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"弓步稳定: {assessment.stability}", (20, 124), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"抬大腿: {assessment.thigh_raise}", (20, 152), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(canvas, f"踢小腿: {assessment.calf_kick}", (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return canvas
