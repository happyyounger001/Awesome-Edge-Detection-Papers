from __future__ import annotations

from typing import Any

import numpy as np

from config import AppConfig
from pose.models import PoseSequence
from rules.evaluator import angle_deg


def _lead_side(sequence: PoseSequence) -> str:
    return "left" if float(np.mean(sequence.arrays["left_ankle"][:, 0])) >= float(np.mean(sequence.arrays["right_ankle"][:, 0])) else "right"


def _line_angle_deg(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    delta = p2 - p1
    return np.abs(np.degrees(np.arctan2(delta[:, 1], delta[:, 0])))


def _stability_std(points: np.ndarray, start: int, end: int) -> float:
    window = points[start:end]
    if len(window) <= 1:
        return 0.0
    center = np.mean(window, axis=0)
    return float(np.std(np.linalg.norm(window - center, axis=1)))


def extract_video_features(sequence: PoseSequence, config: AppConfig, video_path: str, category: str) -> dict[str, Any]:
    side = _lead_side(sequence)
    back = "right" if side == "left" else "left"
    fps = sequence.fps
    dt = 1.0 / fps

    front_hip = sequence.arrays[f"{side}_hip"]
    front_knee = sequence.arrays[f"{side}_knee"]
    front_ankle = sequence.arrays[f"{side}_ankle"]
    rear_hip = sequence.arrays[f"{back}_hip"]
    rear_knee = sequence.arrays[f"{back}_knee"]
    rear_ankle = sequence.arrays[f"{back}_ankle"]
    shoulder = sequence.arrays[f"{side}_shoulder"]
    elbow = sequence.arrays[f"{side}_elbow"]
    wrist = sequence.arrays[f"{side}_wrist"]

    stride = np.abs(front_ankle[:, 0] - rear_ankle[:, 0])
    max_stride_idx = int(np.argmax(stride)) if len(stride) else 0
    hand_speed = np.abs(np.gradient(wrist[:, 0], dt))
    foot_speed = np.abs(np.gradient(front_ankle[:, 0], dt))
    hand_start_idx = int(np.argmax(hand_speed >= config.hand_speed_threshold)) if np.any(hand_speed >= config.hand_speed_threshold) else 0
    foot_start_idx = int(np.argmax(foot_speed >= config.foot_speed_threshold)) if np.any(foot_speed >= config.foot_speed_threshold) else 0

    front_knee_angles = np.array([angle_deg(front_hip[i], front_knee[i], front_ankle[i]) for i in range(sequence.frame_count)])
    rear_knee_angles = np.array([angle_deg(rear_hip[i], rear_knee[i], rear_ankle[i]) for i in range(sequence.frame_count)])
    trunk_lean = np.abs(_line_angle_deg(shoulder, front_hip) - 90.0)
    head_tilt = _line_angle_deg(sequence.arrays["left_eye"], sequence.arrays["right_eye"])
    elbow_extension = np.array([angle_deg(shoulder[i], elbow[i], wrist[i]) for i in range(sequence.frame_count)])
    hip_knee_ground_angle = _line_angle_deg(front_hip, front_knee)

    torso_length = np.maximum(np.linalg.norm(shoulder - front_hip, axis=1), 1e-6)
    max_stride_norm = float(np.max(stride / torso_length))
    wrist_forward = np.abs(wrist[:, 0] - shoulder[:, 0]) / torso_length
    time_to_max_stride = max_stride_idx / fps
    time_to_max_wrist_extension = int(np.argmax(wrist_forward)) / fps

    hold_end = min(sequence.frame_count, max_stride_idx + max(1, int(config.stability_seconds_threshold * fps)))
    wrist_std = _stability_std(wrist, max_stride_idx, hold_end)
    elbow_std = _stability_std(elbow, max_stride_idx, hold_end)
    shoulder_std = _stability_std(shoulder, max_stride_idx, hold_end)
    head_center = (sequence.arrays["left_eye"] + sequence.arrays["right_eye"]) / 2.0
    head_std = _stability_std(head_center, max_stride_idx, hold_end)
    trunk_std = float(np.std(trunk_lean[max_stride_idx:hold_end])) if hold_end > max_stride_idx else 0.0
    stable_hold_duration = max(0.0, (hold_end - max_stride_idx) / fps)

    hand_foot_delta = abs(hand_start_idx - foot_start_idx) / fps
    coordination_raw = max(0.0, 1.0 - hand_foot_delta)
    head_trunk_alignment = max(0.0, 1.0 - float(np.max(head_tilt)) / 45.0)

    return {
        "video_path": video_path,
        "category": category,
        "front_knee_angle_min": float(np.min(front_knee_angles)),
        "rear_knee_angle_min": float(np.min(rear_knee_angles)),
        "trunk_lean_max_deg": float(np.max(trunk_lean)),
        "head_tilt_max_deg": float(np.max(head_tilt)),
        "stride_max_norm": max_stride_norm,
        "wrist_forward_dist_max_norm": float(np.max(wrist_forward)),
        "elbow_extension_angle_min": float(np.min(elbow_extension)),
        "hip_knee_line_ground_angle_max": float(np.max(hip_knee_ground_angle)),
        "hand_start_time": hand_start_idx / fps,
        "foot_start_time": foot_start_idx / fps,
        "hand_foot_start_delta": hand_foot_delta,
        "lunge_out_duration": max(0.0, (max_stride_idx - min(hand_start_idx, foot_start_idx)) / fps),
        "lunge_return_duration": max(0.0, (sequence.frame_count - 1 - max_stride_idx) / fps),
        "lunge_total_duration": sequence.frame_count / fps,
        "time_to_max_stride": time_to_max_stride,
        "time_to_max_wrist_extension": time_to_max_wrist_extension,
        "wrist_stability_std": wrist_std,
        "elbow_stability_std": elbow_std,
        "shoulder_stability_std": shoulder_std,
        "head_stability_std": head_std,
        "trunk_stability_std": trunk_std,
        "stable_hold_duration": stable_hold_duration,
        "hand_first_ratio": 1.0 if hand_start_idx <= foot_start_idx else 0.0,
        "foot_first_ratio": 1.0 if foot_start_idx < hand_start_idx else 0.0,
        "coordination_score_raw": coordination_raw,
        "head_trunk_alignment_score": head_trunk_alignment,
    }
