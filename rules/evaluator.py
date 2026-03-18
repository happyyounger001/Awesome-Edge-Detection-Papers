from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from analysis.models import FrameAssessment, LungeQuality, TimingResult
from config import AppConfig
from pose.models import PoseSequence


@dataclass
class SequenceEvaluation:
    metrics: pd.DataFrame
    timing: TimingResult
    quality: LungeQuality
    frame_assessments: list[FrameAssessment]
    detected_go: bool
    peak_frame: int
    warning_frame: int
    hand_frame: int | None
    finish_frame: int | None


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return 0.0
    value = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(value)))


def _lead_side(sequence: PoseSequence) -> str:
    left_mean = float(np.mean(sequence.arrays["left_ankle"][:, 0]))
    right_mean = float(np.mean(sequence.arrays["right_ankle"][:, 0]))
    return "left" if left_mean >= right_mean else "right"


def _first_crossing(values: np.ndarray, threshold: float) -> int | None:
    indices = np.where(values >= threshold)[0]
    return int(indices[0]) if len(indices) else None


def _bool_label(ok: bool, good_text: str, warn_text: str) -> tuple[str, str]:
    return (good_text, "👍") if ok else (warn_text, "🚨")


def _describe(quality: LungeQuality) -> list[str]:
    return [
        f"手脚顺序：{quality.hand_foot_order} {quality.hand_foot_status}",
        f"弓步稳定：{quality.stability} {quality.stability_status}",
        f"抬大腿：{quality.thigh_raise} {quality.thigh_raise_status}",
        f"踢小腿：{quality.calf_kick} {quality.calf_kick_status}",
    ]


def evaluate_sequence(sequence: PoseSequence, config: AppConfig, manual_go_time: float | None) -> SequenceEvaluation:
    side = _lead_side(sequence)
    back = "right" if side == "left" else "left"
    fps = sequence.fps
    dt = 1.0 / fps

    lead_ankle_x = sequence.arrays[f"{side}_ankle"][:, 0]
    lead_wrist_x = sequence.arrays[f"{side}_wrist"][:, 0]
    lead_knee = sequence.arrays[f"{side}_knee"]
    lead_hip = sequence.arrays[f"{side}_hip"]
    lead_ankle = sequence.arrays[f"{side}_ankle"]
    lead_shoulder = sequence.arrays[f"{side}_shoulder"]
    back_ankle = sequence.arrays[f"{back}_ankle"]

    hand_speed = np.abs(np.gradient(lead_wrist_x, dt))
    foot_speed = np.abs(np.gradient(lead_ankle_x, dt))
    body_speed = np.abs(np.gradient((lead_hip[:, 0] + sequence.arrays[f"{back}_hip"][:, 0]) / 2.0, dt))
    calf_speed = np.abs(np.gradient(lead_ankle[:, 0] - lead_knee[:, 0], dt))

    knee_angles = np.array([angle_deg(lead_hip[i], lead_knee[i], lead_ankle[i]) for i in range(sequence.frame_count)])
    thigh_raise_angles = np.array([angle_deg(lead_shoulder[i], lead_hip[i], lead_knee[i]) for i in range(sequence.frame_count)])
    stride_distance = np.abs(lead_ankle[:, 0] - back_ankle[:, 0])

    peak_frame = int(np.argmax(stride_distance)) if len(stride_distance) else 0
    hand_frame = _first_crossing(hand_speed, config.hand_speed_threshold)
    foot_frame = _first_crossing(foot_speed, config.foot_speed_threshold)

    go_time = manual_go_time
    detected_go = manual_go_time is not None
    if go_time is None:
        auto_frame = min([idx for idx in [hand_frame, foot_frame] if idx is not None], default=None)
        if auto_frame is not None:
            go_time = max(0.0, (auto_frame / fps) - 0.15)

    post_peak = body_speed[peak_frame:]
    stable_window = max(1, int(config.stability_seconds_threshold * fps))
    finish_frame = None
    for start in range(len(post_peak)):
        window = post_peak[start : start + stable_window]
        if len(window) < stable_window:
            break
        if float(np.max(window)) <= config.lunge_finish_speed_threshold:
            finish_frame = peak_frame + start
            break
    if finish_frame is None:
        finish_frame = peak_frame

    stability_duration = max(0.0, (finish_frame - peak_frame) / fps)
    hand_time = None if hand_frame is None else hand_frame / fps
    foot_time = None if foot_frame is None else foot_frame / fps
    finish_time = None if finish_frame is None else finish_frame / fps

    order_ok = hand_frame is not None and foot_frame is not None and hand_frame <= foot_frame
    hand_foot_order, hand_foot_status = _bool_label(order_ok, "手先脚后", "脚先手后")
    stability_ok = stability_duration <= config.stability_seconds_threshold
    stability, stability_status = _bool_label(stability_ok, "稳定", "晃动超时")
    thigh_ok = float(np.nanmax(thigh_raise_angles)) < config.thigh_raise_angle_threshold
    thigh_raise, thigh_raise_status = _bool_label(thigh_ok, "未抬大腿", "抬大腿")
    calf_ok = float(np.nanmax(calf_speed)) >= config.calf_kick_speed_threshold
    calf_kick, calf_kick_status = _bool_label(calf_ok, "有踢小腿", "无明显踢小腿")

    good_count = sum(status == "👍" for status in [hand_foot_status, stability_status, calf_kick_status]) + int(thigh_raise_status == "👍")
    overall_status = "👍" if good_count >= 3 else "🚨"

    quality = LungeQuality(
        hand_foot_order=hand_foot_order,
        hand_foot_status=hand_foot_status,
        stability=stability,
        stability_status=stability_status,
        thigh_raise=thigh_raise,
        thigh_raise_status=thigh_raise_status,
        calf_kick=calf_kick,
        calf_kick_status=calf_kick_status,
        overall_status=overall_status,
        explanations=[],
    )
    quality.explanations = _describe(quality)

    timing = TimingResult(
        go_time=go_time,
        hand_start_time=hand_time,
        foot_start_time=foot_time,
        lunge_finish_time=finish_time,
    )

    df = pd.DataFrame(
        {
            "time": np.arange(sequence.frame_count) / fps,
            "hand_speed": hand_speed,
            "foot_speed": foot_speed,
            "body_speed": body_speed,
            "calf_speed": calf_speed,
            "knee_angle": knee_angles,
            "thigh_raise_angle": thigh_raise_angles,
            "stride_distance": stride_distance,
        }
    )

    warning_frame = peak_frame if overall_status == "🚨" else finish_frame
    assessments = _build_frame_assessments(df["time"], hand_foot_order, stability, thigh_raise, calf_kick, hand_frame, foot_frame, finish_frame)

    return SequenceEvaluation(
        metrics=df,
        timing=timing,
        quality=quality,
        frame_assessments=assessments,
        detected_go=detected_go,
        peak_frame=peak_frame,
        warning_frame=warning_frame,
        hand_frame=hand_frame,
        finish_frame=finish_frame,
    )


def _build_frame_assessments(
    timestamps: Iterable[float],
    hand_foot_order: str,
    stability: str,
    thigh_raise: str,
    calf_kick: str,
    hand_frame: int | None,
    foot_frame: int | None,
    finish_frame: int | None,
) -> list[FrameAssessment]:
    assessments: list[FrameAssessment] = []
    for frame_index, timestamp in enumerate(timestamps):
        if hand_frame is not None and frame_index <= hand_frame:
            current_text = "等待启动"
            current_icon = "⏳"
        elif finish_frame is not None and frame_index >= finish_frame:
            current_text = f"动作完成：{stability}"
            current_icon = "👍" if stability == "稳定" else "🚨"
        elif foot_frame is not None and hand_frame is not None and frame_index >= min(hand_frame, foot_frame):
            current_text = f"启动阶段：{hand_foot_order}"
            current_icon = "👍" if hand_foot_order == "手先脚后" else "🚨"
        else:
            current_text = "准备阶段"
            current_icon = "⏳"

        assessments.append(
            FrameAssessment(
                frame_index=frame_index,
                timestamp=float(timestamp),
                current_text=current_text,
                current_icon=current_icon,
                hand_foot_order=hand_foot_order,
                stability=stability,
                thigh_raise=thigh_raise,
                calf_kick=calf_kick,
            )
        )
    return assessments
