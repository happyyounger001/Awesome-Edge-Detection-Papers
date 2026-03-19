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
    completed_lunges: int


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


def _hip_knee_line_angle(hip: np.ndarray, knee: np.ndarray) -> float:
    vector = knee - hip
    horizontal = np.array([1.0, 0.0])
    denom = np.linalg.norm(vector) * np.linalg.norm(horizontal)
    if denom == 0:
        return 0.0
    cosine = np.clip(np.dot(vector, horizontal) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _detect_lunge_numbers(stride_distance: np.ndarray) -> tuple[np.ndarray, int]:
    if len(stride_distance) == 0:
        return np.zeros(0, dtype=int), 0

    baseline = float(np.median(stride_distance[: max(3, min(len(stride_distance), 10))]))
    peak = float(np.max(stride_distance))
    activation_thr = baseline + max(0.03, (peak - baseline) * 0.25)
    return_thr = baseline + max(0.015, (peak - baseline) * 0.08)

    numbers = np.zeros(len(stride_distance), dtype=int)
    active = False
    current_number = 1
    completed = 0
    start_index = 0

    for i, stride in enumerate(stride_distance):
        if not active and stride >= activation_thr:
            active = True
            start_index = i
            current_number = completed + 1

        if active:
            numbers[i] = current_number
            if i > start_index and stride <= return_thr:
                completed += 1
                active = False
        else:
            numbers[i] = max(completed, 1 if peak > activation_thr else 0)

    if active:
        numbers[start_index:] = current_number

    return numbers, completed


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
    back_hip = sequence.arrays[f"{back}_hip"]

    hand_speed = np.abs(np.gradient(lead_wrist_x, dt))
    foot_speed = np.abs(np.gradient(lead_ankle_x, dt))
    body_speed = np.abs(np.gradient((lead_hip[:, 0] + back_hip[:, 0]) / 2.0, dt))
    calf_speed = np.abs(np.gradient(lead_ankle[:, 0] - lead_knee[:, 0], dt))

    knee_angles = np.array([angle_deg(lead_hip[i], lead_knee[i], lead_ankle[i]) for i in range(sequence.frame_count)])
    thigh_raise_angles = np.array([_hip_knee_line_angle(lead_hip[i], lead_knee[i]) for i in range(sequence.frame_count)])
    knee_below_hip = lead_knee[:, 1] > lead_hip[:, 1]
    stride_distance = np.abs(lead_ankle[:, 0] - back_ankle[:, 0])
    lunge_numbers, completed_lunges = _detect_lunge_numbers(stride_distance)

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

    # Keep the existing product logic: hand starts before foot is considered the correct order.
    order_ok = hand_frame is not None and foot_frame is not None and hand_frame <= foot_frame
    hand_foot_order, hand_foot_status = _bool_label(order_ok, "先手后脚", "其他")
    stability_ok = stability_duration <= config.stability_seconds_threshold
    stability, stability_status = _bool_label(stability_ok, "达标", "未达标")

    thigh_angle_ok = thigh_raise_angles >= config.thigh_raise_angle_threshold
    if config.require_knee_below_hip_for_thigh_raise:
        thigh_raise_flags = knee_below_hip & thigh_angle_ok
    else:
        thigh_raise_flags = thigh_angle_ok
    thigh_abnormal = bool(np.any(thigh_raise_flags))
    thigh_raise = "是" if thigh_abnormal else "否"
    thigh_raise_status = "🚨" if thigh_abnormal else "👍"

    calf_ok = float(np.nanmax(calf_speed)) >= config.calf_kick_speed_threshold
    calf_kick, calf_kick_status = _bool_label(calf_ok, "有", "无")

    overall_good = (not thigh_abnormal) and stability_ok and order_ok
    overall_status = "👍" if overall_good else "🚨"

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
            "knee_below_hip": knee_below_hip.astype(int),
            "stride_distance": stride_distance,
            "lunge_number": lunge_numbers,
            "thigh_raise_flag": thigh_raise_flags.astype(int),
        }
    )

    warning_frame = peak_frame if overall_status == "🚨" else finish_frame
    assessments = _build_frame_assessments(
        df["time"],
        df["lunge_number"],
        order_ok,
        stability_ok,
        thigh_raise_flags,
        hand_frame,
        foot_frame,
        finish_frame,
    )

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
        completed_lunges=completed_lunges,
    )


def _build_frame_assessments(
    timestamps: Iterable[float],
    lunge_numbers: Iterable[int],
    order_ok: bool,
    stability_ok: bool,
    thigh_raise_flags: Iterable[bool],
    hand_frame: int | None,
    foot_frame: int | None,
    finish_frame: int | None,
) -> list[FrameAssessment]:
    assessments: list[FrameAssessment] = []
    sequence_order = "先手后脚" if order_ok else "其他"
    sequence_stability = "达标" if stability_ok else "未达标"

    for frame_index, (timestamp, raw_lunge_number, thigh_flag) in enumerate(zip(timestamps, lunge_numbers, thigh_raise_flags)):
        lunge_index = int(raw_lunge_number) if int(raw_lunge_number) > 0 else 1
        frame_order_ok = not (foot_frame is not None and hand_frame is not None and frame_index >= min(hand_frame, foot_frame) and not order_ok)
        score_good = (not bool(thigh_flag)) and stability_ok and frame_order_ok
        current_text = "很棒，得分！" if score_good else "加油，还能更好！"
        current_icon = "👍" if score_good else "🚨"

        assessments.append(
            FrameAssessment(
                frame_index=frame_index,
                timestamp=float(timestamp),
                lunge_index=lunge_index,
                current_text=current_text,
                current_icon=current_icon,
                score_text="得分" if score_good else "未得分",
                hand_foot_order=sequence_order,
                stability=sequence_stability,
                thigh_raise="是" if bool(thigh_flag) else "否",
                calf_kick="有" if finish_frame is not None and frame_index >= finish_frame else "检测中",
                show_thigh_warning=bool(thigh_flag),
                show_order_warning=not order_ok and foot_frame is not None and hand_frame is not None and frame_index >= min(hand_frame, foot_frame),
            )
        )
    return assessments
