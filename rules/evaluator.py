from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from analysis.models import FrameAssessment, LungeQuality, TimingResult
from config import AppConfig
from pose.models import PoseSequence

logger = logging.getLogger(__name__)


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
        f"头部姿态：{quality.head_tilt} {quality.head_tilt_status}",
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


def _line_angle_degrees(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    delta = p2 - p1
    return np.degrees(np.arctan2(delta[:, 1], delta[:, 0]))


def _compute_lunge_state_machine(
    stride_distance: np.ndarray,
    leg_extension_signed: np.ndarray,
    arm_extension_signed: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, list[str], int]:
    if len(stride_distance) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int), [], 0

    velocity = np.gradient(stride_distance)
    baseline = float(np.median(stride_distance[: max(3, min(len(stride_distance), 10))]))
    peak = float(np.max(stride_distance))
    amplitude = max(peak - baseline, 1e-6)
    prepare_thr = baseline + max(0.01, amplitude * float(getattr(sequence_config, "lunge_prepare_ratio", 0.12)))
    start_thr = baseline + max(0.03, amplitude * float(getattr(sequence_config, "lunge_start_ratio", 0.28)))
    reach_thr = baseline + max(0.05, amplitude * float(getattr(sequence_config, "lunge_reach_ratio", 0.72)))
    return_thr = baseline + max(0.015, amplitude * float(getattr(sequence_config, "lunge_return_ratio", 0.10)))
    rearm_thr = baseline + max(0.012, amplitude * float(getattr(sequence_config, "lunge_rearm_ratio", 0.07)))
    min_gap_frames = max(1, int(round(fps * float(getattr(sequence_config, "lunge_min_gap_sec", 0.25)))))
    hold_frames_required = max(1, int(round(fps * float(getattr(sequence_config, "lunge_hold_sec", 0.12)))))
    leg_out_thr = max(0.012, float(np.percentile(np.abs(leg_extension_signed), 75)) * 0.6)
    arm_extend_thr = max(0.010, float(np.percentile(np.abs(arm_extension_signed), 75)) * 0.6)
    ready_leg_thr = leg_out_thr * 0.35
    ready_arm_thr = arm_extend_thr * 0.35

    current_numbers = np.zeros(len(stride_distance), dtype=int)
    completed_numbers = np.zeros(len(stride_distance), dtype=int)
    states: list[str] = []
    state = "READY"
    completed = 0
    current = 1 if peak > start_thr else 0
    hold_frames = 0
    lunge_start_frame: int | None = None
    last_completion_frame = -min_gap_frames
    leg_confirm_frames = 0
    arm_confirm_frames = 0
    reached_frame: int | None = None
    recovering_frames = 0

    for i, (stride, speed, leg_delta, arm_delta) in enumerate(zip(stride_distance, velocity, leg_extension_signed, arm_extension_signed)):
        previous_state = state
        leg_out = leg_delta >= leg_out_thr
        arm_out = arm_delta >= arm_extend_thr
        backstep_motion = leg_delta < -ready_leg_thr
        if state == "READY":
            current = max(completed + 1, 1) if peak > start_thr else max(completed, 1)
            leg_confirm_frames = 0
            arm_confirm_frames = 0
            if i - last_completion_frame < min_gap_frames:
                state = "READY"
            elif stride >= prepare_thr and speed >= 0 and leg_out and not backstep_motion:
                state = "STARTED"
                lunge_start_frame = i
        elif state == "STARTED":
            current = completed + 1
            leg_confirm_frames = leg_confirm_frames + 1 if leg_out else 0
            arm_confirm_frames = arm_confirm_frames + 1 if arm_out else 0
            if backstep_motion:
                state = "ABORTED"
                lunge_start_frame = None
            elif stride >= start_thr and speed >= 0 and leg_confirm_frames >= 2:
                state = "EXTENDING"
            elif stride < prepare_thr * 0.9:
                state = "READY"
                lunge_start_frame = None
            elif lunge_start_frame is not None and i - lunge_start_frame > int(fps * float(getattr(sequence_config, "lunge_started_timeout_sec", 0.9))):
                state = "ABORTED"
        elif state == "EXTENDING":
            current = completed + 1
            if backstep_motion:
                state = "ABORTED"
                lunge_start_frame = None
            elif stride >= reach_thr and leg_out and arm_out:
                hold_frames += 1
                if hold_frames >= hold_frames_required:
                    state = "HOLD"
                    reached_frame = i
            elif lunge_start_frame is not None and i - lunge_start_frame > int(fps * float(getattr(sequence_config, "lunge_extending_timeout_sec", 1.2))):
                state = "ABORTED"
        elif state == "HOLD":
            current = completed + 1
            if speed < 0 or stride < reach_thr * 0.95:
                state = "RETURN"
                recovering_frames = 0
            elif lunge_start_frame is not None and i - lunge_start_frame > int(fps * float(getattr(sequence_config, "lunge_hold_timeout_sec", 1.6))):
                state = "ABORTED"
        elif state == "RETURN":
            current = completed + 1
            recovering_frames += 1
            back_to_ready = abs(leg_delta) <= ready_leg_thr and abs(arm_delta) <= ready_arm_thr
            stable_recover = recovering_frames >= 2 and back_to_ready and stride <= return_thr
            if stable_recover and reached_frame is not None and i - last_completion_frame >= min_gap_frames:
                completed += 1
                last_completion_frame = i
                state = "COMPLETED_LOCK"
                lunge_start_frame = None
                hold_frames = 0
                recovering_frames = 0
                logger.debug("Lunge completed at frame %s -> completed=%s", i, completed)
            elif lunge_start_frame is not None and i - lunge_start_frame > int(fps * float(getattr(sequence_config, "lunge_recover_timeout_sec", 2.2))):
                state = "ABORTED"
        elif state == "COMPLETED_LOCK":
            current = max(completed, 1)
            if stride <= rearm_thr and abs(speed) <= np.percentile(np.abs(velocity), 60):
                state = "READY"
                reached_frame = None
        elif state == "ABORTED":
            current = max(completed + 1, 1)
            if stride <= rearm_thr and abs(speed) <= np.percentile(np.abs(velocity), 60):
                state = "READY"
                lunge_start_frame = None
                reached_frame = None

        if previous_state != state:
            logger.debug("Lunge state transition at frame %s: %s -> %s", i, previous_state, state)

        current_numbers[i] = current
        completed_numbers[i] = completed
        states.append(state)

    return current_numbers, completed_numbers, states, completed


def _stability_metrics(sequence: PoseSequence, side: str, start_frame: int, window_frames: int) -> tuple[dict[str, float], bool]:
    end_frame = min(sequence.frame_count, start_frame + window_frames)
    if end_frame <= start_frame:
        return {"wrist": 0.0, "elbow": 0.0, "shoulder": 0.0}, True

    checks = {
        "wrist": sequence.arrays[f"{side}_wrist"][start_frame:end_frame],
        "elbow": sequence.arrays[f"{side}_elbow"][start_frame:end_frame],
        "shoulder": sequence.arrays[f"{side}_shoulder"][start_frame:end_frame],
    }
    jitter = {}
    for name, arr in checks.items():
        base = arr[0]
        displacement = np.linalg.norm(arr - base, axis=1)
        jitter[name] = float(np.max(displacement))
    stable = all(value <= sequence_config.upper_body_motion_threshold for value in jitter.values())
    return jitter, stable


def evaluate_sequence(sequence: PoseSequence, config: AppConfig, manual_go_time: float | None) -> SequenceEvaluation:
    global sequence_config
    sequence_config = config
    side = _lead_side(sequence)
    back = "right" if side == "left" else "left"
    fps = sequence.fps
    dt = 1.0 / fps

    lead_ankle_x = sequence.arrays[f"{side}_ankle"][:, 0]
    lead_wrist_x = sequence.arrays[f"{side}_wrist"][:, 0]
    lead_knee = sequence.arrays[f"{side}_knee"]
    lead_hip = sequence.arrays[f"{side}_hip"]
    lead_ankle = sequence.arrays[f"{side}_ankle"]
    back_ankle = sequence.arrays[f"{back}_ankle"]
    back_hip = sequence.arrays[f"{back}_hip"]
    shoulder = sequence.arrays[f"{side}_shoulder"]
    elbow = sequence.arrays[f"{side}_elbow"]
    wrist = sequence.arrays[f"{side}_wrist"]

    hand_speed = np.abs(np.gradient(lead_wrist_x, dt))
    foot_speed = np.abs(np.gradient(lead_ankle_x, dt))
    body_speed = np.abs(np.gradient((lead_hip[:, 0] + back_hip[:, 0]) / 2.0, dt))
    calf_speed = np.abs(np.gradient(lead_ankle[:, 0] - lead_knee[:, 0], dt))

    knee_angles = np.array([angle_deg(lead_hip[i], lead_knee[i], lead_ankle[i]) for i in range(sequence.frame_count)])
    thigh_raise_angles = np.array([_hip_knee_line_angle(lead_hip[i], lead_knee[i]) for i in range(sequence.frame_count)])
    knee_below_hip = lead_knee[:, 1] > lead_hip[:, 1]
    stride_distance = np.abs(lead_ankle[:, 0] - back_ankle[:, 0])
    forward_sign = 1.0 if float(np.mean(lead_ankle_x[: max(3, int(0.2 * fps))])) >= float(np.mean(back_ankle[:, 0][: max(3, int(0.2 * fps))])) else -1.0
    lead_ankle_baseline = float(np.median(lead_ankle_x[: max(3, int(0.2 * fps))]))
    lead_wrist_baseline = float(np.median(lead_wrist_x[: max(3, int(0.2 * fps))]))
    leg_extension_signed = forward_sign * (lead_ankle_x - lead_ankle_baseline)
    arm_extension_signed = forward_sign * (lead_wrist_x - lead_wrist_baseline)
    current_lunge_numbers, completed_numbers, lunge_states, completed_lunges = _compute_lunge_state_machine(
        stride_distance,
        leg_extension_signed,
        arm_extension_signed,
        fps,
    )
    torso_length = np.maximum(np.linalg.norm(shoulder - lead_hip, axis=1), 1e-6)

    eye_angle = np.abs(_line_angle_degrees(sequence.arrays["left_eye"], sequence.arrays["right_eye"]))
    head_tilt_flags = eye_angle > config.head_tilt_angle_threshold

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
    upper_body_jitter, upper_body_stable = _stability_metrics(sequence, side, peak_frame, stable_window)
    hand_time = None if hand_frame is None else hand_frame / fps
    foot_time = None if foot_frame is None else foot_frame / fps
    finish_time = None if finish_frame is None else finish_frame / fps

    # Keep the existing product logic: hand starts before foot is considered the correct order.
    order_ok = hand_frame is not None and foot_frame is not None and hand_frame <= foot_frame
    hand_foot_order, hand_foot_status = _bool_label(order_ok, "先手后脚", "其他")
    stability_ok = stability_duration <= config.stability_seconds_threshold and upper_body_stable
    stability, stability_status = _bool_label(stability_ok, "达标", "未达标")

    thigh_angle_ok = thigh_raise_angles >= config.thigh_raise_angle_threshold
    if config.require_knee_below_hip_for_thigh_raise:
        thigh_raise_flags = knee_below_hip & thigh_angle_ok
    else:
        thigh_raise_flags = thigh_angle_ok
    thigh_abnormal = bool(np.any(thigh_raise_flags))
    thigh_raise = "是" if thigh_abnormal else "否"
    thigh_raise_status = "🚨" if thigh_abnormal else "👍"

    head_tilt_abnormal = bool(np.any(head_tilt_flags))
    head_tilt, head_tilt_status = _bool_label(not head_tilt_abnormal, "正常", "歪头提醒")

    calf_ok = float(np.nanmax(calf_speed)) >= config.calf_kick_speed_threshold
    calf_kick, calf_kick_status = _bool_label(calf_ok, "有", "无")

    overall_good = (not thigh_abnormal) and stability_ok and order_ok and (not head_tilt_abnormal)
    overall_status = "👍" if overall_good else "🚨"

    quality = LungeQuality(
        hand_foot_order=hand_foot_order,
        hand_foot_status=hand_foot_status,
        stability=stability,
        stability_status=stability_status,
        thigh_raise=thigh_raise,
        thigh_raise_status=thigh_raise_status,
        head_tilt=head_tilt,
        head_tilt_status=head_tilt_status,
        calf_kick=calf_kick,
        calf_kick_status=calf_kick_status,
        overall_status=overall_status,
        status_text="很棒，得分！" if overall_good else "加油，还能更好！",
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
            "front_knee_angle": knee_angles,
            "thigh_raise_angle": thigh_raise_angles,
            "trunk_lean_angle": np.abs(_line_angle_degrees(shoulder, lead_hip) - 90.0),
            "knee_below_hip": knee_below_hip.astype(int),
            "head_tilt_angle": eye_angle,
            "stride_length_norm": stride_distance / torso_length,
            "wrist_forward_dist_norm": np.abs(wrist[:, 0] - shoulder[:, 0]) / torso_length,
            "stride_distance": stride_distance,
            "current_lunge_index": current_lunge_numbers,
            "completed_lunge_count": completed_numbers,
            "lunge_state": lunge_states,
            "thigh_raise_flag": thigh_raise_flags.astype(int),
            "head_tilt_flag": head_tilt_flags.astype(int),
        }
    )

    warning_frame = peak_frame if overall_status == "🚨" else finish_frame
    stability_details = (
        f"手晃动 {upper_body_jitter['wrist']:.3f} / 肘晃动 {upper_body_jitter['elbow']:.3f} / 肩晃动 {upper_body_jitter['shoulder']:.3f}"
    )
    assessments = _build_frame_assessments(
        df["time"],
        df["current_lunge_index"],
        df["completed_lunge_count"],
        df["lunge_state"],
        order_ok,
        stability_ok,
        stability_details,
        thigh_raise_flags,
        head_tilt_flags,
        overall_good,
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
    current_lunge_numbers: Iterable[int],
    completed_numbers: Iterable[int],
    lunge_states: Iterable[str],
    order_ok: bool,
    stability_ok: bool,
    stability_details: str,
    thigh_raise_flags: Iterable[bool],
    head_tilt_flags: Iterable[bool],
    overall_good: bool,
    hand_frame: int | None,
    foot_frame: int | None,
    finish_frame: int | None,
) -> list[FrameAssessment]:
    assessments: list[FrameAssessment] = []
    sequence_order = "先手后脚" if order_ok else "其他"
    sequence_stability = "达标" if stability_ok else "未达标"

    for frame_index, values in enumerate(
        zip(
            timestamps,
            current_lunge_numbers,
            completed_numbers,
            lunge_states,
            thigh_raise_flags,
            head_tilt_flags,
        )
    ):
        timestamp, raw_lunge_number, completed_count, lunge_state, thigh_flag, head_flag = values
        current_lunge_index = int(raw_lunge_number) if int(raw_lunge_number) > 0 else max(int(completed_count), 1)
        frame_order_ok = not (foot_frame is not None and hand_frame is not None and frame_index >= min(hand_frame, foot_frame) and not order_ok)
        score_good = (not bool(thigh_flag)) and stability_ok and frame_order_ok and (not bool(head_flag))
        current_text = "很棒，得分！" if score_good else "加油，还能更好！"
        current_icon = "👍" if score_good else "🚨"
        if score_good or overall_good:
            top_issue = "动作整体达标"
            coaching_advice = "继续保持当前节奏和稳定性"
        elif bool(thigh_flag):
            top_issue = "抬大腿"
            coaching_advice = "注意不要抬大腿，保持髋膝角度稳定"
        elif not frame_order_ok:
            top_issue = "先脚后手"
            coaching_advice = "手启动慢，建议先手再脚"
        elif bool(head_flag):
            top_issue = "头部倾斜"
            coaching_advice = "头部倾斜，注意保持正直"
        else:
            top_issue = "到位不稳"
            coaching_advice = "到位不稳，建议停留0.3秒"

        assessments.append(
            FrameAssessment(
                frame_index=frame_index,
                timestamp=float(timestamp),
                current_lunge_index=current_lunge_index,
                completed_lunge_count=int(completed_count),
                lunge_state=lunge_state,
                current_text=current_text,
                current_icon=current_icon,
                score_text="得分" if score_good else "未得分",
                top_issue=top_issue,
                coaching_advice=coaching_advice,
                hand_foot_order=sequence_order,
                stability=sequence_stability,
                stability_details=stability_details,
                thigh_raise="是" if bool(thigh_flag) else "否",
                head_tilt="歪头提醒" if bool(head_flag) else "正常",
                calf_kick="有" if finish_frame is not None and frame_index >= finish_frame else "检测中",
                show_thigh_warning=bool(thigh_flag),
                show_order_warning=not order_ok and foot_frame is not None and hand_frame is not None and frame_index >= min(hand_frame, foot_frame),
                show_head_tilt_warning=bool(head_flag),
            )
        )
    return assessments
