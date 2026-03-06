from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from .config import AnalyzerConfig
from .lunge_detect import LungeEvent
from .pose import PoseSequence
from .utils import angle_deg, safe_mean


@dataclass
class LungeMetrics:
    lunge_id: int
    start_time: float
    end_time: float
    duration: float
    knee_angle_min: float
    hip_flexion_max: float
    trunk_lean_max: float
    stride_max: float
    reaction_time: Optional[float]
    foot_hand_order: str


def _side_prefix(pose: PoseSequence) -> str:
    return "left" if float(np.mean(pose.x["left_ankle"])) > float(np.mean(pose.x["right_ankle"])) else "right"


def _build_frame_metrics(pose: PoseSequence) -> Dict[str, np.ndarray]:
    side = _side_prefix(pose)
    other = "left" if side == "right" else "right"

    frames = pose.frames
    knee = np.full(frames, np.nan)
    hip_flex = np.full(frames, np.nan)
    trunk = np.full(frames, np.nan)
    stride = np.full(frames, np.nan)

    for i in range(frames):
        hip = np.array([pose.x[f"{side}_hip"][i], pose.y[f"{side}_hip"][i]])
        knee_pt = np.array([pose.x[f"{side}_knee"][i], pose.y[f"{side}_knee"][i]])
        ankle = np.array([pose.x[f"{side}_ankle"][i], pose.y[f"{side}_ankle"][i]])
        shoulder = np.array([pose.x[f"{side}_shoulder"][i], pose.y[f"{side}_shoulder"][i]])
        other_hip = np.array([pose.x[f"{other}_hip"][i], pose.y[f"{other}_hip"][i]])

        knee[i] = angle_deg(hip, knee_pt, ankle)
        hip_flex[i] = angle_deg(shoulder, hip, knee_pt)
        vertical_anchor = np.array([hip[0], hip[1] - 1.0])
        trunk[i] = angle_deg(shoulder, hip, vertical_anchor)
        stride[i] = abs(pose.x[f"{side}_ankle"][i] - pose.x[f"{other}_ankle"][i]) / (
            abs(shoulder[1] - other_hip[1]) + 1e-6
        )

    return {
        "knee_angle": knee,
        "hip_flexion": hip_flex,
        "trunk_lean": trunk,
        "stride_ratio": stride,
    }


def classify_foot_hand_order(pose: PoseSequence, event: LungeEvent, cfg: AnalyzerConfig) -> str:
    side = _side_prefix(pose)
    wrist = pose.x[f"{side}_wrist"]
    ankle = pose.x[f"{side}_ankle"]
    dt = 1.0 / pose.fps
    start = event.start_frame
    end = min(event.end_frame + int(0.2 * pose.fps), pose.frames - 1)
    wv = np.abs(np.gradient(wrist[start : end + 1], dt))
    av = np.abs(np.gradient(ankle[start : end + 1], dt))

    w_idx = np.argmax(wv >= cfg.v_start_thr)
    a_idx = np.argmax(av >= cfg.v_start_thr)

    if not np.any(wv >= cfg.v_start_thr):
        w_idx = 9999
    if not np.any(av >= cfg.v_start_thr):
        a_idx = 9999

    if abs(w_idx - a_idx) <= 2:
        return "sync"
    return "foot_first" if a_idx < w_idx else "hand_first"


def summarize_lunges(
    pose: PoseSequence,
    events: List[LungeEvent],
    cfg: AnalyzerConfig,
    beep_times: Optional[List[float]] = None,
) -> tuple[List[LungeMetrics], Dict[str, np.ndarray]]:
    frame_metrics = _build_frame_metrics(pose)
    results: List[LungeMetrics] = []

    for idx, e in enumerate(events):
        s, t = e.start_frame, e.end_frame + 1
        reaction_time = None
        if beep_times:
            ref = min(beep_times, key=lambda b: abs((s / pose.fps) - b))
            reaction_time = (s / pose.fps) - ref

        results.append(
            LungeMetrics(
                lunge_id=e.lunge_id,
                start_time=s / pose.fps,
                end_time=e.end_frame / pose.fps,
                duration=e.duration_frames / pose.fps,
                knee_angle_min=float(np.nanmin(frame_metrics["knee_angle"][s:t])),
                hip_flexion_max=float(np.nanmax(frame_metrics["hip_flexion"][s:t])),
                trunk_lean_max=float(np.nanmax(frame_metrics["trunk_lean"][s:t])),
                stride_max=float(np.nanmax(frame_metrics["stride_ratio"][s:t])),
                reaction_time=reaction_time,
                foot_hand_order=classify_foot_hand_order(pose, e, cfg),
            )
        )

    return results, frame_metrics


def metric_flags(m: LungeMetrics, cfg: AnalyzerConfig) -> Dict[str, bool]:
    return {
        "knee_ok": m.knee_angle_min <= cfg.knee_angle_thr,
        "hip_ok": m.hip_flexion_max >= cfg.hip_flexion_thr,
        "trunk_ok": m.trunk_lean_max <= cfg.trunk_lean_thr,
        "stride_ok": m.stride_max >= cfg.stride_ratio_thr,
    }


def aggregate_summary(metrics: List[LungeMetrics]) -> Dict[str, float]:
    return {
        "lunge_count": len(metrics),
        "duration_mean": safe_mean([m.duration for m in metrics]),
        "knee_angle_min_mean": safe_mean([m.knee_angle_min for m in metrics]),
        "hip_flexion_max_mean": safe_mean([m.hip_flexion_max for m in metrics]),
        "trunk_lean_max_mean": safe_mean([m.trunk_lean_max for m in metrics]),
        "stride_max_mean": safe_mean([m.stride_max for m in metrics]),
    }
