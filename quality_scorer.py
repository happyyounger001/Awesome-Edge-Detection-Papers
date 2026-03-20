from __future__ import annotations

from typing import Any

DEFAULT_WEIGHTS = {"posture": 0.30, "timing": 0.30, "stability": 0.25, "coordination": 0.15}

FEATURE_METADATA: dict[str, dict[str, str]] = {
    "hand_foot_start_delta": {"label": "手脚启动不同步", "advice": "手启动慢，建议先手再脚", "direction": "low"},
    "hand_first_ratio": {"label": "先脚后手", "advice": "先脚后手预警，建议手先启动", "direction": "high"},
    "stable_hold_duration": {"label": "到位不稳", "advice": "到位不稳，建议停留0.3秒", "direction": "high"},
    "wrist_stability_std": {"label": "手腕不稳定", "advice": "到位后手腕保持稳定，减少晃动", "direction": "low"},
    "elbow_stability_std": {"label": "肘部不稳定", "advice": "到位后保持0.3秒不动", "direction": "low"},
    "shoulder_stability_std": {"label": "肩部不稳定", "advice": "肩部摆动过大，注意上肢固定", "direction": "low"},
    "head_tilt_max_deg": {"label": "头部倾斜", "advice": "头部倾斜，注意保持正直", "direction": "low"},
    "head_trunk_alignment_score": {"label": "头干不对齐", "advice": "保持头颈躯干一条线", "direction": "high"},
    "hip_knee_line_ground_angle_max": {"label": "抬大腿", "advice": "注意不要抬大腿，保持髋膝角度稳定", "direction": "low"},
    "trunk_lean_max_deg": {"label": "躯干前倾过大", "advice": "上体前倾过大，注意核心稳定", "direction": "low"},
}

GROUPS = {
    "posture": ["front_knee_angle_min", "rear_knee_angle_min", "trunk_lean_max_deg", "head_tilt_max_deg", "stride_max_norm"],
    "timing": ["hand_foot_start_delta", "lunge_out_duration", "lunge_return_duration", "time_to_max_stride"],
    "stability": ["wrist_stability_std", "elbow_stability_std", "shoulder_stability_std", "head_stability_std", "trunk_stability_std", "stable_hold_duration"],
    "coordination": ["coordination_score_raw", "head_trunk_alignment_score", "hand_first_ratio"],
}


def _score_from_band(value: float, stats: dict[str, float]) -> float:
    p10 = stats.get("p10", value)
    p25 = stats.get("p25", value)
    p75 = stats.get("p75", value)
    p90 = stats.get("p90", value)
    if p25 <= value <= p75:
        return 100.0
    if value < p10 or value > p90:
        return 40.0
    if value < p25:
        return 60.0 + 40.0 * (value - p10) / max(p25 - p10, 1e-6)
    return 60.0 + 40.0 * (p90 - value) / max(p90 - p75, 1e-6)


def _normalized_gap(value: float, stats: dict[str, float], direction: str) -> float:
    p25 = stats.get("p25", value)
    p75 = stats.get("p75", value)
    p10 = stats.get("p10", p25)
    p90 = stats.get("p90", p75)
    if p25 <= value <= p75:
        return 0.0
    if direction == "high":
        if value > p75:
            return (value - p75) / max(p90 - p75, 1e-6)
        return (p25 - value) / max(p25 - p10, 1e-6)
    if value < p25:
        return (p25 - value) / max(p25 - p10, 1e-6)
    return (value - p75) / max(p90 - p75, 1e-6)


def build_training_feedback(features: dict[str, float], standard: dict[str, Any]) -> dict[str, Any]:
    feature_stats = standard.get("features", {})
    candidates: list[dict[str, Any]] = []
    for feature_name, metadata in FEATURE_METADATA.items():
        if feature_name not in features or feature_name not in feature_stats:
            continue
        gap = _normalized_gap(float(features[feature_name]), feature_stats[feature_name], metadata.get("direction", "low"))
        if gap <= 0:
            continue
        candidates.append(
            {
                "feature": feature_name,
                "label": metadata["label"],
                "advice": metadata["advice"],
                "gap": gap,
                "text": f"{metadata['label']}（{feature_name} 偏离优秀区间）",
            }
        )

    if not candidates:
        return {
            "status_text": "很棒，得分！",
            "top_issue": "动作整体达标",
            "top_advice": "继续保持当前节奏和稳定性",
            "deviations": ["当前动作整体位于优秀样本区间内"],
        }

    candidates.sort(key=lambda item: item["gap"], reverse=True)
    top = candidates[0]
    return {
        "status_text": "加油，还能更好！",
        "top_issue": top["label"],
        "top_advice": top["advice"],
        "deviations": [item["text"] for item in candidates[:5]],
    }


def score_features(features: dict[str, float], standard: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    feature_stats = standard.get("features", {})
    weights = standard.get("score_weights", DEFAULT_WEIGHTS)

    scores = {}
    for group_name, names in GROUPS.items():
        group_scores = []
        for name in names:
            if name not in features or name not in feature_stats:
                continue
            group_scores.append(_score_from_band(float(features[name]), feature_stats[name]))
        scores[f"{group_name}_score"] = round(sum(group_scores) / len(group_scores), 2) if group_scores else 0.0

    scores["overall_score"] = round(
        scores.get("posture_score", 0.0) * weights.get("posture", DEFAULT_WEIGHTS["posture"])
        + scores.get("timing_score", 0.0) * weights.get("timing", DEFAULT_WEIGHTS["timing"])
        + scores.get("stability_score", 0.0) * weights.get("stability", DEFAULT_WEIGHTS["stability"])
        + scores.get("coordination_score", 0.0) * weights.get("coordination", DEFAULT_WEIGHTS["coordination"]),
        2,
    )
    feedback = build_training_feedback(features, standard)
    return scores, feedback["deviations"]
