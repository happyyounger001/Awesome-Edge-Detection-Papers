from __future__ import annotations

from typing import Any


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


def score_features(features: dict[str, float], standard: dict[str, Any]) -> tuple[dict[str, float], list[str]]:
    feature_stats = standard.get("features", {})
    weights = standard.get("score_weights", {"posture": 0.30, "timing": 0.30, "stability": 0.25, "coordination": 0.15})

    groups = {
        "posture": ["front_knee_angle_min", "rear_knee_angle_min", "trunk_lean_max_deg", "head_tilt_max_deg", "stride_max_norm"],
        "timing": ["hand_foot_start_delta", "lunge_out_duration", "lunge_return_duration", "time_to_max_stride"],
        "stability": ["wrist_stability_std", "elbow_stability_std", "shoulder_stability_std", "head_stability_std", "trunk_stability_std", "stable_hold_duration"],
        "coordination": ["coordination_score_raw", "head_trunk_alignment_score", "hand_first_ratio"],
    }

    scores = {}
    deviations: list[str] = []
    for group_name, names in groups.items():
        group_scores = []
        for name in names:
            if name not in features or name not in feature_stats:
                continue
            score = _score_from_band(float(features[name]), feature_stats[name])
            group_scores.append(score)
            stats = feature_stats[name]
            if float(features[name]) < stats.get("p25", float(features[name])):
                deviations.append(f"{name} 低于优秀样本区间")
            elif float(features[name]) > stats.get("p75", float(features[name])):
                deviations.append(f"{name} 高于优秀样本区间")
        scores[f"{group_name}_score"] = round(sum(group_scores) / len(group_scores), 2) if group_scores else 0.0

    scores["overall_score"] = round(
        scores.get("posture_score", 0.0) * weights.get("posture", 0.30)
        + scores.get("timing_score", 0.0) * weights.get("timing", 0.30)
        + scores.get("stability_score", 0.0) * weights.get("stability", 0.25)
        + scores.get("coordination_score", 0.0) * weights.get("coordination", 0.15),
        2,
    )
    explanations = deviations[:5] or ["当前动作整体位于优秀样本区间内"]
    return scores, explanations
