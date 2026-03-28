from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from config import AppConfig

DEFAULT_SCORE_WEIGHTS = {
    "posture": 0.30,
    "timing": 0.30,
    "stability": 0.25,
    "coordination": 0.15,
}


def load_quality_standard(path: str | Path) -> dict[str, Any]:
    standard_path = Path(path)
    if not standard_path.exists():
        raise FileNotFoundError(f"质量标准文件不存在: {standard_path}")
    return json.loads(standard_path.read_text(encoding="utf-8"))


def save_quality_standard(path: str | Path, data: dict[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def apply_standard_to_config(config: "AppConfig", standard: dict[str, Any]) -> "AppConfig":
    thresholds = standard.get("thresholds", {})
    stage_rules = standard.get("stage_rules", {})
    config.stability_seconds_threshold = float(thresholds.get("stability_window_sec", config.stability_seconds_threshold))
    config.head_tilt_angle_threshold = float(thresholds.get("head_tilt_deg_thr", config.head_tilt_angle_threshold))
    config.thigh_raise_angle_threshold = float(thresholds.get("hip_knee_ground_angle_thr", config.thigh_raise_angle_threshold))
    config.upper_body_motion_threshold = float(
        stage_rules.get("hold", {}).get("shoulder_stability_std_p90", config.upper_body_motion_threshold)
    )
    config.lunge_min_gap_sec = float(stage_rules.get("cycle", {}).get("min_gap_sec", config.lunge_min_gap_sec))
    config.lunge_hold_sec = float(stage_rules.get("hold", {}).get("stable_hold_duration_p75", config.lunge_hold_sec))
    config.lunge_started_timeout_sec = float(stage_rules.get("cycle", {}).get("started_timeout_sec", config.lunge_started_timeout_sec))
    config.lunge_extending_timeout_sec = float(stage_rules.get("cycle", {}).get("extending_timeout_sec", config.lunge_extending_timeout_sec))
    config.lunge_hold_timeout_sec = float(stage_rules.get("cycle", {}).get("hold_timeout_sec", config.lunge_hold_timeout_sec))
    config.lunge_recover_timeout_sec = float(stage_rules.get("cycle", {}).get("recover_timeout_sec", config.lunge_recover_timeout_sec))
    return config


def summarize_feature(values: list[float]) -> dict[str, float]:
    arr = sorted(float(value) for value in values)
    count = len(arr)
    if count == 0:
        raise ValueError("values must not be empty")

    def percentile(p: float) -> float:
        if count == 1:
            return arr[0]
        rank = (count - 1) * (p / 100.0)
        lower = math.floor(rank)
        upper = math.ceil(rank)
        if lower == upper:
            return arr[lower]
        fraction = rank - lower
        return arr[lower] + (arr[upper] - arr[lower]) * fraction

    mean = sum(arr) / count
    variance = sum((value - mean) ** 2 for value in arr) / count
    midpoint = count // 2
    if count % 2 == 1:
        median = arr[midpoint]
    else:
        median = (arr[midpoint - 1] + arr[midpoint]) / 2.0

    return {
        "mean": float(mean),
        "median": float(median),
        "std": float(math.sqrt(variance)),
        "p10": float(percentile(10)),
        "p25": float(percentile(25)),
        "p75": float(percentile(75)),
        "p90": float(percentile(90)),
    }


def build_quality_standard(feature_rows: list[dict[str, Any]], source_categories: list[str]) -> dict[str, Any]:
    feature_names = [
        key for key in feature_rows[0].keys()
        if key not in {"video_path", "category"} and isinstance(feature_rows[0][key], (int, float))
    ]
    features = {name: summarize_feature([float(row[name]) for row in feature_rows]) for name in feature_names}

    hold_rule_source = features
    return {
        "meta": {
            "name": "U6 Foil Quality Standard",
            "version": "v1",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "source_categories": source_categories,
            "note": "score_weights 当前为默认权重，后续可由数据学习替换",
        },
        "lunge_definition": {
            "rule": "lunge_out + return = one complete lunge"
        },
        "thresholds": {
            "stability_window_sec": features.get("stable_hold_duration", {}).get("p75", 0.30),
            "head_tilt_deg_thr": features.get("head_tilt_max_deg", {}).get("p90", 15.0),
            "hip_knee_ground_angle_thr": features.get("hip_knee_line_ground_angle_max", {}).get("p75", 20.0),
        },
        "features": features,
        "stage_rules": {
            "prepare": {
                "head_trunk_alignment_score_p75": features.get("head_trunk_alignment_score", {}).get("p75", 0.8),
            },
            "lunge_out": {
                "hand_foot_start_delta_p90": features.get("hand_foot_start_delta", {}).get("p90", 0.12),
                "time_to_max_stride_p90": features.get("time_to_max_stride", {}).get("p90", 0.40),
            },
            "hold": {
                "wrist_stability_std_p90": hold_rule_source.get("wrist_stability_std", {}).get("p90", 0.04),
                "elbow_stability_std_p90": hold_rule_source.get("elbow_stability_std", {}).get("p90", 0.05),
                "shoulder_stability_std_p90": hold_rule_source.get("shoulder_stability_std", {}).get("p90", 0.03),
                "head_tilt_deg_p90": hold_rule_source.get("head_tilt_max_deg", {}).get("p90", 12.0),
                "stable_hold_duration_p75": hold_rule_source.get("stable_hold_duration", {}).get("p75", 0.30),
            },
            "return": {
                "lunge_return_duration_p90": features.get("lunge_return_duration", {}).get("p90", 0.60),
            },
        },
        "score_weights": DEFAULT_SCORE_WEIGHTS,
    }
