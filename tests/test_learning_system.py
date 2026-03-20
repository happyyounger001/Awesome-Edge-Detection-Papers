import json

from learning_system import build_quality_standard_from_feature_rows
from quality_scorer import score_features
from quality_standard import build_quality_standard


def test_build_quality_standard_contains_required_sections():
    rows = [
        {
            "video_path": "a.mp4",
            "category": "training_standard",
            "front_knee_angle_min": 50.0,
            "rear_knee_angle_min": 70.0,
            "trunk_lean_max_deg": 18.0,
            "head_tilt_max_deg": 10.0,
            "stride_max_norm": 1.1,
            "wrist_forward_dist_max_norm": 0.8,
            "elbow_extension_angle_min": 145.0,
            "hip_knee_line_ground_angle_max": 28.0,
            "hand_start_time": 0.1,
            "foot_start_time": 0.12,
            "hand_foot_start_delta": 0.02,
            "lunge_out_duration": 0.3,
            "lunge_return_duration": 0.4,
            "lunge_total_duration": 0.8,
            "time_to_max_stride": 0.25,
            "time_to_max_wrist_extension": 0.22,
            "wrist_stability_std": 0.01,
            "elbow_stability_std": 0.02,
            "shoulder_stability_std": 0.015,
            "head_stability_std": 0.012,
            "trunk_stability_std": 0.02,
            "stable_hold_duration": 0.35,
            "hand_first_ratio": 1.0,
            "foot_first_ratio": 0.0,
            "coordination_score_raw": 0.95,
            "head_trunk_alignment_score": 0.9,
        }
    ]
    standard = build_quality_standard(rows, ["training_standard"])
    assert set(["meta", "thresholds", "features", "stage_rules", "score_weights"]).issubset(standard.keys())


def test_score_features_returns_scores_and_explanations():
    rows = [{
        "video_path": "a.mp4", "category": "training_standard", "front_knee_angle_min": 50.0, "rear_knee_angle_min": 70.0,
        "trunk_lean_max_deg": 18.0, "head_tilt_max_deg": 10.0, "stride_max_norm": 1.1, "wrist_forward_dist_max_norm": 0.8,
        "elbow_extension_angle_min": 145.0, "hip_knee_line_ground_angle_max": 28.0, "hand_start_time": 0.1, "foot_start_time": 0.12,
        "hand_foot_start_delta": 0.02, "lunge_out_duration": 0.3, "lunge_return_duration": 0.4, "lunge_total_duration": 0.8,
        "time_to_max_stride": 0.25, "time_to_max_wrist_extension": 0.22, "wrist_stability_std": 0.01, "elbow_stability_std": 0.02,
        "shoulder_stability_std": 0.015, "head_stability_std": 0.012, "trunk_stability_std": 0.02, "stable_hold_duration": 0.35,
        "hand_first_ratio": 1.0, "foot_first_ratio": 0.0, "coordination_score_raw": 0.95, "head_trunk_alignment_score": 0.9,
    }]
    standard = build_quality_standard(rows, ["training_standard"])
    scores, explanations = score_features(rows[0], standard)
    assert "overall_score" in scores
    assert isinstance(explanations, list)


def test_build_quality_standard_from_feature_rows_writes_aggregated_outputs(tmp_path):
    rows = [{
        "video_path": "a.mp4", "category": "training_standard", "front_knee_angle_min": 50.0, "rear_knee_angle_min": 70.0,
        "trunk_lean_max_deg": 18.0, "head_tilt_max_deg": 10.0, "stride_max_norm": 1.1, "wrist_forward_dist_max_norm": 0.8,
        "elbow_extension_angle_min": 145.0, "hip_knee_line_ground_angle_max": 28.0, "hand_start_time": 0.1, "foot_start_time": 0.12,
        "hand_foot_start_delta": 0.02, "lunge_out_duration": 0.3, "lunge_return_duration": 0.4, "lunge_total_duration": 0.8,
        "time_to_max_stride": 0.25, "time_to_max_wrist_extension": 0.22, "wrist_stability_std": 0.01, "elbow_stability_std": 0.02,
        "shoulder_stability_std": 0.015, "head_stability_std": 0.012, "trunk_stability_std": 0.02, "stable_hold_duration": 0.35,
        "hand_first_ratio": 1.0, "foot_first_ratio": 0.0, "coordination_score_raw": 0.95, "head_trunk_alignment_score": 0.9,
    }]
    output_path = tmp_path / "quality_standard.json"
    features_dir = tmp_path / "learning_features"

    standard = build_quality_standard_from_feature_rows(rows, output_path, features_dir)

    assert output_path.exists()
    assert (features_dir / "aggregated_features.json").exists()
    assert standard["meta"]["sample_count"] == 1
    assert json.loads((features_dir / "aggregated_features.json").read_text(encoding="utf-8")) == rows
