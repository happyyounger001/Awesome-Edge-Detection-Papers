from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from quality_standard import build_quality_standard, save_quality_standard


def load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def load_samples_from_input_dir(input_dir: Path) -> list[dict[str, str]]:
    items = []
    for category in ["training_standard", "competition_effective"]:
        for video in sorted((input_dir / category).glob("*.mp4")):
            items.append({"video_path": str(video), "category": category})
    return items


def build_quality_standard_from_feature_rows(
    feature_rows: list[dict[str, Any]],
    output_path: Path,
    features_output_dir: Path,
) -> dict[str, Any]:
    if not feature_rows:
        raise RuntimeError("未找到可学习的视频样本")

    features_output_dir.mkdir(parents=True, exist_ok=True)
    by_category: dict[str, list[dict[str, Any]]] = {}
    for row in feature_rows:
        by_category.setdefault(row["category"], []).append(row)
    for category, rows in by_category.items():
        (features_output_dir / f"{category}_features.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    merged_path = features_output_dir / "aggregated_features.json"
    merged_path.write_text(json.dumps(feature_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    standard = build_quality_standard(feature_rows, sorted(by_category.keys()))
    standard.setdefault("meta", {})["sample_count"] = len(feature_rows)
    save_quality_standard(output_path, standard)
    return standard


def build_structured_lunge_sample(
    sequence,
    evaluation,
    feature_row: dict[str, Any],
    category: str,
    lunge_index: int = 1,
) -> dict[str, Any]:
    import numpy as np

    side = "left" if float(np.mean(sequence.arrays["left_ankle"][:, 0])) >= float(np.mean(sequence.arrays["right_ankle"][:, 0])) else "right"
    back = "right" if side == "left" else "left"
    fps = sequence.fps
    dt = 1.0 / fps

    shoulder = sequence.arrays[f"{side}_shoulder"]
    hip = sequence.arrays[f"{side}_hip"]
    elbow = sequence.arrays[f"{side}_elbow"]
    wrist = sequence.arrays[f"{side}_wrist"]
    front_knee = sequence.arrays[f"{side}_knee"]
    front_ankle = sequence.arrays[f"{side}_ankle"]
    rear_knee = sequence.arrays[f"{back}_knee"]
    rear_hip = sequence.arrays[f"{back}_hip"]
    rear_ankle = sequence.arrays[f"{back}_ankle"]
    left_eye = sequence.arrays["left_eye"]
    right_eye = sequence.arrays["right_eye"]

    torso_len = np.maximum(np.linalg.norm(shoulder - hip, axis=1), 1e-6)
    states = evaluation.metrics["lunge_state"].tolist()
    stride_norm = np.abs(front_ankle[:, 0] - rear_ankle[:, 0]) / torso_len
    wrist_forward_norm = np.abs(wrist[:, 0] - shoulder[:, 0]) / torso_len
    trunk_lean = np.abs(np.degrees(np.arctan2((hip - shoulder)[:, 1], (hip - shoulder)[:, 0])) - 90.0)
    head_tilt = np.abs(np.degrees(np.arctan2((right_eye - left_eye)[:, 1], (right_eye - left_eye)[:, 0])))

    def angle_deg(a, b, c):
        ba = a - b
        bc = c - b
        denom = np.linalg.norm(ba) * np.linalg.norm(bc)
        if denom == 0:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(ba, bc) / denom, -1.0, 1.0))))

    front_knee_angle = np.array([angle_deg(hip[i], front_knee[i], front_ankle[i]) for i in range(sequence.frame_count)])
    rear_knee_angle = np.array([angle_deg(rear_hip[i], rear_knee[i], rear_ankle[i]) for i in range(sequence.frame_count)])
    wrist_speed = np.abs(np.gradient(wrist_forward_norm, dt))

    prepare_indices = [i for i, state in enumerate(states) if state in {"IDLE", "PREPARE"}]
    baseline_indices = prepare_indices[: max(1, min(len(prepare_indices), int(0.2 * fps)))] if prepare_indices else list(range(max(1, int(0.2 * fps))))
    hold_indices = [i for i, state in enumerate(states) if state in {"HOLD", "RETURN"}]
    extended_index = int(np.argmax(stride_norm)) if len(stride_norm) else 0
    pre_return_indices = hold_indices[: max(1, min(len(hold_indices), int(0.3 * fps)))] if hold_indices else [extended_index]

    sample_step = max(1, int(round(fps * 0.05)))
    dynamic_series = {
        "fps": fps,
        "sampling_interval_sec": sample_step / fps,
        "frames": [
            {
                "frame": int(i),
                "time": float(i / fps),
                "stage": states[i],
                "front_knee_angle": float(front_knee_angle[i]),
                "trunk_lean_deg": float(trunk_lean[i]),
                "head_tilt_deg": float(head_tilt[i]),
                "wrist_forward_dist_norm": float(wrist_forward_norm[i]),
                "stride_length_norm": float(stride_norm[i]),
                "wrist_speed_norm": float(wrist_speed[i]),
            }
            for i in range(0, sequence.frame_count, sample_step)
        ],
    }

    static_baseline = {
        "front_knee_angle": float(np.mean(front_knee_angle[baseline_indices])),
        "rear_knee_angle": float(np.mean(rear_knee_angle[baseline_indices])),
        "trunk_lean_deg": float(np.mean(trunk_lean[baseline_indices])),
        "head_tilt_deg": float(np.mean(head_tilt[baseline_indices])),
        "stance_width_norm": float(np.mean(stride_norm[baseline_indices])),
        "wrist_to_shoulder_norm": float(np.mean(wrist_forward_norm[baseline_indices])),
    }

    extended_phase = {
        "frame": extended_index,
        "time_sec": float(extended_index / fps),
        "front_knee_angle": float(front_knee_angle[extended_index]),
        "trunk_lean_deg": float(trunk_lean[extended_index]),
        "wrist_forward_dist_norm": float(wrist_forward_norm[extended_index]),
        "stride_length_norm": float(stride_norm[extended_index]),
        "wrist_speed_norm": float(wrist_speed[extended_index]),
        "target_line_alignment_hint": float(max(0.0, 1.0 - trunk_lean[extended_index] / 90.0)),
    }

    pre_return_phase = {
        "frame_start": int(pre_return_indices[0]),
        "frame_end": int(pre_return_indices[-1]),
        "duration_sec": float((pre_return_indices[-1] - pre_return_indices[0] + 1) / fps),
        "front_knee_angle_mean": float(np.mean(front_knee_angle[pre_return_indices])),
        "trunk_lean_std": float(np.std(trunk_lean[pre_return_indices])),
        "head_tilt_std": float(np.std(head_tilt[pre_return_indices])),
        "wrist_stability_std": float(np.std(wrist_forward_norm[pre_return_indices])),
    }

    max_wrist_speed_norm = float(np.max(wrist_speed))
    summary = {
        "time_to_extension_sec": float(extended_index / fps),
        "max_stride_length_norm": float(np.max(stride_norm)),
        "max_wrist_forward_dist_norm": float(np.max(wrist_forward_norm)),
        "max_wrist_speed_norm": max_wrist_speed_norm,
        "stable_duration_sec": float(feature_row.get("stable_hold_duration", 0.0)),
        "hand_foot_delta_sec": float(feature_row.get("hand_foot_start_delta", 0.0)),
        "biomech_focus": {
            "trunk_beta_like_deg": float(extended_phase["trunk_lean_deg"]),
            "front_knee_alpha_like_deg": float(extended_phase["front_knee_angle"]),
            "extension_line_norm": float(extended_phase["wrist_forward_dist_norm"]),
            "time_t_like_sec": float(extended_phase["time_sec"]),
            "speed_v_like_norm": max_wrist_speed_norm,
            "terminal_delta_hint": float(max(0.0, 1.0 - pre_return_phase["head_tilt_std"])),
        },
    }

    return {
        "lunge_index": lunge_index,
        "category": category,
        "static_baseline": static_baseline,
        "dynamic_series": dynamic_series,
        "extended_phase": extended_phase,
        "pre_return_phase": pre_return_phase,
        "summary": summary,
    }


def build_quality_standard_from_sources(
    input_dir: Path | None,
    manifest_path: Path | None,
    output_path: Path,
    features_output_dir: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    from config import load_config
    from feature_extractor import extract_video_features
    from pose.estimator import PoseEstimator

    config = load_config("config.yaml")
    estimator = PoseEstimator(config)
    try:
        samples = load_manifest(manifest_path) if manifest_path else load_samples_from_input_dir(input_dir)
        if not samples:
            raise RuntimeError("未找到可学习的视频样本")

        feature_rows = []
        for sample in samples:
            sequence = estimator.extract(sample["video_path"])
            feature_rows.append(extract_video_features(sequence, config, sample["video_path"], sample["category"]))
        standard = build_quality_standard_from_feature_rows(feature_rows, output_path, features_output_dir)
        return standard, feature_rows
    finally:
        estimator.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quality standard from high-quality lunge videos")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features_output_dir", type=Path, default=Path("outputs/learning_features"))
    args = parser.parse_args()

    standard, _feature_rows = build_quality_standard_from_sources(
        input_dir=args.input,
        manifest_path=args.manifest,
        output_path=args.output,
        features_output_dir=args.features_output_dir,
    )
    print(f"Quality standard written to {args.output}")
    print(f"Loaded feature keys: {', '.join(sorted(standard.get('features', {}).keys())[:8])} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
