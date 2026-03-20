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
