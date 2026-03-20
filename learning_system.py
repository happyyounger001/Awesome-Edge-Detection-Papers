from __future__ import annotations

import argparse
import json
from pathlib import Path

from config import load_config
from feature_extractor import extract_video_features
from pose.estimator import PoseEstimator
from quality_standard import build_quality_standard, save_quality_standard


def _load_manifest(manifest_path: Path) -> list[dict[str, str]]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _load_from_input_dir(input_dir: Path) -> list[dict[str, str]]:
    items = []
    for category in ["training_standard", "competition_effective"]:
        for video in sorted((input_dir / category).glob("*.mp4")):
            items.append({"video_path": str(video), "category": category})
    return items


def main() -> int:
    parser = argparse.ArgumentParser(description="Build quality standard from high-quality lunge videos")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--features_output_dir", type=Path, default=Path("outputs/learning_features"))
    args = parser.parse_args()

    config = load_config("config.yaml")
    estimator = PoseEstimator(config)
    try:
        samples = _load_manifest(args.manifest) if args.manifest else _load_from_input_dir(args.input)
        if not samples:
            raise RuntimeError("未找到可学习的视频样本")

        feature_rows = []
        for sample in samples:
            sequence = estimator.extract(sample["video_path"])
            feature_rows.append(extract_video_features(sequence, config, sample["video_path"], sample["category"]))

        args.features_output_dir.mkdir(parents=True, exist_ok=True)
        by_category = {}
        for row in feature_rows:
            by_category.setdefault(row["category"], []).append(row)
        for category, rows in by_category.items():
            (args.features_output_dir / f"{category}_features.json").write_text(
                json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
            )

        standard = build_quality_standard(feature_rows, sorted(by_category.keys()))
        save_quality_standard(args.output, standard)
        print(f"Quality standard written to {args.output}")
        return 0
    finally:
        estimator.close()


if __name__ == "__main__":
    raise SystemExit(main())
