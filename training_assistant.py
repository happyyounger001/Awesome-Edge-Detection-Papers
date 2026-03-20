from __future__ import annotations

import argparse
from pathlib import Path

from analysis.engine import AnalysisEngine
from config import load_config
from quality_standard import apply_standard_to_config, load_quality_standard


def main() -> int:
    parser = argparse.ArgumentParser(description="Run training assistant using a learned quality standard")
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--standard", type=Path, default=Path("standards/quality_standard_u6_foil_v1.json"))
    args = parser.parse_args()

    config = load_config("config.yaml")
    standard = load_quality_standard(args.standard)
    apply_standard_to_config(config, standard)

    engine = AnalysisEngine(config, quality_standard=standard)
    try:
        output_dir = Path("outputs") / args.video.stem
        result = engine.analyze_video(args.video, output_dir, manual_go_time=None)
        print(f"Report: {result.report_html}")
        print(f"Overlay: {result.overlay_video}")
        print(f"Overall score: {result.quality.overall_score}")
        print(f"Status: {result.quality.status_text}")
        print(f"Top issue: {result.quality.top_issue}")
        print(f"Advice: {result.quality.top_advice}")
        for item in result.quality.deviations:
            print(f"- {item}")
    finally:
        engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
