from __future__ import annotations

import argparse
from pathlib import Path

from .analyzer import analyze_video
from .config import load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="fencing_analyzer")
    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser("analyze", help="Analyze a single video")
    analyze.add_argument("--input", required=True, type=Path)
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--config", type=Path, default=None)
    analyze.add_argument("--render_overlay", action="store_true")

    batch = sub.add_parser("batch", help="Analyze all MP4 files in directory")
    batch.add_argument("--input_dir", required=True, type=Path)
    batch.add_argument("--output_dir", required=True, type=Path)
    batch.add_argument("--config", type=Path, default=None)
    batch.add_argument("--render_overlay", action="store_true")

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    cfg = load_config(args.config)

    if args.command == "analyze":
        analyze_video(args.input, args.output, cfg, render=args.render_overlay)
        return

    for vid in sorted(args.input_dir.glob("*.mp4")):
        out = args.output_dir / vid.stem
        analyze_video(vid, out, cfg, render=args.render_overlay)


if __name__ == "__main__":
    main()
