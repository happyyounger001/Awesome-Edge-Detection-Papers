from __future__ import annotations

import csv
from pathlib import Path
from typing import List

from .audio import BeepUnavailableError, detect_beeps
from .config import AnalyzerConfig
from .lunge_detect import detect_lunges
from .metrics import LungeMetrics, summarize_lunges
from .overlay import render_overlay
from .pose import extract_pose_sequence
from .report import generate_report


CSV_FIELDS = [
    "lunge_id",
    "start_time",
    "end_time",
    "duration",
    "knee_angle_min",
    "hip_flexion_max",
    "trunk_lean_max",
    "stride_max",
    "reaction_time",
    "foot_hand_order",
]


def write_csv(path: Path, metrics: List[LungeMetrics]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for m in metrics:
            writer.writerow(m.__dict__)


def analyze_video(
    input_path: Path,
    output_dir: Path,
    cfg: AnalyzerConfig,
    render: bool = True,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    pose = extract_pose_sequence(input_path, ema_alpha=cfg.ema_alpha)
    events = detect_lunges(pose, cfg)

    beep_times = None
    beep_enabled = False
    try:
        beep_times = detect_beeps(input_path, cfg)
        beep_enabled = True
    except BeepUnavailableError:
        beep_times = None

    metrics, _frame_metrics = summarize_lunges(pose, events, cfg, beep_times)

    write_csv(output_dir / "lunges.csv", metrics)
    generate_report(output_dir, input_path.stem, metrics, cfg, beep_enabled)

    if render:
        render_overlay(input_path, output_dir / "overlay.mp4", pose, events, metrics)
