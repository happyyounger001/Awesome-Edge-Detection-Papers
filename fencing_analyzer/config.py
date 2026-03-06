from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import yaml


@dataclass
class AnalyzerConfig:
    v_start_thr: float = 0.015
    v_end_thr: float = 0.005
    min_lunge_duration: float = 0.2
    min_gap_between_lunges: float = 0.3
    settle_frames: int = 4

    knee_angle_thr: float = 120.0
    hip_flexion_thr: float = 60.0
    trunk_lean_thr: float = 20.0
    stride_ratio_thr: float = 0.55

    ema_alpha: float = 0.35
    visibility_thr: float = 0.5

    beep_energy_thr: float = 2.5
    beep_min_separation_s: float = 0.25


DEFAULT_CONFIG = AnalyzerConfig()


def load_config(path: str | Path | None) -> AnalyzerConfig:
    if path is None:
        return DEFAULT_CONFIG
    cfg_path = Path(path)
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found: {cfg_path}")
    data: Dict[str, Any] = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    return AnalyzerConfig(**{**DEFAULT_CONFIG.__dict__, **data})
