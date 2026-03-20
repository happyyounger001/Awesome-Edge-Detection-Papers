from __future__ import annotations

import importlib.util
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


@dataclass
class AppConfig:
    hand_speed_threshold: float = 0.018
    foot_speed_threshold: float = 0.02
    lunge_finish_speed_threshold: float = 0.006
    stability_seconds_threshold: float = 0.30
    upper_body_motion_threshold: float = 0.025
    thigh_raise_angle_threshold: float = 20.0
    head_tilt_angle_threshold: float = 15.0
    calf_kick_speed_threshold: float = 0.025
    require_knee_below_hip_for_thigh_raise: bool = True
    smoothing_alpha: float = 0.35
    visibility_threshold: float = 0.5
    preview_width: int = 960
    preview_height: int = 540
    standard_path: str = "standards/quality_standard_u6_foil_v1.json"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = AppConfig()


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        return DEFAULT_CONFIG
    config_path = Path(path)
    yaml_spec = importlib.util.find_spec("yaml")
    if yaml_spec is None:
        return DEFAULT_CONFIG
    import yaml

    data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    return AppConfig(**{**DEFAULT_CONFIG.to_dict(), **data})
