from pathlib import Path

from config import load_config


def test_load_config_overrides_defaults(tmp_path: Path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("hand_speed_threshold: 0.123\npreview_width: 800\n", encoding="utf-8")
    cfg = load_config(config_path)
    assert cfg.hand_speed_threshold == 0.123
    assert cfg.preview_width == 800
