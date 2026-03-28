from __future__ import annotations

import json

import pytest

from core.standard_loader import StandardLoader, StandardProfileError


def _valid_payload():
    return {
        "lesson_id": "L1",
        "action_type": "lunge",
        "version": "v1",
        "feature_definitions": [{"feature_key": "f1", "display_name_cn": "特征1", "description_cn": "desc"}],
        "phase_definitions": [
            {
                "phase_key": "READY",
                "display_name_cn": "准备位稳定",
                "description_cn": "desc",
                "enter_threshold": 0.8,
                "exit_threshold": 0.6,
                "min_hold_frames": 1,
                "timeout_ms": 1000,
            }
        ],
        "transition_rules": [{"from_phase": "READY", "to_phase": "STARTED", "required": True}],
        "scoring_rules": [{"phase_key": "READY", "feature_weights": {"f1": 1.0}}],
        "ui_labels": {"READY": "准备位稳定"},
    }


def test_standard_loader_missing_fields_raises(tmp_path):
    payload = _valid_payload()
    payload.pop("phase_definitions")
    p = tmp_path / "bad.json"
    p.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(StandardProfileError):
        StandardLoader(p).load_profile("L1", "lunge")


def test_standard_loader_success(tmp_path):
    p = tmp_path / "ok.json"
    p.write_text(json.dumps(_valid_payload(), ensure_ascii=False), encoding="utf-8")
    profile = StandardLoader(p).load_profile("L1", "lunge")
    assert profile.phase_definitions_map["READY"].display_name_cn == "准备位稳定"
