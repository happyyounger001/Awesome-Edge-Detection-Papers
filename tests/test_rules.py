from config import AppConfig
from rules.evaluator import evaluate_sequence
from tests.helpers import make_sequence


def test_evaluate_sequence_generates_timing_and_quality():
    result = evaluate_sequence(make_sequence(), AppConfig(), manual_go_time=0.2)
    assert result.timing.go_to_hand is not None
    assert result.timing.go_to_finish is not None
    assert result.quality.hand_foot_order in {"手先脚后", "脚先手后"}
    assert result.quality.overall_status in {"👍", "🚨"}
    assert len(result.frame_assessments) == 80
