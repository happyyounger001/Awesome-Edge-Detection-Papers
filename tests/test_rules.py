from config import AppConfig
from rules.evaluator import evaluate_sequence
from tests.helpers import make_sequence


def test_evaluate_sequence_generates_timing_quality_and_realtime_fields():
    result = evaluate_sequence(make_sequence(), AppConfig(), manual_go_time=0.2)
    assert result.timing.go_to_hand is not None
    assert result.timing.go_to_finish is not None
    assert result.quality.hand_foot_order in {"先手后脚", "其他"}
    assert result.quality.head_tilt in {"正常", "歪头提醒"}
    assert result.quality.overall_status in {"👍", "🚨"}
    assert result.quality.status_text in {"很棒，得分！", "加油，还能更好！"}
    assert len(result.frame_assessments) == 80
    assert result.frame_assessments[0].current_lunge_index >= 1
    assert result.frame_assessments[-1].current_text in {"很棒，得分！", "加油，还能更好！"}
    assert result.frame_assessments[-1].coaching_advice
