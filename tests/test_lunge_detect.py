from config import AppConfig
from rules.evaluator import evaluate_sequence
from tests.helpers import make_sequence


def test_detect_key_event_order_and_lunge_counts_from_synthetic_sequence():
    result = evaluate_sequence(make_sequence(), AppConfig(), manual_go_time=0.2)
    assert result.hand_frame is not None
    assert result.finish_frame is not None
    assert result.hand_frame < result.finish_frame
    assert result.frame_assessments[result.finish_frame].current_lunge_index >= 1
    assert result.frame_assessments[-1].completed_lunge_count >= 1
