from pathlib import Path

from fencing_analyzer.config import AnalyzerConfig
from fencing_analyzer.metrics import LungeMetrics
from fencing_analyzer.report import generate_report


def test_report_generation(tmp_path: Path):
    metrics = [
        LungeMetrics(
            lunge_id=1,
            start_time=0.1,
            end_time=0.7,
            duration=0.6,
            knee_angle_min=100.0,
            hip_flexion_max=70.0,
            trunk_lean_max=15.0,
            stride_max=0.6,
            reaction_time=None,
            foot_hand_order="foot_first",
        )
    ]
    out = generate_report(tmp_path, "demo", metrics, AnalyzerConfig(), beep_enabled=False)
    assert out.exists()
    assert (tmp_path / "summary.json").exists()
    assert (tmp_path / "figures" / "posture_metrics.png").exists()
