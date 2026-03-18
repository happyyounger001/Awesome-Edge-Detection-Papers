from pathlib import Path

from config import AppConfig
from report.generator import ReportGenerator
from rules.evaluator import evaluate_sequence
from tests.helpers import make_sequence


def test_generate_report_outputs_charts_and_html(tmp_path: Path):
    evaluation = evaluate_sequence(make_sequence(), AppConfig(), manual_go_time=0.2)
    generator = ReportGenerator(AppConfig())
    keyframes_dir = tmp_path / "frames"
    keyframes_dir.mkdir()
    keyframes = {}
    for name in ["出手瞬间", "弓步完成瞬间", "预警触发瞬间"]:
        path = keyframes_dir / f"{name}.png"
        path.write_bytes(b"fake")
        keyframes[name] = path
    html_path, charts = generator.generate(Path("demo.mp4"), tmp_path, evaluation, keyframes)
    assert html_path.exists()
    assert charts["timing"].exists()
    assert charts["stability"].exists()
    assert (tmp_path / "metrics.csv").exists()
