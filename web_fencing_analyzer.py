from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from flask import Flask, redirect, render_template_string, request, send_from_directory, url_for

from analysis.engine import AnalysisEngine
from config import load_config
from quality_standard import apply_standard_to_config, load_quality_standard

app = Flask(__name__)
BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "web_uploads"
OUTPUT_DIR = BASE_DIR / "web_outputs"
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

config = load_config("config.yaml")
quality_standard = None
standard_path = Path(config.standard_path)
if standard_path.exists():
    quality_standard = load_quality_standard(standard_path)
    apply_standard_to_config(config, quality_standard)
engine = AnalysisEngine(config, quality_standard=quality_standard)

PAGE = """
<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <title>Fencing Lunge Web Analyzer</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 24px; background:#f8fafc; }
    .card { background:#fff; border:1px solid #e2e8f0; border-radius:10px; padding:16px; margin-bottom:16px; }
    button { padding:8px 14px; border:none; border-radius:8px; background:#2563eb; color:#fff; }
    a { color:#2563eb; text-decoration:none; }
  </style>
</head>
<body>
  <h1>Fencing Lunge Web Analyzer</h1>
  <div class="card">
    <h3>上传视频并分析</h3>
    <form action="{{ url_for('analyze') }}" method="post" enctype="multipart/form-data">
      <input type="file" name="video" accept="video/*" required>
      <button type="submit">开始分析</button>
    </form>
  </div>
  {% if result %}
  <div class="card">
    <h3>分析结果</h3>
    <p>视频：{{ result.video_name }}</p>
    <p>已完成弓步：{{ result.completed }}</p>
    <p><a href="{{ result.report_url }}" target="_blank">打开HTML报告</a></p>
    <p><a href="{{ result.overlay_url }}" target="_blank">打开叠加视频</a></p>
  </div>
  {% endif %}
</body>
</html>
"""


@app.get("/")
def index():
    return render_template_string(PAGE, result=None)


@app.post("/analyze")
def analyze():
    file = request.files.get("video")
    if file is None or file.filename is None or file.filename == "":
        return render_template_string(PAGE, result={"video_name": "未选择文件", "completed": 0, "report_url": "#", "overlay_url": "#"})

    safe_name = f"{uuid4().hex}_{Path(file.filename).name}"
    video_path = UPLOAD_DIR / safe_name
    file.save(video_path)

    output_dir = OUTPUT_DIR / video_path.stem
    result = engine.analyze_video(video_path, output_dir, manual_go_time=None)
    completed = result.frame_assessments[-1].completed_lunge_count if result.frame_assessments else 0

    response = {
        "video_name": file.filename,
        "completed": completed,
        "report_url": url_for("artifacts", folder=output_dir.name, filename=result.report_html.name),
        "overlay_url": url_for("artifacts", folder=output_dir.name, filename=result.overlay_video.name),
    }
    return render_template_string(PAGE, result=response)


@app.get("/artifacts/<folder>/<path:filename>")
def artifacts(folder: str, filename: str):
    return send_from_directory(OUTPUT_DIR / folder, filename)


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
