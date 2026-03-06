# Fencing Lunge Analyzer (MVP)

A Windows-friendly Python prototype for side-view fencing lunge training analysis.

## What it does

Given a video (`.mp4`) it runs:
1. **MediaPipe Pose** keypoint extraction
2. **Automatic lunge detection and counting**
3. **Posture metrics** (knee angle, hip flexion, trunk lean, stride ratio)
4. Optional **beep-to-move reaction time** (if `ffmpeg` is available)
5. **Foot-hand start order** (foot first / hand first / sync)

Outputs per video:
- `overlay.mp4` (skeleton + lunge info)
- `lunges.csv` (per-lunge metrics)
- `summary.json` (aggregates)
- `report.html` (charts + explanations)

## Project structure

```text
fencing_analyzer/
  pose.py
  lunge_detect.py
  metrics.py
  overlay.py
  report.py
  audio.py
  cli.py
templates/
  report.html.j2
config.yaml
requirements.txt
```

## Input recording rules (important)

- Side-view camera, fixed tripod
- Full body visible from head to feet
- Stable lighting and clean background
- Minimize occlusion of hips/knees
- Prefer 60fps (30fps supported but less precise for reaction time)
- Include 5–10 lunge repetitions per video

## Install (Windows)

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

## Run

Single video:

```powershell
python -m fencing_analyzer analyze --input input_videos/demo.mp4 --output outputs/demo --config config.yaml --render_overlay
```

Batch mode:

```powershell
python -m fencing_analyzer batch --input_dir input_videos --output_dir outputs --config config.yaml --render_overlay
```

## Output layout

```text
outputs/
  <video_stem>/
    overlay.mp4
    lunges.csv
    summary.json
    report.html
    figures/
      duration.png
      posture_metrics.png
      reaction_time.png
      foot_hand_order.png
```

## CSV columns

```text
lunge_id,start_time,end_time,duration,knee_angle_min,hip_flexion_max,trunk_lean_max,stride_max,reaction_time,foot_hand_order
```

## Config

Tune thresholds in `config.yaml`, including:
- speed thresholds (`v_start_thr`, `v_end_thr`)
- lunge duration/gap constraints
- posture target thresholds
- smoothing (`ema_alpha`)
- beep detection parameters

## Tests

```bash
pytest -q
```

## Limitations

- 2D landmarks only (no depth)
- Wrist landmark approximates hand action; no sword-tip tracking
- Quality drops with occlusion, camera motion, or poor lighting
- Reaction-time analysis requires audio track + `ffmpeg`
