from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from pose.estimator import KEYPOINTS, SKELETON
from pose.models import PoseSequence
from rules.evaluator import SequenceEvaluation

FONT_CANDIDATES = [
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/arphic/ukai.ttc",
]


def render_overlay_video(
    video_path: Path,
    output_dir: Path,
    pose_sequence: PoseSequence,
    evaluation: SequenceEvaluation,
) -> tuple[Path, dict[str, Path]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = output_dir / "overlay.mp4"

    cap = cv2.VideoCapture(str(video_path))
    writer = cv2.VideoWriter(
        str(overlay_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        pose_sequence.fps,
        (pose_sequence.width, pose_sequence.height),
    )

    keyframe_indices = {
        "出手瞬间": evaluation.hand_frame,
        "弓步完成瞬间": evaluation.finish_frame,
        "预警触发瞬间": evaluation.warning_frame,
    }
    saved_keyframes: dict[str, Path] = {}
    frame_index = 0

    while True:
        ok, frame = cap.read()
        if not ok or frame_index >= pose_sequence.frame_count:
            break
        assessment = evaluation.frame_assessments[min(frame_index, len(evaluation.frame_assessments) - 1)]
        annotated = draw_pose_overlay(frame, pose_sequence, assessment.frame_index, assessment)
        writer.write(annotated)

        for title, idx in keyframe_indices.items():
            if idx is not None and frame_index == idx and title not in saved_keyframes:
                keyframe_path = frames_dir / f"{title}.png"
                cv2.imwrite(str(keyframe_path), annotated)
                saved_keyframes[title] = keyframe_path
        frame_index += 1

    cap.release()
    writer.release()
    return overlay_path, saved_keyframes


def draw_pose_overlay(frame, pose_sequence: PoseSequence, frame_index: int, assessment):
    canvas = frame.copy()
    points: dict[str, tuple[int, int]] = {}
    for name in KEYPOINTS:
        x = int(pose_sequence.arrays[name][frame_index, 0] * pose_sequence.width)
        y = int(pose_sequence.arrays[name][frame_index, 1] * pose_sequence.height)
        points[name] = (x, y)
        cv2.circle(canvas, (x, y), 3 if "eye" in name or "ear" in name or name == "nose" else 4, (0, 255, 0), -1)

    for start, end in SKELETON:
        cv2.line(canvas, points[start], points[end], (255, 215, 0), 2)

    warning_lines = [f"第 {assessment.current_lunge_index} 个弓步"]
    warning_lines.append(f"状态：{assessment.current_text}")
    if assessment.show_thigh_warning:
        warning_lines.append("抬大腿预警")
    if assessment.show_order_warning:
        warning_lines.append("先脚后手预警")
    if assessment.show_head_tilt_warning:
        warning_lines.append("歪头提醒")
    warning_lines.append(f"建议：{assessment.coaching_advice}")
    return _draw_chinese_labels(canvas, warning_lines)


def _draw_chinese_labels(frame: np.ndarray, lines: list[str]) -> np.ndarray:
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(image)
    font = _load_font(28)
    x = 20
    y = 20

    for line in lines:
        bbox = draw.textbbox((x, y), line, font=font)
        draw.rounded_rectangle((bbox[0] - 12, bbox[1] - 8, bbox[2] + 12, bbox[3] + 8), radius=8, fill=(0, 0, 0, 160))
        draw.text((x, y), line, font=font, fill=(255, 255, 255))
        y += (bbox[3] - bbox[1]) + 20

    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for candidate in FONT_CANDIDATES:
        font_path = Path(candidate)
        if font_path.exists():
            return ImageFont.truetype(str(font_path), size=size)
    return ImageFont.load_default()
