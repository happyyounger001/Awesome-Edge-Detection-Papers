from __future__ import annotations

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication, QTimer, Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from analysis.engine import AnalysisEngine
from analysis.models import AnalysisResult
from config import load_config
from pose.estimator import PoseEstimator
from quality_standard import apply_standard_to_config, load_quality_standard
from rules.evaluator import evaluate_sequence
from ui.learning_window import LearningSystemWindow
from ui.dialogs.settings_dialog import SettingsDialog
from ui.progress_dialog import ProgressDialog
from ui.report_window import ReportWindow
from ui.video_overlay import draw_pose_layer, draw_pose_overlay
from ui.video_panel import VideoPanel

logger = logging.getLogger(__name__)


class FencingMainWindow(QMainWindow):
    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.default_config = load_config(config_path)
        self.quality_standard = None
        standard_path = Path(self.config.standard_path) if hasattr(self.config, "standard_path") else Path("standards/quality_standard_u6_foil_v1.json")
        if standard_path.exists():
            self.quality_standard = load_quality_standard(standard_path)
            apply_standard_to_config(self.config, self.quality_standard)
        self.engine = AnalysisEngine(self.config, quality_standard=self.quality_standard)
        self.preview_estimator = PoseEstimator(self.config)
        self.video_path: Path | None = None
        self.capture: cv2.VideoCapture | None = None
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.current_frame = 0
        self.manual_go_time: float | None = None
        self.analysis_result: AnalysisResult | None = None
        self.preview_sequence = None
        self.report_window: ReportWindow | None = None
        self.learning_window: LearningSystemWindow | None = None
        self.progress_dialog: ProgressDialog | None = None
        self.settings_dialog: SettingsDialog | None = None
        self.preview_evaluation = None
        self.preview_eval_future: Future | None = None
        self.bg_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="preview-eval")
        self.phase_label_map = self._build_phase_label_map()

        self.is_video_loading = False
        self.video_load_progress = 0
        self.is_playing = False
        self.current_time = 0.0
        self.play_start_monotonic = 0.0
        self.play_start_frame = 0
        self.duration = 0.0
        self.playback_rate = 1.0
        self.rotation_fix_deg = 0.0
        self.player_state = "IDLE"

        self.setWindowTitle("Fencing Lunge AI Trainer")
        self.resize(1360, 820)
        self._build_ui()
        self._apply_common_styles()

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        self.engine.close()
        self.preview_estimator.close()
        self.bg_executor.shutdown(wait=False, cancel_futures=True)
        super().closeEvent(event)

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QGridLayout(root)

        input_group = QGroupBox("1️⃣ 视频输入区")
        input_layout = QHBoxLayout(input_group)
        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        import_button = QPushButton("导入视频")
        import_button.clicked.connect(self.import_video)
        input_layout.addWidget(import_button)
        input_layout.addWidget(self.path_edit)

        playback_group = QGroupBox("2️⃣ 视频播放区")
        playback_layout = QVBoxLayout(playback_group)
        self.video_panel_widget = VideoPanel()
        self.video_panel_widget.setMinimumSize(int(self.config.preview_width * 0.85), int(self.config.preview_height * 0.85))
        playback_controls = QHBoxLayout()
        self.play_toggle_button = QPushButton("播放")
        self.play_toggle_button.clicked.connect(self.toggle_play)
        self.replay_button = QPushButton("重播")
        self.replay_button.clicked.connect(self.replay)
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["0.5x", "1.0x", "2.0x"])
        self.rate_combo.setCurrentText("1.0x")
        self.rate_combo.currentTextChanged.connect(self.change_playback_rate)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.seek_video)
        playback_controls.addWidget(self.play_toggle_button)
        playback_controls.addWidget(self.replay_button)
        playback_controls.addWidget(QLabel("倍速"))
        playback_controls.addWidget(self.rate_combo)
        playback_controls.addWidget(self.progress_slider)
        playback_layout.addWidget(self.video_panel_widget)
        playback_layout.addLayout(playback_controls)

        realtime_group = QGroupBox("3️⃣ 实时分析区")
        realtime_layout = QVBoxLayout(realtime_group)
        self.realtime_text = QTextEdit()
        self.realtime_text.setReadOnly(True)
        realtime_layout.addWidget(self.realtime_text)

        algorithm_group = QGroupBox("4️⃣ 当前判断标准和算法")
        algorithm_layout = QVBoxLayout(algorithm_group)
        self.algorithm_text = QTextEdit()
        self.algorithm_text.setReadOnly(True)
        algorithm_layout.addWidget(self.algorithm_text)

        control_group = QGroupBox("5️⃣ 控制与报告区")
        control_layout = QVBoxLayout(control_group)
        operation_group = QGroupBox("训练控制")
        operation_layout = QVBoxLayout(operation_group)
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.run_analysis)
        go_button = QPushButton("手动标记")
        go_button.clicked.connect(self.mark_go)
        self.settings_button = QPushButton("参数设置")
        self.settings_button.clicked.connect(self.open_settings_dialog)
        operation_layout.addWidget(self.analyze_button)
        operation_layout.addWidget(go_button)
        operation_layout.addWidget(self.settings_button)

        report_group = QGroupBox("报告与标准")
        report_layout = QVBoxLayout(report_group)
        self.report_button = QPushButton("查看报告")
        self.report_button.clicked.connect(self.open_report)
        self.learning_button = QPushButton("学习系统")
        self.learning_button.clicked.connect(self.open_learning_system)
        self.standard_button = QPushButton("当前标准")
        self.standard_button.clicked.connect(self.open_standard_dialog)
        report_layout.addWidget(self.report_button)
        report_layout.addWidget(self.learning_button)
        report_layout.addWidget(self.standard_button)
        for btn in [
            self.analyze_button,
            go_button,
            self.settings_button,
            self.report_button,
            self.learning_button,
            self.standard_button,
        ]:
            btn.setFixedSize(112, 34)

        self.status_label = QLabel("状态：等待导入视频")
        button_matrix = QHBoxLayout()
        button_matrix.addWidget(operation_group)
        button_matrix.addWidget(report_group)
        control_layout.addLayout(button_matrix)
        for widget in [self.status_label]:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)

        layout.addWidget(input_group, 0, 0, 1, 2)
        layout.addWidget(playback_group, 1, 0, 4, 2)
        layout.addWidget(realtime_group, 1, 2)
        layout.addWidget(control_group, 2, 2, 3, 1)
        layout.setColumnStretch(0, 3)
        layout.setColumnStretch(1, 3)
        layout.setColumnStretch(2, 2)
        self.setCentralWidget(root)
        self._update_realtime_placeholder()
        self._update_algorithm_panel(None)

    def _apply_common_styles(self) -> None:
        self.setStyleSheet(
            """
            QGroupBox { font-weight: 600; border: 1px solid #d0d7de; border-radius: 8px; margin-top: 8px; }
            QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 4px; color: #1f4f82; }
            QPushButton { background: #2f6fed; color: white; border-radius: 6px; padding: 4px 8px; min-height: 30px; min-width: 92px; font-size: 12px; }
            QPushButton:hover { background: #2458bc; }
            QTextEdit { border: 1px solid #d0d7de; border-radius: 8px; background: #ffffff; }
            """
        )

    def open_standard_dialog(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("当前判断标准和算法")
        dialog.resize(520, 420)
        layout = QVBoxLayout(dialog)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(self.algorithm_text.toPlainText())
        layout.addWidget(text)
        dialog.exec()

    def _build_phase_label_map(self) -> dict[str, str]:
        labels = {
            "IDLE": "准备位稳定",
            "PREPARE": "启动准备",
            "LEG_OUT": "推进出击中",
            "ARM_EXTEND": "推进出击中",
            "LUNGE_ACTIVE": "到位保持",
            "HOLD": "到位保持",
            "RETURN": "回收复原中",
            "READY": "准备位稳定",
            "STARTED": "启动阶段",
            "EXTENDING": "推进/出击",
            "REACHED": "到位阶段",
            "RECOVERING": "回收复原中",
        }
        if isinstance(self.quality_standard, dict):
            labels.update(self.quality_standard.get("ui_labels", {}))
        return labels

    def _phase_label(self, phase_key: str) -> str:
        simple = {
            "READY": "准备",
            "IDLE": "准备",
            "STARTED": "出击",
            "EXTENDING": "出击",
            "HOLD": "出击",
            "RETURN": "收回",
            "RECOVERING": "收回",
            "COMPLETED_LOCK": "收回",
        }
        if phase_key in simple:
            return simple[phase_key]
        return self.phase_label_map.get(phase_key, phase_key)

    def _update_realtime_placeholder(self) -> None:
        self.realtime_text.setText(
            "当前第 1 个弓步\n"
            "已完成弓步数：0\n"
            "当前状态：等待分析\n"
            "当前阶段：准备位稳定\n"
            "是否得分：--\n"
            "本次最需要改进：--\n"
            "训练建议：--\n"
            "细项判定：\n"
            "抬大腿：--\n"
            "稳定性：--\n"
            "手脚顺序：--\n"
            "头部姿态：--"
        )

    def _update_algorithm_panel(self, assessment) -> None:
        lunge_state = assessment.lunge_state if assessment is not None else "IDLE"
        current_phase_cn = self._phase_label(lunge_state)
        standard_name = "未配置"
        standard_version = "未配置"
        standard_rules = "当前学习标准未加载完成"
        if isinstance(self.quality_standard, dict):
            meta = self.quality_standard.get("meta", {})
            standard_name = str(meta.get("name", "未命名标准"))
            standard_version = str(meta.get("version", "未配置"))
            stage_rules = self.quality_standard.get("stage_rules", {})
            standard_rules = "；".join(stage_rules.keys()) if stage_rules else "无"
        self.algorithm_text.setText(
            f"动作识别：MediaPipe Pose\n"
            f"当前阶段：{current_phase_cn}\n"
            f"当前状态机：{lunge_state}\n"
            f"判断标准：{standard_name}\n"
            f"关键阶段定义：{standard_rules}\n"
            f"当前标准版本：{standard_version}"
        )

    def _show_progress(self, title: str, percent: int, status: str = "running") -> None:
        if self.progress_dialog is None:
            self.progress_dialog = ProgressDialog(title, self)
            self.progress_dialog.show()
        self.progress_dialog.update_progress(title, percent, status)
        QCoreApplication.processEvents()
        if status in {"success", "failed"}:
            QTimer.singleShot(400, self.progress_dialog.close)
            self.progress_dialog = None

    def import_video(self) -> None:
        file_path, _ = QFileDialog.getOpenFileName(self, "选择训练视频", "", "Video Files (*.mp4 *.mov *.avi)")
        if not file_path:
            return
        self.timer.stop()
        self.is_playing = False
        self.play_toggle_button.setText("播放")
        self.player_state = "LOADING"
        self.video_path = Path(file_path)
        self.path_edit.setText(file_path)
        self.manual_go_time = None
        self.analysis_result = None
        self.preview_evaluation = None
        self.preview_eval_future = None
        self._open_video()
        self.status_label.setText("状态：已导入视频，可播放或开始分析")

    def _open_video(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.video_path))
        self.rotation_fix_deg = self._detect_rotation_fix(self.capture)
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = frame_count / fps if fps else 0.0
        self.progress_slider.setMaximum(max(0, frame_count - 1))
        self.current_frame = 0
        self.is_video_loading = True
        self.play_toggle_button.setEnabled(False)
        self._show_progress("视频加载中", 0)

        def update_progress(percent: int) -> None:
            self.video_load_progress = percent
            self._show_progress("视频加载中", percent)

        self.preview_sequence = self.preview_estimator.extract(self.video_path, progress_callback=update_progress)
        self.preview_eval_future = self.bg_executor.submit(
            evaluate_sequence, self.preview_sequence, self.config, self.manual_go_time
        )
        self.is_video_loading = False
        self.player_state = "READY"
        self.play_toggle_button.setEnabled(True)
        self._show_progress("视频加载完成", 100, status="success")
        logger.info("Video loaded: %s frames, duration %.2fs", frame_count, self.duration)
        self._show_frame(0)

    def toggle_play(self) -> None:
        if self.capture is None or self.is_video_loading or self.player_state == "LOADING":
            return
        if self.is_playing:
            self.pause()
            return
        if self.player_state not in {"READY", "PAUSED", "ENDED"}:
            return
        if self.current_frame >= self.progress_slider.maximum():
            self._show_frame(0)
        self.is_playing = True
        self.player_state = "PLAYING"
        self.play_start_monotonic = time.monotonic()
        self.play_start_frame = self.current_frame
        self.play_toggle_button.setText("暂停")
        self._restart_timer()
        logger.info("Playback started at frame=%s rate=%s", self.current_frame, self.playback_rate)

    def pause(self) -> None:
        self.timer.stop()
        self.is_playing = False
        if self.player_state != "LOADING":
            self.player_state = "PAUSED"
        self.play_toggle_button.setText("播放")
        logger.info("Playback paused at frame=%s", self.current_frame)

    def replay(self) -> None:
        if self.capture is None:
            return
        self.pause()
        self._show_frame(0)
        self.toggle_play()
        logger.info("Playback replayed")

    def change_playback_rate(self, text: str) -> None:
        self.playback_rate = float(text.replace("x", ""))
        logger.info("Playback rate changed to %s", self.playback_rate)
        if self.is_playing:
            self._restart_timer()

    def _restart_timer(self) -> None:
        if self.capture is None:
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        interval = max(1, int(1000 / (fps * self.playback_rate)))
        self.timer.start(interval)

    def _on_slider_pressed(self) -> None:
        self._was_playing_before_seek = self.is_playing
        self.pause()

    def seek_video(self) -> None:
        if self.capture is None or self.player_state == "LOADING":
            return
        frame_index = self.progress_slider.value()
        logger.info("Seek to frame=%s", frame_index)
        self._show_frame(frame_index)
        if getattr(self, "_was_playing_before_seek", False):
            self.toggle_play()

    def _next_frame(self) -> None:
        if self.capture is None:
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        elapsed = max(0.0, time.monotonic() - self.play_start_monotonic)
        expected = self.play_start_frame + int(elapsed * fps * self.playback_rate)
        next_frame = max(self.current_frame + 1, expected)
        if next_frame > self.progress_slider.maximum():
            self.pause()
            self.player_state = "ENDED"
            self._show_frame(self.progress_slider.maximum())
            return
        self._show_frame(next_frame)

    def _show_frame(self, frame_index: int) -> None:
        if self.capture is None or self.preview_sequence is None:
            return
        total = self.preview_sequence.frame_count
        frame_index = max(0, min(frame_index, total - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.timer.stop()
            self.is_playing = False
            self.player_state = "ENDED"
            self.play_toggle_button.setText("播放")
            return
        self.current_frame = frame_index
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_time = frame_index / fps
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(frame_index)
        self.progress_slider.blockSignals(False)

        normalized = self._normalize_frame_orientation(frame)
        if self.analysis_result is not None and self.analysis_result.frame_assessments:
            assessment = self.analysis_result.frame_assessments[min(frame_index, len(self.analysis_result.frame_assessments) - 1)]
        else:
            if self.preview_evaluation is None and self.preview_eval_future is not None and self.preview_eval_future.done():
                self.preview_evaluation = self.preview_eval_future.result()
            if self.preview_evaluation is None:
                self.status_label.setText("状态：后台分析中（播放不受阻塞）")
                pose_preview = draw_pose_layer(self.preview_sequence, frame_index, frame.shape)
                composed_preview = draw_pose_overlay(frame, self.preview_sequence, frame_index, type("A", (), {
                    "current_lunge_index": 0,
                    "show_thigh_warning": False,
                    "show_order_warning": False,
                    "show_head_tilt_warning": False,
                })(), include_status_text=False)
                return self._display_layers(
                    self._normalize_frame_orientation(composed_preview),
                    pose_preview * 0,
                    ["当前阶段：分析中", "提醒：后台分析尚未完成", "建议：继续播放，结果将自动更新"],
                )
            assessment = self.preview_evaluation.frame_assessments[min(frame_index, len(self.preview_evaluation.frame_assessments) - 1)]
        composed = draw_pose_overlay(frame, self.preview_sequence, frame_index, assessment, include_status_text=False)
        composed = self._normalize_frame_orientation(composed)
        pose_layer = draw_pose_layer(self.preview_sequence, frame_index, normalized.shape)
        phase_cn = self._phase_label(assessment.lunge_state)
        status_lines = [
            f"当前阶段：{phase_cn}",
            f"提示：{assessment.top_issue or '注意稳定'}",
            f"建议：{assessment.coaching_advice or '继续向前'}",
        ]
        self._display_layers(composed, pose_layer * 0, status_lines)
        self._update_realtime_panel(assessment)
        self._update_algorithm_panel(assessment)

    def _display_layers(self, video_frame, pose_layer, status_lines: list[str]) -> None:
        self.video_panel_widget.update_layers(video_frame, pose_layer, status_lines)

    def _update_realtime_panel(self, assessment) -> None:
        is_good = assessment.current_text == "很棒，得分！"
        status_color = "#15803d" if is_good else "#b91c1c"
        issue_bg = "#dcfce7" if is_good else "#fee2e2"
        phase_cn = self._phase_label(assessment.lunge_state)
        def metric_row(name: str, target: str, current: str, ok: bool) -> str:
            color = "#15803d" if ok else "#b91c1c"
            bg = "#dcfce7" if ok else "#fee2e2"
            return (
                f"<tr>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #eee;'>{name}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #eee;color:#334155;'>{target}</td>"
                f"<td style='padding:6px 8px;border-bottom:1px solid #eee;background:{bg};color:{color};font-weight:700;'>{current}</td>"
                f"</tr>"
            )

        stability_ok = assessment.stability == "达标"
        thigh_ok = assessment.thigh_raise == "否"
        order_ok = assessment.hand_foot_order == "先手后脚"
        head_ok = "正常" in assessment.head_tilt
        yes_no = lambda ok: "达标" if ok else "未达标"

        metrics_table = (
            "<table style='width:100%;border-collapse:collapse;font-size:13px;'>"
            "<tr><th style='text-align:left;padding:6px 8px;'>项目</th><th style='text-align:left;padding:6px 8px;'>目标值/区间</th><th style='text-align:left;padding:6px 8px;'>当前值</th></tr>"
            + metric_row("抬大腿", f"≥ {self.config.thigh_raise_angle_threshold:.1f}°", yes_no(thigh_ok), thigh_ok)
            + metric_row("稳定性", f"≤ {self.config.stability_seconds_threshold:.2f}s晃动阈值", yes_no(stability_ok), stability_ok)
            + metric_row("手脚顺序", "先手后脚", yes_no(order_ok), order_ok)
            + metric_row("头部姿态", f"偏斜 ≤ {self.config.head_tilt_angle_threshold:.1f}°", yes_no(head_ok), head_ok)
            + "</table>"
        )
        self.realtime_text.setHtml(
            f"<h3>当前弓步：第 {assessment.current_lunge_index} 个（已完成 {assessment.completed_lunge_count} 个）</h3>"
            f"<p><b>当前阶段：</b>{phase_cn}</p>"
            f"<p style='color:{status_color};font-size:18px;'><b>当前状态：{assessment.current_text}</b></p>"
            f"<p style='background:{issue_bg};padding:8px;border-radius:6px;'><b>本次最需要改进：{assessment.top_issue}</b></p>"
            f"<p><b>训练建议：</b>{assessment.coaching_advice}</p>"
            f"<hr>"
            f"{metrics_table}"
            f"<p style='margin-top:8px;'><b>稳定性细节：</b>{assessment.stability_details}</p>"
        )

    def _detect_rotation_fix(self, capture: cv2.VideoCapture) -> float:
        if hasattr(cv2, "CAP_PROP_ORIENTATION_META"):
            try:
                orientation = capture.get(cv2.CAP_PROP_ORIENTATION_META)
                if orientation in {90.0, 180.0, 270.0}:
                    return float(orientation)
            except Exception:
                return 0.0
        return 0.0

    def _normalize_frame_orientation(self, frame):
        if self.rotation_fix_deg == 90.0:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        if self.rotation_fix_deg == 180.0:
            return cv2.rotate(frame, cv2.ROTATE_180)
        if self.rotation_fix_deg == 270.0:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
        return frame

    def mark_go(self) -> None:
        if self.capture is None:
            QMessageBox.information(self, "提示", "请先导入视频。")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.manual_go_time = self.current_frame / fps
        logger.info("Manual go marked at %.3fs", self.manual_go_time)
        self.status_label.setText(f"状态：已手动标记 go = {self.manual_go_time:.3f} 秒")
        self._show_frame(self.current_frame)

    def open_settings_dialog(self) -> None:
        if self.settings_dialog is None:
            self.settings_dialog = SettingsDialog(self.config, self.apply_parameters, self.reset_parameters, self)
        self.settings_dialog.reload_from_config()
        self.settings_dialog.show()
        self.settings_dialog.raise_()
        self.settings_dialog.activateWindow()

    def apply_parameters(self, stability: float, thigh_angle: float, head_tilt: float, knee_rule: bool) -> None:
        self.config.stability_seconds_threshold = stability
        self.config.thigh_raise_angle_threshold = thigh_angle
        self.config.head_tilt_angle_threshold = head_tilt
        self.config.require_knee_below_hip_for_thigh_raise = knee_rule
        logger.info(
            "Parameters applied: stability=%.2f thigh=%.1f head=%.1f knee_rule=%s",
            stability,
            thigh_angle,
            head_tilt,
            knee_rule,
        )
        self.status_label.setText("状态：参数已应用到当前判定逻辑")
        if self.preview_sequence is not None:
            self.analysis_result = None
            self._show_frame(self.current_frame)

    def reset_parameters(self) -> None:
        self.config.stability_seconds_threshold = self.default_config.stability_seconds_threshold
        self.config.thigh_raise_angle_threshold = self.default_config.thigh_raise_angle_threshold
        self.config.head_tilt_angle_threshold = self.default_config.head_tilt_angle_threshold
        self.config.require_knee_below_hip_for_thigh_raise = self.default_config.require_knee_below_hip_for_thigh_raise
        self.status_label.setText("状态：已恢复默认参数")
        if self.preview_sequence is not None:
            self.analysis_result = None
            self._show_frame(self.current_frame)

    def run_analysis(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "提示", "请先导入视频。")
            return
        if not isinstance(self.quality_standard, dict):
            QMessageBox.warning(self, "提示", "当前学习标准未加载完成，无法启动正式分析。")
            return
        self.analyze_button.setEnabled(False)
        output_dir = Path("outputs") / self.video_path.stem
        self.status_label.setText("状态：分析中，请稍候...")
        self._show_progress("分析中", 0)
        self.analysis_result = self.engine.analyze_video(
            self.video_path,
            output_dir,
            self.manual_go_time,
            progress_callback=lambda percent: self._show_progress("分析中", percent),
        )
        self._show_progress("分析完成", 100, status="success")
        self.analyze_button.setEnabled(True)
        last_assessment = self.analysis_result.frame_assessments[-1]
        transitions: list[str] = []
        prev_state: str | None = None
        for fa in self.analysis_result.frame_assessments:
            if fa.lunge_state != prev_state:
                transitions.append(f"{fa.frame_index}:{fa.lunge_state}")
                prev_state = fa.lunge_state
        logger.info("Phase/state transitions: %s", " -> ".join(transitions[:30]))
        logger.info("Completed lunge count: %s", last_assessment.completed_lunge_count)
        self.status_label.setText(
            f"状态：分析完成，当前第 {last_assessment.current_lunge_index} 个弓步，已完成 {last_assessment.completed_lunge_count} 个，报告已生成：{self.analysis_result.report_html}"
        )
        logger.info("Analysis finished for %s", self.video_path)
        self._show_frame(self.current_frame)


    def open_learning_system(self) -> None:
        if self.learning_window is None:
            self.learning_window = LearningSystemWindow(import_callback=self.import_quality_standard)
        self.learning_window.show()
        self.learning_window.raise_()
        self.learning_window.activateWindow()

    def import_quality_standard(self, standard: dict, standard_path: Path) -> None:
        self.quality_standard = standard
        apply_standard_to_config(self.config, standard)
        self.phase_label_map = self._build_phase_label_map()
        self.engine.close()
        self.engine = AnalysisEngine(self.config, quality_standard=self.quality_standard)
        if self.settings_dialog is not None:
            self.settings_dialog.reload_from_config()
        self.status_label.setText(f"状态：已导入质量标准 {standard_path.name}")
        self._update_algorithm_panel(None)
        if self.preview_sequence is not None:
            self.analysis_result = None
            self._show_frame(self.current_frame)

    def open_report(self) -> None:
        if self.analysis_result is None:
            QMessageBox.information(self, "提示", "请先完成分析。")
            return
        self.report_button.setEnabled(False)
        for percent in [20, 65, 100]:
            status = "success" if percent == 100 else "running"
            self._show_progress("加载报告", percent, status=status)
        self.report_window = ReportWindow(self.analysis_result.report_html)
        self.report_window.show()
        self.report_button.setEnabled(True)
        logger.info("Report window opened: %s", self.analysis_result.report_html)
