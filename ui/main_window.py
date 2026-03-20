from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PySide6.QtCore import QCoreApplication, QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
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
from ui.progress_dialog import ProgressDialog
from ui.report_window import ReportWindow
from ui.video_overlay import draw_pose_overlay

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

        self.is_video_loading = False
        self.video_load_progress = 0
        self.is_playing = False
        self.current_time = 0.0
        self.duration = 0.0
        self.playback_rate = 1.0

        self.setWindowTitle("Fencing Lunge AI Trainer")
        self.resize(1640, 960)
        self._build_ui()
        self._load_config_to_widgets()

    def closeEvent(self, event):
        if self.capture is not None:
            self.capture.release()
        self.engine.close()
        self.preview_estimator.close()
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
        self.video_label = QLabel("请先导入视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(self.config.preview_width, self.config.preview_height)
        self.video_label.setStyleSheet("background:#111;color:#fff;border-radius:8px;")
        playback_controls = QHBoxLayout()
        self.play_button = QPushButton("播放")
        self.play_button.clicked.connect(self.toggle_play)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self.pause)
        self.replay_button = QPushButton("重播")
        self.replay_button.clicked.connect(self.replay)
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["0.5x", "1.0x", "1.5x", "2.0x"])
        self.rate_combo.setCurrentText("1.0x")
        self.rate_combo.currentTextChanged.connect(self.change_playback_rate)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderPressed.connect(self._on_slider_pressed)
        self.progress_slider.sliderReleased.connect(self.seek_video)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.pause_button)
        playback_controls.addWidget(self.replay_button)
        playback_controls.addWidget(QLabel("倍速"))
        playback_controls.addWidget(self.rate_combo)
        playback_controls.addWidget(self.progress_slider)
        playback_layout.addWidget(self.video_label)
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
        self.analyze_button = QPushButton("开始分析")
        self.analyze_button.clicked.connect(self.run_analysis)
        go_button = QPushButton("手动标记 go")
        go_button.clicked.connect(self.mark_go)
        self.report_button = QPushButton("查看分析报告")
        self.report_button.clicked.connect(self.open_report)
        self.learning_button = QPushButton("打开学习系统")
        self.learning_button.clicked.connect(self.open_learning_system)
        self.status_label = QLabel("状态：等待导入视频")
        for widget in [self.analyze_button, go_button, self.report_button, self.learning_button, self.status_label]:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)

        params_group = QGroupBox("6️⃣ 当前参数设置")
        params_layout = QVBoxLayout(params_group)
        form = QFormLayout()
        self.stability_spin = QDoubleSpinBox()
        self.stability_spin.setRange(0.1, 5.0)
        self.stability_spin.setDecimals(2)
        self.stability_spin.setSuffix(" 秒")
        form.addRow("弓步稳定性：", self.stability_spin)

        self.thigh_angle_spin = QDoubleSpinBox()
        self.thigh_angle_spin.setRange(1.0, 90.0)
        self.thigh_angle_spin.setDecimals(1)
        self.thigh_angle_spin.setSuffix(" 度")
        form.addRow("抬大腿角度阈值：", self.thigh_angle_spin)

        self.head_tilt_spin = QDoubleSpinBox()
        self.head_tilt_spin.setRange(1.0, 45.0)
        self.head_tilt_spin.setDecimals(1)
        self.head_tilt_spin.setSuffix(" 度")
        form.addRow("头部偏斜阈值：", self.head_tilt_spin)

        self.knee_below_hip_checkbox = QCheckBox("膝关节点需低于髋关节点")
        form.addRow("膝髋位置规则：", self.knee_below_hip_checkbox)

        params_layout.addLayout(form)
        params_layout.addWidget(QLabel("稳定性规则：手 / 肘 / 肩在设定时间内晃动不超过阈值"))
        params_layout.addWidget(QLabel("抬大腿规则：膝-髋连线与地面夹角 >= 阈值，且可选要求膝低于髋"))
        button_row = QHBoxLayout()
        apply_button = QPushButton("应用参数")
        apply_button.clicked.connect(self.apply_parameters)
        reset_button = QPushButton("恢复默认参数")
        reset_button.clicked.connect(self.reset_parameters)
        button_row.addWidget(apply_button)
        button_row.addWidget(reset_button)
        params_layout.addLayout(button_row)
        params_layout.addStretch(1)

        layout.addWidget(input_group, 0, 0, 1, 2)
        layout.addWidget(playback_group, 1, 0, 4, 2)
        layout.addWidget(realtime_group, 1, 2)
        layout.addWidget(algorithm_group, 2, 2)
        layout.addWidget(control_group, 3, 2)
        layout.addWidget(params_group, 4, 2)
        self.setCentralWidget(root)
        self._update_realtime_placeholder()
        self._update_algorithm_panel(None)

    def _load_config_to_widgets(self) -> None:
        self.stability_spin.setValue(self.config.stability_seconds_threshold)
        self.thigh_angle_spin.setValue(self.config.thigh_raise_angle_threshold)
        self.head_tilt_spin.setValue(self.config.head_tilt_angle_threshold)
        self.knee_below_hip_checkbox.setChecked(self.config.require_knee_below_hip_for_thigh_raise)

    def _update_realtime_placeholder(self) -> None:
        self.realtime_text.setText(
            "当前第 1 个弓步\n"
            "已完成弓步数：0\n"
            "当前状态：等待分析\n"
            "当前阶段：IDLE\n"
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
        self.algorithm_text.setText(
            f"动作识别：MediaPipe Pose\n"
            f"计数规则：打出 + 收回 = 1 个完整弓步\n"
            f"稳定性规则：手 / 肘 / 肩在 {self.config.stability_seconds_threshold:.2f} 秒内晃动不超阈值\n"
            f"抬大腿规则：膝-髋连线与地面夹角 >= {self.config.thigh_raise_angle_threshold:.1f} 度\n"
            f"头部规则：头部偏斜角 > {self.config.head_tilt_angle_threshold:.1f} 度触发提醒\n"
            f"膝髋位置规则：{self.config.require_knee_below_hip_for_thigh_raise}\n"
            f"当前状态机：{lunge_state}"
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
        self.video_path = Path(file_path)
        self.path_edit.setText(file_path)
        self.manual_go_time = None
        self.analysis_result = None
        self._open_video()
        self.status_label.setText("状态：已导入视频，可播放或开始分析")

    def _open_video(self) -> None:
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(self.video_path))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.duration = frame_count / fps if fps else 0.0
        self.progress_slider.setMaximum(max(0, frame_count - 1))
        self.current_frame = 0
        self.is_video_loading = True
        self.play_button.setEnabled(False)
        self._show_progress("视频加载中", 0)

        def update_progress(percent: int) -> None:
            self.video_load_progress = percent
            self._show_progress("视频加载中", percent)

        self.preview_sequence = self.preview_estimator.extract(self.video_path, progress_callback=update_progress)
        self.is_video_loading = False
        self.play_button.setEnabled(True)
        self._show_progress("视频加载完成", 100, status="success")
        logger.info("Video loaded: %s frames, duration %.2fs", frame_count, self.duration)
        self._show_frame(0)

    def toggle_play(self) -> None:
        if self.capture is None or self.is_video_loading:
            return
        if self.current_frame >= self.progress_slider.maximum():
            self.replay()
        self.is_playing = True
        self._restart_timer()
        logger.info("Playback started at frame=%s rate=%s", self.current_frame, self.playback_rate)

    def pause(self) -> None:
        self.timer.stop()
        self.is_playing = False
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
        if self.capture is None:
            return
        frame_index = self.progress_slider.value()
        logger.info("Seek to frame=%s", frame_index)
        self._show_frame(frame_index)
        if getattr(self, "_was_playing_before_seek", False):
            self.toggle_play()

    def _next_frame(self) -> None:
        next_frame = self.current_frame + max(1, int(round(self.playback_rate)))
        if next_frame > self.progress_slider.maximum():
            self.pause()
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
            return
        self.current_frame = frame_index
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_time = frame_index / fps
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(frame_index)
        self.progress_slider.blockSignals(False)

        if self.analysis_result is not None and self.analysis_result.frame_assessments:
            assessment = self.analysis_result.frame_assessments[min(frame_index, len(self.analysis_result.frame_assessments) - 1)]
        else:
            preview_eval = evaluate_sequence(self.preview_sequence, self.config, self.manual_go_time)
            assessment = preview_eval.frame_assessments[min(frame_index, len(preview_eval.frame_assessments) - 1)]
        annotated = draw_pose_overlay(frame, self.preview_sequence, frame_index, assessment)
        self._display_image(annotated)
        self._update_realtime_panel(assessment)
        self._update_algorithm_panel(assessment)

    def _display_image(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.video_label.setPixmap(pixmap)

    def _update_realtime_panel(self, assessment) -> None:
        self.realtime_text.setText(
            f"当前第 {assessment.current_lunge_index} 个弓步\n"
            f"已完成弓步数：{assessment.completed_lunge_count}\n"
            f"当前状态：{assessment.current_text}\n"
            f"当前阶段：{assessment.lunge_state}\n"
            f"是否得分：{assessment.score_text}\n"
            f"本次最需要改进：{assessment.top_issue}\n"
            f"训练建议：{assessment.coaching_advice}\n"
            f"细项判定：\n"
            f"抬大腿：{assessment.thigh_raise}\n"
            f"稳定性：{assessment.stability}\n"
            f"手脚顺序：{assessment.hand_foot_order}\n"
            f"头部姿态：{assessment.head_tilt}\n"
            f"稳定性细节：{assessment.stability_details}"
        )

    def mark_go(self) -> None:
        if self.capture is None:
            QMessageBox.information(self, "提示", "请先导入视频。")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.manual_go_time = self.current_frame / fps
        logger.info("Manual go marked at %.3fs", self.manual_go_time)
        self.status_label.setText(f"状态：已手动标记 go = {self.manual_go_time:.3f} 秒")
        self._show_frame(self.current_frame)

    def apply_parameters(self) -> None:
        self.config.stability_seconds_threshold = self.stability_spin.value()
        self.config.thigh_raise_angle_threshold = self.thigh_angle_spin.value()
        self.config.head_tilt_angle_threshold = self.head_tilt_spin.value()
        self.config.require_knee_below_hip_for_thigh_raise = self.knee_below_hip_checkbox.isChecked()
        logger.info(
            "Parameters applied: stability=%.2f thigh=%.1f head=%.1f knee_rule=%s",
            self.config.stability_seconds_threshold,
            self.config.thigh_raise_angle_threshold,
            self.config.head_tilt_angle_threshold,
            self.config.require_knee_below_hip_for_thigh_raise,
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
        self._load_config_to_widgets()
        self.status_label.setText("状态：已恢复默认参数")
        if self.preview_sequence is not None:
            self.analysis_result = None
            self._show_frame(self.current_frame)

    def run_analysis(self) -> None:
        if self.video_path is None:
            QMessageBox.warning(self, "提示", "请先导入视频。")
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
        self.engine.close()
        self.engine = AnalysisEngine(self.config, quality_standard=self.quality_standard)
        self._load_config_to_widgets()
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
