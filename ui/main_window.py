from __future__ import annotations

from pathlib import Path

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
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
from rules.evaluator import evaluate_sequence
from ui.report_window import ReportWindow
from ui.video_overlay import draw_pose_overlay


class FencingMainWindow(QMainWindow):
    def __init__(self, config_path: str | None = None):
        super().__init__()
        self.config = load_config(config_path)
        self.default_config = load_config(config_path)
        self.engine = AnalysisEngine(self.config)
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

        self.setWindowTitle("Fencing Lunge AI Trainer")
        self.resize(1520, 920)
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
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderReleased.connect(self.seek_video)
        playback_controls.addWidget(self.play_button)
        playback_controls.addWidget(self.pause_button)
        playback_controls.addWidget(self.progress_slider)
        playback_layout.addWidget(self.video_label)
        playback_layout.addLayout(playback_controls)

        realtime_group = QGroupBox("3️⃣ 实时分析区")
        realtime_layout = QVBoxLayout(realtime_group)
        self.realtime_text = QTextEdit()
        self.realtime_text.setReadOnly(True)
        realtime_layout.addWidget(self.realtime_text)

        control_group = QGroupBox("4️⃣ 控制与报告区")
        control_layout = QVBoxLayout(control_group)
        analyze_button = QPushButton("开始分析")
        analyze_button.clicked.connect(self.run_analysis)
        go_button = QPushButton("手动标记 go")
        go_button.clicked.connect(self.mark_go)
        report_button = QPushButton("查看分析报告")
        report_button.clicked.connect(self.open_report)
        self.status_label = QLabel("状态：等待导入视频")
        for widget in [analyze_button, go_button, report_button, self.status_label]:
            control_layout.addWidget(widget)
        control_layout.addStretch(1)

        params_group = QGroupBox("5️⃣ 当前参数设置")
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

        self.knee_below_hip_checkbox = QCheckBox("膝关节点需低于髋关节点")
        form.addRow("抬大腿条件 1：", self.knee_below_hip_checkbox)

        params_layout.addLayout(form)
        params_layout.addWidget(QLabel("抬大腿条件 2：膝-髋连线与地面夹角 >= 上述阈值"))
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
        layout.addWidget(playback_group, 1, 0, 3, 2)
        layout.addWidget(realtime_group, 1, 2)
        layout.addWidget(control_group, 2, 2)
        layout.addWidget(params_group, 3, 2)
        self.setCentralWidget(root)
        self._update_realtime_placeholder()

    def _load_config_to_widgets(self) -> None:
        self.stability_spin.setValue(self.config.stability_seconds_threshold)
        self.thigh_angle_spin.setValue(self.config.thigh_raise_angle_threshold)
        self.knee_below_hip_checkbox.setChecked(self.config.require_knee_below_hip_for_thigh_raise)

    def _update_realtime_placeholder(self) -> None:
        self.realtime_text.setText(
            "第 1 个弓步\n"
            "当前状态：等待分析\n"
            "是否得分：--\n"
            "细项判定：\n"
            "抬大腿：--\n"
            "稳定性：--\n"
            "手脚顺序：--"
        )

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
        self.progress_slider.setMaximum(max(0, frame_count - 1))
        self.current_frame = 0
        self.preview_sequence = self.preview_estimator.extract(self.video_path)
        self._show_frame(0)

    def toggle_play(self) -> None:
        if self.capture is None:
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.timer.start(int(1000 / fps))

    def pause(self) -> None:
        self.timer.stop()

    def seek_video(self) -> None:
        if self.capture is None:
            return
        frame_index = self.progress_slider.value()
        self._show_frame(frame_index)

    def _next_frame(self) -> None:
        self._show_frame(self.current_frame + 1)

    def _show_frame(self, frame_index: int) -> None:
        if self.capture is None or self.preview_sequence is None:
            return
        total = self.preview_sequence.frame_count
        frame_index = max(0, min(frame_index, total - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.timer.stop()
            return
        self.current_frame = frame_index
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
            f"第 {assessment.lunge_index} 个弓步\n"
            f"当前状态：{assessment.current_text}\n"
            f"是否得分：{assessment.score_text}\n"
            f"细项判定：\n"
            f"抬大腿：{assessment.thigh_raise}\n"
            f"稳定性：{assessment.stability}\n"
            f"手脚顺序：{assessment.hand_foot_order}"
        )

    def mark_go(self) -> None:
        if self.capture is None:
            QMessageBox.information(self, "提示", "请先导入视频。")
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 30.0
        self.manual_go_time = self.current_frame / fps
        self.status_label.setText(f"状态：已手动标记 go = {self.manual_go_time:.3f} 秒")
        self._show_frame(self.current_frame)

    def apply_parameters(self) -> None:
        self.config.stability_seconds_threshold = self.stability_spin.value()
        self.config.thigh_raise_angle_threshold = self.thigh_angle_spin.value()
        self.config.require_knee_below_hip_for_thigh_raise = self.knee_below_hip_checkbox.isChecked()
        self.status_label.setText("状态：参数已应用到当前判定逻辑")
        if self.preview_sequence is not None:
            self.analysis_result = None
            self._show_frame(self.current_frame)

    def reset_parameters(self) -> None:
        self.config.stability_seconds_threshold = self.default_config.stability_seconds_threshold
        self.config.thigh_raise_angle_threshold = self.default_config.thigh_raise_angle_threshold
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
        output_dir = Path("outputs") / self.video_path.stem
        self.status_label.setText("状态：分析中，请稍候...")
        self.analysis_result = self.engine.analyze_video(self.video_path, output_dir, self.manual_go_time)
        self.status_label.setText(
            f"状态：分析完成，共识别 {self.analysis_result.frame_assessments[-1].lunge_index} 个弓步，报告已生成：{self.analysis_result.report_html}"
        )
        self._show_frame(self.current_frame)

    def open_report(self) -> None:
        if self.analysis_result is None:
            QMessageBox.information(self, "提示", "请先完成分析。")
            return
        self.report_window = ReportWindow(self.analysis_result.report_html)
        self.report_window.show()
