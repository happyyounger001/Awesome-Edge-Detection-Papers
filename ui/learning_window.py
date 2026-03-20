from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import cv2
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSlider,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from config import load_config
from feature_extractor import extract_video_features
from learning_system import build_quality_standard_from_feature_rows
from pose.estimator import PoseEstimator
from quality_standard import load_quality_standard
from rules.evaluator import evaluate_sequence
from ui.video_overlay import render_overlay_video


class LearningSystemWindow(QMainWindow):
    def __init__(self, import_callback: Callable[[dict, Path], None] | None = None):
        super().__init__()
        self.import_callback = import_callback
        self.current_standard: dict | None = None
        self.current_standard_path: Path | None = None
        self.config = load_config("config.yaml")
        self.estimator = PoseEstimator(self.config)
        self.feature_rows: list[dict] = []
        self.samples: list[dict] = []
        self.current_sample: dict | None = None
        self.capture: cv2.VideoCapture | None = None
        self.current_frame_index = 0
        self.playback_rate = 1.0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._next_frame)
        self.setWindowTitle("Learning System - 样本分析模式")
        self.resize(1520, 980)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QGridLayout(root)

        source_group = QGroupBox("学习样本输入")
        source_layout = QGridLayout(source_group)
        self.output_path_edit = QLineEdit("standards/quality_standard_u6_foil_v1.json")
        choose_output = QPushButton("选择标准输出文件")
        choose_output.clicked.connect(self.select_output_file)
        self.training_checkbox = QCheckBox("训练视频")
        self.training_checkbox.setChecked(True)
        self.competition_checkbox = QCheckBox("比赛视频")
        self.training_checkbox.toggled.connect(lambda checked: self._on_category_checked("training", checked))
        self.competition_checkbox.toggled.connect(lambda checked: self._on_category_checked("competition", checked))
        import_button = QPushButton("导入视频")
        import_button.clicked.connect(self.select_videos)
        analyze_button = QPushButton("分析样本")
        analyze_button.clicked.connect(self.analyze_selected_sample)
        load_button = QPushButton("加载已有标准")
        load_button.clicked.connect(self.load_existing_standard)
        apply_button = QPushButton("应用为当前标准")
        apply_button.clicked.connect(self.import_to_training_assistant)
        source_layout.addWidget(QLabel("学习分类："), 0, 0)
        category_row = QHBoxLayout()
        category_row.addWidget(self.training_checkbox)
        category_row.addWidget(self.competition_checkbox)
        category_row.addStretch(1)
        source_layout.addLayout(category_row, 0, 1, 1, 3)
        source_layout.addWidget(QLabel("标准输出："), 1, 0)
        source_layout.addWidget(self.output_path_edit, 1, 1, 1, 2)
        source_layout.addWidget(choose_output, 1, 3)
        source_layout.addWidget(import_button, 2, 0)
        source_layout.addWidget(analyze_button, 2, 1)
        source_layout.addWidget(load_button, 2, 2)
        source_layout.addWidget(apply_button, 2, 3)

        queue_group = QGroupBox("样本队列（默认不进入学习库）")
        queue_layout = QVBoxLayout(queue_group)
        self.sample_list = QListWidget()
        self.sample_list.currentRowChanged.connect(self._load_selected_sample_analysis)
        queue_layout.addWidget(self.sample_list)

        player_group = QGroupBox("样本分析模式 A - 骨架叠加视频")
        player_layout = QVBoxLayout(player_group)
        self.video_label = QLabel("请先选择并分析样本视频")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(880, 480)
        self.video_label.setStyleSheet("background:#111;color:#fff;border-radius:8px;")
        controls = QHBoxLayout()
        play_button = QPushButton("播放")
        play_button.clicked.connect(self.play_video)
        pause_button = QPushButton("暂停")
        pause_button.clicked.connect(self.pause_video)
        replay_button = QPushButton("重播")
        replay_button.clicked.connect(self.replay_video)
        self.rate_combo = QComboBox()
        self.rate_combo.addItems(["0.5x", "1.0x", "2.0x"])
        self.rate_combo.setCurrentText("1.0x")
        self.rate_combo.currentTextChanged.connect(self._change_playback_rate)
        self.progress_slider = QSlider(Qt.Horizontal)
        self.progress_slider.sliderReleased.connect(self.seek_video)
        controls.addWidget(play_button)
        controls.addWidget(pause_button)
        controls.addWidget(replay_button)
        controls.addWidget(QLabel("倍速"))
        controls.addWidget(self.rate_combo)
        controls.addWidget(self.progress_slider)
        player_layout.addWidget(self.video_label)
        player_layout.addLayout(controls)

        status_group = QGroupBox("实时状态信息")
        status_layout = QVBoxLayout(status_group)
        self.status_text = QTextEdit()
        self.status_text.setReadOnly(True)
        status_layout.addWidget(self.status_text)

        feature_group = QGroupBox("特征展示区")
        feature_layout = QGridLayout(feature_group)
        self.frame_feature_text = QTextEdit()
        self.frame_feature_text.setReadOnly(True)
        self.lunge_feature_text = QTextEdit()
        self.lunge_feature_text.setReadOnly(True)
        feature_layout.addWidget(QLabel("当前帧特征"), 0, 0)
        feature_layout.addWidget(QLabel("当前弓步整体特征"), 0, 1)
        feature_layout.addWidget(self.frame_feature_text, 1, 0)
        feature_layout.addWidget(self.lunge_feature_text, 1, 1)

        confirm_group = QGroupBox("样本确认区")
        confirm_layout = QHBoxLayout(confirm_group)
        include_button = QPushButton("纳入学习库")
        include_button.clicked.connect(self.include_current_sample)
        reject_button = QPushButton("不纳入学习")
        reject_button.clicked.connect(self.reject_current_sample)
        confirm_layout.addWidget(include_button)
        confirm_layout.addWidget(reject_button)

        meta_group = QGroupBox("学习结果概览")
        meta_layout = QVBoxLayout(meta_group)
        self.meta_text = QTextEdit()
        self.meta_text.setReadOnly(True)
        meta_layout.addWidget(self.meta_text)

        table_group = QGroupBox("学习后的质量标准衡量指标")
        table_layout = QVBoxLayout(table_group)
        self.metrics_table = QTableWidget(0, 6)
        self.metrics_table.setHorizontalHeaderLabels(["指标", "mean", "median", "p25", "p75", "p90"])
        self.metrics_table.horizontalHeader().setStretchLastSection(True)
        table_layout.addWidget(self.metrics_table)

        layout.addWidget(source_group, 0, 0, 1, 2)
        layout.addWidget(queue_group, 1, 0, 3, 1)
        layout.addWidget(player_group, 1, 1, 2, 2)
        layout.addWidget(status_group, 3, 1)
        layout.addWidget(feature_group, 3, 2)
        layout.addWidget(confirm_group, 4, 0, 1, 3)
        layout.addWidget(meta_group, 5, 0, 1, 3)
        layout.addWidget(table_group, 6, 0, 1, 3)
        self.setCentralWidget(root)
        self._render_meta()

    def closeEvent(self, event) -> None:
        self.timer.stop()
        if self.capture is not None:
            self.capture.release()
        self.estimator.close()
        super().closeEvent(event)

    def select_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择标准输出文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if path:
            self.output_path_edit.setText(path)

    def _on_category_checked(self, category: str, checked: bool) -> None:
        if not checked:
            if not self.training_checkbox.isChecked() and not self.competition_checkbox.isChecked():
                (self.training_checkbox if category == "training" else self.competition_checkbox).setChecked(True)
            return
        other = self.competition_checkbox if category == "training" else self.training_checkbox
        other.blockSignals(True)
        other.setChecked(False)
        other.blockSignals(False)

    def _selected_category(self) -> str:
        if self.training_checkbox.isChecked():
            return "training_standard"
        if self.competition_checkbox.isChecked():
            return "competition_effective"
        raise RuntimeError("请先勾选训练视频或比赛视频分类。")

    def select_videos(self) -> None:
        category = self._selected_category()
        paths, _ = QFileDialog.getOpenFileNames(self, "选择学习视频", "", "Video Files (*.mp4 *.mov *.avi)")
        if not paths:
            return
        for path in paths:
            if any(sample["video_path"] == path for sample in self.samples):
                continue
            sample = {"video_path": path, "category": category, "status": "pending", "feature_row": None, "analysis": None}
            self.samples.append(sample)
            self._append_sample_item(sample)
        self._render_meta()

    def _append_sample_item(self, sample: dict) -> None:
        item = QListWidgetItem(self._sample_label(sample))
        item.setToolTip(sample["video_path"])
        self.sample_list.addItem(item)

    def _sample_label(self, sample: dict) -> str:
        category_text = "训练" if sample["category"] == "training_standard" else "比赛"
        status_text = {
            "pending": "待分析",
            "analyzed": "待确认",
            "included": "已纳入",
            "rejected": "已丢弃",
        }.get(sample["status"], sample["status"])
        return f"[{category_text}][{status_text}] {Path(sample['video_path']).name}"

    def _refresh_sample_list(self) -> None:
        self.sample_list.blockSignals(True)
        self.sample_list.clear()
        for sample in self.samples:
            self._append_sample_item(sample)
        self.sample_list.blockSignals(False)

    def analyze_selected_sample(self) -> None:
        sample = self._selected_sample()
        if sample is None:
            QMessageBox.information(self, "提示", "请先在样本队列中选择一个视频。")
            return
        try:
            self._analyze_sample(sample)
        except Exception as exc:
            QMessageBox.critical(self, "分析失败", str(exc))
            return
        self._refresh_sample_list()
        self._load_sample_into_player(sample)
        self._render_meta()

    def _analyze_sample(self, sample: dict) -> None:
        sample_path = Path(sample["video_path"])
        output_dir = Path("outputs") / f"sample_{sample_path.stem}"
        output_dir.mkdir(parents=True, exist_ok=True)
        sequence = self.estimator.extract(sample_path)
        evaluation = evaluate_sequence(sequence, self.config, manual_go_time=None)
        feature_row = extract_video_features(sequence, self.config, str(sample_path), sample["category"])
        overlay_path, _ = render_overlay_video(sample_path, output_dir, sequence, evaluation)
        features_path = output_dir / "features.json"
        lunge_segments_path = output_dir / "lunge_segments.json"
        metrics = evaluation.metrics[["current_lunge_index", "lunge_state"]].to_dict(orient="records")
        features_path.write_text(json.dumps(feature_row, ensure_ascii=False, indent=2), encoding="utf-8")
        lunge_segments_path.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        sample["analysis"] = {
            "overlay_path": overlay_path,
            "output_dir": output_dir,
            "evaluation": evaluation,
            "feature_row": feature_row,
            "features_path": features_path,
            "lunge_segments_path": lunge_segments_path,
        }
        sample["feature_row"] = feature_row
        if sample["status"] == "pending":
            sample["status"] = "analyzed"

    def _selected_sample(self) -> dict | None:
        row = self.sample_list.currentRow()
        if row < 0 or row >= len(self.samples):
            return None
        return self.samples[row]

    def _load_selected_sample_analysis(self, row: int) -> None:
        if row < 0 or row >= len(self.samples):
            return
        sample = self.samples[row]
        if sample.get("analysis") is None:
            self.status_text.setText("请点击“分析样本”生成骨架叠加视频、阶段划分和特征数据。")
            return
        self._load_sample_into_player(sample)

    def _load_sample_into_player(self, sample: dict) -> None:
        self.current_sample = sample
        analysis = sample["analysis"]
        if self.capture is not None:
            self.capture.release()
        self.capture = cv2.VideoCapture(str(analysis["overlay_path"]))
        frame_count = int(self.capture.get(cv2.CAP_PROP_FRAME_COUNT))
        self.progress_slider.setMaximum(max(0, frame_count - 1))
        self.current_frame_index = 0
        self._update_lunge_feature_panel(analysis["feature_row"])
        self._show_frame(0)

    def play_video(self) -> None:
        if self.capture is None:
            return
        fps = self.capture.get(cv2.CAP_PROP_FPS) or 20.0
        self.timer.start(max(1, int(1000 / (fps * self.playback_rate))))

    def pause_video(self) -> None:
        self.timer.stop()

    def replay_video(self) -> None:
        if self.capture is None:
            return
        self.pause_video()
        self._show_frame(0)
        self.play_video()

    def _change_playback_rate(self, text: str) -> None:
        self.playback_rate = float(text.replace("x", ""))
        if self.timer.isActive():
            self.play_video()

    def seek_video(self) -> None:
        self._show_frame(self.progress_slider.value())

    def _next_frame(self) -> None:
        if self.capture is None:
            return
        next_frame = self.current_frame_index + 1
        if next_frame > self.progress_slider.maximum():
            self.pause_video()
            return
        self._show_frame(next_frame)

    def _show_frame(self, frame_index: int) -> None:
        if self.capture is None or self.current_sample is None:
            return
        analysis = self.current_sample["analysis"]
        evaluation = analysis["evaluation"]
        frame_index = max(0, min(frame_index, len(evaluation.frame_assessments) - 1))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, frame = self.capture.read()
        if not ok:
            self.pause_video()
            return
        self.current_frame_index = frame_index
        self.progress_slider.blockSignals(True)
        self.progress_slider.setValue(frame_index)
        self.progress_slider.blockSignals(False)
        self._display_image(frame)
        self._update_status_panel(frame_index)
        self._update_frame_feature_panel(frame_index)

    def _display_image(self, frame) -> None:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.video_label.setPixmap(pixmap)

    def _update_status_panel(self, frame_index: int) -> None:
        if self.current_sample is None:
            return
        analysis = self.current_sample["analysis"]
        evaluation = analysis["evaluation"]
        assessment = evaluation.frame_assessments[frame_index]
        fps = (self.capture.get(cv2.CAP_PROP_FPS) or 20.0) if self.capture is not None else 20.0
        self.status_text.setText(
            f"当前帧：{frame_index}\n"
            f"当前时间：{frame_index / fps:.2f}s\n"
            f"当前弓步：第 {assessment.current_lunge_index} 个\n"
            f"当前阶段：{assessment.lunge_state}\n"
            f"当前状态：{assessment.current_text}\n"
            f"样本状态：{self._sample_label(self.current_sample)}"
        )

    def _update_frame_feature_panel(self, frame_index: int) -> None:
        if self.current_sample is None:
            return
        metrics = self.current_sample["analysis"]["evaluation"].metrics.iloc[frame_index]
        self.frame_feature_text.setText(
            f"前膝角：{metrics['front_knee_angle']:.2f}°\n"
            f"头部倾斜：{metrics['head_tilt_angle']:.2f}°\n"
            f"手腕位置：{metrics['wrist_forward_dist_norm']:.3f}\n"
            f"步幅归一化：{metrics['stride_length_norm']:.3f}\n"
            f"躯干前倾：{metrics['trunk_lean_angle']:.2f}°"
        )

    def _update_lunge_feature_panel(self, feature_row: dict) -> None:
        self.lunge_feature_text.setText(
            f"最大步幅：{feature_row.get('stride_max_norm', 0.0):.3f}\n"
            f"反应时间（手启动）：{feature_row.get('hand_start_time', 0.0):.3f}s\n"
            f"稳定时长：{feature_row.get('stable_hold_duration', 0.0):.3f}s\n"
            f"手脚时差：{feature_row.get('hand_foot_start_delta', 0.0):.3f}s\n"
            f"头部倾斜最大值：{feature_row.get('head_tilt_max_deg', 0.0):.2f}°"
        )

    def include_current_sample(self) -> None:
        sample = self.current_sample or self._selected_sample()
        if sample is None or sample.get("analysis") is None:
            QMessageBox.information(self, "提示", "请先完成样本分析。")
            return
        if sample["status"] == "included":
            QMessageBox.information(self, "提示", "该样本已经纳入学习库。")
            return
        sample["status"] = "included"
        feature_row = sample["feature_row"]
        self.feature_rows = [row for row in self.feature_rows if row["video_path"] != feature_row["video_path"]]
        self.feature_rows.append(feature_row)
        self._rebuild_standard()
        self._refresh_sample_list()
        QMessageBox.information(self, "完成", "样本已纳入学习库并更新质量标准。")

    def reject_current_sample(self) -> None:
        sample = self.current_sample or self._selected_sample()
        if sample is None:
            QMessageBox.information(self, "提示", "请先选择样本。")
            return
        sample["status"] = "rejected"
        if sample.get("feature_row") is not None:
            self.feature_rows = [row for row in self.feature_rows if row["video_path"] != sample["feature_row"]["video_path"]]
            self._rebuild_standard(show_message=False)
        self._refresh_sample_list()
        self._render_meta()

    def _rebuild_standard(self, show_message: bool = True) -> None:
        if not self.feature_rows:
            self.current_standard = None
            self.current_standard_path = None
            self._render_meta()
            return
        output_path = Path(self.output_path_edit.text())
        standard = build_quality_standard_from_feature_rows(
            feature_rows=self.feature_rows,
            output_path=output_path,
            features_output_dir=Path("outputs/learning_features"),
        )
        self.current_standard = standard
        self.current_standard_path = output_path
        self._render_standard(standard, sample_count=len(self.feature_rows))
        if show_message:
            QMessageBox.information(self, "完成", f"当前学习库已更新，共 {len(self.feature_rows)} 个已确认样本。")

    def load_existing_standard(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择质量标准文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if not path:
            return
        standard = load_quality_standard(path)
        self.current_standard = standard
        self.current_standard_path = Path(path)
        self.output_path_edit.setText(path)
        self._render_standard(standard, sample_count=standard.get("meta", {}).get("sample_count"))

    def import_to_training_assistant(self) -> None:
        if self.current_standard is None or self.current_standard_path is None:
            QMessageBox.information(self, "提示", "学习未完成，禁止导入。请先确认至少一个样本。")
            return
        if self.import_callback is not None:
            self.import_callback(self.current_standard, self.current_standard_path)
        QMessageBox.information(self, "导入完成", "当前学习结果已应用为训练系统判断标准。")

    def _render_meta(self) -> None:
        included_count = sum(1 for sample in self.samples if sample["status"] == "included")
        analyzed_count = sum(1 for sample in self.samples if sample["status"] in {"analyzed", "included", "rejected"})
        pending_count = sum(1 for sample in self.samples if sample["status"] == "pending")
        self.meta_text.setText(
            f"当前模式：样本分析模式（模式A）\n"
            f"待分析样本数：{pending_count}\n"
            f"已分析样本数：{analyzed_count}\n"
            f"已确认纳入样本数：{included_count}\n"
            f"当前学习库样本数：{len(self.feature_rows)}\n"
            f"提示：默认所有样本都不会进入学习库，只有点击“纳入学习库”才会更新标准。"
        )

    def _render_standard(self, standard: dict, sample_count: int | None = None) -> None:
        meta = standard.get("meta", {})
        thresholds = standard.get("thresholds", {})
        self.meta_text.setText(
            f"名称：{meta.get('name', '--')}\n"
            f"版本：{meta.get('version', '--')}\n"
            f"来源类别：{', '.join(meta.get('source_categories', []))}\n"
            f"累计纳入样本数：{sample_count if sample_count is not None else '已加载标准'}\n"
            f"待分析样本数：{sum(1 for sample in self.samples if sample['status'] == 'pending')}\n"
            f"稳定窗口：{thresholds.get('stability_window_sec', '--')} 秒\n"
            f"头部偏斜阈值：{thresholds.get('head_tilt_deg_thr', '--')} 度\n"
            f"髋膝线角度阈值：{thresholds.get('hip_knee_ground_angle_thr', '--')} 度"
        )
        features = standard.get("features", {})
        self.metrics_table.setRowCount(len(features))
        for row_index, (name, stats) in enumerate(features.items()):
            values = [name, stats.get("mean", "--"), stats.get("median", "--"), stats.get("p25", "--"), stats.get("p75", "--"), stats.get("p90", "--")]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, (int, float)) and column_index > 0 else str(value))
                if column_index == 0:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.metrics_table.setItem(row_index, column_index, item)
