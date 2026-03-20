from __future__ import annotations
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
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
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from learning_system import build_quality_standard_from_feature_rows
from config import load_config
from feature_extractor import extract_video_features
from pose.estimator import PoseEstimator
from quality_standard import load_quality_standard


class LearningSystemWindow(QMainWindow):
    def __init__(self, import_callback: Callable[[dict, Path], None] | None = None):
        super().__init__()
        self.import_callback = import_callback
        self.current_standard: dict | None = None
        self.current_standard_path: Path | None = None
        self.config = load_config("config.yaml")
        self.estimator = PoseEstimator(self.config)
        self.feature_rows: list[dict] = []
        self.samples: list[dict[str, str]] = []
        self.processed_sample_paths: set[str] = set()
        self.setWindowTitle("Learning System - 质量标准学习")
        self.resize(1280, 860)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        source_group = QGroupBox("学习样本输入")
        source_layout = QGridLayout(source_group)
        self.output_path_edit = QLineEdit("standards/quality_standard_u6_foil_v1.json")
        choose_output = QPushButton("选择输出文件")
        choose_output.clicked.connect(self.select_output_file)
        self.training_checkbox = QCheckBox("训练视频")
        self.training_checkbox.setChecked(True)
        self.competition_checkbox = QCheckBox("比赛视频")
        self.training_checkbox.toggled.connect(lambda checked: self._on_category_checked("training", checked))
        self.competition_checkbox.toggled.connect(lambda checked: self._on_category_checked("competition", checked))
        import_button = QPushButton("选择视频")
        import_button.clicked.connect(self.select_videos)
        build_button = QPushButton("导入并开始学习")
        build_button.clicked.connect(self.import_and_learn)
        load_button = QPushButton("加载已有标准")
        load_button.clicked.connect(self.load_existing_standard)
        training_import_button = QPushButton("一键导入数据到训练辅助系统")
        training_import_button.clicked.connect(self.import_to_training_assistant)
        source_layout.addWidget(QLabel("学习分类："), 0, 0)
        category_row = QHBoxLayout()
        category_row.addWidget(self.training_checkbox)
        category_row.addWidget(self.competition_checkbox)
        category_row.addStretch(1)
        source_layout.addLayout(category_row, 0, 1, 1, 2)
        source_layout.addWidget(QLabel("标准输出："), 1, 0)
        source_layout.addWidget(self.output_path_edit, 1, 1)
        source_layout.addWidget(choose_output, 1, 2)
        source_layout.addWidget(import_button, 2, 0)
        source_layout.addWidget(build_button, 2, 1)
        source_layout.addWidget(load_button, 2, 2)
        source_layout.addWidget(training_import_button, 3, 0, 1, 3)

        queue_group = QGroupBox("待学习视频")
        queue_layout = QVBoxLayout(queue_group)
        self.sample_list = QListWidget()
        queue_layout.addWidget(self.sample_list)

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

        layout.addWidget(source_group)
        layout.addWidget(queue_group)
        layout.addWidget(meta_group)
        layout.addWidget(table_group)
        self.setCentralWidget(root)

    def closeEvent(self, event) -> None:
        self.estimator.close()
        super().closeEvent(event)

    def select_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择标准输出文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if path:
            self.output_path_edit.setText(path)

    def _on_category_checked(self, category: str, checked: bool) -> None:
        if not checked:
            if not self.training_checkbox.isChecked() and not self.competition_checkbox.isChecked():
                if category == "training":
                    self.training_checkbox.setChecked(True)
                else:
                    self.competition_checkbox.setChecked(True)
            return
        if category == "training":
            self.competition_checkbox.blockSignals(True)
            self.competition_checkbox.setChecked(False)
            self.competition_checkbox.blockSignals(False)
        else:
            self.training_checkbox.blockSignals(True)
            self.training_checkbox.setChecked(False)
            self.training_checkbox.blockSignals(False)

    def _selected_category(self) -> str:
        if self.training_checkbox.isChecked():
            return "training_standard"
        if self.competition_checkbox.isChecked():
            return "competition_effective"
        raise RuntimeError("请先勾选训练视频或比赛视频分类。")

    def select_videos(self) -> None:
        try:
            category = self._selected_category()
        except RuntimeError as exc:
            QMessageBox.information(self, "提示", str(exc))
            return
        paths, _ = QFileDialog.getOpenFileNames(self, "选择学习视频", "", "Video Files (*.mp4 *.mov *.avi)")
        if not paths:
            return
        for path in paths:
            if any(sample["video_path"] == path for sample in self.samples):
                continue
            sample = {"video_path": path, "category": category}
            self.samples.append(sample)
            label = f"[{'训练' if category == 'training_standard' else '比赛'}] {Path(path).name}"
            item = QListWidgetItem(label)
            item.setToolTip(path)
            self.sample_list.addItem(item)

    def import_and_learn(self) -> None:
        if not self.samples:
            QMessageBox.information(self, "提示", "请先选择需要学习的视频。")
            return
        output_path = Path(self.output_path_edit.text())
        try:
            new_feature_rows = []
            for sample in self.samples:
                if sample["video_path"] in self.processed_sample_paths:
                    continue
                sequence = self.estimator.extract(sample["video_path"])
                feature_row = extract_video_features(sequence, self.config, sample["video_path"], sample["category"])
                new_feature_rows.append(feature_row)
                self.processed_sample_paths.add(sample["video_path"])
            if not new_feature_rows and self.current_standard is not None:
                QMessageBox.information(self, "提示", "当前没有新的学习视频，展示的是累计学习结果。")
                return
            self.feature_rows.extend(new_feature_rows)
            standard = build_quality_standard_from_feature_rows(
                feature_rows=self.feature_rows,
                output_path=output_path,
                features_output_dir=Path("outputs/learning_features"),
            )
        except Exception as exc:
            QMessageBox.critical(self, "学习失败", str(exc))
            return
        self.current_standard = standard
        self.current_standard_path = output_path
        self._render_standard(standard, sample_count=len(self.feature_rows))
        QMessageBox.information(self, "完成", f"已完成累计学习，共整合 {len(self.feature_rows)} 个视频样本。")

    def load_existing_standard(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择质量标准文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if not path:
            return
        standard = load_quality_standard(path)
        self.current_standard = standard
        self.current_standard_path = Path(path)
        self.output_path_edit.setText(path)
        self.feature_rows = []
        self.processed_sample_paths.clear()
        self._render_standard(standard, sample_count=standard.get("meta", {}).get("sample_count"))

    def import_to_training_assistant(self) -> None:
        if self.current_standard is None or self.current_standard_path is None:
            QMessageBox.information(self, "提示", "请先生成或加载质量标准文件。")
            return
        if self.import_callback is not None:
            self.import_callback(self.current_standard, self.current_standard_path)
        QMessageBox.information(self, "导入完成", "质量标准已导入 Training Assistant System。")

    def _render_standard(self, standard: dict, sample_count: int | None = None) -> None:
        meta = standard.get("meta", {})
        thresholds = standard.get("thresholds", {})
        self.meta_text.setText(
            f"名称：{meta.get('name', '--')}\n"
            f"版本：{meta.get('version', '--')}\n"
            f"来源类别：{', '.join(meta.get('source_categories', []))}\n"
            f"累计样本数：{sample_count if sample_count is not None else '已加载标准'}\n"
            f"当前待学习视频数：{len(self.samples)}\n"
            f"稳定窗口：{thresholds.get('stability_window_sec', '--')} 秒\n"
            f"头部偏斜阈值：{thresholds.get('head_tilt_deg_thr', '--')} 度\n"
            f"髋膝线角度阈值：{thresholds.get('hip_knee_ground_angle_thr', '--')} 度"
        )

        features = standard.get("features", {})
        self.metrics_table.setRowCount(len(features))
        for row_index, (name, stats) in enumerate(features.items()):
            values = [
                name,
                stats.get("mean", "--"),
                stats.get("median", "--"),
                stats.get("p25", "--"),
                stats.get("p75", "--"),
                stats.get("p90", "--"),
            ]
            for column_index, value in enumerate(values):
                item = QTableWidgetItem(f"{value:.4f}" if isinstance(value, (int, float)) and column_index > 0 else str(value))
                if column_index == 0:
                    item.setTextAlignment(Qt.AlignLeft | Qt.AlignVCenter)
                self.metrics_table.setItem(row_index, column_index, item)
