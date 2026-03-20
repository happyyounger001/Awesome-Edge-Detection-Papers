from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
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

from learning_system import build_quality_standard_from_sources
from quality_standard import load_quality_standard


class LearningSystemWindow(QMainWindow):
    def __init__(self, import_callback: Callable[[dict, Path], None] | None = None):
        super().__init__()
        self.import_callback = import_callback
        self.current_standard: dict | None = None
        self.current_standard_path: Path | None = None
        self.setWindowTitle("Learning System - 质量标准学习")
        self.resize(1280, 860)
        self._build_ui()

    def _build_ui(self) -> None:
        root = QWidget()
        layout = QVBoxLayout(root)

        source_group = QGroupBox("学习样本输入")
        source_layout = QGridLayout(source_group)
        self.input_dir_edit = QLineEdit("data/learning_samples")
        self.output_path_edit = QLineEdit("standards/quality_standard_u6_foil_v1.json")
        choose_input = QPushButton("选择样本目录")
        choose_input.clicked.connect(self.select_input_dir)
        choose_output = QPushButton("选择输出文件")
        choose_output.clicked.connect(self.select_output_file)
        build_button = QPushButton("开始学习并生成标准")
        build_button.clicked.connect(self.build_standard)
        load_button = QPushButton("加载已有标准")
        load_button.clicked.connect(self.load_existing_standard)
        import_button = QPushButton("一键导入到 Training Assistant System")
        import_button.clicked.connect(self.import_to_training_assistant)
        source_layout.addWidget(QLabel("样本目录："), 0, 0)
        source_layout.addWidget(self.input_dir_edit, 0, 1)
        source_layout.addWidget(choose_input, 0, 2)
        source_layout.addWidget(QLabel("标准输出："), 1, 0)
        source_layout.addWidget(self.output_path_edit, 1, 1)
        source_layout.addWidget(choose_output, 1, 2)
        source_layout.addWidget(build_button, 2, 0)
        source_layout.addWidget(load_button, 2, 1)
        source_layout.addWidget(import_button, 2, 2)

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
        layout.addWidget(meta_group)
        layout.addWidget(table_group)
        self.setCentralWidget(root)

    def select_input_dir(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择学习样本目录", self.input_dir_edit.text())
        if directory:
            self.input_dir_edit.setText(directory)

    def select_output_file(self) -> None:
        path, _ = QFileDialog.getSaveFileName(self, "选择标准输出文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if path:
            self.output_path_edit.setText(path)

    def build_standard(self) -> None:
        input_dir = Path(self.input_dir_edit.text())
        output_path = Path(self.output_path_edit.text())
        try:
            standard, feature_rows = build_quality_standard_from_sources(
                input_dir=input_dir,
                manifest_path=None,
                output_path=output_path,
                features_output_dir=Path("outputs/learning_features"),
            )
        except Exception as exc:
            QMessageBox.critical(self, "学习失败", str(exc))
            return
        self.current_standard = standard
        self.current_standard_path = output_path
        self._render_standard(standard, sample_count=len(feature_rows))
        QMessageBox.information(self, "完成", f"质量标准已生成：{output_path}")

    def load_existing_standard(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择质量标准文件", self.output_path_edit.text(), "JSON Files (*.json)")
        if not path:
            return
        standard = load_quality_standard(path)
        self.current_standard = standard
        self.current_standard_path = Path(path)
        self.output_path_edit.setText(path)
        self._render_standard(standard)

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
            f"样本数：{sample_count if sample_count is not None else '已加载标准'}\n"
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
