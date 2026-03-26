from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QVBoxLayout, QWidget


class VideoPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.video_layer = QLabel("VideoLayer")
        self.pose_layer = QLabel("PoseLayer")
        self.status_overlay_layer = QLabel("StatusOverlayLayer")
        self.status_overlay_layer.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        layout.addWidget(self.video_layer)
        layout.addWidget(self.pose_layer)
        layout.addWidget(self.status_overlay_layer)

    def update_overlay(self, result):
        self.status_overlay_layer.setText(
            f"阶段: {result.current_phase_cn} | 计数: {result.lunge_count} | 状态: {result.lunge_state}"
        )
