from __future__ import annotations

import cv2
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QLabel, QStackedLayout, QWidget


class VideoPanel(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumSize(960, 540)
        self.video_layer = QLabel("请先导入视频")
        self.video_layer.setAlignment(Qt.AlignCenter)
        self.video_layer.setStyleSheet("background:#111;color:#fff;border-radius:8px;")

        self.pose_layer = QLabel("")
        self.pose_layer.setAlignment(Qt.AlignCenter)
        self.pose_layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)

        self.status_overlay_layer = QLabel("")
        self.status_overlay_layer.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.status_overlay_layer.setAttribute(Qt.WA_TransparentForMouseEvents, True)
        self.status_overlay_layer.setStyleSheet(
            "color:#ffffff;background:rgba(0,0,0,120);padding:8px;border-radius:6px;font-size:14px;"
        )

        stack = QStackedLayout(self)
        stack.setStackingMode(QStackedLayout.StackAll)
        stack.addWidget(self.video_layer)
        stack.addWidget(self.pose_layer)
        stack.addWidget(self.status_overlay_layer)

    def _to_pixmap(self, frame, keep_alpha: bool = False) -> QPixmap:
        if keep_alpha:
            rgba = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGBA)
            h, w, ch = rgba.shape
            image = QImage(rgba.data, w, h, ch * w, QImage.Format_RGBA8888)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            image = QImage(rgb.data, w, h, ch * w, QImage.Format_RGB888)
        return QPixmap.fromImage(image).scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)

    def update_layers(self, video_frame, pose_layer, status_lines: list[str]) -> None:
        self.video_layer.setPixmap(self._to_pixmap(video_frame, keep_alpha=False))
        self.pose_layer.setPixmap(self._to_pixmap(pose_layer, keep_alpha=True))
        self.status_overlay_layer.setText("\n".join(status_lines))
