from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QPushButton, QWidget


class ControlPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QHBoxLayout(self)
        self.upload_btn = QPushButton("上传视频")
        self.play_btn = QPushButton("播放/暂停")
        self.replay_btn = QPushButton("重播")
        self.export_btn = QPushButton("导出报告")
        self.settings_btn = QPushButton("参数设置")
        for btn in [self.upload_btn, self.play_btn, self.replay_btn, self.export_btn, self.settings_btn]:
            layout.addWidget(btn)
