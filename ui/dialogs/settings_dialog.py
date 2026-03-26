from __future__ import annotations

from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("参数设置")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("参数设置改为弹窗，不再常驻主界面。"))
