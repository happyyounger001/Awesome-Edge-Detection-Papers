from __future__ import annotations

from PySide6.QtCore import Qt, QRectF
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QDialog, QLabel, QVBoxLayout, QWidget


class CircularProgressWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.percent = 0
        self.setMinimumSize(120, 120)

    def set_percent(self, percent: int) -> None:
        self.percent = max(0, min(100, int(percent)))
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = QRectF(10, 10, self.width() - 20, self.height() - 20)

        painter.setPen(QPen(QColor("#d9e2f2"), 10))
        painter.drawArc(rect, 0, 360 * 16)

        painter.setPen(QPen(QColor("#1f6feb"), 10))
        painter.drawArc(rect, 90 * 16, -int(360 * 16 * self.percent / 100))

        painter.setPen(QColor("#111"))
        font = QFont()
        font.setPointSize(16)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.percent}%")


class ProgressDialog(QDialog):
    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setModal(True)
        self.setFixedSize(220, 220)
        self.setWindowFlags(self.windowFlags() & ~Qt.WindowContextHelpButtonHint)

        layout = QVBoxLayout(self)
        self.title_label = QLabel(title)
        self.title_label.setAlignment(Qt.AlignCenter)
        self.progress_widget = CircularProgressWidget()
        self.status_label = QLabel("准备中")
        self.status_label.setAlignment(Qt.AlignCenter)

        layout.addWidget(self.title_label)
        layout.addWidget(self.progress_widget, alignment=Qt.AlignCenter)
        layout.addWidget(self.status_label)

    def update_progress(self, title: str, percent: int, status: str = "running") -> None:
        self.title_label.setText(title)
        self.progress_widget.set_percent(percent)
        if status == "success":
            self.status_label.setText("已完成")
        elif status == "failed":
            self.status_label.setText("失败")
        else:
            self.status_label.setText(f"{title} {percent}%")
