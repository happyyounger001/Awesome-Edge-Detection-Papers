from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class AnalysisPanel(QWidget):
    def __init__(self):
        super().__init__()
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)

    def update_result(self, result):
        self.text.setText(
            f"当前阶段: {result.current_phase_cn}\n弓步计数: {result.lunge_count}\n状态机: {result.lunge_state}"
        )
