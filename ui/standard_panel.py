from __future__ import annotations

from PySide6.QtWidgets import QTextEdit, QVBoxLayout, QWidget


class StandardPanel(QWidget):
    def __init__(self, profile):
        super().__init__()
        self.profile = profile
        layout = QVBoxLayout(self)
        self.text = QTextEdit()
        self.text.setReadOnly(True)
        layout.addWidget(self.text)
        self.text.setText(self._profile_text())

    def _profile_text(self) -> str:
        phase_lines = [f"- {p.display_name_cn}({p.phase_key})" for p in self.profile.phase_definitions]
        return (
            f"课次: {self.profile.lesson_id}\n"
            f"动作: {self.profile.action_type}\n"
            f"版本: {self.profile.version}\n"
            f"阶段:\n" + "\n".join(phase_lines)
        )

    def update_by_result(self, result):
        self.text.append(f"\n当前阶段: {result.current_phase_cn}")
