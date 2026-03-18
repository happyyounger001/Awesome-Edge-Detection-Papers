from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QUrl
from PySide6.QtWidgets import QMainWindow, QTextBrowser


class ReportWindow(QMainWindow):
    def __init__(self, report_path: Path):
        super().__init__()
        self.setWindowTitle("击剑弓步AI分析报告")
        self.resize(1100, 800)
        browser = QTextBrowser()
        browser.setOpenExternalLinks(True)
        browser.setSource(QUrl.fromLocalFile(str(report_path.resolve())))
        self.setCentralWidget(browser)
