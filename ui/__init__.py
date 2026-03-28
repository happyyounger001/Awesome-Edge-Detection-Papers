"""UI package exports.

Avoid importing heavy Qt window modules at package import time to prevent
circular-import side effects when non-GUI entry points (e.g. web API)
import submodules like ``ui.video_overlay``.
"""

__all__ = ["FencingMainWindow", "LearningSystemWindow", "ProgressDialog", "ReportWindow"]


def __getattr__(name: str):
    if name == "FencingMainWindow":
        from .main_window import FencingMainWindow

        return FencingMainWindow
    if name == "LearningSystemWindow":
        from .learning_window import LearningSystemWindow

        return LearningSystemWindow
    if name == "ProgressDialog":
        from .progress_dialog import ProgressDialog

        return ProgressDialog
    if name == "ReportWindow":
        from .report_window import ReportWindow

        return ReportWindow
    raise AttributeError(f"module 'ui' has no attribute {name!r}")
