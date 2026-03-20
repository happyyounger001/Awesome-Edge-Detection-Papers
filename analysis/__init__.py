from .models import AnalysisResult, FrameAssessment, LungeQuality, TimingResult

__all__ = ["AnalysisEngine", "AnalysisResult", "FrameAssessment", "LungeQuality", "TimingResult"]


def __getattr__(name: str):
    if name == "AnalysisEngine":
        from .engine import AnalysisEngine

        return AnalysisEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
