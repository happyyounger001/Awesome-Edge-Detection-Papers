from __future__ import annotations

from core.models import FrameFeatures


class FeatureExtractor:
    def extract(self, frame, pose_data, timestamp_ms: int, frame_index: int) -> FrameFeatures:
        values = {}
        if isinstance(pose_data, dict):
            values = {k: float(v) for k, v in pose_data.items() if isinstance(v, (int, float))}
        return FrameFeatures(timestamp_ms=timestamp_ms, frame_index=frame_index, feature_values=values)
