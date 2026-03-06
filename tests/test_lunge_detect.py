import numpy as np

from fencing_analyzer.config import AnalyzerConfig
from fencing_analyzer.lunge_detect import detect_lunges
from fencing_analyzer.pose import PoseSequence


def make_pose(frames=80):
    x = {k: np.zeros(frames) for k in [
        "left_shoulder", "right_shoulder", "left_hip", "right_hip", "left_knee", "right_knee",
        "left_ankle", "right_ankle", "left_wrist", "right_wrist"
    ]}
    y = {k: np.zeros(frames) for k in x}
    vis = {k: np.ones(frames) for k in x}
    # left is front, build one movement burst
    x["left_ankle"][20:35] = np.linspace(0, 0.6, 15)
    x["left_ankle"][35:] = 0.6
    x["right_ankle"][:] = -0.2
    return PoseSequence(fps=30.0, width=640, height=480, frames=frames, x=x, y=y, vis=vis)


def test_detect_lunge_from_speed_burst():
    pose = make_pose()
    cfg = AnalyzerConfig(v_start_thr=0.2, v_end_thr=0.05, min_lunge_duration=0.1, settle_frames=2)
    events = detect_lunges(pose, cfg)
    assert len(events) == 1
    assert events[0].start_frame <= 21
    assert events[0].end_frame >= 33
