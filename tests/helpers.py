import numpy as np

from pose.models import PoseFrame, PoseSequence


def make_sequence() -> PoseSequence:
    frames = 80
    fps = 20.0
    names = [
        "left_shoulder", "right_shoulder", "left_wrist", "right_wrist", "left_hip",
        "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle"
    ]
    arrays = {name: np.zeros((frames, 2), dtype=float) for name in names}
    time = np.arange(frames) / fps

    arrays["left_shoulder"][:] = [0.38, 0.25]
    arrays["right_shoulder"][:] = [0.30, 0.25]
    arrays["left_hip"][:] = [0.40, 0.48]
    arrays["right_hip"][:] = [0.31, 0.49]
    arrays["left_knee"][:] = [0.49, 0.63]
    arrays["right_knee"][:] = [0.28, 0.64]
    arrays["left_ankle"][:] = [0.48, 0.83]
    arrays["right_ankle"][:] = [0.20, 0.84]
    arrays["left_wrist"][:] = [0.44, 0.33]
    arrays["right_wrist"][:] = [0.26, 0.34]

    arrays["left_wrist"][10:16, 0] += np.linspace(0.0, 0.18, 6)
    arrays["left_wrist"][16:, 0] += 0.18
    arrays["left_ankle"][14:24, 0] += np.linspace(0.0, 0.24, 10)
    arrays["left_ankle"][24:, 0] += 0.24
    arrays["left_hip"][24:34, 0] += np.linspace(0.0, 0.03, 10)
    arrays["left_hip"][34:, 0] += 0.03
    arrays["right_hip"][24:34, 0] += np.linspace(0.0, 0.02, 10)
    arrays["right_hip"][34:, 0] += 0.02
    arrays["left_knee"][12:20, 1] -= np.linspace(0.0, 0.08, 8)
    arrays["left_knee"][20:, 1] -= 0.08

    pose_frames = [
        PoseFrame(index=i, timestamp=float(time[i]), points={name: tuple(arrays[name][i]) for name in names}, visibility={name: 1.0 for name in names})
        for i in range(frames)
    ]
    return PoseSequence(fps=fps, width=640, height=480, frames=pose_frames, arrays=arrays)
