import numpy as np

from rules.evaluator import angle_deg


def test_angle_deg_right_angle():
    a = np.array([1.0, 0.0])
    b = np.array([0.0, 0.0])
    c = np.array([0.0, 1.0])
    assert abs(angle_deg(a, b, c) - 90.0) < 1e-6
