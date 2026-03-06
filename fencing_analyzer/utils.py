from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    ba = a - b
    bc = c - b
    denom = np.linalg.norm(ba) * np.linalg.norm(bc)
    if denom == 0:
        return float("nan")
    cosine = np.clip(np.dot(ba, bc) / denom, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def ema_smooth(values: np.ndarray, alpha: float) -> np.ndarray:
    if len(values) == 0:
        return values
    out = np.zeros_like(values)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out


def safe_mean(items: Iterable[float]) -> float:
    vals = [x for x in items if not math.isnan(x)]
    if not vals:
        return float("nan")
    return float(np.mean(vals))
