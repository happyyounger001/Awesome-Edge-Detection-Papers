from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import List

import numpy as np
from scipy.io import wavfile

from .config import AnalyzerConfig


class BeepUnavailableError(RuntimeError):
    pass


def detect_beeps(video_path: str | Path, cfg: AnalyzerConfig) -> List[float]:
    if shutil.which("ffmpeg") is None:
        raise BeepUnavailableError("ffmpeg not found; skipping beep analysis.")

    with tempfile.TemporaryDirectory() as tmp:
        wav_path = Path(tmp) / "audio.wav"
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(video_path),
            "-ac",
            "1",
            "-ar",
            "16000",
            str(wav_path),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BeepUnavailableError(f"ffmpeg audio extraction failed: {proc.stderr[-200:]}")

        sr, signal = wavfile.read(wav_path)
        signal = signal.astype(float)
        if np.max(np.abs(signal)) > 0:
            signal /= np.max(np.abs(signal))

        window = int(0.02 * sr)
        if window <= 1:
            return []
        energy = np.convolve(signal**2, np.ones(window) / window, mode="same")
        median = np.median(energy) + 1e-9
        threshold = median * cfg.beep_energy_thr

        candidates = np.where(energy >= threshold)[0]
        if len(candidates) == 0:
            return []

        sep = int(cfg.beep_min_separation_s * sr)
        kept = [int(candidates[0])]
        for c in candidates[1:]:
            if c - kept[-1] >= sep:
                kept.append(int(c))

        return [k / sr for k in kept]
