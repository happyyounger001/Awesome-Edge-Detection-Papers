from __future__ import annotations

import time
from pathlib import Path



class PlaybackController:
    def __init__(self):
        self.video_path: Path | None = None
        self._base_ms = 0
        self._start_monotonic = 0.0
        self._playing = False

    def load(self, video_path: str) -> None:
        self.video_path = Path(video_path)
        self._base_ms = 0
        self._start_monotonic = time.monotonic()
        self._playing = False

    def play(self) -> None:
        if not self._playing:
            self._start_monotonic = time.monotonic() - (self._base_ms / 1000.0)
            self._playing = True

    def pause(self) -> None:
        self._base_ms = self.current_time_ms()
        self._playing = False

    def seek(self, ms: int) -> None:
        self._base_ms = max(0, int(ms))
        if self._playing:
            self._start_monotonic = time.monotonic() - (self._base_ms / 1000.0)

    def current_time_ms(self) -> int:
        if not self._playing:
            return self._base_ms
        return int((time.monotonic() - self._start_monotonic) * 1000.0)
