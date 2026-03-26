from __future__ import annotations

import time

from core.playback_controller import PlaybackController


def test_playback_clock_keeps_near_real_time():
    ctl = PlaybackController()
    ctl.load("dummy.mp4")
    ctl.play()
    time.sleep(0.12)
    t = ctl.current_time_ms()
    assert 80 <= t <= 220
