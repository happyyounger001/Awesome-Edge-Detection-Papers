from __future__ import annotations

from queue import Empty, Full, Queue


def push_latest(queue_obj: Queue, item):
    try:
        queue_obj.put_nowait(item)
    except Full:
        try:
            queue_obj.get_nowait()
        except Empty:
            pass
        queue_obj.put_nowait(item)


class AnalysisWorker:
    def __init__(self, frame_queue: Queue, result_queue: Queue, engine):
        self.frame_queue = frame_queue
        self.result_queue = result_queue
        self.engine = engine
        self.running = True

    def _get_latest_frame(self):
        latest = None
        while True:
            try:
                latest = self.frame_queue.get(timeout=0.02 if latest is None else 0.0)
            except Empty:
                break
        return latest

    def _push_latest_result(self, result):
        push_latest(self.result_queue, result)

    def run_once(self):
        frame_packet = self._get_latest_frame()
        if frame_packet is None:
            return None
        result = self.engine.analyze(frame_packet)
        self._push_latest_result(result)
        return result

    def run(self):
        while self.running:
            self.run_once()
