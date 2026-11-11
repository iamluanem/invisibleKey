"""Threaded camera capture: always hands out the newest frame (no queue buildup)."""
from __future__ import annotations

import logging
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

log = logging.getLogger(__name__)


@dataclass(slots=True)
class CapturedFrame:
    image: np.ndarray   # BGR
    timestamp: float    # time.monotonic() right after grab
    seq: int


def _default_capture(index: int):
    import cv2

    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else (
        cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY)
    return cv2.VideoCapture(index, backend)


def list_cameras(max_index: int = 5, factory: Callable = _default_capture) -> list[int]:
    found = []
    for i in range(max_index):
        cap = factory(i)
        try:
            if cap.isOpened() and cap.read()[0]:
                found.append(i)
        finally:
            cap.release()
    return found


class CameraManager:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, fps: int = 30,
                 factory: Callable = _default_capture) -> None:
        self.index, self.width, self.height, self.fps = index, width, height, fps
        self._factory = factory
        self._cap = None
        self._thread: threading.Thread | None = None
        self._cond = threading.Condition()
        self._latest: CapturedFrame | None = None
        self._running = False
        self._consumed = True
        self.dropped = 0
        self.failures = 0

    def open(self) -> None:
        import cv2

        cap = self._factory(self.index)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open camera {self.index} (macOS: allow Camera access for your terminal)")
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap, self._running = cap, True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        seq = 0
        while self._running:
            ok, img = self._cap.read()
            if not ok:
                self.failures += 1
                time.sleep(0.01)
                continue
            seq += 1
            with self._cond:
                if self._latest is not None and self._latest.seq == seq - 1 and not self._consumed:
                    self.dropped += 1
                self._latest, self._consumed = CapturedFrame(img, time.monotonic(), seq), False
                self._cond.notify_all()

    def read(self, timeout: float = 1.0) -> CapturedFrame | None:
        """Newest unseen frame, or None on timeout."""
        with self._cond:
            if not self._cond.wait_for(lambda: self._latest is not None and not self._consumed, timeout):
                return None
            self._consumed = True
            return self._latest

    def close(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap is not None:
            self._cap.release()
        self._cap = self._thread = None

    def __enter__(self) -> "CameraManager":
        self.open()
        return self

    def __exit__(self, *exc) -> None:
        self.close()
