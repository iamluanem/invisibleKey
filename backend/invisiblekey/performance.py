"""Measured (not claimed) performance: FPS, processing latency, detection delay."""
from __future__ import annotations

from collections import Counter, deque

import numpy as np


class PerformanceMonitor:
    def __init__(self, window: int = 120) -> None:
        self.frame_times: deque[float] = deque(maxlen=window)
        self.processing_ms: deque[float] = deque(maxlen=window)
        self.detection_delay_ms: deque[float] = deque(maxlen=window)
        self.e2e_ms: deque[float] = deque(maxlen=window)  # capture -> key dispatched
        self.decisions: Counter[str] = Counter()
        self.suppressed = 0
        self.chars = 0
        self.first_key_t: float | None = None
        self.last_key_t: float | None = None

    def record_frame(self, timestamp: float, processing_ms: float) -> None:
        self.frame_times.append(timestamp)
        self.processing_ms.append(processing_ms)

    def record_tap(self, decision: str, impact_t: float, detected_t: float) -> None:
        self.decisions[decision] += 1
        self.detection_delay_ms.append((detected_t - impact_t) * 1000)

    def record_e2e(self, ms: float) -> None:
        self.e2e_ms.append(ms)

    def record_key(self, t: float) -> None:
        self.chars += 1
        self.first_key_t = self.first_key_t if self.first_key_t is not None else t
        self.last_key_t = t

    @staticmethod
    def _pct(values, q: float) -> float | None:
        return round(float(np.percentile(values, q)), 2) if values else None

    def snapshot(self) -> dict:
        ft = self.frame_times
        fps = (len(ft) - 1) / (ft[-1] - ft[0]) if len(ft) > 1 and ft[-1] > ft[0] else 0.0
        span = (self.last_key_t or 0) - (self.first_key_t or 0)
        wpm = (self.chars / 5) / (span / 60) if span > 0 else 0.0
        return {
            "fps": round(fps, 1),
            "processing_ms_p50": self._pct(self.processing_ms, 50),
            "processing_ms_p95": self._pct(self.processing_ms, 95),
            "detection_delay_ms_p50": self._pct(self.detection_delay_ms, 50),
            "e2e_ms_p50": self._pct(self.e2e_ms, 50),
            "e2e_ms_p95": self._pct(self.e2e_ms, 95),
            "taps": dict(self.decisions),
            "suppressed_duplicates": self.suppressed,
            "chars": self.chars,
            "wpm": round(wpm, 1),
        }
