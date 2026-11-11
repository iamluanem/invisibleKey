"""Detector interface: rule-based now, trained temporal classifiers later."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..observations import FingerId, FingerSample


class FingerPhase(str, Enum):
    HOVER = "hover"
    MOVING = "moving"
    DESCENDING = "descending"
    CONTACT = "contact"
    LIFTING = "lifting"
    LOST = "lost"


@dataclass(slots=True)
class TapCandidate:
    finger: FingerId
    impact_time: float
    detected_time: float
    position: tuple[float, float]  # surface coords of the fingertip at impact
    peak_velocity: float
    deceleration: float            # 0..1, how sharply the descent stopped
    travel: float
    lateral_speed: float
    tracking: float


class TapDetector(Protocol):
    """Any detector (rules, ONNX model...) consumes per-finger samples in time order."""

    def update(self, sample: FingerSample) -> TapCandidate | None: ...
    def mark_missing(self, finger: FingerId, timestamp: float) -> None: ...
    def phase(self, finger: FingerId) -> FingerPhase: ...
    def reset(self) -> None: ...
