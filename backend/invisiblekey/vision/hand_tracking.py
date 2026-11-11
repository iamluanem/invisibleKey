"""MediaPipe Hand Landmarker (Tasks API, VIDEO mode) → FrameObservation."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..observations import FrameObservation, HandObservation
from .model import ensure_model


def to_observation(result, timestamp: float) -> FrameObservation:
    """Convert a HandLandmarkerResult. Hands are labeled by screen position when two
    are visible (MediaPipe's handedness can flip between frames and depends on
    mirroring); with one hand we keep MediaPipe's label."""
    hands = []
    for lms, cats in zip(result.hand_landmarks, result.handedness):
        arr = np.array([[p.x, p.y, p.z] for p in lms], dtype=float)
        cat = cats[0]
        hands.append(HandObservation(cat.category_name.lower(), arr, float(cat.score)))
    if len(hands) == 2:
        hands.sort(key=lambda h: h.landmarks[0, 0])  # by wrist x
        hands[0].handedness, hands[1].handedness = "left", "right"
    return FrameObservation(timestamp, hands)


class HandTrackingService:
    def __init__(self, model_path: Path | None = None, num_hands: int = 2,
                 min_detection: float = 0.5, min_presence: float = 0.5, min_tracking: float = 0.5,
                 mirror: bool = True) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.mirror = mirror
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(ensure_model(model_path))),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(opts)
        self._last_ms = -1

    def prepare(self, bgr: np.ndarray) -> np.ndarray:
        """Frame as the user sees it (mirrored like a selfie view when enabled)."""
        return np.ascontiguousarray(bgr[:, ::-1]) if self.mirror else bgr

    def process(self, bgr_view: np.ndarray, timestamp: float) -> FrameObservation:
        """`bgr_view` must come from prepare(), so landmarks match what is drawn."""
        rgb = np.ascontiguousarray(bgr_view[:, :, ::-1])
        ms = max(int(timestamp * 1000), self._last_ms + 1)  # VIDEO mode needs strictly increasing ms
        self._last_ms = ms
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        return to_observation(self._landmarker.detect_for_video(image, ms), timestamp)

    def close(self) -> None:
        self._landmarker.close()
