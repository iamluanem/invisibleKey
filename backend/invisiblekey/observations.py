"""Hand observations (MediaPipe-shaped) and per-finger feature extraction."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .geometry import CoordinateTransformer

WRIST, MIDDLE_MCP = 0, 9
FINGERS = ("thumb", "index", "middle", "ring", "pinky")
# finger -> (press reference joint, knuckle, tip)
JOINTS = {
    "thumb": (2, 1, 4),
    "index": (6, 5, 8),
    "middle": (10, 9, 12),
    "ring": (14, 13, 16),
    "pinky": (18, 17, 20),
}
FingerId = tuple[str, str]  # (hand, finger)


@dataclass(slots=True)
class HandObservation:
    handedness: str               # "left" | "right"
    landmarks: np.ndarray         # (21, 3) normalized image coords
    score: float = 1.0            # tracker confidence


@dataclass(slots=True)
class FrameObservation:
    timestamp: float              # capture time, seconds (monotonic)
    hands: list[HandObservation] = field(default_factory=list)


@dataclass(slots=True)
class FingerSample:
    finger: FingerId
    timestamp: float
    press: float                  # tip below reference joint, in hand-scale units
    tip: tuple[float, float]      # surface coords
    knuckle: tuple[float, float]  # surface coords
    tracking: float


def hand_samples(hand: HandObservation, t: float, tf: CoordinateTransformer) -> dict[str, FingerSample] | None:
    lm = hand.landmarks
    if lm is None or lm.shape != (21, 3) or not np.all(np.isfinite(lm[:, :2])):
        return None
    scale = float(np.linalg.norm(lm[WRIST, :2] - lm[MIDDLE_MCP, :2]))
    if scale < 1e-4:
        return None
    idx = [j for f in FINGERS for j in JOINTS[f][1:]]  # knuckle, tip per finger
    surf = tf.to_surface(lm[idx, :2])
    out = {}
    for i, f in enumerate(FINGERS):
        ref, _, tip = JOINTS[f]
        out[f] = FingerSample(
            finger=(hand.handedness, f),
            timestamp=t,
            press=float((lm[tip, 1] - lm[ref, 1]) / scale),
            tip=(float(surf[2 * i + 1, 0]), float(surf[2 * i + 1, 1])),
            knuckle=(float(surf[2 * i, 0]), float(surf[2 * i, 1])),
            tracking=float(hand.score),
        )
    return out
