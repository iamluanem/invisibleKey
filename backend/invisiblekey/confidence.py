"""Turns a tap candidate + key hit into a scored, explainable decision."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum

from .config import ConfidenceConfig
from .gestures.base import TapCandidate
from .layout import KeyHit


class TapDecision(str, Enum):
    ACCEPTED = "accepted"     # dispatched
    UNCERTAIN = "uncertain"   # shown in UI, never typed
    REJECTED = "rejected"


@dataclass(slots=True)
class TapAssessment:
    score: float
    decision: TapDecision
    components: dict[str, float] = field(default_factory=dict)
    reason: str | None = None


class TapConfidenceEstimator:
    def __init__(self, config: ConfidenceConfig | None = None) -> None:
        self.config = config or ConfidenceConfig()

    def assess(self, cand: TapCandidate, hit: KeyHit | None) -> TapAssessment:
        c = self.config
        if hit is None:
            return TapAssessment(0.0, TapDecision.REJECTED, reason="off_layout")
        comps = {
            "velocity": min(1.0, cand.peak_velocity / c.velocity_ref),
            "deceleration": cand.deceleration,
            "travel": min(1.0, cand.travel / c.travel_ref),
            "stability": 1.0 - min(1.0, cand.lateral_speed / c.max_lateral_speed),
            "tracking": max(0.0, min(1.0, cand.tracking)),
            "key": hit.centrality,
        }
        # Weighted geometric mean: one very weak signal sinks the score.
        wsum = sum(c.weights.get(k, 0.0) for k in comps) or 1.0
        log = sum(c.weights.get(k, 0.0) * math.log(max(v, 1e-3)) for k, v in comps.items())
        score = math.exp(log / wsum)
        if score >= c.accept_threshold:
            decision = TapDecision.ACCEPTED
        elif score >= c.uncertain_threshold:
            decision = TapDecision.UNCERTAIN
        else:
            decision = TapDecision.REJECTED
        return TapAssessment(round(score, 4), decision, {k: round(v, 4) for k, v in comps.items()})
