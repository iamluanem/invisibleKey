"""One Euro filter: heavy smoothing at rest, low lag during fast motion."""
from __future__ import annotations

import math


class OneEuroFilter:
    def __init__(self, min_cutoff: float, beta: float, d_cutoff: float) -> None:
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x: float | None = None
        self.dx = 0.0

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def __call__(self, value: float, dt: float) -> float:
        if self.x is None or dt <= 0:
            self.x = value
            return value
        a_d = self._alpha(self.d_cutoff, dt)
        self.dx = a_d * (value - self.x) / dt + (1 - a_d) * self.dx
        a = self._alpha(self.min_cutoff + self.beta * abs(self.dx), dt)
        self.x = a * value + (1 - a) * self.x
        return self.x
