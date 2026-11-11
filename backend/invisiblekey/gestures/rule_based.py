"""Rule-based temporal tap detector (per-finger state machine).

A 2D camera cannot prove that a finger touched the desk. What we detect is the
*kinematic signature* of a tap on the press signal (fingertip moving below its
own reference joint, normalized by hand size):

    HOVER/MOVING --fast descent--> DESCENDING --abrupt stop--> CONTACT (tap)
    CONTACT --partial lift--> LIFTING --lift past rearm level--> HOVER (re-armed)

Slow lowering (resting hands), whole-hand motion (joints move together, press
stays constant), small jitter (travel below threshold) and bounces (no re-arm)
do not produce taps.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from ..config import TapDetectionConfig
from ..observations import FingerId, FingerSample
from .base import FingerPhase, TapCandidate
from .filters import OneEuroFilter


@dataclass
class _Track:
    filt: OneEuroFilter
    t: float
    press: float
    knuckle: tuple[float, float]
    phase: FingerPhase = FingerPhase.HOVER
    velocity: float = 0.0
    lateral: float = 0.0
    reacquire: int = 0
    start_t: float = 0.0
    start_press: float = 0.0
    peak_v: float = 0.0
    max_press: float = 0.0
    max_pos: tuple[float, float] = (0.0, 0.0)
    impact_t: float = 0.0
    lateral_at_impact: float = 0.0
    contact_t: float = 0.0
    travel: float = 0.0
    noise_var: float = 0.0
    raw_press: float = 0.0
    pending: TapCandidate | None = field(default=None, repr=False)


class RuleBasedTapDetector:
    def __init__(self, config: TapDetectionConfig | None = None) -> None:
        self.config = config or TapDetectionConfig()
        self._tracks: dict[FingerId, _Track] = {}

    def reset(self) -> None:
        self._tracks.clear()

    def phase(self, finger: FingerId) -> FingerPhase:
        tr = self._tracks.get(finger)
        return tr.phase if tr else FingerPhase.LOST

    def phases(self) -> dict[FingerId, FingerPhase]:
        return {f: tr.phase for f, tr in self._tracks.items()}

    def mark_missing(self, finger: FingerId, timestamp: float) -> None:
        tr = self._tracks.get(finger)
        if tr and (timestamp - tr.t) * 1000 > self.config.loss_timeout_ms:
            self._lose(tr)

    def _lose(self, tr: _Track) -> None:
        tr.phase, tr.reacquire, tr.pending, tr.velocity = FingerPhase.LOST, 0, None, 0.0

    def _new_track(self, s: FingerSample) -> _Track:
        c = self.config
        f = OneEuroFilter(c.filter_min_cutoff, c.filter_beta, c.filter_d_cutoff)
        tr = _Track(filt=f, t=s.timestamp, press=f(s.press, 0), knuckle=s.knuckle)
        tr.phase, tr.reacquire = FingerPhase.LOST, 1
        tr.noise_var = (c.min_travel / c.noise_k) ** 2 if c.noise_k else 0.0
        tr.raw_press = s.press
        return tr

    def travel_threshold(self, finger: FingerId) -> float:
        """Adaptive: min_travel, raised when this finger's landmarks are jittery."""
        tr, c = self._tracks.get(finger), self.config
        sigma = tr.noise_var ** 0.5 if tr else 0.0
        return max(c.min_travel, c.noise_k * sigma)

    def update(self, s: FingerSample) -> TapCandidate | None:
        c = self.config
        tr = self._tracks.get(s.finger)
        if tr is None:
            self._tracks[s.finger] = self._new_track(s)
            return None
        dt = s.timestamp - tr.t
        if dt <= 0:
            return None
        if dt * 1000 > c.loss_timeout_ms:
            self._tracks[s.finger] = self._new_track(s)
            return None

        press = tr.filt(s.press, dt)
        tr.velocity = (press - tr.press) / dt
        speed = ((s.knuckle[0] - tr.knuckle[0]) ** 2 + (s.knuckle[1] - tr.knuckle[1]) ** 2) ** 0.5 / dt
        tr.lateral += c.lateral_smoothing * (speed - tr.lateral)
        prev_press = tr.press
        tr.t, tr.press, tr.knuckle = s.timestamp, press, s.knuckle
        if tr.phase in (FingerPhase.HOVER, FingerPhase.MOVING, FingerPhase.LOST):
            # white jitter: var(x_t - x_{t-1}) = 2 sigma^2 (estimated on the raw signal)
            tr.noise_var += c.noise_adapt_rate * ((s.press - tr.raw_press) ** 2 / 2 - tr.noise_var)
        tr.raw_press = s.press

        if tr.phase is FingerPhase.LOST:
            tr.reacquire += 1
            if tr.reacquire >= c.reacquire_frames:
                tr.phase = FingerPhase.HOVER
            return None
        return self._step(tr, s, press, prev_press)

    def _step(self, tr: _Track, s: FingerSample, press: float, prev_press: float) -> TapCandidate | None:
        c, v = self.config, tr.velocity
        if tr.phase in (FingerPhase.HOVER, FingerPhase.MOVING):
            if v >= c.descend_velocity:
                tr.phase = FingerPhase.DESCENDING
                tr.start_t, tr.start_press, tr.peak_v = s.timestamp, prev_press, v
                tr.max_press, tr.max_pos, tr.impact_t = press, s.tip, s.timestamp
                tr.lateral_at_impact = tr.lateral
            else:
                tr.phase = FingerPhase.MOVING if tr.lateral >= c.moving_speed else FingerPhase.HOVER
            return None

        if tr.phase is FingerPhase.DESCENDING:
            tr.peak_v = max(tr.peak_v, v)
            if press >= tr.max_press:
                tr.max_press, tr.max_pos, tr.impact_t = press, s.tip, s.timestamp
                tr.lateral_at_impact = tr.lateral
            if v <= c.stop_ratio * tr.peak_v:
                tr.travel = tr.max_press - tr.start_press
                if tr.travel < self.travel_threshold(s.finger):
                    tr.phase = FingerPhase.HOVER
                    return None
                tr.phase, tr.contact_t = FingerPhase.CONTACT, s.timestamp
                cand = TapCandidate(
                    finger=s.finger, impact_time=tr.impact_t, detected_time=s.timestamp,
                    position=tr.max_pos, peak_velocity=tr.peak_v,
                    deceleration=max(0.0, min(1.0, 1 - max(v, 0.0) / tr.peak_v)),
                    travel=tr.travel, lateral_speed=tr.lateral_at_impact, tracking=s.tracking,
                )
                if c.confirm_on_lift:
                    tr.pending = cand
                    return None
                return cand
            if (s.timestamp - tr.start_t) * 1000 > c.max_descent_ms:
                tr.phase = FingerPhase.HOVER
            return None

        if tr.phase is FingerPhase.CONTACT:
            if press <= tr.max_press - c.lift_fraction * tr.travel:
                tr.phase = FingerPhase.LIFTING
                cand, tr.pending = tr.pending, None
                if cand and (s.timestamp - tr.contact_t) * 1000 <= c.max_dwell_ms:
                    cand.detected_time = s.timestamp
                    return cand
            elif tr.pending and (s.timestamp - tr.contact_t) * 1000 > c.max_dwell_ms:
                tr.pending = None  # finger is resting, not tapping
            return None

        if tr.phase is FingerPhase.LIFTING and press <= tr.max_press - c.rearm_fraction * tr.travel:
            tr.phase = FingerPhase.HOVER
        return None
