"""Frame → finger samples → tap candidates → keys → dispatch."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from .confidence import TapConfidenceEstimator, TapDecision
from .dispatcher import KeyboardEventDispatcher
from .gestures import FingerPhase, RuleBasedTapDetector, TapDetector
from .geometry import CoordinateTransformer
from .layout import VirtualKeyboardLayout
from .observations import FINGERS, FingerId, FrameObservation, hand_samples
from .performance import PerformanceMonitor
from .policy import KeyPressPolicy


@dataclass(slots=True)
class TapEvent:
    finger: str
    key_id: str | None
    output: str | None
    impact_time: float
    detected_time: float
    position: tuple[float, float]
    score: float
    decision: str
    components: dict[str, float]
    dispatched: bool = False
    sent_to_os: bool = False
    reason: str | None = None


@dataclass(slots=True)
class FrameResult:
    timestamp: float
    taps: list[TapEvent] = field(default_factory=list)
    fingertips: dict[str, tuple[float, float]] = field(default_factory=dict)
    phases: dict[str, str] = field(default_factory=dict)


def _fid(f: FingerId) -> str:
    return f"{f[0]}.{f[1]}"


class TypingPipeline:
    def __init__(
        self,
        transformer: CoordinateTransformer,
        layout: VirtualKeyboardLayout | None = None,
        detector: TapDetector | None = None,
        estimator: TapConfidenceEstimator | None = None,
        policy: KeyPressPolicy | None = None,
        dispatcher: KeyboardEventDispatcher | None = None,
        monitor: PerformanceMonitor | None = None,
    ) -> None:
        self.transformer = transformer
        self.layout = layout or VirtualKeyboardLayout()
        self.detector = detector or RuleBasedTapDetector()
        self.estimator = estimator or TapConfidenceEstimator()
        self.policy = policy or KeyPressPolicy()
        self.dispatcher = dispatcher or KeyboardEventDispatcher()
        self.monitor = monitor or PerformanceMonitor()
        self._known: set[FingerId] = set()

    def process(self, frame: FrameObservation) -> FrameResult:
        t0 = time.perf_counter()
        t = frame.timestamp
        res = FrameResult(t)
        seen: set[FingerId] = set()
        by_hand: dict[str, list] = {}
        for hand in frame.hands:
            samples = hand_samples(hand, t, self.transformer)
            if samples is None:
                continue
            for f in FINGERS:
                s = samples[f]
                seen.add(s.finger)
                res.fingertips[_fid(s.finger)] = s.tip
                cand = self.detector.update(s)
                if cand:
                    by_hand.setdefault(hand.handedness, []).append(cand)
        for f in self._known - seen:
            self.detector.mark_missing(f, t)
        self._known |= seen

        for cands in by_hand.values():
            scored = []
            for c in cands:
                hit = self.layout.hit_test(*c.position)
                scored.append((self.estimator.assess(c, hit), c, hit))
            scored.sort(key=lambda x: x[0].score, reverse=True)
            for i, (a, c, hit) in enumerate(scored):
                ev = TapEvent(_fid(c.finger), hit.key.id if hit else None, None, c.impact_time,
                              c.detected_time, c.position, a.score, a.decision.value, a.components,
                              reason=a.reason)
                if i > 0 and a.decision is TapDecision.ACCEPTED:
                    ev.decision, ev.reason = TapDecision.REJECTED.value, "coalesced"
                elif a.decision is TapDecision.ACCEPTED:
                    if self.policy.allow(c.finger, hit.key.id, c.impact_time):
                        out = self.dispatcher.dispatch(hit.key)
                        ev.output, ev.dispatched, ev.sent_to_os = out.output, True, out.sent_to_os
                        if out.output != "<shift>":
                            self.monitor.record_key(c.impact_time)
                    else:
                        ev.reason = "duplicate"
                        self.monitor.suppressed += 1
                self.monitor.record_tap(ev.decision, c.impact_time, c.detected_time)
                res.taps.append(ev)

        res.phases = {_fid(f): self.detector.phase(f).value for f in self._known}
        self.monitor.record_frame(t, (time.perf_counter() - t0) * 1000)
        return res

    @property
    def text(self) -> str:
        return self.dispatcher.buffer.text

    def reset(self) -> None:
        self.detector.reset()
        self.policy.reset()
        self._known.clear()


def build_pipeline(settings, backend=None) -> TypingPipeline:
    """Wire a pipeline from AppSettings (requires a saved calibration)."""
    if settings.calibration is None:
        raise ValueError("no calibration: run the calibration wizard first")
    return TypingPipeline(
        settings.calibration.transformer(),
        VirtualKeyboardLayout(settings.layout),
        RuleBasedTapDetector(settings.detection),
        TapConfidenceEstimator(settings.confidence),
        KeyPressPolicy(settings.policy),
        KeyboardEventDispatcher(backend),
    )


__all__ = ["TypingPipeline", "TapEvent", "FrameResult", "FingerPhase", "build_pipeline"]
