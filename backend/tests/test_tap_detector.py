"""Tap state machine on hand-crafted press signals (one finger, 30 FPS)."""
import pytest

from invisiblekey.config import TapDetectionConfig
from invisiblekey.gestures import FingerPhase, RuleBasedTapDetector
from invisiblekey.observations import FingerSample

F = ("right", "index")
DT = 1 / 30


def feed(det, presses, t0=0.0, knuckle=lambda i: (0.5, 0.5), finger=F):
    taps, phases = [], []
    for i, p in enumerate(presses):
        s = FingerSample(finger, t0 + i * DT, p, (0.5, 0.5), knuckle(i), 0.95)
        c = det.update(s)
        if c:
            taps.append(c)
        phases.append(det.phase(finger))
    return taps, phases


REST = [0.3] * 10
TAP = [0.42, 0.55, 0.55, 0.47, 0.38, 0.3, 0.3, 0.3]


def test_single_tap_and_phase_sequence():
    det = RuleBasedTapDetector()
    taps, phases = feed(det, REST + TAP + REST)
    assert len(taps) == 1
    seen = [p for i, p in enumerate(phases) if i == 0 or p != phases[i - 1]]
    assert seen[-5:] == [FingerPhase.HOVER, FingerPhase.DESCENDING, FingerPhase.CONTACT,
                         FingerPhase.LIFTING, FingerPhase.HOVER]
    c = taps[0]
    assert c.travel == pytest.approx(0.25, abs=0.08) and 0 <= c.deceleration <= 1


def test_hover_jitter_is_not_a_tap():
    det = RuleBasedTapDetector()
    taps, _ = feed(det, [0.3 + (0.01 if i % 2 else -0.01) for i in range(120)])
    assert taps == []


def test_slow_lowering_is_not_a_tap():
    det = RuleBasedTapDetector()
    ramp = [0.3 + 0.3 * i / 30 for i in range(30)]  # 0.3 scale over 1 s
    taps, _ = feed(det, REST + ramp + [0.6] * 10 + ramp[::-1] + REST)
    assert taps == []


def test_bounce_without_rearm_is_single_tap():
    det = RuleBasedTapDetector()
    bounce = [0.42, 0.55, 0.55, 0.51, 0.55, 0.55, 0.47, 0.38, 0.3, 0.3]  # small re-press, no full lift
    taps, _ = feed(det, REST + bounce + REST)
    assert len(taps) == 1


def test_repeated_taps_after_lift_are_allowed():
    det = RuleBasedTapDetector()
    taps, _ = feed(det, REST + TAP + TAP + TAP + REST)
    assert len(taps) == 3


def test_whole_hand_motion_keeps_press_constant_and_does_not_tap():
    det = RuleBasedTapDetector()
    taps, phases = feed(det, [0.3] * 40, knuckle=lambda i: (0.5 + 0.02 * i, 0.5))
    assert taps == [] and FingerPhase.MOVING in phases


def test_short_tracking_gap_is_tolerated():
    det = RuleBasedTapDetector()
    feed(det, REST)
    det.mark_missing(F, 10 * DT + 0.05)  # 50 ms gap < 120 ms timeout
    assert det.phase(F) is FingerPhase.HOVER
    taps, _ = feed(det, TAP, t0=10 * DT + 0.066)
    assert len(taps) == 1


def test_long_tracking_loss_resets_and_requires_reacquire():
    det = RuleBasedTapDetector()
    taps, _ = feed(det, REST + TAP[:2])          # finger mid-descent...
    det.mark_missing(F, 12 * DT + 0.5)           # ...then lost for 500 ms
    assert det.phase(F) is FingerPhase.LOST
    taps, phases = feed(det, [0.55, 0.47, 0.3, 0.3], t0=12 * DT + 0.5)
    assert taps == []                             # no phantom tap from stale state
    assert phases[0] is FingerPhase.LOST


def test_confirm_on_lift_rejects_resting_finger():
    det = RuleBasedTapDetector(TapDetectionConfig(confirm_on_lift=True, max_dwell_ms=200))
    taps, _ = feed(det, REST + TAP + REST)
    assert len(taps) == 1
    det2 = RuleBasedTapDetector(TapDetectionConfig(confirm_on_lift=True, max_dwell_ms=200))
    rest_down = [0.42, 0.55] + [0.55] * 20 + [0.47, 0.38, 0.3]  # stays down ~700 ms
    taps, _ = feed(det2, REST + rest_down + REST)
    assert taps == []
