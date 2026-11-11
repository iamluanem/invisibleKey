"""End-to-end: synthetic landmarks → pipeline → typed text (no camera)."""
import numpy as np
import pytest

from invisiblekey.dispatcher import KeyboardEventDispatcher, RecordingBackend
from invisiblekey.geometry import CoordinateTransformer
from invisiblekey.layout import VirtualKeyboardLayout
from invisiblekey.observations import FrameObservation
from invisiblekey.pipeline import TypingPipeline
from invisiblekey.simulation import DEFAULT_CORNERS, SimConfig, SyntheticTypist, run_simulation


def setup(cfg=None):
    tf, lay = CoordinateTransformer(DEFAULT_CORNERS), VirtualKeyboardLayout()
    be = RecordingBackend()
    d = KeyboardEventDispatcher(be)
    d.enable()
    return TypingPipeline(tf, lay, dispatcher=d), SyntheticTypist(tf, lay, cfg), be


@pytest.mark.parametrize("text", ["hello world", "Hello, World!", "abc\nxyz", "1234567890"])
def test_clean_typing_is_exact(text):
    r = run_simulation(text)
    assert r["typed"] == text and r["missed"] == 0 and r["spurious"] == 0


def test_backspace():
    pipe, typ, _ = setup()
    for f in typ.type_text("abx\bc"):
        pipe.process(f)
    assert pipe.text == "abc"


def test_repeated_letters():
    assert run_simulation("aaa bookkeeper")["typed"] == "aaa bookkeeper"


def test_moderate_noise_and_dropout():
    for seed in range(3):
        r = run_simulation("the quick brown fox", SimConfig(noise=0.0015, dropout=0.03, seed=seed))
        assert r["char_error_rate"] <= 0.1, r
        assert r["spurious"] <= 1, r


def test_idle_hands_produce_no_keys():
    pipe, typ, be = setup(SimConfig(noise=0.002, seed=7))
    for f in typ.idle(300):
        pipe.process(f)
    assert pipe.text == "" and be.events == []


def test_hand_repositioning_and_resting_produce_no_keys():
    pipe, typ, be = setup()
    frames = typ.idle(10) + typ.shift_hand("left", 0.05, 0.03, 6) + typ.shift_hand("left", -0.05, -0.03, 6)
    frames += typ.slow_press(("right", "index"), 1.0)
    for f in frames:
        pipe.process(f)
    assert be.events == []


def test_missing_hands_and_corrupt_landmarks_are_handled():
    pipe, typ, _ = setup()
    for f in typ.idle(10):
        pipe.process(f)
    pipe.process(FrameObservation(1.0, []))                  # no hands
    bad = typ.emit()
    bad.hands[0].landmarks[:] = np.nan                       # corrupt landmarks
    pipe.process(bad)
    for f in typ.type_text("ok", settle=5):
        f.timestamp += 1.5
        pipe.process(f)
    assert pipe.text == "ok"


def test_disabled_output_still_fills_test_area():
    tf, lay = CoordinateTransformer(DEFAULT_CORNERS), VirtualKeyboardLayout()
    be = RecordingBackend()
    pipe = TypingPipeline(tf, lay, dispatcher=KeyboardEventDispatcher(be))
    for f in SyntheticTypist(tf, lay).type_text("hi"):
        pipe.process(f)
    assert pipe.text == "hi" and be.events == []


def test_metrics_are_measured():
    r = run_simulation("hello world")
    m = r["metrics"]
    assert m["chars"] == 11 and m["processing_ms_p50"] is not None and m["fps"] == pytest.approx(30, abs=0.5)
    assert 0 < r["true_detection_delay_ms_p50"] <= 100
