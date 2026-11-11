"""Phase 2 components without camera/MediaPipe: fakes stand in for the hardware."""
import time
import zipfile
from types import SimpleNamespace as NS

import numpy as np
import pytest

from invisiblekey.app import DebugApp
from invisiblekey.dispatcher import RecordingBackend
from invisiblekey.geometry import CoordinateTransformer
from invisiblekey.layout import VirtualKeyboardLayout
from invisiblekey.recording import LandmarkRecorder, load_recording, replay
from invisiblekey.settings import AppSettings, SettingsManager
from invisiblekey.simulation import DEFAULT_CORNERS, SyntheticTypist
from invisiblekey.vision.hand_tracking import to_observation
from invisiblekey.vision.model import ensure_model, is_valid_model


@pytest.fixture(autouse=True)
def isolated_home(tmp_path, monkeypatch):
    monkeypatch.setenv("INVISIBLEKEY_HOME", str(tmp_path / "home"))


def sim_frames(text):
    tf, lay = CoordinateTransformer(DEFAULT_CORNERS), VirtualKeyboardLayout()
    return SyntheticTypist(tf, lay).type_text(text)


# --- MediaPipe result conversion ---------------------------------------------
def fake_result(*hands):
    lms = [[NS(x=wx + i * 0.001, y=0.5, z=0.0) for i in range(21)] for wx, _ in hands]
    cats = [[NS(category_name=label, score=0.9)] for _, label in hands]
    return NS(hand_landmarks=lms, handedness=cats)


def test_two_hands_labeled_by_screen_position():
    obs = to_observation(fake_result((0.7, "Left"), (0.2, "Left")), 1.0)  # MediaPipe mislabels both
    assert [h.handedness for h in obs.hands] == ["left", "right"]
    assert obs.hands[0].landmarks.shape == (21, 3) and obs.hands[0].landmarks[0, 0] == pytest.approx(0.2)


def test_single_hand_keeps_mediapipe_label_and_empty_result():
    assert to_observation(fake_result((0.5, "Right")), 0).hands[0].handedness == "right"
    assert to_observation(fake_result(), 0).hands == []


# --- model cache ---------------------------------------------------------------
def test_model_validation_and_cache(tmp_path):
    good = tmp_path / "m.task"
    with zipfile.ZipFile(good, "w") as z:
        z.writestr("hand_detector.tflite", b"x")
    assert is_valid_model(good) and ensure_model(good, url="http://invalid.invalid/x") == good
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not a zip")
    with pytest.raises(RuntimeError):
        ensure_model(tmp_path / "new.task", url=bad.as_uri())
    assert not (tmp_path / "new.task").exists()


# --- recording / replay -------------------------------------------------------
def test_recording_round_trip_and_replay(tmp_path):
    from invisiblekey.calibration import DeskCalibration

    cal = DeskCalibration(image_corners=DEFAULT_CORNERS).model_dump_json()
    path = tmp_path / "rec.jsonl"
    rec = LandmarkRecorder(path, cal)
    frames = sim_frames("hi there")
    for f in frames:
        rec.write(f)
    rec.close()
    cal2, loaded = load_recording(path)
    assert cal2 == cal and len(loaded) == len(frames)
    np.testing.assert_allclose(loaded[5].hands[0].landmarks, frames[5].hands[0].landmarks, atol=1e-5)
    out = replay(path, expected="hi there")
    assert out["typed"] == "hi there" and out["char_error_rate"] == 0


# --- camera (fake device) -------------------------------------------------------
class FakeCap:
    def __init__(self, ok=True):
        self.ok, self.n = ok, 0

    def isOpened(self):
        return self.ok

    def set(self, *a):
        return True

    def read(self):
        self.n += 1
        time.sleep(0.002)
        return True, np.full((4, 4, 3), self.n % 255, np.uint8)

    def release(self):
        self.ok = False


def test_camera_returns_newest_frames_and_closes():
    pytest.importorskip("cv2")
    from invisiblekey.vision.camera import CameraManager, list_cameras

    with CameraManager(0, factory=lambda i: FakeCap()) as cam:
        a = cam.read()
        time.sleep(0.03)
        b = cam.read()
        assert a and b and b.seq > a.seq + 1 and cam.dropped > 0
    assert cam._cap is None
    with pytest.raises(RuntimeError):
        CameraManager(0, factory=lambda i: FakeCap(ok=False)).open()
    assert list_cameras(3, factory=lambda i: FakeCap(ok=i == 1)) == [1]


# --- debug app state machine ----------------------------------------------------
class FakeHotkey:
    def __init__(self, cb, fail=False):
        self.cb, self.fail = cb, fail

    def start(self):
        if self.fail:
            raise PermissionError("input monitoring not granted")


def calibrated_app(backend=None, hotkey=FakeHotkey):
    mgr = SettingsManager()
    app = DebugApp(AppSettings(), mgr, backend, hotkey)
    assert app.mode == "calibrate"
    for x, y in DEFAULT_CORNERS:
        app.click(x, y)
    app.click(0.5, 0.5)  # extra clicks ignored once complete
    app.key(13)
    assert app.mode == "type" and mgr.load().calibration is not None
    return app


def test_calibrate_then_type():
    app = calibrated_app()
    for f in sim_frames("ok go"):
        app.process(f, time.monotonic())
    assert app.pipe.text == "ok go"
    assert app.pipe.monitor.snapshot()["e2e_ms_p50"] is not None


def test_output_requires_backend_and_working_hotkey():
    assert not (app := calibrated_app()).toggle_output() and not app.output_enabled
    app = calibrated_app(RecordingBackend(), lambda cb: FakeHotkey(cb, fail=True))
    app.toggle_output()
    assert not app.output_enabled and "hotkey failed" in app.status


def test_keys_ignored_while_output_on_and_emergency_stop():
    be = RecordingBackend()
    app = calibrated_app(be)
    app.key(ord("o"))
    assert app.output_enabled
    app.key(ord("q"))
    assert app.running                       # typed 'q' must not quit the app
    app.hotkey.cb()                          # global hotkey fires
    assert not app.output_enabled and "EMERGENCY" in app.status
    app.key(ord("o"))
    app.key(27)                              # ESC also turns output off
    assert not app.output_enabled


def test_tuning_keys_change_live_config_and_recalibration():
    app = calibrated_app()
    v, w = app.settings.detection.descend_velocity, app.settings.layout.width
    app.key(ord("+"))
    app.key(ord("["))
    app.key(ord("l"))
    assert app.pipe.detector.config.descend_velocity < v
    assert app.pipe.layout.config.width < w and app.pipe.layout.config.x > 0
    app.key(ord("c"))
    assert app.mode == "calibrate" and app.calib.corners == []
    app.key(27)
    assert app.mode == "type"


def test_overlay_renders():
    pytest.importorskip("cv2")
    from invisiblekey.vision import overlay as ov

    img = np.zeros((360, 640, 3), np.uint8)
    tf, lay = CoordinateTransformer(DEFAULT_CORNERS), VirtualKeyboardLayout()
    frame = sim_frames("a")[20]
    ov.draw_hands(img, frame)
    ov.draw_keyboard(img, lay, tf, {"a": (0, 255, 0)})
    ov.draw_fingertips(img, {"left.index": (0.3, 0.6)}, {"left.index": "contact"}, tf)
    ov.draw_corners(img, DEFAULT_CORNERS[:2], "bottom_right")
    ov.draw_hud(img, ["x"])
    ov.draw_text_panel(img, "hello\nworld")
    assert img.any()
