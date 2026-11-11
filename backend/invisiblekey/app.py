"""Local debug app (OpenCV window): live camera, calibration wizard, typing.

    python -m invisiblekey.app [--camera 0] [--no-mirror] [--record session.jsonl]

Calibration: click the 4 corners of the desk typing area (TL, TR, BR, BL), ENTER saves.
Typing mode keys (window focused, OS output OFF):
  C recalibrate · O toggle OS output · +/- sensitivity · [ ] resize keyboard
  I J K L move keyboard · S save settings · Z clear text · R record · Q quit
While OS output is ON the window ignores keys (they may be typed by us!) except ESC,
which turns output off. Global emergency stop from any app: Control+Option+K.
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from .calibration import DeskCalibrationService
from .layout import VirtualKeyboardLayout
from .pipeline import TypingPipeline, build_pipeline
from .recording import LandmarkRecorder
from .settings import AppSettings, SettingsManager

log = logging.getLogger(__name__)
WIN = "Invisible AI Keyboard"
FLASH_S = 0.35


class DebugApp:
    """UI-agnostic state machine; main() plugs in OpenCV, the camera and MediaPipe."""

    def __init__(self, settings: AppSettings, manager: SettingsManager | None = None,
                 backend=None, hotkey_factory=None) -> None:
        self.settings, self.manager = settings, manager
        self.backend, self.hotkey_factory, self.hotkey = backend, hotkey_factory, None
        self.calib = DeskCalibrationService()
        self.pipe: TypingPipeline | None = None
        self.mode = "calibrate"
        self.running = True
        self.recorder: LandmarkRecorder | None = None
        self.flashes: dict[str, tuple[tuple[int, int, int], float]] = {}
        self.status = ""
        self.frame_size: tuple[int, int] | None = None
        if settings.calibration is not None:
            self._start_typing()

    # --- state ------------------------------------------------------------
    def _start_typing(self) -> None:
        old = self.pipe
        self.pipe = build_pipeline(self.settings, self.backend)
        if old:
            self.pipe.dispatcher.buffer.text = old.dispatcher.buffer.text
        self.mode = "type"

    @property
    def output_enabled(self) -> bool:
        return bool(self.pipe and self.pipe.dispatcher.enabled)

    def emergency_stop(self) -> None:
        if self.pipe:
            self.pipe.dispatcher.emergency_stop()
        self.status = "EMERGENCY STOP - OS output OFF"

    def toggle_output(self) -> None:
        if self.output_enabled:
            self.pipe.dispatcher.disable()
            self.status = "OS output OFF"
            return
        if self.backend is None:
            self.status = "OS output unavailable: pip install pynput"
            return
        if self.hotkey is None:  # never inject keys without a working global kill switch
            try:
                self.hotkey = self.hotkey_factory(self.emergency_stop)
                self.hotkey.start()
            except Exception as exc:
                self.hotkey, self.status = None, f"emergency hotkey failed ({exc}); output stays OFF"
                return
        self.pipe.dispatcher.enable()
        self.status = "OS output ON - Ctrl+Option+K to stop"

    # --- input ------------------------------------------------------------
    def click(self, x: float, y: float) -> None:
        if self.mode == "calibrate" and not self.calib.complete:
            self.calib.add_corner(x, y)

    def key(self, ch: int) -> None:
        if ch < 0:
            return
        if self.output_enabled:
            if ch == 27:
                self.toggle_output()
            return
        c = chr(ch).lower() if ch < 128 else ""
        if self.mode == "calibrate":
            if c == "u":
                self.calib.undo()
            elif c == "r":
                self.calib.reset()
            elif ch in (10, 13) and self.calib.complete:
                self.settings.calibration = self.calib.build(str(self.settings.camera_index))
                self._save()
                self._start_typing()
            elif ch == 27:
                if self.settings.calibration is not None:
                    self.mode = "type"
                else:
                    self.running = False
            return
        det, lay = self.settings.detection, self.settings.layout
        if c == "q":
            self.running = False
        elif c == "c":
            self.calib.reset()
            self.mode = "calibrate"
        elif c == "o":
            self.toggle_output()
        elif c in "+=":
            det.descend_velocity = max(0.3, det.descend_velocity * 0.9)   # more sensitive
        elif c == "-":
            det.descend_velocity = min(5.0, det.descend_velocity * 1.1)
        elif c in "[]":
            f = 0.95 if c == "[" else 1.05
            lay.width, lay.height = lay.width * f, lay.height * f
        elif c in "ijkl":
            dx, dy = {"i": (0, -0.01), "k": (0, 0.01), "j": (-0.01, 0), "l": (0.01, 0)}[c]
            lay.x, lay.y = lay.x + dx, lay.y + dy
        elif c == "s":
            self._save()
        elif c == "z":
            self.pipe.dispatcher.buffer.clear()

    def _save(self) -> None:
        if self.manager:
            self.manager.save(self.settings)
            self.status = f"saved {self.manager.path}"

    # --- frames -----------------------------------------------------------
    def process(self, obs, capture_time: float, now: float | None = None):
        if self.recorder:
            self.recorder.write(obs)
        if self.mode != "type":
            return None
        res = self.pipe.process(obs)
        now = time.monotonic() if now is None else now
        for ev in res.taps:
            if ev.key_id and ev.decision in ("accepted", "uncertain"):
                color = (0, 220, 0) if ev.dispatched else (0, 200, 255)
                self.flashes[ev.key_id] = (color, now + FLASH_S)
            if ev.dispatched:
                self.pipe.monitor.record_e2e((time.monotonic() - capture_time) * 1000)
        self.flashes = {k: v for k, v in self.flashes.items() if v[1] > now}
        return res

    def toggle_recording(self, path: Path) -> None:
        if self.recorder:
            self.recorder.close()
            self.status, self.recorder = f"recorded {self.recorder.frames} frames -> {path}", None
        else:
            cal = self.settings.calibration.model_dump_json() if self.settings.calibration else None
            self.recorder = LandmarkRecorder(path, cal)
            self.status = f"recording -> {path}"


def _preview_layout(app: DebugApp):
    """Transformer + layout to draw: saved calibration, or the 4 corners being placed."""
    if app.mode == "type":
        return app.pipe.transformer, app.pipe.layout
    if app.calib.complete:
        return app.calib.build().transformer(), VirtualKeyboardLayout(app.settings.layout)
    return None, None


def main() -> None:
    import cv2

    from .dispatcher import PynputBackend
    from .safety import EmergencyHotkey
    from .vision import overlay as ov
    from .vision.camera import CameraManager, list_cameras
    from .vision.hand_tracking import HandTrackingService

    ap = argparse.ArgumentParser(description="Invisible AI Keyboard - debug app")
    ap.add_argument("--camera", type=int)
    ap.add_argument("--list-cameras", action="store_true")
    ap.add_argument("--no-mirror", action="store_true", help="for cameras looking down from behind the hands")
    ap.add_argument("--record", type=Path, default=Path("recordings") / f"session-{int(time.time())}.jsonl")
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO)
    if a.list_cameras:
        print("cameras:", list_cameras())
        return

    mgr = SettingsManager()
    settings = mgr.load()
    if a.camera is not None and a.camera != settings.camera_index:
        settings.camera_index, settings.calibration = a.camera, None  # new camera -> recalibrate
    try:
        backend = PynputBackend()
    except Exception as exc:  # pynput missing or no permission
        log.warning("OS keyboard output unavailable: %s", exc)
        backend = None
    app = DebugApp(settings, mgr, backend, lambda cb: EmergencyHotkey(cb))
    tracker = HandTrackingService(mirror=not a.no_mirror)

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and app.frame_size:
            w, h = app.frame_size
            app.click(x / w, y / h)

    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(WIN, on_mouse)
    with CameraManager(settings.camera_index, a.width, a.height) as cam:
        while app.running:
            cf = cam.read()
            if cf is None:
                continue
            view = tracker.prepare(cf.image)
            app.frame_size = (view.shape[1], view.shape[0])
            obs = tracker.process(view, cf.timestamp)
            res = app.process(obs, cf.timestamp)
            ov.draw_hands(view, obs)
            tf, layout = _preview_layout(app)
            if tf is not None:
                ov.draw_keyboard(view, layout, tf, {k: c for k, (c, _) in app.flashes.items()})
            if app.mode == "calibrate":
                ov.draw_corners(view, app.calib.corners, app.calib.next_corner)
            else:
                ov.draw_fingertips(view, res.fingertips, res.phases, tf)
                m = app.pipe.monitor.snapshot()
                out = "ON (Ctrl+Option+K stops)" if app.output_enabled else "OFF (O enables)"
                ov.draw_banner(view, f"OS output: {out}   {app.status}",
                               (0, 0, 255) if app.output_enabled else (0, 220, 0))
                ov.draw_hud(view, [
                    f"fps {m['fps']}  proc p50 {m['processing_ms_p50']} ms  e2e p50 {m['e2e_ms_p50']} ms",
                    f"taps {m['taps']}  dup {m['suppressed_duplicates']}  wpm {m['wpm']}",
                    f"sensitivity v>{settings.detection.descend_velocity:.2f}  "
                    f"dropped {cam.dropped}  {'REC' if app.recorder else ''}",
                ])
                ov.draw_text_panel(view, app.pipe.text)
            cv2.imshow(WIN, view)
            k = cv2.waitKey(1)
            if not app.output_enabled and k >= 0 and chr(k & 0xFF).lower() == "r" and app.mode == "type":
                app.toggle_recording(a.record)
            else:
                app.key(k & 0xFF if k >= 0 else -1)
    if app.recorder:
        app.recorder.close()
    if app.hotkey:
        app.hotkey.stop()
    tracker.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
