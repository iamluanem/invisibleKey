"""Synthetic MediaPipe-shaped landmark sequences: test and tune without a camera.

    python -m invisiblekey.simulation --text "hello world" --noise 0.002 --dropout 0.05
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass

import numpy as np

from .dispatcher import KeyboardEventDispatcher, RecordingBackend
from .geometry import CoordinateTransformer
from .layout import UNSHIFTED, VirtualKeyboardLayout
from .observations import FINGERS, JOINTS, FingerId, FrameObservation, HandObservation
from .pipeline import TypingPipeline

# Perspective of a camera looking at the desk from the front and above.
DEFAULT_CORNERS = [(0.2, 0.45), (0.8, 0.45), (0.95, 0.9), (0.05, 0.9)]

_ASSIGN = {
    ("left", "pinky"): ["1", "q", "a", "z", "shift_l"],
    ("left", "ring"): ["2", "w", "s", "x"],
    ("left", "middle"): ["3", "e", "d", "c"],
    ("left", "index"): ["4", "5", "r", "t", "f", "g", "v", "b"],
    ("right", "index"): ["6", "7", "y", "u", "h", "j", "n", "m"],
    ("right", "middle"): ["8", "i", "k", ","],
    ("right", "ring"): ["9", "o", "l", "."],
    ("right", "pinky"): ["0", "-", "=", "p", "[", "]", ";", "'", "/", "enter", "backspace", "shift_r"],
    ("right", "thumb"): ["space"],
}
FINGER_FOR_KEY: dict[str, FingerId] = {k: f for f, keys in _ASSIGN.items() for k in keys}
HOME_KEYS = {("left", "pinky"): "a", ("left", "ring"): "s", ("left", "middle"): "d",
             ("left", "index"): "f", ("right", "index"): "j", ("right", "middle"): "k",
             ("right", "ring"): "l", ("right", "pinky"): ";"}


def keys_for_text(text: str) -> list[str]:
    seq = []
    for ch in text:
        if ch == " ":
            seq.append("space")
        elif ch == "\n":
            seq.append("enter")
        elif ch == "\b":
            seq.append("backspace")
        else:
            base = ch.lower() if ch.isalpha() else UNSHIFTED.get(ch, ch)
            if base not in FINGER_FOR_KEY:
                raise ValueError(f"unsupported character: {ch!r}")
            if base != ch:
                seq.append("shift_r" if FINGER_FOR_KEY[base][0] == "left" else "shift_l")
            seq.append(base)
    return seq


@dataclass
class SimConfig:
    fps: float = 30.0
    hand_scale: float = 0.10     # wrist→middle-MCP distance, image units
    travel: float = 0.25         # tap depth, in hand-scale units
    move_frames: int = 4
    descend_frames: int = 2
    hold_frames: int = 1
    lift_frames: int = 3
    pause_frames: int = 3
    noise: float = 0.0           # landmark jitter std, image units
    dropout: float = 0.0         # probability a hand is missing in a frame
    tracking_score: float = 0.95
    seed: int = 0


class SyntheticTypist:
    def __init__(self, transformer: CoordinateTransformer, layout: VirtualKeyboardLayout,
                 config: SimConfig | None = None) -> None:
        self.tf, self.layout, self.cfg = transformer, layout, config or SimConfig()
        self.rng = np.random.default_rng(self.cfg.seed)
        self.frame_idx = 0
        self.impacts: list[tuple[str, float]] = []  # ground truth: (key_id, contact time)
        self.press: dict[FingerId, float] = {(h, f): 0.0 for h in ("left", "right") for f in FINGERS}
        self.hover: dict[FingerId, np.ndarray] = {}
        for fid, key in HOME_KEYS.items():
            self.hover[fid] = self._hover_over(self.layout.key_center_surface(key))
        sx, sy = self.layout.key_center_surface("space")
        unit = self.layout.config.width / self.layout.UNITS_W
        self.hover[("left", "thumb")] = self._hover_over((sx - 1.5 * unit, sy))
        self.hover[("right", "thumb")] = self._hover_over((sx + 1.5 * unit, sy))

    def _hover_over(self, surface_pt) -> np.ndarray:
        x, y = self.tf.to_image(surface_pt)[0]
        return np.array([x, y - self.cfg.travel * self.cfg.hand_scale])

    def _landmarks(self, hand: str) -> np.ndarray:
        c, lm = self.cfg, np.zeros((21, 3))
        length = 0.6 * c.hand_scale
        for f in FINGERS:
            hx, hy = self.hover[(hand, f)]
            depth = self.press[(hand, f)] * c.travel * c.hand_scale
            ref, knuckle, tip = JOINTS[f]
            lm[knuckle, :2] = hx, hy - 0.8 * length
            lm[ref, :2] = hx, hy - 0.45 * length
            lm[tip - 1, :2] = hx, hy - 0.2 * length + 0.5 * depth
            lm[tip, :2] = hx, hy + depth
        lm[0, :2] = lm[9, 0], lm[9, 1] - c.hand_scale
        if c.noise:
            lm[:, :2] += self.rng.normal(0, c.noise, (21, 2))
        return lm

    def emit(self) -> FrameObservation:
        t = self.frame_idx / self.cfg.fps
        self.frame_idx += 1
        hands = [HandObservation(h, self._landmarks(h), self.cfg.tracking_score)
                 for h in ("left", "right") if self.rng.random() >= self.cfg.dropout]
        return FrameObservation(t, hands)

    def idle(self, n: int) -> list[FrameObservation]:
        return [self.emit() for _ in range(n)]

    def keystroke(self, key_id: str) -> list[FrameObservation]:
        c, finger = self.cfg, FINGER_FOR_KEY[key_id]
        start, target = self.hover[finger].copy(), self._hover_over(self.layout.key_center_surface(key_id))
        frames = []
        for i in range(1, c.move_frames + 1):
            self.hover[finger] = start + (target - start) * i / c.move_frames
            frames.append(self.emit())
        for i in range(1, c.descend_frames + 1):
            self.press[finger] = i / c.descend_frames
            frames.append(self.emit())
        self.impacts.append((key_id, frames[-1].timestamp))
        frames += self.idle(c.hold_frames)
        for i in range(1, c.lift_frames + 1):
            self.press[finger] = 1 - i / c.lift_frames
            frames.append(self.emit())
        return frames + self.idle(c.pause_frames)

    def type_text(self, text: str, settle: int = 10) -> list[FrameObservation]:
        frames = self.idle(settle)
        for k in keys_for_text(text):
            frames += self.keystroke(k)
        return frames + self.idle(settle)

    def slow_press(self, finger: FingerId, seconds: float, depth: float = 1.0) -> list[FrameObservation]:
        """Resting a finger down slowly — must NOT be a tap."""
        n = max(2, int(seconds * self.cfg.fps))
        frames = []
        for i in range(1, n + 1):
            self.press[finger] = depth * i / n
            frames.append(self.emit())
        frames += self.idle(5)
        for i in range(1, n + 1):
            self.press[finger] = depth * (1 - i / n)
            frames.append(self.emit())
        return frames

    def shift_hand(self, hand: str, dx: float, dy: float, frames: int) -> list[FrameObservation]:
        """Whole-hand translation (repositioning) — must NOT be a tap."""
        start = {f: self.hover[(hand, f)].copy() for f in FINGERS}
        out = []
        for i in range(1, frames + 1):
            for f in FINGERS:
                self.hover[(hand, f)] = start[f] + np.array([dx, dy]) * i / frames
            out.append(self.emit())
        return out


def levenshtein(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def run_simulation(text: str, config: SimConfig | None = None) -> dict:
    tf = CoordinateTransformer(DEFAULT_CORNERS)
    layout = VirtualKeyboardLayout()
    backend = RecordingBackend()
    dispatcher = KeyboardEventDispatcher(backend)
    dispatcher.enable()
    pipe = TypingPipeline(tf, layout, dispatcher=dispatcher)
    typist = SyntheticTypist(tf, layout, config)
    taps = [ev for frame in typist.type_text(text) for ev in pipe.process(frame).taps if ev.dispatched]
    delays, used = [], set()
    for key, t in typist.impacts:  # match each true keystroke to the first detection of that key
        for i, ev in enumerate(taps):
            if i not in used and ev.key_id == key and ev.detected_time >= t - 0.05:
                used.add(i)
                delays.append((ev.detected_time - t) * 1000)
                break
    errors = levenshtein(pipe.text, text)
    return {
        "expected": text,
        "typed": pipe.text,
        "char_error_rate": round(errors / max(1, len(text)), 4),
        "os_events": len(backend.events),
        "keystrokes": len(typist.impacts),
        "missed": len(typist.impacts) - len(delays),
        "spurious": len(taps) - len(used),
        "true_detection_delay_ms_p50": round(float(np.median(delays)), 1) if delays else None,
        "metrics": pipe.monitor.snapshot(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Run the tap pipeline on synthetic landmarks")
    ap.add_argument("--text", default="hello world")
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--dropout", type=float, default=0.0)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = SimConfig(noise=a.noise, dropout=a.dropout, fps=a.fps, seed=a.seed)
    print(json.dumps(run_simulation(a.text, cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
