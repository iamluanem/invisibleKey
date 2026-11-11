"""Debug rendering: landmarks, finger phases, projected keyboard, HUD."""
from __future__ import annotations

import cv2
import numpy as np

from ..geometry import CoordinateTransformer
from ..layout import VirtualKeyboardLayout
from ..observations import FrameObservation

HAND_EDGES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
              (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (17, 18), (18, 19), (19, 20), (0, 17)]
PHASE_COLORS = {  # BGR
    "hover": (200, 200, 200), "moving": (255, 190, 0), "descending": (0, 200, 255),
    "contact": (0, 0, 255), "lifting": (0, 255, 120), "lost": (90, 90, 90),
}
ASCII_LABELS = {"backspace": "BKSP", "enter": "ENTER", "shift_l": "SHIFT", "shift_r": "SHIFT", "space": "SPACE"}
FONT = cv2.FONT_HERSHEY_SIMPLEX


def _px(img: np.ndarray, pts) -> np.ndarray:
    h, w = img.shape[:2]
    return (np.asarray(pts, float) * [w, h]).astype(np.int32)


def draw_hands(img: np.ndarray, frame: FrameObservation) -> None:
    for hand in frame.hands:
        p = _px(img, hand.landmarks[:, :2])
        for a, b in HAND_EDGES:
            cv2.line(img, tuple(p[a]), tuple(p[b]), (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(img, f"{hand.handedness} {hand.score:.2f}", tuple(p[0] + [0, 20]), FONT, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)


def draw_keyboard(img: np.ndarray, layout: VirtualKeyboardLayout, tf: CoordinateTransformer,
                  highlights: dict[str, tuple[int, int, int]] | None = None) -> None:
    highlights = highlights or {}
    overlay = img.copy()
    for key in layout.keys:
        x, y, w, h = layout.key_rect_surface(key)
        quad = _px(img, tf.to_image([[x, y], [x + w, y], [x + w, y + h], [x, y + h]]))
        color = highlights.get(key.id)
        if color:
            cv2.fillPoly(overlay, [quad], color)
        cv2.polylines(img, [quad], True, (255, 220, 120), 1, cv2.LINE_AA)
        c = quad.mean(axis=0).astype(int)
        label = ASCII_LABELS.get(key.id, key.label)
        cv2.putText(img, label, (c[0] - 5 * len(label), c[1] + 5), FONT, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.45, img, 0.55, 0, dst=img)


def draw_fingertips(img: np.ndarray, tips: dict[str, tuple[float, float]], phases: dict[str, str],
                    tf: CoordinateTransformer) -> None:
    if not tips:
        return
    names = list(tips)
    pts = _px(img, tf.to_image([tips[n] for n in names]))
    for n, p in zip(names, pts):
        cv2.circle(img, tuple(p), 6, PHASE_COLORS.get(phases.get(n, "hover"), (255, 255, 255)), -1, cv2.LINE_AA)


def draw_corners(img: np.ndarray, corners: list[tuple[float, float]], next_name: str | None) -> None:
    p = _px(img, corners) if corners else []
    for i, c in enumerate(p):
        cv2.circle(img, tuple(c), 7, (0, 255, 255), -1, cv2.LINE_AA)
        if i:
            cv2.line(img, tuple(p[i - 1]), tuple(c), (0, 255, 255), 2, cv2.LINE_AA)
    if len(p) == 4:
        cv2.line(img, tuple(p[3]), tuple(p[0]), (0, 255, 255), 2, cv2.LINE_AA)
    msg = f"Click the {next_name.replace('_', ' ').upper()} corner of the typing area" if next_name else \
        "ENTER = save | U = undo | R = reset"
    draw_banner(img, msg, (0, 200, 255))


def draw_banner(img: np.ndarray, text: str, color=(0, 200, 255)) -> None:
    cv2.rectangle(img, (0, 0), (img.shape[1], 34), (20, 20, 20), -1)
    cv2.putText(img, text, (12, 23), FONT, 0.6, color, 1, cv2.LINE_AA)


def draw_hud(img: np.ndarray, lines: list[str]) -> None:
    y = 56
    for line in lines:
        cv2.putText(img, line, (12, y), FONT, 0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(img, line, (12, y), FONT, 0.5, (230, 255, 230), 1, cv2.LINE_AA)
        y += 20


def draw_text_panel(img: np.ndarray, text: str) -> None:
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, h - 44), (w, h), (15, 15, 15), -1)
    shown = text.replace("\n", " / ")[-max(10, w // 11):]
    cv2.putText(img, shown + "_", (12, h - 15), FONT, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
