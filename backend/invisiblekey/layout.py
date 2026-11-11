"""Virtual QWERTY layout with key regions in calibrated surface space."""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .config import LayoutConfig


class SpecialKey(str, Enum):
    SPACE = "space"
    ENTER = "enter"
    BACKSPACE = "backspace"
    SHIFT = "shift"


@dataclass(frozen=True, slots=True)
class Key:
    id: str
    label: str
    x: float  # layout units (1 unit = 1 standard key)
    y: float
    w: float = 1.0
    h: float = 1.0
    char: str | None = None
    special: SpecialKey | None = None

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.w / 2, self.y + self.h / 2


@dataclass(frozen=True, slots=True)
class KeyHit:
    key: Key
    inside: bool
    centrality: float  # 1 at center, 0.5 at edge, ->0 at tolerance limit


SHIFTED = dict(zip("1234567890-=[];',./", "!@#$%^&*()_+{}:\"<>?"))
UNSHIFTED = {v: k for k, v in SHIFTED.items()}


def shifted(char: str) -> str:
    return char.upper() if char.isalpha() else SHIFTED.get(char, char)


def _row(chars: str, x0: float, y: float) -> list[Key]:
    return [Key(c, c.upper(), x0 + i, y, char=c) for i, c in enumerate(chars)]


def build_qwerty() -> list[Key]:
    keys = _row("1234567890-=", 0, 0)
    keys.append(Key("backspace", "⌫", 12, 0, w=2, special=SpecialKey.BACKSPACE))
    keys += _row("qwertyuiop[]", 1.5, 1)
    keys += _row("asdfghjkl;'", 1.75, 2)
    keys.append(Key("enter", "⏎", 12.75, 2, w=1.25, special=SpecialKey.ENTER))
    keys.append(Key("shift_l", "⇧", 0, 3, w=2.25, special=SpecialKey.SHIFT))
    keys += _row("zxcvbnm,./", 2.25, 3)
    keys.append(Key("shift_r", "⇧", 12.25, 3, w=1.75, special=SpecialKey.SHIFT))
    keys.append(Key("space", "Space", 4, 4, w=6, special=SpecialKey.SPACE))
    return keys


class VirtualKeyboardLayout:
    UNITS_W = 14.0
    UNITS_H = 5.0

    def __init__(self, config: LayoutConfig | None = None, keys: list[Key] | None = None) -> None:
        self.config = config or LayoutConfig()
        self.keys = keys or build_qwerty()
        self._by_id = {k.id: k for k in self.keys}

    def key(self, key_id: str) -> Key:
        return self._by_id[key_id]

    def to_units(self, u: float, v: float) -> tuple[float, float]:
        c = self.config
        return (u - c.x) / c.width * self.UNITS_W, (v - c.y) / c.height * self.UNITS_H

    def to_surface(self, kx: float, ky: float) -> tuple[float, float]:
        c = self.config
        return c.x + kx / self.UNITS_W * c.width, c.y + ky / self.UNITS_H * c.height

    def key_center_surface(self, key_id: str) -> tuple[float, float]:
        return self.to_surface(*self.key(key_id).center)

    def key_rect_surface(self, key: Key) -> tuple[float, float, float, float]:
        x, y = self.to_surface(key.x, key.y)
        x2, y2 = self.to_surface(key.x + key.w, key.y + key.h)
        return x, y, x2 - x, y2 - y

    def hit_test(self, u: float, v: float) -> KeyHit | None:
        kx, ky = self.to_units(u, v)
        best: tuple[float, Key] | None = None
        for k in self.keys:
            dx = max(k.x - kx, 0.0, kx - (k.x + k.w))
            dy = max(k.y - ky, 0.0, ky - (k.y + k.h))
            if dx == 0 and dy == 0:
                cx, cy = k.center
                off = max(abs(kx - cx) / (k.w / 2), abs(ky - cy) / (k.h / 2))
                return KeyHit(k, True, 1.0 - 0.5 * min(off, 1.0))
            d = (dx * dx + dy * dy) ** 0.5
            if best is None or d < best[0]:
                best = (d, k)
        tol = self.config.hit_tolerance
        if best and tol > 0 and best[0] <= tol:
            return KeyHit(best[1], False, 0.5 * (1 - best[0] / tol))
        return None
