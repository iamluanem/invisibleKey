"""Duplicate suppression that still allows deliberate repeats."""
from __future__ import annotations

from .config import PolicyConfig
from .observations import FingerId


class KeyPressPolicy:
    """Same-finger repeats are gated by the detector's re-arm (finger must lift).
    This adds two cross-checks:
      * same key from a *different* finger within a short window -> duplicate
        (sympathetic finger motion, or two fingers resolving to one key);
      * same finger+key faster than a human can repeat -> bounce.
    """

    def __init__(self, config: PolicyConfig | None = None) -> None:
        self.config = config or PolicyConfig()
        self._last: dict[str, tuple[FingerId, float]] = {}

    def allow(self, finger: FingerId, key_id: str, t: float) -> bool:
        c, last = self.config, self._last.get(key_id)
        if last:
            gap_ms = (t - last[1]) * 1000
            if last[0] != finger and gap_ms < c.same_key_window_ms:
                return False
            if last[0] == finger and gap_ms < c.min_repeat_interval_ms:
                return False
        self._last[key_id] = (finger, t)
        return True

    def reset(self) -> None:
        self._last.clear()
