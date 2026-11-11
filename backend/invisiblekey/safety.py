"""Global emergency stop hotkey (works while another app has focus)."""
from __future__ import annotations

import logging
from collections.abc import Callable

log = logging.getLogger(__name__)

EMERGENCY_COMBO = "<ctrl>+<alt>+k"  # macOS: Control + Option + K


class EmergencyHotkey:
    """Requires pynput. macOS: grant Input Monitoring to the terminal/app."""

    def __init__(self, on_trigger: Callable[[], None], combo: str = EMERGENCY_COMBO) -> None:
        from pynput import keyboard

        self.combo = combo
        self._listener = keyboard.GlobalHotKeys({combo: self._fire})
        self._on_trigger = on_trigger
        self.triggered = 0

    def _fire(self) -> None:
        self.triggered += 1
        try:
            self._on_trigger()
        except Exception:  # never let the listener thread die
            log.exception("emergency stop callback failed")

    def start(self) -> None:
        self._listener.start()

    def stop(self) -> None:
        self._listener.stop()

    @property
    def alive(self) -> bool:
        return self._listener.is_alive()
