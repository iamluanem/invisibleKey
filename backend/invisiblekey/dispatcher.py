"""Keyboard output: in-app text buffer always, OS injection only when enabled."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Protocol

from .layout import Key, SpecialKey, shifted

log = logging.getLogger(__name__)


class InputBackend(Protocol):
    def send_text(self, text: str) -> None: ...
    def send_special(self, key: SpecialKey) -> None: ...


class RecordingBackend:
    """Test/simulation backend that records what would be sent to the OS."""

    def __init__(self) -> None:
        self.events: list[str] = []

    def send_text(self, text: str) -> None:
        self.events.append(text)

    def send_special(self, key: SpecialKey) -> None:
        self.events.append(f"<{key.value}>")


class PynputBackend:
    """Cross-platform injection via pynput (macOS needs Accessibility permission)."""

    def __init__(self) -> None:
        from pynput.keyboard import Controller, Key as PKey  # lazy: optional dependency

        self._kb = Controller()
        self._map = {SpecialKey.SPACE: PKey.space, SpecialKey.ENTER: PKey.enter,
                     SpecialKey.BACKSPACE: PKey.backspace}

    def send_text(self, text: str) -> None:
        self._kb.type(text)

    def send_special(self, key: SpecialKey) -> None:
        k = self._map[key]
        self._kb.press(k)
        self._kb.release(k)


class TextBuffer:
    def __init__(self, max_len: int = 10_000) -> None:
        self.text, self.max_len = "", max_len

    def apply(self, output: str) -> None:
        if output == "<backspace>":
            self.text = self.text[:-1]
        elif output == "<enter>":
            self.text += "\n"
        elif output == "<space>":
            self.text += " "
        else:
            self.text += output
        self.text = self.text[-self.max_len:]

    def clear(self) -> None:
        self.text = ""


@dataclass(slots=True)
class DispatchResult:
    key_id: str
    output: str          # character, or "<special>"
    sent_to_os: bool
    error: str | None = None


class KeyboardEventDispatcher:
    """OS output starts DISABLED. Shift is one-shot (sticky): tap Shift, then a key."""

    def __init__(self, backend: InputBackend | None = None) -> None:
        self.backend = backend
        self.enabled = False
        self.shift_armed = False
        self.buffer = TextBuffer()
        self.last_error: str | None = None

    def enable(self) -> None:
        if self.backend is None:
            raise RuntimeError("no OS input backend configured")
        self.enabled, self.last_error = True, None

    def disable(self) -> None:
        self.enabled = False

    def emergency_stop(self) -> None:
        self.enabled, self.shift_armed = False, False
        log.warning("emergency stop: OS keyboard output disabled")

    def dispatch(self, key: Key) -> DispatchResult:
        if key.special is SpecialKey.SHIFT:
            self.shift_armed = not self.shift_armed
            return DispatchResult(key.id, "<shift>", False)
        if key.special:
            output, char = f"<{key.special.value}>", None
        else:
            char = shifted(key.char) if self.shift_armed else key.char
            output = char
        self.shift_armed = False
        self.buffer.apply(output)
        if not (self.enabled and self.backend):
            return DispatchResult(key.id, output, False)
        try:
            if char is None:
                self.backend.send_special(key.special)
            else:
                self.backend.send_text(char)
            return DispatchResult(key.id, output, True)
        except Exception as exc:  # fail safe: never keep injecting after an error
            self.enabled, self.last_error = False, str(exc)
            log.exception("OS input failed; output disabled")
            return DispatchResult(key.id, output, False, error=str(exc))
