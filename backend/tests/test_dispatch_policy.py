import pytest

from invisiblekey.dispatcher import KeyboardEventDispatcher, RecordingBackend
from invisiblekey.layout import VirtualKeyboardLayout
from invisiblekey.policy import KeyPressPolicy

L = VirtualKeyboardLayout()
A, B = ("right", "index"), ("right", "middle")


def test_policy_suppresses_cross_finger_duplicate():
    p = KeyPressPolicy()
    assert p.allow(A, "j", 1.0)
    assert not p.allow(B, "j", 1.03)      # 30 ms, another finger → duplicate
    assert p.allow(B, "j", 1.2)           # later → legitimate


def test_policy_allows_deliberate_repeat_but_not_bounce():
    p = KeyPressPolicy()
    assert p.allow(A, "l", 1.0)
    assert not p.allow(A, "l", 1.04)      # 40 ms: physically implausible repeat
    assert p.allow(A, "l", 1.15)          # "ll" in "hello"


def test_output_disabled_by_default_but_buffer_updates():
    be = RecordingBackend()
    d = KeyboardEventDispatcher(be)
    r = d.dispatch(L.key("h"))
    assert r.output == "h" and not r.sent_to_os and be.events == [] and d.buffer.text == "h"


def test_enabled_sends_text_and_specials():
    be = RecordingBackend()
    d = KeyboardEventDispatcher(be)
    d.enable()
    for k in ["h", "i", "space", "backspace", "enter"]:
        d.dispatch(L.key(k))
    assert be.events == ["h", "i", "<space>", "<backspace>", "<enter>"]
    assert d.buffer.text == "hi\n"


def test_one_shot_shift():
    d = KeyboardEventDispatcher(RecordingBackend())
    for k in ["shift_l", "h", "i", "shift_r", "1"]:
        d.dispatch(L.key(k))
    assert d.buffer.text == "Hi!"


def test_emergency_stop_blocks_output_immediately():
    be = RecordingBackend()
    d = KeyboardEventDispatcher(be)
    d.enable()
    d.dispatch(L.key("a"))
    d.dispatch(L.key("shift_l"))
    d.emergency_stop()
    r = d.dispatch(L.key("b"))
    assert be.events == ["a"] and not r.sent_to_os and r.output == "b"  # shift cleared too


def test_backend_failure_disables_output():
    class Broken(RecordingBackend):
        def send_text(self, text):
            raise PermissionError("accessibility not granted")

    d = KeyboardEventDispatcher(Broken())
    d.enable()
    r = d.dispatch(L.key("a"))
    assert r.error and not d.enabled and "accessibility" in d.last_error


def test_enable_without_backend_raises():
    with pytest.raises(RuntimeError):
        KeyboardEventDispatcher().enable()
