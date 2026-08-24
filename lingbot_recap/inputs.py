from __future__ import annotations

import os
import select
import sys
import termios
import tty
from collections import deque
from contextlib import AbstractContextManager
from typing import Protocol

from .types import InputEvent


class EventSource(Protocol):
    def poll(self) -> InputEvent | None: ...


KEYMAP = {
    " ": InputEvent.TAKEOVER_OR_HAND_BACK,
    "s": InputEvent.SUCCESS,
    "f": InputEvent.FAILURE,
    "r": InputEvent.RESUME_AUTO,
    "q": InputEvent.QUIT,
    "\x1b": InputEvent.QUIT,
}


class KeyboardEventSource(AbstractContextManager):
    """Non-blocking terminal keys. A USB pedal mapped to Space works unchanged."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdin
        self._previous = None

    def __enter__(self):
        if not self.stream.isatty():
            raise RuntimeError("keyboard takeover requires an interactive TTY")
        self._previous = termios.tcgetattr(self.stream.fileno())
        tty.setcbreak(self.stream.fileno())
        return self

    def poll(self) -> InputEvent | None:
        ready, _, _ = select.select([self.stream], [], [], 0)
        if not ready:
            return None
        return KEYMAP.get(os.read(self.stream.fileno(), 1).decode(errors="ignore"))

    def __exit__(self, exc_type, exc, tb):
        if self._previous is not None:
            termios.tcsetattr(self.stream.fileno(), termios.TCSADRAIN, self._previous)
        return False


class QueueEventSource:
    """Test/button adapter: enqueue logical events from any external device."""

    def __init__(self):
        self.events = deque()

    def put(self, event: InputEvent) -> None:
        self.events.append(event)

    def poll(self) -> InputEvent | None:
        return self.events.popleft() if self.events else None
