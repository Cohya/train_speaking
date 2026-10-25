"""Injectable clocks; session elapsed time excludes paused periods."""

from __future__ import annotations

import time
from typing import Protocol


class Clock(Protocol):
    def now(self) -> float: ...


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self._now = start

    def now(self) -> float:
        return self._now

    def advance(self, seconds: float) -> None:
        if seconds < 0:
            raise ValueError("cannot move a clock backwards")
        self._now += seconds


class SessionClock:
    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._started_at: float | None = None
        self._paused_since: float | None = None
        self._paused_total = 0.0

    @property
    def started(self) -> bool:
        return self._started_at is not None

    @property
    def paused(self) -> bool:
        return self._paused_since is not None

    def start(self) -> None:
        if self._started_at is None:
            self._started_at = self._clock.now()

    def pause(self) -> None:
        if self._started_at is not None and self._paused_since is None:
            self._paused_since = self._clock.now()

    def resume(self) -> None:
        if self._paused_since is not None:
            self._paused_total += self._clock.now() - self._paused_since
            self._paused_since = None

    def elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        end = self._paused_since if self._paused_since is not None else self._clock.now()
        return max(0.0, end - self._started_at - self._paused_total)
