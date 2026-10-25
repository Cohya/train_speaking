from __future__ import annotations

import pytest

from train_speaking.session.clock import FakeClock, SessionClock


def test_elapsed_is_zero_before_start() -> None:
    clock = FakeClock()
    session_clock = SessionClock(clock)
    clock.advance(5)
    assert session_clock.elapsed() == 0.0
    assert not session_clock.started


def test_elapsed_excludes_paused_time() -> None:
    clock = FakeClock(100.0)
    session_clock = SessionClock(clock)
    session_clock.start()
    clock.advance(10)
    session_clock.pause()
    clock.advance(30)
    assert session_clock.elapsed() == pytest.approx(10)
    session_clock.resume()
    clock.advance(5)
    assert session_clock.elapsed() == pytest.approx(15)


def test_repeated_pause_and_start_are_idempotent() -> None:
    clock = FakeClock()
    session_clock = SessionClock(clock)
    session_clock.start()
    clock.advance(4)
    session_clock.start()
    session_clock.pause()
    clock.advance(2)
    session_clock.pause()
    session_clock.resume()
    session_clock.resume()
    clock.advance(1)
    assert session_clock.elapsed() == pytest.approx(5)


def test_fake_clock_cannot_go_backwards() -> None:
    with pytest.raises(ValueError):
        FakeClock().advance(-1)
