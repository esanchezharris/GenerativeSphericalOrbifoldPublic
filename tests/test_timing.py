"""The per-phase timer must be a strict no-op when disabled and honest when on."""

from __future__ import annotations

import time

from escher.misc.timing import PhaseTimer


def test_disabled_timer_records_nothing():
    t = PhaseTimer(enabled=False)
    with t.phase("solve"):
        pass
    t.tick()
    assert t.totals == {}
    assert t.steps == 0
    assert t.means_ms(("solve", "render")) == [0.0, 0.0]


def test_enabled_timer_accumulates_and_resets():
    t = PhaseTimer(enabled=True)
    for _ in range(4):
        with t.phase("solve"):
            time.sleep(0.002)
        t.tick()
    # snapshot-style phase firing once in the window is amortized over the ticks
    with t.phase("snapshot"):
        time.sleep(0.002)

    solve_ms, snap_ms, missing_ms = t.means_ms(("solve", "snapshot", "render"))
    assert solve_ms >= 2.0, "mean per tick must reflect the sleep in every tick"
    assert 0.0 < snap_ms < solve_ms, "one firing amortized over 4 ticks"
    assert missing_ms == 0.0

    t.reset()
    assert t.totals == {} and t.steps == 0


def test_timer_propagates_exceptions_but_still_records():
    t = PhaseTimer(enabled=True)
    try:
        with t.phase("solve"):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    assert "solve" in t.totals, "the finally block must record the elapsed time"
