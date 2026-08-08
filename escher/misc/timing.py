"""Per-phase wall-clock timing for the optimization loops (``TIMING: true``).

The only timing datum the project had was a cumulative s/step average and one
measured anecdote (the split-backward note in ``main_sphere.step``); every speedup
decision needs a per-phase breakdown instead of a guess. Disabled (the default) the
timer is a strict no-op, so ordinary runs are byte-identical with or without it.
"""

from __future__ import annotations

import time
from contextlib import contextmanager


class PhaseTimer:
    """Accumulate wall-clock seconds per named phase between resets.

    ``cuda_sync`` inserts ``torch.cuda.synchronize()`` at each phase boundary so
    asynchronous GPU work is billed to the phase that launched it rather than to
    whichever phase happens to block next. The synchronization itself slows the
    loop, which is why timing is opt-in and never on by default.
    """

    def __init__(self, enabled: bool = False, cuda_sync: bool = False) -> None:
        self.enabled = bool(enabled)
        self.cuda_sync = bool(cuda_sync)
        self.totals: dict[str, float] = {}
        self.steps = 0

    def _sync(self) -> None:
        if self.cuda_sync:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()

    @contextmanager
    def phase(self, name: str):
        if not self.enabled:
            yield
            return
        self._sync()
        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.totals[name] = self.totals.get(name, 0.0) + time.perf_counter() - t0

    def tick(self) -> None:
        """Count one loop iteration against the current window."""
        if self.enabled:
            self.steps += 1

    def means_ms(self, names) -> list[float]:
        """Mean milliseconds per counted iteration, in the order of ``names``.

        Phases that fire less often than every iteration (snapshots, checkpoints)
        come back amortized over the window, which is the number that matters for
        s/step.
        """
        n = max(self.steps, 1)
        return [1000.0 * self.totals.get(name, 0.0) / n for name in names]

    def reset(self) -> None:
        self.totals.clear()
        self.steps = 0
