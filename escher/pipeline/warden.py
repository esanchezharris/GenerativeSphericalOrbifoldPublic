"""GPU-warden integration: wrap GPU-lane children in a heartbeated lease.

Opt-in (``--warden``): on this machine the warden's busy-detection counts two
permanently-resident Claude desktop helper processes as live GPU workloads, so a
wrapped command queues forever unless they are excluded. When enabled, the child
runs under ``gpu.py run --wait``, which acquires the lease, heartbeats from a
daemon thread, records the child PID, releases in a finally, and propagates the
child's exit code -- so the stage exit-code contract passes straight through.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


def wrap(cmd: list[str], use_warden: bool, label: str) -> list[str]:
    if not use_warden:
        return cmd
    gpu_dir = Path(os.environ.get("AGENT_GPU_DIR", "/mnt/c/Users/dawha/.agent-gpu"))
    gpu_py = gpu_dir / "gpu.py"
    if not gpu_py.exists():
        print(f"warden requested but {gpu_py} not found -- running unwrapped")
        return cmd
    return [
        sys.executable,
        str(gpu_py),
        "run",
        "--agent",
        "orbifold-pipeline",
        "--project",
        label,
        "--kind",
        "training",
        "--eta",
        "2h",
        "--wait",
        "--",
        *cmd,
    ]
