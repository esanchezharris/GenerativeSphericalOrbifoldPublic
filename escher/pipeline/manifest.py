"""Atomic run manifest: prompt -> dirs -> metrics -> status, for humans and --resume.

Nothing in the repo recorded which prompt produced which run dir; reconstruction
was by directory name and stray .log files. The manifest is rewritten atomically
(temp file + os.replace) on every stage transition, so a killed driver leaves a
consistent file and ``--resume`` can skip completed stages.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

DONE = ("ok",)
TERMINAL = ("ok", "failed", "skipped")


class Manifest:
    def __init__(self, path: str | Path, batch_name: str = "") -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        if self.path.exists():
            self.data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self.data = {
                "batch": batch_name,
                "created": time.strftime("%Y-%m-%d %H:%M:%S"),
                "jobs": {},
            }

    def ensure_job(self, job_id: str, **fields) -> None:
        with self._lock:
            job = self.data["jobs"].setdefault(job_id, {"stages": {}, "artifacts": {}})
            job.update(fields)
            self._save()

    def stage(self, job_id: str, stage: str) -> dict:
        return self.data["jobs"].get(job_id, {}).get("stages", {}).get(stage, {})

    def set_stage(self, job_id: str, stage: str, **fields) -> None:
        with self._lock:
            job = self.data["jobs"].setdefault(job_id, {"stages": {}, "artifacts": {}})
            entry = job["stages"].setdefault(stage, {})
            entry.update(fields)
            self._save()

    def set_artifact(self, job_id: str, key: str, value) -> None:
        with self._lock:
            job = self.data["jobs"].setdefault(job_id, {"stages": {}, "artifacts": {}})
            job["artifacts"][key] = value
            self._save()

    def stage_ok(self, job_id: str, stage: str) -> bool:
        entry = self.stage(job_id, stage)
        if entry.get("status") != "ok":
            return False
        artifact = entry.get("artifact")
        # A recorded success whose artifact vanished is not a success to resume over.
        return artifact is None or Path(artifact).exists()

    def _save(self) -> None:
        tmp = self.path.with_name(self.path.name + ".tmp")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp.write_text(json.dumps(self.data, indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
