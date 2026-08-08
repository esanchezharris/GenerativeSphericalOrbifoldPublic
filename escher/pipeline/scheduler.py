"""Two-lane scheduler: CPU shape work overlaps the exclusive GPU texture runs.

The GPU lane is strictly serial (a texture run needs ~9 GiB of 12 alone); the CPU
lane screens and carves job N+1 while job N's texture holds the GPU. Each stage
runs in a child process via ``python -m escher.pipeline --run-stage``; failures
mark the job's downstream stages skipped and the lanes move on -- one bad prompt
cannot kill the night. CUDA OOM (exit 4) gets exactly one retry in a fresh
process (provably clean VRAM); a second 4 means the config genuinely doesn't fit.
"""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from escher.pipeline import report, stages, warden
from escher.pipeline.manifest import Manifest
from escher.pipeline.spec import JobPlan

TERMINAL = ("ok", "failed", "skipped")


@dataclass
class StageTask:
    plan: JobPlan
    name: str
    lane: str  # "cpu" | "gpu"
    deps: list[str]
    require: str  # "all_ok" | "any_ok" over deps
    build: Callable[[], dict]
    artifact: Path
    status: str = "pending"
    tries: int = 0


def job_tasks(plan: JobPlan, dry_run: bool) -> list[StageTask]:
    tasks: list[StageTask] = []
    if not plan.reuse_targets:
        tasks.append(
            StageTask(
                plan,
                "target",
                "gpu",
                deps=[],
                require="all_ok",
                build=lambda p=plan: stages.target_params(p, dry_run),
                artifact=plan.targets_dir / "target.npy",
            )
        )
    target_deps = ["target"] if not plan.reuse_targets else []
    tasks.append(
        StageTask(
            plan,
            "screen",
            "cpu",
            deps=target_deps,
            require="all_ok",
            build=lambda p=plan: stages.screen_params(p, dry_run),
            artifact=plan.winner_path,
        )
    )
    carve_names = []
    for seed in plan.seeds:
        name = f"carve_s{seed}"
        carve_names.append(name)
        tasks.append(
            StageTask(
                plan,
                name,
                "cpu",
                deps=["screen"],
                require="all_ok",
                build=lambda p=plan, s=seed: stages.carve_params(p, s, dry_run),
                artifact=plan.carve_dir(seed) / "checkpoint.pt",
            )
        )
    for tseed in plan.texture_seeds:
        tex = f"texture_s{tseed}"
        tasks.append(
            StageTask(
                plan,
                tex,
                "gpu",
                deps=carve_names,
                require="any_ok",
                build=None,  # bound by the scheduler: needs the best-carve pick
                artifact=plan.tex_dir(tseed) / "checkpoint.pt",
            )
        )
        tasks.append(
            StageTask(
                plan,
                f"render_s{tseed}",
                "gpu",
                deps=[tex],
                require="all_ok",
                build=lambda p=plan, t=tseed: stages.render_params(p, t, dry_run),
                artifact=plan.tex_dir(tseed) / "tiling.obj",
            )
        )
    return tasks


class Scheduler:
    def __init__(
        self,
        plans: list[JobPlan],
        manifest: Manifest,
        batch_root: Path,
        *,
        dry_run: bool = False,
        use_warden: bool = False,
        repo_root: Path | None = None,
    ) -> None:
        self.manifest = manifest
        self.batch_root = batch_root
        self.dry_run = dry_run
        self.use_warden = use_warden
        self.repo_root = Path(repo_root) if repo_root else Path.cwd()
        self.plans = plans
        self.lock = threading.Lock()
        self.tasks: dict[tuple[str, str], StageTask] = {}
        for plan in plans:
            manifest.ensure_job(
                plan.job_id,
                figure=plan.figure,
                variant=plan.variant,
                prompt=plan.prompt,
            )
            for task in job_tasks(plan, dry_run):
                if manifest.stage_ok(plan.job_id, task.name):
                    task.status = "ok"  # --resume: completed with artifact intact
                self.tasks[(plan.job_id, task.name)] = task

    # ------------------------------------------------------------------ state
    def _job_task(self, plan: JobPlan, name: str) -> StageTask:
        return self.tasks[(plan.job_id, name)]

    def _deps_state(self, task: StageTask) -> tuple[bool, bool]:
        """(all deps terminal, requirement satisfied)."""
        deps = [self._job_task(task.plan, d) for d in task.deps]
        terminal = all(d.status in TERMINAL for d in deps)
        if task.require == "any_ok":
            satisfied = any(d.status == "ok" for d in deps) if deps else True
        else:
            satisfied = all(d.status == "ok" for d in deps)
        return terminal, satisfied

    def _next_ready(self, lane: str) -> StageTask | None:
        with self.lock:
            for plan in self.plans:  # job order: earlier jobs first
                for task in (
                    t for (jid, _), t in self.tasks.items() if jid == plan.job_id
                ):
                    if task.lane != lane or task.status != "pending":
                        continue
                    terminal, satisfied = self._deps_state(task)
                    if not terminal:
                        continue
                    if not satisfied:
                        task.status = "skipped"
                        self.manifest.set_stage(
                            plan.job_id, task.name, status="skipped",
                            error="dependency failed",
                        )
                        continue
                    task.status = "running"
                    return task
        return None

    def _all_terminal(self) -> bool:
        with self.lock:
            return all(t.status in TERMINAL for t in self.tasks.values())

    def _best_carve(self, plan: JobPlan) -> str | None:
        best, best_key = None, None
        for seed in plan.seeds:
            if self._job_task(plan, f"carve_s{seed}").status != "ok":
                continue
            m = self.manifest.stage(plan.job_id, f"carve_s{seed}").get("metrics") or {}
            key = m.get("hard_iou", m.get("iou", 0.0))
            if best_key is None or key > best_key:
                best_key, best = key, str(plan.carve_dir(seed) / "checkpoint.pt")
        return best

    # -------------------------------------------------------------- execution
    def _execute(self, task: StageTask) -> None:
        plan = task.plan
        if task.build is None:  # texture: bind the best-carve pick now
            tseed = int(task.name.split("_s")[1])
            resume = self._best_carve(plan)
            params = stages.texture_params(plan, tseed, resume, self.dry_run)
        else:
            params = task.build()

        plan.root.mkdir(parents=True, exist_ok=True)
        params_path = plan.root / f"stage_{task.name}.json"
        params_path.write_text(json.dumps(params, indent=2), encoding="utf-8")
        log_path = plan.root / f"stage_{task.name}.log"

        cmd = [sys.executable, "-m", "escher.pipeline", "--run-stage", str(params_path)]
        if task.lane == "gpu" and not self.dry_run:
            cmd = warden.wrap(cmd, self.use_warden, label=f"{plan.job_id}:{task.name}")

        self.manifest.set_stage(
            plan.job_id, task.name, status="running",
            started=time.strftime("%Y-%m-%d %H:%M:%S"),
        )
        print(f"[{task.lane}] {plan.job_id}:{task.name} starting", flush=True)

        import os

        env = dict(os.environ)
        env["PYTHONPATH"] = str(self.repo_root)
        with open(log_path, "a", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, cwd=self.repo_root, stdout=log, stderr=subprocess.STDOUT, env=env
            )
        code = proc.returncode

        if code == 4 and task.tries == 0:
            task.tries += 1
            print(
                f"[{task.lane}] {plan.job_id}:{task.name} OOM -- one retry after "
                "cooldown",
                flush=True,
            )
            time.sleep(60)
            with open(log_path, "a", encoding="utf-8") as log:
                proc = subprocess.run(
                    cmd, cwd=self.repo_root, stdout=log, stderr=subprocess.STDOUT,
                    env=env,
                )
            code = proc.returncode

        result_path = params_path.with_suffix(".result.json")
        payload = {}
        if result_path.exists():
            payload = json.loads(result_path.read_text(encoding="utf-8"))

        with self.lock:
            task.status = "ok" if code == 0 else "failed"
        self.manifest.set_stage(
            plan.job_id,
            task.name,
            status=task.status,
            exit_code=code,
            ended=time.strftime("%Y-%m-%d %H:%M:%S"),
            artifact=str(task.artifact),
            metrics=payload.get("result") if code == 0 else None,
            error=None if code == 0 else payload.get("error", f"exit {code}"),
            log=str(log_path),
        )
        print(
            f"[{task.lane}] {plan.job_id}:{task.name} "
            f"{'ok' if code == 0 else f'FAILED (exit {code})'}",
            flush=True,
        )
        report.write_reports(self.manifest, self.batch_root, self.plans)

    def _lane_loop(self, lane: str) -> None:
        while True:
            task = self._next_ready(lane)
            if task is None:
                if self._all_terminal():
                    return
                time.sleep(0.5)
                continue
            try:
                self._execute(task)
            except Exception as e:  # noqa: BLE001 -- a lane must never die silently
                with self.lock:
                    task.status = "failed"
                self.manifest.set_stage(
                    task.plan.job_id, task.name, status="failed",
                    error=f"driver error: {e!r}",
                )

    def run(self) -> bool:
        lanes = [
            threading.Thread(target=self._lane_loop, args=("cpu",), daemon=True),
            threading.Thread(target=self._lane_loop, args=("gpu",), daemon=True),
        ]
        for t in lanes:
            t.start()
        for t in lanes:
            t.join()
        report.write_reports(self.manifest, self.batch_root, self.plans)
        return all(t.status == "ok" for t in self.tasks.values())
