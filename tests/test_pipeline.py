"""Batch driver: spec expansion, manifest semantics, and a dry-run integration
test with failure propagation and resume.

The integration test spawns real child processes through the real scheduler --
only the GPU stages are stubbed (that is what --dry-run is) -- so the exit-code
contract, the two-lane readiness logic, and the artifact handoffs are all the
production code paths.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from escher.pipeline.manifest import Manifest
from escher.pipeline.spec import load_spec


def write_spec(path: Path, body: str) -> Path:
    path.write_text(body, encoding="utf-8")
    return path


def test_spec_sweep_expansion_and_dirs(tmp_path):
    spec = write_spec(
        tmp_path / "b.yaml",
        f"""
batch_name: t
runs_root: {tmp_path.as_posix()}/runs
defaults:
  seeds: [0, 1]
  stages:
    carve: {{SHAPE_STEPS: 5}}
jobs:
  - figure: fish
    prompt: a fish
    sweep:
      orbifold: [[2, 3, 4], [2, 3, 5]]
""",
    )
    name, root, plans = load_spec(spec)
    assert name == "t"
    assert [p.job_id for p in plans] == ["fish__o234", "fish__o235"]
    assert plans[0].orbifold == [2, 3, 4] and plans[1].orbifold == [2, 3, 5]
    assert plans[0].seeds == [0, 1]
    assert plans[0].root == root / "fish__o234"
    assert plans[0].carve_dir(1).name == "carve_s1"
    assert plans[0].stages["carve"]["SHAPE_STEPS"] == 5
    # defaults must not be shared mutable state across expanded jobs
    plans[0].stages["carve"]["SHAPE_STEPS"] = 99
    assert plans[1].stages["carve"]["SHAPE_STEPS"] == 5


def test_spec_rejects_duplicate_job_ids(tmp_path):
    spec = write_spec(
        tmp_path / "b.yaml",
        """
batch_name: t
jobs:
  - figure: fish
  - figure: fish
""",
    )
    with pytest.raises(ValueError, match="duplicate"):
        load_spec(spec)


def test_manifest_roundtrip_and_artifact_check(tmp_path):
    m = Manifest(tmp_path / "m.json", "b")
    art = tmp_path / "ck.pt"
    art.write_text("x")
    m.set_stage("j", "carve_s0", status="ok", artifact=str(art))
    m2 = Manifest(tmp_path / "m.json")
    assert m2.stage_ok("j", "carve_s0")
    art.unlink()
    assert not m2.stage_ok("j", "carve_s0"), "vanished artifact must not resume as ok"
    assert not m2.stage_ok("j", "never_ran")


TINY_SHAPE = """
      MESH_N_THETA: 6
      MESH_N_PHI: 5
      RENDER_SIZE: 64
      MASK_PYRAMID_LEVELS: 3
      TARGET_SMOOTH_RADIUS: 0
      DEVICE: "cpu"
"""


def test_dry_run_failure_propagation_and_resume(tmp_path):
    from escher.pipeline.scheduler import Scheduler

    # A poisoned targets dir: one candidate far below the plausibility band, so
    # the screen stage fails (exit 1) and downstream stages must be skipped.
    bad_targets = tmp_path / "bad_targets"
    bad_targets.mkdir()
    import imageio.v2 as imageio

    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    tiny = (((yy - 32) ** 2 + (xx - 32) ** 2) <= 5**2).astype(np.uint8) * 255
    imageio.imwrite(bad_targets / "target_00.png", tiny)

    spec = write_spec(
        tmp_path / "b.yaml",
        f"""
batch_name: t
runs_root: {tmp_path.as_posix()}/runs
defaults:
  orbifold: null
  seeds: [0]
  texture_seeds: [0]
  stages:
    screen:
{TINY_SHAPE}
      TARGET_SWEEP_STEPS: 2
      TARGET_SWEEP_WORKERS: 1
    carve:
{TINY_SHAPE}
      SHAPE_STEPS: 3
      VISUALIZATION_FREQ: 2
    texture:
      MESH_N_THETA: 6
      MESH_N_PHI: 5
      DEVICE: "cpu"
jobs:
  - figure: good
    prompt: a disk
  - figure: poisoned
    prompt: a dot
    reuse_targets: {bad_targets.as_posix()}
""",
    )
    name, root, plans = load_spec(spec)
    manifest = Manifest(root / "manifest.json", name)
    ok = Scheduler(
        plans, manifest, root, dry_run=True, repo_root=Path.cwd()
    ).run()
    assert not ok, "the poisoned job must fail the batch"

    jobs = manifest.data["jobs"]
    good = jobs["good"]["stages"]
    assert {s["status"] for s in good.values()} == {"ok"}
    poisoned = jobs["poisoned"]["stages"]
    assert poisoned["screen"]["status"] == "failed"
    assert poisoned["carve_s0"]["status"] == "skipped"
    assert poisoned["texture_s0"]["status"] == "skipped"
    assert poisoned["render_s0"]["status"] == "skipped"
    assert (root / "summary.csv").exists()
    assert (root / "contact_sheet.html").exists()

    # Resume: completed stages must not re-run (started timestamp unchanged);
    # the failed screen re-runs and fails again.
    started_before = good["target"]["started"]
    manifest2 = Manifest(root / "manifest.json")
    ok2 = Scheduler(
        plans, manifest2, root, dry_run=True, repo_root=Path.cwd()
    ).run()
    assert not ok2
    good_after = manifest2.data["jobs"]["good"]["stages"]
    assert good_after["target"]["started"] == started_before

    # The good job's winner mask is job-local; the shared bad_targets dir was
    # never mutated by the sweep (WINNER_OUT redirection).
    assert (root / "good" / "target.npy").exists()
    assert sorted(p.name for p in bad_targets.iterdir()) == ["target_00.png"]
