"""CPU validation of the deterministic planar shape phase (main_shape_planar.py).

The real optimization path end to end -- solve, project, soft mask, backward -- with no
mocks: the planar Tutte solve is a dense linear solve and the silhouette is analytic,
so the whole driver runs offline (the test_main_shape.py discipline, planar edition).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

pytest.importorskip("igl")

from escher.main_shape_planar import (  # noqa: E402
    build_planar_shape_run,
    planar_to_pixels,
    run_shape,
    solve_tile,
)


def tiny_args(tmp_path: Path) -> OmegaConf:
    from escher.main import load_planar_args

    args = load_planar_args(OmegaConf.create({"CONF_FILE": "configs/planar_shape.yaml"}))
    args.MESH_RESOLUTION = 10
    args.RENDER_SIZE = 64
    args.MASK_PYRAMID_LEVELS = 3
    args.SHAPE_STEPS = 3
    args.VISUALIZATION_FREQ = 2
    args.OUTPUT_DIR = str(tmp_path / "run")
    args.TARGET_MASK = str(tmp_path / "target.npy")

    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    disk = ((yy - 32) ** 2 + (xx - 32) ** 2 <= 20**2).astype(np.float32)
    np.save(tmp_path / "target.npy", disk)
    return args


def test_planar_to_pixels_conventions():
    pts = torch.tensor([[0.0, 0.0], [2.4, 0.0], [0.0, 2.4]])
    px = planar_to_pixels(pts, 2.4, 100)
    assert torch.allclose(px[0], torch.tensor([50.0, 50.0]))
    assert px[1, 0] == pytest.approx(100.0)  # +x -> right
    assert px[2, 1] == pytest.approx(0.0)  # +y -> UP the image (row 0)


def test_run_shape_moves_w_and_checkpoints(tmp_path):
    args = tiny_args(tmp_path)
    result = run_shape(args)
    out = Path(args.OUTPUT_DIR)

    header = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[3] == "iou"
    assert 0.0 < result["iou"] <= 1.0
    assert result["flips"] == 0

    final = torch.load(out / "checkpoint_000003.pt", weights_only=False)
    assert not torch.allclose(final["W"], torch.zeros_like(final["W"])), (
        "the optimizer must actually move the weights"
    )
    assert (out / "target_overlay.png").exists()
    assert (out / "step_00000.png").exists()

    # the checkpoint feeds main.py's W_INIT_PATH; prove the round trip
    args2 = tiny_args(tmp_path)
    args2.W_INIT_PATH = str(out / "checkpoint_000003.pt")
    e = build_planar_shape_run(args2)
    assert torch.allclose(e.W.detach(), final["W"])


def test_initial_state_is_the_undeformed_tile(tmp_path):
    args = tiny_args(tmp_path)
    e = build_planar_shape_run(args)
    assert torch.allclose(e.W.detach(), torch.zeros_like(e.W))
    mapped = solve_tile(e)
    # OrbifoldI pins its corners at (+-1, +-1); the uniform-weight solve must stay a
    # sane square-ish tile inside that box
    assert mapped.detach().abs().max() <= 1.0 + 1e-6
