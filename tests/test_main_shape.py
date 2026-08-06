"""CPU validation of the deterministic shape-phase driver (escher/main_shape.py).

nvdiffrast is CUDA-only, so the real renderer never runs here; a differentiable
synthetic splat stands in for it, which is enough to prove the driver's contracts:
no diffusion import, a bit-deterministic camera, gradients reaching P, metrics with
the iou column, and checkpoints that round-trip into the ordinary resume path.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from escher.main_sphere import PATH, SphereEscher
from escher.main_shape import (
    build_shape_run,
    fixed_tile_camera,
    run_shape,
    scaled_n_phi,
)


def tiny_args(tmp_path: Path) -> OmegaConf:
    args = OmegaConf.merge(
        OmegaConf.load(PATH / "configs/sphere.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape.yaml"),
    )
    args.DEVICE = "cpu"
    args.MESH_N_THETA = 6
    args.MESH_N_PHI = 5
    args.OUTPUT_DIR = str(tmp_path / "run")
    args.SHAPE_STEPS = 3
    args.VISUALIZATION_FREQ = 2
    args.MASK_PYRAMID_LEVELS = 3
    args.TARGET_MASK = str(tmp_path / "target.npy")

    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    disk = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 8**2).astype(np.float32)
    np.save(tmp_path / "target.npy", disk)
    return args


def splat_render(self, n_views, mv=None, isolated=False, texture=None):
    """Differentiable stand-in for the CUDA renderer: a gaussian splat of the solved
    vertices. Gradients flow alpha -> points -> boundary solve -> P, exactly the chain
    the real renderer provides."""
    points = self.solve_points()
    ax = torch.linspace(-1.2, 1.2, 32, dtype=points.dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    d2 = (yy[..., None] - points[:, 2]) ** 2 + (xx[..., None] - points[:, 1]) ** 2
    alpha = torch.tanh(torch.exp(-d2 / 0.02).sum(-1)).reshape(1, 32, 32, 1)
    images = alpha.expand(-1, -1, -1, 3)
    return images, alpha, points


def test_build_shape_run_never_imports_the_diffusion_stack(tmp_path):
    # Other tests in the same session may have imported sd.py legitimately; the assert
    # only has teeth when this test runs with a clean module table.
    already = "escher.guidance.sd" in sys.modules
    escher = build_shape_run(tiny_args(tmp_path))
    assert escher.P.requires_grad
    if not already:
        assert "escher.guidance.sd" not in sys.modules


def test_fixed_tile_camera_is_deterministic_and_rng_independent(tmp_path):
    args = tiny_args(tmp_path)
    escher = build_shape_run(args)
    mv1 = fixed_tile_camera(escher, args)
    torch.manual_seed(12345)
    torch.rand(100)  # scramble global RNG between calls
    mv2 = fixed_tile_camera(escher, args)
    assert mv1.shape == (1, 4, 4)
    assert torch.equal(mv1, mv2)


def test_run_shape_moves_p_logs_iou_and_checkpoints(tmp_path, monkeypatch):
    args = tiny_args(tmp_path)
    monkeypatch.setattr(SphereEscher, "render", splat_render)

    result = run_shape(args)
    out = Path(args.OUTPUT_DIR)

    header = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[3] == "iou"
    assert result["k"] == args.ORBIFOLD_K
    assert 0.0 <= result["iou"] <= 1.0

    # the optimizer must actually have moved the boundary
    escher = build_shape_run(args)
    init = escher.b_orb.initial_free_points()
    final = torch.load(out / "checkpoint_000003.pt", weights_only=False)["P"]
    assert not torch.allclose(final, init)

    # alignment evidence written for eyeballing
    assert (out / "target_aligned.png").exists()
    assert (out / "target_overlay.png").exists()
    assert (out / "step_00000.png").exists()

    # the checkpoint round-trips through the ordinary resume path
    assert escher.load_checkpoint(out / "checkpoint_000003.pt") == 3
    assert torch.allclose(escher.P.detach(), final)


def test_scaled_n_phi():
    assert scaled_n_phi(19, 4) == 19
    assert scaled_n_phi(19, 6) == 13
    for k in range(2, 13):
        n = scaled_n_phi(19, k)
        assert n % 2 == 1 and n >= 5


def test_texture_init_color(tmp_path):
    args = tiny_args(tmp_path)
    assert args.TEXTURE_INIT_COLOR is None
    escher = build_shape_run(args)
    assert torch.allclose(escher.texture.detach(), torch.full_like(escher.texture, 0.5))

    args.TEXTURE_INIT_COLOR = [0.76, 0.60, 0.42]
    escher = build_shape_run(args)
    tan = torch.tensor([0.76, 0.60, 0.42])
    assert torch.allclose(escher.texture.detach()[0, 0], tan)
    assert torch.allclose(escher.texture.detach()[-1, -1], tan)
