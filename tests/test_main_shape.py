"""CPU validation of the deterministic shape-phase driver (escher/main_shape.py).

The shape loss is renderer-free (escher/soft_silhouette.py) precisely because the
rasterized alpha measured EXACTLY zero vertex gradients -- so these tests run the REAL
optimization path end to end on CPU, no monkeypatched stand-ins: no diffusion import, a
bit-deterministic camera, gradients actually moving P, metrics with the iou column, and
checkpoints that round-trip into the ordinary resume path. (Only the snapshot's pretty
wide view needs CUDA, and it skips itself on CPU.)
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
    args.RENDER_SIZE = 64
    args.TARGET_MASK = str(tmp_path / "target.npy")

    yy, xx = np.meshgrid(np.arange(32), np.arange(32), indexing="ij")
    disk = ((yy - 16) ** 2 + (xx - 16) ** 2 <= 8**2).astype(np.float32)
    np.save(tmp_path / "target.npy", disk)
    return args


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


def test_run_shape_moves_p_logs_iou_and_checkpoints(tmp_path):
    args = tiny_args(tmp_path)

    result = run_shape(args)  # the REAL path: solve -> project -> soft mask -> backward
    out = Path(args.OUTPUT_DIR)

    header = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()[0]
    assert header.split(",")[3] == "iou"
    assert result["k"] == args.ORBIFOLD_K
    assert 0.0 < result["iou"] <= 1.0

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


def test_cross_phase_resume_keeps_fresh_texture(tmp_path):
    """Texture phase resuming a shape checkpoint: shape FROM the checkpoint, texture
    from this run's init -- the shape phase never trained its texture."""
    args = tiny_args(tmp_path)
    donor = build_shape_run(args)
    with torch.no_grad():
        donor.P.add_(0.01)
    path = donor.save_checkpoint(7)

    args2 = tiny_args(tmp_path)
    args2.OUTPUT_DIR = str(tmp_path / "tex")
    args2.TEXTURE_INIT_COLOR = [0.76, 0.60, 0.42]
    e = build_shape_run(args2)
    assert e.load_checkpoint(path, reset_texture=True) == 7
    assert torch.allclose(e.P.detach(), donor.P.detach()), "shape comes from checkpoint"
    tan = torch.tensor([0.76, 0.60, 0.42])
    assert torch.allclose(e.texture.detach()[0, 0], tan), "texture keeps the fresh init"

    # The DEFAULT (render_final's path) must load the trained texture even when the
    # checkpoint's own saved config carries RESET_TEXTURE_ON_RESUME -- reading the flag
    # ambiently once rendered a finished run as flat init blobs.
    args3 = tiny_args(tmp_path)
    args3.OUTPUT_DIR = str(tmp_path / "render")
    args3.RESET_TEXTURE_ON_RESUME = True
    r = build_shape_run(args3)
    r.load_checkpoint(path)
    assert torch.allclose(r.texture.detach(), donor.texture.detach().cpu()), (
        "plain load must take the checkpoint texture"
    )


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
