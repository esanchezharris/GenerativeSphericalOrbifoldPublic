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


def test_run_shape_weights_mode(tmp_path):
    """The GEM full-mesh path: the mask loss reaches ALL edge weights through the
    implicit solve. Same driver, same metrics; no fold rejection (weights-mode solves
    measured fold-free across two orders of weight magnitude -- flips is a tripwire)."""
    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    args.LR_W = 0.05
    args.OUTPUT_DIR = str(tmp_path / "wrun")

    result = run_shape(args)
    out = Path(args.OUTPUT_DIR)

    assert 0.0 < result["iou"] <= 1.0
    assert result["flips"] == 0

    final = torch.load(out / "checkpoint_000003.pt", weights_only=False)
    assert "W" in final and not torch.allclose(final["W"], torch.zeros_like(final["W"])), (
        "the mask gradient must reach the edge weights through the implicit solve"
    )

    rows = (out / "metrics.csv").read_text(encoding="utf-8").splitlines()
    assert all(r.split(",")[9] == "0" for r in rows[1:]), "reverts must stay 0"


def test_weights_mode_smoothness_penalises_a_wobbly_outline(tmp_path):
    """The term must actually charge for boundary wobble, and be off by default.

    Weights mode shipped with NO outline-smoothness prior of any kind -- only the weight
    and area penalties -- so nothing opposed a high-frequency wobble and the carve came
    back with sawtooth serrations. Boundary mode has had such a term all along.
    """
    from escher.main_shape import make_context, shape_step

    args = tiny_args(tmp_path)
    args.PARAM_MODE = "weights"
    escher = build_shape_run(args)
    ctx = make_context(escher, args)
    target = torch.as_tensor(np.load(args.TARGET_MASK), device=escher.device)
    target = torch.nn.functional.interpolate(
        target[None, None], size=(ctx.size, ctx.size), mode="nearest"
    )[0, 0]

    args.BOUNDARY_SMOOTH_WEIGHT_W = 0.0
    off = shape_step(escher, target, ctx, args)["area_reg"]

    escher2 = build_shape_run(args)
    args.BOUNDARY_SMOOTH_WEIGHT_W = 1000.0
    on = shape_step(escher2, target, ctx, args)["area_reg"]

    assert on > off, "the smoothness term must contribute to the regulariser"

    # And it is a genuine second-difference energy: a wobbly loop costs more than a
    # smooth one of the same length scale.
    t = torch.linspace(0, 2 * np.pi, 64, dtype=torch.float64)[:-1]
    smooth = torch.stack([t.cos(), t.sin(), torch.zeros_like(t)], -1)
    wobbly = smooth * (1.0 + 0.05 * torch.sin(12 * t))[:, None]

    def energy(b):
        return float((b.roll(-1, 0) - 2 * b + b.roll(1, 0)).square().sum())

    assert energy(wobbly) > 3 * energy(smooth)


def test_sweep_targets_ranking_is_worker_count_independent(tmp_path):
    """Parallelising the candidate screen must not change which candidate wins."""
    import imageio.v2 as imageio

    from escher.main_shape import sweep_targets

    tdir = tmp_path / "cands"
    tdir.mkdir()
    yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
    for i, r in enumerate((10, 14, 18)):
        m = ((yy - 32) ** 2 + (xx - 32) ** 2 <= r * r).astype(np.uint8) * 255
        imageio.imwrite(tdir / f"target_{i:02d}.png", m)

    def run(workers):
        args = tiny_args(tmp_path)
        args.PARAM_MODE = "weights"
        args.SWEEP_TARGETS = True
        args.TARGET_DIR = str(tdir)
        args.TARGET_SWEEP_STEPS = 2
        args.TARGET_SWEEP_WORKERS = workers
        args.OUTPUT_DIR = str(tmp_path / f"sw{workers}")
        sweep_targets(args)
        rows = Path(f"{args.OUTPUT_DIR}_targets.csv").read_text().strip().splitlines()[1:]
        return [(r.split(",")[0], round(float(r.split(",")[1]), 4)) for r in rows]

    assert run(1) == run(3)


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
