"""UV valid-mask, gutter fill, chroma-floor loss, and the polish scrub."""

from __future__ import annotations

import numpy as np
import torch

from escher.main_sphere import texture_fill_loss
from escher.rendering.texture_mask import gutter_fill, uv_valid_mask


def _unit_square_uv():
    """Two triangles covering the LEFT half of the UV square."""
    uv = np.array([[0.0, 0.0], [0.5, 0.0], [0.5, 1.0], [0.0, 1.0]])
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return uv, faces


def test_uv_valid_mask_covers_the_mapped_region_only():
    uv, faces = _unit_square_uv()
    mask = uv_valid_mask(uv, faces, 64)
    frac = mask.mean()
    # Half the square, plus the one-texel dilation ring.
    assert 0.5 <= frac < 0.56, frac
    assert mask[:, : 64 // 2 - 1].all(), "the mapped half must be fully valid"
    assert not mask[:, -8:].any(), "the far side must stay invalid"


def test_kite_uv_leaves_a_large_gutter():
    from omegaconf import OmegaConf

    from escher.main_shape import build_shape_run
    from escher.main_sphere import PATH
    import tempfile

    args = OmegaConf.merge(
        OmegaConf.load(PATH / "configs/sphere.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape_weights.yaml"),
        OmegaConf.create({"ORBIFOLD_CONES": [2, 3, 4], "DEVICE": "cpu"}),
    )
    with tempfile.TemporaryDirectory() as td:
        args.OUTPUT_DIR = td
        escher = build_shape_run(args)
        mask = uv_valid_mask(escher.mesh.uv, escher.mesh.faces, 128)
    # The kite occupies about half its bounding box; ~60% of texels are gutter.
    assert 0.3 < mask.mean() < 0.55, mask.mean()


def test_gutter_fill_touches_only_invalid_texels():
    rng = np.random.default_rng(0)
    tex = rng.random((32, 32, 3)).astype(np.float32)
    valid = np.zeros((32, 32), dtype=bool)
    valid[:, :16] = True
    filled = gutter_fill(tex, valid)
    assert np.array_equal(filled[valid], tex[valid]), "valid texels must be untouched"
    # Every invalid texel takes its row's nearest valid color (column 15).
    assert np.array_equal(filled[:, 16:], np.repeat(tex[:, 15:16], 16, axis=1))


def test_fill_loss_pushes_near_gray_leaves_color():
    valid = torch.ones(8, 8, dtype=torch.bool)
    # NEAR-gray, the realistic case: exact channel ties are a documented
    # zero-gradient saddle that SDS noise leaves immediately.
    near_gray = torch.full((8, 8, 3), 0.9)
    near_gray[..., 2] = 0.88
    colored = torch.zeros(8, 8, 3)
    colored[..., 0] = 0.8  # strong chroma
    assert texture_fill_loss(near_gray, valid, 0.15) > 0
    assert float(texture_fill_loss(colored, valid, 0.15)) == 0.0
    g = near_gray.clone().requires_grad_(True)
    texture_fill_loss(g, valid, 0.15).backward()
    assert g.grad is not None and g.grad.abs().sum() > 0
    # And the loss only counts VALID texels.
    partial = torch.zeros(8, 8, dtype=torch.bool)
    partial[0, 0] = True
    colored2 = colored.clone()
    colored2[1:, :] = 0.9  # background-like everywhere EXCEPT the one valid texel
    assert float(texture_fill_loss(colored2, partial, 0.15)) == 0.0


def test_polish_scrub_removes_all_offending_texels(tmp_path):
    from escher.texture_polish import offending_texels

    rng = np.random.default_rng(1)
    tex = rng.uniform(0.2, 0.8, size=(32, 32, 3)).astype(np.float32)
    tex[..., 0] += 0.2  # give everything real chroma
    tex = np.clip(tex, 0, 1)
    valid = np.ones((32, 32), dtype=bool)
    tex[4:8, 4:8] = 0.95  # a white-like patch
    init = np.array([0.76, 0.60, 0.42], dtype=np.float32)
    tex[20:24, 20:24] = init  # an init-leak patch

    bad = offending_texels(tex, valid, init)
    assert bad.sum() >= 32

    from scipy import ndimage

    good = valid & ~bad
    idx = ndimage.distance_transform_edt(
        ~good, return_distances=False, return_indices=True
    )
    polished = tex.copy()
    polished[bad] = tex[idx[0][bad], idx[1][bad]]
    assert not offending_texels(polished, valid, init).any()
    assert np.array_equal(polished[good], tex[good])
