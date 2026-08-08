"""The per-pixel spherical-area map must integrate to known closed forms."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.pixel_solid_angle import pixel_solid_angles
from escher.rendering.camera import perspective
from escher.shape_target import align_mask_to


def look_at_mv(distance: float) -> torch.Tensor:
    """Camera on +z at ``distance``, looking at the origin (renderer convention)."""
    mv = torch.eye(4, dtype=torch.float64)
    mv[2, 3] = -distance
    return mv


def test_spherical_zone_area_bracketed():
    """Interior accuracy, tested without the horizon ring.

    Near the horizon the per-pixel spherical area diverges (grazing rays), and
    straddling pixels are deliberately dropped, so a full-cap sum converges only
    O(1/N). Masks in the shape phase live well inside the disc; there the map must
    bracket the analytic zone area 2*pi*(1 - z0) tightly: summing pixels wholly
    inside the zone underestimates it, adding the boundary-straddling pixels
    overestimates it, and both bounds must be close.
    """
    d, z0, res = 2.4, 0.75, 512
    mv = look_at_mv(d)
    proj = perspective(fovy_deg=60.0)
    omega = pixel_solid_angles(mv, proj, res, res)
    assert (omega >= 0).all()

    # Reconstruct each pixel-corner hit's world-z via the same inverse mapping.
    from escher.pixel_solid_angle import _as_matrix, _sphere_hits

    pm_inv = np.linalg.inv(_as_matrix(proj) @ _as_matrix(mv))
    cols = np.arange(res + 1, dtype=np.float64)
    ndc_x = 2.0 * cols / res - 1.0
    ndc_y = -(2.0 * np.arange(res + 1, dtype=np.float64) / res - 1.0)
    gx, gy = np.meshgrid(ndc_x, ndc_y, indexing="xy")
    pts, hit = _sphere_hits(pm_inv, np.stack([gx.ravel(), gy.ravel()], axis=1))
    z = np.where(hit, pts[:, 2], -2.0).reshape(res + 1, res + 1)

    corner_in = z >= z0
    all_in = corner_in[:-1, :-1] & corner_in[:-1, 1:] & corner_in[1:, :-1] & corner_in[1:, 1:]
    any_in = corner_in[:-1, :-1] | corner_in[:-1, 1:] | corner_in[1:, :-1] | corner_in[1:, 1:]

    zone = 2.0 * np.pi * (1.0 - z0)
    lower = omega[all_in].sum()
    upper = omega[any_in].sum()
    assert lower <= zone <= upper
    assert (upper - lower) / zone < 0.03


def test_production_tile_integrates_to_4pi_over_group_order():
    """The actual use case: the undeformed (2,3,4) tile's silhouette, measured in
    steradians on the production shape framing, must integrate to 4pi/24."""
    from omegaconf import OmegaConf

    from escher.main_shape import build_shape_run, make_context, soft_alpha
    from escher.main_sphere import PATH

    args = OmegaConf.merge(
        OmegaConf.load(PATH / "configs/sphere.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape_weights.yaml"),
        OmegaConf.create(
            {
                "ORBIFOLD_CONES": [2, 3, 4],
                "SHAPE_CAMERA_DISTANCE": 2.4,
                "DEVICE": "cpu",
                "OUTPUT_DIR": "unused",
            }
        ),
    )
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        args.OUTPUT_DIR = td
        escher = build_shape_run(args)
        ctx = make_context(escher, args)
        with torch.no_grad():
            alpha = soft_alpha(escher, escher.solve_points(), ctx)[0, ..., 0].cpu().numpy()
        omega = pixel_solid_angles(ctx.mv, ctx.proj, ctx.size, ctx.size)
        tile_sr = omega[alpha > 0.5].sum()
        expected = 4.0 * np.pi / 24.0
        assert abs(tile_sr - expected) / expected < 0.015, (
            f"tile measures {tile_sr:.5f} sr, expected {expected:.5f}"
        )


def test_rim_pixels_cover_more_sphere_than_center():
    d = 2.4
    omega = pixel_solid_angles(look_at_mv(d), perspective(fovy_deg=60.0), 256, 256)
    on = omega > 0
    assert omega[on].max() / omega[128, 128] > 1.5, (
        "per-pixel spherical area must grow toward the horizon"
    )


def test_pixels_off_the_sphere_are_zero():
    omega = pixel_solid_angles(look_at_mv(2.4), perspective(fovy_deg=60.0), 128, 128)
    assert omega[0, 0] == 0.0 and omega[-1, -1] == 0.0


def _disk(h, w, cy, cx, r):
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    return (((yy - cy) ** 2 + (xx - cx) ** 2) <= r * r).astype(np.float32)


def test_achieved_area_bisection_compensates_for_cropping():
    """The scale must be solved on the PLACED mask: a target whose scaled form runs
    off the frame edge loses area to cropping, and the old closed-form scale
    (computed pre-transform) delivered it short -- 1.9% on the shipped fish."""
    ref = _disk(128, 128, 64, 64, 40)
    tgt = _disk(128, 128, 64, 112, 14)  # near the right edge: scaling it crops

    aligned, params, _ = align_mask_to(ref, tgt, match_area=True)
    achieved = float((aligned > 0.5).sum()) / float((ref > 0.5).sum())
    assert abs(achieved - 1.0) < 0.02, f"achieved area ratio {achieved:.4f}"
    assert params["area_measure"] == "pixels"
    assert params["area_ratio"] == pytest.approx(achieved, abs=1e-6)


def test_weighted_area_match_uses_the_weights():
    """With a weight field, equality must hold in the weighted measure."""
    ref = _disk(128, 128, 64, 64, 36)
    tgt = _disk(128, 128, 64, 64, 18)
    yy, xx = np.meshgrid(np.arange(128), np.arange(128), indexing="ij")
    weights = 1.0 + 3.0 * ((xx.astype(np.float64) / 127.0))  # heavier to the right

    aligned, params, _ = align_mask_to(
        ref, tgt, match_area=True, pixel_weights=weights
    )
    w_ref = weights[ref > 0.5].sum()
    w_al = weights[aligned > 0.5].sum()
    assert abs(w_al - w_ref) / w_ref < 0.02
    assert params["area_measure"] == "solid_angle"
