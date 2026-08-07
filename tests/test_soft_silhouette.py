"""CPU validation of the analytic silhouette (escher/soft_silhouette.py).

This module replaced the rasterizer in the shape loss after a measured failure: the
renderer's alpha (and the RGB+alpha composite) return EXACTLY zero vertex gradients
under a uniform texture. These tests pin the replacement's geometry, gradients, and
conventions.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.tilings_sphere.boundary_explicit import BoundaryExplicitDihedral
from escher.rendering.camera import perspective
from escher.soft_silhouette import boundary_loop, project_to_pixels, soft_polygon_mask


def square(x0=10.0, y0=10.0, x1=30.0, y1=30.0, **kw):
    return torch.tensor([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], **kw)


# ------------------------------------------------------------------------- mask
def test_square_mask_matches_analytic_area():
    mask = soft_polygon_mask(square(), 40, 40, tau=0.5)
    assert mask.shape == (40, 40)
    assert mask[20, 20] > 0.99  # deep inside
    assert mask[5, 5] < 0.01  # far outside
    assert abs(float(mask.sum()) - 400.0) / 400.0 < 0.02


def test_mask_is_orientation_independent():
    a = soft_polygon_mask(square(), 40, 40, tau=1.0)
    b = soft_polygon_mask(square().flip(0), 40, 40, tau=1.0)
    assert torch.allclose(a, b, atol=1e-6)


def test_chunking_does_not_change_the_result():
    a = soft_polygon_mask(square(), 40, 40, tau=1.0, chunk=64)
    b = soft_polygon_mask(square(), 40, 40, tau=1.0, chunk=10**9)
    assert torch.allclose(a, b, atol=1e-6)


def test_area_gradient_has_the_right_sign_and_is_dense():
    """Moving the right edge right must GROW the area; every vertex must get grad."""
    pts = square(dtype=torch.float64).requires_grad_(True)
    soft_polygon_mask(pts, 40, 40, tau=1.5).sum().backward()
    g = pts.grad
    assert g is not None and (g != 0).any(dim=-1).all(), "every vertex needs gradient"
    assert g[1, 0] > 0 and g[2, 0] > 0, "right-edge x: outward grows area"
    assert g[0, 0] < 0 and g[3, 0] < 0, "left-edge x: outward (negative x) grows area"


# -------------------------------------------------------------------- projection
def test_projection_conventions():
    proj = perspective(fovy_deg=40.0)
    mv = torch.eye(4)
    pts = torch.tensor([[0.0, 0.0, -2.0], [0.5, 0.0, -2.0], [0.0, 0.5, -2.0]])
    px = project_to_pixels(pts, mv, proj, 100, 100)
    assert torch.allclose(px[0], torch.tensor([50.0, 50.0]), atol=1e-4)
    assert px[1, 0] > 50.0, "+x in view space goes right"
    assert px[2, 1] < 50.0, "+y in view space goes UP the image (renderer's y flip)"


# ------------------------------------------------------------------ boundary loop
@pytest.mark.parametrize("k", [3, 4, 6])
def test_boundary_loop_is_closed_and_complete(k):
    orb = BoundaryExplicitDihedral.from_resolution(k=k, n_theta=6, n_phi=5)
    mesh = orb.mesh
    loop = boundary_loop(mesh)

    assert len(loop) == len(set(loop)), "no repeats"
    assert set(loop) == set(mesh.left) | set(mesh.right) | set(mesh.bottom)

    pts = mesh.points[loop]
    nxt = np.roll(pts, -1, axis=0)
    gaps = np.arccos(np.clip((pts * nxt).sum(-1), -1, 1))
    assert gaps.max() < 4 * np.median(gaps), "consecutive loop points must be neighbors"
