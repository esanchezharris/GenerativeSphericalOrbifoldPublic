"""Validate the boundary-explicit parameterization end to end on CPU.

The critical check is the finite-difference validation of dL/d(free boundary points): the
backward path is adjoint multipliers -> b -> group maps -> free points, and a sign or
ordering error anywhere would still produce plausible-looking gradients.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import BoundaryEmbedder
from escher.OTE.tilings_sphere.boundary_explicit import BoundaryExplicitDihedral
from escher.geometry.sphere_tiler import rotation_matrix
from escher.geometry.spherical_sanity_checks import (
    boundary_arc_ratio,
    check_covers_sphere_once,
    count_flipped_faces,
)


@pytest.fixture(scope="module")
def small():
    orb = BoundaryExplicitDihedral.from_resolution(k=4, n_theta=6, n_phi=5)
    emb = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points, warm_start=False
    )
    return orb, emb


def wiggled(orb, scale=0.06, seed=0) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    P = orb.initial_free_points().numpy()
    P = P + scale * rng.standard_normal(P.shape)
    return torch.as_tensor(P, dtype=torch.float64)


# ------------------------------------------------------------------------- structure
def test_pins_cover_the_whole_boundary(small):
    orb, _ = small
    boundary = set(orb.mesh.left) | set(orb.mesh.right) | set(orb.mesh.bottom)
    assert set(orb.pin_order.tolist()) == boundary
    assert orb.A.shape[0] == 3 * len(boundary)


def test_initial_b_reproduces_the_lune_boundary(small):
    orb, _ = small
    b = orb.boundary_b(orb.initial_free_points()).numpy().reshape(-1, 3)
    assert np.abs(b - orb.mesh.points[orb.pin_order]).max() < 1e-12


def test_generated_mates_are_exact_group_images(small):
    """Interlocking by construction: the right chain IS R_z of the left chain."""
    orb, emb = small
    P = wiggled(orb, seed=1)
    with torch.no_grad():
        pts = emb(orb.boundary_b(P)).numpy()

    rz = rotation_matrix([0, 0, 1], 2 * np.pi / orb.mesh.k)
    left, right = pts[orb.mesh.left], pts[orb.mesh.right]
    assert np.abs(left @ rz.T - right).max() < 1e-9

    rx = rotation_matrix([1, 0, 0], np.pi)
    bottom = pts[orb.mesh.bottom]
    assert np.abs(bottom @ rx.T - bottom[::-1]).max() < 1e-9


# ------------------------------------------------------------------------ validity
def test_initial_solve_stays_near_the_lune(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(orb.initial_free_points())).numpy()
    assert count_flipped_faces(pts, orb.mesh.faces) == 0
    assert np.abs(pts - orb.mesh.points).max() < 0.15


def test_wiggled_boundary_yields_a_valid_tiling(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(wiggled(orb, seed=2))).numpy()

    assert count_flipped_faces(pts, orb.mesh.faces) == 0
    perim = boundary_arc_ratio(
        pts, orb.mesh.points, (orb.mesh.left, orb.mesh.right, orb.mesh.bottom)
    )
    assert perim > 1.005, "a wiggled boundary must lengthen the outline"

    verts, faces, _ = orb.tiler().tile_mesh(pts, orb.mesh.faces)
    ok, msg = check_covers_sphere_once(verts, faces)
    assert ok, msg


def test_cone_points_stay_pinned_regardless_of_parameters(small):
    orb, emb = small
    with torch.no_grad():
        pts = emb(orb.boundary_b(wiggled(orb, seed=3))).numpy()
    assert np.allclose(pts[orb.mesh.pole], [0, 0, 1], atol=1e-9)
    assert np.allclose(pts[orb.mesh.bottom_mid], [1, 0, 0], atol=1e-9)


# ------------------------------------------------------------------ the actual gate
def test_boundary_gradient_matches_finite_differences(small):
    """dL/dP through solve + adjoint + group maps vs central differences.

    Same discipline as the weight-path check: tight solver tolerance, generous h -- the
    forward solve is iterative and its residual error divides by h.
    """
    orb, _ = small
    emb = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points,
        warm_start=False, tol_x=1e-14,
    )
    rng = np.random.default_rng(4)
    P0 = wiggled(orb, scale=0.04, seed=4)
    target = torch.as_tensor(
        rng.standard_normal((orb.mesh.n_verts, 3)), dtype=torch.float64
    )

    def loss_of(P):
        return (emb(orb.boundary_b(P)) * target).sum()

    P = P0.clone().requires_grad_(True)
    loss_of(P).backward()
    analytic = P.grad.clone()
    assert torch.isfinite(analytic).all()
    assert analytic.abs().max() > 1e-8

    h = 1e-4
    rel_errors = []
    picks = [(i, k) for i in rng.choice(orb.n_free, 6, replace=False) for k in range(3)]
    for i, k in picks:
        plus, minus = P0.clone(), P0.clone()
        plus[i, k] += h
        minus[i, k] -= h
        fd = (loss_of(plus) - loss_of(minus)).item() / (2 * h)
        a = analytic[i, k].item()
        rel_errors.append(abs(fd - a) / max(abs(a), 1e-9))
        assert fd == pytest.approx(a, rel=1e-3, abs=1e-8), f"free point {i} axis {k}"
    assert float(np.median(rel_errors)) < 1e-5


def test_gradient_reaches_free_points_through_generated_mates(small):
    """Perturbing only what a GENERATED vertex sees must still grade the free source."""
    orb, emb = small
    P = orb.initial_free_points().clone().requires_grad_(True)
    pts = emb(orb.boundary_b(P))

    # loss touches ONLY the generated right chain
    loss = pts[orb.mesh.right[1:-1]].sum()
    loss.backward()
    left_grad = P.grad[: orb.n_free_left]
    assert left_grad.abs().sum() > 0, "gradient failed to flow through the rotation"


# ---------------------------------------------------------------------- warm start
def test_warm_start_reduces_iterations_and_matches_cold(small):
    orb, _ = small
    cold = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points, warm_start=False
    )
    warm = BoundaryEmbedder(
        orb.mesh.edges, orb.A, orb.pin_order, orb.mesh.points, warm_start=True
    )
    base = wiggled(orb, scale=0.03, seed=5)
    for step in range(4):
        Pn = base * (1.0 + 0.002 * step)
        with torch.no_grad():
            pc = cold(orb.boundary_b(Pn))
            pw = warm(orb.boundary_b(Pn))
        assert torch.allclose(pc, pw, atol=1e-6)
    assert warm.total_iterations < cold.total_iterations
