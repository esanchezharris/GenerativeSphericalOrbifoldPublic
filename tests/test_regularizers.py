"""Validate the shape regularizers, ending with the property that justifies them:
their gradient steers the edge weights back toward an undistorted embedding, through the
implicit solve, on CPU -- so the claim is proven before any GPU time is spent on it.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import SphericalEmbedder
from escher.OTE.core.spherical.regularizers import equal_area_loss, spherical_face_areas
from escher.OTE.tilings_sphere import DihedralOrbifold
from escher.geometry.spherical_base_mesh import get_lune_mesh
from escher.geometry.spherical_sanity_checks import signed_solid_angles


@pytest.fixture(scope="module")
def lune():
    return get_lune_mesh(k=4, n_theta=10, n_phi=7)


def as_t(x, dtype=torch.float64):
    return torch.as_tensor(np.asarray(x), dtype=dtype)


# ------------------------------------------------------------------- torch/numpy parity
def test_torch_areas_match_the_numpy_certificate(lune):
    """The loss and the validity certificate must measure the same quantity."""
    pts = as_t(lune.points)
    faces = torch.as_tensor(lune.faces, dtype=torch.long)
    torch_areas = spherical_face_areas(pts, faces).numpy()
    numpy_areas = signed_solid_angles(lune.points, lune.faces)
    assert np.abs(torch_areas - numpy_areas).max() < 1e-12


def test_torch_areas_match_on_a_distorted_mesh(lune):
    rng = np.random.default_rng(0)
    p = lune.points + rng.normal(scale=0.05, size=lune.points.shape)
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    faces = torch.as_tensor(lune.faces, dtype=torch.long)
    assert np.abs(
        spherical_face_areas(as_t(p), faces).numpy() - signed_solid_angles(p, lune.faces)
    ).max() < 1e-12


# ------------------------------------------------------------------------- loss shape
def test_loss_is_zero_at_the_reference(lune):
    pts = as_t(lune.points)
    faces = torch.as_tensor(lune.faces, dtype=torch.long)
    ref = spherical_face_areas(pts, faces).detach()
    assert float(equal_area_loss(pts, faces, ref)) == pytest.approx(0.0, abs=1e-24)


def test_loss_grows_monotonically_with_distortion(lune):
    faces = torch.as_tensor(lune.faces, dtype=torch.long)
    ref = spherical_face_areas(as_t(lune.points), faces).detach()
    rng = np.random.default_rng(1)
    noise = rng.normal(size=lune.points.shape)

    losses = []
    for scale in (0.01, 0.03, 0.06):
        p = lune.points + scale * noise
        p /= np.linalg.norm(p, axis=1, keepdims=True)
        losses.append(float(equal_area_loss(as_t(p), faces, ref)))
    assert losses[0] > 0
    assert losses == sorted(losses)


def test_flipped_faces_are_penalized_not_rewarded(lune):
    """Signed areas: a flipped face sits at ratio ~-1, cost ~4, rather than being counted
    by magnitude as if nothing happened."""
    faces_np = lune.faces.copy()
    faces = torch.as_tensor(faces_np, dtype=torch.long)
    pts = as_t(lune.points)
    ref = spherical_face_areas(pts, faces).detach()

    flipped = faces_np.copy()
    flipped[0] = flipped[0][[0, 2, 1]]  # invert one face's winding
    loss = float(equal_area_loss(pts, torch.as_tensor(flipped, dtype=torch.long), ref))
    assert loss > 3.0 / len(faces_np)  # the flipped face alone contributes ~4/n


def test_loss_is_differentiable_wrt_points(lune):
    faces = torch.as_tensor(lune.faces, dtype=torch.long)
    ref = spherical_face_areas(as_t(lune.points), faces).detach()
    rng = np.random.default_rng(2)
    p = lune.points + 0.03 * rng.normal(size=lune.points.shape)
    p /= np.linalg.norm(p, axis=1, keepdims=True)

    pts = as_t(p).requires_grad_(True)
    equal_area_loss(pts, faces, ref).backward()
    assert torch.isfinite(pts.grad).all()
    assert pts.grad.abs().sum() > 0


# ------------------------------------------------- the property that earns the GPU run
def test_regularizer_steers_weights_back_toward_uniform_through_the_solve():
    """Distort the embedding via non-uniform weights, then minimize ONLY the equal-area
    loss through the implicit solve. Area spread must shrink. This is the mechanism the
    next SDS run depends on, proven end to end on CPU.
    """
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=6, n_phi=5)
    mesh = orb.mesh
    faces = torch.as_tensor(mesh.faces, dtype=torch.long)
    ref = spherical_face_areas(as_t(mesh.points), faces).detach()
    embedder = SphericalEmbedder(mesh.edges, orb.A, orb.b, orb.initial_guess())

    rng = np.random.default_rng(3)
    W = torch.tensor(rng.normal(scale=2.0, size=len(mesh.edges)), requires_grad=True)

    def weights():
        return torch.special.expit(W) * 0.95 + 0.025

    def spread(points):
        a = np.abs(signed_solid_angles(points.detach().numpy(), mesh.faces))
        return float(np.percentile(a, 99) / np.percentile(a, 1))

    initial_loss = float(equal_area_loss(embedder(weights()), faces, ref))
    initial_spread = spread(embedder(weights()))
    assert initial_loss > 0.05, "distortion too small for the test to mean anything"

    opt = torch.optim.Adam([W], lr=0.1)
    for _ in range(25):
        opt.zero_grad()
        loss = equal_area_loss(embedder(weights()), faces, ref)
        loss.backward()
        opt.step()

    final_points = embedder(weights())
    final_loss = float(equal_area_loss(final_points, faces, ref))
    assert final_loss < 0.5 * initial_loss, (
        f"loss {initial_loss:.4f} -> {final_loss:.4f}: regularizer failed to descend "
        "through the implicit solve"
    )
    assert spread(final_points) < initial_spread
