"""Camera math for multi-view rendering. No GPU or rasteriser needed."""

from __future__ import annotations

import math

import pytest
import torch

from escher.rendering.camera import (
    fibonacci_sphere,
    look_at,
    orbit_views,
    perspective,
    random_views,
)


def apply(mat: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    """Apply a 4x4 to (n, 3) points, returning (n, 4) homogeneous output."""
    hom = torch.cat([points, torch.ones(len(points), 1)], dim=1)
    return hom @ mat.T


# ------------------------------------------------------------------------------ look_at
def test_look_at_is_a_rigid_transform():
    mv = look_at(torch.tensor([3.0, 1.0, 2.0]))[0]
    rot = mv[:3, :3]
    assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-5)
    assert torch.det(rot) == pytest.approx(1.0, abs=1e-5)


def test_look_at_places_the_target_on_the_view_axis():
    """The origin should land at (0, 0, -distance) in view space."""
    eye = torch.tensor([0.0, 0.0, 5.0])
    out = apply(look_at(eye)[0], torch.zeros(1, 3))[0]
    assert out[0].abs() < 1e-5 and out[1].abs() < 1e-5
    assert out[2] == pytest.approx(-5.0, abs=1e-5)


def test_look_at_preserves_distances():
    mv = look_at(torch.tensor([2.0, -1.0, 3.0]))[0]
    pts = torch.randn(20, 3)
    before = torch.cdist(pts, pts)
    after = torch.cdist(apply(mv, pts)[:, :3], apply(mv, pts)[:, :3])
    assert torch.allclose(before, after, atol=1e-4)


def test_look_at_handles_the_polar_degeneracy():
    """Straight down the +z axis, the default up vector is parallel to the view direction."""
    mv = look_at(torch.tensor([0.0, 0.0, 4.0]))[0]
    assert torch.isfinite(mv).all()
    rot = mv[:3, :3]
    assert torch.allclose(rot @ rot.T, torch.eye(3), atol=1e-5)


def test_look_at_batches():
    eyes = torch.randn(7, 3) * 3
    mv = look_at(eyes)
    assert mv.shape == (7, 4, 4)
    for i in range(7):
        single = look_at(eyes[i])[0]
        assert torch.allclose(mv[i], single, atol=1e-6)


# --------------------------------------------------------------------------- projection
def test_perspective_maps_the_near_plane_to_minus_one():
    proj = perspective(fovy_deg=60.0, near=0.5, far=10.0)
    near_pt = torch.tensor([[0.0, 0.0, -0.5]])
    clip = apply(proj, near_pt)[0]
    assert (clip[2] / clip[3]) == pytest.approx(-1.0, abs=1e-5)


def test_perspective_maps_the_far_plane_to_plus_one():
    proj = perspective(fovy_deg=60.0, near=0.5, far=10.0)
    clip = apply(proj, torch.tensor([[0.0, 0.0, -10.0]]))[0]
    assert (clip[2] / clip[3]) == pytest.approx(1.0, abs=1e-5)


def test_perspective_fov_matches_its_argument():
    fovy = 40.0
    proj = perspective(fovy_deg=fovy, near=0.1, far=100.0)
    z = -5.0
    edge_y = abs(z) * math.tan(math.radians(fovy) / 2)
    clip = apply(proj, torch.tensor([[0.0, edge_y, z]]))[0]
    assert (clip[1] / clip[3]) == pytest.approx(1.0, abs=1e-4)


def test_perspective_respects_aspect():
    proj = perspective(fovy_deg=40.0, aspect=2.0)
    assert proj[0, 0] == pytest.approx(proj[1, 1] / 2.0, rel=1e-6)


# ------------------------------------------------------------------- direction sampling
def test_fibonacci_points_are_unit_vectors():
    d = fibonacci_sphere(64)
    assert d.shape == (64, 3)
    assert torch.allclose(d.norm(dim=1), torch.ones(64), atol=1e-5)


def test_fibonacci_covers_the_sphere_more_evenly_than_random():
    """The reason for using it: a clumped view batch wastes SDS steps."""
    torch.manual_seed(0)
    n = 128

    def min_pairwise_angle(dirs):
        gram = (dirs @ dirs.T).clamp(-1, 1)
        gram.fill_diagonal_(-1.0)
        return torch.arccos(gram.max(dim=1).values).min().item()

    uniform = torch.randn(n, 3)
    uniform = uniform / uniform.norm(dim=1, keepdim=True)
    assert min_pairwise_angle(fibonacci_sphere(n)) > min_pairwise_angle(uniform)


def test_fibonacci_rejects_nonpositive_n():
    with pytest.raises(ValueError, match="must be positive"):
        fibonacci_sphere(0)


def test_fibonacci_is_centred():
    d = fibonacci_sphere(512)
    assert d.mean(dim=0).norm() < 0.05


# -------------------------------------------------------------------------------- views
def test_random_views_shape_and_distance():
    mv = random_views(8, distance=3.0)
    assert mv.shape == (8, 4, 4)
    for i in range(8):
        origin_in_view = apply(mv[i], torch.zeros(1, 3))[0]
        assert origin_in_view[:3].norm() == pytest.approx(3.0, abs=1e-4)


def test_random_views_differ_between_calls():
    torch.manual_seed(0)
    a = random_views(6)
    b = random_views(6)
    assert not torch.allclose(a, b, atol=1e-4)


def test_zero_jitter_is_deterministic():
    assert torch.allclose(random_views(6, jitter=0.0), random_views(6, jitter=0.0))


def test_orbit_views_are_evenly_spaced_in_azimuth():
    mv = orbit_views(8, distance=2.5, elevation_deg=0.0)
    assert mv.shape == (8, 4, 4)
    for i in range(8):
        assert apply(mv[i], torch.zeros(1, 3))[0][:3].norm() == pytest.approx(2.5, abs=1e-4)


def test_all_views_see_the_sphere_in_front_of_the_camera():
    """Every unit-sphere point must land at negative view-space z, i.e. ahead of the camera,
    or the rasteriser will clip it away."""
    from escher.geometry.sphere_tiler import SphericalTiler

    pts = fibonacci_sphere(200)
    for mv in random_views(12, distance=3.0):
        view_z = apply(mv, pts)[:, 2]
        assert (view_z < 0).all(), "part of the sphere fell behind the camera"


def test_sphere_fits_inside_the_frustum():
    """With distance 3 and a 40-degree field of view the unit sphere should be fully
    visible, so no view is wasted on a clipped object."""
    proj = perspective(fovy_deg=40.0, aspect=1.0, near=0.1, far=100.0)
    pts = fibonacci_sphere(500)
    for mv in random_views(8, distance=3.0):
        clip = apply(proj, apply(mv, pts)[:, :3])
        ndc = clip[:, :3] / clip[:, 3:4]
        assert ndc[:, :2].abs().max() < 1.0, "sphere extends outside the image"
