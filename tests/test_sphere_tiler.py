"""Validate the spherical symmetry groups and the tiling of a fundamental domain."""

from __future__ import annotations

import numpy as np
import pytest

from escher.OTE.core.spherical.solver import solve_spherical_embedding
from escher.geometry.sphere_tiler import (
    SphericalTiler,
    expected_group_order,
    generate_rotation_group,
    rotation_matrix,
)
from tests.golden import load_golden


# ------------------------------------------------------------------------ group orders
@pytest.mark.parametrize(
    "cones,order",
    [
        ((2, 2, 2), 4),
        ((3, 2, 2), 6),
        ((4, 2, 2), 8),
        ((6, 2, 2), 12),
        ((2, 3, 3), 12),  # tetrahedral
        ((2, 3, 4), 24),  # octahedral
        ((2, 3, 5), 60),  # icosahedral
    ],
)
def test_expected_group_order(cones, order):
    assert expected_group_order(cones) == order


@pytest.mark.parametrize("cones", [(3, 3, 3), (2, 4, 4), (2, 3, 6), (7, 3, 2), (2, 3, 7)])
def test_rejects_non_spherical_orbifolds(cones):
    """1/p+1/q+1/r <= 1 is Euclidean or hyperbolic -- the planar code's territory."""
    with pytest.raises(ValueError, match="not a spherical one"):
        expected_group_order(cones)


# --------------------------------------------------------------------- rotation basics
def test_rotation_matrix_is_a_rotation():
    R = rotation_matrix([1.0, 2.0, 3.0], 0.7)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0)


def test_rotation_matrix_known_case():
    R = rotation_matrix([0, 0, 1], np.pi / 2)
    assert np.allclose(R @ np.array([1.0, 0, 0]), [0, 1, 0], atol=1e-12)


def test_rotation_axis_must_be_nonzero():
    with pytest.raises(ValueError, match="non-zero"):
        rotation_matrix([0.0, 0.0, 0.0], 1.0)


def test_group_closure_detects_non_finite_generators():
    with pytest.raises(ValueError, match="did not close"):
        generate_rotation_group([rotation_matrix([0, 0, 1], 1.0)], max_size=50)


# ----------------------------------------------------------------------------- groups
@pytest.mark.parametrize("k", [2, 3, 4, 5, 6, 8])
def test_dihedral_group_has_order_2k(k):
    tiler = SphericalTiler.dihedral(k)
    assert tiler.order == 2 * k


def test_dihedral_axes_must_be_perpendicular():
    with pytest.raises(ValueError, match="perpendicular"):
        SphericalTiler.dihedral(4, principal_axis=(0, 0, 1), secondary_axis=(0, 0.5, 1))


def test_group_elements_are_distinct_rotations():
    tiler = SphericalTiler.dihedral(4)
    for R in tiler.rotations:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-10)
    flat = tiler.rotations.reshape(tiler.order, 9)
    dists = np.abs(flat[:, None, :] - flat[None, :, :]).max(axis=2)
    np.fill_diagonal(dists, 1.0)
    assert dists.min() > 1e-6, "group contains duplicate elements"


def test_group_is_closed_under_multiplication():
    tiler = SphericalTiler.dihedral(4)
    flat = tiler.rotations.reshape(tiler.order, 9)
    for a in tiler.rotations:
        for b in tiler.rotations:
            prod = (a @ b).reshape(9)
            assert np.abs(flat - prod).max(axis=1).min() < 1e-9


def test_generators_inconsistent_with_cone_orders_are_rejected():
    with pytest.raises(ValueError, match="order"):
        SphericalTiler.from_generators(
            [rotation_matrix([0, 0, 1], 2 * np.pi / 3)], cone_orders=(4, 2, 2)
        )


# ------------------------------------------------------------- tiling the real solution
@pytest.fixture(scope="module")
def embedded_domain():
    """The solved reference embedding: one fundamental domain of the (4,2,2) orbifold."""
    g = load_golden()
    result = solve_spherical_embedding(
        edges=g.edges, weights=g.edge_weights, laplacian=g.wmat, A=g.A, b=g.b, x0=g.x0
    )
    return g, result.points


def test_reference_problem_is_the_422_orbifold(embedded_domain):
    """The pinned cone points identify the orbifold; this documents which one it is.

    Four vertices are pinned by single-nonzero constraint rows: the pole (0,0,1) and three
    equatorial points at longitudes 45/90/135 degrees. That is (k,2,2) with k=4.
    """
    g, points = embedded_domain
    A = g.A.tocsr()
    single_rows = np.where(np.diff(A.indptr) == 1)[0]
    pinned = sorted({int(A.indices[A.indptr[r]]) // 3 for r in single_rows})
    assert len(pinned) == 4

    coords = points[pinned]
    assert np.allclose(np.linalg.norm(coords, axis=1), 1.0)
    # one pole and three points on the equator
    n_polar = int(np.sum(np.abs(np.abs(coords[:, 2]) - 1.0) < 1e-9))
    n_equatorial = int(np.sum(np.abs(coords[:, 2]) < 1e-9))
    assert n_polar == 1 and n_equatorial == 3


def test_domain_reaches_the_pole_and_stops_at_the_equator(embedded_domain):
    """|D_4| = 8, so the domain is 1/8 of the sphere: pole to equator."""
    _, points = embedded_domain
    assert points[:, 2].max() == pytest.approx(1.0, abs=1e-9), "should reach the pole"
    assert points[:, 2].min() > -0.06, "should not extend past the equator"


def test_domain_has_constant_ninety_degree_angular_width(embedded_domain):
    """The domain is a lune of 2*pi/k angular width -- but a *free-form* one.

    Its boundary is not a pair of meridians. The orbifold condition only requires that the
    k-fold rotation carry one side of the cut onto the other, so the boundary is free to
    wiggle as long as the width stays 2*pi/k. Measured here: the width holds at ~90 degrees
    in every latitude band while the absolute longitudes range over 39.7-135 degrees rather
    than a clean 45-135.

    That freedom is exactly what makes an Escher tile possible -- it is the degree of
    freedom the generative pipeline optimises over.
    """
    _, points = embedded_domain
    off_axis_mask = np.hypot(points[:, 0], points[:, 1]) > 0.1

    for lo, hi in [(0.05, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 0.95)]:
        band = off_axis_mask & (points[:, 2] >= lo) & (points[:, 2] < hi)
        assert band.sum() > 20, f"too few samples in band z in [{lo}, {hi})"
        lon = np.degrees(np.arctan2(points[band, 1], points[band, 0]))
        assert lon.max() - lon.min() == pytest.approx(90.0, abs=4.0), (
            f"band z in [{lo}, {hi}) spans {lon.max() - lon.min():.2f} degrees"
        )

    # the boundary genuinely departs from a straight 45-degree meridian
    all_lon = np.degrees(np.arctan2(points[off_axis_mask, 1], points[off_axis_mask, 0]))
    assert all_lon.min() < 44.0, "expected the free-form boundary to bulge past 45 degrees"


def test_eight_copies_cover_the_sphere(embedded_domain):
    _, points = embedded_domain
    tiler = SphericalTiler.dihedral(4)
    assert tiler.order == 8
    assert tiler.coverage_fraction(points, n_bins=200) > 0.99


def test_a_smaller_group_fails_to_cover(embedded_domain):
    """Control: the wrong group leaves gaps, so the coverage check has teeth."""
    _, points = embedded_domain
    assert SphericalTiler.dihedral(2).coverage_fraction(points, n_bins=200) < 0.75


def test_tiled_set_is_invariant_under_the_group(embedded_domain):
    """Applying any group element to the union of copies must reproduce the union."""
    _, points = embedded_domain
    tiler = SphericalTiler.dihedral(4)
    tiled = tiler.tile_points(points).reshape(-1, 3)

    reference = np.sort(np.round(tiled, 6), axis=0)
    for R in tiler.rotations:
        moved = np.sort(np.round(tiled @ R.T, 6), axis=0)
        assert np.abs(moved - reference).max() < 1e-4


def test_tile_mesh_offsets_faces_correctly():
    points = np.array([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = np.array([[0, 1, 2]])
    tiler = SphericalTiler.dihedral(4)

    verts, all_faces, tile_index = tiler.tile_mesh(points, faces)
    assert verts.shape == (8 * 3, 3)
    assert all_faces.shape == (8, 3)
    assert tile_index.shape == (8,)
    assert all_faces.max() == 8 * 3 - 1
    # copy g must reference exactly the vertex block of copy g
    for g in range(tiler.order):
        assert set(all_faces[g].tolist()) == {g * 3, g * 3 + 1, g * 3 + 2}


def test_tiled_vertices_stay_on_the_unit_sphere(embedded_domain):
    _, points = embedded_domain
    tiled = SphericalTiler.dihedral(4).tile_points(points).reshape(-1, 3)
    assert np.allclose(np.linalg.norm(tiled, axis=1), 1.0, atol=1e-12)
