"""End-to-end spherical orbifold embedding with no MATLAB data involved.

This is the test that de-risks the project. Aigerman & Lipman prove bijectivity for embedding
a *given* mesh; the Escher premise instead *generates* a tile by varying the edge weights, and
needs the embedding to stay bijective as they vary. ``test_random_weights_stay_bijective``
below is the direct check: random weights, zero flipped triangles.
"""

from __future__ import annotations

import numpy as np
import pytest

from escher.OTE.core.spherical.affine_space import AffineSpace
from escher.OTE.core.spherical.solver import (
    laplacian_from_edges,
    solve_spherical_embedding,
)
from escher.OTE.tilings_sphere import DihedralOrbifold, SparseSystem3D
from escher.geometry.sphere_tiler import rotation_matrix
from escher.geometry.spherical_base_mesh import get_lune_mesh
from escher.geometry.spherical_sanity_checks import (
    FULL_SPHERE,
    check_covers_sphere_once,
    count_flipped_faces,
    total_solid_angle,
)


def tiled_certificate(orb: DihedralOrbifold, points: np.ndarray) -> tuple[bool, str]:
    """Tile the embedded domain by its group and certify the result covers the sphere once."""
    verts, faces, _ = orb.tiler().tile_mesh(points, orb.mesh.faces)
    return check_covers_sphere_once(verts, faces)


def embed(orb: DihedralOrbifold, weights=None, **kw):
    mesh = orb.mesh
    edges = mesh.edges
    w = np.ones(len(edges)) if weights is None else weights
    return solve_spherical_embedding(
        edges=edges,
        weights=w,
        laplacian=laplacian_from_edges(edges, w, mesh.n_verts),
        A=orb.A,
        b=orb.b,
        x0=orb.initial_guess(),
        **kw,
    )


# ------------------------------------------------------------------------------- mesh
def test_lune_mesh_is_a_disk():
    mesh = get_lune_mesh(k=4, n_theta=12, n_phi=9)
    euler = mesh.n_verts - len(mesh.edges) + len(mesh.faces)
    assert euler == 1, "fundamental domain must have disk topology"


def test_lune_mesh_lies_on_the_unit_sphere():
    mesh = get_lune_mesh(k=5, n_theta=10, n_phi=7)
    assert np.allclose(np.linalg.norm(mesh.points, axis=1), 1.0, atol=1e-12)


def test_lune_faces_are_outward_oriented():
    mesh = get_lune_mesh(k=4, n_theta=10, n_phi=7)
    assert count_flipped_faces(mesh.points, mesh.faces) == 0


def test_lune_spans_the_right_angular_width():
    k = 6
    mesh = get_lune_mesh(k=k, n_theta=10, n_phi=7)
    off_axis = mesh.points[np.hypot(mesh.points[:, 0], mesh.points[:, 1]) > 1e-6]
    lon = np.degrees(np.arctan2(off_axis[:, 1], off_axis[:, 0]))
    assert lon.max() - lon.min() == pytest.approx(360.0 / k, abs=1e-6)


def test_lune_boundary_indices_are_consistent():
    mesh = get_lune_mesh(k=4, n_theta=8, n_phi=9)
    assert mesh.left[0] == mesh.pole and mesh.right[0] == mesh.pole
    assert mesh.left[-1] == mesh.bottom_left
    assert mesh.right[-1] == mesh.bottom_right
    assert mesh.bottom_mid in mesh.bottom.tolist()
    assert np.allclose(mesh.points[mesh.pole], [0, 0, 1])


def test_even_n_phi_is_rejected():
    """An even count puts no vertex on the 2-fold cone at the arc's midpoint."""
    with pytest.raises(ValueError, match="must be odd"):
        get_lune_mesh(k=4, n_theta=8, n_phi=8)


@pytest.mark.parametrize("kwargs", [{"k": 1}, {"n_theta": 1}, {"n_phi": 1}])
def test_degenerate_resolutions_are_rejected(kwargs):
    base = {"k": 4, "n_theta": 8, "n_phi": 9}
    with pytest.raises(ValueError):
        get_lune_mesh(**{**base, **kwargs})


# ------------------------------------------------------------------------ constraints
def test_constraint_matrix_has_full_row_rank():
    """AffineSpace rejects dependent rows, so constructing it is the rank check."""
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=12, n_phi=9)
    affine = AffineSpace(orb.A, orb.b)
    assert affine.n_constraints == orb.A.shape[0]
    assert affine.dim_null == orb.A.shape[1] - orb.A.shape[0]


def test_initial_guess_is_feasible():
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=12, n_phi=9)
    assert AffineSpace(orb.A, orb.b).constraint_violation(orb.initial_guess()) < 1e-10


def test_rotation_constraint_encodes_R_u_minus_v():
    """Three rows per pair, of the form ``R u_i - u_j = 0`` -- the form found in the
    reference A matrix."""
    R = rotation_matrix([0, 0, 1], np.pi / 2)
    sys = SparseSystem3D(n_vertices=4)
    sys.add_rotation_constraints([0], [1], R)
    A, b = sys.build()

    assert A.shape == (3, 12)
    assert np.allclose(b, 0.0)

    x = np.zeros(12)
    u = np.array([0.3, -0.7, 0.5])
    x[0:3] = u
    x[3:6] = R @ u
    assert np.abs(A @ x - b).max() < 1e-12  # satisfied exactly when v = R u

    x[3:6] = u  # and violated otherwise
    assert np.abs(A @ x - b).max() > 1e-3


def test_rotation_constraint_rejects_overlapping_endpoints():
    sys = SparseSystem3D(n_vertices=4)
    with pytest.raises(ValueError, match="both source and target"):
        sys.add_rotation_constraints([0, 1], [1, 2], np.eye(3))


def test_fixed_constraint_pins_all_three_coordinates():
    sys = SparseSystem3D(n_vertices=2)
    sys.add_fixed_constraint(1, [0.0, 1.0, 0.0])
    A, b = sys.build()
    assert A.shape == (3, 6)
    x = np.zeros(6)
    x[3:6] = [0.0, 1.0, 0.0]
    assert np.abs(A @ x - b).max() < 1e-12


# -------------------------------------------------------------------------- end to end
@pytest.mark.parametrize("k", [3, 4, 5, 6, 8])
def test_embeds_and_tiles_for_each_k(k):
    orb = DihedralOrbifold.from_resolution(k=k, n_theta=16, n_phi=11)
    result = embed(orb)

    assert result.constraint_violation < 1e-6
    assert np.allclose(np.linalg.norm(result.points, axis=1), 1.0, atol=1e-12)
    assert result.stage2.proj_grad_inf_history[-1] < 1e-7

    assert orb.tiler().order == 2 * k
    assert count_flipped_faces(result.points, orb.mesh.faces) == 0

    ok, message = tiled_certificate(orb, result.points)
    assert ok, message


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_random_weights_stay_bijective(seed):
    """**The Escher premise.**

    The generative pipeline produces a tile by pushing the edge weights around, so the
    embedding has to remain a valid, fold-free tiling for weights well away from uniform.
    Here they are drawn over more than two orders of magnitude.

    Certified by total signed area rather than by binned coverage: a distorted tile clumps
    its vertices, so bins go empty while the surface is still fully covered. Area does not
    care how the vertices are spread.
    """
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=16, n_phi=11)
    rng = np.random.default_rng(seed)
    weights = 10.0 ** rng.uniform(-1.0, 1.0, size=len(orb.mesh.edges))

    result = embed(orb, weights)

    assert result.constraint_violation < 1e-6
    assert count_flipped_faces(result.points, orb.mesh.faces) == 0, "embedding folded over"

    ok, message = tiled_certificate(orb, result.points)
    assert ok, message


def test_area_certificate_rejects_a_deliberately_broken_tiling():
    """Control: the certificate must fail on something that is not a valid tiling."""
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=12, n_phi=9)
    points = embed(orb).points

    # using too small a group leaves most of the sphere uncovered
    from escher.geometry.sphere_tiler import SphericalTiler

    verts, faces, _ = SphericalTiler.dihedral(2).tile_mesh(points, orb.mesh.faces)
    ok, message = check_covers_sphere_once(verts, faces)
    assert not ok and "gaps" in message

    # and too large a group double-covers it
    verts, faces, _ = SphericalTiler.dihedral(8).tile_mesh(points, orb.mesh.faces)
    ok, message = check_covers_sphere_once(verts, faces)
    assert not ok and "overlaps" in message


def test_single_domain_has_the_expected_area():
    """One fundamental domain must be exactly 1/2k of the sphere."""
    for k in (3, 4, 6):
        orb = DihedralOrbifold.from_resolution(k=k, n_theta=16, n_phi=11)
        area = total_solid_angle(embed(orb).points, orb.mesh.faces)
        assert area == pytest.approx(FULL_SPHERE / (2 * k), rel=1e-6)


def test_different_weights_give_different_tiles():
    """Sanity: the weights actually control the shape, so there is something to optimise."""
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=16, n_phi=11)
    rng = np.random.default_rng(7)
    a = embed(orb, 10.0 ** rng.uniform(-1, 1, size=len(orb.mesh.edges)))
    b = embed(orb, 10.0 ** rng.uniform(-1, 1, size=len(orb.mesh.edges)))
    assert np.abs(a.points - b.points).max() > 1e-2


def test_uniform_weights_are_reproducible():
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=16, n_phi=11)
    assert embed(orb).energy == pytest.approx(embed(orb).energy, rel=1e-12)


@pytest.mark.parametrize("k", [3, 4, 6])
def test_solved_cut_sides_are_exactly_related_by_the_rotation(k):
    """The orbifold property, checked on the *output* rather than the constraint residual.

    The k-fold rotation must carry one side of the cut exactly onto the other. This is what
    makes neighbouring tiles interlock with no seam, so it is worth asserting directly on the
    embedded geometry.
    """
    orb = DihedralOrbifold.from_resolution(k=k, n_theta=16, n_phi=11)
    points = embed(orb).points
    mesh = orb.mesh

    Rz = rotation_matrix([0.0, 0.0, 1.0], 2.0 * np.pi / k)
    rotated_left = points[mesh.left] @ Rz.T
    assert np.abs(rotated_left - points[mesh.right]).max() < 1e-9


@pytest.mark.parametrize("k", [3, 4, 6])
def test_solved_equator_arc_folds_onto_itself(k):
    """The half-turn about the arc's midpoint maps the equator arc to itself, reversed."""
    orb = DihedralOrbifold.from_resolution(k=k, n_theta=16, n_phi=11)
    points = embed(orb).points
    bottom = orb.mesh.bottom

    Rx = rotation_matrix([1.0, 0.0, 0.0], np.pi)
    folded = points[bottom] @ Rx.T
    assert np.abs(folded - points[bottom[::-1]]).max() < 1e-9


# ---------------------------------------------------------------------------- laplacian
def test_laplacian_convention():
    edges = np.array([[0, 1], [1, 2]])
    L = laplacian_from_edges(edges, np.array([2.0, 3.0]), 3).toarray()
    assert np.allclose(L.sum(axis=1), 0.0)
    assert L[0, 1] == 2.0 and L[1, 2] == 3.0
    assert L[0, 0] == -2.0 and L[1, 1] == -5.0
    assert np.linalg.eigvalsh(L).max() < 1e-12, "should be negative semi-definite"


def test_laplacian_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="edges but"):
        laplacian_from_edges(np.array([[0, 1]]), np.array([1.0, 2.0]), 2)
