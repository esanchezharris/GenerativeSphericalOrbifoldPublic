"""Validate the three-cone kite domain (escher/geometry/spherical_kite_mesh.py).

The corner layout is the reference's three_point_tile.m; everything here is pinned
against group theory rather than stored numbers: the kite must have area 4*pi/|G|, its
cut chains must map onto each other under the cone rotations EXACTLY, and the mesh must
be a valid outward-oriented disk.
"""

from __future__ import annotations

import numpy as np
import pytest

from escher.geometry.sphere_tiler import expected_group_order, rotation_matrix
from escher.geometry.spherical_kite_mesh import get_kite_mesh, three_point_tile
from escher.geometry.spherical_sanity_checks import (
    count_flipped_faces,
    signed_solid_angles,
)
from escher.soft_silhouette import boundary_loop

TRIPLES = [(2, 3, 3), (2, 3, 4), (2, 3, 5)]


def cone_rotations(cone_orders, corners):
    """The two cut rotations, signs resolved numerically: about cone1 mapping
    cone2a -> cone2b, and about cone3 doing the same."""
    c1, c2a, c3, c2b = corners
    out = []
    for axis, order in ((c1, cone_orders[0]), (c3, cone_orders[2])):
        for sign in (1.0, -1.0):
            R = rotation_matrix(axis, sign * 2 * np.pi / order)
            if np.allclose(R @ c2a, c2b, atol=1e-9):
                out.append(R)
                break
        else:
            raise AssertionError("no cone rotation maps c2a to c2b")
    return out


# ------------------------------------------------------------------------ corners
@pytest.mark.parametrize("orders", TRIPLES)
def test_corners_are_consistent_with_the_group(orders):
    corners = three_point_tile(orders)
    assert np.allclose(np.linalg.norm(corners, axis=1), 1.0, atol=1e-12)
    R1, R2 = cone_rotations(orders, corners)  # raises if no rotation matches
    c1, c2a, c3, c2b = corners
    assert np.allclose(R1 @ c1, c1, atol=1e-12)
    assert np.allclose(R2 @ c3, c3, atol=1e-12)


@pytest.mark.parametrize("orders", TRIPLES)
def test_kite_area_is_the_fundamental_domain_area(orders):
    mesh = get_kite_mesh(orders, n=16)
    area = signed_solid_angles(mesh.points, mesh.faces).sum()
    expected = 4 * np.pi / expected_group_order(orders)
    # the mesh chords the true spherical kite, so allow a small resolution-dependent gap
    assert area == pytest.approx(expected, rel=2e-3)


# --------------------------------------------------------------------------- mesh
@pytest.mark.parametrize("orders", TRIPLES)
def test_mesh_is_a_valid_outward_disk(orders):
    mesh = get_kite_mesh(orders, n=10)
    assert count_flipped_faces(mesh.points, mesh.faces) == 0
    n_edges = len(mesh.edges)
    assert mesh.n_verts - n_edges + len(mesh.faces) == 1, "Euler characteristic of a disk"
    assert np.allclose(np.linalg.norm(mesh.points, axis=1), 1.0, atol=1e-12)


def test_chains_share_corners_and_have_matching_lengths():
    mesh = get_kite_mesh((2, 3, 4), n=8)
    assert len(mesh.left1) == len(mesh.right1) == 9
    assert len(mesh.left2) == len(mesh.right2) == 9
    assert mesh.left1[0] == mesh.right1[0] == mesh.cone1
    assert mesh.left1[-1] == mesh.cone2a
    assert mesh.right1[-1] == mesh.cone2b
    assert mesh.left2[0] == mesh.cone2a and mesh.left2[-1] == mesh.cone3
    assert mesh.right2[0] == mesh.cone2b and mesh.right2[-1] == mesh.cone3


@pytest.mark.parametrize("orders", TRIPLES)
def test_undeformed_chains_are_exact_rotation_images(orders):
    """right_i[j] == R_i @ left_i[j] EXACTLY on the undeformed kite -- normalized
    barycentric sampling commutes with the rotation, so this is not approximate."""
    mesh = get_kite_mesh(orders, n=12)
    R1, R2 = cone_rotations(orders, mesh.metadata["corners"])
    pts = mesh.points
    assert np.abs(pts[mesh.left1] @ R1.T - pts[mesh.right1]).max() < 1e-12
    assert np.abs(pts[mesh.left2] @ R2.T - pts[mesh.right2]).max() < 1e-12


def test_boundary_loop_stitches_the_kite():
    mesh = get_kite_mesh((2, 3, 4), n=8)
    loop = boundary_loop(mesh)
    assert len(loop) == len(set(loop))
    expected = set(np.concatenate(mesh.boundary_chains).tolist())
    assert set(loop.tolist()) == expected

    pts = mesh.points[loop]
    nxt = np.roll(pts, -1, axis=0)
    gaps = np.arccos(np.clip((pts * nxt).sum(-1), -1, 1))
    assert gaps.max() < 4 * np.median(gaps)


def test_uv_is_injective_and_in_range():
    mesh = get_kite_mesh((2, 3, 4), n=10)
    assert mesh.uv.min() >= 0.049 and mesh.uv.max() <= 0.951
    # no two vertices share a uv (bijective flattening of a convex kite)
    rounded = {tuple(np.round(p, 9)) for p in mesh.uv}
    assert len(rounded) == mesh.n_verts
