"""Assert the invariants the golden data is supposed to satisfy.

These are cheap and run first: if any fails, every downstream test that leans on the
reference dumps is suspect, and we want to know that before debugging the port.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from tests.golden import as_points, load_golden

N_VERTS_CUT = 2554
N_VERTS_UNCUT = 2502
N_FACES = 5000
N_EDGES_CUT = 7553
N_CONSTRAINTS = 165


def test_shapes():
    g = load_golden()
    assert g.A.shape == (N_CONSTRAINTS, 3 * N_VERTS_CUT)
    assert g.b.shape == (N_CONSTRAINTS,)
    assert g.N.shape == (3 * N_VERTS_CUT, 3 * N_VERTS_CUT - N_CONSTRAINTS)
    assert g.x0.shape == (3 * N_VERTS_CUT,)
    assert g.wmat.shape == g.L.shape == (N_VERTS_CUT, N_VERTS_CUT)
    assert g.V.shape == (N_VERTS_UNCUT, 3)
    assert g.F.shape == (N_FACES, 3)


def test_faces_are_zero_based_after_load():
    g = load_golden()
    assert g.F.min() == 0
    assert g.F.max() == N_VERTS_UNCUT - 1


def test_N_is_null_space_of_A():
    g = load_golden()
    assert np.abs((g.A @ g.N).toarray()).max() < 1e-12


def test_N_has_orthonormal_columns():
    g = load_golden()
    gram = (g.N.T @ g.N) - sp.eye(g.N.shape[1], format="csr")
    assert np.abs(gram.data).max() < 1e-12 if gram.nnz else True


def test_N_is_sparse_not_dense():
    """Guards against regressing to ``np.linalg.qr(mode='complete')``, which would be
    a 7662x7662 dense factor (~470 MB) instead of 7884 nonzeros."""
    g = load_golden()
    assert g.N.nnz < 10_000


def test_x0_satisfies_constraints():
    g = load_golden()
    assert np.abs(g.A @ g.x0 - g.b).max() < 1e-12


def test_x0_lies_on_unit_sphere():
    g = load_golden()
    radii = np.linalg.norm(as_points(g.x0), axis=1)
    assert np.allclose(radii, 1.0, atol=1e-12)


def test_uncut_mesh_has_sphere_topology():
    g = load_golden()
    edges = {tuple(sorted(e)) for f in g.F for e in ((f[0], f[1]), (f[1], f[2]), (f[2], f[0]))}
    euler = N_VERTS_UNCUT - len(edges) + N_FACES
    assert len(edges) == 7500
    assert euler == 2, "uncut mesh should be a sphere"


def test_cut_mesh_has_disk_topology():
    """The solver's mesh has 52 vertices duplicated along the cut, making it a disk."""
    g = load_golden()
    assert len(g.edges) == N_EDGES_CUT
    euler = N_VERTS_CUT - N_EDGES_CUT + N_FACES
    assert euler == 1, "cut mesh should be a disk"
    assert N_VERTS_CUT - N_VERTS_UNCUT == 52


def test_laplacians_share_pattern_and_wmat_is_the_clamped_one():
    """``@Solver/Solver.m`` clamps negative cotangent weights to 1e-3."""
    g = load_golden()
    L_off = g.L - sp.diags(g.L.diagonal())
    W_off = g.wmat - sp.diags(g.wmat.diagonal())

    n_negative_in_L = int((L_off.data < 0).sum())
    assert n_negative_in_L == 1696
    assert (W_off.data >= 0).all(), "clamped Laplacian must have no negative weights"
    assert int(np.isclose(W_off.data, 1e-3).sum()) == n_negative_in_L


def test_laplacian_rows_sum_to_zero():
    g = load_golden()
    assert np.abs(np.asarray(g.wmat.sum(axis=1)).ravel()).max() < 1e-10
    assert np.abs(np.asarray(g.L.sum(axis=1)).ravel()).max() < 1e-10


def test_edge_weights_align_with_edges():
    g = load_golden()
    edges, w = g.edges, g.edge_weights
    assert edges.shape == (N_EDGES_CUT, 2)
    assert w.shape == (N_EDGES_CUT,)
    assert (edges[:, 0] < edges[:, 1]).all()
    assert (w > 0).all()
    # weights must match the Laplacian entries they were taken from
    dense_check = np.asarray(g.wmat[edges[:, 0], edges[:, 1]]).ravel()
    assert np.allclose(dense_check, w)
