"""Validate the projector-based constraint space against the reference ``N.mat``.

The projector never forms a null-space basis, so the check is that it *acts* like the
reference one: ``project_tangent(v)`` must equal ``N @ (N.T @ v)`` for the golden ``N``.
"""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from escher.OTE.core.spherical.affine_space import AffineSpace
from escher.OTE.core.spherical.precond import PrecondFixed, PrecondIdentity
from tests.golden import load_golden


@pytest.fixture(scope="module")
def golden_affine():
    g = load_golden()
    return g, AffineSpace(g.A, g.b)


# ------------------------------------------------------------------------ basic structure
def test_dimensions_match_reference(golden_affine):
    g, aff = golden_affine
    assert aff.n_constraints == 165
    assert aff.n_vars == 7662
    assert aff.dim_null == g.N.shape[1] == 7497


def test_particular_solution_satisfies_constraints(golden_affine):
    _, aff = golden_affine
    assert aff.constraint_violation(aff.x_p) < 1e-10


def test_rejects_dependent_constraint_rows():
    A = sp.csr_matrix(np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]))
    with pytest.raises(ValueError, match="linearly dependent"):
        AffineSpace(A, np.array([1.0, 2.0]))


# --------------------------------------------------------- equivalence with reference N
def test_tangent_projection_matches_reference_null_space(golden_affine):
    """The whole justification for dropping the explicit basis."""
    g, aff = golden_affine
    rng = np.random.default_rng(0)
    for _ in range(5):
        v = rng.standard_normal(aff.n_vars)
        via_projector = aff.project_tangent(v)
        via_basis = g.N @ (g.N.T @ v)
        assert np.abs(via_projector - via_basis).max() < 1e-10


def test_our_own_basis_agrees_with_the_projector(golden_affine):
    _, aff = golden_affine
    N = aff.null_space_basis()
    assert N.shape == (7662, 7497)
    assert np.abs((aff.A @ N).toarray()).max() < 1e-10

    rng = np.random.default_rng(1)
    v = rng.standard_normal(aff.n_vars)
    assert np.abs(N @ (N.T @ v) - aff.project_tangent(v)).max() < 1e-10


# --------------------------------------------------------------------- projector algebra
def test_projection_lands_in_the_affine_space(golden_affine):
    _, aff = golden_affine
    rng = np.random.default_rng(2)
    x = rng.standard_normal(aff.n_vars)
    assert aff.constraint_violation(aff.project(x)) < 1e-10


def test_projection_is_idempotent(golden_affine):
    _, aff = golden_affine
    rng = np.random.default_rng(3)
    x = rng.standard_normal(aff.n_vars)
    once = aff.project(x)
    assert np.abs(aff.project(once) - once).max() < 1e-10


def test_x0_is_already_in_the_affine_space(golden_affine):
    """The reference x0 is ``Ab.projectOnto(X)``, so projecting again must be a no-op."""
    g, aff = golden_affine
    assert aff.constraint_violation(g.x0) < 1e-10
    assert np.abs(aff.project(g.x0) - g.x0).max() < 1e-10


def test_tangent_vectors_are_annihilated_by_A(golden_affine):
    _, aff = golden_affine
    rng = np.random.default_rng(4)
    v = aff.project_tangent(rng.standard_normal(aff.n_vars))
    assert np.abs(aff.A @ v).max() < 1e-10


def test_projection_is_orthogonal(golden_affine):
    """<v - Pv, Pw> = 0 for tangent projections."""
    _, aff = golden_affine
    rng = np.random.default_rng(5)
    v, w = rng.standard_normal(aff.n_vars), rng.standard_normal(aff.n_vars)
    Pv, Pw = aff.project_tangent(v), aff.project_tangent(w)
    assert abs(float((v - Pv) @ Pw)) < 1e-8


# ------------------------------------------------------------------------ preconditioner
def test_identity_preconditioner_is_a_no_op():
    v = np.arange(5.0)
    assert np.array_equal(PrecondIdentity().apply(v), v)


@pytest.fixture(scope="module")
def golden_precond(golden_affine):
    g, aff = golden_affine
    return g, aff, PrecondFixed(g.wmat, aff)


def test_fixed_preconditioner_inverts_the_reduced_laplacian(golden_precond):
    r"""``apply`` must invert :math:`-P K P` on the tangent space."""
    g, aff, pre = golden_precond
    K = sp.kron(g.wmat, sp.eye(3, format="csr"), format="csr")

    rng = np.random.default_rng(6)
    q = aff.project_tangent(rng.standard_normal(aff.n_vars))
    r = pre.apply(q)

    # r must be tangent, and -P K r must reproduce q
    assert np.abs(aff.A @ r).max() < 1e-8
    round_trip = aff.project_tangent(-(K @ r))
    rel = np.linalg.norm(round_trip - q) / np.linalg.norm(q)
    assert rel < 1e-8, f"relative round-trip error {rel:.3e}"


def test_fixed_preconditioner_matches_explicit_reduced_solve(golden_precond):
    """Cross-check against the reference's literal ``-N' kron(W,I3) N`` formulation."""
    g, aff, pre = golden_precond
    N = g.N
    K = sp.kron(g.wmat, sp.eye(3, format="csr"), format="csr")
    reduced = sp.csc_matrix(-(N.T @ K @ N))

    rng = np.random.default_rng(7)
    q = aff.project_tangent(rng.standard_normal(aff.n_vars))
    expected = N @ sp.linalg.spsolve(reduced, N.T @ q)
    got = pre.apply(q)

    rel = np.linalg.norm(got - expected) / np.linalg.norm(expected)
    assert rel < 1e-7, f"relative difference {rel:.3e}"


def test_preconditioner_rejects_mismatched_laplacian(golden_affine):
    _, aff = golden_affine
    with pytest.raises(ValueError, match="variables"):
        PrecondFixed(sp.eye(10, format="csr"), aff)
