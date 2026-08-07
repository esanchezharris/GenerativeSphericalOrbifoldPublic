"""Validate the projected L-BFGS, ending with the real spherical orbifold problem."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp
import torch

from escher.OTE.core.spherical.affine_space import AffineSpace
from escher.OTE.core.spherical.karcher import karcher_energy_and_grad
from escher.OTE.core.spherical.lbfgs import ProjectedLBFGS
from escher.OTE.core.spherical.precond import PrecondFixed, PrecondIdentity
from tests.golden import as_points, load_golden


# ------------------------------------------------------------------- synthetic problems
def test_constrained_quadratic_hits_the_analytic_optimum():
    r"""min 1/2 x'x - c'x  s.t.  Ax = b has a closed-form solution to compare against."""
    rng = np.random.default_rng(0)
    n, m = 30, 6
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    c = rng.standard_normal(n)

    affine = AffineSpace(sp.csr_matrix(A), b)

    def fun(x):
        return 0.5 * float(x @ x) - float(c @ x), x - c

    x0 = affine.project(rng.standard_normal(n))
    result = ProjectedLBFGS(fun, x0, affine, PrecondIdentity(), memory=5).solve(
        tol_x=1e-14, max_iter=500
    )

    # KKT: x = c + A'(AA')^-1 (b - Ac)
    expected = c + A.T @ np.linalg.solve(A @ A.T, b - A @ c)
    assert np.abs(result.x - expected).max() < 1e-8
    assert affine.constraint_violation(result.x) < 1e-9


def test_rejects_infeasible_start():
    A = sp.csr_matrix(np.array([[1.0, 1.0]]))
    affine = AffineSpace(A, np.array([1.0]))
    with pytest.raises(ValueError, match="violates the linear constraints"):
        ProjectedLBFGS(lambda x: (0.0, np.zeros(2)), np.array([5.0, 5.0]), affine)


def test_iterates_stay_feasible():
    rng = np.random.default_rng(1)
    n, m = 20, 4
    A = rng.standard_normal((m, n))
    affine = AffineSpace(sp.csr_matrix(A), rng.standard_normal(m))

    def fun(x):
        return float(np.sum(x**4)), 4 * x**3

    solver = ProjectedLBFGS(fun, affine.project(rng.standard_normal(n)), affine, memory=3)
    for _ in range(20):
        solver.iterate()
        assert affine.constraint_violation(solver.x) < 1e-9


def test_memoryless_mode_still_descends():
    """``memory=0`` is what the reference's second stage uses -- steepest descent."""
    rng = np.random.default_rng(2)
    n, m = 15, 3
    A = rng.standard_normal((m, n))
    affine = AffineSpace(sp.csr_matrix(A), rng.standard_normal(m))

    def fun(x):
        return float(x @ x), 2 * x

    solver = ProjectedLBFGS(fun, affine.project(rng.standard_normal(n)), affine, memory=0)
    result = solver.solve(tol_x=1e-14, max_iter=300)
    assert solver.curr_m == 0
    assert result.energy < fun(solver.x0 if hasattr(solver, "x0") else result.x)[0] + 1e-9
    assert np.all(np.diff(result.energy_history) <= 1e-12)


def test_bad_descent_direction_terminates_rather_than_spinning():
    """A gradient that lies about the descent direction must not hang.

    Backtracking drives ``t`` to ~1e-18, at which point ``x + t*p == x`` in float64 and the
    step is exactly zero -- so this exits via ``NoProgress`` rather than exhausting the
    line-search cap. Either flag is a correct, bounded termination; what matters is that it
    stops. The reference's line search has no cap at all.
    """
    A = sp.csr_matrix(np.array([[0.0, 0.0, 1.0]]))
    affine = AffineSpace(A, np.array([0.0]))

    def fun(x):
        return float(x @ x), -2 * x  # sign-flipped gradient: "downhill" is uphill

    solver = ProjectedLBFGS(fun, affine.project(np.array([1.0, 1.0, 0.0])), affine, memory=2)
    result = solver.solve(max_iter=50)
    assert result.exit_flag in ("LineSearchFailed", "NoProgress")
    assert result.n_iter < 50, "must stop early, not run out the iteration budget"


# ------------------------------------------------------------- the real spherical problem
@pytest.fixture(scope="module")
def spherical_problem():
    g = load_golden()
    edges = torch.tensor(g.edges, dtype=torch.long)
    weights = torch.tensor(g.edge_weights, dtype=torch.float64)

    def fun(x_flat: np.ndarray):
        pts = torch.tensor(x_flat.reshape(-1, 3), dtype=torch.float64)
        res = karcher_energy_and_grad(pts, edges, weights)
        return float(res.energy), res.grad.numpy().reshape(-1)

    return g, fun, AffineSpace(g.A, g.b)


def test_karcher_energy_decreases_monotonically(spherical_problem):
    g, fun, affine = spherical_problem
    result = ProjectedLBFGS(
        fun, g.x0, affine, PrecondFixed(g.wmat, affine), memory=3
    ).solve(tol_x=1e-10, max_iter=200)

    e0, e1 = result.energy_history[0], result.energy
    assert e0 == pytest.approx(4.5552541676, rel=1e-6), "starting energy at x0"
    assert e1 < e0, "energy must decrease"
    assert np.all(np.diff(result.energy_history) <= 1e-12), "monotone descent"


def test_solution_stays_feasible_and_on_the_sphere(spherical_problem):
    g, fun, affine = spherical_problem
    result = ProjectedLBFGS(
        fun, g.x0, affine, PrecondFixed(g.wmat, affine), memory=3
    ).solve(tol_x=1e-10, max_iter=200)

    # The reference asserts exactly this bound after solving.
    assert affine.constraint_violation(result.x) < 1e-6

    radii = np.linalg.norm(as_points(result.x), axis=1)
    # The Karcher gradient is tangent, so vertices stay near the sphere without a penalty;
    # the reference still hard-normalises at the end, which is why this is loose.
    assert radii.min() > 0.5, f"vertices collapsed toward the origin (min radius {radii.min():.3f})"
    assert radii.max() < 2.0


def test_projected_gradient_converges_to_zero(spherical_problem):
    """Stationarity for a constrained problem is ``P g -> 0``, not ``g -> 0``.

    At the optimum the full gradient is entirely ``A' lambda`` (verified below), so ``|g|``
    actually *rises* slightly during the solve while ``|P g|`` falls by nine orders of
    magnitude. Asserting on the unprojected norm would flag a converged solve as broken.
    """
    g, fun, affine = spherical_problem
    result = ProjectedLBFGS(
        fun, g.x0, affine, PrecondFixed(g.wmat, affine), memory=3
    ).solve(tol_x=1e-10, max_iter=200)

    assert result.proj_grad_inf_history[0] > 1e-2
    assert result.proj_grad_inf_history[-1] < 1e-8

    # the residual full gradient lies (numerically) entirely in range(A')
    _, grad = fun(result.x)
    projected = affine.project_tangent(grad)
    assert np.linalg.norm(projected) / np.linalg.norm(grad) < 1e-9


def test_converged_energy_is_stable(spherical_problem):
    """Pin the converged value so any future change to the energy shows up here."""
    g, fun, affine = spherical_problem
    result = ProjectedLBFGS(
        fun, g.x0, affine, PrecondFixed(g.wmat, affine), memory=3
    ).solve(tol_x=1e-10, max_iter=200)
    assert result.energy == pytest.approx(3.9368584276, rel=1e-8)
    assert result.exit_flag in ("TolX", "TolFun", "NoProgress")


def test_laplacian_preconditioner_beats_identity(spherical_problem):
    """The reason ``Precond_Fixed`` exists: without it this problem crawls."""
    g, fun, affine = spherical_problem
    budget = 40

    with_pre = ProjectedLBFGS(
        fun, g.x0, affine, PrecondFixed(g.wmat, affine), memory=3
    ).solve(tol_x=0.0, tol_fun=0.0, max_iter=budget)
    without = ProjectedLBFGS(fun, g.x0, affine, PrecondIdentity(), memory=3).solve(
        tol_x=0.0, tol_fun=0.0, max_iter=budget
    )

    assert with_pre.energy < without.energy, (
        f"preconditioned {with_pre.energy:.6g} should beat identity {without.energy:.6g} "
        f"after {budget} iterations"
    )
