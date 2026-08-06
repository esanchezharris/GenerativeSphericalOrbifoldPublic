"""The public SOTESolver front door: input validation and end-to-end behaviour."""

from __future__ import annotations

import numpy as np
import pytest
import scipy.sparse as sp

from escher.OTE.core.SOTESolver import SOTESolver
from tests.golden import load_golden


@pytest.fixture(scope="module")
def solver():
    g = load_golden()
    return g, SOTESolver(g.edges, g.wmat, g.A, g.b, g.x0)


def test_solves_the_reference_problem(solver):
    g, s = solver
    result = s.solve(g.edge_weights)

    assert result.points.shape == (g.n_verts, 3)
    assert np.allclose(np.linalg.norm(result.points, axis=1), 1.0, atol=1e-12)
    assert result.constraint_violation < 1e-6
    assert result.energy == pytest.approx(3.9368584276, rel=1e-8)


def test_warm_start_cuts_the_iteration_count(solver):
    """The lever that makes per-step solves affordable during SDS training."""
    g, s = solver
    cold = s.solve(g.edge_weights)

    perturbed = g.edge_weights * (1.0 + 1e-3)
    from_scratch = s.solve(perturbed)
    warm = s.solve(perturbed, x0=cold.flat)

    cold_iters = from_scratch.stage1.n_iter + from_scratch.stage2.n_iter
    warm_iters = warm.stage1.n_iter + warm.stage2.n_iter
    assert warm_iters < cold_iters, f"warm {warm_iters} should beat cold {cold_iters}"
    assert warm.energy == pytest.approx(from_scratch.energy, rel=1e-6)


def test_rejects_wrong_weight_count(solver):
    _, s = solver
    with pytest.raises(ValueError, match="one entry per edge"):
        s.solve(np.ones(5))


def test_rejects_non_positive_weights(solver):
    g, s = solver
    bad = g.edge_weights.copy()
    bad[0] = 0.0
    with pytest.raises(ValueError, match="strictly positive"):
        s.solve(bad)


def test_rejects_mismatched_shapes():
    g = load_golden()
    with pytest.raises(ValueError, match="edges must be"):
        SOTESolver(np.zeros((3, 3)), g.wmat, g.A, g.b, g.x0)
    with pytest.raises(ValueError, match="x0 has"):
        SOTESolver(g.edges, sp.eye(10, format="csr"), g.A, g.b, g.x0)


def test_module_no_longer_exposes_the_euclidean_energy():
    """Regression guard: the old implementation minimised x'(W kron I3)x.

    If that ever comes back the tangency tests in test_karcher.py would catch the maths,
    but this catches the API surface directly.
    """
    import escher.OTE.core.SOTESolver as module

    source = module.__doc__ or ""
    assert "Karcher" in source
    assert not hasattr(module.SOTESolver, "custom_opt_loop")
    assert not hasattr(module.SOTESolver, "is_laplacian")
