"""Validate implicit differentiation through the spherical solve.

The gate is a finite-difference check on the weights: perturb ``w``, re-solve, and confirm
the resulting change in a scalar loss matches what the implicit VJP predicted. That is an
oracle independent of the derivation, which matters because a wrong adjoint would still
produce plausible-looking gradients.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.differentiable import SphericalEmbedder, assemble_hessian
from escher.OTE.core.spherical.karcher import (
    karcher_energy,
    weight_jacobian_vjp,
)
from escher.OTE.tilings_sphere import DihedralOrbifold


@pytest.fixture(scope="module")
def small_problem():
    """Deliberately coarse: finite differencing needs many re-solves."""
    orb = DihedralOrbifold.from_resolution(k=4, n_theta=6, n_phi=5)
    return orb


def make_embedder(orb, **kw):
    return SphericalEmbedder(orb.mesh.edges, orb.A, orb.b, orb.initial_guess(), **kw)


# ------------------------------------------------------------------------------ Hessian
def test_assembled_hessian_matches_autograd(small_problem):
    orb = small_problem
    edges = torch.as_tensor(orb.mesh.edges, dtype=torch.long)
    pts = torch.as_tensor(orb.mesh.points, dtype=torch.float64)
    w = torch.rand(len(orb.mesh.edges), dtype=torch.float64) + 0.5

    H = assemble_hessian(pts, edges, w).toarray()

    def energy_flat(x_flat):
        return karcher_energy(x_flat.reshape(-1, 3), edges, w)

    expected = torch.autograd.functional.hessian(
        energy_flat, pts.reshape(-1).clone(), vectorize=True
    ).numpy()

    assert np.abs(H - expected).max() < 1e-8, f"max diff {np.abs(H - expected).max():.3e}"


def test_hessian_is_symmetric_and_sparse(small_problem):
    orb = small_problem
    edges = torch.as_tensor(orb.mesh.edges, dtype=torch.long)
    pts = torch.as_tensor(orb.mesh.points, dtype=torch.float64)
    w = torch.ones(len(orb.mesh.edges), dtype=torch.float64)

    H = assemble_hessian(pts, edges, w)
    assert np.abs((H - H.T).data).max() < 1e-10 if (H - H.T).nnz else True
    density = H.nnz / (H.shape[0] * H.shape[1])
    assert density < 0.2, f"Hessian should be sparse, density {density:.3f}"


# -------------------------------------------------------------------------- weight VJP
def test_weight_jacobian_vjp_matches_autograd(small_problem):
    r"""``B^T y`` must equal the autograd VJP of ``dE/dx`` contracted with ``y``."""
    orb = small_problem
    edges = torch.as_tensor(orb.mesh.edges, dtype=torch.long)
    pts = torch.as_tensor(orb.mesh.points, dtype=torch.float64)
    rng = np.random.default_rng(0)
    y = torch.as_tensor(rng.standard_normal(pts.shape), dtype=torch.float64)

    got = weight_jacobian_vjp(pts, edges, y)

    w = torch.ones(len(orb.mesh.edges), dtype=torch.float64, requires_grad=True)
    x = pts.clone().requires_grad_(True)
    grad_x = torch.autograd.grad(karcher_energy(x, edges, w), x, create_graph=True)[0]
    expected = torch.autograd.grad((grad_x * y).sum(), w)[0]

    assert torch.allclose(got, expected, rtol=1e-9, atol=1e-12)


# -------------------------------------------------------------------- the actual gate
def test_implicit_gradient_matches_finite_differences(small_problem):
    """**The Phase 2.1 gate.** Perturb w, re-solve, compare against the predicted change.

    Two settings matter and are easy to get backwards. The forward solve is *iterative*, so
    its own residual error enters every function evaluation; dividing that by ``h`` amplifies
    it, and a *smaller* ``h`` makes the finite difference worse, not better. So this uses a
    tight solver tolerance and a comparatively large ``h = 1e-4``. At ``tol_x = 1e-11`` with
    ``h = 1e-6`` the same (correct) gradient appears to disagree by ~2%, which is entirely
    differencing noise.
    """
    orb = small_problem
    embedder = make_embedder(orb, warm_start=False, tol_x=1e-14)
    n_edges = len(orb.mesh.edges)

    rng = np.random.default_rng(1)
    w0 = torch.as_tensor(
        10.0 ** rng.uniform(-0.3, 0.3, size=n_edges), dtype=torch.float64
    )
    # a fixed random linear functional of the vertices, so the test is not accidentally
    # measuring something the geometry makes trivially zero
    target = torch.as_tensor(rng.standard_normal((orb.mesh.n_verts, 3)), dtype=torch.float64)

    def loss_of(w: torch.Tensor) -> torch.Tensor:
        return (embedder(w) * target).sum()

    w = w0.clone().requires_grad_(True)
    loss_of(w).backward()
    analytic = w.grad.clone()

    assert torch.isfinite(analytic).all()
    assert analytic.abs().max() > 1e-8, "gradient is trivially zero; test proves nothing"

    # Central differences on a random subset of edges. `h` has a sweet spot: too small and
    # the solver's residual error dominates, too large and O(h^2) truncation does. Assert a
    # loose per-edge bound plus a much tighter median, which is the robust aggregate signal
    # -- a structurally wrong adjoint is off by factors, not by fractions of a percent.
    h = 1e-4
    idx = rng.choice(n_edges, size=25, replace=False)
    rel_errors = []
    for e in idx:
        plus, minus = w0.clone(), w0.clone()
        plus[e] += h
        minus[e] -= h
        fd = (loss_of(plus) - loss_of(minus)).item() / (2 * h)
        a = analytic[e].item()
        rel_errors.append(abs(fd - a) / max(abs(a), 1e-12))
        assert fd == pytest.approx(a, rel=1e-3, abs=1e-9), (
            f"edge {e}: finite difference {fd:.10e} vs implicit {a:.10e}"
        )

    median_rel = float(np.median(rel_errors))
    assert median_rel < 1e-5, f"median relative error {median_rel:.3e} is too large"


def test_gradient_is_insensitive_to_forward_solver_tolerance(small_problem):
    """The adjoint linearises about the converged point, so a tighter forward solve should
    not move it. Guards against the gradient quietly depending on where L-BFGS stopped."""
    orb = small_problem
    rng = np.random.default_rng(1)
    w0 = torch.as_tensor(
        10.0 ** rng.uniform(-0.3, 0.3, size=len(orb.mesh.edges)), dtype=torch.float64
    )
    target = torch.as_tensor(rng.standard_normal((orb.mesh.n_verts, 3)), dtype=torch.float64)

    grads = []
    for tol in (1e-11, 1e-14):
        embedder = make_embedder(orb, warm_start=False, tol_x=tol)
        w = w0.clone().requires_grad_(True)
        (embedder(w) * target).sum().backward()
        grads.append(w.grad.clone())

    assert torch.allclose(grads[0], grads[1], rtol=1e-6, atol=1e-12)


def test_gradient_flows_through_a_downstream_loss(small_problem):
    """A realistic shape: the loss depends on the embedding, not on w directly."""
    orb = small_problem
    embedder = make_embedder(orb, warm_start=False)
    w = torch.full((len(orb.mesh.edges),), 1.0, dtype=torch.float64, requires_grad=True)

    points = embedder(w)
    # push the tile's centroid toward the pole
    loss = -points.mean(dim=0)[2]
    loss.backward()

    assert w.grad is not None
    assert torch.isfinite(w.grad).all()
    assert w.grad.abs().sum() > 0


def test_forward_output_is_on_the_sphere_and_feasible(small_problem):
    orb = small_problem
    embedder = make_embedder(orb)
    w = torch.ones(len(orb.mesh.edges), dtype=torch.float64)
    points = embedder(w)

    assert points.shape == (orb.mesh.n_verts, 3)
    assert torch.allclose(points.norm(dim=1), torch.ones(orb.mesh.n_verts, dtype=torch.float64))
    assert embedder.last_result.constraint_violation < 1e-6
    assert embedder.last_stationarity < 1e-7


def test_rejects_wrong_weight_shape(small_problem):
    embedder = make_embedder(small_problem)
    with pytest.raises(ValueError, match="expected"):
        embedder(torch.ones(3, dtype=torch.float64))


# ------------------------------------------------------------------------- warm start
def test_warm_start_reduces_total_iterations(small_problem):
    """The throughput lever for SDS: consecutive solves with nearby weights."""
    orb = small_problem
    rng = np.random.default_rng(2)
    base = torch.as_tensor(
        10.0 ** rng.uniform(-0.3, 0.3, size=len(orb.mesh.edges)), dtype=torch.float64
    )

    cold = make_embedder(orb, warm_start=False)
    warm = make_embedder(orb, warm_start=True)

    for step in range(6):
        nudged = base * (1.0 + 0.002 * step)
        cold(nudged)
        warm(nudged)

    assert warm.total_iterations < cold.total_iterations, (
        f"warm {warm.total_iterations} vs cold {cold.total_iterations}"
    )


def test_warm_and_cold_reach_the_same_solution(small_problem):
    orb = small_problem
    w = torch.ones(len(orb.mesh.edges), dtype=torch.float64)

    cold = make_embedder(orb, warm_start=False)
    warm = make_embedder(orb, warm_start=True)
    warm(w * 1.05)  # prime the cache with a different problem

    assert torch.allclose(cold(w), warm(w), atol=1e-7)


def test_reset_warm_start_clears_the_cache(small_problem):
    orb = small_problem
    embedder = make_embedder(orb)
    embedder(torch.ones(len(orb.mesh.edges), dtype=torch.float64))
    assert embedder._last_x is not None
    embedder.reset_warm_start()
    assert embedder._last_x is None
