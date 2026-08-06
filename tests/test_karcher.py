"""Validate the Karcher energy and its closed-form gradient.

``xgrad.pkl`` is *not* usable as an oracle (see tests/golden/README.md), so the gradient is
checked against two independent references instead: torch autograd, and central finite
differences of the energy. Plus the structural property that actually matters -- tangency.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from escher.OTE.core.spherical.karcher import (
    geodesic_distance,
    karcher_energy,
    karcher_energy_and_grad,
    radial_component,
)
from tests.golden import as_points, load_golden

torch.manual_seed(0)


# --------------------------------------------------------------------------- fixtures
@pytest.fixture(scope="module")
def golden_problem():
    """The reference problem: x0 on the unit sphere with the clamped cotangent weights."""
    g = load_golden()
    points = torch.tensor(as_points(g.x0), dtype=torch.float64)
    edges = torch.tensor(g.edges, dtype=torch.long)
    weights = torch.tensor(g.edge_weights, dtype=torch.float64)
    return points, edges, weights


@pytest.fixture(scope="module")
def random_problem():
    """A small random problem, to exercise points that are *not* a converged solution."""
    n, m = 40, 150
    points = torch.randn(n, 3, dtype=torch.float64)
    points = points / points.norm(dim=-1, keepdim=True)
    edges = torch.stack(
        [torch.randint(0, n, (m,)), torch.randint(0, n, (m,))], dim=1
    )
    edges = edges[edges[:, 0] != edges[:, 1]]
    weights = torch.rand(edges.shape[0], dtype=torch.float64) + 0.1
    return points, edges, weights


# --------------------------------------------------------------------- geodesic distance
def test_geodesic_distance_known_angles():
    e1 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    e2 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    assert torch.allclose(geodesic_distance(e1, e1), torch.zeros(1, dtype=torch.float64))
    assert torch.allclose(geodesic_distance(e1, e2), torch.tensor([math.pi / 2], dtype=torch.float64))
    assert torch.allclose(geodesic_distance(e1, -e1), torch.tensor([math.pi], dtype=torch.float64))


def test_geodesic_distance_beats_arccos_near_antipode():
    """atan2 keeps precision where arccos saturates -- the reason for the formulation."""
    eps = 1e-9
    u = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    v = torch.tensor([[-math.cos(eps), math.sin(eps), 0.0]], dtype=torch.float64)
    d = geodesic_distance(u, v).item()
    assert abs(d - (math.pi - eps)) < 1e-12
    # arccos of the same pair loses ~half the digits
    naive = math.acos(max(-1.0, min(1.0, float((u * v).sum()))))
    assert abs(naive - (math.pi - eps)) > 1e-10


def test_geodesic_distance_is_symmetric(random_problem):
    points, edges, _ = random_problem
    u, v = points[edges[:, 0]], points[edges[:, 1]]
    assert torch.allclose(geodesic_distance(u, v), geodesic_distance(v, u))


# ------------------------------------------------------------------------------- energy
def test_energy_matches_naive_loop(random_problem):
    points, edges, weights = random_problem
    total = 0.0
    for (i, j), w in zip(edges.tolist(), weights.tolist()):
        cosang = float(torch.dot(points[i], points[j]))
        total += w * math.acos(max(-1.0, min(1.0, cosang))) ** 2
    assert karcher_energy(points, edges, weights).item() == pytest.approx(total, rel=1e-9)


def test_energy_is_rotation_invariant(random_problem):
    points, edges, weights = random_problem
    q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.det(q) < 0:
        q[:, 0] *= -1
    before = karcher_energy(points, edges, weights)
    after = karcher_energy(points @ q.T, edges, weights)
    assert torch.allclose(before, after, rtol=1e-12)


def test_both_entry_points_report_the_same_energy(golden_problem):
    points, edges, weights = golden_problem
    a = karcher_energy(points, edges, weights)
    b = karcher_energy_and_grad(points, edges, weights).energy
    assert torch.allclose(a, b, rtol=1e-12)


# ----------------------------------------------------------------------------- gradient
@pytest.mark.parametrize("problem", ["golden_problem", "random_problem"])
def test_analytic_gradient_matches_autograd(problem, request):
    points, edges, weights = request.getfixturevalue(problem)
    p = points.clone().requires_grad_(True)
    karcher_energy(p, edges, weights).backward()

    analytic = karcher_energy_and_grad(points, edges, weights).grad
    assert torch.allclose(analytic, p.grad, rtol=1e-9, atol=1e-12), (
        f"max abs diff {(analytic - p.grad).abs().max():.3e}"
    )


@pytest.mark.parametrize("problem", ["golden_problem", "random_problem"])
def test_analytic_gradient_matches_finite_differences(problem, request):
    """Central differences -- an oracle independent of both torch and the MATLAB port."""
    points, edges, weights = request.getfixturevalue(problem)
    analytic = karcher_energy_and_grad(points, edges, weights).grad

    n = points.shape[0]
    rng = np.random.default_rng(0)
    sample = rng.choice(n, size=min(60, n), replace=False)
    h = 1e-6

    for i in sample:
        for k in range(3):
            plus, minus = points.clone(), points.clone()
            plus[i, k] += h
            minus[i, k] -= h
            fd = (
                karcher_energy(plus, edges, weights) - karcher_energy(minus, edges, weights)
            ) / (2 * h)
            assert fd.item() == pytest.approx(
                analytic[i, k].item(), rel=1e-5, abs=1e-8
            ), f"vertex {i} axis {k}"


def test_gradient_wrt_weights_is_squared_distance(random_problem):
    """E is linear in w, so dE/dw_ij = d_ij**2. Phase 2's implicit gradient relies on this."""
    points, edges, weights = random_problem
    w = weights.clone().requires_grad_(True)
    karcher_energy(points, edges, w).backward()
    res = karcher_energy_and_grad(points, edges, weights)
    assert torch.allclose(w.grad, res.distances.square(), rtol=1e-12)


# ------------------------------------------------------------- the property that matters
def test_gradient_is_tangent_to_the_sphere(golden_problem):
    """The defining property of this energy, and the whole reason it is the right one."""
    points, edges, weights = golden_problem
    grad = karcher_energy_and_grad(points, edges, weights).grad
    radial = radial_component(points, grad)
    assert radial.abs().max() < 1e-9, f"max radial component {radial.abs().max():.3e}"


def test_euclidean_energy_gradient_is_not_tangent(golden_problem):
    """Contrast case: what the old SOTESolver minimised. Documents the actual bug.

    The Euclidean Dirichlet gradient has a large radial component, which is why minimising
    it drags every vertex toward the origin instead of sliding it along the sphere.
    """
    points, edges, weights = golden_problem
    u, v = points[edges[:, 0]], points[edges[:, 1]]
    diff = 2.0 * weights.unsqueeze(-1) * (u - v)
    grad = torch.zeros_like(points)
    grad.index_add_(0, edges[:, 0], diff)
    grad.index_add_(0, edges[:, 1], -diff)

    radial = radial_component(points, grad)
    assert radial.abs().mean() > 0.1, (
        "expected the Euclidean gradient to have a substantial radial component"
    )


# --------------------------------------------------------------------------- degeneracies
def test_coincident_vertices_contribute_nothing():
    points = torch.tensor([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    weights = torch.ones(1, dtype=torch.float64)

    res = karcher_energy_and_grad(points, edges, weights)
    assert res.energy.item() == pytest.approx(0.0)
    assert torch.isfinite(res.grad).all()
    assert torch.allclose(res.grad, torch.zeros_like(res.grad))


def test_no_nans_on_antipodal_vertices():
    points = torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=torch.float64)
    edges = torch.tensor([[0, 1]], dtype=torch.long)
    weights = torch.ones(1, dtype=torch.float64)

    res = karcher_energy_and_grad(points, edges, weights)
    assert res.energy.item() == pytest.approx(math.pi**2)
    assert torch.isfinite(res.grad).all()
