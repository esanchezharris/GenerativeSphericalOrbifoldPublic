r"""Karcher (geodesic) Dirichlet energy on the sphere.

This is the energy the spherical orbifold Tutte embedding minimises:

.. math::

    E(x) = \sum_{(i,j) \in E} w_{ij} \, d(u_i, u_j)^2,
    \qquad d(u, v) = \operatorname{atan2}(\lVert u \times v \rVert,\; u \cdot v)

Port of ``karcher_grad.m`` from the reference MATLAB implementation.

**Why the geodesic distance and not** :math:`\lVert u - v \rVert^2`. The gradient of the
geodesic energy is *tangent to the sphere*: :math:`\langle \partial E/\partial u_i, u_i
\rangle = 0`. That is what keeps vertices on the sphere while the solver runs, and it is why
the Euclidean Dirichlet energy cannot be substituted — minimising that one collapses every
vertex toward the origin. :func:`radial_component` measures the property directly and the
test suite asserts it.

Two entry points, which must agree:

- :func:`karcher_energy` — plain torch ops, differentiable by autograd. The readable one.
- :func:`karcher_energy_and_grad` — closed-form gradient, no autograd graph. The fast one,
  and the basis for the implicit-differentiation path used during SDS training.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import Tensor

__all__ = [
    "KarcherResult",
    "geodesic_distance",
    "karcher_energy",
    "karcher_energy_and_grad",
    "radial_component",
]

# Below this cross-product norm two vertices are treated as coincident and the edge
# contributes nothing. The reference MATLAB zeroes the NaNs that 0/0 produces here; we avoid
# generating them in the first place, which also keeps autograd from seeing a NaN.
_DEGENERATE_EPS = 1e-12


class KarcherResult(NamedTuple):
    """Energy, its gradient, and the per-edge pieces needed downstream."""

    energy: Tensor  #: scalar
    grad: Tensor  #: (n, 3) -- dE/dx
    distances: Tensor  #: (n_edges,) -- geodesic distance per edge
    energy_per_edge: Tensor  #: (n_edges,) -- w_ij * d_ij**2


def _gather_edge_endpoints(points: Tensor, edges: Tensor) -> tuple[Tensor, Tensor]:
    return points[edges[:, 0]], points[edges[:, 1]]


def geodesic_distance(u: Tensor, v: Tensor) -> Tensor:
    """Great-circle angle between rows of ``u`` and ``v``, shape ``(..., 3)`` -> ``(...,)``.

    Uses ``atan2(|u x v|, u . v)`` rather than ``arccos(u . v)``: the latter loses all
    precision as the angle approaches 0 or pi, exactly where a fine mesh puts most of its
    edges.
    """
    cross_norm = torch.linalg.cross(u, v, dim=-1).norm(dim=-1)
    dot = (u * v).sum(dim=-1)
    return torch.atan2(cross_norm, dot)


def karcher_energy(points: Tensor, edges: Tensor, weights: Tensor) -> Tensor:
    """Total Karcher energy. Differentiable by autograd w.r.t. ``points`` and ``weights``.

    Args:
        points: ``(n, 3)`` vertex positions, expected on or near the unit sphere.
        edges: ``(n_edges, 2)`` integer vertex index pairs.
        weights: ``(n_edges,)`` positive per-edge weights.
    """
    u, v = _gather_edge_endpoints(points, edges)
    d = geodesic_distance(u, v)
    return (weights * d.square()).sum()


def karcher_energy_and_grad(points: Tensor, edges: Tensor, weights: Tensor) -> KarcherResult:
    r"""Closed-form energy and gradient, equivalent to :func:`karcher_energy` + autograd.

    With :math:`c = \lVert u \times v \rVert`, :math:`s = u \cdot v`, :math:`d =
    \operatorname{atan2}(c, s)` and :math:`q = c^2 + s^2`:

    .. math::

        \frac{\partial d}{\partial u}
          = \frac{1}{q}\left( \frac{s\,(\lVert v \rVert^2 u - s v)}{c} - c\,v \right)

    and :math:`\partial E/\partial u = 2 w d \cdot \partial d/\partial u`, with the
    :math:`u \leftrightarrow v` swap giving the other endpoint.

    This is algebraically the same expression as ``karcher_grad.m`` but written without the
    three 2x2 minors, which cancel analytically and cost precision when they nearly do.
    """
    u, v = _gather_edge_endpoints(points, edges)

    cross = torch.linalg.cross(u, v, dim=-1)
    c = cross.norm(dim=-1)  # |u x v|
    s = (u * v).sum(dim=-1)  # u . v
    d = torch.atan2(c, s)
    q = c.square() + s.square()

    # Coincident endpoints contribute no energy and no gradient; mask them out before the
    # division rather than cleaning up NaNs afterwards.
    ok = c > _DEGENERATE_EPS
    c_safe = torch.where(ok, c, torch.ones_like(c))
    q_safe = torch.where(ok, q, torch.ones_like(q))

    u_sq = u.square().sum(dim=-1)
    v_sq = v.square().sum(dim=-1)

    def _dd(a: Tensor, b: Tensor, b_sq: Tensor) -> Tensor:
        """d(distance)/d(a), for endpoint ``a`` with opposite endpoint ``b``."""
        term = (s / c_safe).unsqueeze(-1) * (b_sq.unsqueeze(-1) * a - s.unsqueeze(-1) * b)
        return (term - c_safe.unsqueeze(-1) * b) / q_safe.unsqueeze(-1)

    scale = torch.where(ok, 2.0 * weights * d, torch.zeros_like(d)).unsqueeze(-1)
    grad_u = scale * _dd(u, v, v_sq)
    grad_v = scale * _dd(v, u, u_sq)

    grad = torch.zeros_like(points)
    grad.index_add_(0, edges[:, 0], grad_u)
    grad.index_add_(0, edges[:, 1], grad_v)

    energy_per_edge = weights * d.square()
    return KarcherResult(
        energy=energy_per_edge.sum(),
        grad=grad,
        distances=d,
        energy_per_edge=energy_per_edge,
    )


def radial_component(points: Tensor, grad: Tensor) -> Tensor:
    """Per-vertex ``cos`` of the angle between the gradient and the radial direction.

    The defining property of this energy is that its gradient is tangent to the sphere, so
    every entry should be ~0. A value near 1 means the energy is pulling vertices radially
    -- the failure mode of the Euclidean Dirichlet energy.
    """
    gn = grad.norm(dim=-1)
    pn = points.norm(dim=-1)
    denom = torch.clamp(gn * pn, min=1e-30)
    return (grad * points).sum(dim=-1) / denom
