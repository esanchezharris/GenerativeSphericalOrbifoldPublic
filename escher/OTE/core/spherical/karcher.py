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
    "karcher_edge_gradients",
    "karcher_edge_hessians",
    "karcher_energy",
    "karcher_energy_and_grad",
    "radial_component",
    "weight_jacobian_vjp",
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


def karcher_edge_gradients(
    points: Tensor, edges: Tensor, weights: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    r"""Per-edge gradient contributions, *before* they are scattered onto vertices.

    Returns ``(grad_u, grad_v, distances)`` with the gradient arrays shaped
    ``(n_edges, 3)``. Kept separate from :func:`karcher_energy_and_grad` because the
    implicit-differentiation path needs the contributions edge by edge rather than summed.

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
    return scale * _dd(u, v, v_sq), scale * _dd(v, u, u_sq), d


def karcher_energy_and_grad(points: Tensor, edges: Tensor, weights: Tensor) -> KarcherResult:
    r"""Closed-form energy and gradient, equivalent to :func:`karcher_energy` + autograd.

    With :math:`c = \lVert u \times v \rVert`, :math:`s = u \cdot v`, :math:`d =
    \operatorname{atan2}(c, s)` and :math:`q = c^2 + s^2`:

    .. math::

        \frac{\partial d}{\partial u}
          = \frac{1}{q}\left( \frac{s\,(\lVert v \rVert^2 u - s v)}{c} - c\,v \right)

    and :math:`\partial E/\partial u = 2 w d \cdot \partial d/\partial u`, with the
    :math:`u \leftrightarrow v` swap giving the other endpoint.

    See :func:`karcher_edge_gradients` for the derivation; this scatters its output onto
    vertices.
    """
    grad_u, grad_v, d = karcher_edge_gradients(points, edges, weights)

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


def weight_jacobian_vjp(points: Tensor, edges: Tensor, y: Tensor) -> Tensor:
    r"""``B^T y`` where :math:`B = \partial^2 E / \partial x \, \partial w`.

    The energy is **linear in the weights** -- :math:`E = \sum_e w_e d_e^2` -- so
    :math:`\partial E / \partial w_e = d_e^2` and the mixed partial is just
    :math:`\partial (d_e^2)/\partial x`, which is the ordinary per-edge gradient evaluated
    at unit weight. That is why the implicit gradient reuses
    :func:`karcher_edge_gradients` instead of needing anything new.

    Args:
        y: ``(n, 3)`` adjoint vector.

    Returns:
        ``(n_edges,)``.
    """
    unit = torch.ones(len(edges), dtype=points.dtype, device=points.device)
    grad_u, grad_v, _ = karcher_edge_gradients(points, edges, unit)
    return (grad_u * y[edges[:, 0]]).sum(-1) + (grad_v * y[edges[:, 1]]).sum(-1)


def _single_edge_energy(uv: Tensor, w: Tensor) -> Tensor:
    """``w * d(u, v)**2`` for one edge, as a function of the stacked endpoints."""
    u, v = uv[:3], uv[3:]
    d = torch.atan2(torch.linalg.cross(u, v, dim=-1).norm(), u @ v)
    return w * d.square()


def karcher_edge_hessians(points: Tensor, edges: Tensor, weights: Tensor) -> Tensor:
    """``(n_edges, 6, 6)`` Hessian of each edge's energy w.r.t. its stacked endpoints.

    Each edge term depends on only its own two vertices, so the full Hessian is block-sparse
    with the mesh's own sparsity pattern and assembling it from these blocks is cheap. Taken
    by autograd rather than by hand -- the closed-form second derivative of ``atan2`` of a
    cross-product norm is long, error-prone, and buys nothing here since this runs once per
    backward pass rather than once per line-search step.
    """
    uv = torch.cat([points[edges[:, 0]], points[edges[:, 1]]], dim=-1)  # (E, 6)
    hess_fn = torch.func.hessian(_single_edge_energy, argnums=0)
    return torch.func.vmap(hess_fn, in_dims=(0, 0))(uv, weights)


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
