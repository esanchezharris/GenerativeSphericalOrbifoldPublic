r"""Differentiating through the spherical embedding, w.r.t. the edge weights.

The planar :class:`~escher.OTE.core.OTESolver.OTESolver` is differentiable for free: it is a
single linear solve, so autograd walks straight through it. The spherical solve is an
iterative nonlinear minimisation, and unrolling 30-plus L-BFGS iterations -- each with its
own line search and a sparse LU apply -- is neither cheap nor numerically pleasant.

Instead, differentiate the *optimality condition*. The solve returns

.. math:: x^*(w) = \arg\min_{Ax = b} E(x, w)

whose stationarity condition on the constraint null space is :math:`P \nabla_x E = 0` for the
orthogonal projector :math:`P` onto :math:`\ker A`. Differentiating that in :math:`w` and
solving for :math:`\partial x/\partial w` gives, for an upstream gradient
:math:`v = \partial \mathcal{L}/\partial x`,

.. math:: \frac{\partial \mathcal{L}}{\partial w} = -B^\top y,
   \qquad \text{where } y \text{ solves }
   \begin{bmatrix} H & A^\top \\ A & 0 \end{bmatrix}
   \begin{bmatrix} y \\ \mu \end{bmatrix} =
   \begin{bmatrix} v \\ 0 \end{bmatrix}

with :math:`H = \partial^2 E/\partial x^2` and :math:`B = \partial^2 E/\partial x \partial w`.
The KKT block form is the same trick used in :mod:`.precond` and in the planar solver: the
second row forces :math:`y \in \ker A`, and projecting the first then gives :math:`PHy = Pv`.

So one backward pass costs **one sparse solve**, independent of how many iterations the
forward solve took. Two things make it cheap:

- :math:`H` is block-sparse with the mesh's own pattern, since each energy term touches only
  its edge's two endpoints (:func:`~.karcher.karcher_edge_hessians`);
- :math:`E` is **linear in** :math:`w`, so :math:`B` needs no new machinery -- it is the
  per-edge gradient at unit weight (:func:`~.karcher.weight_jacobian_vjp`).

.. rubric:: The radial gauge

There is a subtlety that will silently produce nonsense if ignored. The Karcher energy
measures *angles*, so it is invariant under scaling each vertex independently:
:math:`E(\alpha_i x_i) = E(x)` for any :math:`\alpha_i > 0`. This is the same property that
makes the gradient tangent to the sphere -- and it means the stationary point is not
isolated. :math:`H` annihilates every radial direction, so it has an :math:`n`-dimensional
null space, most of which lies inside :math:`\ker A`. The KKT matrix above is then singular,
and a sparse LU will happily return garbage rather than raise.

The fix is to pin the gauge by adding the linearised unit-sphere conditions
:math:`x_i^\top \delta x_i = 0` as a third block :math:`C`:

.. math::
   \begin{bmatrix} H & A^\top & C^\top \\ A & 0 & 0 \\ C & 0 & 0 \end{bmatrix}
   \begin{bmatrix} y \\ \mu \\ \nu \end{bmatrix} =
   \begin{bmatrix} v \\ 0 \\ 0 \end{bmatrix}

Their multipliers are zero at the solution -- the sphere condition is gauge fixing, not an
active constraint, precisely because :math:`\nabla E` is already tangent -- so the Hessian
of the Lagrangian is still just :math:`H`, and no extra terms appear.

:math:`[A; C]` is itself rank-deficient, and necessarily so: a pinned cone point's gauge row
is a combination of its three position rows, and for a rotation pair :math:`u_j = R u_i` the
gauge row at :math:`j` follows from the one at :math:`i` because rotations preserve norms.
The system stays *consistent* and :math:`y` stays unique -- only the multipliers are
undetermined -- so the blocks carry a small :math:`-\varepsilon I` regularisation, the
standard treatment for a saddle-point system with redundant constraints. The residual of
every solve is checked afterwards regardless, because a sparse LU will return a plausible
vector for a singular system rather than fail.

This also lets the adjoint be taken at the *normalised* output rather than at the raw
iterate. Normalisation commutes with the orbifold constraints (rotations preserve norms, and
the pinned cone points already lie on the sphere), so the normalised points satisfy
:math:`Ax = b` too and are themselves stationary for the constrained problem. That removes
any need for a separate normalisation Jacobian.

One caveat worth stating plainly: this is exact only at a converged solution. If the forward
solve stops early the adjoint is solving for a stationary point the forward pass never
reached. :attr:`SphericalEmbedder.last_stationarity` exposes the achieved
:math:`\lVert P g \rVert_\infty` so callers can check.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch import Tensor

from .karcher import karcher_edge_hessians, weight_jacobian_vjp
from .solver import SphericalEmbeddingResult, laplacian_from_edges, solve_spherical_embedding

__all__ = [
    "BoundaryEmbedder",
    "SphericalEmbedder",
    "assemble_hessian",
    "embed_differentiable",
    "sphere_gauge_matrix",
]

#: Relative residual above which the adjoint solve is considered to have failed.
_ADJOINT_RESIDUAL_TOL = 1e-6

#: Saddle-point regularisation for the redundant constraint rows, relative to ``max|H|``.
_MULTIPLIER_REGULARISATION = 1e-12


def sphere_gauge_matrix(points: np.ndarray) -> sp.csr_matrix:
    r"""``(n, 3n)`` matrix whose row ``i`` is :math:`x_i^\top` in vertex ``i``'s block.

    Encodes the linearised unit-sphere conditions :math:`x_i^\top \delta x_i = 0`, which pin
    the radial gauge that the Karcher energy leaves free. Without these the adjoint system is
    singular.
    """
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    n = len(points)
    rows = np.repeat(np.arange(n), 3)
    cols = np.arange(3 * n)
    return sp.csr_matrix((points.ravel(), (rows, cols)), shape=(n, 3 * n))


def assemble_hessian(
    points: Tensor, edges: Tensor, weights: Tensor
) -> sp.csr_matrix:
    """Scatter the per-edge 6x6 blocks into a sparse ``(3n, 3n)`` Hessian."""
    blocks = karcher_edge_hessians(points, edges, weights).detach().cpu().numpy()
    edges_np = edges.detach().cpu().numpy()
    n = points.shape[0]

    # global indices of the 6 variables each block touches: [3i, 3i+1, 3i+2, 3j, 3j+1, 3j+2]
    idx = np.concatenate(
        [3 * edges_np[:, [0]] + np.arange(3), 3 * edges_np[:, [1]] + np.arange(3)], axis=1
    )  # (E, 6)
    rows = np.repeat(idx, 6, axis=1).ravel()
    cols = np.tile(idx, (1, 6)).ravel()
    return sp.csr_matrix(
        (blocks.ravel(), (rows, cols)), shape=(3 * n, 3 * n)
    )


class _EmbedFunction(torch.autograd.Function):
    """Forward: solve. Backward: implicit VJP via the KKT adjoint system."""

    @staticmethod
    def forward(ctx, weights: Tensor, embedder: "SphericalEmbedder"):
        w = weights.detach().cpu().double().numpy()
        result = embedder._run_solve(w)

        points = torch.as_tensor(
            result.points, dtype=weights.dtype, device=weights.device
        )
        ctx.embedder = embedder
        ctx.save_for_backward(weights, points)
        return points

    @staticmethod
    def backward(ctx, grad_points: Tensor):
        weights, points = ctx.saved_tensors
        embedder: SphericalEmbedder = ctx.embedder

        pts64 = points.detach().double().cpu()
        w64 = weights.detach().double().cpu()
        edges = embedder.edges_t

        H = assemble_hessian(pts64, edges, w64)
        A = embedder.A
        C = sphere_gauge_matrix(pts64.numpy())  # pins the radial gauge; see module docstring
        m, g = A.shape[0], C.shape[0]
        n_vars = 3 * pts64.shape[0]

        # [A; C] is rank-deficient by construction (see module docstring), so regularise the
        # multiplier blocks. Scaled off H so it stays negligible relative to the problem.
        eps = _MULTIPLIER_REGULARISATION * max(abs(H).max(), 1.0)
        zero = lambda r, c: sp.csr_matrix((r, c))  # noqa: E731
        kkt = sp.bmat(
            [
                [H, A.T, C.T],
                [A, -eps * sp.eye(m), zero(m, g)],
                [C, zero(g, m), -eps * sp.eye(g)],
            ],
            format="csc",
        )
        rhs = np.zeros(n_vars + m + g)
        rhs[:n_vars] = grad_points.detach().double().cpu().numpy().reshape(-1)

        try:
            sol = spla.splu(kkt).solve(rhs)
        except RuntimeError as exc:
            raise RuntimeError(
                "the adjoint KKT system could not be factorised. This usually means the "
                "forward solve did not reach a genuine minimum, leaving the Hessian "
                f"indefinite on the constraint null space (stationarity "
                f"{embedder.last_stationarity})."
            ) from exc

        # A sparse LU can succeed on a near-singular system and return nonsense, which is
        # exactly what happens if the gauge block is omitted. Verify rather than trust.
        residual = np.linalg.norm(kkt @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300)
        if not np.isfinite(residual) or residual > _ADJOINT_RESIDUAL_TOL:
            raise RuntimeError(
                f"adjoint solve did not converge (relative residual {residual:.3e}); "
                "the implicit gradient would be unreliable"
            )

        y = torch.as_tensor(sol[:n_vars].reshape(-1, 3), dtype=torch.float64)
        grad_w = -weight_jacobian_vjp(pts64, edges, y)
        return grad_w.to(dtype=weights.dtype, device=weights.device), None


class SphericalEmbedder:
    """A differentiable spherical orbifold embedding, as a function of the edge weights.

    Holds the fixed parts of the problem -- mesh connectivity and orbifold constraints -- and
    turns a weight vector into vertex positions that autograd can backpropagate through.

    Args:
        edges: ``(n_edges, 2)`` vertex index pairs.
        A, b: orbifold boundary equations.
        x0: ``(3n,)`` feasible starting point.
        warm_start: reuse the previous solution to seed the next solve. During SDS the
            weights barely move between steps, so this is the main lever on throughput.

    Example:
        >>> embedder = SphericalEmbedder(edges, A, b, x0)      # doctest: +SKIP
        >>> w = torch.rand(len(edges), requires_grad=True)     # doctest: +SKIP
        >>> points = embedder(w)                               # doctest: +SKIP
        >>> points.sum().backward()                            # doctest: +SKIP
    """

    def __init__(
        self,
        edges: np.ndarray,
        A: sp.spmatrix,
        b: np.ndarray,
        x0: np.ndarray,
        *,
        warm_start: bool = True,
        tol_x: float = 1e-11,
        max_iter: int = 10_000,
    ):
        self.edges = np.asarray(edges)
        self.edges_t = torch.as_tensor(self.edges, dtype=torch.long)
        self.A = sp.csr_matrix(A)
        self.b = np.asarray(b, dtype=np.float64).ravel()
        self.x0 = np.asarray(x0, dtype=np.float64).ravel()
        self.n_verts = self.x0.size // 3
        self.warm_start = warm_start
        self.tol_x = tol_x
        self.max_iter = max_iter

        self._last_x: np.ndarray | None = None
        self.last_result: SphericalEmbeddingResult | None = None
        self.n_solves = 0
        self.total_iterations = 0

    @property
    def last_stationarity(self) -> float | None:
        r"""``max|P g|`` reached by the most recent forward solve.

        The implicit gradient assumes this is ~0. If it is not, the backward pass is
        linearising about a point that is not actually stationary.
        """
        if self.last_result is None:
            return None
        return self.last_result.stage2.proj_grad_inf_history[-1]

    def _run_solve(self, weights: np.ndarray) -> SphericalEmbeddingResult:
        start = self._last_x if (self.warm_start and self._last_x is not None) else self.x0
        result = solve_spherical_embedding(
            edges=self.edges,
            weights=weights,
            laplacian=laplacian_from_edges(self.edges, weights, self.n_verts),
            A=self.A,
            b=self.b,
            x0=start,
            tol_x=self.tol_x,
            max_iter=self.max_iter,
        )
        # Warm-start from the pre-normalisation iterate: it is the one that actually
        # satisfies Ax = b to solver tolerance, and the ProjectedLBFGS constructor rejects
        # an infeasible start.
        self._last_x = result.stage2.x.copy()
        self.last_result = result
        self.n_solves += 1
        self.total_iterations += result.stage1.n_iter + result.stage2.n_iter
        return result

    def __call__(self, weights: Tensor) -> Tensor:
        """``(n_edges,)`` weights -> ``(n, 3)`` unit-sphere vertices, differentiably."""
        if weights.ndim != 1 or weights.shape[0] != len(self.edges):
            raise ValueError(
                f"expected {len(self.edges)} weights, got shape {tuple(weights.shape)}"
            )
        return _EmbedFunction.apply(weights, self)

    def reset_warm_start(self) -> None:
        """Forget the cached solution, so the next solve restarts from ``x0``."""
        self._last_x = None


def embed_differentiable(
    weights: Tensor, edges: np.ndarray, A: sp.spmatrix, b: np.ndarray, x0: np.ndarray
) -> Tensor:
    """One-shot convenience wrapper around :class:`SphericalEmbedder`."""
    return SphericalEmbedder(edges, A, b, x0, warm_start=False)(weights)


class _BoundaryEmbedFunction(torch.autograd.Function):
    r"""Forward: fixed-boundary solve. Backward: :math:`\partial L/\partial b` via the
    adjoint's constraint multipliers.

    Differentiating the optimality system (:math:`\nabla E + A^\top\lambda = 0`,
    :math:`Ax = b`) in :math:`b` and contracting with the upstream gradient :math:`v`
    through the same regularized KKT adjoint used for the weight path gives

    .. math:: \frac{\partial L}{\partial b} = \mu + O(\varepsilon),

    where :math:`\mu` is the :math:`A`-block of the adjoint solution -- the multipliers we
    were already computing and discarding. Validated against finite differences in
    ``tests/test_boundary_explicit.py``.
    """

    @staticmethod
    def forward(ctx, b: Tensor, embedder: "BoundaryEmbedder"):
        result = embedder._run_solve(b.detach().cpu().double().numpy())
        points = torch.as_tensor(result.points, dtype=b.dtype, device=b.device)
        ctx.embedder = embedder
        ctx.save_for_backward(points)
        return points

    @staticmethod
    def backward(ctx, grad_points: Tensor):
        (points,) = ctx.saved_tensors
        embedder: BoundaryEmbedder = ctx.embedder
        pts64 = points.detach().double().cpu()

        H = assemble_hessian(pts64, embedder.edges_t, embedder.weights_t)
        A = embedder.A
        C = sphere_gauge_matrix(pts64.numpy())
        m, g = A.shape[0], C.shape[0]
        n_vars = 3 * pts64.shape[0]

        eps = _MULTIPLIER_REGULARISATION * max(abs(H).max(), 1.0)
        zero = lambda r, c: sp.csr_matrix((r, c))  # noqa: E731
        kkt = sp.bmat(
            [
                [H, A.T, C.T],
                [A, -eps * sp.eye(m), zero(m, g)],
                [C, zero(g, m), -eps * sp.eye(g)],
            ],
            format="csc",
        )
        rhs = np.zeros(n_vars + m + g)
        rhs[:n_vars] = grad_points.detach().double().cpu().numpy().reshape(-1)
        sol = spla.splu(kkt).solve(rhs)

        residual = np.linalg.norm(kkt @ sol - rhs) / max(np.linalg.norm(rhs), 1e-300)
        if not np.isfinite(residual) or residual > _ADJOINT_RESIDUAL_TOL:
            raise RuntimeError(
                f"boundary adjoint solve did not converge (relative residual {residual:.3e})"
            )

        mu = sol[n_vars : n_vars + m]
        grad_b = torch.as_tensor(mu, dtype=torch.float64)
        return grad_b.to(dtype=grad_points.dtype, device=grad_points.device), None


class BoundaryEmbedder:
    """Differentiable fixed-boundary spherical embedding: ``b`` in, vertices out.

    The counterpart of :class:`SphericalEmbedder` for the boundary-explicit
    parameterization -- edge weights are held **uniform** and the boundary right-hand side
    is the optimisation variable. Pair with
    :class:`~escher.OTE.tilings_sphere.boundary_explicit.BoundaryExplicitDihedral`, whose
    ``boundary_b`` builds ``b`` from the free half of the cut differentiably (so gradients
    continue from ``b`` through the group maps onto the free points via autograd).

    Warm starting exploits ``A`` being pure pins: any cached solution is made exactly
    feasible for a *new* ``b`` by overwriting its pinned rows -- projection is assignment.
    """

    def __init__(
        self,
        edges: np.ndarray,
        A: sp.spmatrix,
        pin_order: np.ndarray,
        x0_points: np.ndarray,
        *,
        warm_start: bool = True,
        tol_x: float = 1e-11,
        max_iter: int = 10_000,
    ):
        self.edges = np.asarray(edges)
        self.edges_t = torch.as_tensor(self.edges, dtype=torch.long)
        self.A = sp.csr_matrix(A)
        self.pin_order = np.asarray(pin_order)
        self.n_verts = np.asarray(x0_points).reshape(-1, 3).shape[0]
        self._x0_points = np.asarray(x0_points, dtype=np.float64).reshape(-1, 3)
        self.warm_start = warm_start
        self.tol_x = tol_x
        self.max_iter = max_iter

        self.weights = np.ones(len(self.edges))
        self.weights_t = torch.ones(len(self.edges), dtype=torch.float64)
        self._laplacian = laplacian_from_edges(self.edges, self.weights, self.n_verts)

        self._last_x: np.ndarray | None = None
        self.last_result: SphericalEmbeddingResult | None = None
        self.n_solves = 0
        self.total_iterations = 0

    @property
    def last_stationarity(self) -> float | None:
        if self.last_result is None:
            return None
        return self.last_result.stage2.proj_grad_inf_history[-1]

    def _feasible_start(self, b: np.ndarray) -> np.ndarray:
        base = (
            self._last_x.reshape(-1, 3).copy()
            if (self.warm_start and self._last_x is not None)
            else self._x0_points.copy()
        )
        base[self.pin_order] = b.reshape(-1, 3)
        return base.reshape(-1)

    def _run_solve(self, b: np.ndarray) -> SphericalEmbeddingResult:
        result = solve_spherical_embedding(
            edges=self.edges,
            weights=self.weights,
            laplacian=self._laplacian,
            A=self.A,
            b=b,
            x0=self._feasible_start(b),
            tol_x=self.tol_x,
            max_iter=self.max_iter,
        )
        self._last_x = result.stage2.x.copy()
        self.last_result = result
        self.n_solves += 1
        self.total_iterations += result.stage1.n_iter + result.stage2.n_iter
        return result

    def __call__(self, b: Tensor) -> Tensor:
        """``(3 * n_boundary,)`` pin RHS -> ``(n, 3)`` unit-sphere vertices."""
        if b.ndim != 1 or b.shape[0] != self.A.shape[0]:
            raise ValueError(f"expected b of length {self.A.shape[0]}, got {tuple(b.shape)}")
        return _BoundaryEmbedFunction.apply(b, self)

    def reset_warm_start(self) -> None:
        self._last_x = None
