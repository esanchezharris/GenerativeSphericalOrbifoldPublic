r"""The spherical orbifold Tutte embedding solve.

Port of ``@Solver/solve_bfgs_fast.m``. The reference runs L-BFGS twice, and the schedule
matters -- the old ``SOTESolver`` omitted it entirely:

1. minimise with the **Laplacian preconditioner** :math:`-N^\top (W \otimes I_3) N`, which
   is what makes the problem tractable at mesh resolution;
2. **reset**: swap in the identity preconditioner and clear the correction memory, then
   minimise again. This is memoryless projected steepest descent and acts as a polish pass;
3. **hard-normalise** every vertex onto the unit sphere;
4. assert the orbifold constraints still hold to 1e-6.

Because the Karcher gradient is tangent to the sphere, step 3 only has to undo a small
drift -- on the reference problem, radii reach at most ~1.16 before normalisation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch

from .affine_space import AffineSpace
from .karcher import karcher_energy_and_grad
from .lbfgs import LBFGSResult, ProjectedLBFGS
from .precond import PrecondFixed, PrecondIdentity

__all__ = [
    "SphericalEmbeddingResult",
    "laplacian_from_edges",
    "make_karcher_objective",
    "solve_spherical_embedding",
]


def laplacian_from_edges(
    edges: np.ndarray, weights: np.ndarray, n_vertices: int
) -> sp.csr_matrix:
    """Weighted Laplacian in the convention used throughout: ``+w`` off-diagonal,
    ``-sum(w)`` on the diagonal, so the matrix is negative semi-definite.

    Matches ``Solver.Wmat`` in the reference. No negative-weight clamping is applied --
    unlike cotangent weights, the Tutte weights this pipeline optimises are positive by
    construction.
    """
    edges = np.asarray(edges)
    weights = np.asarray(weights, dtype=np.float64).ravel()
    if len(edges) != len(weights):
        raise ValueError(f"got {len(edges)} edges but {len(weights)} weights")

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    cols = np.concatenate([edges[:, 1], edges[:, 0]])
    vals = np.concatenate([weights, weights])
    L = sp.csr_matrix((vals, (rows, cols)), shape=(n_vertices, n_vertices))
    return L - sp.diags(np.asarray(L.sum(axis=1)).ravel())


@dataclass
class SphericalEmbeddingResult:
    """Outcome of :func:`solve_spherical_embedding`."""

    points: np.ndarray  #: ``(n, 3)`` vertices on the unit sphere
    energy: float  #: Karcher energy before the final normalisation
    constraint_violation: float  #: ``max|Ax - b|`` after normalisation
    max_radius_before_normalisation: float
    min_radius_before_normalisation: float
    stage1: LBFGSResult
    stage2: LBFGSResult

    @property
    def flat(self) -> np.ndarray:
        """colStacked ``(3n,)`` view of :attr:`points`."""
        return self.points.reshape(-1)


def make_karcher_objective(edges: np.ndarray, weights: np.ndarray, device: str = "cpu"):
    """Build the ``x -> (energy, gradient)`` callable the L-BFGS expects.

    Args:
        edges: ``(n_edges, 2)`` vertex index pairs.
        weights: ``(n_edges,)`` positive edge weights.
    """
    edge_t = torch.as_tensor(np.asarray(edges), dtype=torch.long, device=device)
    weight_t = torch.as_tensor(np.asarray(weights), dtype=torch.float64, device=device)

    def objective(x_flat: np.ndarray) -> tuple[float, np.ndarray]:
        pts = torch.as_tensor(
            np.asarray(x_flat, dtype=np.float64).reshape(-1, 3), device=device
        )
        result = karcher_energy_and_grad(pts, edge_t, weight_t)
        return float(result.energy), result.grad.cpu().numpy().reshape(-1)

    return objective


def solve_spherical_embedding(
    edges: np.ndarray,
    weights: np.ndarray,
    laplacian: sp.spmatrix,
    A: sp.spmatrix,
    b: np.ndarray,
    x0: np.ndarray,
    *,
    memory: int = 3,
    tol_x: float = 1e-10,
    tol_fun: float = 0.0,
    max_iter: int = 10_000,
    constraint_tol: float = 1e-6,
    device: str = "cpu",
) -> SphericalEmbeddingResult:
    """Run the two-stage solve and return vertices on the unit sphere.

    Args:
        edges: ``(n_edges, 2)`` vertex index pairs of the cut mesh.
        weights: ``(n_edges,)`` positive edge weights, aligned with ``edges``.
        laplacian: ``(n, n)`` weighted Laplacian (``+w`` off-diagonal, ``-sum(w)`` diagonal),
            used to build the preconditioner.
        A, b: the orbifold boundary equations.
        x0: ``(3n,)`` colStacked starting point; must satisfy ``A x0 = b``.

    Raises:
        ValueError: if the normalised result violates the orbifold constraints by more than
            ``constraint_tol``, mirroring the reference's assertion.
    """
    affine = AffineSpace(A, b)
    objective = make_karcher_objective(edges, weights, device=device)

    # Stage 1 -- Laplacian-preconditioned.
    solver = ProjectedLBFGS(
        objective, x0, affine, PrecondFixed(laplacian, affine), memory=memory
    )
    stage1 = solver.solve(tol_x=tol_x, tol_fun=tol_fun, max_iter=max_iter)

    # Stage 2 -- identity preconditioner, memory cleared. ``memory=0`` keeps ``curr_m`` at 0,
    # so this is projected steepest descent, exactly as ``updateMemory(0)`` gives in MATLAB.
    solver.update_precond(PrecondIdentity())
    solver.update_memory(0)
    stage2 = solver.solve(tol_x=tol_x, tol_fun=tol_fun, max_iter=max_iter)

    points = stage2.x.reshape(-1, 3)
    radii = np.linalg.norm(points, axis=1)
    normalised = points / radii[:, None]

    violation = affine.constraint_violation(normalised.reshape(-1))
    if violation > constraint_tol:
        raise ValueError(
            f"orbifold constraints violated by {violation:.3e} after projecting onto the "
            f"sphere (tolerance {constraint_tol:.1e})"
        )

    return SphericalEmbeddingResult(
        points=normalised,
        energy=stage2.energy,
        constraint_violation=violation,
        max_radius_before_normalisation=float(radii.max()),
        min_radius_before_normalisation=float(radii.min()),
        stage1=stage1,
        stage2=stage2,
    )
