r"""Preconditioners for the projected L-BFGS.

Port of ``LBFGS/Precond.m``, ``Precond_Identity.m``, ``Precond_Fixed.m`` and ``SparseLU.m``.

The reference builds the reduced matrix :math:`-N^\top (W \otimes I_3) N` explicitly and
LU-factorises it. Working projector-only (see :mod:`.affine_space`) we get the same operator
without ever forming :math:`N`, by solving the sparse saddle-point system

.. math::

    \begin{bmatrix} -K & A^\top \\ A & 0 \end{bmatrix}
    \begin{bmatrix} r \\ \lambda \end{bmatrix}
    = \begin{bmatrix} q \\ 0 \end{bmatrix},
    \qquad K = W \otimes I_3 .

The second block row forces :math:`r \in \ker A`; projecting the first row then gives
:math:`-PKr = Pq = q` for :math:`q` already tangent, which is exactly the reduced system.
This is the same KKT trick the planar :class:`~escher.OTE.core.OTESolver.OTESolver` uses, so
both halves of the codebase now solve their constrained problems the same way.

Sign convention: the Laplacians here store ``+w`` off-diagonal and ``-sum(w)`` on the
diagonal, so :math:`W` is negative semi-definite and :math:`-K` is the PSD one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

from .affine_space import AffineSpace

__all__ = ["Precond", "PrecondIdentity", "PrecondFixed", "SparseLU"]


class SparseLU:
    """Thin wrapper over :func:`scipy.sparse.linalg.splu`, mirroring ``SparseLU.m``."""

    def __init__(self, matrix: sp.spmatrix):
        if not sp.issparse(matrix):
            raise TypeError("SparseLU requires a sparse matrix")
        self.shape = matrix.shape
        self._lu = spla.splu(sp.csc_matrix(matrix))

    def solve(self, rhs: np.ndarray) -> np.ndarray:
        return self._lu.solve(np.asarray(rhs, dtype=np.float64))


class Precond(ABC):
    """Return ``y = P_z(x)``: a preconditioner applied to ``x``, possibly depending on ``z``."""

    @abstractmethod
    def apply(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray: ...


class PrecondIdentity(Precond):
    """No preconditioning. Used for the second stage of the reference's two-stage solve."""

    def apply(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        return x


class PrecondFixed(Precond):
    r"""Inverse of the reduced Laplacian :math:`-N^\top (W \otimes I_3) N`, factorised once.

    Args:
        laplacian: ``(n, n)`` weighted Laplacian ``W`` (``+w`` off-diagonal, ``-sum(w)``
            diagonal), as produced by the mesh weights.
        affine: the constraint space; supplies ``A`` and the tangent projection.
        dim: spatial dimension, 3 for the sphere.
    """

    def __init__(self, laplacian: sp.spmatrix, affine: AffineSpace, dim: int = 3):
        n = laplacian.shape[0]
        if n * dim != affine.n_vars:
            raise ValueError(
                f"Laplacian is {n}x{n} with dim={dim} ({n * dim} variables) but the affine "
                f"space has {affine.n_vars}"
            )
        self.affine = affine
        self.n_vars = affine.n_vars
        self.n_constraints = affine.n_constraints

        # K = W (x) I_dim, in colStack ordering [x1 y1 z1 x2 y2 z2 ...].
        K = sp.kron(sp.csr_matrix(laplacian), sp.eye(dim, format="csr"), format="csr")
        A = affine.A
        zero = sp.csr_matrix((self.n_constraints, self.n_constraints))
        kkt = sp.bmat([[-K, A.T], [A, zero]], format="csc")

        try:
            self._lu = SparseLU(kkt)
        except RuntimeError as exc:  # singular factor
            raise ValueError(
                "the KKT system is singular: the constraints do not pin down the "
                "Laplacian's null space (rigid translations). Check that the orbifold "
                "cone points are fixed."
            ) from exc

    def apply(self, x: np.ndarray, z: np.ndarray | None = None) -> np.ndarray:
        rhs = np.zeros(self.n_vars + self.n_constraints, dtype=np.float64)
        rhs[: self.n_vars] = np.asarray(x, dtype=np.float64).ravel()
        sol = self._lu.solve(rhs)
        # Re-project to kill drift from the sparse solve; the second block row already
        # asks for this, but the factorisation only satisfies it to its own tolerance.
        return self.affine.project_tangent(sol[: self.n_vars])
