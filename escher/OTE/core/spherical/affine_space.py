r"""The affine constraint space :math:`\{x : Ax = b\}` of the orbifold boundary conditions.

Port of ``LBFGS/AffineSpace.m``, with one deliberate change of representation.

**Why no explicit null-space basis.** The MATLAB code calls ``qr(A')`` and keeps the
trailing columns as a basis :math:`N`. On sparse input MATLAB returns a sparse factor, and
the reference dump bears that out: ``N.mat`` is 7662x7497 with just 7884 nonzeros and
orthonormal columns. The existing Python port instead used ``np.linalg.qr(mode='complete')``,
which is dense -- a 7662x7662 factor, ~470 MB, and it grows quadratically with mesh
resolution.

Neither is necessary. Every use of :math:`N` in the solver is either

- :math:`N N^\top v` -- orthogonal projection of a full-space vector onto :math:`\ker A`, or
- an inner product between two vectors already lying in :math:`\ker A`,

and both are available from the projector

.. math:: P = I - A^\top (A A^\top)^{-1} A

directly. :math:`A A^\top` is only ``n_constraints`` square -- **165x165** for the reference
problem -- so one dense Cholesky up front makes every projection a pair of sparse matvecs
plus a tiny triangular solve. Nothing scales with ``3n x 3n``.

:meth:`AffineSpace.null_space_basis` still materialises an explicit sparse basis, used by the
tests to check this projector against ``N.mat``, but the solver never calls it.
"""

from __future__ import annotations

import numpy as np
import scipy.linalg
import scipy.sparse as sp

__all__ = ["AffineSpace"]


class AffineSpace:
    r"""Orthogonal projection onto :math:`\{x : Ax = b\}` and onto its tangent
    :math:`\ker A`.

    Args:
        A: ``(m, n)`` constraint matrix, sparse or dense. Rows must be linearly independent.
        b: ``(m,)`` right-hand side.
        rcond: relative tolerance used to detect rank deficiency in :math:`A A^\top`.
    """

    def __init__(self, A, b, rcond: float = 1e-10):
        self.A = sp.csr_matrix(A)
        self.b = np.asarray(b, dtype=np.float64).ravel()
        self.n_constraints, self.n_vars = self.A.shape

        if self.b.shape != (self.n_constraints,):
            raise ValueError(f"b has shape {self.b.shape}, expected ({self.n_constraints},)")

        # A A^T is (m, m) -- small by construction, so dense Cholesky is the right call.
        gram = (self.A @ self.A.T).toarray()
        eigmin = float(np.linalg.eigvalsh(gram).min())
        scale = float(np.trace(gram)) / max(self.n_constraints, 1)
        if eigmin <= rcond * max(scale, 1.0):
            raise ValueError(
                "constraint rows are linearly dependent "
                f"(min eigenvalue of A A^T = {eigmin:.3e}); "
                "drop redundant orbifold constraints before building the affine space"
            )
        self._gram_cho = scipy.linalg.cho_factor(gram, lower=True)

        # Minimum-norm particular solution. Any x_p with A x_p = b gives the same
        # project() result, so the choice is free.
        self.x_p = self.A.T @ scipy.linalg.cho_solve(self._gram_cho, self.b)

    @property
    def dim_null(self) -> int:
        """Dimension of :math:`\\ker A`, i.e. the number of free parameters."""
        return self.n_vars - self.n_constraints

    def project_tangent(self, v: np.ndarray) -> np.ndarray:
        r"""Orthogonal projection onto :math:`\ker A`. Equivalent to ``N @ (N.T @ v)``.

        Use for gradients and search directions -- anything that is a *difference* of points
        in the affine space rather than a point in it.
        """
        v = np.asarray(v, dtype=np.float64)
        correction = self.A.T @ scipy.linalg.cho_solve(self._gram_cho, self.A @ v)
        return v - correction

    def project(self, x: np.ndarray) -> np.ndarray:
        r"""Orthogonal projection onto the affine space :math:`\{x : Ax = b\}`.

        Equivalent to ``x_p + N @ (N.T @ (x - x_p))``, and idempotent.
        """
        x = np.asarray(x, dtype=np.float64)
        residual = self.A @ x - self.b
        return x - self.A.T @ scipy.linalg.cho_solve(self._gram_cho, residual)

    def constraint_violation(self, x: np.ndarray) -> float:
        """``max|Ax - b|`` -- the quantity the reference asserts is below 1e-6."""
        return float(np.abs(self.A @ np.asarray(x) - self.b).max())

    def null_space_basis(self) -> sp.csr_matrix:
        r"""An explicit sparse orthonormal basis of :math:`\ker A`.

        Only used to cross-check the projector against the reference ``N.mat``; the solver
        works projector-only. Cost is a dense QR of ``A.T``, so keep it out of hot paths.
        """
        q, _ = np.linalg.qr(self.A.toarray().T, mode="complete")
        return sp.csr_matrix(q[:, self.n_constraints :])
