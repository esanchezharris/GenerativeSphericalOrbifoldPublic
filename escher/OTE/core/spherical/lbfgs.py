r"""Projected L-BFGS for minimising a smooth energy subject to :math:`Ax = b`.

Port of ``LBFGS/OptimSolverLBFGS_NEW.m`` (Nocedal & Wright alg. 7.4 + 7.5, restricted to the
constraint null space), with one change of representation.

**Reduced vs. projected coordinates.** The reference carries an explicit orthonormal basis
:math:`N` of :math:`\ker A` and stores its L-BFGS memory as reduced vectors :math:`N^\top v`.
Because :math:`N` is orthonormal, :math:`\langle N^\top a, N^\top b \rangle = \langle Pa, Pb
\rangle` for the orthogonal projector :math:`P = NN^\top`, and the two-loop recursion uses
nothing but inner products and linear combinations. So we store the *projected* full-space
vectors instead and never form :math:`N` (see :mod:`.affine_space` for why that matters).
Every scalar in the recursion is identical.

Deliberate deviations from the reference, both documented at their site:

- the backtracking line search is capped (the reference loops unbounded),
- :meth:`solve` reports a structured result rather than printing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

import numpy as np

from .affine_space import AffineSpace
from .precond import Precond, PrecondIdentity

__all__ = ["LBFGSResult", "ProjectedLBFGS"]

ObjectiveFn = Callable[[np.ndarray], tuple[float, np.ndarray]]
ExitFlag = Literal["TolX", "TolFun", "NoProgress", "MaxIter", "LineSearchFailed"]


@dataclass
class LBFGSResult:
    """Outcome of a :meth:`ProjectedLBFGS.solve` call."""

    x: np.ndarray
    energy: float
    n_iter: int
    exit_flag: ExitFlag
    energy_history: list[float] = field(default_factory=list)
    #: ``max|P g|`` per iteration -- the stationarity measure for a *constrained* problem.
    #: The unprojected ``max|g|`` is not one: at the optimum the full gradient equals
    #: :math:`A^\top \lambda`, so it stays large (~2.14 on the reference problem) while the
    #: projected one falls to ~1e-10.
    proj_grad_inf_history: list[float] = field(default_factory=list)
    f_count: int = 0
    grad_count: int = 0

    def __repr__(self) -> str:  # pragma: no cover - diagnostics only
        return (
            f"LBFGSResult(energy={self.energy:.10g}, n_iter={self.n_iter}, "
            f"exit={self.exit_flag!r}, f_count={self.f_count})"
        )


class ProjectedLBFGS:
    """L-BFGS restricted to :math:`\\ker A`, with a pluggable preconditioner.

    Args:
        fun: ``x -> (value, gradient)``, gradient in full ambient coordinates.
        x0: starting point; must already satisfy ``A x0 = b``.
        affine: the constraint space.
        precond: initial preconditioner. Swap it mid-solve with :meth:`update_precond`.
        memory: number of correction pairs. ``0`` gives memoryless preconditioned steepest
            descent, which is what the reference's second stage uses.
    """

    #: Armijo sufficient-decrease constant (``ls_alpha`` in the reference).
    ls_alpha = 0.2
    #: Backtracking factor (``ls_beta`` in the reference).
    ls_beta = 0.5
    #: How many consecutive tolerance hits before stopping (``stopCntAccept``).
    stop_count_accept = 2
    #: Tolerance for the initial feasibility assertion (``tol_Ab``).
    tol_feasible = 1e-10
    #: Cap on backtracking steps. The reference loops unbounded; with ``ls_beta = 0.5`` this
    #: bottoms out around ``t = 1e-18``, far below any useful step.
    max_line_search_steps = 60

    def __init__(
        self,
        fun: ObjectiveFn,
        x0: np.ndarray,
        affine: AffineSpace,
        precond: Precond | None = None,
        memory: int = 3,
    ):
        self.fun = fun
        self.affine = affine
        self.precond: Precond = precond if precond is not None else PrecondIdentity()

        x0 = np.asarray(x0, dtype=np.float64).ravel()
        violation = affine.constraint_violation(x0)
        if violation > self.tol_feasible:
            raise ValueError(
                f"x0 violates the linear constraints by {violation:.3e} "
                f"(tolerance {self.tol_feasible:.1e}); project it first"
            )

        self.x = x0.copy()
        self.x_prev: np.ndarray | None = None
        self.f_count = 0
        self.grad_count = 0
        self.t = float("nan")
        self.gamma = float("nan")

        self.update_memory(memory)
        self.x_f, self.x_grad = self._evaluate_with_grad(self.x)
        self.proj_grad = affine.project_tangent(self.x_grad)

    @property
    def stationarity(self) -> float:
        """``max|P g|`` at the current iterate: 0 exactly at a constrained optimum."""
        return float(np.abs(self.proj_grad).max())

    # ------------------------------------------------------------------ configuration
    def update_precond(self, precond: Precond) -> None:
        """Swap the preconditioner. Used between the two stages of the reference solve."""
        self.precond = precond

    def update_memory(self, memory: int) -> None:
        """Reset the correction-pair memory to hold ``memory`` pairs.

        The dummy first column of ones is from the reference: with ``curr_m == 0`` the
        recursion loops do not run and ``gamma = <s,y>/<y,y>`` evaluates to exactly 1,
        which is the intended :math:`H_k^0` scaling for the first step.
        """
        if memory < 0:
            raise ValueError("memory must be non-negative")
        self.memory = memory
        self.curr_m = 0
        n = self.affine.n_vars
        slots = max(memory, 1)  # always keep the dummy column alive
        self.s_k = np.zeros((n, slots))
        self.y_k = np.zeros((n, slots))
        self.rho_k = np.zeros(slots)
        self.s_k[:, 0] = 1.0
        self.y_k[:, 0] = 1.0

    # -------------------------------------------------------------------- evaluation
    def _evaluate(self, x: np.ndarray) -> float:
        value, _ = self.fun(x)
        self.f_count += 1
        return float(value)

    def _evaluate_with_grad(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        value, grad = self.fun(x)
        self.f_count += 1
        self.grad_count += 1
        return float(value), np.asarray(grad, dtype=np.float64).ravel()

    # --------------------------------------------------------------------- iteration
    def _search_direction(self) -> np.ndarray:
        """Two-loop recursion, in projected full-space coordinates."""
        q = self.affine.project_tangent(self.x_grad)

        alpha = np.zeros(self.curr_m)
        for i in range(self.curr_m):
            alpha[i] = self.rho_k[i] * float(self.s_k[:, i] @ q)
            q = q - alpha[i] * self.y_k[:, i]

        # H_k^0 scaling from the most recent pair (column 0), or the dummy on step 1.
        yy = float(self.y_k[:, 0] @ self.y_k[:, 0])
        self.gamma = float(self.s_k[:, 0] @ self.y_k[:, 0]) / yy if yy > 0 else 1.0
        r = self.gamma * self.precond.apply(q, self.x)

        for i in range(self.curr_m):
            beta = self.rho_k[i] * float(self.y_k[:, i] @ r)
            r = r + self.s_k[:, i] * (alpha[i] - beta)

        return -r

    def _line_search(self, p: np.ndarray) -> tuple[float, bool]:
        """Backtracking Armijo search. Returns ``(step, succeeded)``."""
        slope = float(self.x_grad @ p)
        t = 1.0
        for _ in range(self.max_line_search_steps):
            if self._evaluate(self.x + t * p) <= self.x_f + self.ls_alpha * t * slope:
                return t, True
            t *= self.ls_beta
        return t, False

    def iterate(self) -> bool:
        """One L-BFGS step. Returns ``False`` if the line search failed to make progress."""
        self.x_prev = self.x.copy()
        self.x_f_prev = self.x_f
        grad_prev = self.x_grad.copy()

        p = self._search_direction()
        self.t, ok = self._line_search(p)
        if not ok:
            return False

        self.x = self.x + self.t * p
        self.x_f, self.x_grad = self._evaluate_with_grad(self.x)
        self.proj_grad = self.affine.project_tangent(self.x_grad)

        # Shift memory right, newest pair into column 0.
        if self.memory > 0:
            self.s_k[:, 1:] = self.s_k[:, :-1]
            self.y_k[:, 1:] = self.y_k[:, :-1]
            self.rho_k[1:] = self.rho_k[:-1]
        self.curr_m = min(self.curr_m + 1, self.memory)

        s = self.affine.project_tangent(self.x - self.x_prev)
        y = self.affine.project_tangent(self.x_grad - grad_prev)
        self.s_k[:, 0] = s
        self.y_k[:, 0] = y
        ys = float(y @ s)
        # A non-positive curvature product would make the update non-descent; the reference
        # divides regardless, which yields inf/nan. Skipping keeps the previous scaling.
        self.rho_k[0] = 1.0 / ys if abs(ys) > 1e-300 else 0.0
        return True

    # ------------------------------------------------------------------------- driver
    def solve(
        self,
        tol_x: float = 1e-10,
        tol_fun: float = 0.0,
        max_iter: int = 10_000,
        record_history: bool = True,
    ) -> LBFGSResult:
        """Iterate to convergence.

        Stopping mirrors the reference: relative step size below ``tol_x``, or relative
        energy change below ``tol_fun``, each required on ``stop_count_accept`` consecutive
        iterations; plus an exact-no-progress guard.
        """
        tol_x_count = 0
        tol_fun_count = 0
        exit_flag: ExitFlag = "MaxIter"
        energy_history: list[float] = [self.x_f] if record_history else []
        grad_history: list[float] = [self.stationarity] if record_history else []
        n_iter = 0

        for n_iter in range(1, max_iter + 1):
            if not self.iterate():
                exit_flag = "LineSearchFailed"
                break

            if record_history:
                energy_history.append(self.x_f)
                grad_history.append(self.stationarity)

            step = self.x - self.x_prev
            if np.abs(step).max() == 0.0:
                exit_flag = "NoProgress"
                break

            rel_delta_x = np.linalg.norm(step) / (1.0 + np.linalg.norm(self.x_prev))
            tol_x_count = tol_x_count + 1 if rel_delta_x < tol_x else 0
            if tol_x_count >= self.stop_count_accept:
                exit_flag = "TolX"
                break

            if abs(self.x_f_prev - self.x_f) < tol_fun * (1.0 + abs(self.x_f_prev)):
                tol_fun_count += 1
            else:
                tol_fun_count = 0
            if tol_fun_count >= self.stop_count_accept:
                exit_flag = "TolFun"
                break

        return LBFGSResult(
            x=self.x.copy(),
            energy=self.x_f,
            n_iter=n_iter,
            exit_flag=exit_flag,
            energy_history=energy_history,
            proj_grad_inf_history=grad_history,
            f_count=self.f_count,
            grad_count=self.grad_count,
        )
