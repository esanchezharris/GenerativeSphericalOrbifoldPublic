# Golden data

Dumps from a run of the reference MATLAB implementation
([`noamaig/spherical_orbifolds`](https://github.com/noamaig/spherical_orbifolds)) on a
2502-vertex sphere-topology mesh, used to validate the Python port.

Load them with `tests.golden.load_golden()`.

## Contents

| File | Shape | What it is |
|---|---|---|
| `A.mat` | 165 × 7662 sparse | Orbifold boundary equations, from `generateBoundaryEquations`. Full row rank. |
| `b.mat` | 165 × 1 | RHS of the same. |
| `N.mat` | 7662 × 7497 sparse | Null-space basis of `A` from `AffineSpace`. **Orthonormal and sparse** — 7884 nnz. |
| `x0.mat` | 7662 × 1 | `Ab.projectOnto(X)` — the lifted initial guess. Every vertex has radius exactly 1. |
| `wMat.mat` | 2554 × 2554 sparse | `Solver.Wmat` — cotangent Laplacian with negative weights clamped to `1e-3`. |
| `L.mat` | 2554 × 2554 sparse | Raw cotangent Laplacian, same sparsity pattern, 1696 negative off-diagonals. |
| `V.mat` | 2502 × 3 | Source mesh vertices, **uncut**. |
| `F.mat` | 5000 × 3 | Faces, **1-based** (MATLAB convention). |

Coordinates are `colStack`ed: `[x₁ y₁ z₁ x₂ y₂ z₂ …]`, so `x.reshape(-1, 3)` recovers
per-vertex rows.

## Verified invariants

These are asserted by `tests/test_golden_data.py`:

- `‖A · N‖∞ = 2.4e-16` — `N` really is the null space
- `‖NᵀN − I‖∞ = 2.2e-16` — orthonormal columns
- `max|A · x0 − b| = 3.3e-16` — `x0` satisfies the constraints
- all `x0` radii `= 1.000000` — `x0` lies on the unit sphere
- uncut mesh `V − E + F = 2502 − 7500 + 5000 = 2` — sphere topology
- cut mesh `2554 − 7553 + 5000 = 1` — disk topology, 52 vertices duplicated along the cut
- `wMat` has zero negative off-diagonals; exactly 1696 entries equal `0.001`, matching the
  1696 negative off-diagonals in `L` — confirms the `clamp=1e-3` in `@Solver/Solver.m`

## Excluded

- **`xgrad.pkl`** — *not* ground truth. It is a torch tensor with norm `4.9e+30`, 80.6%
  radially aligned where a Karcher gradient must be tangent to the sphere. It is a snapshot
  of the old diverged `SOTESolver`, kept only as a regression curiosity. Do not test against
  it; use finite differences instead.
- **`P.mat`** — a MATLAB object container (`s0/s1/s2/arr` struct), not loadable by
  `scipy.io.loadmat` as an array. Unneeded: the preconditioner rebuilds from `wMat` and `N`.
