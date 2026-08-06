"""Solve the reference spherical orbifold problem end to end and plot the result.

Run from the repo root::

    python escher/sanity_checks/solve_reference_sphere.py

Writes ``output/reference_sphere.png``: the embedded fundamental domain, the same domain
tiled by its symmetry group into the closed sphere, and the L-BFGS convergence curve.

The reference dump is the ``(4,2,2)`` dihedral orbifold -- ``orbifold_type = 4, k = 4`` in
``script_orbifold_sphere.m``. Its rotation group is |D_4| = 8, so the domain is a 90-degree
lune from pole to equator and eight copies close the sphere.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from escher.OTE.core.spherical.solver import solve_spherical_embedding
from escher.geometry.sphere_tiler import SphericalTiler
from escher.rendering.render_sphere_matplotlib import draw_sphere_edges
from tests.golden import load_golden

OUTPUT = Path(os.environ.get("OUTPUT_DIR", "output"))
K = 4


def main() -> None:
    g = load_golden()
    print(f"mesh        : {g.n_verts} vertices (cut), {len(g.edges)} edges")
    print(f"constraints : {g.A.shape[0]}")

    result = solve_spherical_embedding(
        edges=g.edges,
        weights=g.edge_weights,
        laplacian=g.wmat,
        A=g.A,
        b=g.b,
        x0=g.x0,
    )
    s1, s2 = result.stage1, result.stage2

    print()
    print(f"stage 1 (Laplacian precond) : {s1.n_iter:3d} iters, "
          f"{s1.energy_history[0]:.10f} -> {s1.energy:.10f}  [{s1.exit_flag}]")
    print(f"stage 2 (identity, m=0)     : {s2.n_iter:3d} iters, "
          f"-> {s2.energy:.10f}  [{s2.exit_flag}]")
    print(f"stationarity |P g|_inf      : {s1.proj_grad_inf_history[0]:.3e} -> "
          f"{s2.proj_grad_inf_history[-1]:.3e}")
    print(f"radii before normalisation  : "
          f"[{result.min_radius_before_normalisation:.4f}, "
          f"{result.max_radius_before_normalisation:.4f}]")
    print(f"max|Ax - b| after           : {result.constraint_violation:.3e}  "
          f"(reference asserts < 1e-6)")

    tiler = SphericalTiler.dihedral(K)
    coverage = tiler.coverage_fraction(result.points, n_bins=200)
    print()
    print(f"symmetry group              : D_{K}, order {tiler.order}")
    print(f"sphere coverage by {tiler.order} copies : {coverage:.4f}")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(14, 5.0))
    fig.suptitle(
        "Spherical orbifold Tutte embedding — (4,2,2) reference problem  "
        f"({g.n_verts} vertices, {len(g.edges)} edges)",
        fontsize=13,
    )

    ax = fig.add_subplot(1, 4, 1)
    draw_sphere_edges(ax, result.points, g.edges, elev=20, azim=90, color="#12305c")
    ax.set_title("fundamental domain\n(1/8 of the sphere)", fontsize=9)

    # Tiled: replicate the edge set by the group, as one big point/edge array.
    n = len(result.points)
    tiled_points = tiler.tile_points(result.points).reshape(-1, 3)
    tiled_edges = np.concatenate([g.edges + i * n for i in range(tiler.order)])

    for j, (elev, azim, label) in enumerate(
        [(20, 90, "tiled — front"), (-20, 270, "tiled — back")], start=2
    ):
        ax = fig.add_subplot(1, 4, j)
        draw_sphere_edges(ax, tiled_points, tiled_edges, elev=elev, azim=azim, color="#12305c")
        ax.set_title(f"{label}\n({tiler.order} copies, D$_4$)", fontsize=9)

    ax = fig.add_subplot(1, 4, 4)
    hist = s1.energy_history + s2.energy_history[1:]
    ax.plot(hist, color="#12305c", lw=1.6)
    ax.axvline(len(s1.energy_history) - 1, color="#bbb", ls="--", lw=1)
    ax.text(
        len(s1.energy_history) - 1, max(hist) * 0.86,
        "  stage 2\n  (precond reset)", fontsize=7.5, color="#777",
    )
    ax.set_xlabel("L-BFGS iteration", fontsize=8)
    ax.set_ylabel("Karcher energy", fontsize=8)
    ax.set_title(
        f"convergence\n{hist[0]:.4f} → {hist[-1]:.4f}", fontsize=9
    )
    ax.tick_params(labelsize=7)
    ax.grid(alpha=0.3)

    out = OUTPUT / "reference_sphere.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
