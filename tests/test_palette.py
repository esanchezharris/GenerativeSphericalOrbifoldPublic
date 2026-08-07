"""CPU validation of the per-tile hue machinery (escher/rendering/palette.py).

Pins the color math the renderer change relies on, the adjacency read off the group
itself, and the vertex-attribute expansion contract used by render_tiled_sphere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from escher.OTE.tilings_sphere.boundary_explicit import BoundaryExplicitDihedral
from escher.rendering.palette import (
    assign_palette_indices,
    hue_rotation_matrix,
    tile_adjacency,
    tile_color_matrices,
)


# ------------------------------------------------------------------- color matrices
def test_zero_degrees_is_identity():
    assert torch.allclose(hue_rotation_matrix(0.0), torch.eye(3), atol=1e-7)


def test_120_degrees_cycles_the_channels():
    R = hue_rotation_matrix(120.0)
    red, green = torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 1.0, 0.0])
    assert torch.allclose(R @ red, green, atol=1e-6)
    assert torch.allclose(R @ R @ R, torch.eye(3), atol=1e-6)
    assert torch.allclose(hue_rotation_matrix(-120.0), R.T, atol=1e-6)


def test_gray_axis_is_fixed():
    for deg in (37.0, 120.0, -90.0):
        R = hue_rotation_matrix(deg)
        gray = torch.full((3,), 0.5)
        assert torch.allclose(R @ gray, gray, atol=1e-6)


# ----------------------------------------------------------------------- adjacency
@pytest.mark.parametrize("k", [3, 4, 6])
def test_dihedral_adjacency_matches_the_group_parity(k):
    """Measured, not assumed: the southern alignment depends on the parity of k.

    D_k's flip axes sit at azimuth multiples of pi/k. For EVEN k that set includes each
    tile's mid-arc axis, so every southern copy lies directly under a northern one and
    the graph is the k-prism: 3k edges, degree 3. For ODD k the flip axes pass through
    the CORNERS instead -- the southern ring is offset by half a tile and each bottom
    arc is split between two southern neighbours: the k-antiprism (k=3 is the
    octahedron graph), 4k edges, degree 4. Cone-point-only contacts (the pole touches
    every rotation copy) must not count either way."""
    orb = BoundaryExplicitDihedral.from_resolution(k=k, n_theta=6, n_phi=5)
    pairs = tile_adjacency(orb.tiler(), orb.mesh)
    degree = np.zeros(2 * k, dtype=int)
    for a, b in pairs:
        assert a != b
        degree[a] += 1
        degree[b] += 1
    if k % 2 == 0:
        assert len(pairs) == 3 * k
        assert (degree == 3).all()
    else:
        assert len(pairs) == 4 * k
        assert (degree == 4).all()


@pytest.mark.parametrize("k", [3, 4, 6])
def test_three_hues_properly_color_every_k(k):
    orb = BoundaryExplicitDihedral.from_resolution(k=k, n_theta=6, n_phi=5)
    tiler = orb.tiler()
    colors = assign_palette_indices(tiler, orb.mesh, n_colors=3)
    assert colors.shape == (2 * k,)
    assert set(colors) <= {0, 1, 2}
    for a, b in tile_adjacency(tiler, orb.mesh):
        assert colors[a] != colors[b], f"tiles {a},{b} share a chain and a color"


def test_octahedral_tiling_is_three_colorable():
    """The (2,3,4) kite tiling: 24 tiles, 3-regular adjacency (36 shared chains),
    properly 3-colored -- the target image's three-hue look scales to full density."""
    from escher.OTE.tilings_sphere.boundary_explicit_triple import BoundaryExplicitTriple

    orb = BoundaryExplicitTriple.from_resolution((2, 3, 4), n=6)
    tiler = orb.tiler()
    pairs = tile_adjacency(tiler, orb.mesh)
    assert len(pairs) == 36
    degree = np.zeros(24, dtype=int)
    for a, b in pairs:
        degree[a] += 1
        degree[b] += 1
    assert (degree == 3).all()

    colors = assign_palette_indices(tiler, orb.mesh, n_colors=3)
    for a, b in pairs:
        assert colors[a] != colors[b]


def test_tile_color_matrices_draws_from_the_palette():
    orb = BoundaryExplicitDihedral.from_resolution(k=4, n_theta=6, n_phi=5)
    hues = [0.0, 120.0, -120.0]
    mats = tile_color_matrices(orb.tiler(), orb.mesh, hues)
    assert mats.shape == (8, 3, 3)
    palette = torch.stack([hue_rotation_matrix(h) for h in hues])
    for m in mats:
        assert any(torch.allclose(m, p, atol=1e-6) for p in palette)


# ------------------------------------------------------- renderer attribute contract
def test_per_vertex_expansion_matches_per_tile_matmul():
    """render_tiled_sphere flattens (G,3,3) -> (G,9) and repeat_interleaves by the
    per-tile vertex count; build_tiled_sphere lays vertices out in tile-major blocks.
    This pins that the two orderings agree, in pure torch."""
    G, n = 4, 5
    torch.manual_seed(0)
    mats = torch.stack([hue_rotation_matrix(90.0 * g) for g in range(G)])
    colors = torch.rand(G * n, 3)

    vcm = mats.reshape(G, 9).repeat_interleave(n, dim=0)  # what the renderer builds
    out = torch.einsum("vij,vj->vi", vcm.reshape(-1, 3, 3), colors)

    expected = torch.cat([colors[g * n : (g + 1) * n] @ mats[g].T for g in range(G)])
    assert torch.allclose(out, expected, atol=1e-6)
