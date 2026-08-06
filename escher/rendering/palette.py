"""Per-tile hue rotation: the alternating-color Escher look, at render time only.

In the target style every figure is the SAME figure with its hue rotated (blue / pink /
orange), and neighboring figures never share a color -- that alternation is what makes
the interlocking legible. All tiles here sample one shared texture, so the recolor is a
per-tile 3x3 matrix applied to the sampled color: a rotation about the RGB gray axis,
which preserves luminance and turns one gingerbread into the palette's variants.

Training never uses this (each tile view would push a different hue into the one shared
texture and the gradients would fight to gray); it is applied in previews and final
renders via ``render_tiled_sphere(tile_color_matrices=...)``.
"""

from __future__ import annotations

import numpy as np
import torch

__all__ = [
    "hue_rotation_matrix",
    "tile_adjacency",
    "assign_palette_indices",
    "tile_color_matrices",
]


def hue_rotation_matrix(degrees: float) -> torch.Tensor:
    """(3, 3) rotation of RGB space about the gray axis (1,1,1)/sqrt(3).

    Rodrigues' formula; 0 deg is the identity and +120 deg cyclically permutes
    R -> G -> B -> R, so a three-hue palette at 0/+120/-120 costs no saturation.
    """
    theta = np.deg2rad(degrees)
    k = np.ones(3) / np.sqrt(3.0)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    R = np.eye(3) * np.cos(theta) + np.sin(theta) * K + (1 - np.cos(theta)) * np.outer(k, k)
    return torch.as_tensor(R, dtype=torch.float32)


def tile_adjacency(tiler, mesh) -> list[tuple[int, int]]:
    """Tile pairs that share a whole boundary chain on the UNDEFORMED tiling.

    Two tiles are adjacent iff their vertex images coincide at >= 2 positions: chains
    (cut mates, the shared bottom arc) give whole shared curves, while cone points --
    the pole is one vertex on every rotation copy -- give a single position and do not
    count. Robust for any k because it reads the group itself, not an assumed layout.
    """
    boundary = np.concatenate([mesh.left, mesh.right, mesh.bottom])
    pts = mesh.points[boundary]

    seen: dict[tuple, set[int]] = {}
    for g, rot in enumerate(tiler.rotations):
        for p in pts @ rot.T:
            seen.setdefault(tuple(np.round(p, 6)), set()).add(g)

    counts: dict[tuple[int, int], int] = {}
    for tiles in seen.values():
        ordered = sorted(tiles)
        for i, a in enumerate(ordered):
            for b in ordered[i + 1 :]:
                counts[(a, b)] = counts.get((a, b), 0) + 1
    return sorted(pair for pair, n in counts.items() if n >= 2)


def assign_palette_indices(tiler, mesh, n_colors: int) -> np.ndarray:
    """A color index per group element such that adjacent tiles differ, deterministically.

    The adjacency graph of a (k,2,2) dihedral tiling is the k-prism for even k and the
    k-antiprism for odd k (measured: D_k's flip axes hit tile mids for even k and tile
    CORNERS for odd k, offsetting the southern ring by half a tile). Rather than rely
    on either shape, do an exact backtracking search -- the graph has at most a few
    dozen nodes. If ``n_colors`` genuinely cannot properly color it (odd-k antiprisms
    other than k=3 need 4), fall back to greedy least-conflict so rendering still works.
    """
    G = tiler.order
    adj = [[] for _ in range(G)]
    for a, b in tile_adjacency(tiler, mesh):
        adj[a].append(b)
        adj[b].append(a)

    colors = np.full(G, -1, dtype=np.int64)

    def backtrack(i: int) -> bool:
        if i == G:
            return True
        for c in range(n_colors):
            if all(colors[nb] != c for nb in adj[i]):
                colors[i] = c
                if backtrack(i + 1):
                    return True
                colors[i] = -1
        return False

    if backtrack(0):
        return colors

    for i in range(G):  # fallback: minimize conflicts, never fail
        used = [colors[nb] for nb in adj[i] if colors[nb] >= 0]
        colors[i] = min(range(n_colors), key=lambda c: used.count(c))
    return colors


def tile_color_matrices(tiler, mesh, hues_deg) -> torch.Tensor:
    """(G, 3, 3) per-tile color matrices from a palette of hue angles."""
    indices = assign_palette_indices(tiler, mesh, len(hues_deg))
    return torch.stack([hue_rotation_matrix(float(hues_deg[i])) for i in indices])
