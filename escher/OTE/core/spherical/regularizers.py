r"""Shape regularizers for the spherical SDS loop.

Why these exist: an unconstrained strong shape signal degenerates the mesh. Measured on the
"strong silhouette" run (8x weight, every step, no regularization): tile perimeter grew to
1.85x but face-area spread (p99/p1) went from 17.9 undeformed to **632**, with the smallest
triangles at 1.5% of the median -- perimeter bought by collapsing triangles, not by the
outline taking the figure's shape.

The planar pipeline guards against this with ``W_REGULARIZATION * torch.sum(W**2)`` applied
on every step (``main.py`` line 615) plus a planar ``EqualAreaLoss`` for multi-prompt tiles.
The former ports directly (it penalizes sigmoid saturation, where degenerate weight
configurations live). The latter is 2D cross-product area on subtile groups, so the
spherical counterpart here works on per-face **solid angles** instead.

Both act on the embedding through the implicit solve, i.e. their gradients reach the edge
weights the same way the SDS gradient does.
"""

from __future__ import annotations

import torch
from torch import Tensor

__all__ = [
    "area_margin_loss",
    "area_tail_loss",
    "equal_area_loss",
    "spherical_face_areas",
]


def area_margin_loss(
    points: "Tensor", faces: "Tensor", reference_areas: "Tensor", margin: float = 0.15
) -> "Tensor":
    r"""Hinge barrier keeping every face's area ratio above ``margin``.

    :math:`\sum_i \mathrm{relu}(m - a_i/a_i^0)^2` -- exactly zero while all faces hold a
    healthy fraction of their reference area, ramping quadratically as any face approaches
    the fold at ratio 0 (and growing through it, since the ratio goes negative).

    Why this exists, measured: rejection alone (undo folding steps) kept run B2 valid but
    rejected **78% of steps** -- Adam has no gradient telling it the wall is there, so it
    rams the same direction forever and the outline stalls. The hinge supplies that
    gradient. Rejection stays on as the hard guarantee; this makes it rarely needed.
    Diagnostic basis: run B1's folds were boundary-adjacent interior faces (12/19), i.e.
    approachable smoothly -- a barrier can catch them before the sign change.
    """
    ratio = spherical_face_areas(points, faces) / reference_areas
    return torch.relu(margin - ratio).square().sum()


def spherical_face_areas(points: Tensor, faces: Tensor) -> Tensor:
    """``(n_faces,)`` signed solid angles, differentiable.

    Torch twin of :func:`escher.geometry.spherical_sanity_checks.signed_solid_angles`
    (Van Oosterom-Strackee); that one is numpy for the certificates, this one carries
    gradients for the loss. ``tests/test_regularizers.py`` pins them equal.
    """
    p = points / points.norm(dim=-1, keepdim=True).clamp_min(1e-30)
    a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
    numerator = (a * torch.linalg.cross(b, c, dim=-1)).sum(-1)
    denominator = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
    return 2.0 * torch.atan2(numerator, denominator)


def equal_area_loss(points: Tensor, faces: Tensor, reference_areas: Tensor) -> Tensor:
    r"""``mean((area_i / area_i^0 - 1)^2)`` -- per-face area drift from the undeformed mesh.

    Relative to the **initial** per-face areas rather than to a uniform value: the lune's own
    faces legitimately differ in size (the pole fan vs the equator quads), and a uniform
    target would fight the base mesh itself.

    Exactly zero at the start, grows quadratically as faces shrink or stretch, and -- because
    the areas are *signed* -- a flipped face lands at ratio :math:`\approx -1` and is
    penalized at :math:`(-2)^2` rather than sailing through on its magnitude.
    """
    areas = spherical_face_areas(points, faces)
    return ((areas / reference_areas - 1.0) ** 2).mean()


def area_tail_loss(
    points: Tensor, faces: Tensor, reference_areas: Tensor, fraction: float = 0.05
) -> Tensor:
    r"""Mean squared area drift of the **worst** ``fraction`` of faces.

    Why the mean term alone cannot work, measured across two runs: with a weak mean
    (EA_W=2000) the boundary faces collapsed anyway -- a minority of degenerating faces
    dilutes into an average over 990 -- and with a strong mean (EA_W=5000) the *global*
    restoring force overpowered the shape signal and pulled the tile back to the lune
    (perimeter regressed 1.11 -> 1.04). The mean punishes legitimate whole-tile deformation
    and degeneracy with the same coin.

    This term separates them: every face drifting moderately (a genuine shape change) barely
    moves the top-5% statistic, while a few faces collapsing dominate it. So the barrier can
    be strong against degeneracy without taxing deformation.

    ``topk`` is differentiable -- gradients flow to the currently-worst faces, which is
    exactly where the pushback belongs.
    """
    drift = (spherical_face_areas(points, faces) / reference_areas - 1.0) ** 2
    k = max(1, int(len(drift) * fraction))
    return torch.topk(drift, k).values.mean()
