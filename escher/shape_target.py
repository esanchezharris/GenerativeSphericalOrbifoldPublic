"""Target-silhouette utilities for the deterministic shape phase.

The shape phase replaces SDS as the outline driver: a clean figure silhouette is
generated ONCE (full denoising, ``make_target.py``), reduced to a binary mask here, and
the free boundary points are then optimized so the tile's rendered alpha matches it.
That trades SDS's noisy, unrepeatable silhouette signal for a deterministic loss with
exact gradients -- the measured reason every SDS-driven run stalled near a 1.13x outline.

Everything in this module is CPU-only numpy/scipy/torch; nothing imports the renderer
or the diffusion stack, so it is fully unit-testable offline.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage
from torch import Tensor

__all__ = ["binarize_mask", "align_mask_to", "soft_iou", "hard_iou", "mask_pyramid_loss"]


def _to_grayscale(image: np.ndarray) -> np.ndarray:
    """Any (H, W[, C]) uint8/float image -> float64 grayscale in [0, 1]."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim == 3:
        img = img[..., :3].mean(axis=-1)
    if img.max() > 1.0:
        img = img / 255.0
    return img


def _otsu_threshold(gray: np.ndarray) -> float:
    """Classic Otsu: the threshold maximizing between-class variance, 256 bins."""
    hist, edges = np.histogram(gray, bins=256, range=(0.0, 1.0))
    p = hist.astype(np.float64) / max(hist.sum(), 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    w0 = np.cumsum(p)
    w1 = 1.0 - w0
    mu0 = np.cumsum(p * centers) / np.clip(w0, 1e-12, None)
    mu_total = (p * centers).sum()
    mu1 = (mu_total - np.cumsum(p * centers)) / np.clip(w1, 1e-12, None)
    variance = w0 * w1 * (mu0 - mu1) ** 2
    # The UPPER edge of the winning bin, so values inside that bin count as foreground
    # (a delta-function histogram puts the whole dark class in one bin; its center would
    # exclude it).
    return float(edges[int(np.argmax(variance)) + 1])


def binarize_mask(
    image: np.ndarray, threshold: float | None = None, smooth_radius: int = 0
) -> np.ndarray:
    """A generated silhouette image -> a clean binary figure mask (float32, {0, 1}).

    Convention: the figure is DARK on a light background (matching the silhouette
    prompt). Diffusion output is never perfectly clean, so after thresholding keep only
    the largest connected component (drops speckle and stray marks) and fill interior
    holes (icing details rendered as white).

    ``smooth_radius`` > 0 additionally closes then opens the mask with a disk of that
    radius. Diffusion silhouettes arrive with ragged, pixel-scale jaggies along the
    contour -- clearly visible on the fish target's tail and lower edge -- and the shape
    phase fits them faithfully, so they come back as sawtooth serrations on the tile
    outline. Closing first fills notches, opening then removes spurs, and the pair leaves
    the figure's overall extent alone. Keep the radius small: it will erode genuinely thin
    features (a tail fin, a limb) before it runs out of jaggies to remove.
    """
    gray = _to_grayscale(image)
    if threshold is None:
        threshold = _otsu_threshold(gray)
    mask = gray < threshold
    if not mask.any():
        raise ValueError("binarize_mask: no foreground below the threshold")

    labels, n = ndimage.label(mask)
    if n > 1:
        sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
        mask = labels == (1 + int(np.argmax(sizes)))
    mask = ndimage.binary_fill_holes(mask)

    if smooth_radius > 0:
        r = int(smooth_radius)
        yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
        disk = (xx**2 + yy**2) <= r * r
        mask = ndimage.binary_closing(mask, structure=disk)
        mask = ndimage.binary_opening(mask, structure=disk)
        # Opening can split off a fragment; keep the body, and re-fill any notch the
        # closing opened up.
        labels, n = ndimage.label(mask)
        if n > 1:
            sizes = ndimage.sum_labels(mask, labels, index=np.arange(1, n + 1))
            mask = labels == (1 + int(np.argmax(sizes)))
        mask = ndimage.binary_fill_holes(mask)
        if not mask.any():
            raise ValueError(f"binarize_mask: smooth_radius {r} erased the figure")
    return mask.astype(np.float32)


def _moments(mask: np.ndarray) -> tuple[np.ndarray, float]:
    """(centroid (row, col), RMS radius), intensity-weighted so soft alphas work too."""
    w = np.asarray(mask, dtype=np.float64)
    total = w.sum()
    if total <= 0:
        raise ValueError("empty mask")
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in w.shape], indexing="ij"), -1)
    centroid = (w[..., None] * coords).sum(axis=(0, 1)) / total
    r2 = ((coords - centroid) ** 2).sum(axis=-1)
    return centroid, float(np.sqrt((w * r2).sum() / total))


def hard_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Binary IoU after thresholding both masks at 0.5.

    This is the number that corresponds to "filler" on the sphere. :func:`soft_iou`
    compares a sigmoid-softened alpha against a binary target, so a mathematically
    perfect carve reports ~0.93 at ``MASK_TAU`` 3 -- and the shortfall *grows with
    perimeter*, biasing any soft-IoU ranking against articulated outlines. With the
    sigmoid mask, alpha > 0.5 exactly iff the pixel center is inside the polygon, so
    thresholding removes the tau dependence entirely.
    """
    a = np.asarray(a) > 0.5
    b = np.asarray(b) > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(union, 1))


_hard_iou = hard_iou  # internal alias, kept for call sites below


def align_mask_to(
    reference: np.ndarray,
    target: np.ndarray,
    scales: tuple[float, float, int] = (0.6, 1.3, 15),
    # Full circle at the same 7.5-degree step. A half-turn range assumes the tile has a
    # symmetry that makes +t and t-180 equivalent, and the kites do not: the (2,3,4)
    # carve selected exactly +90.0, i.e. it was pinned at the edge of its own grid and
    # the true optimum lay outside. (The range was widened once before, from +-30, for
    # the same reason -- an alignment sitting on a grid boundary is never a fit, it is a
    # truncation.)
    angles_deg: tuple[float, float, int] = (-180.0, 180.0, 49),
    match_area: bool = False,
) -> tuple[np.ndarray, dict, float]:
    """Place ``target`` over ``reference`` by a similarity transform, maximizing IoU.

    ``reference`` is the undeformed tile's rendered alpha; among all placements of the
    figure, the one overlapping the tile most is the most *reachable* one -- the
    boundary optimization then only has to cover the residual. Moments give the initial
    translation and scale; a coarse grid over (scale multiplier, rotation) refines
    them, with the translation re-derived from centroids at each candidate. One-time,
    CPU, seconds.

    ``match_area=True`` replaces the scale search with the single AREA-MATCHED scale
    (target area == reference area), searching only rotation. This matters whenever the
    tile's area is conserved -- which is every orbifold parameterization here: the
    tiling must cover its surface, so the tile's area is pinned at ``|surface|/|G|``
    and a smaller target caps IoU at ``target_area / tile_area`` STRUCTURALLY.
    Measured: the planar torus carve converged to 0.7106 with the free-scale
    alignment's area ratio at exactly 0.71.

    Returns ``(aligned_mask, {"scale", "angle_deg", "shift"}, best_iou)`` with
    ``aligned_mask`` sampled on the reference's grid.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    c_ref, r_ref = _moments(ref)
    c_tgt, r_tgt = _moments(tgt)
    base_scale = r_ref / r_tgt

    if match_area:
        area_scale = float(np.sqrt((ref > 0.5).sum() / max((tgt > 0.5).sum(), 1)))
        scale_grid = [area_scale / base_scale]  # one multiplier: exact area match
    else:
        scale_grid = np.linspace(*scales)

    best = (None, None, -1.0)
    for mult in scale_grid:
        for angle in np.linspace(*angles_deg):
            s = base_scale * mult
            theta = np.deg2rad(angle)
            rot = np.array(
                [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
            )
            # scipy's affine_transform pulls: output[y] = input[M @ y + offset], so M is
            # the INVERSE map (reference frame -> target frame).
            M = rot.T / s
            offset = c_tgt - M @ c_ref
            aligned = ndimage.affine_transform(
                tgt, M, offset=offset, output_shape=ref.shape, order=0, cval=0.0
            )
            iou = _hard_iou(aligned, ref)
            if iou > best[2]:
                best = (aligned, {"scale": s, "angle_deg": float(angle)}, iou)

    aligned, params, iou = best
    params["shift"] = tuple((c_ref - c_tgt).tolist())
    return aligned.astype(np.float32), params, iou


def soft_iou(alpha: Tensor, target: Tensor) -> Tensor:
    """Soft intersection-over-union of rendered alpha vs the target mask, in [0, 1].

    The *metric* (logged, used for sweep selection), not the loss -- see
    :func:`mask_pyramid_loss` for why the loss is a pyramid instead.
    """
    a = alpha.reshape(alpha.shape[0], -1)
    t = target.reshape(1, -1).to(a)
    inter = (a * t).sum(dim=1)
    union = (a + t - a * t).sum(dim=1)
    return (inter / union.clamp(min=1e-9)).mean()


def mask_pyramid_loss(alpha: Tensor, target: Tensor, levels: int = 5) -> Tensor:
    """Multi-scale MSE between rendered alpha ``(B, H, W, 1)`` and the target ``(H, W)``.

    The pyramid helps, but NOT for the reason originally recorded here. That rationale --
    coarse levels reaching misalignments the fine levels cannot -- was written for the
    rasterized alpha this loss was first built against, where pooling hard 0/1 pixels
    genuinely manufactures a coverage gradient. It does not carry over to the analytic
    ``soft_polygon_mask``, and long-range reach was never the mechanism anyway: that
    mask's soft band is centered on the MOVING polygon, so every boundary vertex always
    sits in its own band and always feels whether the target is 1 or 0 just outside it,
    however far the target's structure extends.

    What the pyramid actually does is amplify. Measured on the real target from the
    area-matched start (196 boundary vertices, 84 of them more than 20 px from the target
    outline): peak vertex gradient rises monotonically with depth -- 7.6e-2 (1 level),
    2.3e-1 (3), 4.0e-1 (5), 6.0e-1 (7), 6.1e-1 (9) -- while the share of gradient
    magnitude carried by those far vertices barely moves (45% -> 52%). Raising ``tau``
    instead *lowers* peak gradient, which is why there is no tau schedule here.
    """
    a = alpha.permute(0, 3, 1, 2)  # (B, 1, H, W)
    t = target.to(a).reshape(1, 1, *target.shape).expand(a.shape[0], -1, -1, -1)
    loss = a.new_zeros(())
    for _ in range(levels):
        loss = loss + F.mse_loss(a, t)
        if min(a.shape[-2:]) < 2:
            break
        a = F.avg_pool2d(a, 2)
        t = F.avg_pool2d(t, 2)
    return loss
