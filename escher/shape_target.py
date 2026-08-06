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

__all__ = ["binarize_mask", "align_mask_to", "soft_iou", "mask_pyramid_loss"]


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


def binarize_mask(image: np.ndarray, threshold: float | None = None) -> np.ndarray:
    """A generated silhouette image -> a clean binary figure mask (float32, {0, 1}).

    Convention: the figure is DARK on a light background (matching the silhouette
    prompt). Diffusion output is never perfectly clean, so after thresholding keep only
    the largest connected component (drops speckle and stray marks) and fill interior
    holes (icing details rendered as white).
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


def _hard_iou(a: np.ndarray, b: np.ndarray) -> float:
    a = a > 0.5
    b = b > 0.5
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / max(union, 1))


def align_mask_to(
    reference: np.ndarray,
    target: np.ndarray,
    scales: tuple[float, float, int] = (0.6, 1.3, 15),
    angles_deg: tuple[float, float, int] = (-90.0, 90.0, 25),
) -> tuple[np.ndarray, dict, float]:
    """Place ``target`` over ``reference`` by a similarity transform, maximizing IoU.

    ``reference`` is the undeformed tile's rendered alpha; among all placements of the
    figure, the one overlapping the tile most is the most *reachable* one -- the
    boundary optimization then only has to cover the residual. Moments give the initial
    translation and scale; a coarse grid over (scale multiplier, rotation) refines
    them, with the translation re-derived from centroids at each candidate. One-time,
    CPU, seconds.

    Returns ``(aligned_mask, {"scale", "angle_deg", "shift"}, best_iou)`` with
    ``aligned_mask`` sampled on the reference's grid.
    """
    ref = np.asarray(reference, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    c_ref, r_ref = _moments(ref)
    c_tgt, r_tgt = _moments(tgt)
    base_scale = r_ref / r_tgt

    best = (None, None, -1.0)
    for mult in np.linspace(*scales):
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

    The pyramid is load-bearing, not a nicety: nvdiffrast's alpha carries gradients only
    in the antialiased EDGE pixels, so a plain per-pixel loss is nearly gradient-dead
    whenever the rendered and target boundaries are more than a pixel apart. Average
    pooling widens the soft edge at each level, so coarse levels see misalignments the
    fine levels cannot, and their gradients point the right way from far away.
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
