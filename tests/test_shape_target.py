"""CPU validation of the target-silhouette utilities (escher/shape_target.py).

The alignment and loss functions drive the deterministic shape phase, so their
conventions are pinned here on synthetic masks where the right answer is known exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy import ndimage

from escher.shape_target import (
    align_mask_to,
    binarize_mask,
    mask_pyramid_loss,
    soft_iou,
)


def disk(shape, center, radius):
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    return ((coords - np.asarray(center)) ** 2).sum(-1) <= radius**2


# ---------------------------------------------------------------------- binarize
def test_binarize_keeps_largest_component_and_fills_holes():
    img = np.ones((128, 128))
    img[disk(img.shape, (64, 64), 30)] = 0.1  # the figure
    img[disk(img.shape, (64, 64), 6)] = 1.0  # a white "icing" hole inside it
    img[disk(img.shape, (20, 100), 4)] = 0.1  # speckle far away

    mask = binarize_mask(img)
    assert mask.dtype == np.float32
    assert mask[64, 64] == 1.0, "interior hole must be filled"
    assert mask[20, 100] == 0.0, "small stray component must be dropped"
    area = mask.sum()
    assert abs(area - np.pi * 30**2) / (np.pi * 30**2) < 0.05


def test_binarize_explicit_threshold_overrides_otsu():
    img = np.ones((32, 32))
    img[disk(img.shape, (16, 16), 8)] = 0.4
    assert binarize_mask(img, threshold=0.5).sum() > 0
    with pytest.raises(ValueError):
        binarize_mask(img, threshold=0.1)  # nothing darker than 0.1


def test_binarize_rejects_blank_image():
    with pytest.raises(ValueError):
        binarize_mask(np.ones((16, 16)))


# ---------------------------------------------------------------------- alignment
def ellipse(shape, center, a, b):
    coords = np.stack(np.meshgrid(*[np.arange(s) for s in shape], indexing="ij"), -1)
    d = coords - np.asarray(center)
    return ((d[..., 0] / a) ** 2 + (d[..., 1] / b) ** 2 <= 1.0).astype(np.float32)


def test_align_identity_is_near_perfect():
    ref = ellipse((128, 128), (64, 64), 40, 25)
    aligned, params, iou = align_mask_to(ref, ref.copy())
    assert iou > 0.95
    assert params["scale"] == pytest.approx(1.0, abs=0.1)


def test_align_recovers_a_known_similarity_transform():
    ref = ellipse((128, 128), (64, 64), 40, 25)

    # Build the target by pushing the reference through a known similarity: rotate 10
    # degrees, scale 1.15, shift (+5, -8). (affine_transform pulls, so we hand it the
    # inverse pieces.)
    theta = np.deg2rad(10.0)
    s = 1.15
    rot = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    M = rot.T / s
    c = np.array([64.0, 64.0])
    shifted_center = c + np.array([5.0, -8.0])
    target = ndimage.affine_transform(
        ref, M, offset=c - M @ shifted_center, order=0, cval=0.0
    )
    assert 0.2 < _iou_np(target, ref) < 0.9, "test premise: target starts misaligned"

    aligned, _, iou = align_mask_to(ref, target)
    assert iou > 0.85
    assert _iou_np(aligned, ref) == pytest.approx(iou, abs=1e-6)


def _iou_np(a, b):
    a = a > 0.5
    b = b > 0.5
    return np.logical_and(a, b).sum() / max(np.logical_or(a, b).sum(), 1)


def test_match_area_alignment_equalizes_areas():
    """Area-matched mode: the aligned target's area must equal the reference's, because
    the tile's area is conserved and a smaller target caps IoU structurally."""
    ref = ellipse((128, 128), (64, 64), 40, 30)
    small = ellipse((128, 128), (64, 64), 20, 15)  # a quarter of the area
    aligned, _, _ = align_mask_to(ref, small, match_area=True)
    ref_area = (ref > 0.5).sum()
    assert abs((aligned > 0.5).sum() - ref_area) / ref_area < 0.05


# ---------------------------------------------------------------------- soft IoU
def test_soft_iou_hand_computed():
    target = torch.ones(8, 8)
    alpha = torch.zeros(1, 8, 8, 1)
    alpha[0, :, :4, 0] = 1.0  # left half
    assert soft_iou(alpha, target).item() == pytest.approx(0.5)
    assert soft_iou(torch.ones(2, 8, 8, 1), target).item() == pytest.approx(1.0)


# ---------------------------------------------------------------------- pyramid loss
def soft_disk(radius: torch.Tensor, size: int = 64, edge: float = 1.5) -> torch.Tensor:
    """(1, H, W, 1) alpha with a differentiable soft edge, standing in for a render."""
    ax = torch.arange(size, dtype=torch.float64)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    dist = torch.sqrt((yy - size / 2) ** 2 + (xx - size / 2) ** 2)
    return torch.sigmoid((radius - dist) / edge).reshape(1, size, size, 1)


def test_pyramid_loss_zero_on_identical_positive_on_shifted():
    t = torch.as_tensor(disk((64, 64), (32, 32), 12).astype(np.float32))
    same = t.reshape(1, 64, 64, 1)
    assert mask_pyramid_loss(same, t).item() == pytest.approx(0.0, abs=1e-12)

    near = torch.as_tensor(disk((64, 64), (32, 36), 12).astype(np.float32))
    far = torch.as_tensor(disk((64, 64), (32, 52), 12).astype(np.float32))
    l_near = mask_pyramid_loss(near.reshape(1, 64, 64, 1), t).item()
    l_far = mask_pyramid_loss(far.reshape(1, 64, 64, 1), t).item()
    assert 0 < l_near < l_far, "loss must grow with misalignment"


def test_pyramid_loss_gradient_points_toward_the_target():
    target = torch.as_tensor(disk((64, 64), (32, 32), 20).astype(np.float64))
    r = torch.tensor(15.0, dtype=torch.float64, requires_grad=True)
    loss = mask_pyramid_loss(soft_disk(r), target)
    loss.backward()
    assert r.grad is not None and r.grad.abs() > 0
    assert r.grad < 0, "radius below target: growing it must reduce the loss"

    closer = mask_pyramid_loss(soft_disk(torch.tensor(19.0, dtype=torch.float64)), target)
    assert closer.item() < loss.item()
