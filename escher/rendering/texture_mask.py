"""Which texels does the mesh actually sample, and how to fill the rest.

The kite UV layout covers roughly 40% of the texture square; the other ~60% of
texels are never rasterized directly, stay at the flat init color, and leak into
every minified view through the differentiable mip chain
(``dr.texture(..., linear-mipmap-linear, max_mip_level=9)``) -- measured as 1.17%
of the shipped sphere's pixels reading as init-tan. The classic fix is UV gutter
fill: flood the unsampled texels with their nearest sampled color so mip
averages stay inside the figure's palette.

Row convention matches the renderer: ``v`` indexes texture rows top-down (the
OBJ export writes ``1 - v`` because OBJ's v axis points up).
"""

from __future__ import annotations

import numpy as np


def uv_valid_mask(uv: np.ndarray, faces: np.ndarray, resolution: int) -> np.ndarray:
    """``(R, R)`` bool: texels covered by some UV triangle, dilated by one texel.

    The one-texel dilation covers bilinear taps that straddle the kite edge.
    """
    uv = np.asarray(uv, dtype=np.float64)
    mask = np.zeros((resolution, resolution), dtype=bool)

    # Texel centers in UV space.
    centers = (np.arange(resolution) + 0.5) / resolution

    for tri in np.asarray(faces):
        a, b, c = uv[tri[0]], uv[tri[1]], uv[tri[2]]
        lo = np.floor(np.minimum(np.minimum(a, b), c) * resolution - 1).astype(int)
        hi = np.ceil(np.maximum(np.maximum(a, b), c) * resolution + 1).astype(int)
        lo = np.clip(lo, 0, resolution - 1)
        hi = np.clip(hi, 0, resolution - 1)
        us = centers[lo[0] : hi[0] + 1]
        vs = centers[lo[1] : hi[1] + 1]
        if us.size == 0 or vs.size == 0:
            continue
        uu, vv = np.meshgrid(us, vs, indexing="xy")  # rows = v, cols = u
        # Barycentric sign test, tolerant of either winding.
        d = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
        if abs(d) < 1e-15:
            continue
        w0 = ((b[1] - c[1]) * (uu - c[0]) + (c[0] - b[0]) * (vv - c[1])) / d
        w1 = ((c[1] - a[1]) * (uu - c[0]) + (a[0] - c[0]) * (vv - c[1])) / d
        w2 = 1.0 - w0 - w1
        eps = -1e-9
        inside = (w0 >= eps) & (w1 >= eps) & (w2 >= eps)
        mask[lo[1] : hi[1] + 1, lo[0] : hi[0] + 1] |= inside

    from scipy import ndimage

    return ndimage.binary_dilation(mask, iterations=1)


def gutter_fill(texture: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Fill invalid texels with the color of their NEAREST valid texel.

    Exact nearest-neighbor via the indexed distance transform -- one shot, no
    iteration, and valid texels are provably untouched.
    """
    from scipy import ndimage

    valid = np.asarray(valid, dtype=bool)
    if valid.all():
        return np.array(texture, copy=True)
    idx = ndimage.distance_transform_edt(
        ~valid, return_distances=False, return_indices=True
    )
    return np.asarray(texture)[idx[0], idx[1]]
