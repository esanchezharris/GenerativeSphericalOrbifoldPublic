"""Offline background scrub: the guaranteed floor for the zero-filler render gate.

Finds texels that would read as background on the tinted sphere -- low-chroma
brights (fixed points of the hue-rotation tint) and near-init colors (mip-leaked
gutter) -- and recolors each with its NEAREST non-offending sampled texel, the
same mechanism as the UV gutter fill. Deterministic, no diffusion, reversible:
writes ``checkpoint_polished.pt`` beside the input and never touches the
original.

Usage (inside the WSL venv)::

    python escher/texture_polish.py output/texF_fish/checkpoint.pt

Then render/measure the polished artifact::

    python escher/render_final.py output/texF_fish/checkpoint_polished.pt TINT=1
    python escher/metrics_background.py output/texF_fish/checkpoint_polished.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

from escher.metrics_background import CHROMA_MAX, DEFAULT_INIT, INIT_L2, LUMA_MIN
from escher.rendering.texture_mask import uv_valid_mask


def offending_texels(
    texture: np.ndarray, valid: np.ndarray, init: np.ndarray
) -> np.ndarray:
    """Bool mask of sampled texels that would read as background when rendered."""
    chroma = texture.max(-1) - texture.min(-1)
    luma = texture.mean(-1)
    white_like = (chroma < CHROMA_MAX) & (luma > LUMA_MIN)
    init_like = np.linalg.norm(texture - init, axis=-1) < INIT_L2
    return (white_like | init_like) & valid


def polish(checkpoint: str | Path) -> Path:
    checkpoint = Path(checkpoint)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = state["config"]
    texture = state["texture"].numpy().copy()

    from escher.render_final import load_run

    escher, _ = load_run(checkpoint)
    valid = uv_valid_mask(
        escher.mesh.uv, escher.mesh.faces, int(cfg["TEXTURE_RESOLUTION"])
    )

    init_color = cfg.get("TEXTURE_INIT_COLOR")
    init = np.asarray(
        DEFAULT_INIT if init_color is None else list(init_color), dtype=np.float32
    )

    bad = offending_texels(texture, valid, init)
    good = valid & ~bad
    if not good.any():
        raise ValueError("no non-offending sampled texels to recolor from")
    print(
        f"{int(bad.sum())} offending texels of {int(valid.sum())} sampled "
        f"({100 * bad.sum() / max(valid.sum(), 1):.2f}%)"
    )

    from scipy import ndimage

    idx = ndimage.distance_transform_edt(
        ~good, return_distances=False, return_indices=True
    )
    polished = texture.copy()
    polished[bad] = texture[idx[0][bad], idx[1][bad]]

    # The recolor source set contains no offending texels by construction, so the
    # scrub is guaranteed complete in one pass; verify anyway.
    still = offending_texels(polished, valid, init)
    assert not still.any(), "scrub left offending texels -- classifier inconsistency"

    state["texture"] = torch.as_tensor(polished)
    out = checkpoint.with_name(checkpoint.stem + "_polished.pt")
    torch.save(state, out)
    print(f"wrote {out}")
    return out


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit(f"usage: {sys.argv[0]} <checkpoint.pt>")
    polish(sys.argv[1])


if __name__ == "__main__":
    main()
