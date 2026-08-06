"""Generate figure-silhouette targets with FULL Stable Diffusion denoising (not SDS).

The deterministic shape phase needs one clean binary mask of the figure. SDS never
produces one -- it only ever nudges pixels -- but the very same weights run as a normal
txt2img sampler produce crisp silhouettes in seconds. Generate a batch of candidates,
binarize each (``shape_target.binarize_mask``), and write a contact sheet; the largest
candidate is auto-selected as ``target.npy``, and a second invocation with ``CHOOSE=i``
overrides that choice without touching the GPU.

Usage (inside the WSL venv)::

    python escher/make_target.py                 # generate N candidates + contact sheet
    python escher/make_target.py CHOOSE=3        # re-pick: candidate 3 -> target.npy
"""

from __future__ import annotations

from pathlib import Path

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import OmegaConf

from escher.shape_target import binarize_mask

DEFAULTS = OmegaConf.create(
    {
        "PROMPT": (
            "a plain solid black silhouette of a gingerbread man, thick rounded limbs, "
            "white background, minimal flat logo, centered, full body"
        ),
        "NEGATIVE": "photo, texture, shading, gradient, outline only, cropped, text",
        "MODEL": "Manojb/stable-diffusion-2-1-base",
        "N": 8,
        "SEED": 0,
        "STEPS": 30,
        "GUIDANCE": 7.5,
        "SIZE": 512,
        "OUT_DIR": "assets/targets/gingerbread",
        "CHOOSE": -1,  # >= 0: skip generation, just re-point target.npy
    }
)


def write_choice(out_dir: Path, index: int) -> None:
    mask = imageio.imread(out_dir / f"target_{index:02d}.png").astype(np.float32)
    mask = (mask / mask.max() > 0.5).astype(np.float32)
    np.save(out_dir / "target.npy", mask)
    print(f"target.npy <- candidate {index} (area {mask.mean():.3f} of frame)")


def generate(args) -> None:
    import torch
    from diffusers import StableDiffusionPipeline

    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    pipe = StableDiffusionPipeline.from_pretrained(
        args.MODEL, torch_dtype=torch.float16, safety_checker=None,
        requires_safety_checker=False,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)

    candidates = []  # (index, raw, mask-or-None, area)
    for i in range(args.N):
        gen = torch.Generator(device="cuda").manual_seed(args.SEED + i)
        raw = pipe(
            args.PROMPT,
            negative_prompt=args.NEGATIVE,
            num_inference_steps=args.STEPS,
            guidance_scale=args.GUIDANCE,
            height=args.SIZE,
            width=args.SIZE,
            generator=gen,
        ).images[0]
        raw = np.asarray(raw)
        imageio.imwrite(out_dir / f"target_{i:02d}_raw.png", raw)
        try:
            mask = binarize_mask(raw)
        except ValueError:
            mask = None  # sampler produced no dark figure; keep the slot for the sheet
        if mask is not None:
            imageio.imwrite(out_dir / f"target_{i:02d}.png", (mask * 255).astype(np.uint8))
        area = float(mask.mean()) if mask is not None else 0.0
        candidates.append((i, raw, mask, area))
        print(f"candidate {i}: seed {args.SEED + i}, figure area {area:.3f}", flush=True)

    fig, axes = plt.subplots(2, args.N, figsize=(2.2 * args.N, 4.8))
    for i, raw, mask, area in candidates:
        axes[0, i].imshow(raw)
        axes[0, i].set_title(f"{i}  (area {area:.2f})", fontsize=9)
        if mask is not None:
            axes[1, i].imshow(mask, cmap="gray")
        for row in (0, 1):
            axes[row, i].set_axis_off()
    fig.suptitle(f'"{args.PROMPT}"', fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / "contact_sheet.png", dpi=120, bbox_inches="tight")
    print(f"wrote {out_dir / 'contact_sheet.png'}")

    # Auto-pick the largest figure; CHOOSE=i on a re-run overrides without the GPU.
    valid = [(i, a) for i, _, m, a in candidates if m is not None]
    if not valid:
        raise SystemExit("no candidate binarized cleanly -- adjust PROMPT and rerun")
    best = max(valid, key=lambda t: t[1])[0]
    write_choice(out_dir, best)


def main() -> None:
    args = OmegaConf.merge(DEFAULTS, OmegaConf.from_cli())
    if args.CHOOSE >= 0:
        write_choice(Path(args.OUT_DIR), int(args.CHOOSE))
    else:
        generate(args)


if __name__ == "__main__":
    main()
