"""Deterministic planar shape phase: fit the tile outline to a target silhouette mask.

The planar counterpart of ``main_shape.py``, and the mechanism verification the
GEM-articulation plan rests on: the tile outline is carved by FULL-MESH deformation --
every vertex moves through the differentiable planar orbifold-Tutte solve over all edge
weights (``OTESolver``), the parameterization whose bijectivity holds at any weight
extremity. The loss is the same deterministic pyramid mask loss as the sphere
(``shape_target``), on the same analytic soft silhouette (``soft_silhouette``): the
planar tile's boundary polygon is just ``mapped[bdry]`` -- ``igl.boundary_loop`` output
that ``Escher.init_mesh_and_solver`` already stores, ordered, no camera, one fixed
affine map to pixels.

No renderer and no diffusion model anywhere in the loop; runs on CPU (GPU only
accelerates the distance field). The texture phase afterwards is upstream ``main.py``
with ``W_INIT_PATH`` + ``ONLY_TEXTURE_FROM_THIS_POINT: 0``.

Usage (inside the WSL venv)::

    python escher/main_shape_planar.py                    # configs/planar_shape.yaml
    python escher/main_shape_planar.py MASK_LOSS_WEIGHT=10000
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.geometry.sanity_checks import check_triangle_orientation
from escher.main import Escher, PATH, load_planar_args
from escher.shape_target import align_mask_to, mask_pyramid_loss, soft_iou
from escher.soft_silhouette import soft_polygon_mask


@dataclass
class PlanarShapeContext:
    """Everything fixed for the whole run."""

    loop: torch.Tensor  #: ordered boundary vertex indices (igl.boundary_loop)
    half_width: float  #: the frame shows [-h, h]^2 -- tile INSIDE with margins
    size: int
    tau: float
    device: torch.device
    ref_areas: torch.Tensor  #: signed face areas of the W=0 solve, for the drift reg
    perimeter0: float  #: boundary length of the W=0 solve, for the length reg


def planar_face_areas(mapped: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Signed triangle areas via the 2D cross product, differentiably."""
    a, b, c = mapped[faces[:, 0]], mapped[faces[:, 1]], mapped[faces[:, 2]]
    ab, ac = b - a, c - a
    return 0.5 * (ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0])


def build_planar_shape_run(args) -> Escher:
    """An ``Escher`` with mesh + solver only: no renderer, no guidance, no optimizer.

    The driver owns a plain Adam over ``W``. ``W`` starts at zero (sigmoid(0) = 0.5 ->
    uniform weights -> the undeformed tile), not upstream's ``randn`` -- the mask loss
    wants a deterministic, feasible start, same as every other phase in this project.
    """
    if isinstance(args.PROMPT, str):
        args.PROMPT = [args.PROMPT]  # init_mesh_and_solver reads len(PROMPT)
    from escher.misc.misc import seed_everything

    seed_everything(args.SEED)

    escher = Escher.__new__(Escher)
    escher.args = args
    escher.device = torch.device("cpu")  # the linear solve lives on CPU
    Path(args.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    escher.init_mesh_and_solver()
    if not args.get("W_INIT_PATH", None):
        # init_mesh_and_solver already applied W_INIT_PATH when given (resume); only a
        # fresh run gets the deterministic uniform start.
        with torch.no_grad():
            escher.W.zero_()
    return escher


def planar_to_pixels(points2: torch.Tensor, half_width: float, size: int) -> torch.Tensor:
    """``(n, 2)`` tile coordinates -> ``(n, 2)`` pixel ``(col, row)``, y flipped to
    image row order. A fixed affine map -- the planar analogue of the fixed camera."""
    col = (points2[:, 0] + half_width) / (2 * half_width) * size
    row = (half_width - points2[:, 1]) / (2 * half_width) * size
    return torch.stack([col, row], dim=-1)


def solve_tile(escher: Escher) -> torch.Tensor:
    """One differentiable planar Tutte solve -> ``(n, 2)`` vertices."""
    mapped, _, success = escher.solver.solve(escher.map_weights())
    if not success:
        print("  !! solver reported failure", flush=True)
    return mapped[:, :2]


def soft_alpha_planar(mapped: torch.Tensor, ctx: PlanarShapeContext) -> torch.Tensor:
    """``(1, H, W, 1)`` differentiable silhouette of the current tile."""
    px = planar_to_pixels(mapped[ctx.loop], ctx.half_width, ctx.size)
    px = px.to(device=ctx.device, dtype=torch.float32)
    return soft_polygon_mask(px, ctx.size, ctx.size, tau=ctx.tau).reshape(
        1, ctx.size, ctx.size, 1
    )


def load_target_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    import imageio.v2 as imageio

    img = imageio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    return (img / max(img.max(), 1e-9) > 0.5).astype(np.float32)


def prepare_target(escher: Escher, ctx: PlanarShapeContext, args) -> torch.Tensor:
    raw = load_target_mask(args.TARGET_MASK)
    with torch.no_grad():
        alpha0 = soft_alpha_planar(solve_tile(escher), ctx)[0, ..., 0].cpu().numpy()

    aligned, params, iou = align_mask_to(alpha0, raw)
    print(
        f"target aligned: scale {params['scale']:.3f}, angle {params['angle_deg']:+.1f} deg, "
        f"initial IoU {iou:.3f}"
    )

    import imageio.v2 as imageio

    out = Path(args.OUTPUT_DIR)
    imageio.imwrite(out / "target_aligned.png", (aligned * 255).astype(np.uint8))
    overlay = np.stack([aligned, alpha0, np.zeros_like(aligned)], axis=-1)
    imageio.imwrite(out / "target_overlay.png", (overlay * 255).astype(np.uint8))
    return torch.as_tensor(aligned, device=ctx.device)


def shape_step(escher, optimizer, target, ctx: PlanarShapeContext, args) -> dict:
    optimizer.zero_grad()
    mapped = solve_tile(escher)

    alpha = soft_alpha_planar(mapped, ctx)
    mask = args.MASK_LOSS_WEIGHT * mask_pyramid_loss(
        alpha, target, levels=args.MASK_PYRAMID_LEVELS
    )
    # Measured failure without these: the boundary grows HAIRY TENDRILS -- thin spikes
    # into the target's soft band that buy coarse-pyramid coverage with unpenalized
    # length (perimeter 8 -> 13.3 by step 100, IoU static). The sphere runs were
    # protected by their area barrier / chain regs; the plane needs its own:
    boundary = mapped[ctx.loop]
    perimeter = (boundary.roll(-1, 0) - boundary).norm(dim=-1).sum()
    reg = args.W_REGULARIZATION * (escher.W**2).sum()
    # Slack matters: the figure's outline is legitimately LONGER than the undeformed
    # square (hinging at perimeter0 exactly froze the run at IoU 0.685); the hinge only
    # exists to make runaway tendrils expensive.
    reg = reg + args.PERIMETER_WEIGHT * torch.relu(
        perimeter - args.PERIMETER_SLACK * ctx.perimeter0
    )
    if args.EQUAL_AREA_WEIGHT_2D > 0:
        areas = planar_face_areas(mapped, escher.faces)
        reg = reg + args.EQUAL_AREA_WEIGHT_2D * (
            (areas / ctx.ref_areas - 1.0) ** 2
        ).mean()
    loss = mask + reg.to(mask)
    loss.backward()
    optimizer.step()

    # Tripwire only: planar Tutte with positive weights is provably injective, so a
    # flip here means a solver failure, not an optimization event to project away.
    flips = 0
    try:
        check_triangle_orientation(mapped.detach(), escher.faces)
    except Exception:
        flips = 1

    boundary = mapped.detach()[ctx.loop]
    seg = (boundary.roll(-1, 0) - boundary).norm(dim=-1).sum()
    return {
        "loss": float(loss.detach()),
        "mask": float(mask.detach()),
        "iou": float(soft_iou(alpha.detach(), target)),
        "reg": float(reg.detach()),
        "perimeter": float(seg),
        "w_max": float(escher.W.detach().abs().max()),
        "flips": flips,
        "mapped": mapped,
    }


def log_metrics(output_dir: Path, iteration: int, info: dict) -> None:
    path = output_dir / "metrics.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write("step,loss,mask,iou,reg,perimeter,w_max,flips\n")
        f.write(
            f"{iteration},{info['loss']:.4f},{info['mask']:.4f},{info['iou']:.4f},"
            f"{info['reg']:.4f},{info['perimeter']:.4f},{info['w_max']:.3f},"
            f"{info['flips']}\n"
        )


def save_snapshot(escher, target, ctx: PlanarShapeContext, iteration: int, args) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        mapped = solve_tile(escher)
        alpha = soft_alpha_planar(mapped, ctx)

    achieved = alpha[0, ..., 0].cpu().numpy()
    tgt = target.cpu().numpy()
    overlay = np.stack([tgt, achieved, np.zeros_like(tgt)], axis=-1)

    panels = [(tgt, "target"), (achieved, "achieved"), (overlay, "overlay (R=target, G=achieved)")]
    fig, axes = plt.subplots(1, len(panels) + 1, figsize=(4.0 * (len(panels) + 1), 4.2))
    for ax, (img, title) in zip(axes, panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()

    # Interlocking check: the group images of the boundary, straight from the
    # constraint data -- the planar analogue of the tiled-sphere panel.
    ax = axes[-1]
    try:
        loop_np = ctx.loop.cpu().numpy()
        maps = escher.constraint_data.get_tiling(
            half_grid_size=1, vertices=mapped, sides=escher.constraint_data.sides
        )
        for m in maps:
            b = m.map(mapped)[loop_np].cpu().numpy()
            ax.fill(b[:, 0], b[:, 1], alpha=0.45, linewidth=0.8, edgecolor="k")
        ax.set_aspect("equal")
        ax.set_title("tiling", fontsize=10)
    except Exception as exc:  # get_tiling signatures vary per group; panel is optional
        ax.set_axis_off()
        ax.set_title(f"tiling panel unavailable: {type(exc).__name__}", fontsize=8)

    fig.suptitle(f"planar shape step {iteration}", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        Path(args.OUTPUT_DIR) / f"step_{iteration:05d}.png", dpi=110, bbox_inches="tight"
    )
    plt.close(fig)


def save_checkpoint(escher, iteration: int, args) -> Path:
    payload = {
        "iteration": iteration,
        "W": escher.W.detach().cpu(),
        "config": OmegaConf.to_container(args, resolve=True),
    }
    out = Path(args.OUTPUT_DIR)
    tagged = out / f"checkpoint_{iteration:06d}.pt"
    torch.save(payload, tagged)
    torch.save(payload, out / "checkpoint.pt")
    return tagged


def run_shape(args) -> dict:
    escher = build_planar_shape_run(args)
    with torch.no_grad():
        mapped0 = solve_tile(escher)
        loop_t = torch.as_tensor(np.asarray(escher.bdry), dtype=torch.long)
        b0 = mapped0[loop_t]
        ctx = PlanarShapeContext(
            loop=loop_t,
            half_width=float(args.PLANAR_FRAME_HALF_WIDTH),
            size=int(args.RENDER_SIZE),
            tau=float(args.MASK_TAU),
            device=torch.device(args.DEVICE),
            ref_areas=planar_face_areas(mapped0, escher.faces).detach(),
            perimeter0=float((b0.roll(-1, 0) - b0).norm(dim=-1).sum()),
        )
    target = prepare_target(escher, ctx, args)
    optimizer = torch.optim.Adam([escher.W], lr=float(args.LR_SHAPE))

    start = time.time()
    info: dict = {}
    for iteration in range(int(args.SHAPE_STEPS) + 1):
        info = shape_step(escher, optimizer, target, ctx, args)

        if iteration % 10 == 0:
            log_metrics(Path(args.OUTPUT_DIR), iteration, info)
            per_step = (time.time() - start) / max(iteration, 1)
            print(
                f"step {iteration:5d} | loss {info['loss']:9.1f} | "
                f"mask {info['mask']:8.1f} | iou {info['iou']:.4f} | "
                f"reg {info['reg']:8.1f} | perim {info['perimeter']:7.3f} | "
                f"|W| {info['w_max']:5.2f} | flp {info['flips']} | "
                f"{per_step:5.2f} s/step",
                flush=True,
            )

        if iteration % args.VISUALIZATION_FREQ == 0:
            save_snapshot(escher, target, ctx, iteration, args)
            save_checkpoint(escher, iteration, args)

    save_checkpoint(escher, int(args.SHAPE_STEPS), args)
    print(f"done in {(time.time() - start) / 60:.1f} min -> {args.OUTPUT_DIR}")
    return {"iou": info["iou"], "flips": info["flips"], "output_dir": args.OUTPUT_DIR}


def main() -> None:
    cli = OmegaConf.from_cli()
    if "CONF_FILE" not in cli:
        cli["CONF_FILE"] = "configs/planar_shape.yaml"
    run_shape(load_planar_args(cli))


if __name__ == "__main__":
    main()
