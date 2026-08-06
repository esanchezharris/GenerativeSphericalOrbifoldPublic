"""Deterministic shape phase: fit the tile outline to a target silhouette mask.

The SDS silhouette pass was believed to drive the outline; a gradient probe showed it
never did -- the rasterizer's alpha (and the whole RGB+alpha composite) return EXACTLY
zero vertex gradients under a uniform texture, so every outline change in the SDS runs
came from the textured pass's texture-slide gradients plus noise. This driver replaces
that non-signal with a real one, twice over:

1. The target is a figure mask generated ONCE by full denoising (``make_target.py``),
   aligned over the undeformed tile -- deterministic, no SDS noise.
2. The tile silhouette is computed analytically (``soft_silhouette``): project the
   boundary loop with the renderer's own camera conventions, soft-fill the polygon.
   Dense exact gradients, and NO renderer in the optimization loop at all -- the shape
   phase runs on CPU; the GPU is only used by snapshots' pretty wide views.

All validity machinery carries over unchanged: fold rejection (``ensure_valid_shape``),
the margin hinge, chain spacing/smoothness, the equal-area terms. Checkpoints are
ordinary ``SphereEscher`` checkpoints, so the texture phase resumes one with the
existing ``RESUME=`` path.

Usage (inside the WSL venv)::

    python escher/main_shape.py                          # single run, sphere_shape.yaml
    python escher/main_shape.py "SWEEP_K=[4,6]"          # k sweep, pick by IoU
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.main_sphere import PATH, SphereEscher
from escher.rendering.camera import orbit_views, perspective, tile_centric_views
from escher.shape_target import align_mask_to, mask_pyramid_loss, soft_iou
from escher.soft_silhouette import boundary_loop, project_to_pixels, soft_polygon_mask
from escher.OTE.core.spherical.regularizers import (
    area_margin_loss,
    area_tail_loss,
    equal_area_loss,
)
from escher.geometry.spherical_sanity_checks import signed_solid_angles


@dataclass
class ShapeContext:
    """Everything fixed for the whole run: camera, projection, boundary order."""

    mv: torch.Tensor  # (1, 4, 4), computed once from the UNDEFORMED mesh
    proj: torch.Tensor  # (4, 4)
    loop: torch.Tensor  # ordered boundary vertex indices
    size: int
    tau: float


def build_shape_run(args) -> SphereEscher:
    """A ``SphereEscher`` with geometry and parameters but NO guidance model.

    The ``__new__`` construction is the same one ``render_final.load_run`` and the CPU
    tests use; ``__init__`` is skipped, so the seeding it normally does happens here.
    """
    # Boundary mode drives ~38 outline points directly; weights mode drives ALL edge
    # weights through the implicit solve -- the GEM full-mesh parameterization, with
    # far more articulation capacity. Both use the same mask loss and metrics.
    torch.manual_seed(args.SEED)
    np.random.seed(args.SEED)

    escher = SphereEscher.__new__(SphereEscher)
    escher.args = args
    escher.device = torch.device(args.DEVICE)
    escher.output_dir = Path(args.OUTPUT_DIR)
    escher.output_dir.mkdir(parents=True, exist_ok=True)
    escher._init_geometry()
    escher._init_parameters()
    return escher


def fixed_tile_camera(escher: SphereEscher, args) -> torch.Tensor:
    """One ``(1, 4, 4)`` view of the fundamental domain, computed ONCE.

    From the UNDEFORMED ``mesh.points``, with zero jitter. Re-deriving the camera from
    the current deformed points every step (what the SDS render path does) would slide
    the fixed target under the tile as it deforms, corrupting the loss.
    """
    centers = escher.solo_tiler.tile_centers(escher.mesh.points)
    return tile_centric_views(
        torch.as_tensor(centers, dtype=torch.float32),
        1,
        distance=float(args.SHAPE_CAMERA_DISTANCE),
        angular_jitter_deg=0.0,
    )


def make_context(escher: SphereEscher, args) -> ShapeContext:
    return ShapeContext(
        mv=fixed_tile_camera(escher, args),
        proj=perspective(fovy_deg=float(args.CAMERA_FOV)),
        loop=torch.as_tensor(boundary_loop(escher.mesh), dtype=torch.long),
        size=int(args.RENDER_SIZE),
        tau=float(args.MASK_TAU),
    )


def soft_alpha(escher: SphereEscher, points: torch.Tensor, ctx: ShapeContext) -> torch.Tensor:
    """``(1, H, W, 1)`` differentiable silhouette of the current tile."""
    px = project_to_pixels(points[ctx.loop], ctx.mv, ctx.proj, ctx.size, ctx.size)
    # float32 on the run device for the (pixels x segments) field; gradients flow back
    # through the cast into the float64 solve.
    px = px.to(device=escher.device, dtype=torch.float32)
    mask = soft_polygon_mask(px, ctx.size, ctx.size, tau=ctx.tau)
    return mask.reshape(1, ctx.size, ctx.size, 1)


def load_target_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path).astype(np.float32)
    import imageio.v2 as imageio

    img = imageio.imread(path).astype(np.float32)
    if img.ndim == 3:
        img = img[..., :3].mean(-1)
    return (img / max(img.max(), 1e-9) > 0.5).astype(np.float32)


def prepare_target(escher: SphereEscher, ctx: ShapeContext, args) -> torch.Tensor:
    """Align the raw mask over the undeformed tile's silhouette; save the evidence."""
    raw = load_target_mask(args.TARGET_MASK)
    with torch.no_grad():
        alpha0 = soft_alpha(escher, escher.solve_points(), ctx)[0, ..., 0].cpu().numpy()

    aligned, params, iou = align_mask_to(alpha0, raw)
    print(
        f"target aligned: scale {params['scale']:.3f}, angle {params['angle_deg']:+.1f} deg, "
        f"initial IoU {iou:.3f}"
    )

    import imageio.v2 as imageio

    imageio.imwrite(escher.output_dir / "target_aligned.png", (aligned * 255).astype(np.uint8))
    overlay = np.stack([aligned, alpha0, np.zeros_like(aligned)], axis=-1)
    imageio.imwrite(
        escher.output_dir / "target_overlay.png", (overlay * 255).astype(np.uint8)
    )
    return torch.as_tensor(aligned, device=escher.device)


def shape_step(escher: SphereEscher, target: torch.Tensor, ctx: ShapeContext, args) -> dict:
    """One optimization step: ``SphereEscher.step()``'s skeleton, minus everything SDS.

    One solve per step: ``ensure_valid_shape`` already solved, and the silhouette is
    computed from those points analytically -- there is no render here to re-solve for.
    """
    escher.optimizer.zero_grad()

    points, flips, reverted = escher.ensure_valid_shape()
    if reverted:
        escher._n_reverts += 1
        # Rejection resets P but not Adam's momentum, which otherwise re-proposes the
        # same folding step forever (measured: 394 consecutive reverts, state frozen).
        escher.reset_shape_optimizer_state()

    alpha = soft_alpha(escher, points, ctx)
    mask = args.MASK_LOSS_WEIGHT * mask_pyramid_loss(
        alpha, target, levels=args.MASK_PYRAMID_LEVELS
    )

    reg = points.new_zeros(())
    if args.EQUAL_AREA_WEIGHT > 0:
        reg = reg + args.EQUAL_AREA_WEIGHT * equal_area_loss(
            points, escher.faces_t, escher.ref_areas
        )
    if args.AREA_TAIL_WEIGHT > 0:
        reg = reg + args.AREA_TAIL_WEIGHT * area_tail_loss(
            points, escher.faces_t, escher.ref_areas, fraction=args.AREA_TAIL_FRACTION
        )
    if args.PARAM_MODE == "boundary":
        reg = reg + escher.boundary_chain_regularizers()
        if args.BOUNDARY_MARGIN_WEIGHT > 0:
            reg = reg + args.BOUNDARY_MARGIN_WEIGHT * area_margin_loss(
                points, escher.faces_t, escher.ref_areas, margin=args.BOUNDARY_MARGIN
            )
    elif args.W_REGULARIZATION > 0:
        # The weights-mode priors, mirroring step(): area terms above guard against the
        # C-series degeneracy, this one against sigmoid saturation.
        reg = reg + args.W_REGULARIZATION * (escher.W**2).sum()

    loss = mask + reg.to(mask)
    loss.backward()
    escher.optimizer.step()

    areas = np.abs(signed_solid_angles(points.detach().cpu().numpy(), escher.mesh.faces))
    return {
        "loss": float(loss.detach()),
        "mask": float(mask.detach()),
        "iou": float(soft_iou(alpha.detach(), target)),
        "area_reg": float(reg.detach()),
        "boundary_ratio": escher.boundary_ratio(points),
        "area_spread": float(
            np.percentile(areas, 99) / max(np.percentile(areas, 1), 1e-30)
        ),
        "flips": flips,
        "reverts": escher._n_reverts,
        "energy": escher.embedder.last_result.energy,
        "solver_iters": (
            escher.embedder.last_result.stage1.n_iter
            + escher.embedder.last_result.stage2.n_iter
        ),
        "points": points,
    }


def log_shape_metrics(output_dir: Path, iteration: int, info: dict) -> None:
    """Own CSV, own header: ``log_metrics``'s header is hardcoded for the SDS loop."""
    path = output_dir / "metrics.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "step,loss,mask,iou,area_reg,karcher,"
                "boundary_ratio,area_spread,flips,reverts,solver_iters\n"
            )
        f.write(
            f"{iteration},{info['loss']:.4f},{info['mask']:.4f},{info['iou']:.4f},"
            f"{info['area_reg']:.4f},{info['energy']:.6f},"
            f"{info['boundary_ratio']:.6f},{info['area_spread']:.2f},"
            f"{info['flips']},{info['reverts']},{info['solver_iters']}\n"
        )


def save_shape_snapshot(
    escher: SphereEscher, target: torch.Tensor, ctx: ShapeContext, iteration: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        alpha = soft_alpha(escher, escher.solve_points(), ctx)

        # The wide tiled view needs nvdiffrast (CUDA); on CPU the analytic panels stand
        # alone, which is what lets the whole driver run in the CPU test suite.
        wide = None
        if escher.device.type == "cuda":
            wide_mv = orbit_views(
                1, distance=escher.args.PREVIEW_DISTANCE, elevation_deg=18.0
            )
            images, walpha, _ = escher.render(1, mv=wide_mv)
            wide = (images * walpha + 1.0 * (1 - walpha)).clamp(0, 1).cpu().numpy()[0]

    achieved = alpha[0, ..., 0].cpu().numpy()
    tgt = target.cpu().numpy()
    overlay = np.stack([tgt, achieved, np.zeros_like(tgt)], axis=-1)

    panels = [
        (tgt, "target"),
        (achieved, "achieved silhouette"),
        (overlay, "overlay (R=target, G=achieved)"),
    ]
    if wide is not None:
        panels.append((wide, "tiled sphere"))

    fig, axes = plt.subplots(1, len(panels), figsize=(4.0 * len(panels), 4.2))
    for ax, (img, title) in zip(np.atleast_1d(axes), panels):
        ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
        ax.set_title(title, fontsize=10)
        ax.set_axis_off()
    fig.suptitle(f"shape step {iteration}", fontsize=12)
    fig.tight_layout()
    fig.savefig(
        escher.output_dir / f"step_{iteration:05d}.png", dpi=110, bbox_inches="tight"
    )
    plt.close(fig)


def run_shape(args) -> dict:
    escher = build_shape_run(args)
    ctx = make_context(escher, args)
    target = prepare_target(escher, ctx, args)

    start = time.time()
    info: dict = {}
    for iteration in range(int(args.SHAPE_STEPS) + 1):
        info = shape_step(escher, target, ctx, args)

        if iteration % 10 == 0:
            log_shape_metrics(escher.output_dir, iteration, info)
            per_step = (time.time() - start) / max(iteration, 1)
            print(
                f"step {iteration:5d} | loss {info['loss']:9.1f} | "
                f"mask {info['mask']:8.1f} | iou {info['iou']:.4f} | "
                f"reg {info['area_reg']:8.1f} | "
                f"perim {info['boundary_ratio']:6.4f}x | "
                f"spread {info['area_spread']:6.1f} | "
                f"flp {info['flips']:2d}/{info['reverts']:3d} | "
                f"{per_step:5.2f} s/step",
                flush=True,
            )

        if iteration % args.VISUALIZATION_FREQ == 0:
            ok, message = escher.check_geometry(info["points"])
            if not ok:
                print(f"  !! geometry check failed: {message}")
            save_shape_snapshot(escher, target, ctx, iteration)
            escher.save_checkpoint(iteration)

    escher.save_checkpoint(int(args.SHAPE_STEPS))
    ok, message = escher.check_geometry(info["points"])
    print(f"final geometry: {message}")
    print(f"done in {(time.time() - start) / 60:.1f} min -> {escher.output_dir}")
    return {
        "k": int(args.ORBIFOLD_K),
        "iou": info["iou"],
        "boundary_ratio": info["boundary_ratio"],
        "area_spread": info["area_spread"],
        "flips": info["flips"],
        "valid": ok,
        "output_dir": str(escher.output_dir),
    }


def scaled_n_phi(n_phi_k4: int, k: int) -> int:
    """Scale the k=4 across-the-lune resolution to a 2*pi/k lune, keeping it odd.

    The lune's angular width is proportional to 1/k; the 2-fold cone at the bottom
    midpoint needs a vertex exactly there, hence odd.
    """
    n = int(round(n_phi_k4 * 4 / k))
    if n % 2 == 0:
        n += 1
    return max(n, 5)


def sweep(args) -> None:
    """Run the shape phase per k; pick by IoU among valid runs (perimeter tiebreak)."""
    results = []
    for k in args.SWEEP_K:
        run_args = OmegaConf.merge(
            args,
            {
                "ORBIFOLD_K": int(k),
                "MESH_N_PHI": scaled_n_phi(int(args.MESH_N_PHI), int(k)),
                "OUTPUT_DIR": f"{args.OUTPUT_DIR}_k{k}",
                "SWEEP_K": [],
            },
        )
        print(f"\n=== k={k} ({2 * int(k)} tiles), n_phi={run_args.MESH_N_PHI} ===")
        results.append(run_shape(run_args))

    lines = ["k,iou,boundary_ratio,area_spread,flips,valid,output_dir"]
    for r in results:
        lines.append(
            f"{r['k']},{r['iou']:.4f},{r['boundary_ratio']:.4f},"
            f"{r['area_spread']:.1f},{r['flips']},{r['valid']},{r['output_dir']}"
        )
    sweep_csv = Path(f"{args.OUTPUT_DIR}_sweep.csv")
    sweep_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    valid = [r for r in results if r["valid"] and r["flips"] == 0]
    if valid:
        best = max(valid, key=lambda r: (r["iou"], r["boundary_ratio"]))
        print(f"\nwinner: k={best['k']} (iou {best['iou']:.4f}) -> {best['output_dir']}")
    else:
        print("\nno valid run -- inspect the overlays before rerunning")


def main() -> None:
    cli = OmegaConf.from_cli()
    conf_file = cli.pop("CONF_FILE", "configs/sphere_shape.yaml")
    # sphere.yaml -> sphere_shape.yaml -> CONF_FILE (if different) -> CLI, so overlays
    # like sphere_shape_weights.yaml stay small deltas on the shape defaults.
    layers = [
        OmegaConf.load(PATH / "configs/sphere.yaml"),
        OmegaConf.load(PATH / "configs/sphere_shape.yaml"),
    ]
    if conf_file != "configs/sphere_shape.yaml":
        layers.append(OmegaConf.load(PATH / conf_file))
    args = OmegaConf.merge(*layers, cli)
    if args.SWEEP_K:
        sweep(args)
    else:
        run_shape(args)


if __name__ == "__main__":
    main()
