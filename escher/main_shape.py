"""Deterministic shape phase: fit the tile outline to a target silhouette mask.

The SDS silhouette pass was the outline's only driver and its weakest link: noisy
gradients, CUDA-chaotic trajectories, and a measured equilibrium of ~1.13x perimeter
before the noise and the validity terms stalemate. This driver replaces it. A figure
mask is generated once (``make_target.py``), aligned over the undeformed tile, and the
free boundary points are optimized so the isolated tile's rendered alpha matches it --
a fixed camera, a deterministic loss (``shape_target.mask_pyramid_loss``), exact
gradients, and NO diffusion model in memory. Steps cost a solve plus one raster, so a
whole run takes minutes and A/B experiments (targets, ``ORBIFOLD_K``) are cheap.

All validity machinery carries over unchanged: fold rejection (``ensure_valid_shape``),
the margin hinge, chain spacing/smoothness, and the equal-area terms. Checkpoints are
ordinary ``SphereEscher`` checkpoints, so the texture phase resumes one with the
existing ``RESUME=`` path.

Usage (inside the WSL venv)::

    python escher/main_shape.py                          # single run, sphere_shape.yaml
    python escher/main_shape.py "SWEEP_K=[4,6]"          # k sweep, pick by IoU
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.main_sphere import PATH, SphereEscher
from escher.rendering.camera import orbit_views, tile_centric_views
from escher.shape_target import align_mask_to, mask_pyramid_loss, soft_iou
from escher.OTE.core.spherical.regularizers import (
    area_margin_loss,
    area_tail_loss,
    equal_area_loss,
)
from escher.geometry.spherical_sanity_checks import signed_solid_angles


def build_shape_run(args) -> SphereEscher:
    """A ``SphereEscher`` with geometry and parameters but NO guidance model.

    The ``__new__`` construction is the same one ``render_final.load_run`` and the CPU
    tests use; ``__init__`` is skipped, so the seeding it normally does happens here.
    """
    assert args.PARAM_MODE == "boundary", "the shape phase drives the boundary directly"
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

    From the UNDEFORMED ``mesh.points``, with zero jitter. The default render path
    re-derives tile centers from the *current deformed* points every step and consumes
    global RNG -- under a fixed target mask that would slide the camera (and so the
    target) under the tile as it deforms, corrupting the loss.
    """
    centers = escher.solo_tiler.tile_centers(escher.mesh.points)
    return tile_centric_views(
        torch.as_tensor(centers, dtype=torch.float32),
        1,
        distance=float(args.SHAPE_CAMERA_DISTANCE),
        angular_jitter_deg=0.0,
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


def prepare_target(escher: SphereEscher, mv: torch.Tensor, args) -> torch.Tensor:
    """Align the raw mask over the undeformed tile's silhouette; save the evidence."""
    raw = load_target_mask(args.TARGET_MASK)
    with torch.no_grad():
        _, alpha, _ = escher.render(1, mv=mv, isolated=True, texture=escher.solid_texture)
    alpha0 = alpha[0, ..., 0].detach().cpu().numpy()

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


def shape_step(escher: SphereEscher, target: torch.Tensor, mv: torch.Tensor, args) -> dict:
    """One optimization step: ``SphereEscher.step()``'s skeleton, minus everything SDS."""
    escher.optimizer.zero_grad()

    # Validity projection first, exactly as the SDS loop does it.
    _, flips, reverted = escher.ensure_valid_shape()
    if reverted:
        escher._n_reverts += 1

    # The second solve after a revert is warm-started and near-free.
    _, alpha, points = escher.render(1, mv=mv, isolated=True, texture=escher.solid_texture)

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
    reg = reg + escher.boundary_chain_regularizers()
    if args.BOUNDARY_MARGIN_WEIGHT > 0:
        reg = reg + args.BOUNDARY_MARGIN_WEIGHT * area_margin_loss(
            points, escher.faces_t, escher.ref_areas, margin=args.BOUNDARY_MARGIN
        )

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
    escher: SphereEscher, target: torch.Tensor, mv: torch.Tensor, iteration: int
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with torch.no_grad():
        _, alpha, points = escher.render(
            1, mv=mv, isolated=True, texture=escher.solid_texture
        )
        wide_mv = orbit_views(1, distance=escher.args.PREVIEW_DISTANCE, elevation_deg=18.0)
        images, walpha, _ = escher.render(1, mv=wide_mv)
        wide = (images * walpha + 1.0 * (1 - walpha)).clamp(0, 1).cpu().numpy()[0]

    achieved = alpha[0, ..., 0].cpu().numpy()
    tgt = target.cpu().numpy()
    overlay = np.stack([tgt, achieved, np.zeros_like(tgt)], axis=-1)

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.2))
    for ax, img, title in zip(
        axes,
        [tgt, achieved, overlay, wide],
        ["target", "achieved silhouette", "overlay (R=target, G=achieved)", "tiled sphere"],
    ):
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
    mv = fixed_tile_camera(escher, args)
    target = prepare_target(escher, mv, args)

    start = time.time()
    info: dict = {}
    for iteration in range(int(args.SHAPE_STEPS) + 1):
        info = shape_step(escher, target, mv, args)

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
            save_shape_snapshot(escher, target, mv, iteration)
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
    args = OmegaConf.merge(
        OmegaConf.load(PATH / "configs/sphere.yaml"),
        OmegaConf.load(PATH / conf_file),
        cli,
    )
    if args.SWEEP_K:
        sweep(args)
    else:
        run_shape(args)


if __name__ == "__main__":
    main()
