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

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from escher.main_sphere import PATH, SphereEscher
from escher.rendering.camera import orbit_views, perspective, tile_centric_views
from escher.shape_target import (
    align_mask_to,
    binarize_mask,
    hard_iou,
    mask_pyramid_loss,
    soft_iou,
)
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
    # Per-pixel solid angles for SOLID_ANGLE_LOSS_WEIGHTING; None = unweighted.
    loss_weights: torch.Tensor | None = None


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
    # The shape phase may use its own framing (SHAPE_RENDER_SIZE / SHAPE_CAMERA_FOV,
    # null = the render defaults). The production (2,3,4) framing left 4.2 px of
    # margin -- the PINNED cone corners pressed against the frame edge, and outline
    # excursions past the edge are invisible to BOTH the loss and the metric.
    # Widening FOV with a proportionally larger grid adds margin without shrinking
    # pixels-per-steradian (the "tile too small -> flat mask loss" failure mode).
    return ShapeContext(
        mv=fixed_tile_camera(escher, args),
        proj=perspective(fovy_deg=float(args.get("SHAPE_CAMERA_FOV") or args.CAMERA_FOV)),
        loop=torch.as_tensor(boundary_loop(escher.mesh), dtype=torch.long),
        size=int(args.get("SHAPE_RENDER_SIZE") or args.RENDER_SIZE),
        tau=float(args.MASK_TAU),
    )


def frame_margin_px(escher: SphereEscher, ctx: ShapeContext) -> float:
    """Min distance (px) from the UNDEFORMED tile boundary to the frame edge.

    Negative means boundary vertices start OUTSIDE the frame (the shipped
    sphere_shape_ico framing had 8 of them at margin -17.5 px) -- pixels the carve
    can neither move nor score.
    """
    pts = torch.as_tensor(escher.mesh.points, dtype=torch.float64)
    with torch.no_grad():
        px = project_to_pixels(pts[ctx.loop], ctx.mv, ctx.proj, ctx.size, ctx.size)
    return float(min(px.min(), ctx.size - 1 - px.max()))


def soft_alpha(escher: SphereEscher, points: torch.Tensor, ctx: ShapeContext) -> torch.Tensor:
    """``(1, H, W, 1)`` differentiable silhouette of the current tile."""
    px = project_to_pixels(points[ctx.loop], ctx.mv, ctx.proj, ctx.size, ctx.size)
    # float32 on the run device for the (pixels x segments) field; gradients flow back
    # through the cast into the float64 solve.
    px = px.to(device=escher.device, dtype=torch.float32)
    mask = soft_polygon_mask(px, ctx.size, ctx.size, tau=ctx.tau)
    return mask.reshape(1, ctx.size, ctx.size, 1)


def load_target_mask(path: str | Path, smooth_radius: int = 0) -> np.ndarray:
    """Load an already-binary target mask, optionally de-jaggying its contour.

    ``smooth_radius`` runs the same close-then-open pair as :func:`binarize_mask`; the
    masks reaching here are binary already, so only the morphology applies. Diffusion
    contours are ragged at pixel scale and the carve reproduces that raggedness as
    sawtooth serrations on the tile outline.
    """
    path = Path(path)
    if path.suffix == ".npy":
        mask = np.load(path).astype(np.float32)
    else:
        import imageio.v2 as imageio

        img = imageio.imread(path).astype(np.float32)
        if img.ndim == 3:
            img = img[..., :3].mean(-1)
        mask = (img / max(img.max(), 1e-9) > 0.5).astype(np.float32)
    if smooth_radius > 0:
        # binarize_mask expects the figure DARK on a light ground, so invert in and back.
        mask = binarize_mask(1.0 - mask, threshold=0.5, smooth_radius=int(smooth_radius))
    return mask


def prepare_target(escher: SphereEscher, ctx: ShapeContext, args) -> torch.Tensor:
    """Align the raw mask over the undeformed tile's silhouette; save the evidence."""
    raw = load_target_mask(args.TARGET_MASK, int(args.get("TARGET_SMOOTH_RADIUS", 0)))
    with torch.no_grad():
        alpha0 = soft_alpha(escher, escher.solve_points(), ctx)[0, ..., 0].cpu().numpy()

    weights = None
    if bool(args.get("SOLID_ANGLE_AREA_MATCH", False)):
        from escher.pixel_solid_angle import pixel_solid_angles

        weights = pixel_solid_angles(ctx.mv, ctx.proj, ctx.size, ctx.size)

    # The four PINNED cone corners (kite domains only): outline points the carve can
    # never move. Projected here for alignment scoring and for the diagnostics below.
    corner_px = None
    mesh = escher.mesh
    if all(hasattr(mesh, k) for k in ("cone1", "cone2a", "cone2b", "cone3")):
        idx = [mesh.cone1, mesh.cone2a, mesh.cone3, mesh.cone2b]
        pts = torch.as_tensor(mesh.points[idx], dtype=torch.float64)
        with torch.no_grad():
            corner_px = (
                project_to_pixels(pts, ctx.mv, ctx.proj, ctx.size, ctx.size)
                .cpu()
                .numpy()
            )

    aligned, params, iou = align_mask_to(
        alpha0,
        raw,
        match_area=bool(args.get("MATCH_TILE_AREA", True)),
        pixel_weights=weights,
        corner_px=corner_px,
        corner_weight=float(args.get("ALIGN_CORNER_WEIGHT", 0.0)),
        translation_px=float(args.get("ALIGN_TRANSLATION_SEARCH_PX", 0.0)),
        translation_steps=int(args.get("ALIGN_TRANSLATION_STEPS", 5)),
    )
    print(
        f"target aligned: scale {params['scale']:.3f}, angle {params['angle_deg']:+.1f} deg, "
        f"initial IoU {iou:.3f}, area ratio {params['area_ratio']:.4f} "
        f"({params['area_measure']})"
    )

    # Step-0 residual split + corner status: where would filler land TODAY?
    inside_t = aligned > 0.5
    inside_a = alpha0 > 0.5
    not_covered = float((inside_t & ~inside_a).sum()) / max(inside_t.sum(), 1)
    outside_tgt = float((inside_a & ~inside_t).sum()) / max(inside_a.sum(), 1)
    print(
        f"step-0 residual: target-not-covered {not_covered:.1%}, "
        f"tile-outside-target {outside_tgt:.1%}"
    )
    if corner_px is not None:
        status = []
        for name, (col, row) in zip(("cone1", "cone2a", "cone3", "cone2b"), corner_px):
            r, c = int(round(row)), int(round(col))
            ok = 0 <= r < inside_t.shape[0] and 0 <= c < inside_t.shape[1] and inside_t[r, c]
            status.append(f"{name}={'inside' if ok else 'OUTSIDE'}")
        print(f"pinned corners vs target: {', '.join(status)}")

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
        alpha, target, levels=args.MASK_PYRAMID_LEVELS, pixel_weights=ctx.loss_weights
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
    else:
        # The weights-mode priors, mirroring step(): area terms above guard against the
        # C-series degeneracy, this one against sigmoid saturation.
        if args.W_REGULARIZATION > 0:
            reg = reg + args.W_REGULARIZATION * (escher.W**2).sum()
        smooth_w = float(args.get("BOUNDARY_SMOOTH_WEIGHT_W", 0.0))
        if smooth_w > 0:
            # Second differences around the boundary loop -- the same term boundary mode
            # gets from boundary_chain_regularizers and the planar driver applies to its
            # own loop. Weights mode had NO outline-smoothness prior of any kind, only the
            # weight and area penalties, so nothing charged for a high-frequency wobble in
            # the boundary and the carve came back with sawtooth serrations along the
            # edges (visible as the green excess in the fish overlay).
            b = points[ctx.loop]
            second = b.roll(-1, 0) - 2.0 * b + b.roll(1, 0)
            reg = reg + smooth_w * second.square().sum()

    loss = mask + reg.to(mask)
    loss.backward()
    escher.optimizer.step()

    areas = np.abs(signed_solid_angles(points.detach().cpu().numpy(), escher.mesh.faces))
    return {
        "loss": float(loss.detach()),
        "mask": float(mask.detach()),
        "iou": float(soft_iou(alpha.detach(), target)),
        "hard_iou": hard_iou(
            alpha.detach()[0, ..., 0].cpu().numpy(), target.detach().cpu().numpy()
        ),
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


def evaluate_shape(escher: SphereEscher, target: torch.Tensor, ctx: ShapeContext, args) -> dict:
    """No-grad metrics of the CURRENT parameters -- exactly what a checkpoint holds.

    ``shape_step`` measures alpha BEFORE its optimizer step, so the loop's last info
    describes the state one Adam step before the checkpoint saved after it. Ranking a
    sweep or gating a carve on that stale number is off by one step; this eval is what
    ``run_shape`` reports and returns.
    """
    with torch.no_grad():
        points, flips, _ = escher.ensure_valid_shape()
        alpha = soft_alpha(escher, points, ctx)
        mask = args.MASK_LOSS_WEIGHT * mask_pyramid_loss(
            alpha, target, levels=args.MASK_PYRAMID_LEVELS, pixel_weights=ctx.loss_weights
        )
    areas = np.abs(signed_solid_angles(points.detach().cpu().numpy(), escher.mesh.faces))
    return {
        "mask": float(mask),
        "iou": float(soft_iou(alpha, target)),
        "hard_iou": hard_iou(alpha[0, ..., 0].cpu().numpy(), target.cpu().numpy()),
        "boundary_ratio": escher.boundary_ratio(points),
        "area_spread": float(
            np.percentile(areas, 99) / max(np.percentile(areas, 1), 1e-30)
        ),
        "flips": flips,
        "energy": escher.embedder.last_result.energy,
        "solver_iters": (
            escher.embedder.last_result.stage1.n_iter
            + escher.embedder.last_result.stage2.n_iter
        ),
        "points": points,
    }


def log_shape_metrics(output_dir: Path, iteration: int, info: dict) -> None:
    """Own CSV, own header: ``log_metrics``'s header is hardcoded for the SDS loop.

    ``hard_iou`` is appended LAST: tests and downstream parsers index the earlier
    columns by position.
    """
    path = output_dir / "metrics.csv"
    new = not path.exists()
    with open(path, "a", encoding="utf-8") as f:
        if new:
            f.write(
                "step,loss,mask,iou,area_reg,karcher,"
                "boundary_ratio,area_spread,flips,reverts,solver_iters,hard_iou\n"
            )
        f.write(
            f"{iteration},{info['loss']:.4f},{info['mask']:.4f},{info['iou']:.4f},"
            f"{info['area_reg']:.4f},{info['energy']:.6f},"
            f"{info['boundary_ratio']:.6f},{info['area_spread']:.2f},"
            f"{info['flips']},{info['reverts']},{info['solver_iters']},"
            f"{info['hard_iou']:.4f}\n"
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

    margin = frame_margin_px(escher, ctx)
    print(f"frame margin: {margin:.1f} px (undeformed tile on the {ctx.size}px grid)")
    min_margin = float(args.get("MIN_FRAME_MARGIN_PX", 0.0))
    if margin < min_margin:
        raise ValueError(
            f"undeformed tile margin {margin:.1f} px < MIN_FRAME_MARGIN_PX "
            f"{min_margin}; widen SHAPE_CAMERA_FOV / SHAPE_RENDER_SIZE or raise "
            f"SHAPE_CAMERA_DISTANCE"
        )
    if margin <= 0:
        print(
            "  !! boundary vertices start OUTSIDE the frame -- the loss and metric "
            "cannot see them; fix the framing before trusting any number from this run"
        )

    if bool(args.get("SOLID_ANGLE_LOSS_WEIGHTING", False)):
        from escher.pixel_solid_angle import pixel_solid_angles

        ctx.loss_weights = torch.as_tensor(
            pixel_solid_angles(ctx.mv, ctx.proj, ctx.size, ctx.size),
            dtype=torch.float32,
            device=escher.device,
        )

    target = prepare_target(escher, ctx, args)

    # Optional tau anneal over the final stretch: hard IoU is tau-invariant (the
    # alpha>0.5 contour is the polygon for ANY tau), but the LOSS's gradient band
    # narrows and sharpens as tau falls -- right when the residual is a thin
    # boundary ribbon. Late-only because narrow bands from step 0 risk instability
    # (raising tau LOWERS peak gradient; see mask_pyramid_loss docstring).
    base_tau = float(args.MASK_TAU)
    tau_end = float(args.get("MASK_TAU_END", 0.0) or 0.0)
    anneal_start = float(args.get("MASK_TAU_ANNEAL_START", 0.8))
    n_steps = max(int(args.SHAPE_STEPS), 1)

    start = time.time()
    info: dict = {}
    for iteration in range(int(args.SHAPE_STEPS) + 1):
        if tau_end > 0:
            f = iteration / n_steps
            if f > anneal_start:
                t = (f - anneal_start) / max(1.0 - anneal_start, 1e-9)
                ctx.tau = base_tau + (tau_end - base_tau) * min(t, 1.0)
        info = shape_step(escher, target, ctx, args)

        if iteration % 10 == 0:
            log_shape_metrics(escher.output_dir, iteration, info)
            per_step = (time.time() - start) / max(iteration, 1)
            print(
                f"step {iteration:5d} | loss {info['loss']:9.1f} | "
                f"mask {info['mask']:8.1f} | iou {info['iou']:.4f} | "
                f"hard {info['hard_iou']:.4f} | "
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
    # Post-step eval: the loop's last info is one Adam step stale relative to the
    # checkpoint just saved. Log it as the LAST row so last-row consumers see the
    # checkpoint-accurate numbers, and return/rank on it.
    final = evaluate_shape(escher, target, ctx, args)
    final_row = {**info, **final, "loss": final["mask"] + info["area_reg"]}
    log_shape_metrics(escher.output_dir, int(args.SHAPE_STEPS), final_row)
    ok, message = escher.check_geometry(final["points"])
    print(f"final geometry: {message}")
    print(
        f"final iou {final['iou']:.4f} (hard {final['hard_iou']:.4f}) | "
        f"perim {final['boundary_ratio']:.4f}x"
    )
    print(f"done in {(time.time() - start) / 60:.1f} min -> {escher.output_dir}")
    return {
        "k": int(args.ORBIFOLD_K),
        "iou": final["iou"],
        "hard_iou": final["hard_iou"],
        "boundary_ratio": final["boundary_ratio"],
        "area_spread": final["area_spread"],
        "flips": final["flips"],
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


def _sweep_rank_key(args):
    """Winner-selection key for sweeps, by SWEEP_RANK_METRIC.

    ``soft`` (default, the historical behavior) ranks by soft IoU with a HIGHER-
    perimeter tiebreak. ``hard`` ranks by hard IoU -- the number that corresponds
    to filler on the sphere, free of the tau bias that penalizes articulated
    outlines (a perfect carve scores ~0.93 soft at tau 3, and the shortfall grows
    with perimeter) -- with a LOWER-perimeter tiebreak (less filler boundary).
    """
    metric = str(args.get("SWEEP_RANK_METRIC", "soft"))
    if metric == "hard":
        return lambda r: (r["hard_iou"], -r["boundary_ratio"])
    if metric != "soft":
        raise ValueError(f"SWEEP_RANK_METRIC must be 'soft' or 'hard', got {metric!r}")
    return lambda r: (r["iou"], r["boundary_ratio"])


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

    lines = ["k,iou,boundary_ratio,area_spread,flips,valid,output_dir,hard_iou"]
    for r in results:
        lines.append(
            f"{r['k']},{r['iou']:.4f},{r['boundary_ratio']:.4f},"
            f"{r['area_spread']:.1f},{r['flips']},{r['valid']},{r['output_dir']},"
            f"{r['hard_iou']:.4f}"
        )
    sweep_csv = Path(f"{args.OUTPUT_DIR}_sweep.csv")
    sweep_csv.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n" + "\n".join(lines))

    valid = [r for r in results if r["valid"] and r["flips"] == 0]
    if valid:
        best = max(valid, key=_sweep_rank_key(args))
        print(f"\nwinner: k={best['k']} (iou {best['iou']:.4f}) -> {best['output_dir']}")
    else:
        print("\nno valid run -- inspect the overlays before rerunning")


def _sweep_one(cfg: dict) -> dict | None:
    """One candidate screen, in its own process. Module-level so it pickles.

    Takes a plain container rather than an ``OmegaConf`` object, and returns ``None``
    instead of raising, so one bad candidate cannot take the pool down. RuntimeError
    is caught too -- a solver/adjoint blowup on one candidate used to kill the whole
    screen -- and the traceback is preserved in the candidate's output dir rather
    than silenced.
    """
    torch.set_num_threads(1)
    try:
        return run_shape(OmegaConf.create(cfg))
    except (ValueError, RuntimeError) as e:
        import traceback

        out = Path(cfg.get("OUTPUT_DIR", "."))
        try:
            out.mkdir(parents=True, exist_ok=True)
            (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        print(f"  {Path(cfg['TARGET_MASK']).stem}: failed ({type(e).__name__}: {e})")
        return None


def sweep_targets(args) -> None:
    """Carve against every candidate silhouette; keep the one the tiling can actually be.

    ``make_target.py`` picks a candidate by figure AREA, which says nothing about whether
    the tiling can adopt that outline. Shapes that tile a surface under a fixed group are
    a thin subset -- every limb needs a complementary notch in a neighbour -- and Stable
    Diffusion draws with no such constraint, so a candidate can be a perfect gingerbread
    man and still be unreachable. Measured: against the area-picked candidate the carve
    converges to 0.75 and will not move (regularizers off changes it by +0.007), while
    the same optimizer on a target known to be reachable climbs past 0.87 in 20 steps.
    Reachability is the binding constraint, so select on it.

    Short carves, since the ranking settles long before convergence. CPU only.
    """
    candidates = sorted(Path(args.TARGET_DIR).glob("target_[0-9]*.png"))
    candidates = [p for p in candidates if not p.stem.endswith("_raw")]
    if not candidates:
        raise FileNotFoundError(f"no target_*.png candidates in {args.TARGET_DIR}")

    # Reachability alone is gameable, and measurably so: a candidate whose binarization
    # failed and covers the whole frame gets area-matched down to a tile-shaped blob and
    # then "wins" by not deforming at all (measured on (2,3,4): candidates 1/2/6 scored
    # IoU 0.757 at boundary_ratio 1.019, against 0.729 for the real figure). So keep
    # make_target.py's plausibility band as a gate and rank by IoU only within it.
    lo, hi = float(args.TARGET_AREA_MIN), float(args.TARGET_AREA_MAX)
    plausible = []
    for path in candidates:
        area = float(load_target_mask(path).mean())
        if lo <= area <= hi:
            plausible.append(path)
        else:
            print(f"  {path.stem}: figure area {area:.3f} outside [{lo}, {hi}] -- skipped")
    if not plausible:
        raise ValueError(f"no candidate in {args.TARGET_DIR} has a plausible figure area")
    candidates = plausible

    cfgs = [
        OmegaConf.to_container(
            OmegaConf.merge(
                args,
                {
                    "TARGET_MASK": str(path),
                    "OUTPUT_DIR": f"{args.OUTPUT_DIR}_{path.stem}",
                    "SHAPE_STEPS": int(args.TARGET_SWEEP_STEPS),
                    "SWEEP_TARGETS": False,
                    "SWEEP_K": [],
                },
            ),
            resolve=True,
        )
        for path in candidates
    ]

    # Candidates are completely independent solves, so screening them is embarrassingly
    # parallel -- and it was running one at a time on one of 12 cores. Each worker pins
    # itself to a single BLAS thread, otherwise N processes each spawn N threads and
    # oversubscribe the machine into being slower than sequential.
    workers = int(args.get("TARGET_SWEEP_WORKERS", 0)) or min(8, max(1, (os.cpu_count() or 2) - 2))
    workers = max(1, min(workers, len(cfgs)))
    print(f"screening {len(cfgs)} candidates on {workers} worker(s)")

    results = []
    if workers == 1:
        pairs = [(c, _sweep_one(c)) for c in cfgs]
    else:
        # SPAWN, not fork: the parent has already run autograd by the time a sweep starts,
        # and forking after that raises "Unable to handle autograd's threading in
        # combination with fork-based multiprocessing". Spawn re-imports this module in
        # each child, which is why _sweep_one is module-level and takes a plain container.
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as pool:
            pairs = list(zip(cfgs, pool.map(_sweep_one, cfgs)))
    for cfg, r in pairs:
        if r is None:  # binarize_mask rejected an all-background candidate
            print(f"  {Path(cfg['TARGET_MASK']).stem}: skipped (no usable foreground)")
            continue
        r["target"] = cfg["TARGET_MASK"]
        results.append(r)

    lines = ["target,iou,boundary_ratio,area_spread,flips,valid,output_dir,hard_iou"]
    for r in results:
        lines.append(
            f"{Path(r['target']).stem},{r['iou']:.4f},{r['boundary_ratio']:.4f},"
            f"{r['area_spread']:.1f},{r['flips']},{r['valid']},{r['output_dir']},"
            f"{r['hard_iou']:.4f}"
        )
    Path(f"{args.OUTPUT_DIR}_targets.csv").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(lines))

    valid = [r for r in results if r["valid"] and r["flips"] == 0]
    if not valid:
        print("\nno valid candidate -- inspect the overlays before rerunning")
        return
    best = max(valid, key=_sweep_rank_key(args))
    chosen = Path(args.TARGET_DIR) / "target.npy"
    # Save the winner RAW (no morphology): the production carve applies
    # TARGET_SMOOTH_RADIUS itself on load, and saving it smoothed here meant the
    # radius was applied TWICE -- eroding exactly the thin fins that make a fish
    # read as a fish.
    np.save(chosen, load_target_mask(best["target"]))
    print(
        f"\nwinner: {Path(best['target']).stem} (iou {best['iou']:.4f}, "
        f"hard {best['hard_iou']:.4f}) -> {chosen}\n"
        f"  re-run the full carve with TARGET_MASK={chosen}"
    )


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
    if args.get("SWEEP_TARGETS", False):
        sweep_targets(args)
    elif args.SWEEP_K:
        sweep(args)
    else:
        run_shape(args)


if __name__ == "__main__":
    main()
