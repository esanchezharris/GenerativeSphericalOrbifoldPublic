"""Stage runner: driver-side param building + child-side execution.

The driver resolves every stage's full config (the same yaml layering the manual
CLIs use) and writes it to ``<run>/stage_<name>.json``; the child process loads
that, calls the in-process function, writes ``.result.json``, and exits with a
code from the contract below. Children are separate processes on purpose: clean
CUDA teardown at every boundary, and one crashed stage cannot take the driver
down.

Exit-code contract (the driver's whole failure vocabulary):
    0  ok
    2  no usable candidate (target generation / reachability screen)
    3  geometry certificate failed (carve / render)
    4  CUDA out of memory (retried once in a fresh process)
    5  no-op resume (checkpoint already at/past N_STEPS)
    1  anything else (traceback preserved in the result json)
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from omegaconf import OmegaConf

from escher.main_sphere import PATH  # escher/ -- configs resolve against it
from escher.pipeline.spec import JobPlan


class StageFailure(Exception):
    def __init__(self, exit_code: int, message: str) -> None:
        super().__init__(message)
        self.exit_code = exit_code


# ----------------------------------------------------------------- driver side
def _merged(base_files: list[str], stage_over: dict, extra: dict) -> dict:
    """Entry-point-identical layering: bases -> stage CONF_FILE overlays ->
    remaining stage overrides -> driver-derived keys (paths, seeds)."""
    over = dict(stage_over or {})
    conf_files = over.pop("CONF_FILE", [])
    if isinstance(conf_files, str):
        conf_files = [conf_files]
    layers = [OmegaConf.load(PATH / f) for f in base_files]
    layers += [OmegaConf.load(PATH / f) for f in conf_files]
    merged = OmegaConf.merge(*layers, OmegaConf.create(over), OmegaConf.create(extra))
    return OmegaConf.to_container(merged, resolve=True)


def target_params(plan: JobPlan, dry_run: bool) -> dict:
    from escher.make_target import DEFAULTS

    over = dict(plan.stages.get("target") or {})
    args = OmegaConf.to_container(
        OmegaConf.merge(
            DEFAULTS,
            OmegaConf.create(over),
            OmegaConf.create(
                {"PROMPT": plan.silhouette_prompt, "OUT_DIR": str(plan.targets_dir)}
            ),
        ),
        resolve=True,
    )
    return {"stage": "target", "dry_run": dry_run, "args": args}


def _shape_extra(plan: JobPlan) -> dict:
    extra: dict = {}
    if plan.orbifold:
        extra["ORBIFOLD_CONES"] = list(plan.orbifold)
    return extra


def screen_params(plan: JobPlan, dry_run: bool) -> dict:
    extra = {
        **_shape_extra(plan),
        "SWEEP_TARGETS": True,
        "TARGET_DIR": str(plan.targets_dir),
        "WINNER_OUT": str(plan.winner_path),
        "OUTPUT_DIR": str(plan.root / "screen" / "cand"),
        # Screens run on the CPU lane so they can overlap a texture run that
        # owns the GPU exclusively.
        "TARGET_SWEEP_DEVICE": "cpu",
    }
    args = _merged(
        ["configs/sphere.yaml", "configs/sphere_shape.yaml"],
        plan.stages.get("screen"),
        extra,
    )
    return {"stage": "screen", "dry_run": dry_run, "args": args}


def carve_params(plan: JobPlan, seed: int, dry_run: bool) -> dict:
    extra = {
        **_shape_extra(plan),
        "SWEEP_TARGETS": False,
        "SWEEP_K": [],
        "TARGET_MASK": str(plan.winner_path),
        "OUTPUT_DIR": str(plan.carve_dir(seed)),
        "SEED": int(seed),
        "DEVICE": "cpu",  # CPU lane; carve on GPU only when the lane owns it
    }
    args = _merged(
        ["configs/sphere.yaml", "configs/sphere_shape.yaml"],
        plan.stages.get("carve"),
        extra,
    )
    return {"stage": "carve", "dry_run": dry_run, "args": args}


def texture_params(plan: JobPlan, tseed: int, resume_from: str, dry_run: bool) -> dict:
    extra = {
        **_shape_extra(plan),
        "PROMPT": plan.prompt,
        "OUTPUT_DIR": str(plan.tex_dir(tseed)),
        "SEED": int(tseed),
    }
    args = _merged(["configs/sphere.yaml"], plan.stages.get("texture"), extra)
    return {
        "stage": "texture",
        "dry_run": dry_run,
        "args": args,
        "resume_from": resume_from,
    }


def render_params(plan: JobPlan, tseed: int, dry_run: bool) -> dict:
    over = dict(plan.stages.get("render") or {})
    return {
        "stage": "render",
        "dry_run": dry_run,
        "checkpoint": str(plan.tex_dir(tseed) / "checkpoint.pt"),
        "tint": over.get("TINT"),
    }


# ------------------------------------------------------------------ child side
def _stage_target(params: dict) -> dict:
    args = OmegaConf.create(params["args"])
    out_dir = Path(args.OUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    if params.get("dry_run"):
        # Synthetic candidates exercising the full downstream plumbing with no
        # diffusion stack: distinct ellipses in the plausibility band.
        import imageio.v2 as imageio
        import numpy as np

        yy, xx = np.meshgrid(np.arange(64), np.arange(64), indexing="ij")
        for i, (ry, rx) in enumerate(((14, 14), (18, 12), (12, 19))):
            m = (
                ((yy - 32) / ry) ** 2 + ((xx - 32) / rx) ** 2 <= 1.0
            ).astype(np.uint8) * 255
            imageio.imwrite(out_dir / f"target_{i:02d}.png", m)
        np.save(out_dir / "target.npy", (m > 0).astype(np.float32))
        return {"chosen": 2, "out_dir": str(out_dir), "dry_run": True}
    from escher.make_target import generate

    try:
        return generate(args)
    except ValueError as e:
        raise StageFailure(2, str(e)) from e


def _stage_screen(params: dict) -> dict:
    from escher.main_shape import sweep_targets

    result = sweep_targets(OmegaConf.create(params["args"]))
    if result is None:
        raise StageFailure(2, "no valid screen candidate")
    result.pop("points", None)
    return result


def _stage_carve(params: dict) -> dict:
    from escher.main_shape import run_shape

    result = run_shape(OmegaConf.create(params["args"]))
    if not result["valid"]:
        raise StageFailure(3, "carve failed the geometry certificate")
    return result


def _stage_texture(params: dict) -> dict:
    args = OmegaConf.create(params["args"])
    if params.get("dry_run"):
        # Exercise the real cross-phase resume artifacts (checkpoint load with
        # reset_texture, fresh step counter) without the diffusion stack.
        from escher.main_shape import build_shape_run

        escher = build_shape_run(args)
        escher.load_checkpoint(params["resume_from"], reset_texture=True)
        escher.save_checkpoint(0)
    else:
        from escher.main_sphere import SphereEscher

        SphereEscher(args).run(resume_from=params["resume_from"], start_step=0)
    return {"checkpoint": str(Path(args.OUTPUT_DIR) / "checkpoint.pt")}


def _stage_render(params: dict) -> dict:
    from escher.render_final import finalize

    result = finalize(
        params["checkpoint"],
        tint=params.get("tint"),
        turntable=not params.get("dry_run"),
    )
    if not result["geometry_ok"]:
        raise StageFailure(3, "final geometry certificate failed")
    return result


_STAGES = {
    "target": _stage_target,
    "screen": _stage_screen,
    "carve": _stage_carve,
    "texture": _stage_texture,
    "render": _stage_render,
}


def run_stage_from_file(params_path: str | Path) -> None:
    """Child entry point: execute one stage, write the result json, exit."""
    params_path = Path(params_path)
    params = json.loads(params_path.read_text(encoding="utf-8"))
    result_path = params_path.with_suffix(".result.json")

    code, payload = 0, {}
    try:
        payload = {"status": "ok", "result": _STAGES[params["stage"]](params)}
    except StageFailure as e:
        code = e.exit_code
        payload = {"status": "failed", "error": str(e)}
    except SystemExit as e:
        code = int(e.code) if isinstance(e.code, int) else 1
        payload = {"status": "failed", "error": f"SystemExit({e.code})"}
    except BaseException as e:  # noqa: BLE001 -- the child must always report
        import torch

        code = 4 if isinstance(e, torch.cuda.OutOfMemoryError) else 1
        payload = {"status": "failed", "error": traceback.format_exc()}
    payload["exit_code"] = code
    result_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    sys.exit(code)
