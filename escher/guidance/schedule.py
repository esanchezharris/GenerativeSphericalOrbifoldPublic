"""Per-step SDS schedule arming, shared by the planar and spherical pipelines.

Two facts this module encodes, both measured:

- ``update_step`` is the ONLY code that arms ``grad_clip_val`` from the
  ``CLIP_GRADIENTS_IN_SDS`` schedule, and NEITHER pipeline ever called it -- SDS
  gradient clipping was silently OFF in every historical run, planar and spherical.
- Annealing the max sampled timestep downward moves SDS from layout-scale composition
  (high noise) to detail polish (low noise); ``set_step_range`` is the hook
  (``StableDiffusion`` only -- DeepFloyd lacks it, hence the ``hasattr`` guard).
"""

from __future__ import annotations

__all__ = ["annealed_max_step", "arm_sds"]


def annealed_max_step(args, iteration: int) -> float:
    """Linear anneal of the SDS max-timestep fraction over ``[0, SDS_ANNEAL_END]``."""
    t = min(max(iteration / max(args.SDS_ANNEAL_END, 1), 0.0), 1.0)
    return float(
        args.get("SDS_MAX_START", 0.98)
        + (args.get("SDS_MAX_END", 0.5) - args.get("SDS_MAX_START", 0.98)) * t
    )


def arm_sds(guidance, args, iteration: int) -> None:
    """Call once per optimization step, before ``train_step``."""
    guidance.update_step(0, iteration)
    if args.get("SDS_ANNEAL_END", 0) > 0 and hasattr(guidance, "set_step_range"):
        guidance.set_step_range(
            guidance.cfg.min_step_percent, annealed_max_step(args, iteration)
        )
