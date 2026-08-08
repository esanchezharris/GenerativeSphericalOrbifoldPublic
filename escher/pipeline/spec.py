"""Batch spec: YAML jobs x seeds x sweeps -> concrete per-figure run plans.

Every output path is derived here, replacing the hardcoded OUTPUT_DIR literals
that made two prompts collide (the live repo accumulated 310 hand-named output
dirs). Layout, all under ``<runs_root>/<batch_name>/``::

    <figure>[__<variant>]/
        targets/            silhouette candidates (unless reuse_targets)
        screen/cand_*/      per-candidate screen runs
        target.npy          the screen winner (job-local, never assets/)
        carve_s<seed>/      one carve per seed
        tex_s<seed>/        texture run(s), resumed from the best carve
    manifest.json  summary.csv  contact_sheet.html
"""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf


@dataclass
class JobPlan:
    """One fully-expanded job: a figure at one point of the sweep grid."""

    job_id: str  # "<figure>" or "<figure>__<variant>"
    figure: str
    variant: str
    prompt: str
    silhouette_prompt: str
    orbifold: list | None
    seeds: list[int]
    texture_seeds: list[int]
    reuse_targets: str | None
    stages: dict = field(default_factory=dict)  # per-stage raw config overrides
    root: Path = Path(".")

    @property
    def targets_dir(self) -> Path:
        return Path(self.reuse_targets) if self.reuse_targets else self.root / "targets"

    @property
    def winner_path(self) -> Path:
        return self.root / "target.npy"

    def carve_dir(self, seed: int) -> Path:
        return self.root / f"carve_s{seed}"

    def tex_dir(self, tseed: int) -> Path:
        return self.root / f"tex_s{tseed}"


def _deep_merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _slug(value) -> str:
    if isinstance(value, (list, tuple)):
        return "".join(str(v) for v in value)
    return re.sub(r"[^A-Za-z0-9]+", "", str(value)) or "x"


def _set_dotted(d: dict, dotted: str, value) -> None:
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value


def _default_silhouette_prompt(figure: str) -> str:
    return (
        f"a plain solid black silhouette of a {figure}, simple flat shape, "
        "white background, minimal flat logo, centered, full body"
    )


def load_spec(path: str | Path) -> tuple[str, Path, list[JobPlan]]:
    """Parse + validate a batch spec; returns (batch_name, batch_root, plans)."""
    path = Path(path)
    raw = OmegaConf.to_container(OmegaConf.load(path), resolve=True)

    batch_name = str(raw.get("batch_name") or path.stem)
    runs_root = Path(str(raw.get("runs_root", "runs")))
    batch_root = runs_root / batch_name
    defaults = raw.get("defaults") or {}
    jobs = raw.get("jobs") or []
    if not jobs:
        raise ValueError(f"{path}: spec has no jobs")

    plans: list[JobPlan] = []
    for job in jobs:
        if "figure" not in job:
            raise ValueError(f"{path}: every job needs a 'figure'")
        merged = _deep_merge(defaults, {k: v for k, v in job.items() if k != "sweep"})

        sweep = job.get("sweep") or {}
        keys = sorted(sweep.keys())
        combos = (
            list(itertools.product(*(sweep[k] for k in keys))) if keys else [()]
        )
        for combo in combos:
            cfg = dict(merged)
            cfg["stages"] = {
                k: dict(v) for k, v in (merged.get("stages") or {}).items()
            }
            variant_bits = []
            for k, v in zip(keys, combo):
                if k == "orbifold":
                    cfg["orbifold"] = v
                    variant_bits.append(f"o{_slug(v)}")
                else:
                    _set_dotted(cfg, k, v)
                    variant_bits.append(f"{k.split('.')[-1].lower()}{_slug(v)}")
            variant = "_".join(variant_bits)

            figure = str(cfg["figure"])
            job_id = f"{figure}__{variant}" if variant else figure
            seeds = [int(s) for s in (cfg.get("seeds") or [0])]
            tseeds = [int(s) for s in (cfg.get("texture_seeds") or [0])]
            plans.append(
                JobPlan(
                    job_id=job_id,
                    figure=figure,
                    variant=variant,
                    prompt=str(cfg.get("prompt", figure)),
                    silhouette_prompt=str(
                        cfg.get("silhouette_prompt")
                        or _default_silhouette_prompt(figure)
                    ),
                    orbifold=(
                        list(cfg["orbifold"]) if cfg.get("orbifold") else None
                    ),
                    seeds=seeds,
                    texture_seeds=tseeds,
                    reuse_targets=cfg.get("reuse_targets"),
                    stages=cfg["stages"],
                    root=batch_root / job_id,
                )
            )

    ids = [p.job_id for p in plans]
    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        raise ValueError(f"{path}: duplicate job ids after sweep expansion: {sorted(dupes)}")
    for p in plans:
        if p.reuse_targets and not Path(p.reuse_targets).exists():
            raise ValueError(f"{path}: {p.job_id}: reuse_targets {p.reuse_targets} does not exist")
    return batch_name, batch_root, plans
