"""CLI: run a batch spec, or (internally) one stage in a child process."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(
        prog="python -m escher.pipeline",
        description="Unattended multi-figure batch driver (see batch_example.yaml).",
    )
    ap.add_argument("spec", nargs="?", help="batch spec YAML")
    ap.add_argument("--run-stage", metavar="PARAMS_JSON", help="(internal) child mode")
    ap.add_argument("--dry-run", action="store_true", help="stub GPU stages; <1 min")
    ap.add_argument("--resume", action="store_true", help="skip stages already ok")
    ap.add_argument(
        "--warden", action="store_true",
        help="wrap GPU stages in a gpu.py lease (see pipeline/warden.py caveat)",
    )
    args = ap.parse_args()

    if args.run_stage:
        from escher.pipeline.stages import run_stage_from_file

        run_stage_from_file(args.run_stage)
        return

    if not args.spec:
        ap.error("a batch spec is required (or --run-stage)")

    from escher.pipeline.manifest import Manifest
    from escher.pipeline.scheduler import Scheduler
    from escher.pipeline.spec import load_spec

    batch_name, batch_root, plans = load_spec(args.spec)
    manifest_path = batch_root / "manifest.json"
    if manifest_path.exists() and not args.resume:
        raise SystemExit(
            f"{manifest_path} exists -- pass --resume to continue it, or choose a "
            "new batch_name"
        )
    manifest = Manifest(manifest_path, batch_name)

    print(
        f"batch {batch_name}: {len(plans)} job(s) -> {batch_root} "
        f"(dry_run={args.dry_run}, warden={args.warden})"
    )
    ok = Scheduler(
        plans,
        manifest,
        batch_root,
        dry_run=args.dry_run,
        use_warden=args.warden,
        repo_root=Path.cwd(),
    ).run()
    print(f"batch {'complete' if ok else 'FINISHED WITH FAILURES'} -> {batch_root}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
