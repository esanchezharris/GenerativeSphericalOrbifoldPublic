"""Unattended multi-figure batch driver for the spherical Escher pipeline.

The five stages (silhouette generation, reachability screen, carve, texture SDS,
final render) were five hand-typed commands with hand-copied paths between them.
This package chains them: a YAML spec of jobs expands into per-figure run plans,
a two-lane scheduler overlaps CPU shape work with the exclusive GPU texture runs,
every stage runs in its own child process (clean CUDA teardown, one crash cannot
take the night down), and a manifest + contact sheet record what happened.

    python -m escher.pipeline escher/configs/batch_example.yaml
    python -m escher.pipeline escher/configs/batch_smoke.yaml --dry-run
"""
