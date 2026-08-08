"""Batch reports: summary.csv + contact_sheet.html, regenerated per completion.

The HTML sheet is dependency-free (relative <img> links to each job's final.png)
so it is useful mid-batch: rows appear as jobs finish.
"""

from __future__ import annotations

import html
from pathlib import Path

from escher.pipeline.manifest import Manifest
from escher.pipeline.spec import JobPlan

_STAGE_ORDER = ("target", "screen", "carve", "texture", "render")


def _job_stage_summary(stages: dict) -> dict:
    """Collapse per-seed stages into one status + best metrics per family."""
    out = {}
    for family in _STAGE_ORDER:
        members = {
            name: st
            for name, st in stages.items()
            if name == family or name.startswith(f"{family}_s")
        }
        if not members:
            out[family] = {"status": "-"}
            continue
        statuses = {st.get("status", "?") for st in members.values()}
        status = (
            "ok"
            if statuses == {"ok"}
            else ("ok*" if "ok" in statuses else "/".join(sorted(statuses)))
        )
        best = None
        for st in members.values():
            m = st.get("metrics") or {}
            if "hard_iou" in m and (best is None or m["hard_iou"] > best.get("hard_iou", -1)):
                best = m
        out[family] = {"status": status, "metrics": best or {}}
    return out


def write_reports(manifest: Manifest, batch_root: Path, plans: list[JobPlan]) -> None:
    batch_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for plan in plans:
        job = manifest.data["jobs"].get(plan.job_id, {})
        fam = _job_stage_summary(job.get("stages", {}))
        carve_m = fam["carve"].get("metrics") or {}
        rows.append(
            {
                "job": plan.job_id,
                "figure": plan.figure,
                "variant": plan.variant,
                "orbifold": "".join(str(c) for c in plan.orbifold or []) or "-",
                **{f"{f}_status": fam[f]["status"] for f in _STAGE_ORDER},
                "hard_iou": f"{carve_m.get('hard_iou', float('nan')):.4f}"
                if carve_m.get("hard_iou") is not None
                else "-",
                "iou": f"{carve_m.get('iou', float('nan')):.4f}"
                if carve_m.get("iou") is not None
                else "-",
                "perim": f"{carve_m.get('boundary_ratio', float('nan')):.4f}"
                if carve_m.get("boundary_ratio") is not None
                else "-",
            }
        )

    if rows:
        cols = list(rows[0].keys())
        lines = [",".join(cols)]
        lines += [",".join(str(r[c]) for c in cols) for r in rows]
        (batch_root / "summary.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    cells = []
    for plan, row in zip(plans, rows):
        final = None
        for tseed in plan.texture_seeds:
            candidate = plan.tex_dir(tseed) / "final.png"
            if candidate.exists():
                final = candidate.relative_to(batch_root)
                break
        img = (
            f'<img src="{html.escape(str(final).replace(chr(92), "/"))}" '
            'style="max-width:420px">'
            if final
            else "<em>no render yet</em>"
        )
        meta = " | ".join(
            f"{f}: {row[f'{f}_status']}" for f in _STAGE_ORDER
        )
        cells.append(
            f"<td><h3>{html.escape(plan.job_id)}</h3>{img}"
            f"<p>hard IoU {row['hard_iou']} | perim {row['perim']}<br>"
            f"<small>{meta}</small></p></td>"
        )

    per_row = 3
    trs = [
        "<tr>" + "".join(cells[i : i + per_row]) + "</tr>"
        for i in range(0, len(cells), per_row)
    ]
    (batch_root / "contact_sheet.html").write_text(
        "<html><body><h1>"
        + html.escape(manifest.data.get("batch", ""))
        + "</h1><table>"
        + "".join(trs)
        + "</table></body></html>",
        encoding="utf-8",
    )
