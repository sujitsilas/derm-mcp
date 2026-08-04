"""`brief()` — the single most important call in the server.

A ≤1500-token state summary that lets a 32k-context local model pick up a
project it has never seen. Everything here is *descriptive*: it reports what
happened and what defaults are on record. It never issues instructions, and the
parameters it surfaces are offered, not applied (§5 guardrail).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ..config import CONFIG
from . import store


def _ds_line(d: dict[str, Any]) -> str:
    label = f" [{d['label']}]" if d.get("label") else ""
    parent = f" <- {d['parent_id']}" if d.get("parent_id") else ""
    return (f"{d['dataset_id']}{label}: {d['n_obs']}x{d['n_vars']} "
            f"x={d.get('x_state', '?')} op={d.get('op', '?')}{parent}")


def _artifact_lines(project_id: str, limit: int = 14) -> list[str]:
    """Recent figures and tables as `kind path`, relative to `output_dir`.

    Relative because the briefing has a 4 KB budget and absolute paths are ~80
    bytes each: spelling them out truncated the list to five of fourteen, which
    is exactly the information a resuming session needs in full.
    """
    root = Path(store.get_output_dir(project_id) or CONFIG.project_dir(project_id))
    out = []
    for a in store.list_artifacts(project_id, limit=limit):
        p = Path(a.get("path") or "")
        try:
            rel = p.relative_to(root) if p.is_absolute() else p
        except ValueError:
            rel = p  # outside the output dir; keep it absolute
        out.append(f"{a.get('kind', '?')} {rel}")
    return out


def open_flags(project_id: str, steps: list[dict[str, Any]]) -> list[str]:
    """Warnings from recent steps that nothing has yet resolved.

    Collapsed by tool+error: one stuck loop used to emit eight identical
    "set_label FAILED" lines and push everything the project had actually
    achieved out of the briefing.
    """
    counts: dict[str, int] = {}
    latest: dict[str, int] = {}
    for s in steps:
        if s.get("ok"):
            continue
        key = f"{s['tool']}: {(s.get('error') or '')[:120]}"
        counts[key] = counts.get(key, 0) + 1
        latest[key] = s["step_id"]
    out = []
    for key, n in list(counts.items())[-6:]:
        times = f" (x{n}, last step {latest[key]})" if n > 1 else f" (step {latest[key]})"
        out.append(f"{key}{times}")
    return out


def workflow_state(project_id: str) -> dict[str, Any]:
    """Infer where the project is, so `skin.help.workflow` can propose next steps."""
    datasets = store.list_datasets(project_id)
    steps = store.get_steps(project_id, limit=400)
    tools_run = {s["tool"] for s in steps if s.get("ok")}
    anns = store.get_annotations(project_id)

    state = {
        "has_data": bool(datasets),
        "has_qc": any(t.startswith("skin.qc.") for t in tools_run),
        "has_filtered": "skin.qc.apply_filters" in tools_run,
        "has_doublets": "skin.doublet.call" in tools_run,
        "has_preprocess": "skin.integrate.preprocess" in tools_run,
        "has_integration": "skin.integrate.harmony" in tools_run,
        "has_neighbors": "skin.cluster.neighbors" in tools_run,
        "has_umap": "skin.cluster.umap" in tools_run,
        "has_clusters": "skin.cluster.leiden" in tools_run,
        "has_markers": "skin.cluster.marker_genes" in tools_run,
        "has_labels": "skin.annotate.apply_labels" in tools_run or bool(anns),
        "has_audit": "skin.annotate.contamination_audit" in tools_run,
        "has_de": any(t.startswith("skin.de.") for t in tools_run),
        "has_enrich": any(t.startswith("skin.enrich.") for t in tools_run),
    }

    order = [
        ("has_data", "skin.io.build_multisample", "Load your samples first."),
        ("has_qc", "skin.qc.sample_stats", "Discover per-sample thresholds before filtering."),
        ("has_filtered", "skin.qc.preview_filters", "Preview, then skin.qc.apply_filters."),
        ("has_doublets", "skin.doublet.call", "Call doublets per sample (do not filter yet)."),
        ("has_preprocess", "skin.integrate.preprocess", "Normalize, HVG, scale, PCA."),
        ("has_integration", "skin.integrate.harmony", "Batch-correct across samples."),
        ("has_neighbors", "skin.cluster.neighbors", "Build the kNN graph on X_pca_harmony."),
        ("has_umap", "skin.cluster.umap", "Embed for visualisation."),
        ("has_clusters", "skin.cluster.leiden", "Cluster; consider skin.cluster.leiden_sweep."),
        ("has_markers", "skin.cluster.marker_genes", "Rank genes per cluster."),
        ("has_labels", "skin.annotate.marker_report", "Propose labels, then apply_labels."),
        ("has_audit", "skin.annotate.contamination_audit", "Audit cross-lineage contamination."),
        ("has_de", "skin.de.pseudobulk", "Pseudobulk DE is the default; Wilcoxon is exploratory."),
        ("has_enrich", "skin.enrich.ora", "Enrich the DE results, then plot the tile."),
    ]
    nxt: list[dict[str, str]] = []
    for key, tool, why in order:
        if not state[key]:
            nxt.append({"tool": tool, "why": why})
            if len(nxt) >= 3:
                break
    if not nxt:
        nxt = [{"tool": "skin.export.report", "why": "Pipeline complete — export the lab notebook."}]
    state["next_suggested_tools"] = nxt
    return state


def brief(project_id: str, max_steps: int = 5) -> dict[str, Any]:
    proj = store.get_project(project_id) or {}
    datasets = store.list_datasets(project_id)
    steps = store.get_steps(project_id, limit=max_steps)
    all_steps = store.get_steps(project_id, limit=500)
    params = store.list_params(project_id)
    anns = store.get_annotations(project_id)
    decisions = store.get_decisions(project_id, limit=6)
    notes = store.get_notes(project_id, limit=5)
    runs = store.list_runs(project_id, limit=8)

    ann_sets: dict[str, dict[str, Any]] = {}
    for a in anns:
        k = f"{a['dataset_id']}:{a['obs_key']}"
        e = ann_sets.setdefault(k, {"n_labels": 0, "labels": [], "authors": set()})
        e["n_labels"] += 1
        if len(e["labels"]) < 12:
            e["labels"].append(f"{a['cluster']}->{a['label']}")
        e["authors"].add(a["author"])
    for e in ann_sets.values():
        e["authors"] = sorted(e["authors"])

    return {
        "project": {
            "project_id": project_id,
            "name": proj.get("name"),
            "organism": proj.get("organism"),
            "created_at": proj.get("created_at"),
            "description": (proj.get("description") or "")[:400],
            "design_notes": (proj.get("design_notes") or "")[:400],
        },
        "datasets": [_ds_line(d) for d in datasets][-12:],
        "n_datasets": len(datasets),
        "annotation_sets": ann_sets,
        "parameters_in_force": {
            f"{p['name']}@{p['scope']}": p["value"] for p in params
        },
        "parameter_note": (
            "Parameters are recorded defaults, not automatic behaviour. Pass them "
            "explicitly to a tool if you want them applied."
        ),
        "decisions": [
            {"q": d["question"][:120], "choice": d["choice"][:80]} for d in decisions
        ],
        "notes": [{"tag": n["tag"], "body": n["body"][:160]} for n in notes],
        "runs": runs,
        # What already exists on disk. Without this a session resuming after a
        # context compaction cannot tell finished work from work still to do,
        # and re-runs the whole analysis: the handoff named the figures, but
        # nothing said where they were or that they were already rendered.
        "output_dir": store.get_output_dir(project_id) or str(
            CONFIG.project_dir(project_id)),
        "artifacts": _artifact_lines(project_id),
        "n_artifacts": len(store.list_artifacts(project_id, limit=500)),
        "resume_note": (
            "artifact paths are relative to output_dir, newest first. These runs "
            "and files ALREADY EXIST — do not recompute them. Re-render a figure "
            "only if the user asked for a change, and reuse the listed run_id "
            "rather than starting a new DE or enrichment run. If n_artifacts "
            "exceeds the list shown, skin.memory.export has the full record."
        ),
        "last_steps": [
            {"id": s["step_id"], "tool": s["tool"], "ok": bool(s["ok"]),
             "t": round(s["duration_s"], 2)}
            for s in steps
        ],
        "n_steps_total": len(all_steps),
        "open_flags": open_flags(project_id, all_steps),
        "workflow": workflow_state(project_id),
    }


def as_markdown(project_id: str) -> str:
    """Lab-notebook markdown of the whole project (`skin.memory.export`)."""
    proj = store.get_project(project_id) or {}
    lines = [
        f"# {proj.get('name', project_id)}",
        "",
        f"- **project_id**: `{project_id}`",
        f"- **organism**: {proj.get('organism')}",
        f"- **created**: {proj.get('created_at')}",
        "",
        proj.get("description") or "",
        "",
    ]
    if proj.get("design_notes"):
        lines += ["## Experimental design", "", proj["design_notes"], ""]

    lines += ["## Datasets", ""]
    for d in store.list_datasets(project_id):
        lines.append(f"- `{d['dataset_id']}` **{d.get('label') or ''}** — {d['n_obs']} cells × "
                     f"{d['n_vars']} genes, X={d.get('x_state')}, op=`{d.get('op')}`"
                     + (f", parent=`{d['parent_id']}`" if d.get("parent_id") else ""))

    anns = store.get_annotations(project_id, include_superseded=True)
    if anns:
        lines += ["", "## Annotations", ""]
        for a in anns:
            sup = " *(superseded)*" if a.get("superseded_by") else ""
            lines.append(
                f"### `{a['obs_key']}` cluster {a['cluster']} → **{a['label']}**{sup}\n"
                f"- confidence: {a.get('confidence')}\n"
                f"- author: {a.get('author')}\n"
                f"- rationale: {a.get('rationale') or '_none recorded_'}\n"
            )

    dec = store.get_decisions(project_id, limit=200)
    if dec:
        lines += ["", "## Decisions", ""]
        for d in dec:
            lines.append(f"### {d['question']}\n- **chose**: {d['choice']}\n"
                         f"- alternatives: {d.get('alternatives_json')}\n"
                         f"- rationale: {d.get('rationale')}\n- author: {d.get('author')}\n")

    params = store.list_params(project_id)
    if params:
        lines += ["", "## Parameters on record", "", "| name | scope | value | rationale |",
                  "|---|---|---|---|"]
        for p in params:
            lines.append(f"| `{p['name']}` | {p['scope']} | `{p['value']}` | {p.get('rationale', '')} |")

    notes = store.get_notes(project_id, limit=500)
    if notes:
        lines += ["", "## Notes", ""]
        for n in reversed(notes):
            lines.append(f"- **[{n['tag']}]** ({n['created_at']}) {n['body']}")

    steps = store.get_steps(project_id, limit=1000)
    lines += ["", "## Provenance log", "", "| # | tool | ok | seconds | when |", "|---|---|---|---|---|"]
    for s in steps:
        lines.append(f"| {s['step_id']} | `{s['tool']}` | {'✓' if s['ok'] else '✗'} | "
                     f"{s['duration_s']:.2f} | {s['started_at']} |")
    return "\n".join(lines)
