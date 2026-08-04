"""`skin.memory.*` — the persistent project memory.

This is what lets a local model pick up a project three weeks later and what
lets a PI audit how a label was assigned. Memory is descriptive, never
directive: nothing recorded here changes server behaviour.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from ..config import CONFIG
from ..errors import BadParam, NotFound
from ..memory import recall, semantic, store
from ._base import Ctx, get_active_project, set_active_project, tool


@tool("skin.memory.open_project", category="memory", needs_project=False,
      summary="Create or re-attach a project; returns a resume briefing.")
def open_project(
    name: str,
    organism: str = "mouse",
    description: str = "",
    design_notes: str = "",
    output_dir: str = "",
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Create a project, or re-attach to an existing one with the same name.

    Call this first, every session. The returned briefing is the whole project
    state in under 1500 tokens.

    Args:
        name: Human name, e.g. "burn_sham_macs_2026".
        organism: "mouse" or "human". Fixed for the life of the project.
        description: What the experiment is.
        design_notes: Conditions, timepoints, replicates. Free text; it ends up
            in the exported methods draft.
        output_dir: Where figures/ and tables/ are written. Defaults to the
            directory the data is loaded from, so results sit beside the .h5ad.
            Set this when the user names somewhere else.
        project_id: Re-attach to an exact id instead of matching by name.
        dry_run: Validate only.
        seed: RNG seed recorded with the step.
    """
    from ..knowledge import _check_organism

    org = _check_organism(organism)
    existing = store.get_project(project_id) if project_id else store.find_project_by_name(name)

    if dry_run:
        ctx.summary = {"would_create": existing is None, "name": name, "organism": org}
        return

    if existing:
        pid = existing["project_id"]
        if existing.get("organism") != org:
            ctx.warn(f"Project {pid} is registered as {existing.get('organism')}; the "
                     f"requested organism {org!r} was ignored. Organism is immutable.")
        resumed = True
    else:
        pid = store.create_project(name, org, description, design_notes, project_id or None)
        resumed = False

    if output_dir:
        store.set_output_dir(pid, str(Path(output_dir).expanduser().resolve()))

    set_active_project(pid)
    ctx.project_id = pid
    b = recall.brief(pid)

    # State the runtime up front. Without this the caller cannot tell whether it
    # must provision an environment before doing anything, and burns turns
    # looking for one. The server's own interpreter already has every core
    # dependency; a project-local runtime is only for pinned reproducibility.
    from ..runtimes import bridge, python_manifest

    pkgs = python_manifest.capture()
    r = bridge.runtime_status()
    runtime = {
        "python": "ready (in-process)",
        "scanpy": pkgs.get("scanpy"),
        "r": "ready" if r["available"] else "not installed",
        "note": ("No provisioning is needed to start — load data and analyse. "
                 "skin.runtime.create(kind='python'|'r') builds a pinned per-project "
                 "environment when you want the versions frozen for reproduction."),
    }
    if not r["available"]:
        runtime["r_note"] = ("R tools (scDblFinder, DecontX, miloR, CellChat) will "
                             "return RUNTIME_UNAVAILABLE naming a Python fallback.")

    ctx.summary = {
        "project_id": pid,
        "resumed": resumed,
        "root": str(CONFIG.project_dir(pid)),
        "runtime": runtime,
        "memory": semantic.status(pid)["backend"],
        "briefing": b,
    }
    ctx.code = (f"# project: {name} ({org})\n"
                f"PROJECT_ID = {pid!r}\n"
                f"PROJECT_ROOT = {str(CONFIG.project_dir(pid))!r}\n")
    if resumed:
        ctx.suggest("skin.memory.brief", "skin.io.lineage", "skin.help.workflow")
    else:
        ctx.suggest("skin.io.build_multisample", "skin.io.load_h5ad", "skin.help.workflow")


@tool("skin.memory.list_projects", category="memory", needs_project=False,
      summary="List every project under the project root.")
def list_projects(dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """List all known projects with dataset and step counts.

    Args:
        dry_run: No effect; this tool never writes.
        seed: Unused.
    """
    ctx.summary = {
        "project_root": str(CONFIG.project_root),
        "active": get_active_project(),
        "projects": [
            {"project_id": p["project_id"], "name": p.get("name"),
             "organism": p.get("organism"), "n_datasets": p.get("n_datasets"),
             "n_steps": p.get("n_steps"), "created_at": p.get("created_at")}
            for p in store.list_projects()
        ],
    }
    ctx.suggest("skin.memory.open_project")


@tool("skin.memory.brief", category="memory",
      summary="THE resume call: full project state in <=1500 tokens.")
def brief(project_id: str = "", max_steps: int = 5, dry_run: bool = False,
          seed: int = 0, *, ctx: Ctx) -> None:
    """Return the project state: datasets, annotations, parameters, recent steps, flags.

    Put this at the top of every session and after any long gap. It is cheap.

    Args:
        project_id: Defaults to the active project.
        max_steps: How many recent steps to include.
        dry_run: No effect.
        seed: Unused.
    """
    ctx.summary = recall.brief(ctx.project_id, max_steps=max_steps)
    nxt = ctx.summary.get("workflow", {}).get("next_suggested_tools", [])
    ctx.suggest(*[d["tool"] for d in nxt])


@tool("skin.memory.record_annotation", category="memory",
      summary="Record cluster->label assignments with evidence and rationale.")
def record_annotation(
    dataset_id: str,
    obs_key: str,
    mapping: dict[str, str],
    rationale: str,
    evidence: dict[str, Any] | None = None,
    confidence: float = 0.8,
    author: str = "model",
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Record why each cluster got the label it got. This is the gold.

    Call it alongside skin.annotate.apply_labels — apply_labels writes the obs
    column, this writes the justification that survives the session.

    Args:
        dataset_id: Handle the labels apply to.
        obs_key: Target obs column, e.g. "cell_types" or "macrophage_subtypes".
        mapping: {cluster_id: label}, e.g. {"5": "LAM-I"}.
        rationale: Why. Free text. Cite the markers you used.
        evidence: Optional {cluster: {"top_markers": [...], "scores": {...},
            "artifact_ids": [...]}}.
        confidence: 0-1, your confidence in the assignment.
        author: "model:qwen3-35b", "model:opus-5", or "user:sujit".
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    if not mapping:
        raise BadParam("mapping is empty", remedy="Pass at least one {cluster: label} pair.")
    if not rationale.strip():
        raise BadParam(
            "rationale is required",
            remedy="A label without a rationale is unauditable. One sentence naming the "
                   "markers you used is enough.",
        )
    ev = evidence or {}
    if dry_run:
        ctx.summary = {"would_record": len(mapping), "obs_key": obs_key}
        return
    ids = []
    for cluster, label in mapping.items():
        ids.append(store.record_annotation(
            ctx.project_id, dataset_id, obs_key, str(cluster), str(label),
            ev.get(str(cluster), {}), rationale, float(confidence), author))
    ctx.dataset_id = dataset_id
    ctx.summary = {"n_recorded": len(ids), "annotation_ids": ids, "obs_key": obs_key,
                   "labels": sorted(set(mapping.values()))}
    ctx.suggest("skin.annotate.apply_labels", "skin.annotate.contamination_audit")


@tool("skin.memory.get_annotations", category="memory",
      summary="Read back recorded annotations with their rationales.")
def get_annotations(dataset_id: str = "", obs_key: str = "", include_superseded: bool = False,
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Retrieve annotations, optionally including superseded ones.

    Args:
        dataset_id: Filter by handle.
        obs_key: Filter by obs column.
        include_superseded: Include revised-away labels (the audit trail).
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    rows = store.get_annotations(ctx.project_id, dataset_id, obs_key, include_superseded)
    ctx.summary = {
        "n": len(rows),
        "annotations": [
            {"id": r["annotation_id"], "obs_key": r["obs_key"], "cluster": r["cluster"],
             "label": r["label"], "confidence": r["confidence"], "author": r["author"],
             "superseded_by": r["superseded_by"], "rationale": (r["rationale"] or "")[:200]}
            for r in rows
        ],
    }


@tool("skin.memory.revise_annotation", category="memory",
      summary="Supersede an annotation with a new label; never deletes.")
def revise_annotation(annotation_id: int, new_label: str, rationale: str,
                      author: str = "model", project_id: str = "", dry_run: bool = False,
                      seed: int = 0, *, ctx: Ctx) -> None:
    """Change a label, keeping the original and the reason for the change.

    Args:
        annotation_id: Id from record_annotation / get_annotations.
        new_label: The corrected label.
        rationale: Why the original was wrong.
        author: Who is revising.
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    if not rationale.strip():
        raise BadParam("rationale is required for a revision",
                       remedy="Say what changed your mind. That is the point of the record.")
    if dry_run:
        ctx.summary = {"would_revise": annotation_id, "to": new_label}
        return
    try:
        new_id = store.revise_annotation(ctx.project_id, int(annotation_id), new_label,
                                         rationale, author)
    except KeyError as e:
        raise NotFound(str(e), remedy="Call skin.memory.get_annotations to list valid ids.") from e
    ctx.summary = {"superseded": annotation_id, "new_annotation_id": new_id, "label": new_label}


@tool("skin.memory.set_param", category="memory",
      summary="Record a named, reusable threshold with its rationale.")
def set_param(name: str, value: Any, rationale: str, scope: str = "global",
              set_by: str = "model", project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Record a parameter so later sessions and figures stay consistent.

    Parameters are DEFAULTS OFFERED to you via skin.memory.brief. They are never
    applied automatically — pass them explicitly to a tool if you want them used.

    Args:
        name: Dotted name, e.g. "qc.min_genes" or "palette.Type".
        value: Any JSON value.
        rationale: Why this value. Required.
        scope: "global", or a scoped key like "sample:B_D7_1".
        set_by: Who set it.
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    if not rationale.strip():
        raise BadParam("rationale is required",
                       remedy="A threshold with no recorded reason is not reproducible.")
    if dry_run:
        ctx.summary = {"would_set": name, "scope": scope, "value": value}
        return
    store.set_param(ctx.project_id, name, value, scope, set_by, rationale)
    ctx.summary = {"name": name, "scope": scope, "value": value}


@tool("skin.memory.get_param", category="memory", summary="Read a recorded parameter.")
def get_param(name: str, scope: str = "global", project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Read one parameter, falling back from a scoped value to the global one.

    Args:
        name: Parameter name.
        scope: Scope to look in first.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    p = store.get_param(ctx.project_id, name, scope)
    ctx.summary = {"found": p is not None, "parameter": p}


@tool("skin.memory.list_params", category="memory", summary="List all recorded parameters.")
def list_params(project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """List every parameter on record for this project.

    Args:
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    rows = store.list_params(ctx.project_id)
    ctx.summary = {"n": len(rows), "parameters": [
        {"name": r["name"], "scope": r["scope"], "value": r["value"],
         "rationale": (r.get("rationale") or "")[:140]} for r in rows]}


@tool("skin.memory.record_decision", category="memory",
      summary="Record a non-parametric choice and the alternatives rejected.")
def record_decision(question: str, choice: str, rationale: str,
                    alternatives: list[str] | None = None, author: str = "model",
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Record a judgement call: which cluster to drop, which DE method, which root.

    Args:
        question: What was being decided.
        choice: What was chosen.
        rationale: Why.
        alternatives: What was considered and rejected.
        author: Who decided.
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    if dry_run:
        ctx.summary = {"would_record": question[:80]}
        return
    did = store.record_decision(ctx.project_id, question, choice, alternatives or [],
                                rationale, author)
    ctx.summary = {"decision_id": did, "question": question[:120], "choice": choice[:80]}


@tool("skin.memory.note", category="memory", summary="Append a free-text note.")
def note(tag: str, body: str, author: str = "model", project_id: str = "",
         dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Leave a note for your future self. Indexed for full-text search.

    Args:
        tag: Short category, e.g. "qc", "todo", "reviewer".
        body: The note.
        author: Who wrote it.
        project_id: Defaults to the active project.
        dry_run: Validate only.
        seed: Unused.
    """
    if dry_run:
        ctx.summary = {"would_note": tag}
        return
    nid = store.add_note(ctx.project_id, tag, body, author)
    ctx.summary = {"note_id": nid, "tag": tag}


@tool("skin.memory.search", category="memory",
      summary="Full-text search over rationales, decisions and notes.")
def search(query: str, limit: int = 15, project_id: str = "", dry_run: bool = False,
           seed: int = 0, *, ctx: Ctx) -> None:
    """Search everything free-text that was ever written about this project.

    Args:
        query: FTS5 query, e.g. "Trem2" or "why dropped cluster".
        limit: Max hits.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    hits = store.search(ctx.project_id, query, limit)
    ctx.summary = {"query": query, "n_hits": len(hits), "hits": hits}


@tool("skin.memory.timeline", category="memory", summary="Compact provenance log.")
def timeline(limit: int = 25, project_id: str = "", dry_run: bool = False, seed: int = 0,
             *, ctx: Ctx) -> None:
    """The last `limit` steps: tool, ok, duration, key params.

    Args:
        limit: How many steps.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    rows = store.get_steps(ctx.project_id, limit=limit)
    out = []
    for r in rows:
        params = json.loads(r["params_json"] or "{}")
        keep = {k: v for k, v in params.items()
                if k not in ("project_id", "dry_run", "seed") and v not in (None, "", [], {})}
        out.append({"id": r["step_id"], "tool": r["tool"], "ok": bool(r["ok"]),
                    "s": round(r["duration_s"], 2), "params": keep,
                    "error": (r.get("error") or "")[:100] or None})
    ctx.summary = {"n": len(out), "steps": out}


@tool("skin.memory.export", category="memory",
      summary="Export the whole project memory as markdown or JSON.")
def export(fmt: str = "md", project_id: str = "", dry_run: bool = False, seed: int = 0,
           *, ctx: Ctx) -> None:
    """Write a lab-notebook document of every annotation, decision and step.

    Args:
        fmt: "md" or "json".
        project_id: Defaults to the active project.
        dry_run: Report the target path without writing.
        seed: Unused.
    """
    if fmt not in ("md", "json"):
        raise BadParam(f"fmt must be 'md' or 'json', got {fmt!r}")
    root = CONFIG.ensure_project_dirs(ctx.project_id)
    path = root / f"memory_export.{fmt}"
    if dry_run:
        ctx.summary = {"would_write": str(path)}
        return
    if fmt == "md":
        path.write_text(recall.as_markdown(ctx.project_id), encoding="utf-8")
    else:
        payload = {
            "project": store.get_project(ctx.project_id),
            "datasets": store.list_datasets(ctx.project_id),
            "annotations": store.get_annotations(ctx.project_id, include_superseded=True),
            "parameters": store.list_params(ctx.project_id),
            "decisions": store.get_decisions(ctx.project_id, limit=10000),
            "notes": store.get_notes(ctx.project_id, limit=10000),
            "steps": store.get_steps(ctx.project_id, limit=10000),
        }
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    ctx.add_artifact("text", path, caption=f"project memory export ({fmt})")
    ctx.summary = {"path": str(path), "format": fmt}
    ctx.suggest("skin.export.report", "skin.export.notebook")
