"""`skin.help.*` — tool discovery and state-aware next steps.

With ~80 tools, discovery matters more than any single tool. `workflow` reads
the project state and tells the caller what to do next; `list_tools` keeps the
catalogue out of the system prompt.
"""

from __future__ import annotations

from typing import Any

from ..config import CONFIG
from ..memory import recall
from ._base import REGISTRY, Ctx, tool

CATEGORY_BLURB = {
    "memory": "Project memory: open/resume, brief, annotations, parameters, decisions, notes.",
    "io": "Ingest and handles: load 10x/h5ad/mtx/Seurat, describe, lineage, save.",
    "qc": "Per-sample QC statistics, threshold discovery, filtering, ambient RNA.",
    "meta": "Sample metadata, categorical ordering, colour palettes.",
    "doublet": "Doublet calling (always per sample) and cluster-level enrichment.",
    "integrate": "Normalization, HVG, PCA, Harmony and alternative batch correction.",
    "cluster": "Neighbours, UMAP, Leiden, resolution sweep, marker ranking, cluster QC.",
    "annotate": "Lineage scoring, label proposal, contamination audit, refinement loop.",
    "sub": "Subclustering with proper re-normalization; drop clusters; map labels back.",
    "de": "Differential expression: pseudobulk (default) and cell-wise (exploratory).",
    "enrich": "ORA, GSEA, TF activity, per-cell signature and Hallmark scoring.",
    "abundance": "Differential abundance: Milo, scCODA, per-sample proportions.",
    "traj": "Trajectories: principal graph, PAGA, DPT, CellRank real-time kernel.",
    "ccc": "Cell-cell communication: LIANA and CellChat, with a min-cells floor.",
    "plot": "Every figure: UMAPs, dotplots, volcano grids, enrichment tiles, legends.",
    "atlas": "Reference atlases: CellTypist, Census, label transfer, orthologs.",
    "runtime": "Runtime status, R container, version manifest, vetted R scripts.",
    "export": "Executable notebook, PI report, methods draft, reproducibility bundle.",
    "help": "This.",
}


@tool("skin.help.list_tools", category="help",
      summary="List available tools, optionally filtered by category.")
def list_tools(category: str = "", project_id: str = "", dry_run: bool = False,
               seed: int = 0, *, ctx: Ctx) -> None:
    """Browse the tool catalogue without putting it all in your context.

    Args:
        category: One of memory, io, qc, meta, doublet, integrate, cluster,
            annotate, sub, de, enrich, abundance, traj, ccc, plot, atlas,
            runtime, export, help. Empty = the category index only.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    enabled = {n: s for n, s in REGISTRY.items() if CONFIG.namespace_enabled(n)}
    by_cat: dict[str, list[dict[str, Any]]] = {}
    for name, spec in sorted(enabled.items()):
        by_cat.setdefault(spec.category, []).append({
            "tool": name, "summary": spec.summary[:110],
            **({"destructive": True} if spec.destructive else {}),
            **({"needs_network": True} if spec.needs_network else {}),
            **({"needs_r": True} if spec.needs_r else {}),
        })

    if category:
        if category not in by_cat:
            ctx.summary = {"error": f"unknown category {category!r}",
                           "categories": sorted(by_cat)}
            return
        ctx.summary = {"category": category, "blurb": CATEGORY_BLURB.get(category, ""),
                       "n_tools": len(by_cat[category]), "tools": by_cat[category]}
    else:
        ctx.summary = {
            "profile": CONFIG.profile,
            "n_tools_total": len(enabled),
            "categories": {c: {"n": len(v), "about": CATEGORY_BLURB.get(c, "")}
                           for c, v in sorted(by_cat.items())},
            "hint": "Call skin.help.list_tools(category='qc') for one category's tools, "
                    "or skin.help.workflow for state-aware next steps.",
        }
    ctx.suggest("skin.help.workflow", "skin.memory.brief")


@tool("skin.help.workflow", category="help",
      summary="State-aware next steps: reads the project and tells you what to do next.")
def workflow(project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """What should I do next? Answered from the actual project state.

    Args:
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    st = recall.workflow_state(ctx.project_id)
    nxt = st.pop("next_suggested_tools", [])
    done = [k.replace("has_", "") for k, v in st.items() if v]
    todo = [k.replace("has_", "") for k, v in st.items() if not v]
    ctx.summary = {
        "completed": done, "not_yet_done": todo, "next_steps": nxt,
        "sops": ["sop_new_project", "sop_qc_and_filter", "sop_first_pass_annotation",
                 "sop_decontamination_loop", "sop_subcluster", "sop_pseudobulk_de",
                 "sop_trajectory", "sop_abundance", "sop_communication",
                 "sop_finalize_and_export"],
        "hint": "The SOP prompts are short numbered procedures naming the exact tools. "
                "Load the one that matches what you are about to do.",
    }
    ctx.suggest(*[d["tool"] for d in nxt])


@tool("skin.help.explain_tool", category="help",
      summary="Full docstring and argument list for one tool.")
def explain_tool(tool_name: str, project_id: str = "", dry_run: bool = False,
                 seed: int = 0, *, ctx: Ctx) -> None:
    """Read one tool's full documentation without loading every docstring.

    Args:
        tool_name: Full dotted name, e.g. "skin.de.pseudobulk".
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    import inspect

    spec = REGISTRY.get(tool_name)
    if spec is None:
        near = [n for n in REGISTRY if tool_name.split(".")[-1] in n]
        ctx.summary = {"error": f"unknown tool {tool_name!r}", "did_you_mean": near[:8],
                       "hint": "skin.help.list_tools lists everything."}
        return
    sig = inspect.signature(spec.fn)
    ctx.summary = {
        "tool": tool_name, "category": spec.category,
        "destructive": spec.destructive, "needs_network": spec.needs_network,
        "needs_r": spec.needs_r,
        "parameters": {k: {"default": (None if p.default is inspect.Parameter.empty
                                       else repr(p.default)),
                           "required": p.default is inspect.Parameter.empty}
                       for k, p in sig.parameters.items()},
        "doc": (spec.fn.__doc__ or "")[:2500],
    }
