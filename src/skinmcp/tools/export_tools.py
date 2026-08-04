"""`skin.export.*` — notebook, report, bundle, methods draft.

Reproducibility is the product. `notebook` assembles every logged `step.code`
into a linear, executable document with handles resolved to real file paths, so
the exported notebook regenerates the results without the MCP server in the
loop. If a result cannot be regenerated from that notebook, it is a bug.
"""

from __future__ import annotations

import json
import logging
import zipfile
from pathlib import Path
from typing import Any

from ..config import CONFIG
from ..errors import BadParam
from ..memory import recall, store
from ..runtimes import bridge, python_manifest
from ._base import Ctx, tool

logger = logging.getLogger(__name__)

R_TOOLS = ("deseq2_r", "milo_r", "cellchat", "scdblfinder", "estimate_ambient",
           "exec_r", "seurat_rds")


def _is_r_step(tool_name: str) -> bool:
    return any(t in tool_name for t in R_TOOLS)


def _comment_out(code: str, why: str) -> str:
    return why + "\n" + "\n".join(f"# {ln}" for ln in code.splitlines())


def _step_code(step: dict[str, Any]) -> str:
    """Build one runnable cell: bind `adata` from the step's input, run, rebind.

    Without this the cells are orphans — each tool's code operates on `adata`,
    but nothing in the notebook ever binds it. The step rows record which handle
    went in and which came out, so the exporter can stitch the chain together.
    """
    ins = json.loads(step.get("inputs_json") or "{}") or {}
    outs = json.loads(step.get("outputs_json") or "{}") or {}
    params = json.loads(step.get("params_json") or "{}") or {}
    code = step["code"]

    if _is_r_step(step["tool"]):
        return _comment_out(
            code,
            "# This step ran in the pinned R container. Reproduce it with\n"
            "#   skin.runtime.exec_r(script_id=..., params=...)\n"
            "# or use the .Rmd export, which emits a real knitr chunk.")
    if outs.get("code_executable") is False:
        return _comment_out(code, "# Reference only — see the note below.")

    in_id = ins.get("dataset_id") or params.get("dataset_id")
    out_id = outs.get("dataset_id")
    head, tail = [], []
    if in_id and isinstance(in_id, str) and in_id.startswith("ds_"):
        head.append(f"adata = load({in_id!r})")
    if out_id and out_id != in_id:
        tail.append(f"# -> {out_id}"
                    + (f"   (server handle; reload with load({out_id!r}))" if out_id else ""))
    return "\n".join([*head, code.rstrip(), *tail]) + "\n"


def _params_cell(project_id: str) -> str:
    """A params cell with every handle resolved to a real path — no MCP calls."""
    proj = store.get_project(project_id) or {}
    ds = store.list_datasets(project_id)
    params = store.list_params(project_id)
    lines = [
        "# ─── Resolved parameters ───────────────────────────────────────────────",
        "# Every dataset handle below is a real file path. This notebook does not",
        "# call the MCP server; it re-runs the analysis directly.",
        "from pathlib import Path",
        "import anndata as ad, scanpy as sc, numpy as np, pandas as pd",
        "",
        f"PROJECT_ID = {project_id!r}",
        f"PROJECT_ROOT = Path({str(CONFIG.project_dir(project_id))!r})",
        f"ORGANISM = {proj.get('organism')!r}",
        "FIGDIR = PROJECT_ROOT / 'figures'",
        "TABLEDIR = PROJECT_ROOT / 'tables'",
        "",
        "HANDLES = {",
    ]
    for d in ds:
        alias = f"  # {d['label']}" if d.get("label") else ""
        lines.append(f"    {d['dataset_id']!r}: {d['path']!r},{alias}")
    lines.append("}")
    if params:
        lines += ["", "PARAMS = {"]
        for p in params:
            lines.append(f"    {p['name']!r}: {p['value']!r},  # {p['scope']}: "
                         f"{(p.get('rationale') or '')[:60]}")
        lines.append("}")
    lines += ["", "def load(handle):",
              '    """Resolve a skin-mcp handle to an AnnData."""',
              "    return ad.read_h5ad(HANDLES[handle])", ""]
    return "\n".join(lines)


def _header_md(project_id: str) -> str:
    proj = store.get_project(project_id) or {}
    m = python_manifest.full_manifest()
    pkgs = m["python"]["packages"]
    keep = ("scanpy", "anndata", "numpy", "scipy", "pandas", "pydeseq2", "harmonypy",
            "gseapy", "leidenalg", "matplotlib", "skin-mcp")
    ver = "\n".join(f"| `{k}` | {pkgs[k]} |" for k in keep if k in pkgs)
    r = bridge.runtime_status()
    return (
        f"# {proj.get('name', project_id)}\n\n"
        f"Exported from **skin-mcp** on {store.now()}.\n\n"
        f"- **project_id**: `{project_id}`\n"
        f"- **organism**: {proj.get('organism')}\n"
        f"- **created**: {proj.get('created_at')}\n\n"
        f"{proj.get('description') or ''}\n\n"
        + (f"## Experimental design\n\n{proj['design_notes']}\n\n"
           if proj.get("design_notes") else "")
        + f"## Session info\n\n| package | version |\n|---|---|\n"
          f"| python | {m['python']['version']} |\n{ver}\n\n"
          f"R runtime: `{r['backend']}` (available={r['available']}); "
          f"Python: uv.\n\n"
          f"Pins: {m['pinned']}\n"
    )


@tool("skin.export.notebook", category="export",
      summary="Assemble every logged step into a runnable .ipynb / .Rmd.")
def notebook(fmt: str = "ipynb", include_steps: list[int] | None = None, path: str = "",
             skip_failed: bool = True, project_id: str = "", dry_run: bool = False,
             seed: int = 0, *, ctx: Ctx) -> None:
    """Export the whole project as an executable notebook.

    Each step becomes a markdown cell (tool, timestamp, recorded rationale)
    followed by the code cell that reproduces it. Figures are regenerated by the
    code rather than embedded from cache.

    Args:
        fmt: "ipynb", "rmd", or "both".
        include_steps: Specific step_ids. Empty = every successful step.
        path: Output path. Empty = {project}/notebooks/analysis.{ext}.
        skip_failed: Omit steps that failed.
        project_id: Defaults to the active project.
        dry_run: Report the step count and target path.
        seed: Unused.
    """
    if fmt not in ("ipynb", "rmd", "both"):
        raise BadParam("fmt must be ipynb|rmd|both")
    steps = store.all_steps_for_export(ctx.project_id) if skip_failed else \
        store.get_steps(ctx.project_id, limit=10000, include_code=True)
    if include_steps:
        want = set(int(x) for x in include_steps)
        steps = [s for s in steps if int(s["step_id"]) in want]
    steps = [s for s in steps if (s.get("code") or "").strip()]

    nb_dir = CONFIG.ensure_project_dirs(ctx.project_id) / "notebooks"
    ann = {a["dataset_id"]: a for a in store.get_annotations(ctx.project_id)}
    if dry_run:
        ctx.summary = {"n_steps": len(steps), "formats": fmt,
                       "target_dir": str(nb_dir)}
        return
    if not steps:
        ctx.warn("No steps carry reproducing code yet. Run some analysis first.")

    made = []
    if fmt in ("ipynb", "both"):
        made.append(_write_ipynb(ctx, steps, nb_dir, path, ann))
    if fmt in ("rmd", "both"):
        made.append(_write_rmd(ctx, steps, nb_dir, path))

    n_r = sum(1 for s in steps if _is_r_step(s["tool"]))
    if n_r:
        ctx.warn(f"{n_r} steps ran through the R bridge. In the .ipynb they appear as "
                 f"documented placeholders naming the vetted script and its params — rerun "
                 f"them with skin.runtime.exec_r, or use the .Rmd export, which emits real "
                 f"knitr chunks.")
    ctx.summary = {"files": made, "n_steps": len(steps), "n_r_steps": n_r}
    ctx.suggest("skin.export.report", "skin.export.bundle")


def _write_ipynb(ctx: Ctx, steps: list[dict[str, Any]], nb_dir: Path, path: str,
                 ann: dict[str, Any]) -> str:
    import nbformat as nbf

    nb = nbf.v4.new_notebook()
    cells = [nbf.v4.new_markdown_cell(_header_md(ctx.project_id)),
             nbf.v4.new_code_cell(_params_cell(ctx.project_id))]
    for s in steps:
        params = json.loads(s.get("params_json") or "{}")
        keep = {k: v for k, v in params.items()
                if k not in ("project_id", "dry_run") and v not in (None, "", [], {})}
        md = [f"## step {s['step_id']} — `{s['tool']}`", "",
              f"*{s['started_at']}  ·  {s['duration_s']:.2f}s*", ""]
        if keep:
            md += ["```json", json.dumps(keep, indent=2, default=str)[:900], "```", ""]
        out = json.loads(s.get("outputs_json") or "{}")
        if out.get("dataset_id"):
            a = ann.get(out["dataset_id"])
            if a and a.get("rationale"):
                md += [f"> **Recorded rationale:** {a['rationale']}", ""]
        cells.append(nbf.v4.new_markdown_cell("\n".join(md)))
        cells.append(nbf.v4.new_code_cell(_step_code(s)))

    nb["cells"] = cells
    nb["metadata"] = {"kernelspec": {"name": "python3", "display_name": "Python 3",
                                     "language": "python"},
                      "language_info": {"name": "python"},
                      "skinmcp": {"project_id": ctx.project_id, "exported_at": store.now()}}
    p = Path(path) if path else nb_dir / "analysis.ipynb"
    p.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(p))
    ctx.add_artifact("notebook", p, caption="executable analysis notebook")
    return str(p)


def _write_rmd(ctx: Ctx, steps: list[dict[str, Any]], nb_dir: Path, path: str) -> str:
    proj = store.get_project(ctx.project_id) or {}
    out = [
        "---",
        f'title: "{proj.get("name", ctx.project_id)}"',
        f'date: "{store.now()}"',
        "output: html_document",
        "---",
        "",
        "```{r setup, include=FALSE}",
        "library(reticulate)",
        "knitr::opts_chunk$set(echo = TRUE, warning = FALSE, message = FALSE)",
        "```",
        "",
        _header_md(ctx.project_id),
        "",
        "```{python params}",
        _params_cell(ctx.project_id),
        "```",
        "",
    ]
    for s in steps:
        out += [f"## step {s['step_id']} — `{s['tool']}`", "",
                f"*{s['started_at']} · {s['duration_s']:.2f}s*", ""]
        if _is_r_step(s["tool"]):
            out += ["```{r}", "# R-bridge step; see runtimes/r/scripts/",
                    *[f"# {ln}" for ln in s["code"].splitlines()], "```", ""]
        else:
            out += ["```{python}", _step_code(s), "```", ""]
    p = Path(path).with_suffix(".Rmd") if path else nb_dir / "analysis.Rmd"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("\n".join(out), encoding="utf-8")
    ctx.add_artifact("notebook", p, caption="R Markdown analysis document")
    return str(p)


@tool("skin.export.report", category="export",
      summary="PI-facing lab-notebook narrative: annotations, decisions, figures, warnings.")
def report(fmt: str = "md", path: str = "", include_figures: bool = True,
           project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Write the human-readable project narrative.

    Args:
        fmt: "md" or "html".
        path: Output path. Empty = {project}/report.{ext}.
        include_figures: Embed figure links inline.
        project_id: Defaults to the active project.
        dry_run: Report the target path only.
        seed: Unused.
    """
    if fmt not in ("md", "html"):
        raise BadParam("fmt must be md|html")
    root = CONFIG.ensure_project_dirs(ctx.project_id)
    p = Path(path) if path else root / f"report.{fmt}"
    if dry_run:
        ctx.summary = {"would_write": str(p)}
        return

    anns = store.get_annotations(ctx.project_id, include_superseded=True)
    decs = store.get_decisions(ctx.project_id, limit=500)
    notes = store.get_notes(ctx.project_id, limit=500)
    steps = store.get_steps(ctx.project_id, limit=5000)
    figs = store.list_artifacts(ctx.project_id, kind="figure", limit=200)
    tables = store.list_artifacts(ctx.project_id, kind="table", limit=200)

    warn_steps = [s for s in steps if not s["ok"]]
    md = [_header_md(ctx.project_id), "",
          "## What was done", "",
          f"{len(steps)} recorded steps, {len(store.list_datasets(ctx.project_id))} dataset "
          f"handles, {len(figs)} figures, {len(tables)} tables.", ""]

    if anns:
        md += ["## Cell type annotations", ""]
        by_key: dict[str, list[dict[str, Any]]] = {}
        for a in anns:
            by_key.setdefault(f"{a['obs_key']} ({a['dataset_id']})", []).append(a)
        for key, rows in by_key.items():
            md += [f"### `{key}`", "", "| cluster | label | confidence | author | status |",
                   "|---|---|---|---|---|"]
            for a in rows:
                st = "superseded" if a.get("superseded_by") else "current"
                md.append(f"| {a['cluster']} | **{a['label']}** | {a.get('confidence')} | "
                          f"{a.get('author')} | {st} |")
            md += [""]
            seen = set()
            for a in rows:
                r = (a.get("rationale") or "").strip()
                if r and r not in seen:
                    seen.add(r)
                    md += [f"> {r}", ""]

    if decs:
        md += ["## Decisions", ""]
        for d in decs:
            md += [f"### {d['question']}", "",
                   f"- **Chose:** {d['choice']}",
                   f"- **Alternatives considered:** {d.get('alternatives_json')}",
                   f"- **Rationale:** {d.get('rationale')}",
                   f"- **Author:** {d.get('author')}  ·  {d.get('created_at')}", ""]

    if include_figures and figs:
        md += ["## Figures", ""]
        for f in figs:
            png = Path(f["path"]).with_suffix(".png")
            md += [f"### {f.get('caption') or Path(f['path']).stem}", "",
                   f"![{f['artifact_id']}]({png})", "",
                   f"`{f['artifact_id']}` · step {f.get('step_id')} · `{png}`", ""]

    if tables:
        md += ["## Tables", ""]
        for t in tables[:40]:
            md += [f"- `{t['artifact_id']}` — {t.get('caption') or ''} → `{t['path']}`"]
        md += [""]

    if notes:
        md += ["## Notes", ""]
        for n in reversed(notes):
            md += [f"- **[{n['tag']}]** ({n['created_at']}) {n['body']}"]
        md += [""]

    if warn_steps:
        md += ["## Failed steps", "", "| step | tool | error |", "|---|---|---|"]
        for s in warn_steps:
            md.append(f"| {s['step_id']} | `{s['tool']}` | {(s.get('error') or '')[:150]} |")
        md += [""]

    md += ["## Provenance", "",
           f"Full step log: `skin://project/{ctx.project_id}/provenance`", "",
           "```", recall.as_markdown(ctx.project_id)[-4000:], "```"]

    text = "\n".join(md)
    if fmt == "html":
        text = ("<!doctype html><meta charset='utf-8'>"
                "<style>body{font-family:system-ui,sans-serif;max-width:56rem;margin:2rem auto;"
                "padding:0 1rem;line-height:1.6}img{max-width:100%}table{border-collapse:collapse}"
                "td,th{border:1px solid #ddd;padding:4px 8px}pre{overflow-x:auto;background:#f6f8fa;"
                "padding:1rem}blockquote{border-left:3px solid #ccc;margin-left:0;padding-left:1rem;"
                "color:#444}</style>\n<pre>" + text.replace("<", "&lt;") + "</pre>")
    p.write_text(text, encoding="utf-8")
    ctx.add_artifact("text", p, caption="PI-facing project report")
    ctx.summary = {"path": str(p), "format": fmt, "n_annotations": len(anns),
                   "n_decisions": len(decs), "n_figures": len(figs),
                   "n_failed_steps": len(warn_steps)}
    ctx.suggest("skin.export.bundle", "skin.export.methods_paragraph")


@tool("skin.export.methods_paragraph", category="export",
      summary="Draft a methods section from the provenance log. DRAFT ONLY.")
def methods_paragraph(project_id: str = "", dry_run: bool = False, seed: int = 0,
                      *, ctx: Ctx) -> None:
    """Draft methods text with the real versions and parameters filled in.

    This is a DRAFT. It reports what the provenance log says happened; it cannot
    know what you meant, and it will not caveat your biology for you.

    Args:
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    proj = store.get_project(ctx.project_id) or {}
    steps = store.all_steps_for_export(ctx.project_id)
    by_tool: dict[str, dict[str, Any]] = {}
    for s in steps:
        by_tool.setdefault(s["tool"], json.loads(s.get("params_json") or "{}"))
    pkgs = python_manifest.capture()
    org = proj.get("organism", "mouse")

    def pv(name: str) -> str:
        return f"{name} v{pkgs[name]}" if name in pkgs else name

    parts: list[str] = []
    parts.append(
        f"Single-cell RNA-seq data ({'Mus musculus' if org == 'mouse' else 'Homo sapiens'}) "
        f"were processed with {pv('scanpy')} on {pv('anndata')} objects, orchestrated by "
        f"skin-mcp v{pkgs.get('skin-mcp', '0.1.0')} with full provenance logging.")

    if "skin.qc.apply_filters" in by_tool:
        th = by_tool["skin.qc.apply_filters"].get("thresholds", {})
        mt = th.get("max_pct_mt")
        parts.append(
            f"Cells were filtered to {th.get('min_genes', '?')}–{th.get('max_genes', '?')} "
            f"detected genes and {th.get('min_counts', '?')}–{th.get('max_counts', '?')} UMIs"
            + (f", with a maximum mitochondrial fraction of {mt}%" if mt is not None
               else " (the mitochondrial filter was not applied for this chemistry)")
            + "; thresholds were derived per sample by median-absolute-deviation outlier "
              "detection on the log1p scale.")
    if "skin.doublet.call" in by_tool:
        m = by_tool["skin.doublet.call"].get("method", "scrublet")
        parts.append(f"Doublets were called per sample with {m}, with the expected rate taken "
                     f"from the 10x multiplet table for each library's recovered cell count.")
    if "skin.integrate.preprocess" in by_tool:
        p = by_tool["skin.integrate.preprocess"]
        ex = p.get("exclude_gene_groups") or []
        parts.append(
            f"Counts were normalized to {p.get('target_sum', 1e4):g} counts per cell and "
            f"log1p-transformed; {p.get('n_hvg', 2000)} highly variable genes were selected "
            f"({p.get('hvg_flavor', 'seurat')} flavor), scaled to a maximum of "
            f"{p.get('scale_max', 10)}, and reduced to {p.get('n_comps', 50)} principal "
            f"components"
            + (f". Genes in the groups {ex} were excluded from the feature space prior to "
               f"variable-gene selection." if ex else "."))
    if "skin.integrate.harmony" in by_tool:
        p = by_tool["skin.integrate.harmony"]
        parts.append(f"Batch effects across {p.get('batch_key', 'Sample')} were corrected with "
                     f"Harmony ({pv('harmonypy')}).")
    if "skin.cluster.leiden" in by_tool:
        p = by_tool["skin.cluster.leiden"]
        nk = by_tool.get("skin.cluster.neighbors", {})
        parts.append(f"A k-nearest-neighbour graph (k={nk.get('n_neighbors', 15)}, "
                     f"{nk.get('n_pcs', 30)} components) was built on the integrated "
                     f"embedding and clustered with the Leiden algorithm "
                     f"({pv('leidenalg')}) at resolution {p.get('resolution', 0.8)}.")
    if "skin.annotate.apply_labels" in by_tool:
        n_ann = len(store.get_annotations(ctx.project_id))
        parts.append(f"Clusters were annotated by canonical marker expression and module "
                     f"scoring against curated skin lineage sets; {n_ann} cluster-to-label "
                     f"assignments were recorded with their supporting evidence.")
    if "skin.de.pseudobulk" in by_tool:
        p = by_tool["skin.de.pseudobulk"]
        c = p.get("contrast", ["A", "B"])
        parts.append(
            f"Differential expression was performed on pseudobulk profiles: raw counts were "
            f"summed within each sample × cell-type unit (minimum {p.get('min_cells', 10)} "
            f"cells per unit) and tested with {pv('pydeseq2')} using the design "
            f"`{p.get('design', '~ condition')}`, contrasting {c[0]} against {c[1]}. Cell "
            f"types with fewer than {p.get('min_samples_per_arm', 3)} biological replicates "
            f"in either arm were excluded rather than tested. Log2 fold changes were shrunk; "
            f"significance was called at FDR < 0.05 and |log2FC| > 0.5.")
    elif "skin.de.wilcoxon" in by_tool:
        parts.append("Differential expression was assessed cell-wise with a Wilcoxon "
                     "rank-sum test. NOTE: these p-values are pseudo-replicated across cells "
                     "within a sample and are exploratory; state this or switch to pseudobulk.")
    if "skin.enrich.ora" in by_tool:
        p = by_tool["skin.enrich.ora"]
        parts.append(
            f"Over-representation analysis was performed with {pv('gseapy')} against "
            f"{p.get('library')}; fold enrichment was computed as (k/N)/(n/M) with "
            f"M = {p.get('background_size', 15000)}"
            + (f". Terms matching the preset '{p.get('exclusion_preset')}' were excluded."
               if p.get("exclusion_preset") else "."))
    if "skin.abundance.milo_py" in by_tool:
        p = by_tool["skin.abundance.milo_py"]
        parts.append(
            f"Differential abundance was tested on kNN neighbourhoods (Milo-style): "
            f"{p.get('prop', 0.1):.0%} of cells were sampled as neighbourhood indices on a "
            f"k={p.get('k', 30)} graph, counts per neighbourhood per sample were fitted with "
            f"a quasi-Poisson GLM under `{p.get('design')}` with a shared dispersion, and "
            f"p-values were Benjamini-Hochberg corrected at FDR < "
            f"{p.get('alpha_fdr', 0.1)}.")
    if "skin.traj.monocle" in by_tool:
        p = by_tool["skin.traj.monocle"]
        parts.append(
            f"Trajectories were inferred by fitting a principal graph "
            f"({p.get('n_centroids', 20)} centroids) on the {p.get('basis')} embedding "
            f"({p.get('implementation')}), rooted on {p.get('root_label')!r}; the Spearman "
            f"correlation between pseudotime and the experimental timepoint is reported "
            f"as a sanity check.")

    parts.append("All figures were produced with "
                 f"{pv('matplotlib')} at 300 dpi (PNG) and as vector PDFs with editable text "
                 "(fonttype 42). Complete parameters, package versions and the executable "
                 "code for every step are in the exported notebook and provenance log.")

    text = "\n\n".join(parts)
    if not dry_run:
        p = CONFIG.ensure_project_dirs(ctx.project_id) / "methods_draft.md"
        p.write_text("# Methods (DRAFT — generated from the provenance log)\n\n" + text,
                     encoding="utf-8")
        ctx.add_artifact("text", p, caption="methods section draft")
        ctx.summary = {"path": str(p), "n_paragraphs": len(parts), "text": text[:2500]}
    else:
        ctx.summary = {"n_paragraphs": len(parts)}
    ctx.warn("DRAFT ONLY. This reports what the provenance log recorded; it does not know "
             "what you intended, and it will not caveat your biology. Read every sentence "
             "before it goes near a manuscript.")
    ctx.suggest("skin.runtime.manifest", "skin.export.bundle")


@tool("skin.export.bundle", category="export",
      summary="Zip the notebook, figures, tables, lockfiles, manifest and memory.db.")
def bundle(path: str = "", include_objects: bool = False, project_id: str = "",
           dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Package everything needed to reproduce and review the project.

    Args:
        path: Output .zip path. Empty = {project}/{project_id}_bundle.zip.
        include_objects: Include the .h5ad objects. Large; off by default.
        project_id: Defaults to the active project.
        dry_run: Report what would be included.
        seed: Unused.
    """
    root = CONFIG.ensure_project_dirs(ctx.project_id)
    out = Path(path) if path else root / f"{ctx.project_id}_bundle.zip"

    members: list[tuple[Path, str]] = []
    for sub in ("notebooks", "figures", "tables", "runtimes"):
        d = root / sub
        if d.is_dir():
            for f in d.rglob("*"):
                if f.is_file() and f.suffix not in (".h5ad",):
                    members.append((f, str(f.relative_to(root))))
    for name in ("memory.db", "report.md", "report.html", "methods_draft.md",
                 "memory_export.md"):
        f = root / name
        if f.is_file():
            members.append((f, name))
    repo = Path(__file__).resolve().parents[3]
    for name in ("uv.lock", "pyproject.toml", "README.md", "NOTICE"):
        f = repo / name
        if f.is_file():
            members.append((f, f"env/{name}"))
    rl = Path(bridge.R_DIR) / "renv.lock"
    if rl.is_file():
        members.append((rl, "env/renv.lock"))
    if include_objects:
        for f in (root / "objects").glob("*.h5ad"):
            members.append((f, f"objects/{f.name}"))

    if dry_run:
        ctx.summary = {"would_write": str(out), "n_files": len(members),
                       "approx_mb": round(sum(f.stat().st_size for f, _ in members) / 1e6, 1)}
        return

    manifest_json = json.dumps({
        "project": store.get_project(ctx.project_id),
        "datasets": store.list_datasets(ctx.project_id),
        "runtime": python_manifest.full_manifest(),
        "r_runtime": bridge.runtime_status(),
        "exported_at": store.now(),
        "n_steps": len(store.get_steps(ctx.project_id, limit=100000)),
    }, indent=2, default=str)

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.json", manifest_json)
        for f, arc in members:
            try:
                z.write(f, arc)
            except OSError as e:
                logger.warning("skipping %s: %s", f, e)

    ctx.add_artifact("bundle", out, caption="reproducibility bundle")
    ctx.summary = {"path": str(out), "n_files": len(members) + 1,
                   "size_mb": round(out.stat().st_size / 1e6, 2),
                   "includes_objects": include_objects}
    if not include_objects:
        ctx.warn("The .h5ad objects were not included (they are large). The notebook "
                 "references them by absolute path, so a reviewer on another machine needs "
                 "them separately — re-run with include_objects=True for a self-contained "
                 "archive.")
    ctx.suggest("skin.export.report", "skin.runtime.manifest")
