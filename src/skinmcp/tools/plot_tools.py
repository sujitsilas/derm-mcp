"""`skin.plot.*` — every figure the pipeline produces.

Every plot tool writes both `.pdf` (vector, fonttype 42 so text stays editable in
Illustrator) and `.png`, registers an artifact with a sidecar `.json` carrying
the tool, params, dataset_id and step_id, and returns the artifact id and paths
— never image bytes, unless `return_image=True` for a vision-capable client.
"""

from __future__ import annotations

import base64
import logging
from pathlib import Path
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import BadParam, NotFound
from ..memory import store
from ..style import palettes as PAL
from ..style import panels
from ..style.rcparams import font_warning, savefig, style
from ._base import Ctx, require_obs, tool

logger = logging.getLogger(__name__)


def _finish(ctx: Ctx, paths: dict[str, str], caption: str, params: dict[str, Any],
            return_image: bool, group: str = "") -> str:
    aid = ctx.add_artifact("figure", paths["pdf"], caption=caption, params=params)
    fw = font_warning()
    if fw:
        ctx.warn(fw)
    ctx.summary.update({"artifact_id": aid, "paths": paths,
                        "uri": f"skin://figure/{aid}"})
    if return_image:
        b = Path(paths["png"]).read_bytes()
        if len(b) < 700_000:
            ctx.summary["png_base64"] = base64.b64encode(b).decode()
        else:
            ctx.warn("PNG too large to inline; read it from the resource URI instead.")
    return aid


@tool("skin.plot.set_style", category="plot", summary="Set the figure style profile.")
def set_style(profile: str = "standard", project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Choose the size profile for every subsequent figure, and record it.

    Args:
        profile: "standard" (UMAPs, dotplots, QC) or "large" (DE/enrichment
            panels: base font 30, axis labels 28 bold).
        project_id: Defaults to the active project.
        dry_run: Report the choice only.
        seed: Unused.
    """
    from ..config import CONFIG
    from ..style.rcparams import PROFILES

    if profile not in PROFILES:
        raise BadParam(f"profile must be one of {sorted(PROFILES)}")
    if not dry_run:
        CONFIG.style_profile = profile
        store.set_param(ctx.project_id, "style.profile", profile, "global",
                        "skin.plot.set_style", "keeps a project's figures internally consistent")
    ctx.summary = {"profile": profile, "font_warning": font_warning() or None}


@tool("skin.plot.umap", category="plot", summary="UMAP coloured by obs keys or genes.")
def umap(dataset_id: str, color: list[str], save_prefix: str = "umap", ncols: int = 2,
         point_size: float = 0.0, legend_loc: str = "right margin", cmap: str = "viridis",
         group: str = "umap", return_image: bool = False, project_id: str = "",
         dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Plot the UMAP, coloured by one or more obs columns or genes.

    Args:
        dataset_id: Handle or label with X_umap.
        color: obs columns and/or gene names.
        save_prefix: Output filename stem.
        ncols: Panels per row.
        point_size: Point size. 0 = scanpy's automatic size.
        legend_loc: "right margin", "on data", or "none".
        cmap: Colormap for continuous colours.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64 (vision-capable clients only).
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "X_umap" not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding("X_umap", list(adata.obsm.keys()))
    keys = [c for c in color if c in adata.obs.columns or c in adata.var_names]
    missing = [c for c in color if c not in keys]
    if missing:
        ctx.warn(f"not found in obs or var: {missing} (skipped)")
    if not keys:
        raise NotFound(f"none of {color} are obs columns or genes",
                       remedy="skin.io.describe lists the obs columns.")

    ctx.code = (f"sc.pl.umap(adata, color={keys!r}, ncols={ncols}, cmap={cmap!r},\n"
                f"           legend_loc={legend_loc!r}, frameon=True, show=False)\n")
    if dry_run:
        ctx.summary = {"color": keys}
        return

    with style():
        nc = min(ncols, len(keys))
        nr = int(-(-len(keys) // nc))
        fig, axes = plt.subplots(nr, nc, figsize=(6.2 * nc, 5.8 * nr), squeeze=False)
        af = axes.flatten()
        kw: dict[str, Any] = {"show": False, "frameon": True, "cmap": cmap}
        if point_size:
            kw["size"] = point_size
        for ax, k in zip(af, keys):
            sc.pl.umap(adata, color=k, ax=ax,
                       legend_loc=(legend_loc if k in adata.obs.columns else None), **kw)
            panels.style_umap_axis(ax, title=k)
        for ax in af[len(keys):]:
            ax.set_visible(False)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"color": keys, "n_panels": len(keys), "n_cells": int(adata.n_obs)}
    _finish(ctx, paths, f"UMAP coloured by {keys}", {"color": keys, "cmap": cmap},
            return_image, group)
    ctx.suggest("skin.plot.umap_split", "skin.plot.dotplot", "skin.plot.legend_only")


@tool("skin.plot.umap_split", category="plot",
      summary="UMAP faceted by a categorical (the Type x Timepoint grid).")
def umap_split(dataset_id: str, color_key: str, split_key: str, ncols: int = 4,
               second_split: str = "", save_prefix: str = "umap_split", group: str = "umap",
               return_image: bool = False, project_id: str = "", dry_run: bool = False,
               seed: int = 0, *, ctx: Ctx) -> None:
    """Facet the UMAP by one or two categoricals, with the rest of the cells as grey context.

    Args:
        dataset_id: Handle or label with X_umap.
        color_key: obs column to colour by.
        split_key: obs column to facet on (columns of the grid).
        ncols: Panels per row when there is no second split.
        second_split: Optional second obs column (rows of the grid).
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the grid shape only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, color_key)
    require_obs(adata, split_key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "X_umap" not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding("X_umap", list(adata.obsm.keys()))

    def levels(k: str) -> list[str]:
        s = adata.obs[k]
        return (list(map(str, s.cat.categories)) if isinstance(s.dtype, pd.CategoricalDtype)
                else PAL.natural_order(s.astype(str)))

    cols = levels(split_key)
    rows = levels(second_split) if second_split else [""]
    if second_split:
        require_obs(adata, second_split)
    ctx.code = (f"for r in {rows!r}:\n  for c in {cols!r}:\n"
                f"    sc.pl.umap(adata[mask], color={color_key!r}, ...)\n")
    if dry_run:
        ctx.summary = {"grid": [len(rows), len(cols)], "columns": cols, "rows": rows}
        return

    XY = np.asarray(adata.obsm["X_umap"])
    lab = adata.obs[color_key].astype(str).to_numpy()
    pal = PAL.get_from_adata(adata, color_key) or PAL.celltype_palette(sorted(set(lab)))
    sk = adata.obs[split_key].astype(str).to_numpy()
    rk = adata.obs[second_split].astype(str).to_numpy() if second_split else None

    nc = len(cols) if second_split else min(ncols, len(cols))
    nr = len(rows) if second_split else int(-(-len(cols) // nc))
    with style():
        fig, axes = plt.subplots(nr, nc, figsize=(4.6 * nc, 4.4 * nr), squeeze=False)
        af = axes.flatten()
        panels_list = ([(r, c) for r in rows for c in cols] if second_split
                       else [("", c) for c in cols])
        counts = {}
        for ax, (r, c) in zip(af, panels_list):
            m = sk == c
            if second_split:
                m = m & (rk == r)
            ax.scatter(XY[:, 0], XY[:, 1], s=2, c=PAL.CONTEXT_GREY, linewidths=0,
                       rasterized=True, zorder=1)
            for lv in sorted(set(lab[m])):
                mm = m & (lab == lv)
                ax.scatter(XY[mm, 0], XY[mm, 1], s=4, c=pal.get(lv, PAL.UNKNOWN),
                           linewidths=0, rasterized=True, zorder=2, label=lv)
            title = f"{r} {c}".strip()
            counts[title] = int(m.sum())
            panels.style_umap_axis(ax, title=f"{title}  (n={int(m.sum())})", label_fs=16,
                                   title_fs=17)
        for ax in af[len(panels_list):]:
            ax.set_visible(False)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"color_key": color_key, "split_key": split_key,
                   "second_split": second_split or None, "grid": [nr, nc],
                   "cells_per_panel": counts}
    _finish(ctx, paths, f"UMAP of {color_key} split by {split_key}"
            + (f" x {second_split}" if second_split else ""),
            {"color_key": color_key, "split_key": split_key}, return_image, group)
    ctx.suggest("skin.plot.legend_only", "skin.abundance.proportions")


@tool("skin.plot.umap_highlight", category="plot",
      summary="Highlight a subset on the UMAP; everything else grey.")
def umap_highlight(dataset_id: str, key: str, labels: list[str], save_prefix: str = "highlight",
                   group: str = "umap", return_image: bool = False, project_id: str = "",
                   dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Highlight chosen labels against a grey background (reference cell 32).

    Args:
        dataset_id: Handle or label with X_umap.
        key: obs column holding the labels.
        labels: Labels to highlight.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "X_umap" not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding("X_umap", list(adata.obsm.keys()))
    present = set(map(str, adata.obs[key].astype(str).unique()))
    want = [str(x) for x in labels if str(x) in present]
    if not want:
        raise NotFound(f"none of {labels} are in {key!r}", remedy=f"Present: {sorted(present)}")
    if dry_run:
        ctx.summary = {"highlight": want}
        return

    XY = np.asarray(adata.obsm["X_umap"])
    lab = adata.obs[key].astype(str).to_numpy()
    pal = PAL.get_from_adata(adata, key) or PAL.celltype_palette(want)
    with style():
        fig, ax = plt.subplots(figsize=(7, 6.5))
        ax.scatter(XY[:, 0], XY[:, 1], s=3, c=PAL.CONTEXT_GREY, linewidths=0,
                   rasterized=True, zorder=1)
        for lv in want:
            m = lab == lv
            ax.scatter(XY[m, 0], XY[m, 1], s=8, c=pal.get(lv, PAL.UNKNOWN), linewidths=0,
                       rasterized=True, zorder=2, label=f"{lv} (n={int(m.sum())})")
        panels.style_umap_axis(ax, title="")
        ax.legend(frameon=False, fontsize=12, loc="best", markerscale=2.0)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"key": key, "highlighted": want,
                   "n_highlighted": int(np.isin(lab, want).sum())}
    _finish(ctx, paths, f"UMAP highlighting {want} in {key}", {"key": key, "labels": want},
            return_image, group)


@tool("skin.plot.dotplot", category="plot", summary="Marker dotplot with the lab's styling.")
def dotplot(dataset_id: str, groupby: str, markers: list[str] | None = None,
            marker_groups: dict[str, list[str]] | None = None, cmap: str = "Reds",
            dot_max: float = 1.0, standard_scale: str = "var", dendrogram: bool = False,
            cluster_genes: bool = False, save_prefix: str = "dotplot", group: str = "dotplot",
            return_image: bool = False, project_id: str = "", dry_run: bool = False,
            seed: int = 0, *, ctx: Ctx) -> None:
    """Dotplot of markers per group (§8.4 styling).

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        groupby: obs column for the rows.
        markers: Flat gene list. Empty = the shipped lineage markers for the organism.
        marker_groups: {"Keratinocytes": ["Krt5",...]} for bracketed lineage headers.
        cmap: "Reds" for curated panels, "plasma_r" for major cell types.
        dot_max: Maximum dot size fraction.
        standard_scale: "var" (per gene), "group", or "none".
        dendrogram: Add a hierarchical dendrogram over the groups.
        cluster_genes: Reorder genes by profile similarity (reference cell 35).
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the gene list only.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, groupby)
    organism = registry.get_organism(adata)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")
        ctx.warn("X was z-scaled; restored layers['lognorm'] so dot colour means expression.")

    genes: Any
    if marker_groups:
        genes = {k: [g for g in v if g in adata.var_names] for k, v in marker_groups.items()}
        genes = {k: v for k, v in genes.items() if v}
        flat = [g for v in genes.values() for g in v]
    elif markers:
        flat = [g for g in markers if g in adata.var_names]
        genes = flat
    else:
        lin = K.lineages(organism)
        genes = {k: K.present(adata, v)[:4] for k, v in lin.items()}
        genes = {k: v for k, v in genes.items() if v}
        flat = [g for v in genes.values() for g in v]

    if dry_run:
        ctx.summary = {"n_genes": len(flat), "grouped": bool(marker_groups or not markers)}
        return
    if not flat:
        raise NotFound("none of the requested genes are in var_names")

    if cluster_genes and not isinstance(genes, dict):
        s = adata.obs[groupby]
        order = (list(map(str, s.cat.categories))
                 if isinstance(s.dtype, pd.CategoricalDtype)
                 else PAL.natural_order(s.astype(str)))
        genes = panels.cluster_gene_order(adata, flat, groupby, order)
        flat = list(genes)

    res = panels.dotplot(adata, genes, groupby, ctx.figdir(group) / save_prefix,
                         cmap=cmap, dot_max=dot_max,
                         standard_scale=(None if standard_scale == "none" else standard_scale),
                         dendrogram=dendrogram)
    ctx.summary = {"groupby": groupby, "n_genes": res["n_genes"], "n_groups": res["n_groups"],
                   "genes": flat[:40]}
    _finish(ctx, res["paths"], f"dotplot of {len(flat)} markers by {groupby}",
            {"groupby": groupby, "cmap": cmap, "n_genes": len(flat)}, return_image, group)
    ctx.suggest("skin.annotate.apply_labels", "skin.plot.umap")


@tool("skin.plot.volcano_grid", category="plot",
      summary="Grid of volcano panels from a DE run (reference cell 10).")
def volcano_grid(de_run_id: str, labels: list[str] | None = None, ncols: int = 0,
                 must_label: list[str] | None = None, highlight_genes: list[str] | None = None,
                 fdr: float = 0.05, lfc: float = 0.5, n_label: int = 9,
                 xlim_abs: float = 8.0, save_prefix: str = "volcano", group: str = "de",
                 return_image: bool = False, project_id: str = "", dry_run: bool = False,
                 seed: int = 0, *, ctx: Ctx) -> None:
    """Volcano panels, one per cell label, with adjustText label placement.

    Layout follows the reference: ncols = min(4, n_labels), figsize (7*ncols,
    7*nrows). Pass ncols=2 for the 2xN layout.

    Args:
        de_run_id: run_id from a skin.de.* tool.
        labels: Labels to plot. Empty = every label in the run.
        ncols: Panels per row. 0 = min(4, n_labels).
        must_label: Genes always labelled, whether or not they pass thresholds.
        highlight_genes: Genes whose labels are drawn in red.
        fdr: Adjusted-p threshold for the guide line and the counts.
        lfc: Log2FC threshold for the guide lines and the counts.
        n_label: Labels drawn per side.
        xlim_abs: X-axis limit, symmetric.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the panels only.
        seed: RNG seed.
    """
    import pandas as pd

    run = store.get_run(ctx.project_id, de_run_id)
    if run is None:
        raise NotFound(f"unknown de_run_id {de_run_id!r}",
                       remedy="skin.memory.brief lists recent runs.")
    tables = run["result"].get("tables", {})
    want = [str(x) for x in (labels or tables.keys())]
    missing = [x for x in want if x not in tables]
    if missing:
        raise NotFound(f"labels not in this run: {missing}",
                       remedy=f"Available: {sorted(tables)}")
    contrast = run["params"].get("contrast", ["A", "B"])
    a, b = str(contrast[0]), str(contrast[1])
    level = run["result"].get("inference_level", run["params"].get("method"))

    if dry_run:
        ctx.summary = {"panels": want, "contrast": [a, b], "ncols": ncols or min(4, len(want))}
        return

    de_tables = {lb: pd.read_csv(tables[lb]) for lb in want}
    fname = f"{save_prefix}_{panels.slug(a)}_vs_{panels.slug(b)}"
    res = panels.volcano_grid(
        de_tables, ctx.figdir(group) / fname, group_a=a, group_b=b,
        ncols=(ncols or None), fdr=fdr, lfc=lfc, n_label=n_label,
        must_label=must_label or [], highlight_genes=highlight_genes or [],
        xlim=(-abs(xlim_abs), abs(xlim_abs)))

    if run["params"].get("method") == "wilcoxon" or level == "cell":
        ctx.warn("This DE run is cell-wise (EXPLORATORY). The per-panel counts are inflated "
                 "relative to pseudobulk because cells within a sample are treated as "
                 "independent replicates. The caption metadata records this.")

    ctx.summary = {"de_run_id": de_run_id, "contrast": [a, b], "counts": res["counts"],
                   "n_panels": res["n_panels"], "inference_level": level,
                   "thresholds": {"fdr": fdr, "lfc": lfc}}
    _finish(ctx, res["paths"], f"volcano grid {a} vs {b} ({level} inference)",
            {"de_run_id": de_run_id, "contrast": [a, b], "fdr": fdr, "lfc": lfc,
             "inference_level": level, "method": run["params"].get("method")},
            return_image, group)
    ctx.suggest("skin.enrich.ora", "skin.plot.enrichment_tile", "skin.plot.de_panel")


@tool("skin.plot.enrichment_tile", category="plot",
      summary="Pathway x direction enrichment tile (reference make_enrichment_tile).")
def enrichment_tile(enrich_run_ids: list[str], title: str = "", save_prefix: str = "enrich_tile",
                    tile_size: float = 0.7, limit: int = 40, group: str = "de",
                    return_image: bool = False, project_id: str = "", dry_run: bool = False,
                    seed: int = 0, *, ctx: Ctx) -> None:
    """Render the enrichment tile: pathways as rows, directions as columns.

    Fill is -log10(padj) on a blue->red map; alpha encodes the gene Count; the
    bold number in each tile is fold enrichment.

    Args:
        enrich_run_ids: One or more run_ids from skin.enrich.ora.
        title: Figure title.
        save_prefix: Output filename stem.
        tile_size: Tile edge length in inches.
        limit: Character limit for term labels before truncation.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the term count only.
        seed: RNG seed.
    """
    import pandas as pd

    frames, params_seen = [], []
    for rid in enrich_run_ids:
        run = store.get_run(ctx.project_id, rid)
        if run is None:
            raise NotFound(f"unknown enrich run_id {rid!r}")
        p = run["result"].get("table")
        if not p or not Path(p).exists():
            ctx.warn(f"{rid}: no result table on disk; skipped")
            continue
        frames.append(pd.read_csv(p))
        params_seen.append(run["params"])
    if not frames:
        raise NotFound("no enrichment tables to plot")
    df = pd.concat(frames, ignore_index=True)
    if df.empty:
        ctx.warn("The enrichment results are empty — nothing to plot.")
        ctx.summary = {"skipped": True, "reason": "no enriched terms"}
        return

    contrast = params_seen[0].get("contrast", ["Sham", "Burn"])
    order = [str(contrast[1]), str(contrast[0])]  # reference (control) column first
    if dry_run:
        ctx.summary = {"n_terms": int(df["pathway_clean"].nunique()), "direction_order": order}
        return

    res = panels.enrichment_tile(df, ctx.figdir(group) / save_prefix, title=title,
                                 tile_size=tile_size, limit=limit, direction_order=order)
    if res.get("skipped"):
        ctx.summary = res
        return

    excl = params_seen[0].get("exclusion_preset")
    if excl:
        ctx.warn(f"This figure was produced with the exclusion preset {excl!r}. That is "
                 f"recorded in the sidecar JSON and must be stated in the figure legend.")
    ctx.summary = {"n_terms": res["n_terms"], "directions": res["directions"],
                   "enrich_run_ids": enrich_run_ids, "exclusion_preset": excl}
    _finish(ctx, res["paths"], title or "GO enrichment tile",
            {"enrich_run_ids": enrich_run_ids, "library": params_seen[0].get("library"),
             "exclusion_preset": excl, "excluded_terms": params_seen[0].get("excluded_terms"),
             "background_size": params_seen[0].get("background_size")},
            return_image, group)


@tool("skin.plot.de_panel", category="plot", needs_network=True,
      summary="Volcano grid + one enrichment tile per label, in one call.")
def de_panel(de_run_id: str, labels: list[str] | None = None, library: str = "GO_Biological_Process_2025",
             ncols: int = 0, must_label: list[str] | None = None,
             highlight_genes: list[str] | None = None, fdr: float = 0.05, lfc: float = 0.5,
             exclude_preset: str = "", top_n_terms: int = 5, group: str = "de",
             project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """The standard DE output: one volcano grid plus a GO tile per label.

    Args:
        de_run_id: run_id from a skin.de.* tool.
        labels: Labels to include. Empty = all.
        library: Enrichment library.
        ncols: Volcano panels per row.
        must_label: Genes always labelled on the volcanoes.
        highlight_genes: Genes labelled in red.
        fdr: FDR threshold.
        lfc: Log2FC threshold.
        exclude_preset: Optional named term-exclusion preset. Off by default.
        top_n_terms: Terms per direction on each tile.
        group: Figure subdirectory.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from . import enrich_tools

    run = store.get_run(ctx.project_id, de_run_id)
    if run is None:
        raise NotFound(f"unknown de_run_id {de_run_id!r}")
    want = [str(x) for x in (labels or run["result"].get("tables", {}).keys())]
    if dry_run:
        ctx.summary = {"labels": want, "library": library}
        return

    v = volcano_grid(de_run_id=de_run_id, labels=want, ncols=ncols,
                     must_label=must_label or [], highlight_genes=highlight_genes or [],
                     fdr=fdr, lfc=lfc, group=group, project_id=ctx.project_id, seed=seed)
    for w in v.get("warnings", []):
        ctx.warn(f"[volcano_grid] {w}")
    made = [{"kind": "volcano_grid", "ok": v.get("ok"),
             "artifact_id": v.get("summary", {}).get("artifact_id")}]

    for lb in want:
        e = enrich_tools.ora(de_run_id=de_run_id, label=lb, library=library, fdr=fdr, lfc=lfc,
                             exclude_preset=exclude_preset, top_n_terms=top_n_terms,
                             project_id=ctx.project_id, seed=seed)
        if not e.get("ok"):
            ctx.warn(f"[ora:{lb}] {e.get('error', {}).get('message', 'failed')}")
            continue
        rid = e["summary"]["run_id"]
        t = enrichment_tile(enrich_run_ids=[rid],
                            title=f"{library.replace('_', ' ')} — {lb}",
                            save_prefix=f"enrich_tile_{panels.slug(lb)}", group=group,
                            project_id=ctx.project_id, seed=seed)
        made.append({"kind": "enrichment_tile", "label": lb, "ok": t.get("ok"),
                     "artifact_id": t.get("summary", {}).get("artifact_id")})

    ctx.summary = {"de_run_id": de_run_id, "figures": made, "library": library,
                   "labels": want}
    ctx.suggest("skin.export.report", "skin.memory.note")


@tool("skin.plot.score_umap_grid", category="plot",
      summary="Grid of UMAPs coloured by signature scores (reference cell 5).")
def score_umap_grid(dataset_id: str, scores: list[str] | None = None, cmap: str = "RdBu_r",
                    vcenter: float = 0.0, ncols: int = 3, save_prefix: str = "score_umaps",
                    group: str = "scores", return_image: bool = False, project_id: str = "",
                    dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Plot every signature score on the UMAP, diverging around zero.

    Args:
        dataset_id: Handle or label with X_umap and score_* columns.
        scores: obs columns to plot. Empty = every `score_*` column.
        cmap: Diverging colormap.
        vcenter: Centre value for the diverging norm.
        ncols: Panels per row.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the score columns only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    adata = registry.load(ctx.project_id, dataset_id)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "X_umap" not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding("X_umap", list(adata.obsm.keys()))
    cols = scores or [c for c in adata.obs.columns if str(c).startswith("score_")
                      or str(c).startswith("HALLMARK_")]
    cols = [c for c in cols if c in adata.obs.columns]
    if not cols:
        raise NotFound("no score columns found",
                       remedy="Run skin.enrich.score_panel or score_signature first.",
                       suggested_tool="skin.enrich.score_panel")
    if dry_run:
        ctx.summary = {"scores": cols}
        return

    XY = np.asarray(adata.obsm["X_umap"])
    nc = min(ncols, len(cols))
    nr = int(-(-len(cols) // nc))
    with style():
        fig, axes = plt.subplots(nr, nc, figsize=(5.4 * nc, 5.0 * nr), squeeze=False)
        af = axes.flatten()
        for ax, c in zip(af, cols):
            v = adata.obs[c].to_numpy(dtype=float)
            lim = float(np.nanpercentile(np.abs(v - vcenter), 99)) or 1.0
            norm = TwoSlopeNorm(vmin=vcenter - lim, vcenter=vcenter, vmax=vcenter + lim)
            order = np.argsort(v)  # draw high scores last
            s = ax.scatter(XY[order, 0], XY[order, 1], c=v[order], s=5, cmap=cmap,
                           norm=norm, linewidths=0, rasterized=True)
            panels.style_umap_axis(ax, title=c.replace("score_", "").replace("_", " "),
                                   label_fs=16, title_fs=18)
            fig.colorbar(s, ax=ax, fraction=0.04, pad=0.02)
        for ax in af[len(cols):]:
            ax.set_visible(False)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"scores": cols, "n_panels": len(cols)}
    _finish(ctx, paths, f"signature score UMAPs ({len(cols)} panels)",
            {"scores": cols, "cmap": cmap}, return_image, group)
    ctx.suggest("skin.plot.state_space", "skin.abundance.proportions")


@tool("skin.plot.state_space", category="plot",
      summary="2D confidence-ellipse or 3D ellipsoid state-space plot (reference cells 86/92).")
def state_space(dataset_id: str, x_score: str, y_score: str, groupby: str,
                z_score: str = "", color_score: str = "", style_: str = "ellipse",
                n_std: float = 1.0, save_prefix: str = "state_space", group: str = "scores",
                return_image: bool = False, project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Plot groups in signature space with confidence ellipses (or 3D ellipsoids).

    Args:
        dataset_id: Handle or label with the score columns.
        x_score: obs column for the x axis.
        y_score: obs column for the y axis.
        groupby: obs column defining the groups to draw ellipses for.
        z_score: obs column for the z axis (3D only).
        color_score: Optional obs column used to colour the points.
        style_: "ellipse" (2D) or "ellipsoid3d".
        n_std: Ellipse radius in standard deviations.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Ellipse

    adata = registry.load(ctx.project_id, dataset_id)
    for c in (x_score, y_score, groupby):
        require_obs(adata, c)
    if style_ == "ellipsoid3d" and not z_score:
        raise BadParam("z_score is required for style_='ellipsoid3d'")
    if z_score:
        require_obs(adata, z_score)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if dry_run:
        ctx.summary = {"x": x_score, "y": y_score, "z": z_score or None, "style": style_}
        return

    lab = adata.obs[groupby].astype(str).to_numpy()
    groups = PAL.natural_order(set(lab))
    pal = PAL.get_from_adata(adata, groupby) or PAL.celltype_palette(groups)
    centroids = {}

    with style():
        if style_ == "ellipsoid3d":
            fig = plt.figure(figsize=(9, 8))
            ax = fig.add_subplot(111, projection="3d")
            for g in groups:
                m = lab == g
                P = np.column_stack([adata.obs[x_score][m], adata.obs[y_score][m],
                                     adata.obs[z_score][m]]).astype(float)
                c = P.mean(0)
                centroids[g] = [round(float(v), 4) for v in c]
                ax.scatter(*P.T, s=4, alpha=0.25, color=pal.get(g, PAL.UNKNOWN),
                           linewidths=0, rasterized=True)
                cov = np.cov(P.T)
                w, V = np.linalg.eigh(cov)
                u, v = np.mgrid[0:2 * np.pi:24j, 0:np.pi:12j]
                sph = np.stack([np.cos(u) * np.sin(v), np.sin(u) * np.sin(v),
                                np.cos(v)], -1)
                E = sph @ np.diag(n_std * np.sqrt(np.maximum(w, 0))) @ V.T + c
                ax.plot_wireframe(E[..., 0], E[..., 1], E[..., 2], linewidth=0.5,
                                  alpha=0.5, color=pal.get(g, PAL.UNKNOWN))
            ax.set_xlabel(x_score.replace("score_", ""), fontweight="bold")
            ax.set_ylabel(y_score.replace("score_", ""), fontweight="bold")
            ax.set_zlabel(z_score.replace("score_", ""), fontweight="bold")
        else:
            fig, ax = plt.subplots(figsize=(8, 7))
            for g in groups:
                m = lab == g
                x = adata.obs[x_score][m].to_numpy(dtype=float)
                y = adata.obs[y_score][m].to_numpy(dtype=float)
                col = pal.get(g, PAL.UNKNOWN)
                ax.scatter(x, y, s=5, alpha=0.28, color=col, linewidths=0, rasterized=True)
                if len(x) > 3:
                    cov = np.cov(x, y)
                    w, V = np.linalg.eigh(cov)
                    ang = np.degrees(np.arctan2(V[1, -1], V[0, -1]))
                    e = Ellipse((x.mean(), y.mean()),
                                2 * n_std * np.sqrt(max(w[-1], 0)),
                                2 * n_std * np.sqrt(max(w[0], 0)),
                                angle=ang, facecolor="none", edgecolor=col, lw=2.4, zorder=5)
                    ax.add_patch(e)
                ax.scatter([x.mean()], [y.mean()], s=140, marker="X", color=col,
                           edgecolor="black", lw=1.2, zorder=6, label=g)
                centroids[g] = [round(float(x.mean()), 4), round(float(y.mean()), 4)]
            ax.axhline(0, color="0.7", lw=0.9, ls="--")
            ax.axvline(0, color="0.7", lw=0.9, ls="--")
            ax.set_xlabel(x_score.replace("score_", ""), fontsize=20, fontweight="bold")
            ax.set_ylabel(y_score.replace("score_", ""), fontsize=20, fontweight="bold")
            ax.legend(frameon=False, fontsize=12, loc="best")
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"x_score": x_score, "y_score": y_score, "z_score": z_score or None,
                   "groupby": groupby, "style": style_, "centroids": centroids}
    _finish(ctx, paths, f"state space {x_score} vs {y_score} by {groupby}",
            {"x": x_score, "y": y_score, "z": z_score, "n_std": n_std}, return_image, group)


@tool("skin.plot.legend_only", category="plot",
      summary="Standalone legend figure, for panel assembly in Illustrator.")
def legend_only(dataset_id: str, key: str, orientation: str = "vertical", ncol: int = 1,
                title: str = "", save_prefix: str = "legend", group: str = "legends",
                project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Render just the legend for a categorical, as its own vector file.

    Args:
        dataset_id: Handle or label.
        key: Categorical obs column.
        orientation: "vertical" or "horizontal".
        ncol: Legend columns when vertical.
        title: Legend title.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        project_id: Defaults to the active project.
        dry_run: Report the entries only.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    s = adata.obs[key]
    levels = (list(map(str, s.cat.categories)) if isinstance(s.dtype, pd.CategoricalDtype)
              else PAL.natural_order(s.astype(str)))
    pal = PAL.get_from_adata(adata, key) or PAL.celltype_palette(levels)
    mapping = {lv: pal.get(lv, PAL.UNKNOWN) for lv in levels}
    if dry_run:
        ctx.summary = {"entries": mapping}
        return
    res = panels.legend_only(mapping, ctx.figdir(group) / f"{save_prefix}_{panels.slug(key)}",
                             orientation=orientation, title=title or key, ncol=ncol)
    ctx.summary = {"key": key, "n_entries": len(mapping), "palette": mapping}
    _finish(ctx, res["paths"], f"legend for {key}", {"key": key}, False, group)


@tool("skin.plot.heatmap", category="plot", summary="Expression heatmap by group.")
def heatmap(dataset_id: str, genes: list[str], groupby: str, standard_scale: str = "var",
            cmap: str = "RdBu_r", save_prefix: str = "heatmap", group: str = "heatmap",
            return_image: bool = False, project_id: str = "", dry_run: bool = False,
            seed: int = 0, *, ctx: Ctx) -> None:
    """Mean-expression heatmap of chosen genes across groups.

    Args:
        dataset_id: Handle or label.
        genes: Genes to show.
        groupby: obs column for the columns of the heatmap.
        standard_scale: "var" (per gene), "group", or "none".
        cmap: Colormap.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the gene list only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, groupby)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
    present = [g for g in genes if g in adata.var_names]
    if not present:
        raise NotFound("none of the requested genes are in var_names")
    if dry_run:
        ctx.summary = {"n_genes": len(present)}
        return

    X = adata[:, present].X
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    df = pd.DataFrame(X, columns=present)
    df["_g"] = adata.obs[groupby].astype(str).to_numpy()
    M = df.groupby("_g", observed=True).mean()
    s = adata.obs[groupby]
    order = (list(map(str, s.cat.categories)) if isinstance(s.dtype, pd.CategoricalDtype)
             else PAL.natural_order(M.index))
    M = M.reindex([o for o in order if o in M.index])
    if standard_scale == "var":
        M = (M - M.mean(0)) / M.std(0).replace(0, 1)
    elif standard_scale == "group":
        M = M.sub(M.mean(1), axis=0).div(M.std(1).replace(0, 1), axis=0)

    with style():
        fig, ax = plt.subplots(figsize=(0.30 * len(present) + 3, 0.42 * len(M) + 2.5))
        lim = float(np.nanmax(np.abs(M.to_numpy()))) or 1.0
        im = ax.imshow(M.to_numpy(), aspect="auto", cmap=cmap, vmin=-lim, vmax=lim)
        ax.set_xticks(range(len(present)))
        ax.set_xticklabels(present, rotation=90, fontsize=11, fontweight="bold")
        ax.set_yticks(range(len(M)))
        ax.set_yticklabels(M.index, fontsize=13, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02,
                     label="z-score" if standard_scale != "none" else "mean expression")
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"n_genes": len(present), "n_groups": int(len(M)), "groupby": groupby}
    _finish(ctx, paths, f"heatmap of {len(present)} genes by {groupby}",
            {"genes": present[:40], "groupby": groupby}, return_image, group)


@tool("skin.plot.violin_qc", category="plot", summary="Violin plot of a numeric obs column.")
def violin_qc(dataset_id: str, keys: list[str], groupby: str = "Sample",
              save_prefix: str = "violin", group: str = "qc", return_image: bool = False,
              project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Violins of numeric obs columns split by a categorical.

    Args:
        dataset_id: Handle or label.
        keys: Numeric obs columns.
        groupby: Categorical obs column.
        save_prefix: Output filename stem.
        group: Figure subdirectory.
        return_image: Inline the PNG as base64.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, groupby)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    cols = [k for k in keys if k in adata.obs.columns]
    if not cols:
        raise NotFound(f"none of {keys} are obs columns")
    if dry_run:
        ctx.summary = {"keys": cols}
        return

    order = PAL.natural_order(adata.obs[groupby].astype(str).unique())
    with style():
        fig, axes = plt.subplots(len(cols), 1, figsize=(max(6, 0.6 * len(order)), 3.4 * len(cols)),
                                 squeeze=False)
        for ax, c in zip(axes.flatten(), cols):
            data = [adata.obs.loc[adata.obs[groupby].astype(str) == o, c].to_numpy(dtype=float)
                    for o in order]
            parts = ax.violinplot(data, showextrema=False, widths=0.85)
            for pc in parts["bodies"]:
                pc.set_facecolor("#4E79A7")
                pc.set_alpha(0.65)
            for j, d in enumerate(data, start=1):
                ax.scatter([j], [np.median(d)], color="black", s=14, zorder=4)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order, rotation=90, fontsize=11)
            ax.set_ylabel(c, fontsize=14, fontweight="bold")
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir(group) / save_prefix)
        plt.close(fig)

    ctx.summary = {"keys": cols, "groupby": groupby, "n_groups": len(order)}
    _finish(ctx, paths, f"violins of {cols} by {groupby}", {"keys": cols}, return_image, group)
