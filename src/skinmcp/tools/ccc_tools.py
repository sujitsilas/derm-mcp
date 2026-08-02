"""`skin.ccc.*` — ligand-receptor and cell-cell communication.

Guard enforced throughout: LR inference on fewer than ~30 cells in a sending or
receiving population is noise. Populations below the floor are dropped and
listed, never quietly included.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import registry
from ..config import CONFIG
from ..errors import BadParam, DependencyMissing, NetworkUnavailable, NotFound
from ..memory import store
from ..style import palettes as PAL
from ..style.rcparams import savefig, style
from ._base import Ctx, require_obs, tool

logger = logging.getLogger(__name__)

MIN_CELLS_DEFAULT = 30


def _drop_small(adata: Any, label_key: str, min_cells: int, ctx: Ctx) -> tuple[Any, list[str]]:
    vc = adata.obs[label_key].astype(str).value_counts()
    small = sorted(vc[vc < min_cells].index.astype(str))
    if small:
        ctx.warn(f"dropped {len(small)} populations with fewer than {min_cells} cells: "
                 f"{small[:8]}. Ligand-receptor scores from populations that small are "
                 f"dominated by sampling noise.")
        keep = ~adata.obs[label_key].astype(str).isin(small)
        adata = adata[keep.to_numpy()].copy()
        import pandas as pd

        if isinstance(adata.obs[label_key].dtype, pd.CategoricalDtype):
            adata.obs[label_key] = adata.obs[label_key].cat.remove_unused_categories()
    return adata, small


@tool("skin.ccc.liana", category="ccc", needs_network=True,
      summary="LIANA rank_aggregate ligand-receptor scoring, run per context group.")
def liana(dataset_id: str, label_key: str, groupby_context: str = "",
          method: str = "rank_aggregate", resource: str = "", expr_prop: float = 0.1,
          min_cells: int = MIN_CELLS_DEFAULT, top_n: int = 25, project_id: str = "",
          dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Score ligand-receptor interactions between labelled populations.

    Run per context group (e.g. per `Type x Timepoint`) so the conditions are
    comparable — scoring the pooled object and then splitting is not the same
    thing, because the expression proportions are computed across all cells.

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        label_key: obs column of the populations.
        groupby_context: obs column defining context groups, e.g. "Type_Timepoint".
            Empty = one run over everything.
        method: "rank_aggregate" (consensus), "cellphonedb", "natmi", "logfc",
            "connectome", or "singlecellsignalr".
        resource: "mouseconsensus" or "consensus". Empty = chosen by organism.
        expr_prop: Minimum fraction of a population expressing a gene.
        min_cells: Populations below this are dropped and listed.
        top_n: Interactions reported inline per context; the full table is an artifact.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    organism = registry.get_organism(adata)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    res_name = resource or ("mouseconsensus" if organism == "mouse" else "consensus")

    ctx.code = (
        "import liana as li\n"
        f"li.mt.{method}(adata, groupby={label_key!r}, resource_name={res_name!r},\n"
        f"               expr_prop={expr_prop}, use_raw=False, verbose=False)\n"
        "df = adata.uns['liana_res']\n"
    )
    if dry_run:
        ctx.summary = {"method": method, "resource": res_name,
                       "contexts": (int(adata.obs[groupby_context].nunique())
                                    if groupby_context else 1)}
        return
    if CONFIG.offline:
        raise NetworkUnavailable(
            "LIANA needs to download its LR resource and --offline is set",
            remedy="Restart without --offline, or use skin.ccc.cellchat_r with a "
                   "pre-built container image.",
            suggested_tool="skin.enrich.score_signature")
    try:
        import liana as li
    except ImportError as e:
        raise DependencyMissing(
            "liana is not installed",
            remedy='uv pip install "skin-mcp[ccc]"',
            suggested_tool="skin.ccc.cellchat_r") from e

    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")
        ctx.warn("X was z-scaled; restored layers['lognorm'] — LR scoring needs expression "
                 "on the natural scale.")

    contexts = ([""] if not groupby_context
                else PAL.natural_order(adata.obs[groupby_context].astype(str).unique()))
    if groupby_context:
        require_obs(adata, groupby_context)

    run_id = store.new_id("ccc", 8)
    per_context, dropped_all, tables = [], {}, {}
    fn = getattr(li.mt, method, None)
    if fn is None:
        raise BadParam(f"unknown liana method {method!r}",
                       remedy=f"Available: {[m for m in dir(li.mt) if not m.startswith('_')]}")

    for cxt in contexts:
        sub = (adata if not groupby_context
               else adata[(adata.obs[groupby_context].astype(str) == cxt).to_numpy()].copy())
        sub, small = _drop_small(sub, label_key, min_cells, ctx)
        dropped_all[cxt or "all"] = small
        if int(sub.obs[label_key].astype(str).nunique()) < 2:
            ctx.warn(f"context {cxt or 'all'}: fewer than 2 populations survive the "
                     f"min_cells={min_cells} floor; skipped.")
            continue
        try:
            fn(sub, groupby=label_key, resource_name=res_name, expr_prop=expr_prop,
               use_raw=False, verbose=False, seed=seed)
        except TypeError:
            fn(sub, groupby=label_key, resource_name=res_name, expr_prop=expr_prop,
               use_raw=False, verbose=False)
        df = sub.uns.get("liana_res")
        if df is None or len(df) == 0:
            ctx.warn(f"context {cxt or 'all'}: LIANA returned no interactions.")
            continue
        df = pd.DataFrame(df)
        p = ctx.tabledir() / f"liana_{run_id}_{PAL.norm_key(cxt) or 'all'}.csv"
        df.to_csv(p, index=False)
        ctx.add_artifact("table", p, caption=f"LIANA {method} — {cxt or 'all'}")
        tables[cxt or "all"] = str(p)
        rank_col = next((c for c in ("magnitude_rank", "specificity_rank", "lr_means")
                         if c in df.columns), df.columns[-1])
        asc = "rank" in rank_col
        top = df.sort_values(rank_col, ascending=asc).head(top_n)
        per_context.append({
            "context": cxt or "all", "n_interactions": int(len(df)),
            "n_populations": int(sub.obs[label_key].nunique()),
            "dropped_populations": small,
            "top": top[[c for c in ("source", "target", "ligand_complex",
                                    "receptor_complex", rank_col) if c in top.columns]]
            .head(10).to_dict("records"),
            "table_path": str(p),
        })

    if not per_context:
        raise BadParam("no context produced LIANA results",
                       remedy="Check the min_cells floor and that populations overlap "
                              "across contexts.")

    store.record_run(ctx.project_id, run_id, "ccc", resolved or dataset_id,
                     {"method": method, "label_key": label_key, "resource": res_name,
                      "groupby_context": groupby_context, "min_cells": min_cells},
                     {"per_context": per_context, "tables": tables})
    ctx.summary = {"run_id": run_id, "method": method, "resource": res_name,
                   "n_contexts": len(per_context), "per_context": per_context[:6],
                   "dropped_populations": dropped_all}
    ctx.suggest("skin.ccc.liana_differential", "skin.ccc.plot_lr_dotplot")


@tool("skin.ccc.liana_differential", category="ccc",
      summary="Differential ligand-receptor table between two LIANA contexts.")
def liana_differential(run_id: str, context_a: str, context_b: str, top_n: int = 10,
                       specificity_cutoff: float = 0.05, make_plot: bool = True,
                       project_id: str = "", dry_run: bool = False, seed: int = 0,
                       *, ctx: Ctx) -> None:
    """Compare two contexts of a LIANA run: delta of LR expression, filtered by specificity.

    Follows the reference notebook's `diff_table` pattern — join on the LR pair,
    take `delta = lr_expr_a - lr_expr_b`, keep only pairs that are specific in
    the arm they are up in, and report the top N in each direction.

    Args:
        run_id: run_id from skin.ccc.liana.
        context_a: First context name (the "treated" arm).
        context_b: Second context name (the reference arm).
        top_n: Interactions per direction.
        specificity_cutoff: Maximum specificity p-value in the arm a pair is up in.
        make_plot: Render the horizontal delta bar chart.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    run = store.get_run(ctx.project_id, run_id)
    if run is None:
        raise NotFound(f"unknown ccc run_id {run_id!r}")
    tables = run["result"].get("tables", {})
    for c in (context_a, context_b):
        if c not in tables:
            raise NotFound(f"context {c!r} not in this run",
                           remedy=f"Contexts available: {sorted(tables)}")
    if dry_run:
        ctx.summary = {"context_a": context_a, "context_b": context_b}
        return

    A = pd.read_csv(tables[context_a])
    B = pd.read_csv(tables[context_b])
    keys = [c for c in ("source", "target", "ligand_complex", "receptor_complex")
            if c in A.columns and c in B.columns]
    expr_col = next((c for c in ("lr_means", "expr_prod", "lrscore", "magnitude")
                     if c in A.columns), None)
    spec_col = next((c for c in ("cellphone_pvals", "specificity_rank", "scaled_weight")
                     if c in A.columns), None)
    if not keys or expr_col is None:
        raise BadParam("LIANA tables lack the expected columns",
                       remedy=f"Columns found: {list(A.columns)[:12]}")

    m = A.merge(B, on=keys, suffixes=("_a", "_b"))
    m["delta"] = m[f"{expr_col}_a"] - m[f"{expr_col}_b"]
    if spec_col:
        up_ok = m[f"{spec_col}_a"] <= specificity_cutoff
        dn_ok = m[f"{spec_col}_b"] <= specificity_cutoff
    else:
        up_ok = dn_ok = pd.Series(True, index=m.index)
        ctx.warn("no specificity column in these LIANA results; the differential table is "
                 "unfiltered and will include non-specific pairs.")

    up = m[(m["delta"] > 0) & up_ok].nlargest(top_n, "delta")
    dn = m[(m["delta"] < 0) & dn_ok].nsmallest(top_n, "delta")
    diff = pd.concat([up, dn])
    diff["pair"] = (diff["source"].astype(str) + " → " + diff["target"].astype(str) + "  ("
                    + diff["ligand_complex"].astype(str) + "→"
                    + diff["receptor_complex"].astype(str) + ")")

    did = store.new_id("ccd", 8)
    p = ctx.tabledir() / f"liana_diff_{did}.csv"
    diff.to_csv(p, index=False)
    ctx.add_artifact("table", p, caption=f"LIANA differential {context_a} vs {context_b}")

    fig_info = None
    if make_plot and len(diff):
        pal = PAL.condition_palette([context_b, context_a])
        with style("standard"):
            d = diff.sort_values("delta")
            fig, ax = plt.subplots(figsize=(9, 0.36 * len(d) + 2.2))
            cols = [pal.get(context_a, PAL.BURN) if v > 0 else pal.get(context_b, PAL.SHAM)
                    for v in d["delta"]]
            ax.barh(np.arange(len(d)), d["delta"], color=cols)
            ax.set_yticks(np.arange(len(d)))
            ax.set_yticklabels(d["pair"], fontsize=9)
            ax.axvline(0, color="0.4", lw=1.2)
            ax.set_xlabel(f"Δ LR expression  ({context_b} ← → {context_a})",
                          fontsize=14, fontweight="bold")
            fig.tight_layout()
            paths = savefig(fig, ctx.figdir("ccc") / f"liana_diff_{did}")
            plt.close(fig)
        aid = ctx.add_artifact("figure", paths["pdf"],
                               caption=f"differential LR: {context_a} vs {context_b}")
        fig_info = {"artifact_id": aid, "paths": paths}

    store.record_run(ctx.project_id, did, "ccc", run["dataset_id"],
                     {"method": "liana_differential", "context_a": context_a,
                      "context_b": context_b}, {"table": str(p)})
    ctx.summary = {"run_id": did, "context_a": context_a, "context_b": context_b,
                   "n_compared": int(len(m)),
                   "top_up": up[["source", "target", "ligand_complex", "receptor_complex",
                                 "delta"]].head(8).to_dict("records"),
                   "top_down": dn[["source", "target", "ligand_complex", "receptor_complex",
                                   "delta"]].head(8).to_dict("records"),
                   "figure": fig_info, "table_path": str(p)}
    ctx.suggest("skin.ccc.plot_lr_dotplot", "skin.memory.note")


@tool("skin.ccc.plot_lr_dotplot", category="ccc",
      summary="Dotplot of top ligand-receptor pairs for chosen source/target populations.")
def plot_lr_dotplot(run_id: str, context: str = "", sources: list[str] | None = None,
                    targets: list[str] | None = None, top_n: int = 25,
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Plot LR magnitude and specificity for selected sender/receiver pairs.

    Args:
        run_id: run_id from skin.ccc.liana.
        context: Which context to plot. Empty = the first one.
        sources: Sender populations. Empty = all.
        targets: Receiver populations. Empty = all.
        top_n: Interactions to show.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    run = store.get_run(ctx.project_id, run_id)
    if run is None:
        raise NotFound(f"unknown ccc run_id {run_id!r}")
    tables = run["result"].get("tables", {})
    cxt = context or next(iter(tables))
    if cxt not in tables:
        raise NotFound(f"context {cxt!r} not found", remedy=f"Available: {sorted(tables)}")
    if dry_run:
        ctx.summary = {"context": cxt}
        return

    df = pd.read_csv(tables[cxt])
    if sources:
        df = df[df["source"].astype(str).isin(sources)]
    if targets:
        df = df[df["target"].astype(str).isin(targets)]
    if df.empty:
        raise NotFound("no interactions match those sources/targets")

    mag = next((c for c in ("lr_means", "expr_prod", "magnitude_rank") if c in df.columns),
               df.columns[-1])
    spec = next((c for c in ("cellphone_pvals", "specificity_rank") if c in df.columns), None)
    d = df.sort_values(mag, ascending="rank" in mag).head(top_n).copy()
    d["pair"] = d["ligand_complex"].astype(str) + " → " + d["receptor_complex"].astype(str)
    d["st"] = d["source"].astype(str) + " → " + d["target"].astype(str)

    with style("standard"):
        pairs = list(dict.fromkeys(d["pair"]))
        sts = list(dict.fromkeys(d["st"]))
        fig, ax = plt.subplots(figsize=(0.9 * len(sts) + 5, 0.34 * len(pairs) + 3))
        v = d[mag].to_numpy(dtype=float)
        size = 40 + 260 * (v - v.min()) / (np.ptp(v) or 1)
        col = (-np.log10(np.clip(d[spec].to_numpy(dtype=float), 1e-10, None))
               if spec and "pval" in spec else v)
        s = ax.scatter([sts.index(x) for x in d["st"]], [pairs.index(x) for x in d["pair"]],
                       s=size, c=col, cmap="viridis", edgecolor="black", linewidth=0.4)
        ax.set_xticks(range(len(sts)))
        ax.set_xticklabels(sts, rotation=45, ha="right", fontsize=10)
        ax.set_yticks(range(len(pairs)))
        ax.set_yticklabels(pairs, fontsize=10)
        fig.colorbar(s, ax=ax, fraction=0.03, pad=0.02,
                     label=("-log10(specificity p)" if spec and "pval" in spec else mag))
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("ccc") / f"lr_dotplot_{PAL.norm_key(cxt)}")
        plt.close(fig)

    aid = ctx.add_artifact("figure", paths["pdf"], caption=f"LR dotplot — {cxt}")
    ctx.summary = {"context": cxt, "n_shown": int(len(d)), "artifact_id": aid,
                   "paths": paths}


@tool("skin.ccc.cellchat_r", category="ccc", needs_r=True,
      summary="CellChat in the R container, per split group.")
def cellchat_r(dataset_id: str, label_key: str, split_by: str = "", organism: str = "",
               min_cells: int = MIN_CELLS_DEFAULT, project_id: str = "",
               dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Run CellChat with CellChatDB, one object per split level.

    Args:
        dataset_id: Handle or label.
        label_key: obs column of the populations.
        split_by: obs column to split on, e.g. "Type_Timepoint". Empty = one run.
        organism: "mouse" or "human". Defaults to the dataset's organism.
        min_cells: Populations below this are dropped.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    org = organism or registry.get_organism(adata)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    if dry_run:
        ctx.summary = {"backend": "CellChat (R)", "organism": org, "split_by": split_by or None}
        return

    adata, small = _drop_small(adata, label_key, min_cells, ctx)
    res = run_r_script("ccc_cellchat", adata=adata, project_id=ctx.project_id,
                       params={"label_key": label_key, "split_by": split_by,
                               "organism": org, "min_cells": min_cells, "seed": seed},
                       python_fallback="skin.ccc.liana")
    run_id = store.new_id("ccc", 8)
    store.record_run(ctx.project_id, run_id, "ccc", resolved or dataset_id,
                     {"method": "cellchat", "label_key": label_key, "split_by": split_by,
                      "organism": org}, res)
    ctx.summary = {"run_id": run_id, "method": "cellchat", "organism": org,
                   "per_split": res.get("per_split", []),
                   "dropped_populations": small,
                   "r_log_tail": (res.get("log") or "")[-400:]}
    ctx.suggest("skin.ccc.cellchat_compare", "skin.ccc.plot_chord")


@tool("skin.ccc.cellchat_compare", category="ccc", needs_r=True,
      summary="mergeCellChat: information-flow scatter and pathway x split heatmap.")
def cellchat_compare(run_ids: list[str], project_id: str = "", dry_run: bool = False,
                     seed: int = 0, *, ctx: Ctx) -> None:
    """Compare CellChat runs: rankNet information flow and the pathway log2 ratio heatmap.

    Args:
        run_ids: Two or more run_ids from skin.ccc.cellchat_r.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    runs = []
    for rid in run_ids:
        r = store.get_run(ctx.project_id, rid)
        if r is None:
            raise NotFound(f"unknown ccc run_id {rid!r}")
        runs.append(r)
    if len(runs) < 2:
        raise BadParam("need at least two run_ids to compare")
    if dry_run:
        ctx.summary = {"run_ids": run_ids}
        return

    res = run_r_script("ccc_cellchat_compare", adata=None, project_id=ctx.project_id,
                       params={"work_dirs": [r["result"].get("work_dir") for r in runs],
                               "labels": run_ids},
                       python_fallback="skin.ccc.liana_differential")
    for f in res.get("figures", []):
        ctx.add_artifact("figure", f, caption="CellChat comparison")
    ctx.summary = {"run_ids": run_ids, "information_flow": res.get("information_flow", [])[:20],
                   "figures": res.get("figures", []),
                   "r_log_tail": (res.get("log") or "")[-400:]}
    ctx.suggest("skin.ccc.plot_chord", "skin.export.report")


@tool("skin.ccc.plot_chord", category="ccc", needs_r=True,
      summary="Chord diagram for chosen sources, targets and pathways.")
def plot_chord(run_id: str, pathways: list[str] | None = None, sources: list[str] | None = None,
               targets: list[str] | None = None, project_id: str = "", dry_run: bool = False,
               seed: int = 0, *, ctx: Ctx) -> None:
    """Render a CellChat chord diagram.

    Args:
        run_id: run_id from skin.ccc.cellchat_r.
        pathways: Signalling pathways to draw. Empty = the top aggregate.
        sources: Sender populations.
        targets: Receiver populations.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    run = store.get_run(ctx.project_id, run_id)
    if run is None:
        raise NotFound(f"unknown ccc run_id {run_id!r}")
    if dry_run:
        ctx.summary = {"run_id": run_id, "pathways": pathways or []}
        return
    res = run_r_script("ccc_chord", adata=None, project_id=ctx.project_id,
                       params={"work_dir": run["result"].get("work_dir"),
                               "pathways": pathways or [], "sources": sources or [],
                               "targets": targets or []},
                       python_fallback="skin.ccc.plot_lr_dotplot")
    for f in res.get("figures", []):
        ctx.add_artifact("figure", f, caption="CellChat chord diagram")
    ctx.summary = {"run_id": run_id, "figures": res.get("figures", []),
                   "r_log_tail": (res.get("log") or "")[-300:]}
