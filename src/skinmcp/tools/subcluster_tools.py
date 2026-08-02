"""`skin.sub.*` — subclustering, with re-normalization done properly.

Subsetting an object and reusing the parent's HVGs, scaling, and embedding is
the single most common subclustering error: the feature space was chosen to
separate keratinocytes from fibroblasts, and it cannot resolve LAM-I from
LAM-II. `extract` restores raw counts so the downstream `preprocess` selects
features for the compartment you actually care about.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import registry
from ..errors import BadParam, NotFound
from ..memory import store
from ._base import Ctx, confirm_or_raise, require_obs, tool

logger = logging.getLogger(__name__)


@tool("skin.sub.extract", category="sub",
      summary="Subset to chosen labels and restore raw counts for re-normalization.")
def extract(dataset_id: str, label_key: str, labels: list[str], new_label: str = "",
            project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Subset to a set of labels and reset X to raw counts.

    Drops the parent's HVG flags, scaling, PCA, neighbours and UMAP — they were
    computed for a different question and reusing them is the classic
    subclustering error. Run skin.integrate.preprocess on the result.

    Args:
        dataset_id: Handle or label.
        label_key: obs column holding the labels.
        labels: Labels to keep.
        new_label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the cell count without minting.
        seed: RNG seed.
    """

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, label_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    present = set(map(str, adata.obs[label_key].astype(str).unique()))
    want = [str(x) for x in labels]
    missing = [x for x in want if x not in present]
    if missing:
        raise NotFound(f"labels not present: {missing}",
                       remedy=f"Labels available in {label_key!r}: {sorted(present)}")

    mask = adata.obs[label_key].astype(str).isin(want).to_numpy()
    n = int(mask.sum())
    ctx.code = (
        f"sub = adata[adata.obs[{label_key!r}].astype(str).isin({want!r})].copy()\n"
        "# restore raw counts: HVGs/scaling/PCA from the parent were chosen for a\n"
        "# different question and cannot resolve substructure inside one compartment\n"
        "sub.X = sub.layers['counts'].copy()\n"
        "for k in list(sub.obsm): del sub.obsm[k]\n"
        "sub.raw = None; sub.uns.pop('pca', None); sub.uns.pop('neighbors', None)\n"
    )
    if dry_run:
        ctx.summary = {"labels": want, "n_cells": n,
                       "pct_of_parent": round(100 * n / adata.n_obs, 1)}
        return
    if n < 50:
        raise BadParam(f"only {n} cells match those labels",
                       remedy="Subclustering under ~50 cells is not meaningful. Merge "
                              "labels, or analyse this population without subclustering.")

    sub = adata[mask].copy()
    if "counts" not in sub.layers:
        from ..errors import missing_counts

        raise missing_counts(parent or dataset_id)
    sub.X = sub.layers["counts"].copy()
    registry.set_x_state(sub, "counts")
    sub.raw = None
    for k in list(sub.obsm.keys()):
        del sub.obsm[k]
    for k in list(sub.obsp.keys()):
        del sub.obsp[k]
    for k in ("pca", "neighbors", "umap"):
        sub.uns.pop(k, None)
    for c in ("highly_variable", "means", "dispersions", "dispersions_norm",
              "mean", "std", "highly_variable_rank", "variances",
              "variances_norm", "highly_variable_nbatches"):
        if c in sub.var.columns:
            del sub.var[c]
    # Drop now-empty categories so downstream groupbys do not produce empty groups.
    import pandas as pd

    for c in sub.obs.columns:
        if isinstance(sub.obs[c].dtype, pd.CategoricalDtype):
            sub.obs[c] = sub.obs[c].cat.remove_unused_categories()

    dsid = registry.mint(ctx.project_id, sub, parent_id=parent, op="sub.extract",
                         params={"label_key": label_key, "labels": want},
                         label=new_label or f"sub_{'_'.join(want)[:24]}")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "labels": want, "n_obs": int(sub.n_obs),
                   "n_vars": int(sub.n_vars), "x_state": "counts",
                   "pct_of_parent": round(100 * n / adata.n_obs, 1)}
    ctx.warn("X reset to raw counts and the parent's feature selection/embeddings were "
             "dropped. Run skin.integrate.preprocess next to re-select HVGs for this "
             "compartment.")
    ctx.suggest("skin.integrate.preprocess", "skin.sub.pipeline")


@tool("skin.sub.pipeline", category="sub",
      summary="extract -> preprocess -> harmony -> neighbors -> umap -> leiden -> markers, one call.")
def pipeline(dataset_id: str, label_key: str, labels: list[str], resolution: float = 1.0,
             batch_key: str = "Sample", n_hvg: int = 2000, n_comps: int = 30,
             n_neighbors: int = 15, exclude_gene_groups: list[str] | None = None,
             skip_harmony: bool = False, new_label: str = "", project_id: str = "",
             dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Run the whole subclustering chain in one call. Every step is logged separately.

    Args:
        dataset_id: Handle or label.
        label_key: obs column holding the labels.
        labels: Labels to subcluster.
        resolution: Leiden resolution. Higher than the whole-object pass —
            1.0-1.6 is typical for a single compartment.
        batch_key: Batch key for the Harmony step.
        n_hvg: HVGs to select within the compartment.
        n_comps: PCA components.
        n_neighbors: kNN size.
        exclude_gene_groups: Gene groups to drop from the feature space, e.g.
            ["collagen","keratin","muscle"] when subclustering immune cells.
        skip_harmony: Skip integration (use for a single-sample subset).
        new_label: Human alias for the final handle.
        project_id: Defaults to the active project.
        dry_run: Report the chain without executing.
        seed: RNG seed.
    """
    from . import cluster_tools, integrate_tools

    if dry_run:
        ctx.summary = {
            "chain": ["skin.sub.extract", "skin.integrate.preprocess"]
            + ([] if skip_harmony else ["skin.integrate.harmony"])
            + ["skin.cluster.neighbors", "skin.cluster.umap", "skin.cluster.leiden",
               "skin.cluster.marker_genes"],
            "labels": labels, "resolution": resolution,
            "exclude_gene_groups": exclude_gene_groups or [],
        }
        return

    steps: list[dict[str, Any]] = []

    def _run(fn: Any, name: str, **kw: Any) -> str:
        out = fn(project_id=ctx.project_id, seed=seed, **kw)
        steps.append({"tool": name, "ok": out.get("ok"), "dataset_id": out.get("dataset_id"),
                      "memory_ref": out.get("memory_ref")})
        for w in out.get("warnings", []):
            ctx.warn(f"[{name}] {w}")
        if not out.get("ok"):
            err = out.get("error") or {}
            raise BadParam(f"{name} failed: {err.get('message', 'unknown')}",
                           remedy=err.get("remedy", ""), suggested_tool=name)
        return out["dataset_id"]

    cur = _run(extract, "skin.sub.extract", dataset_id=dataset_id, label_key=label_key,
               labels=labels, new_label=new_label)
    cur = _run(integrate_tools.preprocess, "skin.integrate.preprocess", dataset_id=cur,
               n_hvg=n_hvg, n_comps=n_comps, exclude_gene_groups=exclude_gene_groups or [])

    rep = "X_pca"
    if not skip_harmony:
        adata = registry.load(ctx.project_id, cur)
        if batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1:
            cur = _run(integrate_tools.harmony, "skin.integrate.harmony", dataset_id=cur,
                       batch_key=batch_key)
            rep = "X_pca_harmony"
        else:
            ctx.warn(f"{batch_key!r} has fewer than 2 levels in this subset; skipped Harmony.")

    cur = _run(cluster_tools.neighbors, "skin.cluster.neighbors", dataset_id=cur,
               use_rep=rep, n_neighbors=n_neighbors, n_pcs=n_comps)
    cur = _run(cluster_tools.umap, "skin.cluster.umap", dataset_id=cur)
    cur = _run(cluster_tools.leiden, "skin.cluster.leiden", dataset_id=cur,
               resolution=resolution)
    cluster_key = f"leiden_res{resolution:g}"
    cur = _run(cluster_tools.marker_genes, "skin.cluster.marker_genes", dataset_id=cur,
               groupby=cluster_key)

    if new_label:
        store.set_label(ctx.project_id, cur, new_label)
    adata = registry.load(ctx.project_id, cur)
    ctx.dataset_id = cur
    ctx.summary = {
        "dataset_id": cur, "labels": labels, "cluster_key": cluster_key,
        "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
        "n_clusters": int(adata.obs[cluster_key].nunique()),
        "cluster_sizes": adata.obs[cluster_key].value_counts().head(25).to_dict(),
        "steps": steps, "use_rep": rep,
    }
    ctx.suggest("skin.annotate.marker_report", "skin.annotate.apply_labels",
                "skin.sub.map_back")


@tool("skin.sub.drop_clusters", category="sub", destructive=True,
      summary="Remove contaminating sub-clusters. `reason` is required and is recorded.")
def drop_clusters(dataset_id: str, cluster_key: str, clusters: list[str], reason: str,
                  confirm: bool = False, new_label: str = "", project_id: str = "",
                  dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Drop clusters and record why. The reason goes to skin.memory.decision.

    Args:
        dataset_id: Handle or label.
        cluster_key: obs column of cluster ids.
        clusters: Cluster ids to remove.
        reason: Why. Required — a dropped cluster with no recorded reason is
            unauditable, and this is exactly the decision a reviewer asks about.
        confirm: Required. Removing cells is not reversible within a handle.
        new_label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report what would be removed.
        seed: RNG seed.
    """
    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, cluster_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if not reason.strip():
        raise BadParam("reason is required",
                       remedy="One sentence: what made these clusters non-biological?")

    present = set(map(str, adata.obs[cluster_key].astype(str).unique()))
    want = [str(c) for c in clusters]
    missing = [c for c in want if c not in present]
    if missing:
        raise NotFound(f"clusters not present: {missing}",
                       remedy=f"Clusters in {cluster_key!r}: {sorted(present)}")

    drop = adata.obs[cluster_key].astype(str).isin(want).to_numpy()
    n_drop, pct = int(drop.sum()), round(100 * drop.mean(), 1)
    ctx.code = (f"keep = ~adata.obs[{cluster_key!r}].astype(str).isin({want!r})\n"
                "adata = adata[keep].copy()\n")
    if dry_run:
        ctx.summary = {"would_remove": want, "n_cells": n_drop, "pct": pct}
        return
    confirm_or_raise(confirm, dry_run, "skin.sub.drop_clusters",
                     f"This removes clusters {want} — {n_drop} cells ({pct}%).")
    if pct > 60:
        ctx.warn(f"Dropping {pct}% of the object. If most sub-clusters look contaminated, "
                 f"the problem is probably upstream (ambient, or the wrong parent labels) "
                 f"rather than in these clusters.")

    sub = adata[~drop].copy()
    import pandas as pd

    for c in sub.obs.columns:
        if isinstance(sub.obs[c].dtype, pd.CategoricalDtype):
            sub.obs[c] = sub.obs[c].cat.remove_unused_categories()

    dsid = registry.mint(ctx.project_id, sub, parent_id=parent, op="sub.drop_clusters",
                         params={"cluster_key": cluster_key, "clusters": want,
                                 "reason": reason},
                         label=new_label or "cleaned")
    did = store.record_decision(
        ctx.project_id, question=f"drop clusters {want} from {cluster_key!r}?",
        choice=f"dropped {n_drop} cells ({pct}%) -> {dsid}",
        alternatives=["keep and label as contamination", "subcluster further",
                      "treat as ambient and fix upstream"],
        rationale=reason, author="skin.sub.drop_clusters")

    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "dropped": want, "n_removed": n_drop,
                   "pct_removed": pct, "n_obs": int(sub.n_obs), "decision_id": did}
    ctx.warn("The embedding and clustering are now stale — they were fit including the "
             "dropped cells. Run skin.sub.recluster before interpreting anything.")
    ctx.suggest("skin.sub.recluster", "skin.integrate.preprocess")


@tool("skin.sub.recluster", category="sub",
      summary="Re-run preprocess -> harmony -> neighbors -> umap -> leiden after dropping cells.")
def recluster(dataset_id: str, resolution: float = 1.0, batch_key: str = "Sample",
              n_hvg: int = 2000, n_comps: int = 30, exclude_gene_groups: list[str] | None = None,
              new_label: str = "", project_id: str = "", dry_run: bool = False,
              seed: int = 0, *, ctx: Ctx) -> None:
    """Rebuild the whole embedding from counts. Never reuse an embedding after dropping cells.

    Args:
        dataset_id: Handle or label.
        resolution: Leiden resolution.
        batch_key: Batch key for Harmony.
        n_hvg: HVGs to select.
        n_comps: PCA components.
        exclude_gene_groups: Gene groups to drop from the feature space.
        new_label: Human alias for the final handle.
        project_id: Defaults to the active project.
        dry_run: Report the chain only.
        seed: RNG seed.
    """
    from . import cluster_tools, integrate_tools

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if dry_run:
        ctx.summary = {"chain": ["preprocess", "harmony", "neighbors", "umap", "leiden",
                                 "marker_genes"], "resolution": resolution}
        return

    steps: list[dict[str, Any]] = []

    def _run(fn: Any, name: str, **kw: Any) -> str:
        out = fn(project_id=ctx.project_id, seed=seed, **kw)
        steps.append({"tool": name, "ok": out.get("ok"), "dataset_id": out.get("dataset_id")})
        if not out.get("ok"):
            err = out.get("error") or {}
            raise BadParam(f"{name} failed: {err.get('message')}", remedy=err.get("remedy", ""))
        for w in out.get("warnings", []):
            ctx.warn(f"[{name}] {w}")
        return out["dataset_id"]

    cur = _run(integrate_tools.preprocess, "skin.integrate.preprocess", dataset_id=parent,
               n_hvg=n_hvg, n_comps=n_comps, exclude_gene_groups=exclude_gene_groups or [])
    rep = "X_pca"
    if batch_key in adata.obs.columns and adata.obs[batch_key].nunique() > 1:
        cur = _run(integrate_tools.harmony, "skin.integrate.harmony", dataset_id=cur,
                   batch_key=batch_key)
        rep = "X_pca_harmony"
    cur = _run(cluster_tools.neighbors, "skin.cluster.neighbors", dataset_id=cur,
               use_rep=rep, n_pcs=n_comps)
    cur = _run(cluster_tools.umap, "skin.cluster.umap", dataset_id=cur)
    cur = _run(cluster_tools.leiden, "skin.cluster.leiden", dataset_id=cur,
               resolution=resolution)
    cluster_key = f"leiden_res{resolution:g}"
    cur = _run(cluster_tools.marker_genes, "skin.cluster.marker_genes", dataset_id=cur,
               groupby=cluster_key)
    if new_label:
        store.set_label(ctx.project_id, cur, new_label)

    a2 = registry.load(ctx.project_id, cur)
    ctx.dataset_id = cur
    ctx.summary = {"dataset_id": cur, "cluster_key": cluster_key,
                   "n_clusters": int(a2.obs[cluster_key].nunique()), "steps": steps}
    ctx.suggest("skin.annotate.marker_report", "skin.annotate.contamination_audit")


@tool("skin.sub.map_back", category="sub",
      summary="Write sub-labels onto the parent object by barcode.")
def map_back(sub_dataset_id: str, parent_dataset_id: str, obs_key: str,
             new_key: str = "", fill_value: str = "Other", force: bool = False,
             new_label: str = "", project_id: str = "", dry_run: bool = False,
             seed: int = 0, *, ctx: Ctx) -> None:
    """Transfer a sub-object's labels back onto its parent, matching on barcode.

    Refuses below a 95% barcode match rate without force=True — a low match rate
    means the two objects do not share an index, and the result would be a
    column of silently misassigned labels.

    Args:
        sub_dataset_id: The subclustered handle carrying the labels.
        parent_dataset_id: The handle to write onto.
        obs_key: Column in the sub-object to transfer.
        new_key: Column name on the parent. Defaults to `obs_key`.
        fill_value: Value for parent cells absent from the sub-object.
        force: Proceed despite a match rate below 95%.
        new_label: Human alias for the new parent handle.
        project_id: Defaults to the active project.
        dry_run: Report the match rate without writing.
        seed: RNG seed.
    """
    import pandas as pd

    from ..style import palettes as PAL

    sub = registry.load(ctx.project_id, sub_dataset_id)
    par = registry.load(ctx.project_id, parent_dataset_id, copy=True)
    require_obs(sub, obs_key)
    parent_id = store.resolve_dataset_ref(ctx.project_id, parent_dataset_id)
    target = new_key or obs_key

    s = pd.Series(sub.obs[obs_key].astype(str).values, index=sub.obs_names.astype(str))
    par_idx = par.obs_names.astype(str)
    matched = s.reindex(par_idx)
    n_match = int(matched.notna().sum())
    rate = n_match / max(sub.n_obs, 1)

    ctx.code = (f"lab = sub.obs[{obs_key!r}].astype(str)\n"
                f"parent.obs[{target!r}] = parent.obs_names.map(lab).fillna({fill_value!r})\n")
    if dry_run:
        ctx.summary = {"match_rate": round(rate, 4), "n_matched": n_match,
                       "n_sub_cells": int(sub.n_obs), "target_key": target}
        return
    if rate < 0.95 and not force:
        raise BadParam(
            f"only {rate:.1%} of the sub-object's barcodes are present in the parent",
            remedy=("The two handles probably do not share an index — check that the parent "
                    "is the object the subset was taken from and that no filtering happened "
                    "in between. skin.io.lineage shows the relationship. Pass force=True to "
                    "write anyway."),
            details={"match_rate": round(rate, 4), "n_matched": n_match,
                     "n_sub": int(sub.n_obs)},
        )
    if rate < 0.95:
        ctx.warn(f"Writing labels at a {rate:.1%} match rate because force=True. "
                 f"{int(sub.n_obs) - n_match} sub-object cells have no parent row.")

    vals = matched.fillna(fill_value)
    cats = [c for c in PAL.natural_order(sub.obs[obs_key].astype(str).unique())] + [fill_value]
    par.obs[target] = pd.Categorical(vals.values, categories=list(dict.fromkeys(cats)))
    PAL.apply_to_adata(par, target, {**PAL.celltype_palette(cats[:-1]),
                                     fill_value: PAL.CONTEXT_GREY})

    dsid = registry.mint(ctx.project_id, par, parent_id=parent_id, op="sub.map_back",
                         params={"sub_dataset_id": sub_dataset_id, "obs_key": obs_key,
                                 "new_key": target, "match_rate": round(rate, 4)},
                         label=new_label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "target_key": target, "match_rate": round(rate, 4),
                   "n_matched": n_match,
                   "counts": par.obs[target].value_counts().head(25).to_dict()}
    ctx.suggest("skin.plot.umap_highlight", "skin.abundance.proportions", "skin.de.pseudobulk")
