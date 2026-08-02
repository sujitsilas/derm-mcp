"""`skin.cluster.*` — neighbours, UMAP, Leiden, marker ranking, cluster QC."""

from __future__ import annotations

import logging
from typing import Any

from .. import registry
from ..errors import BadParam
from ..memory import store
from ._base import Ctx, pick_rep, require_obs, tool

logger = logging.getLogger(__name__)


@tool("skin.cluster.neighbors", category="cluster", summary="Build the kNN graph.")
def neighbors(dataset_id: str, use_rep: str = "X_pca_harmony", n_neighbors: int = 15,
              n_pcs: int = 30, metric: str = "euclidean", label: str = "",
              project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Compute the neighbour graph on the integrated embedding.

    Args:
        dataset_id: Handle or label.
        use_rep: obsm key. Falls back to X_pca_harmony -> X_scVI -> X_pca.
        n_neighbors: Neighbourhood size.
        n_pcs: Components to use from `use_rep`.
        metric: Distance metric.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the resolved representation.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    rep = pick_rep(adata, use_rep)
    if rep != use_rep:
        ctx.warn(f"{use_rep!r} not present; used {rep!r} instead. Run "
                 f"skin.integrate.harmony first if you meant to cluster on the "
                 f"batch-corrected embedding.")
    n_pcs_eff = int(min(n_pcs, adata.obsm[rep].shape[1]))

    ctx.code = (f"import scanpy as sc\n"
                f"sc.pp.neighbors(adata, use_rep={rep!r}, n_neighbors={n_neighbors}, "
                f"n_pcs={n_pcs_eff}, metric={metric!r}, random_state={seed})\n")
    if dry_run:
        ctx.summary = {"use_rep": rep, "n_neighbors": n_neighbors, "n_pcs": n_pcs_eff}
        return

    sc.pp.neighbors(adata, use_rep=rep, n_neighbors=n_neighbors, n_pcs=n_pcs_eff,
                    metric=metric, random_state=seed)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="cluster.neighbors",
                         params={"use_rep": rep, "n_neighbors": n_neighbors,
                                 "n_pcs": n_pcs_eff, "metric": metric, "seed": seed},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "use_rep": rep, "n_neighbors": n_neighbors,
                   "n_pcs": n_pcs_eff}
    ctx.suggest("skin.cluster.umap", "skin.cluster.leiden", "skin.cluster.leiden_sweep")


@tool("skin.cluster.umap", category="cluster", summary="Compute the UMAP embedding.")
def umap(dataset_id: str, min_dist: float = 0.5, spread: float = 1.0, n_components: int = 2,
         label: str = "", project_id: str = "", dry_run: bool = False, seed: int = 0,
         *, ctx: Ctx) -> None:
    """Embed for visualisation. UMAP geometry is for looking at, not for measuring.

    Args:
        dataset_id: Handle or label with a neighbour graph.
        min_dist: Minimum distance in the embedding.
        spread: Effective scale of embedded points.
        n_components: 2 for figures, 3 for the 3D state-space plots.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "neighbors" not in adata.uns:
        raise BadParam("no neighbour graph on this handle",
                       remedy="Run skin.cluster.neighbors first.",
                       suggested_tool="skin.cluster.neighbors")
    ctx.code = (f"sc.tl.umap(adata, min_dist={min_dist}, spread={spread}, "
                f"n_components={n_components}, random_state={seed})\n")
    if dry_run:
        ctx.summary = {"min_dist": min_dist, "spread": spread}
        return

    sc.tl.umap(adata, min_dist=min_dist, spread=spread, n_components=n_components,
               random_state=seed)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="cluster.umap",
                         params={"min_dist": min_dist, "spread": spread,
                                 "n_components": n_components, "seed": seed}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "n_components": n_components,
                   "obsm": list(adata.obsm.keys())}
    ctx.suggest("skin.cluster.leiden", "skin.plot.umap")


@tool("skin.cluster.leiden", category="cluster", summary="Leiden clustering.")
def leiden(dataset_id: str, resolution: float = 0.8, key_added: str = "", flavor: str = "igraph",
           n_iterations: int = 2, directed: bool = False, label: str = "", project_id: str = "",
           dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Cluster the neighbour graph. Key defaults to `leiden_res{resolution}`.

    Args:
        dataset_id: Handle or label with a neighbour graph.
        resolution: Higher = more clusters. 0.8 is a reasonable first pass on
            whole skin; use 1.0-1.6 when subclustering a single compartment.
        key_added: obs column name. Defaults to "leiden_res{resolution}".
        flavor: "igraph" (fast, recommended) or "leidenalg" (legacy).
        n_iterations: Leiden iterations. 2 with the igraph flavor.
        directed: Must be False with the igraph flavor.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the resolved key.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "neighbors" not in adata.uns:
        raise BadParam("no neighbour graph on this handle",
                       remedy="Run skin.cluster.neighbors first.",
                       suggested_tool="skin.cluster.neighbors")
    key = key_added or f"leiden_res{resolution:g}"

    fl = flavor
    if fl == "igraph":
        try:
            import igraph  # noqa: F401
        except ImportError:
            fl = "leidenalg"
            ctx.warn("python-igraph is unavailable; fell back to flavor='leidenalg'. "
                     "Results differ slightly from the igraph flavor.")

    ctx.code = (f"sc.tl.leiden(adata, resolution={resolution}, key_added={key!r},\n"
                f"             flavor={fl!r}, n_iterations={n_iterations}, "
                f"directed={directed if fl != 'igraph' else False}, random_state={seed})\n")
    if dry_run:
        ctx.summary = {"key_added": key, "resolution": resolution, "flavor": fl}
        return

    kw: dict[str, Any] = {"resolution": resolution, "key_added": key, "flavor": fl,
                          "n_iterations": n_iterations, "random_state": seed}
    if fl == "igraph":
        kw["directed"] = False
    else:
        kw["directed"] = bool(directed)
    sc.tl.leiden(adata, **kw)

    vc = adata.obs[key].value_counts()
    tiny = vc[vc < 20]
    if len(tiny):
        ctx.warn(f"{len(tiny)} clusters have fewer than 20 cells ({list(tiny.index)[:8]}). "
                 f"They are unlikely to survive marker ranking or pseudobulk DE; consider a "
                 f"lower resolution.")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="cluster.leiden",
                         params={"resolution": resolution, "key_added": key, "flavor": fl,
                                 "n_iterations": n_iterations, "seed": seed}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "cluster_key": key, "resolution": resolution,
                   "n_clusters": int(vc.size), "cluster_sizes": vc.head(30).to_dict()}
    ctx.suggest("skin.cluster.marker_genes", "skin.cluster.cluster_qc",
                "skin.annotate.score_lineages")


@tool("skin.cluster.leiden_sweep", category="cluster",
      summary="Sweep resolutions; report n_clusters, silhouette, stability, and a recommendation.")
def leiden_sweep(dataset_id: str, resolutions: list[float] | None = None,
                 n_bootstrap: int = 3, subsample: int = 6000, project_id: str = "",
                 dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Compare clustering resolutions and recommend one.

    Reports, per resolution: cluster count, mean silhouette on the integrated
    embedding, and bootstrap ARI stability (how reproducible the partition is
    under resampling). The recommendation balances stability against
    over-splitting; it is a suggestion, not an answer.

    Args:
        dataset_id: Handle or label with a neighbour graph.
        resolutions: Resolutions to try. Default [0.2,0.4,0.6,0.8,1.0,1.2].
        n_bootstrap: Bootstrap replicates for the stability estimate. 0 skips it.
        subsample: Cell cap for silhouette and bootstrap, for speed.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import scanpy as sc
    from sklearn.metrics import adjusted_rand_score, silhouette_score

    res_list = resolutions or [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = parent
    if "neighbors" not in adata.uns:
        raise BadParam("no neighbour graph", remedy="Run skin.cluster.neighbors first.",
                       suggested_tool="skin.cluster.neighbors")
    rep = pick_rep(adata, adata.uns["neighbors"].get("params", {}).get("use_rep", ""))

    ctx.code = ("for r in %r:\n"
                "    sc.tl.leiden(adata, resolution=r, key_added=f'leiden_res{r:g}',\n"
                "                 flavor='igraph', n_iterations=2, random_state=%d)\n"
                % (res_list, seed))
    if dry_run:
        ctx.summary = {"resolutions": res_list, "use_rep": rep}
        return

    rng = np.random.default_rng(seed)
    idx = (rng.choice(adata.n_obs, subsample, replace=False)
           if adata.n_obs > subsample else np.arange(adata.n_obs))
    X = np.asarray(adata.obsm[rep])[idx]

    rows = []
    for r in res_list:
        key = f"leiden_res{r:g}"
        sc.tl.leiden(adata, resolution=r, key_added=key, flavor="igraph", n_iterations=2,
                     directed=False, random_state=seed)
        lab = adata.obs[key].astype(str).to_numpy()
        n_clu = int(len(set(lab)))
        sil = None
        if 1 < n_clu < len(idx):
            try:
                sil = round(float(silhouette_score(X, lab[idx])), 4)
            except ValueError:
                sil = None
        stab = None
        if n_bootstrap > 0 and n_clu > 1:
            aris = []
            for b in range(n_bootstrap):
                sub = adata[rng.choice(adata.n_obs, int(0.8 * adata.n_obs), replace=False)].copy()
                sc.pp.neighbors(sub, use_rep=rep, n_neighbors=15, random_state=seed + b)
                sc.tl.leiden(sub, resolution=r, key_added="_b", flavor="igraph",
                             n_iterations=2, directed=False, random_state=seed + b)
                common = sub.obs_names
                aris.append(adjusted_rand_score(
                    adata.obs.loc[common, key].astype(str), sub.obs["_b"].astype(str)))
            stab = round(float(np.mean(aris)), 4)
        sizes = adata.obs[key].value_counts()
        rows.append({"resolution": r, "n_clusters": n_clu, "silhouette": sil,
                     "bootstrap_ari": stab, "min_cluster_size": int(sizes.min()),
                     "n_clusters_under_20": int((sizes < 20).sum())})

    # Recommendation: prefer high stability, penalise tiny clusters.
    def score(row: dict[str, Any]) -> float:
        s = (row["bootstrap_ari"] or 0.5) * 2.0
        s += (row["silhouette"] or 0.0)
        s -= 0.15 * row["n_clusters_under_20"]
        return s

    best = max(rows, key=score)
    ctx.summary = {
        "use_rep": rep, "per_resolution": rows,
        "recommended_resolution": best["resolution"],
        "recommendation_rationale": (
            f"resolution={best['resolution']} gives {best['n_clusters']} clusters with "
            f"bootstrap ARI {best['bootstrap_ari']} and silhouette {best['silhouette']}, "
            f"and {best['n_clusters_under_20']} clusters under 20 cells. Higher resolutions "
            f"here split without improving stability. Confirm against marker coherence "
            f"before committing — this metric cannot see biology."
        ),
    }
    ctx.suggest("skin.cluster.leiden", "skin.cluster.marker_genes",
                "skin.memory.record_decision")


@tool("skin.cluster.marker_genes", category="cluster",
      summary="Rank genes per cluster; full table exposed as a resource.")
def marker_genes(dataset_id: str, groupby: str, method: str = "wilcoxon", n_genes: int = 50,
                 use_raw: bool = False, pts: bool = True, layer: str = "lognorm",
                 label: str = "", project_id: str = "", dry_run: bool = False,
                 seed: int = 0, *, ctx: Ctx) -> None:
    """Rank marker genes per cluster and expose the full table as a resource.

    Only the top 10 per cluster come back in the tool return; the complete table
    is at `skin://dataset/{dataset_id}/markers/{groupby}` and is written to
    tables/. That keeps the caller's context alive.

    Args:
        dataset_id: Handle or label.
        groupby: obs column of cluster ids or labels.
        method: "wilcoxon" (default), "t-test", "t-test_overestim_var", or "logreg".
        n_genes: Genes to store per cluster.
        use_raw: Use `.raw` instead of X.
        pts: Compute fraction of cells expressing. Worth the cost.
        layer: Layer to test on. "lognorm" is correct after preprocess scaled X.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, groupby)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    key = f"rank_genes_{groupby}"

    use_layer = layer if layer and layer in adata.layers else None
    if use_layer is None and registry.get_x_state(adata) == "scaled" and not use_raw:
        ctx.warn("X is z-scaled and no 'lognorm' layer was found. Wilcoxon on scaled data "
                 "gives meaningless log-fold-changes. Re-run skin.integrate.preprocess, "
                 "which keeps layers['lognorm'].")

    ctx.code = (f"sc.tl.rank_genes_groups(adata, groupby={groupby!r}, method={method!r},\n"
                f"                        n_genes={n_genes}, use_raw={use_raw}, pts={pts},\n"
                f"                        layer={use_layer!r}, key_added={key!r})\n"
                f"df = sc.get.rank_genes_groups_df(adata, group=None, key={key!r})\n")
    if dry_run:
        ctx.summary = {"groupby": groupby, "method": method, "layer": use_layer,
                       "n_groups": int(adata.obs[groupby].nunique())}
        return

    sizes = adata.obs[groupby].value_counts()
    tiny = list(sizes[sizes < 3].index.astype(str))
    if tiny:
        ctx.warn(f"groups {tiny} have fewer than 3 cells and were excluded from ranking.")
        keep = ~adata.obs[groupby].astype(str).isin(tiny)
        adata = adata[keep].copy()
        adata.obs[groupby] = adata.obs[groupby].astype(str).astype("category")

    sc.tl.rank_genes_groups(adata, groupby=groupby, method=method, n_genes=n_genes,
                            use_raw=use_raw, pts=pts, layer=use_layer, key_added=key)
    df = sc.get.rank_genes_groups_df(adata, group=None, key=key)

    tbl = ctx.tabledir() / f"markers_{groupby}.csv"
    df.to_csv(tbl, index=False)
    ctx.add_artifact("table", tbl, caption=f"rank_genes_groups({groupby}, {method})",
                     params={"method": method, "n_genes": n_genes})

    top: dict[str, list[str]] = {}
    for g, sub in df.groupby("group", observed=True):
        top[str(g)] = sub.nsmallest(10, "pvals_adj")["names"].tolist()

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="cluster.marker_genes",
                         params={"groupby": groupby, "method": method, "n_genes": n_genes,
                                 "layer": use_layer}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {
        "dataset_id": dsid, "groupby": groupby, "method": method,
        "n_groups": len(top), "top10_per_group": top,
        "full_table_uri": f"skin://dataset/{dsid}/markers/{groupby}",
        "full_table_path": str(tbl),
    }
    ctx.suggest("skin.annotate.marker_report", "skin.annotate.score_lineages",
                "skin.plot.dotplot")


@tool("skin.cluster.cluster_qc", category="cluster",
      summary="Per-cluster QC: size, depth, %mt, doublet fraction, sample composition entropy.")
def cluster_qc(dataset_id: str, cluster_key: str, sample_key: str = "Sample",
               project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Flag clusters that are technical rather than biological.

    A cluster drawn from a single sample is a batch artifact until proven
    otherwise; sample composition entropy is the number to look at.

    Args:
        dataset_id: Handle or label.
        cluster_key: obs column of cluster ids.
        sample_key: obs column identifying samples.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, cluster_key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    has_sample = sample_key in adata.obs.columns
    o = adata.obs
    clusters = list(map(str, o[cluster_key].astype(str).unique()))
    ck = o[cluster_key].astype(str).to_numpy()
    n_samples = int(o[sample_key].nunique()) if has_sample else 1
    max_ent = np.log(n_samples) if n_samples > 1 else 1.0

    rows = []
    for c in sorted(clusters):
        m = ck == c
        sub = o[m]
        r: dict[str, Any] = {
            "cluster": c, "n_cells": int(m.sum()),
            "pct_of_total": round(100 * m.sum() / adata.n_obs, 2),
        }
        for col, name in (("n_genes_by_counts", "median_genes"),
                          ("total_counts", "median_counts"),
                          ("pct_counts_mt", "median_pct_mt")):
            if col in sub:
                r[name] = round(float(sub[col].median()), 2)
        if "predicted_doublet" in sub:
            r["doublet_fraction"] = round(float(sub["predicted_doublet"].astype(float).mean()), 4)
        if has_sample:
            vc = sub[sample_key].astype(str).value_counts(normalize=True)
            ent = float(-(vc * np.log(vc.clip(lower=1e-12))).sum())
            r["sample_entropy"] = round(ent, 3)
            r["sample_entropy_norm"] = round(ent / max_ent, 3)
            r["dominant_sample"] = str(vc.index[0])
            r["dominant_sample_frac"] = round(float(vc.iloc[0]), 3)
        rows.append(r)

    flagged = []
    for r in rows:
        why = []
        if r.get("dominant_sample_frac", 0) > 0.8 and n_samples > 2:
            why.append(f"{r['dominant_sample_frac']:.0%} from {r['dominant_sample']}")
        if r.get("doublet_fraction", 0) > 0.3:
            why.append(f"doublet fraction {r['doublet_fraction']:.2f}")
        if r.get("median_pct_mt", 0) > 25:
            why.append(f"median %mt {r['median_pct_mt']}")
        if r.get("median_genes", 1e9) < 250:
            why.append(f"median genes {r['median_genes']}")
        if why:
            flagged.append({"cluster": r["cluster"], "reasons": why})
    if flagged:
        ctx.warn(f"{len(flagged)} clusters look technical: "
                 f"{[f['cluster'] for f in flagged][:8]}. A cluster from one sample is a "
                 f"batch artifact until proven otherwise — check its markers before "
                 f"deciding.")

    ctx.summary = {"cluster_key": cluster_key, "n_clusters": len(rows), "n_samples": n_samples,
                   "per_cluster": rows[:30], "flagged": flagged}
    ctx.suggest("skin.annotate.marker_report", "skin.doublet.cluster_enrichment",
                "skin.annotate.contamination_audit")
