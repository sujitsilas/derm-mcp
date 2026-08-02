"""`skin.traj.*` — pseudotime and trajectory inference.

Two honest caveats this module surfaces rather than buries:

1. `basis="X_umap"` fits the principal graph on a 2D non-linear embedding. It
   produces the figures the lab wants, but the geometry is UMAP's, not the
   data's. `basis="X_pca_harmony"` is offered and the pseudotime<->timepoint
   correlation is reported for whichever you pick.
2. For a designed timecourse, CellRank's RealTimeKernel uses the actual
   experimental timepoints rather than inferring order from geometry, and is
   often the better-grounded primary method. `skin.traj.cellrank` is a
   first-class alternative, not an afterthought.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from .. import registry
from ..errors import BadParam, DependencyMissing, NotFound
from ..memory import store
from ..style import palettes as PAL
from ..style.rcparams import savefig, style
from ._base import Ctx, pick_rep, require_obs, tool

logger = logging.getLogger(__name__)


def _import_monocle() -> tuple[Any, Any, str]:
    """Prefer the real py_monocle; fall back to the shipped implementation."""
    try:
        from py_monocle import learn_graph, order_cells  # type: ignore

        return learn_graph, order_cells, "py_monocle (upstream)"
    except ImportError:
        from ..vendor.py_monocle import learn_graph, order_cells

        return learn_graph, order_cells, "skinmcp-simpleppt (shipped reimplementation)"


def _chains_from_edges(edges: Any) -> list[list[int]]:
    """Stitch the MST into continuous chains, breaking only at leaves/forks."""
    adj: dict[int, list[int]] = defaultdict(list)
    for a, b in edges:
        a, b = int(a), int(b)
        adj[a].append(b)
        adj[b].append(a)
    deg = {n: len(v) for n, v in adj.items()}

    def eid(u: int, v: int) -> tuple[int, int]:
        return (u, v) if u < v else (v, u)

    seen: set[tuple[int, int]] = set()
    chains: list[list[int]] = []
    for e in [n for n in adj if deg[n] != 2]:
        for nb in adj[e]:
            if eid(e, nb) in seen:
                continue
            seen.add(eid(e, nb))
            path = [e, nb]
            prev, cur = e, nb
            while deg.get(cur, 0) == 2:
                nxt = next((x for x in adj[cur] if x != prev), None)
                if nxt is None or eid(cur, nxt) in seen:
                    break
                seen.add(eid(cur, nxt))
                path.append(nxt)
                prev, cur = cur, nxt
            chains.append(path)
    for a, b in edges:
        a, b = int(a), int(b)
        if eid(a, b) not in seen:
            seen.add(eid(a, b))
            chains.append([a, b])
    return chains


def _chaikin(pts: Any, iters: int = 3) -> Any:
    """Corner-cutting smoothing that keeps the endpoints (no overshoot)."""
    import numpy as np

    pts = np.asarray(pts, dtype=float)
    for _ in range(iters):
        if len(pts) < 3:
            break
        new = [pts[0]]
        for i in range(len(pts) - 1):
            p, q = pts[i], pts[i + 1]
            new += [0.75 * p + 0.25 * q, 0.25 * p + 0.75 * q]
        new.append(pts[-1])
        pts = np.array(new)
    return pts


def _pick_root(centroids: Any, mst: Any, nn_node: Any, ident: Any, time: Any,
               root_label: str, min_frac: float = 0.35) -> dict[str, Any]:
    """Auditable root selection (reference `_pick_inf_root`).

    Candidates are centroids where the root label makes up >= `min_frac` of the
    cells; leaves are preferred; ties break on purity plus the fraction of
    earliest-timepoint cells. Falls back to the nearest centroid that actually
    contains root-label cells.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial.distance import cdist

    K = len(centroids)
    A = sp.csr_matrix(mst)
    deg = np.asarray(((A + A.T) != 0).sum(1)).ravel()
    is_root = np.asarray(ident) == root_label
    n_tot = np.bincount(nn_node, minlength=K).astype(float)
    n_root = np.bincount(nn_node[is_root], minlength=K).astype(float)
    root_frac = n_root / np.maximum(n_tot, 1)

    cand = np.where((n_tot > 0) & (root_frac >= min_frac))[0]
    if cand.size == 0:
        cand = np.where(n_tot > 0)[0]
    leaves = cand[deg[cand] == 1]
    pool = leaves if leaves.size else cand

    t = np.asarray(time, dtype=float)
    if np.isfinite(t).any():
        early = t == np.nanmin(t)
        n_erl = np.bincount(nn_node[early], minlength=K).astype(float)
        frac_early = n_erl / np.maximum(n_tot, 1)
    else:
        frac_early = np.zeros(K)

    root = int(pool[np.argmax(root_frac[pool] + 0.25 * frac_early[pool])])
    fallback = False
    if root_frac[root] == 0 and n_root.sum() > 0:
        centers = np.where(n_root > 0)[0]
        root = int(centers[np.argmin(cdist(centroids[root:root + 1], centroids[centers])[0])])
        fallback = True
    return {"root": root, "root_purity": round(float(root_frac[root]), 3),
            "is_leaf": bool(deg[root] == 1), "used_fallback": fallback,
            "n_candidates": int(cand.size), "n_leaf_candidates": int(leaves.size)}


@tool("skin.traj.monocle", category="traj",
      summary="Principal-graph trajectory with auditable root selection and a time sanity check.")
def monocle(dataset_id: str, cluster_key: str, root_label: str, basis: str = "X_umap",
            n_centroids: int = 20, p_threshold: float = 14.0, split_by: str = "",
            time_key: str = "Timepoint", make_plot: bool = True, label: str = "",
            project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Fit a principal graph and order cells along it.

    Reports Spearman rho between normalised pseudotime and the real experimental
    timepoint. Across a designed timecourse, rho near 0 is a red flag and the
    return says so.

    Args:
        dataset_id: Handle or label with an embedding.
        cluster_key: obs column of subtype labels; drives the graph and the root.
        root_label: The label the trajectory should start from, e.g. "Inf. Mono.".
        basis: "X_umap" (what the reference figures use) or "X_pca_harmony"
            (geometry of the data rather than of the embedding).
        n_centroids: Number of principal points.
        p_threshold: Branch-pruning threshold in the units of `basis`.
        split_by: Fit an independent graph per level of this obs column, e.g.
            "Type". That side-by-side comparison is usually the point.
        time_key: obs column holding the experimental timepoint.
        make_plot: Render the trajectory figure.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import scipy.sparse as sp
    from scipy.spatial import cKDTree
    from scipy.stats import spearmanr

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, cluster_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if basis not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding(basis, list(adata.obsm.keys()))
    ident_all = adata.obs[cluster_key].astype(str).to_numpy()
    if root_label not in set(ident_all):
        raise NotFound(f"root_label {root_label!r} is not in {cluster_key!r}",
                       remedy=f"Labels present: {sorted(set(ident_all))}")

    learn_graph, order_cells, impl = _import_monocle()
    ctx.code = (
        "from py_monocle import learn_graph, order_cells\n"
        f"xy = adata.obsm[{basis!r}]\n"
        f"projected, mst, centroids = learn_graph(matrix=xy, clusters=clu,\n"
        f"    n_centroids={n_centroids}, prune=True, p_threshold={p_threshold})\n"
        f"pt = order_cells(xy, centroids, mst=mst, projected_points=projected,\n"
        f"                 root_pr_cells=root)\n"
    )
    if dry_run:
        ctx.summary = {"implementation": impl, "basis": basis, "root_label": root_label,
                       "n_centroids": n_centroids, "split_by": split_by or None}
        return

    if basis == "X_umap":
        ctx.warn("The principal graph is being fit in UMAP space. That reproduces the "
                 "reference figures, but the geometry is UMAP's, not the data's — distances "
                 "along the graph are not distances in expression space. Re-run with "
                 "basis='X_pca_harmony' and compare the reported rho(pseudotime, timepoint).")

    time = np.full(adata.n_obs, np.nan)
    if time_key in adata.obs.columns:
        ext = adata.obs[time_key].astype(str).str.extract(r"(\d+\.?\d*)")[0]
        time = ext.astype(float).to_numpy()

    splits = ([""] if not split_by else
              PAL.natural_order(adata.obs[split_by].astype(str).unique()))
    if split_by:
        require_obs(adata, split_by)
    sk = adata.obs[split_by].astype(str).to_numpy() if split_by else None

    XY_all = np.asarray(adata.obsm[basis], dtype=float)[:, :2] if basis == "X_umap" \
        else np.asarray(adata.obsm[basis], dtype=float)
    results: dict[str, dict[str, Any]] = {}
    pt_full = np.full(adata.n_obs, np.nan)

    for sv in splits:
        m = np.ones(adata.n_obs, bool) if not split_by else (sk == sv)
        if m.sum() < 50:
            ctx.warn(f"split {sv!r} has only {int(m.sum())} cells; skipped.")
            continue
        XY = XY_all[m]
        ident = ident_all[m]
        import pandas as pd

        clu = pd.factorize(ident)[0]
        projected, mst, centroids = learn_graph(matrix=XY, clusters=clu,
                                                n_centroids=n_centroids, prune=True,
                                                p_threshold=p_threshold,
                                                random_state=seed)
        nn_node = cKDTree(centroids).query(XY)[1]
        rt = _pick_root(centroids, mst, nn_node, ident, time[m], root_label)
        pt = np.asarray(order_cells(XY, centroids, mst=mst, projected_points=projected,
                                    root_pr_cells=rt["root"]), dtype=float)
        pt[np.isinf(pt)] = np.nan
        fin = np.isfinite(pt)
        ptn = np.full_like(pt, np.nan)
        if fin.any():
            lo, hi = np.nanmin(pt), np.nanmax(pt)
            ptn[fin] = (pt[fin] - lo) / (hi - lo + 1e-12)
        pt_full[m] = ptn

        tm = time[m]
        both = fin & np.isfinite(tm)
        rho = float(spearmanr(ptn[both], tm[both]).statistic) if both.sum() > 2 else float("nan")
        ii, jj = sp.triu(sp.csr_matrix(mst) + sp.csr_matrix(mst).T, k=1).nonzero()
        results[sv or "all"] = {
            "n_cells": int(m.sum()), "n_centroids": int(centroids.shape[0]),
            "n_edges": int(len(ii)), **rt,
            "rho_pseudotime_vs_timepoint": (None if np.isnan(rho) else round(rho, 3)),
            "_centroids": centroids, "_edges": np.column_stack([ii, jj]),
            "_xy": XY, "_ident": ident, "_mask": m,
        }
        if not np.isnan(rho) and abs(rho) < 0.15 and both.sum() > 50:
            ctx.warn(f"split {sv or 'all'}: rho(pseudotime, {time_key}) = {rho:+.2f}. Across a "
                     f"designed timecourse a near-zero correlation means the trajectory is "
                     f"not recovering the experimental order. Try basis='X_pca_harmony', a "
                     f"different root, or skin.traj.cellrank, which uses the real timepoints.")
        if rt["used_fallback"]:
            ctx.warn(f"split {sv or 'all'}: no candidate centroid reached {root_label!r} "
                     f"purity; fell back to the nearest centroid that contains root cells "
                     f"(purity {rt['root_purity']}). Treat the direction as weakly supported.")

    if not results:
        raise BadParam("no split had enough cells to fit a trajectory")

    pt_key = f"pseudotime_{cluster_key}"
    adata.obs[pt_key] = pt_full
    fig_info = None
    if make_plot:
        fig_info = _plot_trajectory(ctx, adata, results, cluster_key, root_label, basis,
                                    split_by)

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="traj.monocle",
                         params={"cluster_key": cluster_key, "root_label": root_label,
                                 "basis": basis, "n_centroids": n_centroids,
                                 "p_threshold": p_threshold, "split_by": split_by,
                                 "implementation": impl}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {
        "dataset_id": dsid, "implementation": impl, "basis": basis,
        "pseudotime_key": pt_key, "root_label": root_label,
        "per_split": {k: {kk: vv for kk, vv in v.items() if not kk.startswith("_")}
                      for k, v in results.items()},
        "figure": fig_info,
    }
    ctx.suggest("skin.traj.pseudotime_genes", "skin.traj.cellrank",
                "skin.memory.record_decision")


def _plot_trajectory(ctx: Ctx, adata: Any, results: dict[str, dict[str, Any]],
                     cluster_key: str, root_label: str, basis: str,
                     split_by: str) -> dict[str, Any]:
    """Reference cell 127 rendering: smoothed graph, boxed labels, root star."""
    import matplotlib.patheffects as pe
    import matplotlib.pyplot as plt
    import numpy as np

    from ..style.panels import style_umap_axis

    keys = list(results)
    XY_all = np.asarray(adata.obsm[basis], dtype=float)[:, :2]
    ident_all = adata.obs[cluster_key].astype(str).to_numpy()
    labels = PAL.natural_order(set(ident_all))
    pal = PAL.get_from_adata(adata, cluster_key) or PAL.celltype_palette(labels)
    title_pal = PAL.condition_palette(keys) if split_by else {}

    pad = (XY_all[:, 0].max() - XY_all[:, 0].min()) * 0.04
    xlim = (XY_all[:, 0].min() - pad, XY_all[:, 0].max() + pad)
    ylim = (XY_all[:, 1].min() - pad, XY_all[:, 1].max() + pad)
    label_pos = {s: np.median(XY_all[ident_all == s], axis=0) for s in labels
                 if (ident_all == s).any()}

    with style("standard"):
        fig, axes = plt.subplots(1, len(keys), figsize=(6.5 * len(keys), 6), squeeze=False)
        for ax, k in zip(axes.flatten(), keys):
            d = results[k]
            C, XY, ident = d["_centroids"], d["_xy"], d["_ident"]
            ax.scatter(XY_all[:, 0], XY_all[:, 1], s=4, c="#ECECEC", linewidths=0,
                       rasterized=True, zorder=1)
            for s in labels:
                m = ident == s
                if m.any():
                    ax.scatter(XY[m, 0], XY[m, 1], s=9, color=pal.get(s, PAL.UNKNOWN),
                               linewidths=0, rasterized=True, zorder=2)
            smooths = [_chaikin(C[np.asarray(ch, int)], 3)
                       for ch in _chains_from_edges(d["_edges"])]
            for sm in smooths:
                ax.plot(sm[:, 0], sm[:, 1], color="white", lw=9.0, solid_capstyle="round",
                        solid_joinstyle="round", zorder=3)
            for sm in smooths:
                ax.plot(sm[:, 0], sm[:, 1], color="black", lw=5.0, solid_capstyle="round",
                        solid_joinstyle="round", zorder=4)
            for s, (lx, ly) in label_pos.items():
                ax.text(lx, ly, s, fontsize=15, fontweight="bold", ha="center", va="center",
                        zorder=9,
                        bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                                  edgecolor="none", alpha=0.82),
                        path_effects=[pe.withStroke(linewidth=2.0, foreground="white")])
            ax.scatter(*C[d["root"]], s=800, marker="*", c="#E8412A", edgecolor="black",
                       lw=2.0, zorder=12, label=f"root ({root_label})",
                       path_effects=[pe.withStroke(linewidth=4.0, foreground="white")])
            rho = d.get("rho_pseudotime_vs_timepoint")
            title = k if split_by else ""
            sub = f"ρ(pt,time)={rho:+.2f}" if rho is not None else "ρ n/a"
            ax.set_title(f"{title}\n{sub}".strip(), fontsize=20, fontweight="bold",
                         color=title_pal.get(k, "black"), pad=10)
            style_umap_axis(ax, label_fs=18)
            ax.set_xlim(xlim)
            ax.set_ylim(ylim)
            ax.legend(fontsize=11, frameon=True, loc="lower right", framealpha=0.9)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("trajectory") / f"trajectory_{cluster_key}")
        plt.close(fig)
    aid = ctx.add_artifact("figure", paths["pdf"],
                           caption=f"principal-graph trajectory rooted on {root_label}",
                           params={"basis": basis, "cluster_key": cluster_key})
    return {"artifact_id": aid, "paths": paths}


@tool("skin.traj.paga", category="traj", summary="PAGA connectivity graph over clusters.")
def paga(dataset_id: str, groupby: str, threshold: float = 0.05, make_plot: bool = True,
         label: str = "", project_id: str = "", dry_run: bool = False, seed: int = 0,
         *, ctx: Ctx) -> None:
    """Abstracted graph of cluster-to-cluster connectivity. Cheap and robust.

    Args:
        dataset_id: Handle or label with a neighbour graph.
        groupby: obs column of cluster labels.
        threshold: Minimum connectivity to draw an edge.
        make_plot: Render the PAGA graph.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, groupby)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if "neighbors" not in adata.uns:
        raise BadParam("no neighbour graph", remedy="Run skin.cluster.neighbors first.",
                       suggested_tool="skin.cluster.neighbors")
    ctx.code = f"sc.tl.paga(adata, groups={groupby!r})\nsc.pl.paga(adata, threshold={threshold})\n"
    if dry_run:
        ctx.summary = {"groupby": groupby}
        return

    sc.tl.paga(adata, groups=groupby)
    conn = np.asarray(adata.uns["paga"]["connectivities"].todense())
    groups = list(map(str, adata.obs[groupby].cat.categories)) \
        if hasattr(adata.obs[groupby], "cat") else PAL.natural_order(adata.obs[groupby])
    ii, jj = np.triu_indices_from(conn, k=1)
    edges = [{"a": groups[i], "b": groups[j], "connectivity": round(float(conn[i, j]), 3)}
             for i, j in zip(ii, jj) if conn[i, j] >= threshold]
    edges.sort(key=lambda e: -e["connectivity"])

    fig_info = None
    if make_plot:
        with style("standard"):
            fig, ax = plt.subplots(figsize=(7, 6.5))
            sc.pl.paga(adata, threshold=threshold, ax=ax, show=False, fontsize=11)
            paths = savefig(fig, ctx.figdir("trajectory") / f"paga_{groupby}")
            plt.close(fig)
        aid = ctx.add_artifact("figure", paths["pdf"], caption=f"PAGA over {groupby}")
        fig_info = {"artifact_id": aid, "paths": paths}

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="traj.paga",
                         params={"groupby": groupby, "threshold": threshold}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "groupby": groupby, "n_groups": len(groups),
                   "top_edges": edges[:20], "figure": fig_info}
    ctx.suggest("skin.traj.dpt", "skin.traj.monocle")


@tool("skin.traj.dpt", category="traj", summary="Diffusion pseudotime from a root cell.")
def dpt(dataset_id: str, root_label: str, label_key: str, n_dcs: int = 15,
        time_key: str = "Timepoint", label: str = "", project_id: str = "",
        dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Diffusion pseudotime. Reports the same timepoint sanity check as monocle.

    Args:
        dataset_id: Handle or label with a neighbour graph.
        root_label: Label whose centroid-nearest cell becomes the root.
        label_key: obs column holding the labels.
        n_dcs: Diffusion components.
        time_key: obs column of the experimental timepoint.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import scanpy as sc
    from scipy.stats import spearmanr

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if root_label not in set(adata.obs[label_key].astype(str)):
        raise NotFound(f"{root_label!r} not in {label_key!r}")
    ctx.code = (f"sc.tl.diffmap(adata, n_comps={n_dcs})\n"
                f"adata.uns['iroot'] = root_index\nsc.tl.dpt(adata)\n")
    if dry_run:
        ctx.summary = {"root_label": root_label, "n_dcs": n_dcs}
        return

    sc.tl.diffmap(adata, n_comps=n_dcs, random_state=seed)
    m = (adata.obs[label_key].astype(str) == root_label).to_numpy()
    D = np.asarray(adata.obsm["X_diffmap"])[:, 1:4]
    centre = D[m].mean(0)
    adata.uns["iroot"] = int(np.where(m)[0][np.argmin(((D[m] - centre) ** 2).sum(1))])
    sc.tl.dpt(adata)

    rho = None
    if time_key in adata.obs.columns:
        t = adata.obs[time_key].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
        pt = adata.obs["dpt_pseudotime"]
        ok = np.isfinite(t) & np.isfinite(pt)
        if ok.sum() > 2:
            rho = round(float(spearmanr(pt[ok], t[ok]).statistic), 3)
            if abs(rho) < 0.15:
                ctx.warn(f"rho(dpt_pseudotime, {time_key}) = {rho:+.2f}: the ordering does not "
                         f"track the experimental timecourse.")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="traj.dpt",
                         params={"root_label": root_label, "n_dcs": n_dcs}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "pseudotime_key": "dpt_pseudotime",
                   "root_index": int(adata.uns["iroot"]),
                   "rho_pseudotime_vs_timepoint": rho}
    ctx.suggest("skin.traj.pseudotime_genes", "skin.traj.monocle")


@tool("skin.traj.cellrank", category="traj",
      summary="CellRank RealTimeKernel — uses the actual experimental timepoints.")
def cellrank(dataset_id: str, time_key: str = "Timepoint", label_key: str = "",
             n_states: int = 5, use_rep: str = "X_pca_harmony", label: str = "",
             project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Fate mapping from real timepoints rather than embedding geometry.

    For a designed timecourse this is usually better grounded than a principal
    graph fit in UMAP space, because the transition matrix is constrained by
    when each cell was actually collected.

    Args:
        dataset_id: Handle or label with an embedding and a timepoint column.
        time_key: obs column of the experimental timepoint.
        label_key: obs column of cluster labels, used to name the macrostates.
        n_states: Number of macrostates for GPCCA.
        use_rep: obsm key for the kernel.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, time_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    rep = pick_rep(adata, use_rep)
    ctx.code = ("import cellrank as cr\n"
                f"adata.obs['_t'] = pd.Categorical(...)  # numeric order from {time_key!r}\n"
                "k = cr.kernels.RealTimeKernel.from_moscot(...)  # or .compute_transition_matrix()\n"
                f"g = cr.estimators.GPCCA(k); g.compute_macrostates(n_states={n_states})\n")
    if dry_run:
        ctx.summary = {"time_key": time_key, "use_rep": rep, "n_states": n_states}
        return
    try:
        import cellrank as cr
    except ImportError as e:
        raise DependencyMissing(
            "cellrank is not installed",
            remedy=('uv pip install "skin-mcp[traj]". skin.traj.monocle and skin.traj.dpt '
                    "work without it, but they infer order from geometry rather than from "
                    "the real timepoints."),
            suggested_tool="skin.traj.monocle") from e

    t = adata.obs[time_key].astype(str).str.extract(r"(\d+\.?\d*)")[0].astype(float)
    if t.isna().all():
        raise BadParam(f"could not parse numbers out of {time_key!r}",
                       remedy="Timepoints must contain a number, e.g. D7 / day10 / 14.")
    adata.obs["_skin_time"] = t.to_numpy()
    order = sorted(t.dropna().unique())
    import pandas as pd

    adata.obs["_skin_time_cat"] = pd.Categorical(t.astype(float), categories=order,
                                                 ordered=True)

    kernel = cr.kernels.RealTimeKernel(adata, time_key="_skin_time_cat")
    kernel.compute_transition_matrix(threshold=0.0)
    g = cr.estimators.GPCCA(kernel)
    g.compute_schur(n_components=max(n_states + 1, 4))
    g.compute_macrostates(n_states=n_states, cluster_key=label_key or None)
    g.predict_terminal_states()
    g.compute_fate_probabilities()

    ms = adata.obs.get("macrostates")
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="traj.cellrank",
                         params={"time_key": time_key, "n_states": n_states,
                                 "use_rep": rep}, label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "method": "RealTimeKernel + GPCCA",
                   "timepoints": [float(x) for x in order],
                   "macrostates": (ms.value_counts().head(12).to_dict()
                                   if ms is not None else None),
                   "terminal_states": (list(map(str, g.terminal_states.cat.categories))
                                       if getattr(g, "terminal_states", None) is not None
                                       else [])}
    ctx.suggest("skin.traj.pseudotime_genes", "skin.traj.monocle")


@tool("skin.traj.pseudotime_genes", category="traj",
      summary="Genes varying along pseudotime, with a binned heatmap.")
def pseudotime_genes(dataset_id: str, pseudotime_key: str, groupby: str = "", n_bins: int = 20,
                     top_n: int = 60, min_abs_corr: float = 0.2, make_plot: bool = True,
                     project_id: str = "", dry_run: bool = False, seed: int = 0,
                     *, ctx: Ctx) -> None:
    """Rank genes by their correlation with pseudotime and draw the binned heatmap.

    Args:
        dataset_id: Handle or label carrying a pseudotime column.
        pseudotime_key: obs column of pseudotime.
        groupby: Optional obs column to annotate the bins by majority label.
        n_bins: Pseudotime bins for the heatmap.
        top_n: Genes to show.
        min_abs_corr: Minimum |Spearman rho| to report a gene.
        make_plot: Render the heatmap.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp
    from scipy.stats import spearmanr

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, pseudotime_key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
    if dry_run:
        ctx.summary = {"pseudotime_key": pseudotime_key, "n_bins": n_bins}
        return

    pt = adata.obs[pseudotime_key].to_numpy(dtype=float)
    ok = np.isfinite(pt)
    if ok.sum() < 50:
        raise BadParam(f"only {int(ok.sum())} cells have finite pseudotime")
    sub = adata[ok]
    # Restrict to reasonably expressed genes; correlating sparse noise is useless.
    X = sub.X
    det = (np.asarray((X > 0).mean(0)).ravel() if sp.issparse(X) else (np.asarray(X) > 0).mean(0))
    genes = np.asarray(sub.var_names)[det > 0.10]
    if len(genes) > 4000:
        Xd = sub[:, genes].X
        v = (np.asarray(Xd.power(2).mean(0)).ravel() - np.asarray(Xd.mean(0)).ravel() ** 2
             if sp.issparse(Xd) else np.asarray(Xd).var(0))
        genes = genes[np.argsort(-v)[:4000]]

    M = sub[:, genes].X
    M = M.toarray() if sp.issparse(M) else np.asarray(M)
    p = pt[ok]
    rho = np.array([spearmanr(M[:, i], p).statistic for i in range(M.shape[1])])
    rho = np.nan_to_num(rho)
    sel = np.abs(rho) >= min_abs_corr
    ranked = pd.DataFrame({"gene": genes[sel], "rho": rho[sel]}).sort_values(
        "rho", key=lambda s: -s.abs())
    tbl = ctx.tabledir() / f"pseudotime_genes_{pseudotime_key}.csv"
    ranked.to_csv(tbl, index=False)
    ctx.add_artifact("table", tbl, caption=f"genes correlated with {pseudotime_key}")

    fig_info = None
    show = ranked.head(top_n)["gene"].tolist()
    if make_plot and show:
        bins = np.clip(np.digitize(p, np.quantile(p, np.linspace(0, 1, n_bins + 1)[1:-1])),
                       0, n_bins - 1)
        idx = [list(genes).index(g) for g in show]
        B = np.vstack([M[bins == b][:, idx].mean(0) if (bins == b).any()
                       else np.full(len(idx), np.nan) for b in range(n_bins)])
        Z = (B - np.nanmean(B, 0)) / (np.nanstd(B, 0) + 1e-9)
        with style("standard"):
            fig, ax = plt.subplots(figsize=(9, 0.22 * len(show) + 3))
            im = ax.imshow(Z.T, aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
            ax.set_yticks(range(len(show)))
            ax.set_yticklabels(show, fontsize=8)
            ax.set_xlabel("pseudotime bin", fontsize=15, fontweight="bold")
            fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="z-score")
            fig.tight_layout()
            paths = savefig(fig, ctx.figdir("trajectory") / f"pseudotime_heatmap_{pseudotime_key}")
            plt.close(fig)
        aid = ctx.add_artifact("figure", paths["pdf"],
                               caption=f"binned expression along {pseudotime_key}")
        fig_info = {"artifact_id": aid, "paths": paths}

    ctx.summary = {"pseudotime_key": pseudotime_key, "n_genes_tested": int(len(genes)),
                   "n_correlated": int(len(ranked)),
                   "top_up": ranked.head(12).to_dict("records"),
                   "top_down": ranked.tail(12).iloc[::-1].to_dict("records"),
                   "table_path": str(tbl), "figure": fig_info}
    ctx.suggest("skin.enrich.gsea", "skin.plot.heatmap")
