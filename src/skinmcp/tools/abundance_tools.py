"""`skin.abundance.*` — differential abundance and composition.

`milo_py` is a faithful pure-Python port of the reference notebook's cell 46:
kNN neighbourhoods, a quasi-Poisson GLM per neighbourhood with a shared
dispersion, and BH FDR. It falls back from `~ C(Timepoint) + Type` to `~ Type`
with an explicit warning when the design matrix is rank-deficient, rather than
silently producing garbage coefficients.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import registry
from ..errors import BadParam, DependencyMissing, InsufficientReplicates
from ..memory import store
from ..style import palettes as PAL
from ..style.rcparams import savefig, style
from ._base import Ctx, pick_rep, require_obs, tool

logger = logging.getLogger(__name__)


@tool("skin.abundance.milo_py", category="abundance",
      summary="Milo-style neighbourhood differential abundance, pure Python.")
def milo_py(
    dataset_id: str,
    label_key: str,
    condition_key: str,
    contrast: list[str],
    sample_key: str = "Sample",
    covariates: list[str] | None = None,
    prop: float = 0.10,
    k: int = 30,
    mix_thresh: float = 0.70,
    alpha_fdr: float = 0.10,
    use_rep: str = "X_pca_harmony",
    make_plots: bool = True,
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Test which regions of the manifold change in abundance between conditions.

    Milo tests neighbourhoods rather than discrete labels, so it sees shifts that
    happen inside a cell type as well as changes in cell-type frequency.

    Design is `~ C(covariates) + condition`; the covariate block is dropped with
    a warning if it is collinear with the condition.

    Args:
        dataset_id: Handle or label with a batch-corrected embedding.
        label_key: obs column used to annotate neighbourhoods by majority vote.
        condition_key: obs column of the condition.
        contrast: [test, reference], e.g. ["Burn","Sham"]. Positive LFC = up in test.
        sample_key: obs column of biological replicates.
        covariates: Blocking factors, default ["Timepoint"].
        prop: Fraction of cells sampled as neighbourhood indices.
        k: kNN size for the neighbourhood graph.
        mix_thresh: Majority fraction below which a neighbourhood is labelled "Mixed".
        alpha_fdr: FDR threshold.
        use_rep: obsm key for the graph.
        make_plots: Also render the beeswarm and the neighbourhood-graph UMAP.
        project_id: Defaults to the active project.
        dry_run: Report the design and neighbourhood count.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd
    import scanpy as sc
    import scipy.sparse as sp
    import statsmodels.api as sm
    from scipy.stats import norm as _norm
    from statsmodels.stats.multitest import multipletests

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    for kk in (label_key, condition_key, sample_key):
        require_obs(adata, kk)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    if len(contrast) != 2:
        raise BadParam("contrast must be [test, reference]")
    a, b = str(contrast[0]), str(contrast[1])
    covs = [c for c in (covariates if covariates is not None else ["Timepoint"])
            if c in adata.obs.columns]
    rep = pick_rep(adata, use_rep)

    ctx.code = (
        "import numpy as np, scipy.sparse as sp, statsmodels.api as sm\n"
        f"sc.pp.neighbors(ad, use_rep={rep!r}, n_neighbors={k})\n"
        "A = (ad.obsp['distances'] > 0).astype(np.int8); A = (A + A.T).tolil(); A.setdiag(1)\n"
        "A = (A.tocsr() > 0).astype(np.int8)          # union-symmetrize + binarize\n"
        f"cand = rng.choice(ad.n_obs, int({prop} * ad.n_obs), replace=False)\n"
        "# snap each candidate to its local-centroid nearest neighbour, then dedupe\n"
        "counts = (N.T @ S).toarray(); offset = np.log(counts.sum(0))\n"
        "# per-neighbourhood Poisson GLM, shared dispersion phi, BH FDR\n"
    )
    if dry_run:
        ctx.summary = {"design": f"~ C({covs}) + {condition_key}", "contrast": [a, b],
                       "use_rep": rep, "expected_nhoods": int(prop * adata.n_obs)}
        return

    rng = np.random.default_rng(seed)
    if "distances" not in adata.obsp:
        sc.pp.neighbors(adata, use_rep=rep, n_neighbors=k, random_state=seed)

    A = (adata.obsp["distances"] > 0).astype(np.int8)
    A = (A + A.T).tolil()
    A.setdiag(1)
    A = (A.tocsr() > 0).astype(np.int8)
    R = np.asarray(adata.obsm[rep])

    cand = rng.choice(adata.n_obs, size=max(1, int(prop * adata.n_obs)), replace=False)
    idx = []
    for c in cand:
        nb = A[c].indices
        idx.append(int(nb[np.argmin(((R[nb] - R[nb].mean(0)) ** 2).sum(1))]) if nb.size else int(c))
    idx = np.unique(idx)
    N = A[idx].T.tocsr()
    nh_size = np.asarray(N.sum(0)).ravel()

    smp = adata.obs[sample_key].astype(str).to_numpy()
    samples = sorted(set(smp))
    S = sp.csr_matrix((np.ones(adata.n_obs),
                       (np.arange(adata.n_obs), [samples.index(s) for s in smp])),
                      shape=(adata.n_obs, len(samples)))
    counts = (N.T @ S).toarray()

    cond_of = np.array([pd.Series(adata.obs.loc[smp == s, condition_key].astype(str))
                        .mode().iloc[0] for s in samples])
    keep_s = counts.sum(0) > 0
    counts = counts[:, keep_s]
    cond_keep = cond_of[keep_s]
    samples_keep = [s for s, kp in zip(samples, keep_s) if kp]
    is_a = (cond_keep == a).astype(float)
    n_a, n_b = int((cond_keep == a).sum()), int((cond_keep == b).sum())
    if n_a < 2 or n_b < 2:
        raise InsufficientReplicates(
            f"need at least 2 samples per arm; got {a}={n_a}, {b}={n_b}",
            remedy="Differential abundance across conditions needs biological replicates. "
                   "Use skin.abundance.proportions for a descriptive summary instead.",
            suggested_tool="skin.abundance.proportions")
    offset = np.log(counts.sum(0))

    design_note = f"~ C({'+'.join(covs)}) + {condition_key}" if covs else f"~ {condition_key}"
    X = np.ones((is_a.size, 1))
    for cov in covs:
        lv = np.array([pd.Series(adata.obs.loc[smp == s, cov].astype(str)).mode().iloc[0]
                       for s in samples_keep])
        levels = PAL.order_timepoints(set(lv))
        D = pd.get_dummies(pd.Categorical(lv, categories=levels),
                           drop_first=True).to_numpy(float)
        if D.size:
            X = np.column_stack([X, D])
    X = np.column_stack([X, is_a])
    bcol = X.shape[1] - 1
    if np.linalg.matrix_rank(X) < X.shape[1]:
        X = np.column_stack([np.ones(is_a.size), is_a])
        bcol = 1
        design_note = f"~ {condition_key}  (covariate block dropped: collinear)"
        ctx.warn(f"{covs} are collinear with {condition_key} in this design; fell back to "
                 f"`~ {condition_key}`. The timepoint effect cannot be separated from the "
                 f"condition effect with these samples.")

    n_nh = len(idx)
    coef = np.full(n_nh, np.nan)
    se1 = np.full(n_nh, np.nan)
    phi = np.full(n_nh, np.nan)
    for j in range(n_nh):
        if counts[j].sum() < 5:
            continue
        try:
            r = sm.GLM(counts[j], X, family=sm.families.Poisson(), offset=offset).fit()
            coef[j] = r.params[bcol]
            se1[j] = np.sqrt(r.cov_params()[bcol, bcol])
            phi[j] = r.pearson_chi2 / max(r.df_resid, 1)
        except Exception:  # noqa: BLE001 - a few neighbourhoods always fail to converge
            pass

    ok = np.isfinite(coef) & np.isfinite(se1) & (se1 > 0)
    if not ok.any():
        raise BadParam("no neighbourhood produced a usable GLM fit",
                       remedy="Increase `prop` or `k`, or check that the samples have "
                              "enough cells.")
    phi_common = max(1.0, float(np.nanmedian(phi[ok])))
    z = coef / (se1 * np.sqrt(phi_common))
    lfc = coef / np.log(2)
    pval = np.ones(n_nh)
    pval[ok] = 2 * _norm.sf(np.abs(z[ok]))
    fdr = np.ones(n_nh)
    fdr[ok] = multipletests(pval[ok], method="fdr_bh")[1]
    sig = ok & (fdr < alpha_fdr)

    subs = adata.obs[label_key].astype(str).to_numpy()
    subcats = PAL.natural_order(set(subs))
    SUB1 = sp.csr_matrix((np.ones(adata.n_obs),
                          (np.arange(adata.n_obs), [subcats.index(s) for s in subs])),
                         shape=(adata.n_obs, len(subcats)))
    nsub = (N.T @ SUB1).toarray()
    frac = nsub.max(1) / np.maximum(nsub.sum(1), 1)
    annot = np.array([subcats[i] for i in nsub.argmax(1)], dtype=object)
    annot[frac < mix_thresh] = "Mixed"

    df = pd.DataFrame({"nhood": np.arange(n_nh), "index_cell": idx, "size": nh_size,
                       "label": annot, "lfc": lfc, "z": z, "pval": pval, "fdr": fdr,
                       "significant": sig})
    p = ctx.tabledir() / f"milo_{label_key}_{a}_vs_{b}.csv"
    df.to_csv(p, index=False)
    ctx.add_artifact("table", p, caption=f"Milo neighbourhoods {a} vs {b}")

    per_label = []
    for g in subcats + ["Mixed"]:
        m = annot == g
        if not m.any():
            continue
        per_label.append({
            "label": g, "n_nhoods": int(m.sum()),
            "n_sig": int((m & sig).sum()),
            "n_up": int((m & sig & (lfc > 0)).sum()),
            "n_down": int((m & sig & (lfc < 0)).sum()),
            "median_lfc": round(float(np.nanmedian(lfc[m & ok])) if (m & ok).any() else 0.0, 3),
        })
    per_label.sort(key=lambda r: r["median_lfc"])

    run_id = store.new_id("da", 8)
    figs = []
    if make_plots:
        figs.append(_milo_beeswarm(ctx, df, subcats, a, b, alpha_fdr, run_id))
        if "X_umap" in adata.obsm:
            figs.append(_milo_nhood_graph(ctx, adata, df, idx, a, b, run_id))

    store.record_run(ctx.project_id, run_id, "abundance", resolved or dataset_id,
                     {"method": "milo_py", "label_key": label_key,
                      "condition_key": condition_key, "contrast": [a, b],
                      "design": design_note, "prop": prop, "k": k, "alpha_fdr": alpha_fdr},
                     {"table": str(p), "per_label": per_label})

    ctx.summary = {
        "run_id": run_id, "method": "milo_py", "design": design_note, "contrast": [a, b],
        "n_neighbourhoods": n_nh, "median_nhood_size": int(np.median(nh_size)),
        "dispersion_phi": round(phi_common, 3),
        "n_significant": int(sig.sum()),
        f"n_up_in_{a}": int((sig & (lfc > 0)).sum()),
        f"n_up_in_{b}": int((sig & (lfc < 0)).sum()),
        "per_label": per_label, "n_samples": {a: n_a, b: n_b}, "table_path": str(p),
        "figures": [f for f in figs if f],
    }
    ctx.suggest("skin.abundance.proportions", "skin.abundance.sccoda",
                "skin.memory.record_decision")


def _milo_beeswarm(ctx: Ctx, df: Any, subcats: list[str], a: str, b: str,
                   alpha_fdr: float, run_id: str) -> dict[str, Any] | None:
    """Reference cell 46 figure 1."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    ok = np.isfinite(df["lfc"].to_numpy())
    if not ok.any():
        return None
    lfc = df["lfc"].to_numpy()
    sig = df["significant"].to_numpy().astype(bool)
    annot = df["label"].to_numpy()
    groups = [g for g in subcats if (annot == g).any()]
    groups = sorted(groups, key=lambda g: (np.nanmedian(lfc[(annot == g) & ok])
                                           if ((annot == g) & ok).any() else 0))
    dmax = float(np.nanpercentile(np.abs(lfc[ok]), 98)) or 1.0
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)
    cmap = plt.get_cmap("RdBu_r")
    rng = np.random.default_rng(0)

    with style("standard"):
        fig, ax = plt.subplots(figsize=(7.5, max(4.0, 0.55 * len(groups) + 2)))
        ax.axvline(0, color="0.4", lw=1.4, ls="--", zorder=1)
        for i, g in enumerate(groups):
            ns = (annot == g) & ok & ~sig
            ss = (annot == g) & sig
            ax.scatter(np.clip(lfc[ns], -dmax, dmax), i + rng.uniform(-.34, .34, ns.sum()),
                       s=18, color="0.82", zorder=2)
            ax.scatter(np.clip(lfc[ss], -dmax, dmax), i + rng.uniform(-.34, .34, ss.sum()),
                       s=34, c=np.clip(lfc[ss], -dmax, dmax), cmap=cmap, norm=norm,
                       edgecolor="black", linewidth=0.3, zorder=3)
        ax.set_yticks(range(len(groups)))
        ax.set_yticklabels(groups, fontsize=17, fontweight="bold")
        ax.set_xlim(-dmax * 1.15, dmax * 1.15)
        ax.set_xlabel(f"log$_2$FC  ({b} ← → {a})", fontsize=16, fontweight="bold")
        ax.set_title(f"Differential abundance (FDR<{alpha_fdr})", fontsize=13, fontweight="bold")
        ax.tick_params(axis="x", labelsize=17)
        cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax,
                          fraction=0.03, pad=0.02)
        cb.set_ticks([])
        cb.outline.set_linewidth(0.8)
        cb.ax.text(0.5, 1.03, a, transform=cb.ax.transAxes, ha="center", va="bottom",
                   fontsize=15, fontweight="bold", color=PAL.BURN)
        cb.ax.text(0.5, -0.03, b, transform=cb.ax.transAxes, ha="center", va="top",
                   fontsize=15, fontweight="bold", color=PAL.SHAM_ALT)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("abundance") / f"milo_beeswarm_{run_id}", hero=True)
        plt.close(fig)
    aid = ctx.add_artifact("figure", paths["pdf"],
                           caption=f"Milo beeswarm {a} vs {b} (FDR<{alpha_fdr})",
                           params={"run_id": run_id})
    return {"kind": "beeswarm", "artifact_id": aid, "paths": paths}


def _milo_nhood_graph(ctx: Ctx, adata: Any, df: Any, idx: Any, a: str, b: str,
                      run_id: str) -> dict[str, Any] | None:
    """Reference cell 47: neighbourhood graph drawn on the UMAP."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import TwoSlopeNorm

    XY = np.asarray(adata.obsm["X_umap"])
    lfc = df["lfc"].to_numpy()
    sig = df["significant"].to_numpy().astype(bool)
    size = df["size"].to_numpy(dtype=float)
    ok = np.isfinite(lfc)
    if not ok.any():
        return None
    dmax = float(np.nanpercentile(np.abs(lfc[ok]), 98)) or 1.0
    norm = TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)

    with style("standard"):
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.scatter(XY[:, 0], XY[:, 1], s=2, c=PAL.CONTEXT_GREY, linewidths=0,
                   rasterized=True, zorder=1)
        P = XY[idx]
        s = 12 + 90 * (size - size.min()) / max(size.max() - size.min(), 1)
        ax.scatter(P[~sig, 0], P[~sig, 1], s=s[~sig] * 0.5, c="0.75", linewidths=0,
                   zorder=2, alpha=0.7)
        sc_ = ax.scatter(P[sig, 0], P[sig, 1], s=s[sig], c=np.clip(lfc[sig], -dmax, dmax),
                         cmap="RdBu_r", norm=norm, edgecolor="black", linewidth=0.3, zorder=3)
        from ..style.panels import style_umap_axis

        style_umap_axis(ax, title=f"Differential abundance neighbourhoods\n{a} vs {b}",
                        label_fs=18, title_fs=17)
        cb = fig.colorbar(sc_, ax=ax, fraction=0.04, pad=0.02)
        cb.set_label(f"log$_2$FC ({b} ← → {a})", fontsize=13, fontweight="bold")
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("abundance") / f"milo_nhood_graph_{run_id}")
        plt.close(fig)
    aid = ctx.add_artifact("figure", paths["pdf"],
                           caption=f"Milo neighbourhood graph {a} vs {b}",
                           params={"run_id": run_id})
    return {"kind": "nhood_graph", "artifact_id": aid, "paths": paths}


@tool("skin.abundance.proportions", category="abundance",
      summary="Per-sample proportions with stacked bars, line plots, SEM and significance stars.")
def proportions(dataset_id: str, label_key: str, sample_key: str = "Sample",
                group_keys: list[str] | None = None, transform: str = "arcsine",
                test: str = "welch", split_by: str = "", make_plots: bool = True,
                bar_by: str = "", project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Per-sample composition table, tests, stacked bars and timepoint lines.

    n is samples, not cells. Proportions are compositional (they sum to 1), so
    the test is run on a variance-stabilising transform; even then a rise in one
    population mechanically depresses the others, and skin.abundance.sccoda
    handles that constraint properly.

    Args:
        dataset_id: Handle or label.
        label_key: obs column of cell type labels.
        sample_key: obs column of biological replicates.
        group_keys: Design columns, e.g. ["Type","Timepoint"]. Defaults to what is present.
        transform: "arcsine", "logit", or "none".
        test: "welch" (Welch t-test) or "mannwhitney".
        split_by: Optional obs column to facet the line plot by.
        make_plots: Render the stacked bars and line plots.
        bar_by: Draw ONE bar per level of this obs column instead of one per
            sample — e.g. "Type_Timepoint" for eight bars rather than twenty-one.
            Bars are the mean of the per-sample proportions, so the replicate
            structure is untouched: `sample_key` remains the unit of analysis and
            the statistics are unchanged. Bar order follows the column's category
            order, so skin.meta.order_categorical sets it (Sham_D7, Burn_D7, ...).
            Do NOT get this effect by passing a group column as `sample_key`:
            that makes each group a single replicate and silently destroys every
            p-value in the table.
        project_id: Defaults to the active project.
        dry_run: Report the design only.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd
    from scipy.stats import mannwhitneyu, ttest_ind

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, label_key)
    require_obs(adata, sample_key)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    gk = [g for g in (group_keys if group_keys is not None
                      else ["Type", "Timepoint"]) if g in adata.obs.columns]

    ctx.code = (
        f"counts = (adata.obs.groupby([{sample_key!r}] + {gk!r} + [{label_key!r}])\n"
        "          .size().unstack(fill_value=0))\n"
        "props = counts.div(counts.sum(1), axis=0)\n"
        f"# test on the {transform} transform; stars at .05/.01/.001\n"
    )
    if dry_run:
        ctx.summary = {"group_keys": gk, "n_labels": int(adata.obs[label_key].nunique()),
                       "n_samples": int(adata.obs[sample_key].nunique())}
        return

    o = adata.obs
    keys = [sample_key, *gk]
    counts = (o.groupby(keys + [label_key], observed=True).size()
              .unstack(label_key, fill_value=0))
    props = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).reset_index()
    labels = [c for c in counts.columns]

    def tf(v: Any) -> Any:
        v = np.clip(np.asarray(v, dtype=float), 1e-6, 1 - 1e-6)
        if transform == "arcsine":
            return np.arcsin(np.sqrt(v))
        if transform == "logit":
            return np.log(v / (1 - v))
        return v

    stats_rows: list[dict[str, Any]] = []
    cond_key = gk[0] if gk else ""
    if cond_key:
        arms = PAL.natural_order(props[cond_key].astype(str).unique())
        if len(arms) == 2:
            strata = ([None] if len(gk) < 2
                      else PAL.order_timepoints(props[gk[1]].astype(str).unique()))
            for lab in labels:
                for st in strata:
                    sel = props if st is None else props[props[gk[1]].astype(str) == st]
                    x = tf(sel.loc[sel[cond_key].astype(str) == arms[0], lab].dropna())
                    y = tf(sel.loc[sel[cond_key].astype(str) == arms[1], lab].dropna())
                    if len(x) < 2 or len(y) < 2:
                        p = np.nan
                    elif test == "mannwhitney":
                        p = float(mannwhitneyu(x, y, alternative="two-sided").pvalue)
                    else:
                        p = float(ttest_ind(x, y, equal_var=False).pvalue)
                    stats_rows.append({
                        "label": lab, "stratum": st, "arm_a": arms[0], "arm_b": arms[1],
                        f"n_{arms[0]}": int(len(x)), f"n_{arms[1]}": int(len(y)),
                        "mean_a": round(float(np.mean(x)), 4) if len(x) else None,
                        "mean_b": round(float(np.mean(y)), 4) if len(y) else None,
                        "p": None if not np.isfinite(p) else round(p, 5),
                        "stars": __import__("skinmcp.style.panels", fromlist=["x"])
                        .significance_stars(p),
                    })
        else:
            ctx.warn(f"{cond_key!r} has {len(arms)} levels; pairwise tests need exactly 2. "
                     f"Reporting proportions without a test.")

    n_small = int((counts.sum(axis=1) < 100).sum())
    if n_small:
        ctx.warn(f"{n_small} samples contribute fewer than 100 cells; their proportions are "
                 f"noisy and dominate the SEM.")
    ctx.warn("Proportions are compositional: they sum to 1, so an increase in one population "
             "mechanically decreases the others. skin.abundance.sccoda models that "
             "constraint properly; use it before claiming a specific population 'decreased'.")

    pcsv = ctx.tabledir() / f"proportions_{label_key}.csv"
    props.to_csv(pcsv, index=False)
    ctx.add_artifact("table", pcsv, caption=f"per-sample proportions of {label_key}")
    scsv = None
    if stats_rows:
        scsv = ctx.tabledir() / f"proportions_stats_{label_key}.csv"
        pd.DataFrame(stats_rows).to_csv(scsv, index=False)
        ctx.add_artifact("table", scsv, caption=f"proportion tests for {label_key}")

    figs = []
    if make_plots:
        figs.append(_stacked_bars(ctx, props, labels, sample_key, gk, label_key,
                                  adata, bar_by=bar_by))
        if len(gk) >= 2:
            figs.append(_proportion_lines(ctx, props, labels, gk, stats_rows, label_key, adata))

    run_id = store.new_id("prop", 8)
    store.record_run(ctx.project_id, run_id, "abundance", resolved or dataset_id,
                     {"method": "proportions", "label_key": label_key, "transform": transform,
                      "test": test, "group_keys": gk},
                     {"table": str(pcsv), "stats": str(scsv) if scsv else None})

    ctx.summary = {
        "run_id": run_id, "label_key": label_key, "group_keys": gk,
        "n_samples": int(props.shape[0]), "n_labels": len(labels),
        "transform": transform, "test": test,
        "significant": [r for r in stats_rows if r.get("stars")][:15],
        "n_tests": len(stats_rows),
        "table_path": str(pcsv), "figures": [f for f in figs if f],
    }
    ctx.suggest("skin.abundance.milo_py", "skin.abundance.sccoda", "skin.plot.legend_only")


def _stacked_bars(ctx: Ctx, props: Any, labels: list[str], sample_key: str,
                  gk: list[str], label_key: str, adata: Any,
                  bar_by: str = "") -> dict[str, Any] | None:
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    from matplotlib.ticker import PercentFormatter

    pal = PAL.get_from_adata(adata, label_key) or PAL.celltype_palette(labels)
    x_key = sample_key
    if bar_by and bar_by not in props.columns:
        # props carries sample_key and the design columns, so a composite like
        # "Type_Timepoint" is usually absent. It is constant within a sample by
        # construction, so it can be mapped across — but only if it really is
        # constant, otherwise the bar would average across different groups.
        if bar_by in adata.obs.columns:
            per = (adata.obs.groupby(sample_key, observed=True)[bar_by]
                   .agg(lambda s: s.astype(str).mode().iloc[0]))
            nun = adata.obs.groupby(sample_key, observed=True)[bar_by].nunique()
            if (nun > 1).any():
                ctx.warn(f"{bar_by} is not constant within {sample_key} "
                         f"({int((nun > 1).sum())} samples span several levels); bars stay "
                         f"per {sample_key} rather than averaging across groups.")
                bar_by = ""
            else:
                src_cat = adata.obs[bar_by]
                props = props.copy()
                props[bar_by] = props[sample_key].astype(str).map(per.to_dict())
                if isinstance(src_cat.dtype, pd.CategoricalDtype):
                    props[bar_by] = pd.Categorical(
                        props[bar_by], categories=list(src_cat.cat.categories), ordered=True)
        else:
            ctx.warn(f"bar_by={bar_by!r} is not an obs column; bars stay per {sample_key}.")
            bar_by = ""
    if bar_by and bar_by in props.columns:
        # One bar per group: the mean of the per-sample proportions, which is
        # what a reader expects a group bar to mean. Averaging the proportions
        # rather than pooling cells keeps every sample weighted equally, so a
        # deeply sequenced mouse does not quietly become the group.
        d = (props.groupby(bar_by, observed=True)[labels].mean().reset_index())
        n_per = props.groupby(bar_by, observed=True).size()
        # Order follows the column's own categories, so
        # skin.meta.order_categorical is what sets it.
        src = props[bar_by]
        order = (list(src.cat.categories) if isinstance(src.dtype, pd.CategoricalDtype)
                 else PAL.natural_order(src.astype(str).unique()))
        order = [o for o in order if o in set(d[bar_by].astype(str))]
        d[bar_by] = pd.Categorical(d[bar_by].astype(str), categories=order, ordered=True)
        d = d.sort_values(bar_by)
        x_key = bar_by
        bar_info = {"bar_by": bar_by, "n_bars": int(len(d)),
                    "samples_per_bar": {str(k): int(v) for k, v in n_per.items()}}
        ctx.warn(f"Bars are group means over {sample_key} ({len(d)} bars). The table and "
                 f"every p-value are still computed per {sample_key}; only the drawing "
                 f"is aggregated.")
    else:
        bar_info = {}
        sort_cols = [c for c in gk if c in props.columns] + [sample_key]
        d = props.sort_values(sort_cols)
    x = np.arange(len(d))
    with style("standard"):
        fig, ax = plt.subplots(figsize=(max(6, 0.42 * len(d) + 3), 6))
        bottom = np.zeros(len(d))
        for lab in labels:
            v = d[lab].fillna(0).to_numpy(dtype=float)
            ax.bar(x, v, bottom=bottom, color=pal.get(lab, PAL.UNKNOWN), width=0.85,
                   label=lab, edgecolor="white", linewidth=0.4)
            bottom += v
        ax.set_xticks(x)
        ax.set_xticklabels(d[x_key].astype(str), rotation=90, fontsize=11)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(PercentFormatter(1.0))
        ax.set_ylabel("Proportion of cells", fontsize=18, fontweight="bold")
        ax.legend(frameon=False, fontsize=11, loc="center left", bbox_to_anchor=(1.01, 0.5))
        fig.tight_layout()
        suffix = f"_by_{bar_by}" if bar_by else ""
        paths = savefig(fig,
                        ctx.figdir("abundance") / f"stacked_proportions_{label_key}{suffix}")
        plt.close(fig)
    aid = ctx.add_artifact("figure", paths["pdf"],
                           caption=f"stacked proportions of {label_key} per "
                                   f"{bar_by or sample_key}")
    return {"kind": "stacked_bars", "artifact_id": aid, "paths": paths, **bar_info}


def _proportion_lines(ctx: Ctx, props: Any, labels: list[str], gk: list[str],
                      stats_rows: list[dict[str, Any]], label_key: str,
                      adata: Any) -> dict[str, Any] | None:
    """Reference cell 22: mean +- SEM over timepoints, one panel per condition."""
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.ticker import PercentFormatter

    cond_key, time_key = gk[0], gk[1]
    conds = PAL.natural_order(props[cond_key].astype(str).unique())
    tps = PAL.order_timepoints(props[time_key].astype(str).unique())
    pal = PAL.get_from_adata(adata, label_key) or PAL.celltype_palette(labels)
    title_col = PAL.condition_palette(conds)
    stars = {(r["label"], r["stratum"]): r.get("stars", "") for r in stats_rows}

    with style("standard"):
        fig, axes = plt.subplots(len(conds), 1, figsize=(6.4, 4.4 * len(conds)),
                                 sharey=True, squeeze=False)
        for ax, cd in zip(axes.flatten(), conds):
            sub = props[props[cond_key].astype(str) == cd]
            for lab in labels:
                g = sub.groupby(sub[time_key].astype(str))[lab]
                mean = g.mean().reindex(tps)
                sem = g.sem().reindex(tps).fillna(0)
                xs = np.arange(len(tps))
                ax.errorbar(xs, mean.to_numpy(), yerr=sem.to_numpy(), fmt="-o",
                            color=pal.get(lab, PAL.UNKNOWN), lw=2.6, ms=8,
                            markeredgecolor="white", markeredgewidth=1.0, capsize=4,
                            capthick=1.8, elinewidth=1.8, label=lab, zorder=3)
                for i, tp in enumerate(tps):
                    txt = stars.get((lab, tp), "")
                    m, s = mean.get(tp, np.nan), sem.get(tp, 0)
                    if txt and np.isfinite(m):
                        ax.annotate(txt, (i, m + s), xytext=(0, 7),
                                    textcoords="offset points", ha="center", va="bottom",
                                    color=pal.get(lab, PAL.UNKNOWN), fontsize=15,
                                    fontweight="bold")
            ax.set_title(cd, fontsize=22, fontweight="bold",
                         color=title_col.get(cd, "black"), pad=8)
            ax.set_xticks(range(len(tps)))
            ax.set_xticklabels(tps, fontsize=17, fontweight="bold")
            ax.yaxis.set_major_formatter(PercentFormatter(1.0))
            ax.set_ylabel("Proportion of cells", fontsize=17, fontweight="bold")
            ax.grid(axis="y", alpha=0.25)
        axes.flatten()[-1].legend(frameon=False, fontsize=11, loc="center left",
                                  bbox_to_anchor=(1.02, 0.5), title=label_key)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("abundance") / f"proportion_lines_{label_key}")
        plt.close(fig)
    aid = ctx.add_artifact("figure", paths["pdf"],
                           caption=f"{label_key} proportions over {time_key} by {cond_key}")
    return {"kind": "proportion_lines", "artifact_id": aid, "paths": paths}


@tool("skin.abundance.sccoda", category="abundance",
      summary="Bayesian compositional DA that respects the sum-to-one constraint.")
def sccoda(dataset_id: str, label_key: str, condition_key: str, sample_key: str = "Sample",
           reference_cell_type: str = "automatic", fdr: float = 0.05,
           project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Compositional differential abundance with scCODA.

    Unlike a per-label proportion test, this models the sum-to-one constraint, so
    it will not report every other population as "decreased" when one expands.

    Args:
        dataset_id: Handle or label.
        label_key: obs column of cell type labels.
        condition_key: obs column of the condition.
        sample_key: obs column of biological replicates.
        reference_cell_type: A label assumed unchanged, or "automatic".
        fdr: Target FDR.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    adata = registry.load(ctx.project_id, dataset_id)
    for kk in (label_key, condition_key, sample_key):
        require_obs(adata, kk)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.code = ("import sccoda.util.cell_composition_data as dat\n"
                "from sccoda.util import comp_ana as mod\n"
                f"data = dat.from_pandas(counts, covariate_columns=[{condition_key!r}])\n"
                f"m = mod.CompositionalAnalysis(data, formula={condition_key!r},\n"
                f"                              reference_cell_type={reference_cell_type!r})\n"
                "res = m.sample_hmc()\n")
    if dry_run:
        ctx.summary = {"backend": "sccoda", "reference_cell_type": reference_cell_type}
        return
    try:
        import sccoda.util.cell_composition_data as dat
        from sccoda.util import comp_ana as mod
    except ImportError as e:
        raise DependencyMissing(
            "sccoda is not installed",
            remedy=("uv pip install sccoda. Meanwhile skin.abundance.proportions gives the "
                    "descriptive table and skin.abundance.milo_py gives neighbourhood-level "
                    "testing — neither models compositionality, and both say so."),
            suggested_tool="skin.abundance.proportions") from e

    o = adata.obs
    counts = (o.groupby([sample_key, condition_key, label_key], observed=True)
              .size().unstack(label_key, fill_value=0).reset_index())
    data = dat.from_pandas(counts, covariate_columns=[sample_key, condition_key])
    m = mod.CompositionalAnalysis(data, formula=condition_key,
                                  reference_cell_type=reference_cell_type)
    res = m.sample_hmc()
    res.set_fdr(est_fdr=fdr)
    eff = res.credible_effects()
    sig = [str(i) for i, v in eff.items() if v]

    run_id = store.new_id("da", 8)
    store.record_run(ctx.project_id, run_id, "abundance",
                     store.resolve_dataset_ref(ctx.project_id, dataset_id) or dataset_id,
                     {"method": "sccoda", "label_key": label_key, "fdr": fdr},
                     {"credible": sig})
    ctx.summary = {"run_id": run_id, "method": "sccoda", "fdr": fdr,
                   "credible_effects": sig[:25], "n_credible": len(sig),
                   "reference_cell_type": reference_cell_type}
    ctx.suggest("skin.abundance.proportions", "skin.memory.record_decision")


@tool("skin.abundance.milo_r", category="abundance", needs_r=True,
      summary="The real miloR with spatial FDR, via the R bridge.")
def milo_r(dataset_id: str, label_key: str, condition_key: str, contrast: list[str],
           sample_key: str = "Sample", covariates: list[str] | None = None, k: int = 30,
           prop: float = 0.1, use_rep: str = "X_pca_harmony", project_id: str = "",
           dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Run miloR in the pinned R container, with proper spatial FDR.

    Args:
        dataset_id: Handle or label.
        label_key: obs column for neighbourhood annotation.
        condition_key: obs column of the condition.
        contrast: [test, reference].
        sample_key: obs column of biological replicates.
        covariates: Blocking factors.
        k: kNN size.
        prop: Neighbourhood sampling proportion.
        use_rep: obsm key for the graph.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    adata = registry.load(ctx.project_id, dataset_id)
    for kk in (label_key, condition_key, sample_key):
        require_obs(adata, kk)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if dry_run:
        ctx.summary = {"backend": "miloR (R)", "contrast": contrast}
        return
    res = run_r_script("abundance_milo", adata=adata, project_id=ctx.project_id,
                       params={"label_key": label_key, "condition_key": condition_key,
                               "contrast": list(contrast), "sample_key": sample_key,
                               "covariates": covariates or ["Timepoint"], "k": k,
                               "prop": prop, "use_rep": use_rep, "seed": seed},
                       python_fallback="skin.abundance.milo_py")
    run_id = store.new_id("da", 8)
    store.record_run(ctx.project_id, run_id, "abundance",
                     store.resolve_dataset_ref(ctx.project_id, dataset_id) or dataset_id,
                     {"method": "milo_r", "contrast": list(contrast)}, res)
    ctx.summary = {"run_id": run_id, "method": "milo_r", "per_label": res.get("per_label", []),
                   "n_nhoods": res.get("n_nhoods"),
                   "r_log_tail": (res.get("log") or "")[-400:]}
    ctx.suggest("skin.abundance.milo_py", "skin.memory.record_decision")
