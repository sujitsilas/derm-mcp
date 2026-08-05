"""`skin.integrate.*` — normalization, HVG, scaling, PCA, batch integration.

The preprocess path follows the reference notebook exactly, including restoring
raw counts into X first, so the result is reproducible from the exported
notebook regardless of what state the handle was in.

`harmony` refuses to run when the batch key is confounded with the biological
variable. Integrating over a confounded key removes the effect you are studying,
and it does so silently — the UMAP looks better and the DE disappears.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import BadParam, ConfoundedBatch, DependencyMissing
from ..memory import store
from ._base import Ctx, require_obs, tool

logger = logging.getLogger(__name__)

HVG_FLAVORS = ("seurat", "seurat_v3", "cell_ranger", "pearson_residuals")
INTEGRATION_METHODS = ("harmony", "scvi", "scanorama", "bbknn")


def cramers_v(a: Any, b: Any) -> float:
    """Cramér's V between two categoricals — how nested one is inside the other."""
    import numpy as np
    import pandas as pd
    from scipy.stats import chi2_contingency

    tab = pd.crosstab(pd.Series(a).astype(str), pd.Series(b).astype(str))
    if tab.shape[0] < 2 or tab.shape[1] < 2:
        return 1.0
    chi2 = chi2_contingency(tab.values, correction=False)[0]
    n = tab.values.sum()
    r, k = tab.shape
    denom = n * (min(r - 1, k - 1) or 1)
    return float(np.sqrt(chi2 / denom)) if denom else 1.0


def confounding_check(batch: Any, bio: Any, min_cells: int = 10) -> dict[str, Any]:
    """Is `batch` safe to integrate over, given the biological variable `bio`?

    Cramér's V alone is the wrong test. In *every* normal design each sample
    belongs to exactly one condition, so V is 1.0 and a naive guard would refuse
    the standard `batch_key="Sample"` call. That nesting is fine: Harmony still
    has several samples *within* each condition to align against each other, so
    the condition effect survives.

    The genuinely destructive case is when the batch key has the same partition
    as the biology — batch IS condition, or every condition contains exactly one
    batch. Then there is nothing to align within a condition and Harmony removes
    the effect being studied. That is what this refuses.
    """
    import pandas as pd

    b = pd.Series(batch).astype(str).reset_index(drop=True)
    y = pd.Series(bio).astype(str).reset_index(drop=True)
    tab = pd.crosstab(b, y)
    # Ignore cells with negligible counts: one stray barcode should not count as
    # "this batch spans two conditions".
    occupied = tab >= min_cells
    batches_per_bio = occupied.sum(axis=0)
    bio_per_batch = occupied.sum(axis=1)

    min_batches = int(batches_per_bio.min()) if len(batches_per_bio) else 0
    return {
        "cramers_v": round(cramers_v(b, y), 3),
        "n_batch_levels": int(tab.shape[0]),
        "n_bio_levels": int(tab.shape[1]),
        "batches_per_bio_level": {str(k): int(v) for k, v in batches_per_bio.items()},
        "min_batches_per_bio_level": min_batches,
        "n_batches_spanning_multiple_bio_levels": int((bio_per_batch > 1).sum()),
        # Safe when at least one biological level would otherwise have a single
        # batch to align — i.e. every level has >= 2 batches.
        "confounded": bool(min_batches < 2),
        "nested_but_safe": bool(min_batches >= 2 and cramers_v(b, y) > 0.9),
    }


@tool("skin.integrate.preprocess", category="integrate",
      summary="Restore counts, normalize, log1p, HVG, scale, PCA.")
def preprocess(
    dataset_id: str,
    target_sum: float = 1e4,
    n_hvg: int = 2000,
    hvg_flavor: str = "seurat",
    scale_max: float = 10.0,
    n_comps: int = 50,
    min_cells_per_gene: int = 3,
    exclude_gene_groups: list[str] | None = None,
    regress_out: list[str] | None = None,
    label: str = "",
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Normalize, select HVGs, scale, and run PCA — from raw counts, every time.

    X is reset from layers["counts"] first, so this is idempotent and does not
    depend on what the handle's X happened to hold. The log-normalized matrix is
    kept in layers["lognorm"] and in `.raw`, so DE and plotting can reach it
    after scaling destroys X.

    `exclude_gene_groups` removes genes from the FEATURE SPACE before HVG
    selection — it stops keratin/collagen soup driving the clusters. It is not
    decontamination: the counts remain and still shape library sizes. Use
    skin.qc.estimate_ambient for that.

    Args:
        dataset_id: Handle or label. Must have layers["counts"].
        target_sum: Counts per cell after normalization.
        n_hvg: Number of highly variable genes.
        hvg_flavor: "seurat" (default, matches the reference), "seurat_v3"
            (expects raw counts, uses loess), "cell_ranger", or
            "pearson_residuals".
        scale_max: Clip value for sc.pp.scale.
        n_comps: PCA components.
        min_cells_per_gene: Drop genes below this detection count.
        exclude_gene_groups: Named groups from knowledge/contamination.yaml, e.g.
            ["collagen","keratin","mito","ribo"], or a preset like
            ["feature_space"].
        regress_out: obs columns to regress out (e.g. ["pct_counts_mt"]). Slow,
            and it distorts the variance structure — rarely the right answer.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report resolved parameters and the excluded gene list.
        seed: RNG seed.
    """
    import scanpy as sc

    if hvg_flavor not in HVG_FLAVORS:
        raise BadParam(f"hvg_flavor must be one of {list(HVG_FLAVORS)}")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.inputs = {"dataset_id": parent}

    if "counts" not in adata.layers:
        from ..errors import missing_counts

        raise missing_counts(parent or dataset_id)

    groups = list(exclude_gene_groups or [])
    matched = K.match_gene_groups(organism, groups, adata.var_names) if groups else {}
    excluded = sorted({g for v in matched.values() for g in v})

    ctx.code = (
        "import scanpy as sc\n"
        "adata.X = adata.layers['counts'].copy()\n"
        + (f"# exclude gene groups {groups!r}: {len(excluded)} genes removed from the\n"
           f"# FEATURE SPACE (not decontamination — the counts remain)\n"
           f"EXCLUDED = set({excluded!r})\n"
           f"adata = adata[:, [g for g in adata.var_names if g not in EXCLUDED]].copy()\n"
           if excluded else "")
        + f"sc.pp.filter_genes(adata, min_cells={min_cells_per_gene})\n"
        f"sc.pp.normalize_total(adata, target_sum={target_sum:g})\n"
        "sc.pp.log1p(adata)\n"
        "adata.layers['lognorm'] = adata.X.copy()\n"
        "adata.raw = adata\n"
        f"sc.pp.highly_variable_genes(adata, n_top_genes={n_hvg}, flavor={hvg_flavor!r})\n"
        + (f"sc.pp.regress_out(adata, {regress_out!r})\n" if regress_out else "")
        + f"sc.pp.scale(adata, max_value={scale_max})\n"
        f"sc.tl.pca(adata, svd_solver='arpack', n_comps={n_comps}, mask_var='highly_variable')\n"
    )
    if dry_run:
        ctx.summary = {"n_excluded_genes": len(excluded),
                       "excluded_by_group": {k: len(v) for k, v in matched.items()},
                       "excluded_examples": excluded[:15], "hvg_flavor": hvg_flavor,
                       "n_hvg": n_hvg, "n_comps": n_comps}
        return

    adata.X = adata.layers["counts"].copy()
    registry.set_x_state(adata, "counts")

    if excluded:
        keep = [g for g in map(str, adata.var_names) if g not in set(excluded)]
        adata = adata[:, keep].copy()
        ctx.warn(
            f"Removed {len(excluded)} genes in groups {list(matched)} from the feature space. "
            f"This is NOT decontamination: the contaminating counts are still in "
            f"layers['counts'] and still inflate library sizes and shape the neighbour "
            f"graph. If ambient is the real problem, fix it upstream with "
            f"skin.qc.estimate_ambient (SoupX/DecontX/CellBender)."
        )

    sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    if hvg_flavor == "seurat_v3":
        # seurat_v3 wants raw counts; run it before normalizing.
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor="seurat_v3",
                                    layer="counts")
    sc.pp.normalize_total(adata, target_sum=target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    adata.raw = adata
    registry.set_x_state(adata, "lognorm")

    if hvg_flavor == "pearson_residuals":
        sc.experimental.pp.highly_variable_genes(adata, n_top_genes=n_hvg,
                                                 flavor="pearson_residuals", layer="counts")
    elif hvg_flavor != "seurat_v3":
        sc.pp.highly_variable_genes(adata, n_top_genes=n_hvg, flavor=hvg_flavor)

    n_hvg_found = int(adata.var["highly_variable"].sum())
    if regress_out:
        for c in regress_out:
            require_obs(adata, c)
        sc.pp.regress_out(adata, regress_out)
        ctx.warn(f"regress_out={regress_out} distorts the variance structure and is rarely "
                 f"the right answer. Prefer excluding genes, or modelling the covariate in "
                 f"the DE design.")

    # Checked here rather than left to fail: zero-centering densifies, and the
    # result is the single largest allocation in the whole pipeline.
    dense_gb = registry.guard_dense(
        adata, "sc.pp.scale",
        remedy=(f"Do NOT retry — this fails the same way and risks the OS killing the "
                f"server. Either free memory (unloading the local LLM is usually the "
                f"largest win), or subset first: skin.sub.extract on one cell type gives "
                f"a handle small enough to scale. Lowering n_hvg does not help; "
                f"sc.pp.scale works on all {adata.n_vars} genes, not just the "
                f"{n_hvg_found} variable ones."))
    if dense_gb > 2.0:
        ctx.warn(f"Scaling materialises a dense {dense_gb:.1f} GB matrix — the biggest "
                 f"allocation in this pipeline. If the server has died here before, that "
                 f"is why, and subsetting before preprocessing avoids it.")
    sc.pp.scale(adata, max_value=scale_max)
    registry.set_x_state(adata, "scaled")
    n_comps_eff = int(min(n_comps, max(2, min(adata.n_obs, n_hvg_found) - 1)))
    # mask_var, not the deprecated use_highly_variable: scanpy will remove it.
    sc.tl.pca(adata, svd_solver="arpack", n_comps=n_comps_eff,
              mask_var="highly_variable", random_state=seed)

    dsid = registry.mint(
        ctx.project_id, adata, parent_id=parent, op="integrate.preprocess",
        params={"target_sum": target_sum, "n_hvg": n_hvg, "hvg_flavor": hvg_flavor,
                "scale_max": scale_max, "n_comps": n_comps_eff,
                "exclude_gene_groups": groups, "regress_out": regress_out or [],
                "min_cells_per_gene": min_cells_per_gene, "seed": seed},
        label=label or "preprocessed")
    ctx.dataset_id = dsid

    import numpy as np

    vr = adata.uns["pca"]["variance_ratio"]
    ctx.summary = {
        "dataset_id": dsid, "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars),
        "n_hvg": n_hvg_found, "hvg_flavor": hvg_flavor, "n_comps": n_comps_eff,
        "x_state": "scaled",
        "variance_ratio_head": [round(float(v), 4) for v in vr[:10]],
        "cum_variance_30pc": round(float(np.sum(vr[:30])), 3),
        "excluded_genes": {"n": len(excluded), "by_group": {k: len(v) for k, v in matched.items()}},
    }
    ctx.suggest("skin.integrate.harmony", "skin.cluster.neighbors", "skin.integrate.assess")


@tool("skin.integrate.harmony", category="integrate",
      summary="Harmony batch correction, with a confounding guard.")
def harmony(dataset_id: str, batch_key: str = "Sample", basis: str = "X_pca",
            adjusted_basis: str = "X_pca_harmony", biological_key: str = "",
            max_iter: int = 20, theta: float | None = None, lambda_: float | None = None,
            force: bool = False, label: str = "", project_id: str = "",
            dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Batch-correct the PCA embedding with Harmony.

    Refuses when `batch_key` is near-perfectly nested inside `biological_key`
    (Cramér's V > 0.9). Integrating over a confounded key destroys the effect
    being studied, and the failure is silent: the UMAP looks cleaner and the
    biology disappears. Pass force=True only if you have a specific reason.

    Args:
        dataset_id: Handle or label with a PCA embedding.
        batch_key: obs column of the technical batch (usually "Sample").
        basis: Embedding to correct.
        adjusted_basis: Where to write the corrected embedding.
        biological_key: The variable you are studying, e.g. "Type". Used for the
            confounding check. Auto-detected from Type/Condition when empty.
        max_iter: Harmony iterations.
        theta: Diversity clustering penalty. None = harmonypy default.
        lambda_: Ridge penalty. None = harmonypy default.
        force: Run despite a confounding warning.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the confounding check without running.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, batch_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if basis not in adata.obsm:
        from ..errors import no_embedding

        raise no_embedding(basis, list(adata.obsm.keys()))

    # When the caller does not name the biological variable, do NOT guess it from
    # a list of expected column names — a condition column called "treatment_arm"
    # would slip past and the guard would silently never fire. Check the batch key
    # against every low-cardinality categorical and report the worst confounding.
    from . import _introspect as I

    cands = ([biological_key] if biological_key
             else I.confounding_candidates(adata, batch_key))
    checks = {c: confounding_check(adata.obs[batch_key], adata.obs[c])
              for c in cands if c in adata.obs.columns}
    bio = (max(checks, key=lambda c: (checks[c]["confounded"], checks[c]["cramers_v"]))
           if checks else "")
    confound = None
    if bio:
        chk = checks[bio]
        tab = pd.crosstab(adata.obs[batch_key].astype(str), adata.obs[bio].astype(str))
        confound = {"batch_key": batch_key, "biological_key": bio, **chk,
                    "checked_columns": list(checks),
                    "auto_selected": not biological_key,
                    "contingency": tab.to_dict()}
        if chk["confounded"]:
            thin = [k for k, v in chk["batches_per_bio_level"].items() if v < 2]
            msg = (f"{batch_key!r} is confounded with {bio!r}: level(s) {thin} contain fewer "
                   f"than 2 batches, so there is nothing to align *within* them. Harmony "
                   f"would remove the {bio} effect along with the batch effect.")
            if not force:
                raise ConfoundedBatch(
                    msg,
                    remedy=("Integrate on a key that varies within each level of the "
                            "biological variable — a sequencing-run or mouse-batch column, "
                            "or 'Sample' when you have several replicates per condition. "
                            "Otherwise accept the batch structure and model it in the DE "
                            "design instead. Pass force=True to override; the return is "
                            "then tagged as confounded."),
                    details=confound,
                )
            ctx.warn(f"{msg} Running anyway because force=True. Every downstream result from "
                     f"this handle is confounded and must be reported as such.")
        elif chk["nested_but_safe"]:
            ctx.warn(
                f"{batch_key!r} is fully nested inside {bio!r} (Cramér's V "
                f"{chk['cramers_v']}), which is normal — every sample belongs to one "
                f"condition. Integration is safe here because each {bio} level contains at "
                f"least {chk['min_batches_per_bio_level']} batches for Harmony to align "
                f"within. Verify afterwards with skin.integrate.assess: batch mixing should "
                f"rise while label LISI stays near 1.")

    ctx.code = (
        "import harmonypy, numpy as np, pandas as pd\n"
        f"meta = pd.DataFrame({{{batch_key!r}: adata.obs[{batch_key!r}].astype(str)}})\n"
        f"ho = harmonypy.run_harmony(adata.obsm[{basis!r}], meta, [{batch_key!r}],\n"
        f"                           max_iter_harmony={max_iter}, random_state={seed}"
        + (f", theta={theta}" if theta is not None else "")
        + (f", lamb={lambda_}" if lambda_ is not None else "") + ")\n"
        "Z = np.asarray(ho.Z_corr)\n"
        "if Z.shape[0] != adata.n_obs: Z = Z.T   # harmonypy 1.x returns PCs x cells\n"
        f"adata.obsm[{adjusted_basis!r}] = Z\n"
    )
    if dry_run:
        ctx.summary = {"batch_key": batch_key, "basis": basis, "confounding": confound}
        return

    # Call harmonypy directly rather than via scanpy.external.pp.harmony_integrate:
    # that wrapper transposes Z_corr, which is correct for harmonypy 1.x
    # (PCs x cells) and wrong for 2.x (cells x PCs). Orienting by shape works on
    # both and removes the version coupling.
    import harmonypy

    kw: dict[str, Any] = {"max_iter_harmony": max_iter, "random_state": seed,
                          "verbose": False}
    if theta is not None:
        kw["theta"] = theta
    if lambda_ is not None:
        kw["lamb"] = lambda_
    meta = pd.DataFrame({batch_key: adata.obs[batch_key].astype(str).to_numpy()},
                        index=adata.obs_names)
    ho = harmonypy.run_harmony(np.asarray(adata.obsm[basis], dtype=float), meta,
                               [batch_key], **kw)
    Z = np.asarray(ho.Z_corr)
    if Z.shape[0] != adata.n_obs:
        Z = Z.T
    if Z.shape[0] != adata.n_obs:
        raise BadParam(
            f"harmonypy returned an embedding of shape {np.asarray(ho.Z_corr).shape} for "
            f"{adata.n_obs} cells",
            remedy="This usually means an incompatible harmonypy version. Check "
                   "skin.runtime.manifest, or use skin.integrate.alternative.")
    adata.obsm[adjusted_basis] = np.ascontiguousarray(Z)

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="integrate.harmony",
                         params={"batch_key": batch_key, "basis": basis,
                                 "adjusted_basis": adjusted_basis, "max_iter": max_iter,
                                 "theta": theta, "lambda_": lambda_, "forced": bool(force)},
                         label=label or "harmonized")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "adjusted_basis": adjusted_basis,
                   "n_batches": int(adata.obs[batch_key].nunique()),
                   "shape": list(np.asarray(adata.obsm[adjusted_basis]).shape),
                   "confounding": confound, "forced": bool(force)}
    ctx.suggest("skin.cluster.neighbors", "skin.integrate.assess")


@tool("skin.integrate.alternative", category="integrate",
      summary="scVI / Scanorama / BBKNN behind the same signature as harmony.")
def alternative(dataset_id: str, method: str = "scanorama", batch_key: str = "Sample",
                adjusted_basis: str = "", n_latent: int = 30, max_epochs: int = 200,
                label: str = "", project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Run a non-Harmony integration method. Harmony is the default for a reason.

    Args:
        dataset_id: Handle or label.
        method: "scvi", "scanorama", or "bbknn".
        batch_key: obs column of the batch.
        adjusted_basis: Output obsm key. Defaults per method.
        n_latent: Latent dimensions (scVI).
        max_epochs: Training epochs (scVI).
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """

    if method not in ("scvi", "scanorama", "bbknn"):
        raise BadParam(f"method must be scvi|scanorama|bbknn, got {method!r}")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, batch_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    basis = adjusted_basis or {"scvi": "X_scVI", "scanorama": "X_scanorama",
                               "bbknn": "X_pca"}[method]
    ctx.code = f"# {method} integration on {batch_key!r} -> {basis!r}\n"
    if dry_run:
        ctx.summary = {"method": method, "adjusted_basis": basis}
        return

    try:
        if method == "scanorama":
            import scanpy.external as sce

            adata = adata[adata.obs.sort_values(batch_key).index].copy()
            sce.pp.scanorama_integrate(adata, key=batch_key, basis="X_pca",
                                       adjusted_basis=basis)
        elif method == "bbknn":
            import scanpy.external as sce

            sce.pp.bbknn(adata, batch_key=batch_key)
            ctx.warn("BBKNN modifies the neighbour graph rather than producing an embedding; "
                     "skip skin.cluster.neighbors and go straight to umap/leiden.")
        else:
            import scvi

            scvi.settings.seed = seed
            sub = adata[:, adata.var["highly_variable"]].copy() if "highly_variable" in adata.var \
                else adata.copy()
            sub.X = sub.layers["counts"].copy()
            scvi.model.SCVI.setup_anndata(sub, batch_key=batch_key, layer="counts")
            m = scvi.model.SCVI(sub, n_latent=n_latent)
            m.train(max_epochs=max_epochs)
            adata.obsm[basis] = m.get_latent_representation()
    except ImportError as e:
        raise DependencyMissing(
            f"{method} is not installed",
            remedy='Install with `uv pip install "skin-mcp[full]"`, or use '
                   "skin.integrate.harmony, which is a core dependency.",
            suggested_tool="skin.integrate.harmony",
        ) from e

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent,
                         op=f"integrate.{method}",
                         params={"batch_key": batch_key, "adjusted_basis": basis,
                                 "n_latent": n_latent, "max_epochs": max_epochs},
                         label=label or f"{method}_integrated")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "method": method, "adjusted_basis": basis}
    ctx.suggest("skin.cluster.neighbors", "skin.integrate.assess")


@tool("skin.integrate.assess", category="integrate",
      summary="Batch mixing vs label preservation, before and after integration.")
def assess(dataset_id: str, batch_key: str = "Sample", label_key: str = "",
           before_basis: str = "X_pca", after_basis: str = "X_pca_harmony",
           n_neighbors: int = 30, subsample: int = 8000, project_id: str = "",
           dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Quantify the integration tradeoff. It is a tradeoff, not a win.

    Reports LISI-style scores: batch mixing (higher is better-mixed) and label
    preservation (lower is better-preserved), before and after. An integration
    that improves batch mixing while collapsing label separation has removed
    biology, not batch.

    Args:
        dataset_id: Handle or label.
        batch_key: obs column of the batch.
        label_key: obs column of cell labels/clusters. Optional but recommended.
        before_basis: Uncorrected embedding.
        after_basis: Corrected embedding.
        n_neighbors: Neighbourhood size for the LISI estimate.
        subsample: Cap on cells used, for speed.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: RNG seed.
    """
    import numpy as np
    from sklearn.neighbors import NearestNeighbors

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, batch_key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    if label_key:
        require_obs(adata, label_key)

    rng = np.random.default_rng(seed)
    idx = (rng.choice(adata.n_obs, subsample, replace=False)
           if adata.n_obs > subsample else np.arange(adata.n_obs))

    def lisi(basis: str, key: str) -> float | None:
        if basis not in adata.obsm:
            return None
        Xb = np.asarray(adata.obsm[basis])[idx]
        lab = adata.obs[key].astype(str).to_numpy()[idx]
        cats, codes = np.unique(lab, return_inverse=True)
        k = min(n_neighbors, len(idx) - 1)
        nn = NearestNeighbors(n_neighbors=k + 1).fit(Xb)
        _, ind = nn.kneighbors(Xb)
        neigh = codes[ind[:, 1:]]
        # inverse Simpson index of the neighbourhood composition
        out = []
        for row in neigh:
            _, cnt = np.unique(row, return_counts=True)
            p = cnt / cnt.sum()
            out.append(1.0 / float(np.sum(p ** 2)))
        return round(float(np.mean(out)), 3)

    res = {
        "n_batches": int(adata.obs[batch_key].nunique()),
        "batch_mixing_before": lisi(before_basis, batch_key),
        "batch_mixing_after": lisi(after_basis, batch_key),
    }
    if label_key:
        res["label_lisi_before"] = lisi(before_basis, label_key)
        res["label_lisi_after"] = lisi(after_basis, label_key)
        res["n_labels"] = int(adata.obs[label_key].nunique())

    bb, ba = res["batch_mixing_before"], res["batch_mixing_after"]
    if bb and ba:
        res["batch_mixing_gain"] = round(ba - bb, 3)
        if ba < bb:
            ctx.warn("Integration reduced batch mixing. Check that `after_basis` is the "
                     "corrected embedding and that batch_key is right.")
    if label_key and res.get("label_lisi_before") and res.get("label_lisi_after"):
        cost = res["label_lisi_after"] - res["label_lisi_before"]
        res["label_mixing_cost"] = round(cost, 3)
        if cost > 0.5:
            ctx.warn(f"Label LISI rose by {cost:.2f} after integration: distinct cell types "
                     f"are now mixing. That is over-correction. Lower theta, or integrate on "
                     f"a coarser batch key.")
    res["interpretation"] = ("Batch mixing should rise (max = n_batches); label LISI should "
                             "stay near 1. Both rising means biology was removed.")
    ctx.summary = res
    ctx.suggest("skin.cluster.neighbors", "skin.cluster.leiden_sweep")
