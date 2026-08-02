"""`skin.de.*` — differential expression.

Pseudobulk is the default and the recommendation. Cell-wise Wilcoxon is
available, but its return carries `inference_level: "cell"` and an explicit
caveat, because with n=3-4 mice per arm the p-values are pseudo-replicated
across cells within a sample and the false-positive rate is high.

The module refuses rather than silently degrading: a label with too few
replicates lands in `skipped` with its counts, not in a Wilcoxon fallback.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import BadParam, InsufficientReplicates, NotFound
from ..memory import store
from ..style.panels import slug
from ._base import Ctx, require_obs, tool

logger = logging.getLogger(__name__)


def _aggregate_pseudobulk(adata: Any, label_key: str, sample_key: str, labels: list[str],
                          min_cells: int) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Sum raw counts by sample x label. Sum, never mean."""
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    X = adata.layers["counts"]
    lab = adata.obs[label_key].astype(str).to_numpy()
    smp = adata.obs[sample_key].astype(str).to_numpy()
    rows, index, dropped = [], [], []
    for lb in labels:
        for s in sorted(set(smp)):
            m = (lab == lb) & (smp == s)
            n = int(m.sum())
            if n == 0:
                continue
            if n < min_cells:
                dropped.append({"label": lb, "sample": s, "n_cells": n,
                                "reason": f"fewer than min_cells={min_cells}"})
                continue
            sub = X[m]
            v = np.asarray(sub.sum(0)).ravel() if sp.issparse(sub) else np.asarray(sub).sum(0)
            rows.append(v)
            index.append((lb, s, n))
    if not rows:
        raise InsufficientReplicates(
            "no pseudobulk units survived the min_cells filter",
            remedy=f"Lower min_cells (currently {min_cells}) or merge labels.",
        )
    counts = pd.DataFrame(np.vstack(rows), columns=list(map(str, adata.var_names)))
    meta = pd.DataFrame(index, columns=["label", "sample", "n_cells"])
    counts.index = meta.index = [f"{lb}__{s}" for lb, s, _ in index]
    return counts, meta, dropped


def _sample_metadata(adata: Any, sample_key: str, cols: list[str]) -> Any:
    """One row per sample for the design matrix."""
    import pandas as pd

    o = adata.obs
    out = {}
    for s in sorted(set(o[sample_key].astype(str))):
        m = o[sample_key].astype(str) == s
        out[s] = {c: str(o.loc[m, c].astype(str).mode().iloc[0]) for c in cols if c in o}
    return pd.DataFrame(out).T


@tool("skin.de.pseudobulk", category="de",
      summary="Pseudobulk DE with PyDESeq2. The default and the recommendation.")
def pseudobulk(
    dataset_id: str,
    label_key: str,
    condition_key: str,
    contrast: list[str],
    groups: list[str] | None = None,
    sample_key: str = "Sample",
    covariates: list[str] | None = None,
    min_cells: int = 10,
    min_samples_per_arm: int = 3,
    exclude_gene_groups: list[str] | None = None,
    min_count_sum: int = 10,
    shrink_lfc: bool = True,
    project_id: str = "",
    dry_run: bool = False,
    seed: int = 0,
    *,
    ctx: Ctx,
) -> None:
    """Aggregate raw counts by sample x label and test with PyDESeq2.

    n is the number of SAMPLES, not cells. Labels that lack
    `min_samples_per_arm` replicates in either arm are returned in `skipped`
    with their counts — this tool does not silently fall back to a cell-wise
    test. Opt into skin.de.wilcoxon deliberately if that is what you want.

    Covariate levels that lack both contrast arms (e.g. a D19 timepoint with no
    Sham for neutrophils) are dropped and reported in `dropped_levels`.

    Args:
        dataset_id: Handle or label. Must have layers["counts"].
        label_key: obs column of cell type labels. DE runs per label.
        condition_key: obs column holding the contrast variable, e.g. "Type".
        contrast: [test, reference], e.g. ["Burn","Sham"]. Positive LFC = up in test.
        groups: Labels to test. Empty = all of them.
        sample_key: obs column identifying biological replicates.
        covariates: Blocking factors, e.g. ["Timepoint"]. Design is
            `~ {covariates} + {condition_key}`.
        min_cells: Minimum cells per pseudobulk unit.
        min_samples_per_arm: Minimum replicates per arm. Below this, the label is skipped.
        exclude_gene_groups: Gene groups removed BEFORE size-factor estimation.
            Defaults to ["immune_de"] (collagen, keratin, muscle, ecm_misc,
            stress) — the reference notebook's set. Pass [] to keep everything.
        min_count_sum: Drop genes whose total pseudobulk count is below this.
        shrink_lfc: Apply apeglm/ashr shrinkage. Both shrunk and unshrunk LFC are reported.
        project_id: Defaults to the active project.
        dry_run: Report the design, the units, and what would be skipped.
        seed: RNG seed.
    """
    import pandas as pd

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, label_key)
    require_obs(adata, condition_key)
    require_obs(adata, sample_key)
    organism = registry.get_organism(adata)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.inputs = {"dataset_id": resolved}
    ctx.dataset_id = resolved

    if "counts" not in adata.layers:
        from ..errors import missing_counts

        raise missing_counts(resolved or dataset_id)
    if len(contrast) != 2:
        raise BadParam("contrast must be [test, reference]",
                       remedy='e.g. ["Burn", "Sham"] — positive LFC means up in "Burn".')
    a, b = str(contrast[0]), str(contrast[1])
    levels = set(map(str, adata.obs[condition_key].astype(str).unique()))
    missing = [x for x in (a, b) if x not in levels]
    if missing:
        raise NotFound(f"contrast levels not present: {missing}",
                       remedy=f"Levels in {condition_key!r}: {sorted(levels)}")

    covs = list(covariates) if covariates is not None else ["Timepoint"]
    covs = [c for c in covs if c in adata.obs.columns]
    labels = [str(x) for x in (groups or adata.obs[label_key].astype(str).unique())]
    grp_excl = list(exclude_gene_groups) if exclude_gene_groups is not None else ["immune_de"]
    matched = K.match_gene_groups(organism, grp_excl, adata.var_names) if grp_excl else {}
    excluded = sorted({g for v in matched.values() for g in v})
    design = "~ " + " + ".join([*(f"{c}" for c in covs), condition_key])

    ctx.code = (
        "import numpy as np, pandas as pd, scipy.sparse as sp\n"
        "from pydeseq2.dds import DeseqDataSet\n"
        "from pydeseq2.ds import DeseqStats\n"
        "\n"
        "# 1. sum RAW counts by sample x label (sum, never mean)\n"
        f"LABELS = {labels!r}\n"
        f"EXCLUDED = set({excluded!r})   # {len(excluded)} genes in groups {grp_excl!r},\n"
        "                               # removed BEFORE size-factor estimation\n"
        f"lab = adata.obs[{label_key!r}].astype(str).to_numpy()\n"
        f"smp = adata.obs[{sample_key!r}].astype(str).to_numpy()\n"
        "X = adata.layers['counts']\n"
        "rows, index = [], []\n"
        "for lb in LABELS:\n"
        "    for s in sorted(set(smp)):\n"
        "        m = (lab == lb) & (smp == s)\n"
        f"        if m.sum() < {min_cells}:   # drop thin pseudobulk units\n"
        "            continue\n"
        "        sub = X[m]\n"
        "        rows.append(np.asarray(sub.sum(0)).ravel() if sp.issparse(sub)\n"
        "                    else np.asarray(sub).sum(0))\n"
        "        index.append((lb, s))\n"
        "pb = pd.DataFrame(np.vstack(rows), columns=list(map(str, adata.var_names)),\n"
        "                  index=[f'{lb}__{s}' for lb, s in index])\n"
        "pb = pb.drop(columns=[g for g in EXCLUDED if g in pb.columns])\n"
        "meta = pd.DataFrame(index, columns=['label', 'sample']).set_index(pb.index)\n"
        f"for col in {[condition_key, *covs]!r}:\n"
        f"    per_sample = (adata.obs.groupby({sample_key!r}, observed=True)[col]\n"
        "                  .agg(lambda s: s.astype(str).mode().iloc[0]))\n"
        "    meta[col] = meta['sample'].map(per_sample)\n"
        "\n"
        "# 2. one DESeq2 fit per label\n"
        "for lb in LABELS:\n"
        "    sel = meta['label'] == lb\n"
        "    m, c = meta[sel].copy(), pb[sel].copy()\n"
        f"    m[{condition_key!r}] = pd.Categorical(m[{condition_key!r}], "
        f"categories=[{b!r}, {a!r}])\n"
        f"    c = c.loc[:, c.sum(0) >= {min_count_sum}]\n"
        f"    dds = DeseqDataSet(counts=c.astype(int), metadata=m, design={design!r},\n"
        "                       quiet=True)\n"
        "    dds.deseq2()\n"
        f"    res = DeseqStats(dds, contrast=[{condition_key!r}, {a!r}, {b!r}], quiet=True)\n"
        "    res.summary()\n"
        + ("    # LFC shrinkage: coeff name comes from dds.obsm['design_matrix'].columns\n"
           "    res.lfc_shrink(coeff=next(x for x in dds.obsm['design_matrix'].columns\n"
           f"                             if {condition_key!r} in x and {a!r} in x))\n"
           if shrink_lfc else "")
        + "    df = res.results_df.dropna(subset=['padj'])\n"
        "    df.index.name = 'gene'\n"
        "    df = df.reset_index().rename(columns={'log2FoldChange': 'lfc',\n"
        "                                          'pvalue': 'pval'})\n"
        # Deliberately a different filename from the server's own artifact: an
        # exported notebook must never silently overwrite the tables the figures
        # were built from.
        + f"    df.to_csv(TABLEDIR / ('de_' + _slug(lb) + "
        f"'_{slug(a)}_vs_{slug(b)}_reproduced.csv'), index=False)\n"
    )
    ctx.code = ("import re\n"
                "def _slug(s):\n"
                "    return re.sub(r'[^A-Za-z0-9]+', '_', "
                "str(s).replace('φ','phi').replace('Φ','phi')).strip('_').lower()\n"
                + ctx.code)

    counts, meta, dropped_units = _aggregate_pseudobulk(adata, label_key, sample_key,
                                                        labels, min_cells)
    smeta = _sample_metadata(adata, sample_key, [condition_key, *covs])
    meta = meta.join(smeta, on="sample")

    if excluded:
        counts = counts.drop(columns=[g for g in excluded if g in counts.columns])

    plan_rows = []
    for lb in labels:
        sel = meta["label"] == lb
        arm = meta.loc[sel, condition_key].astype(str)
        plan_rows.append({"label": lb, f"n_{a}": int((arm == a).sum()),
                          f"n_{b}": int((arm == b).sum()),
                          "n_units": int(sel.sum())})
    if dry_run:
        ctx.summary = {"design": design, "contrast": [a, b], "n_labels": len(labels),
                       "n_excluded_genes": len(excluded), "units": plan_rows,
                       "dropped_units": dropped_units[:10]}
        return

    try:
        from pydeseq2.dds import DeseqDataSet
        from pydeseq2.ds import DeseqStats
    except ImportError as e:
        from ..errors import DependencyMissing

        raise DependencyMissing("pydeseq2 is not installed",
                                remedy="uv pip install pydeseq2") from e

    per_label, skipped, dropped_levels = [], [], []
    run_id = store.new_id("de", 8)
    tables: dict[str, str] = {}

    for lb in labels:
        sel = (meta["label"] == lb).to_numpy()
        m = meta.loc[sel].copy()
        c = counts.loc[sel].copy()
        arm = m[condition_key].astype(str)
        na, nb = int((arm == a).sum()), int((arm == b).sum())
        if na < min_samples_per_arm or nb < min_samples_per_arm:
            skipped.append({"label": lb, f"n_{a}": na, f"n_{b}": nb,
                            "reason": f"fewer than min_samples_per_arm={min_samples_per_arm} "
                                      f"in at least one arm"})
            continue

        # Drop covariate levels that lack both arms.
        use_covs = []
        for cov in covs:
            lv = m[cov].astype(str)
            keep_levels = [x for x in lv.unique()
                           if {a, b} <= set(m.loc[lv == x, condition_key].astype(str))]
            dropped = [x for x in lv.unique() if x not in keep_levels]
            if dropped:
                dropped_levels.append({"label": lb, "covariate": cov, "dropped": dropped,
                                       "reason": "level lacks both contrast arms"})
                keep = lv.isin(keep_levels).to_numpy()
                m, c = m.loc[keep], c.loc[keep]
                lv = m[cov].astype(str)
            if lv.nunique() > 1:
                use_covs.append(cov)

        arm = m[condition_key].astype(str)
        na, nb = int((arm == a).sum()), int((arm == b).sum())
        if na < min_samples_per_arm or nb < min_samples_per_arm:
            skipped.append({"label": lb, f"n_{a}": na, f"n_{b}": nb,
                            "reason": "too few replicates after dropping unbalanced "
                                      "covariate levels"})
            continue

        c = c.loc[:, c.sum(0) >= min_count_sum]
        if c.shape[1] < 50:
            skipped.append({"label": lb, "reason": f"only {c.shape[1]} genes pass "
                                                   f"min_count_sum={min_count_sum}"})
            continue

        md = m[[condition_key, *use_covs]].copy()
        md[condition_key] = pd.Categorical(md[condition_key].astype(str),
                                           categories=[b, a])  # reference first
        for cov in use_covs:
            md[cov] = md[cov].astype(str).astype("category")
        d = "~ " + " + ".join([*use_covs, condition_key])

        try:
            dds = DeseqDataSet(counts=c.astype(int), metadata=md, design=d, quiet=True,
                               refit_cooks=True)
            dds.deseq2()
            ds = DeseqStats(dds, contrast=[condition_key, a, b], quiet=True)
            ds.summary()
            res = ds.results_df.copy()
            res["lfc_unshrunk"] = res["log2FoldChange"]
            if shrink_lfc:
                coeff = _find_coeff(dds, condition_key, a, b)
                if coeff:
                    try:
                        ds.lfc_shrink(coeff=coeff)
                        res["log2FoldChange"] = ds.results_df["log2FoldChange"]
                        res["shrunk"] = True
                    except Exception as e:  # noqa: BLE001 - shrinkage is optional
                        ctx.warn(f"{lb}: LFC shrinkage failed ({e}); reporting unshrunk LFC.")
                        res["shrunk"] = False
                else:
                    res["shrunk"] = False
                    ctx.warn(f"{lb}: could not resolve the shrinkage coefficient; "
                             f"reporting unshrunk LFC.")
        except Exception as e:  # noqa: BLE001 - per-label failure must not kill the run
            skipped.append({"label": lb, "reason": f"PyDESeq2 error: {type(e).__name__}: "
                                                   f"{str(e)[:150]}"})
            continue

        # pydeseq2 names the index after the counts frame's columns axis, which is
        # not always "index"; force it so every DE table in this server has the
        # same schema (gene / lfc / padj / pval).
        res.index.name = "gene"
        res = res.reset_index().rename(columns={"log2FoldChange": "lfc", "pvalue": "pval"})
        res = res.dropna(subset=["padj"])
        p = ctx.tabledir() / f"de_{slug(lb)}_{slug(a)}_vs_{slug(b)}.csv"
        res.sort_values("padj").to_csv(p, index=False)
        aid = ctx.add_artifact("table", p, caption=f"pseudobulk DE {lb}: {a} vs {b}",
                               params={"design": d, "contrast": [a, b], "run_id": run_id})
        tables[lb] = str(p)
        per_label.append({
            "label": lb, "n_genes_tested": int(len(res)),
            "n_up": int(((res.padj < 0.05) & (res.lfc > 0.5)).sum()),
            "n_down": int(((res.padj < 0.05) & (res.lfc < -0.5)).sum()),
            f"n_samples_{a}": na, f"n_samples_{b}": nb, "design": d,
            "table_artifact_id": aid, "table_path": str(p),
        })

    if not per_label:
        raise InsufficientReplicates(
            f"no label had >= {min_samples_per_arm} samples in both arms",
            remedy=("Report n_samples, not n_cells: this design does not support "
                    "population-level inference for any label. Options: merge labels, "
                    "lower min_samples_per_arm (and say so in the methods), or run "
                    "skin.de.wilcoxon and label the result exploratory."),
            details={"skipped": skipped[:10]},
        )
    if skipped:
        ctx.warn(f"{len(skipped)} labels were skipped for insufficient replicates: "
                 f"{[s['label'] for s in skipped][:8]}. They are NOT silently downgraded to a "
                 f"cell-wise test — call skin.de.wilcoxon deliberately if you want that, and "
                 f"label it exploratory.")
    if dropped_levels:
        ctx.warn(f"{len(dropped_levels)} covariate levels were dropped for lacking both "
                 f"contrast arms: {dropped_levels[:3]}")
    if excluded:
        ctx.warn(f"{len(excluded)} genes in groups {list(matched)} were removed before size-"
                 f"factor estimation. This is a presentation guard, not decontamination — "
                 f"the counts were still in the cells that produced these pseudobulk sums.")

    store.record_run(ctx.project_id, run_id, "de", resolved or dataset_id,
                     {"label_key": label_key, "condition_key": condition_key,
                      "contrast": [a, b], "covariates": covs, "design": design,
                      "method": "pydeseq2", "exclude_gene_groups": grp_excl,
                      "min_samples_per_arm": min_samples_per_arm},
                     {"per_label": per_label, "tables": tables, "skipped": skipped,
                      "inference_level": "sample"})

    ctx.summary = {
        "run_id": run_id, "method": "pydeseq2", "inference_level": "sample",
        "design": design, "contrast": [a, b], "per_label": per_label,
        "dropped_levels": dropped_levels[:10], "skipped": skipped[:10],
        "n_excluded_genes": len(excluded),
        "dropped_units": dropped_units[:8],
    }
    ctx.suggest("skin.plot.volcano_grid", "skin.enrich.ora", "skin.de.compare_methods")


def _find_coeff(dds: Any, condition_key: str, a: str, b: str) -> str | None:
    """Resolve the design-matrix coefficient name PyDESeq2 uses for this contrast."""
    try:
        cols = list(dds.obsm["design_matrix"].columns)
    except (KeyError, AttributeError):
        return None
    for c in cols:
        if condition_key in c and a in c:
            return c
    return None


@tool("skin.de.wilcoxon", category="de",
      summary="Cell-wise Wilcoxon. EXPLORATORY — p-values are pseudo-replicated.")
def wilcoxon(dataset_id: str, label_key: str, condition_key: str, contrast: list[str],
             groups: list[str] | None = None, exclude_gene_groups: list[str] | None = None,
             min_cells: int = 3, project_id: str = "", dry_run: bool = False,
             seed: int = 0, *, ctx: Ctx) -> None:
    """Per-label cell-wise Wilcoxon test. Reproduces the reference notebook's volcanoes.

    Every return from this tool carries `inference_level: "cell"` and an explicit
    caveat: p-values are pseudo-replicated across cells within a sample and are
    not valid for population-level inference. With n=3-4 mice per arm the counts
    printed on a volcano from this test are typically several-fold higher than
    the pseudobulk counts. Use skin.de.compare_methods to quantify the shift.

    Args:
        dataset_id: Handle or label. X or layers["lognorm"] must be log-normalized.
        label_key: obs column of cell type labels.
        condition_key: obs column holding the contrast variable.
        contrast: [test, reference], e.g. ["Burn","Sham"].
        groups: Labels to test. Empty = all.
        exclude_gene_groups: Gene groups to drop before testing. Defaults to
            ["immune_de"], matching the reference notebook.
        min_cells: Minimum cells per arm within a label.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    require_obs(adata, condition_key)
    organism = registry.get_organism(adata)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    if len(contrast) != 2:
        raise BadParam("contrast must be [test, reference]")
    a, b = str(contrast[0]), str(contrast[1])

    if registry.get_x_state(adata) == "scaled":
        if "lognorm" in adata.layers:
            adata.X = adata.layers["lognorm"].copy()
            registry.set_x_state(adata, "lognorm")
        else:
            raise BadParam("X is z-scaled and there is no 'lognorm' layer",
                           remedy="Wilcoxon LFCs on scaled data are meaningless. Re-run "
                                  "skin.integrate.preprocess, which preserves lognorm.")

    grp_excl = list(exclude_gene_groups) if exclude_gene_groups is not None else ["immune_de"]
    matched = K.match_gene_groups(organism, grp_excl, adata.var_names) if grp_excl else {}
    excluded = sorted({g for v in matched.values() for g in v})
    labels = [str(x) for x in (groups or adata.obs[label_key].astype(str).unique())]

    ctx.code = (
        "import scanpy as sc\n"
        f"# EXPLORATORY: cell-wise test, p-values pseudo-replicated within samples\n"
        f"clean = [g for g in adata.var_names if g not in CONTAM]  # {len(excluded)} removed\n"
        f"for ct in {labels[:5]!r}:\n"
        f"    sub = adata[adata.obs[{label_key!r}] == ct][:, clean].copy()\n"
        f"    sc.tl.rank_genes_groups(sub, groupby={condition_key!r}, groups=[{a!r}],\n"
        f"                            reference={b!r}, method='wilcoxon', use_raw=False,\n"
        f"                            pts=True, key_added='rgg')\n"
        f"    df = sc.get.rank_genes_groups_df(sub, group={a!r}, key='rgg')\n"
    )
    if dry_run:
        ctx.summary = {"n_labels": len(labels), "n_excluded_genes": len(excluded),
                       "inference_level": "cell"}
        return

    clean = [g for g in map(str, adata.var_names) if g not in set(excluded)]
    run_id = store.new_id("de", 8)
    per_label, skipped, tables = [], [], {}

    for lb in labels:
        sel = adata.obs[label_key].astype(str) == lb
        sub = adata[sel.to_numpy()][:, clean].copy()
        arm = sub.obs[condition_key].astype(str)
        na, nb = int((arm == a).sum()), int((arm == b).sum())
        if na < min_cells or nb < min_cells:
            skipped.append({"label": lb, f"n_{a}": na, f"n_{b}": nb,
                            "reason": f"fewer than min_cells={min_cells}"})
            continue
        sc.tl.rank_genes_groups(sub, groupby=condition_key, groups=[a], reference=b,
                                method="wilcoxon", use_raw=False, pts=True, key_added="rgg")
        df = sc.get.rank_genes_groups_df(sub, group=a, key="rgg")
        df = df.dropna(subset=["logfoldchanges", "pvals_adj"]).rename(
            columns={"names": "gene", "logfoldchanges": "lfc", "pvals_adj": "padj",
                     "pvals": "pval"})
        p = ctx.tabledir() / f"de_wilcoxon_{slug(lb)}_{slug(a)}_vs_{slug(b)}.csv"
        df.to_csv(p, index=False)
        aid = ctx.add_artifact("table", p,
                               caption=f"cell-wise Wilcoxon (EXPLORATORY) {lb}: {a} vs {b}",
                               params={"inference_level": "cell", "run_id": run_id})
        tables[lb] = str(p)
        per_label.append({"label": lb, "n_cells_a": na, "n_cells_b": nb,
                          "n_genes_tested": int(len(df)),
                          "n_up": int(((df.padj < 0.05) & (df.lfc > 0.5)).sum()),
                          "n_down": int(((df.padj < 0.05) & (df.lfc < -0.5)).sum()),
                          "table_artifact_id": aid, "table_path": str(p)})

    if not per_label:
        raise BadParam("no label had enough cells in both arms",
                       details={"skipped": skipped[:10]})

    n_samples = (int(adata.obs["Sample"].nunique()) if "Sample" in adata.obs else None)
    caveat = ("p-values are pseudo-replicated across cells within a sample and are NOT valid "
              "for population-level inference. n here is cells, not biological replicates"
              + (f" (this object has {n_samples} samples)." if n_samples else ".")
              + " Use skin.de.pseudobulk for anything that goes in a paper.")
    ctx.warn("EXPLORATORY RESULT — " + caveat)

    store.record_run(ctx.project_id, run_id, "de", resolved or dataset_id,
                     {"label_key": label_key, "condition_key": condition_key,
                      "contrast": [a, b], "method": "wilcoxon",
                      "exclude_gene_groups": grp_excl},
                     {"per_label": per_label, "tables": tables, "inference_level": "cell"})

    ctx.summary = {"run_id": run_id, "method": "wilcoxon", "inference_level": "cell",
                   "caveat": caveat, "contrast": [a, b], "per_label": per_label,
                   "skipped": skipped[:8], "n_excluded_genes": len(excluded),
                   "n_samples": n_samples}
    ctx.suggest("skin.plot.volcano_grid", "skin.de.pseudobulk", "skin.de.compare_methods")


@tool("skin.de.timepoint_interaction", category="de",
      summary="Test whether the condition effect changes over time (~ Type * Timepoint).")
def timepoint_interaction(dataset_id: str, label_key: str, condition_key: str,
                          time_key: str = "Timepoint", groups: list[str] | None = None,
                          sample_key: str = "Sample", min_cells: int = 10,
                          min_samples_per_arm: int = 3,
                          exclude_gene_groups: list[str] | None = None,
                          project_id: str = "", dry_run: bool = False, seed: int = 0,
                          *, ctx: Ctx) -> None:
    """Fit `~ condition * timepoint` when the question is about the interaction.

    Use this when you want "does the burn effect change over time", not "is there
    a burn effect". Interaction terms need substantially more replicates than
    main effects; the requirement is enforced per timepoint.

    Args:
        dataset_id: Handle or label with counts.
        label_key: obs column of cell type labels.
        condition_key: obs column of the condition.
        time_key: obs column of the timepoint.
        groups: Labels to test. Empty = all.
        sample_key: obs column of biological replicates.
        min_cells: Minimum cells per pseudobulk unit.
        min_samples_per_arm: Minimum replicates per condition x timepoint cell.
        exclude_gene_groups: Gene groups removed before size factors.
        project_id: Defaults to the active project.
        dry_run: Report the design and the cell counts.
        seed: RNG seed.
    """

    adata = registry.load(ctx.project_id, dataset_id)
    for k in (label_key, condition_key, time_key, sample_key):
        require_obs(adata, k)
    organism = registry.get_organism(adata)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    if "counts" not in adata.layers:
        from ..errors import missing_counts

        raise missing_counts(resolved or dataset_id)

    labels = [str(x) for x in (groups or adata.obs[label_key].astype(str).unique())]
    design = f"~ {condition_key} + {time_key} + {condition_key}:{time_key}"
    ctx.code = (f"dds = DeseqDataSet(counts=pb_counts, metadata=pb_meta, design={design!r})\n"
                "dds.deseq2()\n"
                "# the interaction coefficients are the answer, not the main effect\n")

    counts, meta, dropped_units = _aggregate_pseudobulk(adata, label_key, sample_key,
                                                        labels, min_cells)
    smeta = _sample_metadata(adata, sample_key, [condition_key, time_key])
    meta = meta.join(smeta, on="sample")
    balance = (meta.groupby(["label", condition_key, time_key], observed=True)
               .size().rename("n_samples").reset_index())
    if dry_run:
        ctx.summary = {"design": design, "balance": balance.head(24).to_dict("records")}
        return

    grp_excl = list(exclude_gene_groups) if exclude_gene_groups is not None else ["immune_de"]
    matched = K.match_gene_groups(organism, grp_excl, adata.var_names) if grp_excl else {}
    excluded = sorted({g for v in matched.values() for g in v})
    if excluded:
        counts = counts.drop(columns=[g for g in excluded if g in counts.columns])

    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats

    run_id = store.new_id("de", 8)
    per_label, skipped = [], []
    for lb in labels:
        sel = (meta["label"] == lb).to_numpy()
        m, c = meta.loc[sel].copy(), counts.loc[sel].copy()
        cells = m.groupby([condition_key, time_key], observed=True).size()
        if cells.min() < min_samples_per_arm or len(cells) < 4:
            skipped.append({"label": lb, "reason": "interaction needs >= "
                            f"{min_samples_per_arm} samples in every condition x timepoint "
                            f"cell; smallest is {int(cells.min()) if len(cells) else 0}",
                            "cells": cells.to_dict()})
            continue
        md = m[[condition_key, time_key]].astype(str).astype("category")
        c = c.loc[:, c.sum(0) >= 10]
        try:
            dds = DeseqDataSet(counts=c.astype(int), metadata=md, design=design, quiet=True)
            dds.deseq2()
            coeffs = [x for x in dds.obsm["design_matrix"].columns if ":" in x]
            rows = []
            for coef in coeffs:
                ds = DeseqStats(dds, contrast=_coef_vector(dds, coef), quiet=True)
                ds.summary()
                r = ds.results_df.dropna(subset=["padj"])
                rows.append({"coefficient": coef,
                             "n_sig": int((r["padj"] < 0.05).sum()),
                             "top_genes": r.nsmallest(8, "padj").index.tolist()})
                p = ctx.tabledir() / f"de_interaction_{slug(lb)}_{slug(coef)}.csv"
                r.reset_index().rename(columns={"index": "gene",
                                                "log2FoldChange": "lfc"}).to_csv(p, index=False)
                ctx.add_artifact("table", p, caption=f"interaction {lb} {coef}")
            per_label.append({"label": lb, "n_interaction_terms": len(coeffs),
                              "terms": rows})
        except Exception as e:  # noqa: BLE001
            skipped.append({"label": lb, "reason": f"{type(e).__name__}: {str(e)[:140]}"})

    if not per_label:
        raise InsufficientReplicates(
            "no label supports an interaction model",
            remedy=("Interaction terms need replicates in every condition x timepoint cell. "
                    "Test each timepoint separately with skin.de.pseudobulk "
                    "(groups=..., covariates=[]) instead."),
            details={"skipped": skipped[:8]})

    store.record_run(ctx.project_id, run_id, "de", resolved or dataset_id,
                     {"design": design, "method": "pydeseq2_interaction"},
                     {"per_label": per_label})
    ctx.summary = {"run_id": run_id, "design": design, "per_label": per_label,
                   "skipped": skipped[:8], "inference_level": "sample"}
    ctx.suggest("skin.de.pseudobulk", "skin.plot.volcano_grid")


def _coef_vector(dds: Any, coef: str) -> Any:
    import numpy as np

    cols = list(dds.obsm["design_matrix"].columns)
    v = np.zeros(len(cols))
    v[cols.index(coef)] = 1.0
    return v


@tool("skin.de.pseudobulk_matrix", category="de",
      summary="Export the aggregated pseudobulk count matrix and sample metadata.")
def pseudobulk_matrix(dataset_id: str, label_key: str, sample_key: str = "Sample",
                      groups: list[str] | None = None, min_cells: int = 10,
                      metadata_keys: list[str] | None = None, project_id: str = "",
                      dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Write the sample x label pseudobulk matrix for use outside this server.

    Args:
        dataset_id: Handle or label with counts.
        label_key: obs column of cell type labels.
        sample_key: obs column of biological replicates.
        groups: Labels to include. Empty = all.
        min_cells: Minimum cells per unit.
        metadata_keys: obs columns to carry into the sample metadata table.
        project_id: Defaults to the active project.
        dry_run: Report the shape only.
        seed: RNG seed.
    """
    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, label_key)
    require_obs(adata, sample_key)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    if "counts" not in adata.layers:
        from ..errors import missing_counts

        raise missing_counts(resolved or dataset_id)
    labels = [str(x) for x in (groups or adata.obs[label_key].astype(str).unique())]
    counts, meta, dropped = _aggregate_pseudobulk(adata, label_key, sample_key, labels,
                                                  min_cells)
    keys = metadata_keys or [c for c in ("Type", "Timepoint", "Batch", "Sex", "Replicate")
                             if c in adata.obs.columns]
    meta = meta.join(_sample_metadata(adata, sample_key, keys), on="sample")
    if dry_run:
        ctx.summary = {"shape": list(counts.shape), "n_units": int(len(meta))}
        return
    pc = ctx.tabledir() / "pseudobulk_counts.csv"
    pm = ctx.tabledir() / "pseudobulk_metadata.csv"
    counts.to_csv(pc)
    meta.to_csv(pm)
    ctx.add_artifact("table", pc, caption="pseudobulk counts (sample x label)")
    ctx.add_artifact("table", pm, caption="pseudobulk unit metadata")
    ctx.summary = {"counts_path": str(pc), "metadata_path": str(pm),
                   "shape": list(counts.shape), "n_units": int(len(meta)),
                   "dropped_units": dropped[:8],
                   "labels": sorted(set(meta["label"]))}
    ctx.suggest("skin.de.pseudobulk", "skin.de.deseq2_r")


@tool("skin.de.compare_methods", category="de",
      summary="Concordance between two DE runs: Spearman rho and top-N Jaccard.")
def compare_methods(run_a: str, run_b: str, label: str = "", top_n: int = 200,
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Quantify how much moves between two DE runs. Worth having when switching
    from cell-wise to pseudobulk — the significant-gene counts typically drop a
    lot, and a reviewer will ask by how much.

    Args:
        run_a: First DE run_id.
        run_b: Second DE run_id.
        label: Compare only this label. Empty = every label the runs share.
        top_n: Top-N size for the Jaccard overlap.
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    import pandas as pd
    from scipy.stats import spearmanr

    ra = store.get_run(ctx.project_id, run_a)
    rb = store.get_run(ctx.project_id, run_b)
    for rid, r in ((run_a, ra), (run_b, rb)):
        if r is None:
            raise NotFound(f"unknown run_id {rid!r}",
                           remedy="skin.memory.brief lists recent runs.")
    ta = ra["result"].get("tables", {})
    tb = rb["result"].get("tables", {})
    shared = sorted(set(ta) & set(tb)) if not label else [label]
    if not shared:
        raise BadParam("the two runs share no labels",
                       remedy=f"run_a labels: {sorted(ta)}; run_b labels: {sorted(tb)}")

    rows = []
    for lb in shared:
        da = pd.read_csv(ta[lb]).set_index("gene")
        db = pd.read_csv(tb[lb]).set_index("gene")
        common = da.index.intersection(db.index)
        if len(common) < 20:
            rows.append({"label": lb, "n_common_genes": int(len(common)),
                         "note": "too few shared genes to compare"})
            continue
        rho = float(spearmanr(da.loc[common, "lfc"], db.loc[common, "lfc"]).statistic)
        sa = set(da.nsmallest(top_n, "padj").index)
        sb = set(db.nsmallest(top_n, "padj").index)
        jac = len(sa & sb) / max(len(sa | sb), 1)
        rows.append({
            "label": lb, "n_common_genes": int(len(common)),
            "spearman_lfc": round(rho, 3), f"jaccard_top{top_n}": round(jac, 3),
            "n_sig_a": int((da["padj"] < 0.05).sum()),
            "n_sig_b": int((db["padj"] < 0.05).sum()),
        })

    ctx.summary = {
        "run_a": {"id": run_a, "method": ra["params"].get("method")},
        "run_b": {"id": run_b, "method": rb["params"].get("method")},
        "per_label": rows,
        "interpretation": ("A large drop in n_sig from a cell-wise run to a pseudobulk run "
                           "is expected, not a bug: cell-wise p-values treat cells from one "
                           "mouse as independent replicates. Spearman rho on LFC is the "
                           "fairer comparison — the rankings usually agree well even when "
                           "the counts do not."),
    }
    ctx.suggest("skin.plot.volcano_grid", "skin.memory.record_decision")


@tool("skin.de.deseq2_r", category="de", needs_r=True,
      summary="DESeq2 (R) cross-check on the same pseudobulk matrix.")
def deseq2_r(dataset_id: str, label_key: str, condition_key: str, contrast: list[str],
             groups: list[str] | None = None, sample_key: str = "Sample",
             covariates: list[str] | None = None, min_cells: int = 10,
             project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Run the reference R DESeq2 on the same pseudobulk matrix, for reviewer requests.

    Args:
        dataset_id: Handle or label with counts.
        label_key: obs column of cell type labels.
        condition_key: obs column of the condition.
        contrast: [test, reference].
        groups: Labels to test. Empty = all.
        sample_key: obs column of biological replicates.
        covariates: Blocking factors.
        min_cells: Minimum cells per pseudobulk unit.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    adata = registry.load(ctx.project_id, dataset_id)
    for k in (label_key, condition_key, sample_key):
        require_obs(adata, k)
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.dataset_id = resolved
    a, b = str(contrast[0]), str(contrast[1])
    covs = list(covariates) if covariates is not None else ["Timepoint"]
    covs = [c for c in covs if c in adata.obs.columns]
    labels = [str(x) for x in (groups or adata.obs[label_key].astype(str).unique())]
    if dry_run:
        ctx.summary = {"backend": "R DESeq2", "labels": labels, "contrast": [a, b],
                       "covariates": covs}
        return

    counts, meta, _ = _aggregate_pseudobulk(adata, label_key, sample_key, labels, min_cells)
    meta = meta.join(_sample_metadata(adata, sample_key, [condition_key, *covs]), on="sample")
    work = ctx.tabledir() / "deseq2_r_input"
    work.mkdir(exist_ok=True)
    counts.T.to_csv(work / "counts.csv")
    meta.to_csv(work / "metadata.csv")

    res = run_r_script("de_deseq2", adata=None, project_id=ctx.project_id,
                       params={"input_dir": str(work), "condition_key": condition_key,
                               "contrast": [a, b], "covariates": covs, "seed": seed},
                       python_fallback="skin.de.pseudobulk")
    run_id = store.new_id("de", 8)
    store.record_run(ctx.project_id, run_id, "de", resolved or dataset_id,
                     {"method": "deseq2_r", "contrast": [a, b]}, res)
    ctx.summary = {"run_id": run_id, "method": "deseq2_r", "inference_level": "sample",
                   "per_label": res.get("per_label", []),
                   "r_log_tail": (res.get("log") or "")[-400:]}
    ctx.suggest("skin.de.compare_methods", "skin.plot.volcano_grid")
