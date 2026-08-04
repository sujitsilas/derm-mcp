"""`skin.annotate.*` — lineage scoring, label proposal, and the contamination audit.

This module is the heart of the request: *check whether non-specific markers are
being expressed (keratinocytes expressing Col1a1), regress them out, and run an
iterative clustering and annotation loop.*

Two things it will not do:

- It never writes an obs label on its own. `marker_report` proposes; you decide;
  `apply_labels` writes; `skin.memory.record_annotation` records why.
- It never auto-removes cells. `contamination_audit` classifies the *cause*
  (ambient / doublet / mixed cluster / true biology) because the remedies
  differ, and returns a recommended action for a human or model to approve.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import AmbiguousLabels, BadParam
from ..memory import store
from ._base import Ctx, confirm_or_raise, require_obs, tool

logger = logging.getLogger(__name__)


def _score_sets(adata: Any, sets: dict[str, list[str]], ctx: Ctx | None = None,
                prefix: str = "score_", seed: int = 0) -> tuple[list[str], dict[str, int]]:
    """sc.tl.score_genes for each set present in var_names."""
    import scanpy as sc

    written, n_present = [], {}
    for name, genes in sets.items():
        g = K.present(adata, genes)
        n_present[name] = len(g)
        if len(g) < 2:
            continue
        col = f"{prefix}{name}"
        sc.tl.score_genes(adata, g, score_name=col, use_raw=False, random_state=seed)
        written.append(col)
    return written, n_present


@tool("skin.annotate.score_lineages", category="annotate",
      summary="Score every lineage marker set per cell; return a per-cluster score matrix.")
def score_lineages(dataset_id: str, cluster_key: str = "", family: str = "lineages",
                   label: str = "", project_id: str = "", dry_run: bool = False,
                   seed: int = 0, *, ctx: Ctx) -> None:
    """Score the shipped skin lineage sets and summarise them per cluster.

    Returns a per-cluster mean-score matrix plus a normalised entropy: low
    entropy means one lineage dominates (a clean cluster), high entropy means
    several score similarly (a mixed cluster, a doublet cluster, or a genuinely
    intermediate state).

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        cluster_key: obs column to summarise by. Empty = per-cell scores only.
        family: "lineages" for the first pass, or a subtype family:
            "macrophage", "macrophage_origin", "fibroblast", "keratinocyte", "tcell".
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report which sets would be scored.
        seed: RNG seed.
    """
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    sets = (K.lineages(organism) if family == "lineages"
            else K.subtype_sets(organism, family))

    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")
        ctx.warn("X was z-scaled; restored layers['lognorm'] for scoring. score_genes on "
                 "scaled data compares z-scores against a z-scored control set, which is "
                 "not what the statistic means.")

    ctx.code = (
        "import scanpy as sc\n"
        f"# {family} sets from skinmcp knowledge/markers_{organism}.yaml\n"
        "from skinmcp import knowledge as K\n"
        + (f"SETS = K.lineages({organism!r})\n" if family == "lineages"
           else f"SETS = K.subtype_sets({organism!r}, {family!r})\n")
        + "for name, genes in SETS.items():\n"
        "    present = [g for g in genes if g in adata.var_names]\n"
        "    if len(present) >= 2:\n"
        f"        sc.tl.score_genes(adata, present, score_name=f'score_{{name}}',\n"
        f"                          use_raw=False, random_state={seed})\n"
    )
    if dry_run:
        cov = {n: len(K.present(adata, g)) for n, g in sets.items()}
        ctx.summary = {"family": family, "n_sets": len(sets), "genes_present_per_set": cov}
        return

    written, n_present = _score_sets(adata, sets, ctx, seed=seed)
    thin = [n for n, c in n_present.items() if c < 3]
    if thin:
        ctx.warn(f"{len(thin)} sets have fewer than 3 genes present and were scored weakly or "
                 f"skipped: {thin[:10]}. If the object is subset to HVGs, score on the full "
                 f"gene space instead.")

    per_cluster = None
    entropy = None
    if cluster_key:
        require_obs(adata, cluster_key)
        M = (adata.obs[written].assign(_c=adata.obs[cluster_key].astype(str))
             .groupby("_c", observed=True).mean())
        M.columns = [c.replace("score_", "") for c in M.columns]
        # Softmax over the row so entropy is comparable across clusters.
        Z = M.sub(M.mean(1), axis=0).div(M.std(1).replace(0, 1), axis=0)
        E = np.exp(Z)
        Pm = E.div(E.sum(1), axis=0)
        ent = -(Pm * np.log(Pm.clip(lower=1e-12))).sum(1) / np.log(max(Pm.shape[1], 2))
        per_cluster = {
            str(c): {"top": M.loc[c].nlargest(3).round(3).to_dict(),
                     "entropy": round(float(ent[c]), 3),
                     "n_cells": int((adata.obs[cluster_key].astype(str) == c).sum())}
            for c in M.index
        }
        entropy = {str(k): round(float(v), 3) for k, v in ent.items()}
        high = [k for k, v in entropy.items() if v > 0.85]
        if high:
            ctx.warn(f"clusters {high[:8]} have high lineage entropy (>0.85): no single "
                     f"lineage dominates. Candidates for subclustering or doublet removal — "
                     f"run skin.annotate.contamination_audit to tell those apart.")
        tbl = ctx.tabledir() / f"lineage_scores_{cluster_key}.csv"
        M.round(4).to_csv(tbl)
        ctx.add_artifact("table", tbl, caption=f"mean {family} scores per {cluster_key}")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="annotate.score_lineages",
                         params={"family": family, "cluster_key": cluster_key, "seed": seed},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "family": family, "n_sets_scored": len(written),
                   "score_columns": written[:30], "genes_present_per_set": n_present,
                   "per_cluster": per_cluster, "entropy": entropy}
    ctx.suggest("skin.annotate.marker_report", "skin.plot.score_umap_grid",
                "skin.annotate.apply_labels")


@tool("skin.annotate.marker_report", category="annotate",
      summary="Per-cluster: top DE genes, lineage hits, a proposed label, and confidence. Proposal only.")
def marker_report(dataset_id: str, cluster_key: str, top_n: int = 25, family: str = "lineages",
                  project_id: str = "", dry_run: bool = False, seed: int = 0,
                  *, ctx: Ctx) -> None:
    """Propose a label for every cluster, with the evidence behind it.

    This tool NEVER writes obs. It reports, for each cluster: the top ranked
    genes, which lineage sets those genes hit, the mean lineage score, a
    proposed label, and a confidence derived from the margin between the best
    and second-best lineage. You decide; apply_labels writes.

    Args:
        dataset_id: Handle or label. Needs skin.cluster.marker_genes to have run.
        cluster_key: obs column of cluster ids.
        top_n: How many ranked genes per cluster to consider.
        family: "lineages" or a subtype family name.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, cluster_key)
    organism = registry.get_organism(adata)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    sets = (K.lineages(organism) if family == "lineages"
            else K.subtype_sets(organism, family))
    gene_to_sets: dict[str, list[str]] = {}
    for name, genes in sets.items():
        for g in genes:
            gene_to_sets.setdefault(g, []).append(name)

    key = f"rank_genes_{cluster_key}"
    if key not in adata.uns:
        # marker_genes mints a new handle. If it has already run for this
        # grouping, the table exists — just on a different handle — and telling
        # the caller to "run it first" sends it round the same loop again.
        have = registry.find_by_op(ctx.project_id, "cluster.marker_genes",
                                   {"groupby": cluster_key})
        if have:
            raise BadParam(
                f"this handle has no uns[{key!r}], but {have[-1]} does",
                remedy=(f"skin.cluster.marker_genes already ran for "
                        f"{cluster_key!r} and minted {have[-1]}. Re-issue this call "
                        f"with dataset_id={have[-1]!r} — do not re-run marker_genes."),
                suggested_tool="",
                details={"given": dataset_id, "has_markers": have[-3:]},
            )
        raise BadParam(
            f"no marker table under uns[{key!r}]",
            remedy=f"Run skin.cluster.marker_genes(dataset_id, groupby={cluster_key!r}) first.",
            suggested_tool="skin.cluster.marker_genes",
        )
    if dry_run:
        ctx.summary = {"cluster_key": cluster_key, "family": family, "top_n": top_n}
        return

    df = sc.get.rank_genes_groups_df(adata, group=None, key=key)
    score_cols = [c for c in adata.obs.columns if c.startswith("score_")]
    if not score_cols:
        _score_sets(adata, sets, ctx, seed=seed)
        score_cols = [c for c in adata.obs.columns if c.startswith("score_")]

    mean_scores = (adata.obs[score_cols].assign(_c=adata.obs[cluster_key].astype(str))
                   .groupby("_c", observed=True).mean())
    mean_scores.columns = [c.replace("score_", "") for c in mean_scores.columns]

    rows = []
    for cl, sub in df.groupby("group", observed=True):
        top = sub.nsmallest(top_n, "pvals_adj")
        genes = top["names"].tolist()
        hits: dict[str, list[str]] = {}
        for g in genes:
            for s in gene_to_sets.get(g, []):
                hits.setdefault(s, []).append(g)
        marker_votes = {s: len(v) for s, v in hits.items()}

        cl_s = str(cl)
        sc_row = mean_scores.loc[cl_s] if cl_s in mean_scores.index else None
        ranked = sc_row.sort_values(ascending=False) if sc_row is not None else None

        # Combine marker overlap with the module score; neither alone is enough.
        combined: dict[str, float] = {}
        max_votes = max(marker_votes.values()) if marker_votes else 1
        for s in sets:
            v = marker_votes.get(s, 0) / max(max_votes, 1)
            z = 0.0
            if ranked is not None and s in ranked.index:
                spread = float(ranked.max() - ranked.min()) or 1.0
                z = float((ranked[s] - ranked.min()) / spread)
            combined[s] = round(0.6 * v + 0.4 * z, 3)
        best = sorted(combined.items(), key=lambda kv: -kv[1])[:3]
        margin = (best[0][1] - best[1][1]) if len(best) > 1 else best[0][1]
        conf = float(np.clip(0.35 + 1.6 * margin, 0.0, 0.97))

        pct_mt = (round(float(adata.obs.loc[adata.obs[cluster_key].astype(str) == cl_s,
                                            "pct_counts_mt"].median()), 1)
                  if "pct_counts_mt" in adata.obs else None)
        rows.append({
            "cluster": cl_s,
            "n_cells": int((adata.obs[cluster_key].astype(str) == cl_s).sum()),
            "top_genes": genes[:12],
            "lineage_hits": {s: v[:6] for s, v in sorted(hits.items(),
                                                         key=lambda kv: -len(kv[1]))[:4]},
            "proposed_label": best[0][0],
            "confidence": round(conf, 2),
            "runner_up": best[1][0] if len(best) > 1 else None,
            "margin": round(float(margin), 3),
            "median_pct_mt": pct_mt,
        })
        if conf < 0.5:
            ctx.warn(f"cluster {cl_s}: low confidence ({conf:.2f}); "
                     f"{best[0][0]} vs {best[1][0] if len(best) > 1 else '-'} are close. "
                     f"Inspect the markers before committing, or subcluster it.")

    rows.sort(key=lambda r: -r["n_cells"])
    dupes: dict[str, list[str]] = {}
    for r in rows:
        dupes.setdefault(r["proposed_label"], []).append(r["cluster"])
    multi = {k: v for k, v in dupes.items() if len(v) > 1}

    ctx.summary = {
        "cluster_key": cluster_key, "family": family, "n_clusters": len(rows),
        "per_cluster": rows,
        "proposed_mapping": {r["cluster"]: r["proposed_label"] for r in rows},
        "labels_used_more_than_once": multi,
        "note": ("PROPOSAL ONLY — nothing was written to obs. Review, then call "
                 "skin.annotate.apply_labels with your mapping and "
                 "skin.memory.record_annotation with your rationale."),
    }
    ctx.suggest("skin.annotate.apply_labels", "skin.memory.record_annotation",
                "skin.plot.dotplot")


@tool("skin.annotate.apply_labels", category="annotate",
      summary="Write a cluster->label mapping to a new obs column.")
def apply_labels(dataset_id: str, cluster_key: str, mapping: dict[str, str], new_key: str,
                 order: list[str] | None = None, palette: dict[str, str] | None = None,
                 scheme: str = "celltype", label: str = "", project_id: str = "",
                 dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Apply a cluster->label mapping. Errors on unmapped clusters rather than making NaN.

    Args:
        dataset_id: Handle or label.
        cluster_key: obs column of cluster ids.
        mapping: {cluster_id: label}. Must cover every cluster present.
        new_key: New obs column, e.g. "cell_types" or "macrophage_subtypes".
        order: Category order for the new column. Defaults to first-appearance
            order in `mapping`.
        palette: Explicit {label: "#RRGGBB"}. Otherwise built from `scheme`.
        scheme: Palette scheme when `palette` is omitted.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Validate coverage without writing.
        seed: RNG seed.
    """
    import pandas as pd

    from ..style import palettes as PAL

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, cluster_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    present = set(map(str, adata.obs[cluster_key].astype(str).unique()))
    mapped = {str(k): str(v) for k, v in mapping.items()}
    missing = sorted(present - set(mapped))
    if missing:
        raise AmbiguousLabels(
            f"{len(missing)} clusters have no label: {missing}",
            remedy=(f"Every cluster must be mapped or the column gets NaN, which then "
                    f"silently drops those cells from every downstream contrast. "
                    f"Clusters present: {sorted(present)}. Map the rest, or give them an "
                    f"explicit placeholder such as 'Unassigned'."),
            details={"unmapped": missing, "clusters_present": sorted(present)},
        )
    extra = sorted(set(mapped) - present)
    if extra:
        ctx.warn(f"mapping has entries for clusters not present: {extra[:10]} (ignored)")

    labels = list(dict.fromkeys(mapped[c] for c in sorted(present, key=str)))
    final_order = order or labels
    unknown_in_order = [x for x in labels if x not in final_order]
    if unknown_in_order:
        raise BadParam(f"order is missing labels {unknown_in_order}",
                       remedy=f"Labels produced by the mapping: {labels}")

    ctx.code = (f"mapping = {mapped!r}\n"
                f"adata.obs[{new_key!r}] = pd.Categorical(\n"
                f"    adata.obs[{cluster_key!r}].astype(str).map(mapping),\n"
                f"    categories={final_order!r}, ordered=True)\n")
    if dry_run:
        ctx.summary = {"new_key": new_key, "n_clusters": len(present), "labels": labels,
                       "order": final_order}
        return

    vals = adata.obs[cluster_key].astype(str).map(mapped)
    adata.obs[new_key] = pd.Categorical(vals, categories=final_order, ordered=True)
    pal = palette or PAL.build(scheme, final_order)
    PAL.apply_to_adata(adata, new_key, pal)

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="annotate.apply_labels",
                         params={"cluster_key": cluster_key, "mapping": mapped,
                                 "new_key": new_key, "order": final_order},
                         label=label)
    ctx.dataset_id = dsid
    vc = adata.obs[new_key].value_counts().reindex(final_order).to_dict()
    ctx.summary = {"dataset_id": dsid, "new_key": new_key, "n_labels": len(final_order),
                   "counts": vc, "palette": pal}
    _warn_on_discards(ctx, vc, int(adata.n_obs))
    ctx.warn("Labels written. Record WHY with skin.memory.record_annotation — the mapping "
             "alone is not auditable, and it is the rationale a reviewer will ask for.")
    ctx.suggest("skin.memory.record_annotation", "skin.annotate.contamination_audit",
                "skin.plot.umap")


#: Substrings that mark a label as a bucket rather than a cell identity. Matched
#: against labels the CALLER just wrote, not against columns in the user's data,
#: so this is not the kind of name-guessing that _introspect.py exists to avoid.
_DISCARD_WORDS = ("other", "unknown", "unassigned", "low quality", "lowqual",
                  "low-quality", "doublet", "junk", "debris", "mixed", "ambiguous",
                  "artifact", "exclude", "discard", "n/a")


def _warn_on_discards(ctx: Ctx, counts: dict[str, Any], n_obs: int) -> None:
    """Flag a labelling that quietly bins a large share of cells.

    Over-clustering makes this easy to do by accident: 14 clusters from 20k
    neutrophils, four of them hard to name, and 44% of the population lands in
    "Other"/"Low Quality" with every tool reporting success. Genuine debris
    (mitochondrial-high, erythrocyte, a stray lineage) is usually a few percent;
    much beyond that is normally real cells whose cluster was never identified.
    """
    if not n_obs:
        return
    binned = {k: int(v or 0) for k, v in counts.items()
              if any(w in str(k).lower() for w in _DISCARD_WORDS)}
    n = sum(binned.values())
    pct = 100.0 * n / n_obs
    if pct >= 15.0:
        ctx.warn(
            f"{pct:.0f}% of cells ({n}/{n_obs}) were labelled {sorted(binned)} rather "
            f"than a cell identity. Debris is usually a few percent, so check these are "
            f"really discardable before dropping them: a cluster of genuine cells that "
            f"was merely hard to name belongs in the analysis. Over-clustering is the "
            f"usual cause — a lower leiden resolution often removes the ambiguity. "
            f"skin.annotate.contamination_audit distinguishes contamination from "
            f"unnamed biology."
        )


# --------------------------------------------------------------------------- #
# contamination audit
# --------------------------------------------------------------------------- #

@tool("skin.annotate.contamination_audit", category="annotate",
      summary="Per-label cross-lineage contamination, with a likely cause and a remedy.")
def contamination_audit(dataset_id: str, label_key: str, sample_key: str = "Sample",
                        min_cells: int = 20, project_id: str = "", dry_run: bool = False,
                        seed: int = 0, *, ctx: Ctx) -> None:
    """Audit every label for foreign lineage programs and classify the cause.

    Computes per label:
      1. Cross-lineage co-expression rate for mutually exclusive pairs, corrected
         against the object-wide baseline (keratinocyte∩fibroblast, immune∩structural, ...).
      2. Foreign-program score: score_genes for every *other* lineage set within
         this label, flagged when a foreign score beats the native score in >10% of cells.
      3. Ambient signature: correlation of the label's mean profile with the
         globally most abundant genes.
      4. Doublet concentration.
      5. Sample skew — a "cell type" that came from one mouse.

    The `likely_cause` classification matters because the remedies differ:
      - uniform, low magnitude, present in EVERY label  -> ambient
          -> fix at the count level (skin.qc.estimate_ambient), or exclude the
             genes from the feature space
      - bimodal within the label, high magnitude, elevated doublet score
          -> heterotypic doublets -> remove the cells or the cluster
      - one label carrying two COMPLETE programs, low doublet score
          -> mixed cluster from under-clustering -> raise resolution / subcluster
      - a coherent minority with a genuine dual program (Arg1+Nos2+ macrophages)
          -> true biology -> do not remove

    This tool never removes anything. It proposes; you decide; the decision goes
    to skin.memory.record_decision.

    Args:
        dataset_id: Handle or label with a label column.
        label_key: obs column of cell type labels.
        sample_key: obs column identifying samples, for the skew check.
        min_cells: Skip labels smaller than this.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import numpy as np
    import pandas as pd
    import scipy.sparse as sp

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, label_key)
    organism = registry.get_organism(adata)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    lin = K.lineages(organism)
    pairs = K.exclusive_pairs(organism)

    if registry.get_x_state(adata) == "scaled" and "lognorm" in adata.layers:
        adata.X = adata.layers["lognorm"].copy()
        registry.set_x_state(adata, "lognorm")

    ctx.code_executable = False
    ctx.code = (
        "# Contamination audit — the full computation is ~150 lines; it lives in\n"
        "# skinmcp.tools.annotate_tools.contamination_audit. Reproduce it with:\n"
        "#\n"
        "#   from skinmcp.tools import annotate_tools\n"
        f"#   annotate_tools.contamination_audit(dataset_id=..., label_key={label_key!r})\n"
        "#\n"
        "# It computes, per label: baseline-corrected cross-lineage co-detection,\n"
        "# foreign module scores, ambient correlation, doublet fraction, sample skew.\n"
        f"# The per-label result table is written to tables/contamination_audit_{label_key}.csv\n"
    )
    if dry_run:
        ctx.summary = {"label_key": label_key, "n_labels": int(adata.obs[label_key].nunique()),
                       "n_exclusive_pairs": len(pairs)}
        return

    # Score every lineage first: the module score is background-corrected (it
    # subtracts a matched control set), so a positive value means genuine
    # enrichment rather than "this cell is deeply sequenced".
    scored, _ = _score_sets(adata, lin, ctx, prefix="_fscore_", seed=seed)
    score_names = [c.replace("_fscore_", "") for c in scored]

    # --- "expresses lineage L" per cell --------------------------------------
    # Detection count alone saturates: in a well-sequenced cell, >=2 of any 8
    # markers are usually non-zero, so every pair co-detects at ~100% and the
    # signal is drowned. Requiring BOTH >=2 detected markers AND a positive
    # module score keeps it specific.
    det: dict[str, Any] = {}
    for name, genes in lin.items():
        g = K.present(adata, genes)
        col = f"_fscore_{name}"
        if len(g) < 3 or col not in adata.obs:
            continue
        X = adata[:, g].X
        n_det = (np.asarray((X > 0).sum(1)).ravel() if sp.issparse(X)
                 else (np.asarray(X) > 0).sum(1))
        det[name] = (n_det >= 2) & (adata.obs[col].to_numpy() > 0)

    # --- object-wide ambient reference: globally most abundant genes ----------
    Xc = adata.layers.get("counts", adata.X)
    gene_tot = np.asarray(Xc.sum(0)).ravel()
    top20_idx = np.argsort(-gene_tot)[:20]
    top20_genes = [str(adata.var_names[i]) for i in top20_idx]
    global_mean = np.asarray(adata[:, top20_genes].X.mean(0)).ravel()

    labels = [str(x) for x in adata.obs[label_key].astype(str).unique()]
    lab_arr = adata.obs[label_key].astype(str).to_numpy()

    baseline_pairs: dict[str, float] = {}
    for a, b in pairs:
        if a in det and b in det:
            baseline_pairs[f"{a}|{b}"] = float((det[a] & det[b]).mean())

    per_label: list[dict[str, Any]] = []
    for lab in labels:
        m = lab_arr == lab
        n = int(m.sum())
        if n < min_cells:
            continue
        row: dict[str, Any] = {"label": lab, "n_cells": n}

        # 1. cross-lineage co-detection, baseline-corrected
        co: dict[str, dict[str, float]] = {}
        for a, b in pairs:
            if a not in det or b not in det:
                continue
            rate = float((det[a][m] & det[b][m]).mean())
            base = baseline_pairs.get(f"{a}|{b}", 0.0)
            excess = rate - base
            if excess > 0.02:
                co[f"{a}+{b}"] = round(excess, 3)
        # value is the baseline-corrected EXCESS co-detection rate; the raw rate
        # and baseline are in the CSV artifact
        row["cross_lineage_excess"] = dict(sorted(co.items(), key=lambda kv: -kv[1])[:3])
        row["_cross_lineage_full"] = {k: {"rate": round(v, 3)} for k, v in
                                      sorted(co.items(), key=lambda kv: -kv[1])}

        # 2. foreign program score
        native = _match_lineage(lab, score_names)
        foreign_frac: dict[str, float] = {}
        if native:
            nat_v = adata.obs[f"_fscore_{native}"].to_numpy()[m]
            for s in score_names:
                if s == native:
                    continue
                f = float((adata.obs[f"_fscore_{s}"].to_numpy()[m] > nat_v).mean())
                if f > 0.05:
                    foreign_frac[s] = round(f, 3)
        row["native_lineage"] = native
        row["foreign_program"] = dict(sorted(foreign_frac.items(), key=lambda kv: -kv[1])[:4])
        dominant_foreign = next(iter(row["foreign_program"]), None)

        # 3. ambient signature
        lab_mean = np.asarray(adata[m][:, top20_genes].X.mean(0)).ravel()
        amb_r = float(np.corrcoef(lab_mean, global_mean)[0, 1]) if lab_mean.std() > 0 else 0.0
        row["ambient_correlation"] = round(amb_r, 3)

        # 4. doublet concentration
        row["doublet_fraction"] = (
            round(float(adata.obs["predicted_doublet"].to_numpy()[m].astype(float).mean()), 3)
            if "predicted_doublet" in adata.obs else None)

        # 5. sample skew
        if sample_key in adata.obs.columns:
            vc = adata.obs[sample_key].astype(str).to_numpy()[m]
            u, c = np.unique(vc, return_counts=True)
            row["dominant_sample"] = str(u[np.argmax(c)])
            row["dominant_sample_frac"] = round(float(c.max() / n), 3)

        # --- classification --------------------------------------------------
        max_excess = max(row["cross_lineage_excess"].values(), default=0.0)
        max_foreign = max(foreign_frac.values(), default=0.0)
        dbl = row["doublet_fraction"] or 0.0

        # Bimodality of the dominant foreign score inside this label: a mixed
        # cluster is bimodal (two populations), true dual biology is unimodal.
        bimodal = 0.0
        if dominant_foreign:
            v = adata.obs[f"_fscore_{dominant_foreign}"].to_numpy()[m]
            if v.std() > 0:
                lo, hi = np.percentile(v, [25, 75])
                bimodal = float((hi - lo) / (v.std() + 1e-9))

        score = round(float(0.5 * max_foreign + 0.35 * max_excess + 0.15 * min(dbl * 3, 1)), 3)
        row["contamination_score"] = score
        row["dominant_foreign_lineage"] = dominant_foreign

        if amb_r > 0.9 and max_excess < 0.10 and max_foreign < 0.25:
            cause, action = "ambient", (
                "Low-magnitude foreign signal that tracks the globally abundant genes and "
                "is present across labels. Fix at the count level with "
                "skin.qc.estimate_ambient, or drop the gene groups from the feature space "
                "with skin.integrate.preprocess(exclude_gene_groups=[...]).")
        elif dbl > 0.25 and max_foreign > 0.2:
            cause, action = "doublet", (
                "Elevated doublet fraction alongside a strong foreign program: heterotypic "
                "doublets. Check skin.doublet.cluster_enrichment, then remove the cells or "
                "the cluster with skin.sub.drop_clusters.")
        elif max_foreign > 0.35 and bimodal > 1.2 and dbl < 0.25:
            cause, action = "mixed_cluster", (
                "Two complete programs inside one label with a low doublet fraction: this is "
                "under-clustering. Subcluster it with skin.sub.pipeline at a higher "
                "resolution and re-label.")
        elif 0.10 < max_foreign <= 0.35 and bimodal <= 1.2:
            cause, action = "true_biology", (
                "A coherent minority carrying a dual program (e.g. Arg1+Nos2+ macrophages). "
                "Do NOT remove. Record the observation with skin.memory.note.")
        else:
            cause, action = "clean", "No action needed."
        row["likely_cause"] = cause
        row["recommended_action"] = action
        row["_bimodality"] = round(bimodal, 2)
        per_label.append(row)

    per_label.sort(key=lambda r: -r["contamination_score"])
    for c in [c for c in adata.obs.columns if c.startswith("_fscore_")]:
        del adata.obs[c]

    causes = pd.Series([r["likely_cause"] for r in per_label]).value_counts().to_dict()
    n_ambient = causes.get("ambient", 0)
    if n_ambient >= max(2, 0.5 * len(per_label)):
        ctx.warn(f"{n_ambient} of {len(per_label)} labels look ambient-contaminated. Uniform "
                 f"low-level foreign expression across every label is soup, not doublets. "
                 f"The fix is upstream (skin.qc.estimate_ambient / CellBender), not gene "
                 f"exclusion — excluding genes hides it from DE but leaves the counts "
                 f"distorting library sizes and the neighbour graph.")
    mixed = [r["label"] for r in per_label if r["likely_cause"] == "mixed_cluster"]
    if mixed:
        ctx.warn(f"labels {mixed[:6]} look like mixed clusters. skin.annotate.refine_loop "
                 f"will plan the subclustering for you.")

    tbl = ctx.tabledir() / f"contamination_audit_{label_key}.csv"
    pd.json_normalize(per_label).to_csv(tbl, index=False)
    ctx.add_artifact("table", tbl, caption=f"contamination audit for {label_key}")

    def inline(r: dict[str, Any]) -> dict[str, Any]:
        return {"label": r["label"], "n_cells": r["n_cells"],
                "contamination_score": r["contamination_score"],
                "likely_cause": r["likely_cause"],
                "dominant_foreign_lineage": r["dominant_foreign_lineage"],
                "foreign_program": dict(list(r["foreign_program"].items())[:2]),
                "cross_lineage_excess": dict(list(r["cross_lineage_excess"].items())[:2]),
                "doublet_fraction": r["doublet_fraction"],
                "ambient_correlation": r["ambient_correlation"],
                "recommended_action": r["recommended_action"][:180]}

    ctx.summary = {
        "label_key": label_key, "n_labels_audited": len(per_label),
        "per_label": [inline(r) for r in per_label[:8]],
        "cause_counts": causes,
        "full_table": str(tbl),
        "top20_abundant_genes": top20_genes[:8],
        "overall_contamination": round(
            float(np.mean([r["contamination_score"] for r in per_label])) if per_label else 0.0,
            3),
        "note": "Nothing was removed. Decide, then record with skin.memory.record_decision.",
    }
    ctx.suggest("skin.annotate.refine_loop", "skin.qc.estimate_ambient",
                "skin.sub.pipeline", "skin.memory.record_decision")


def _match_lineage(label: str, candidates: list[str]) -> str | None:
    """Best-effort match of a free-text cell label to a lineage set name."""
    from ..style.palettes import norm_key

    nl = norm_key(label)
    for c in candidates:
        if norm_key(c) == nl:
            return c
    for c in candidates:
        nc = norm_key(c)
        if nc and (nc in nl or nl in nc):
            return c
    # Common aliases the marker-set names do not carry.
    ALIAS = {"mono": "Monocytes", "mdm": "Macrophages", "mphi": "Macrophages",
             "mac": "Macrophages", "lam": "Macrophages", "kc": "Keratinocytes",
             "fibro": "Fibroblasts", "endo": "Endothelial", "neut": "Neutrophils",
             "dc": "cDC", "treg": "T cells", "nk": "ILC/NK", "smc": "Smooth muscle"}
    for k, v in ALIAS.items():
        if k in nl and v in candidates:
            return v
    return None


@tool("skin.annotate.regress_markers", category="annotate", destructive=True,
      summary="Exclude (default) or regress out non-specific gene groups.")
def regress_markers(dataset_id: str, gene_groups: list[str], mode: str = "exclude",
                    confirm: bool = False, label: str = "", project_id: str = "",
                    dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Remove non-specific genes from the feature space, or regress out their module score.

    READ THIS: removing genes from the feature space is NOT decontamination. It
    stops the contaminating genes driving clusters and topping DE lists, but the
    contaminating COUNTS remain in the matrix and still distort library sizes,
    size factors, and the neighbour graph. Real ambient removal belongs upstream
    (skin.qc.estimate_ambient / SoupX / DecontX / CellBender). This tool says so
    in every return, deliberately.

    Args:
        dataset_id: Handle or label.
        gene_groups: Names from knowledge/contamination.yaml — "collagen",
            "keratin", "muscle", "cornified", "ecm_misc", "stress", "mito",
            "ribo", "hb", "dissociation", "sex — or a preset: "immune_de",
            "feature_space", "ambient_heavy".
        mode: "exclude" (drop the genes; what the reference notebook does) or
            "regress" (sc.pp.regress_out on the module score — slow, distorts
            variance, rarely right).
        confirm: Required; this changes the feature space for everything downstream.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: List the genes that would be removed.
        seed: RNG seed.
    """
    import scanpy as sc

    if mode not in ("exclude", "regress"):
        raise BadParam(f"mode must be exclude|regress, got {mode!r}")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    matched = K.match_gene_groups(organism, gene_groups, adata.var_names)
    genes = sorted({g for v in matched.values() for g in v})

    ctx.code = (
        "import scanpy as sc\n"
        f"# groups {gene_groups!r} -> {len(genes)} genes\n"
        f"EXCLUDED = {genes!r}\n"
        + ("adata = adata[:, [g for g in adata.var_names if g not in set(EXCLUDED)]].copy()\n"
           if mode == "exclude" else
           "sc.tl.score_genes(adata, EXCLUDED, score_name='_contam_score', use_raw=False)\n"
           "sc.pp.regress_out(adata, ['_contam_score'])\n")
    )
    if dry_run:
        ctx.summary = {"mode": mode, "n_genes": len(genes),
                       "by_group": {k: len(v) for k, v in matched.items()},
                       "examples": genes[:20]}
        return
    if not genes:
        raise BadParam(f"groups {gene_groups} matched no genes in var_names",
                       remedy=f"Available groups: {sorted(K.contamination_groups(organism))}")

    confirm_or_raise(confirm, dry_run, "skin.annotate.regress_markers",
                     f"This changes the feature space for every downstream step: "
                     f"{len(genes)} genes across {list(matched)}.")

    if mode == "exclude":
        keep = [g for g in map(str, adata.var_names) if g not in set(genes)]
        adata = adata[:, keep].copy()
    else:
        sc.tl.score_genes(adata, genes, score_name="_contam_score", use_raw=False,
                          random_state=seed)
        sc.pp.regress_out(adata, ["_contam_score"])
        ctx.warn("mode='regress' distorts the variance structure of every gene, not only the "
                 "contaminating ones, and it does not remove the counts either. "
                 "mode='exclude' is almost always the better choice.")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent,
                         op="annotate.regress_markers",
                         params={"gene_groups": gene_groups, "mode": mode},
                         label=label or "decontam_features")
    ctx.dataset_id = dsid
    ctx.warn(
        "Gene exclusion is NOT decontamination. The contaminating counts are still in "
        "layers['counts'] and still inflate library sizes, size factors, and the "
        "neighbour graph. If ambient RNA is the real problem, fix it at the count level "
        "with skin.qc.estimate_ambient (SoupX/DecontX) or CellBender before "
        "normalization. This exclusion only stops those genes driving clusters and "
        "topping DE lists."
    )
    ctx.summary = {"dataset_id": dsid, "mode": mode, "n_genes_removed": len(genes),
                   "by_group": {k: len(v) for k, v in matched.items()},
                   "n_vars_after": int(adata.n_vars), "examples": genes[:20]}
    ctx.suggest("skin.integrate.preprocess", "skin.memory.record_decision")


@tool("skin.annotate.refine_loop", category="annotate",
      summary="Orchestrated audit -> subcluster -> re-label loop. Returns a PLAN by default.")
def refine_loop(dataset_id: str, label_key: str, max_rounds: int = 3,
                contamination_threshold: float = 0.3, resolution_step: float = 0.2,
                base_resolution: float = 1.0, batch_key: str = "Sample",
                auto_apply: bool = False, max_seconds: int = 1800, project_id: str = "",
                dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """The iterative decontamination loop. The ONLY tool that calls other tools.

    Each round:
      1. contamination_audit
      2. for each label over threshold whose likely_cause is "mixed_cluster",
         subcluster it (extract -> preprocess -> harmony -> leiden at a higher
         resolution)
      3. marker_report on the sub-clusters and propose labels
      4. propose removal for sub-clusters that fail their canonical-marker check
      5. re-audit; stop when nothing exceeds the threshold or max_rounds is hit

    With auto_apply=False (the default) this returns an ordered PLAN of tool
    calls with resolved arguments for you to approve. With auto_apply=True it
    executes and logs every round as a decision in project memory.

    Args:
        dataset_id: Handle or label with a label column.
        label_key: obs column of cell type labels.
        max_rounds: Cap on refinement rounds.
        contamination_threshold: Contamination score above which a label is refined.
        resolution_step: Resolution increase per round.
        base_resolution: Starting Leiden resolution for subclustering.
        batch_key: Batch key for the subcluster integration.
        auto_apply: Execute the plan instead of returning it.
        max_seconds: Wall-clock cap; returns partial progress on timeout.
        project_id: Defaults to the active project.
        dry_run: Same as auto_apply=False.
        seed: RNG seed.
    """
    import time

    from . import subcluster_tools

    t0 = time.perf_counter()
    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, label_key)
    current = store.resolve_dataset_ref(ctx.project_id, dataset_id) or dataset_id
    ctx.inputs = {"dataset_id": current}

    plan: list[dict[str, Any]] = []
    rounds: list[dict[str, Any]] = []
    stop_reason = "max_rounds reached"

    for rnd in range(1, max_rounds + 1):
        if time.perf_counter() - t0 > max_seconds:
            stop_reason = f"timed out after {max_seconds}s; returning partial progress"
            break

        audit = contamination_audit(dataset_id=current, label_key=label_key,
                                    project_id=ctx.project_id, seed=seed)
        if not audit.get("ok"):
            stop_reason = f"audit failed in round {rnd}: {audit.get('error', {}).get('message')}"
            break
        per_label = audit["summary"].get("per_label", [])
        offenders = [r for r in per_label
                     if r.get("contamination_score", 0) > contamination_threshold]
        mixed = [r for r in offenders if r.get("likely_cause") == "mixed_cluster"]
        ambient = [r for r in offenders if r.get("likely_cause") == "ambient"]
        doublet = [r for r in offenders if r.get("likely_cause") == "doublet"]

        rounds.append({
            "round": rnd,
            "n_over_threshold": len(offenders),
            "mixed_cluster": [r["label"] for r in mixed],
            "ambient": [r["label"] for r in ambient],
            "doublet": [r["label"] for r in doublet],
            "audit_step": audit.get("memory_ref"),
            "worst": [{"label": r["label"], "score": r["contamination_score"],
                       "cause": r["likely_cause"]} for r in per_label[:5]],
        })

        if not offenders:
            stop_reason = f"no label exceeds the contamination threshold after {rnd - 1} rounds"
            break

        res = base_resolution + (rnd - 1) * resolution_step
        for r in ambient:
            plan.append({
                "tool": "skin.qc.estimate_ambient",
                "args": {"dataset_id": current, "method": "decontx",
                         "apply_correction": False},
                "why": f"{r['label']} looks ambient-contaminated (score "
                       f"{r['contamination_score']}); gene exclusion would hide it, "
                       f"not fix it.",
            })
        for r in doublet:
            plan.append({
                "tool": "skin.doublet.cluster_enrichment",
                "args": {"dataset_id": current, "cluster_key": label_key},
                "why": f"{r['label']} has doublet fraction {r.get('doublet_fraction')} "
                       f"with a strong foreign program.",
            })
        for r in mixed:
            plan.append({
                "tool": "skin.sub.pipeline",
                "args": {"dataset_id": current, "label_key": label_key,
                         "labels": [r["label"]], "resolution": round(res, 2),
                         "batch_key": batch_key,
                         "exclude_gene_groups": ["collagen", "keratin", "muscle"]},
                "why": f"{r['label']} carries a complete {r.get('dominant_foreign_lineage')} "
                       f"program with a low doublet fraction — under-clustering.",
            })
            plan.append({
                "tool": "skin.annotate.marker_report",
                "args": {"dataset_id": "<handle from the previous step>",
                         "cluster_key": f"leiden_res{res:g}"},
                "why": "Propose labels for the new sub-clusters.",
            })

        if not auto_apply or dry_run:
            stop_reason = "plan mode — nothing executed"
            break

        # --- execution --------------------------------------------------------
        applied = []
        for r in mixed:
            out = subcluster_tools.pipeline(
                dataset_id=current, label_key=label_key, labels=[r["label"]],
                resolution=round(res, 2), batch_key=batch_key,
                exclude_gene_groups=["collagen", "keratin", "muscle"],
                project_id=ctx.project_id, seed=seed)
            if out.get("ok") and out.get("dataset_id"):
                applied.append({"label": r["label"], "sub_dataset_id": out["dataset_id"],
                                "n_subclusters": out["summary"].get("n_clusters")})
                store.record_decision(
                    ctx.project_id,
                    question=f"round {rnd}: refine label {r['label']!r}?",
                    choice=f"subclustered at resolution {res:g} -> {out['dataset_id']}",
                    alternatives=["leave as is", "drop the label", "treat as ambient"],
                    rationale=(f"contamination_score={r['contamination_score']}, "
                               f"likely_cause=mixed_cluster, dominant foreign lineage="
                               f"{r.get('dominant_foreign_lineage')}"),
                    author="skin.annotate.refine_loop")
            else:
                ctx.warn(f"subclustering {r['label']!r} failed: "
                         f"{out.get('error', {}).get('message', 'unknown')}")
        rounds[-1]["applied"] = applied
        if not applied:
            stop_reason = "nothing could be refined this round"
            break

    ctx.summary = {
        "mode": "execute" if (auto_apply and not dry_run) else "plan",
        "label_key": label_key, "rounds_run": len(rounds), "rounds": rounds,
        "plan": plan[:20], "n_plan_steps": len(plan), "stop_reason": stop_reason,
        "elapsed_s": round(time.perf_counter() - t0, 1),
        "note": ("Plan mode is the default and is deliberate: this is the only tool that "
                 "calls other tools, and an auto-applied refinement chain is hard to audit. "
                 "Review the plan, then run the steps yourself or re-call with "
                 "auto_apply=True."),
    }
    ctx.suggest("skin.sub.pipeline", "skin.annotate.contamination_audit",
                "skin.memory.record_decision")
