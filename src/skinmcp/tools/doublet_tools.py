"""`skin.doublet.*` — doublet calling, always per sample.

Two rules this module enforces rather than documents:

1. **Per sample, never pooled.** Doublets are generated within a library.
   Running a caller on the pooled object lets one sample's depth distribution
   set another's threshold. Any argument asking for pooled behaviour is ignored.
2. **Call, then cluster, then decide.** `filter` warns when it runs before any
   clustering on this lineage, because homotypic doublets are invisible to these
   methods and only show up as a cluster carrying two lineage programs — that is
   the annotation loop's job (skin.annotate.contamination_audit), not the
   caller's.
"""

from __future__ import annotations

import logging
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import BadParam, DependencyMissing
from ..memory import store
from ._base import Ctx, confirm_or_raise, require_obs, tool

logger = logging.getLogger(__name__)

METHODS = ("scdblfinder", "scrublet", "doubletdetection")


@tool("skin.doublet.call", category="doublet",
      summary="Call doublets per sample and write scores to obs. Does not filter.")
def call(dataset_id: str, method: str = "scrublet", sample_key: str = "Sample",
         expected_rate: float | None = None, label: str = "", project_id: str = "",
         dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Score every cell for doublet likelihood, one sample at a time.

    Writes `doublet_score` and `predicted_doublet` to obs and mints a new handle.
    Nothing is removed — call skin.cluster.leiden next, check where the calls
    concentrate with skin.doublet.cluster_enrichment, and only then decide.

    Args:
        dataset_id: Handle or label. Needs raw counts.
        method: "scdblfinder" (R bridge, best benchmarked), "scrublet"
            (scanpy's built-in re-implementation, the offline/no-Docker
            fallback), or "doubletdetection".
        sample_key: obs column identifying samples. Calling is always per level.
        expected_rate: Expected doublet fraction. Defaults to the 10x multiplet
            table for each sample's recovered cell count.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan and per-sample expected rates.
        seed: RNG seed.
    """
    import numpy as np

    if method not in METHODS:
        raise BadParam(f"method must be one of {list(METHODS)}, got {method!r}")
    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, sample_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.inputs = {"dataset_id": parent}

    if "counts" not in adata.layers:
        from ..errors import missing_counts

        raise missing_counts(parent or dataset_id)

    samples = list(map(str, adata.obs[sample_key].astype(str).unique()))
    rates = {s: (expected_rate if expected_rate is not None
                 else K.expected_doublet_rate(int((adata.obs[sample_key].astype(str) == s).sum())))
             for s in samples}
    chem = registry.skinmcp_uns(adata).get("chemistry", "10x_3prime_v3")
    if K.platform_rules(chem).get("doublet_calls_unreliable"):
        ctx.warn(f"chemistry={chem}: doublet callers are substantially less reliable on FFPE. "
                 f"Treat these scores as weak evidence and lean on cluster-level enrichment.")

    ctx.code = (
        "import scanpy as sc, numpy as np\n"
        f"# ALWAYS per sample — pooling lets one library's depth set another's threshold\n"
        f"for s in adata.obs[{sample_key!r}].astype(str).unique():\n"
        f"    m = adata.obs[{sample_key!r}].astype(str) == s\n"
        f"    sub = adata[m].copy(); sub.X = sub.layers['counts'].copy()\n"
        f"    sc.pp.scrublet(sub, expected_doublet_rate=<per-sample 10x rate>, random_state={seed})\n"
        f"    adata.obs.loc[m, 'doublet_score'] = sub.obs['doublet_score'].values\n"
        f"    adata.obs.loc[m, 'predicted_doublet'] = sub.obs['predicted_doublet'].values\n"
    )
    if dry_run:
        ctx.summary = {"method": method, "n_samples": len(samples),
                       "expected_rates": rates, "sample_key": sample_key}
        return

    if method == "scdblfinder":
        per_sample, scores, preds = _call_scdblfinder(adata, sample_key, rates, ctx, seed)
    elif method == "doubletdetection":
        per_sample, scores, preds = _call_doubletdetection(adata, sample_key, rates, ctx, seed)
    else:
        per_sample, scores, preds = _call_scrublet(adata, sample_key, rates, ctx, seed)

    adata.obs["doublet_score"] = scores
    adata.obs["predicted_doublet"] = preds.astype(int)
    adata.obs["doublet_method"] = method

    total_rate = float(np.mean(preds))
    for r in per_sample:
        if r.get("rate") and r["rate"] > 2.5 * r["expected_rate"]:
            ctx.warn(f"sample {r['sample']}: observed doublet rate {r['rate']:.3f} is "
                     f"{r['rate'] / max(r['expected_rate'], 1e-6):.1f}x the expected "
                     f"{r['expected_rate']:.3f}. Either overloading, or a real biological "
                     f"population with an intermediate profile is being flagged.")

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="doublet.call",
                         params={"method": method, "sample_key": sample_key,
                                 "expected_rate": expected_rate, "seed": seed},
                         label=label)
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "method": method, "per_sample": per_sample,
                   "total_rate": round(total_rate, 4),
                   "n_called": int(preds.sum()), "n_cells": int(adata.n_obs)}
    ctx.warn("Doublet calls written, nothing removed. The correct order is call -> cluster -> "
             "check where the calls concentrate -> decide. Homotypic doublets are invisible "
             "to every caller here and only surface as a cluster with two lineage programs.")
    ctx.suggest("skin.integrate.preprocess", "skin.cluster.leiden",
                "skin.doublet.cluster_enrichment")


def _call_scrublet(adata: Any, sample_key: str, rates: dict[str, float], ctx: Ctx,
                   seed: int) -> tuple[list[dict[str, Any]], Any, Any]:
    import numpy as np
    import scanpy as sc

    scores = np.zeros(adata.n_obs, dtype=float)
    preds = np.zeros(adata.n_obs, dtype=bool)
    rows: list[dict[str, Any]] = []
    key = adata.obs[sample_key].astype(str).to_numpy()
    for s, rate in rates.items():
        m = key == s
        n = int(m.sum())
        if n < 50:
            rows.append({"sample": s, "n_cells": n, "n_called": 0, "rate": 0.0,
                         "expected_rate": round(rate, 4), "skipped": "fewer than 50 cells"})
            continue
        sub = adata[m].copy()
        sub.X = sub.layers["counts"].copy()
        try:
            sc.pp.scrublet(sub, expected_doublet_rate=float(rate), random_state=seed,
                           verbose=False)
        except Exception as e:  # noqa: BLE001 - per-sample failure must not kill the run
            ctx.warn(f"scrublet failed on sample {s}: {e}")
            rows.append({"sample": s, "n_cells": n, "error": str(e)[:120]})
            continue
        sc_ = sub.obs["doublet_score"].to_numpy(dtype=float)
        pd_ = sub.obs["predicted_doublet"].to_numpy().astype(bool)
        scores[m] = sc_
        preds[m] = pd_
        rows.append({"sample": s, "n_cells": n, "n_called": int(pd_.sum()),
                     "rate": round(float(pd_.mean()), 4), "expected_rate": round(rate, 4),
                     "threshold": round(float(sub.uns.get("scrublet", {})
                                              .get("threshold", float("nan"))), 4)})
    return rows, scores, preds


def _call_doubletdetection(adata: Any, sample_key: str, rates: dict[str, float], ctx: Ctx,
                           seed: int) -> tuple[list[dict[str, Any]], Any, Any]:
    import numpy as np

    try:
        import doubletdetection as dd
    except ImportError as e:
        raise DependencyMissing(
            "doubletdetection is not installed",
            remedy='Install with `uv pip install "skin-mcp[full]"`, or use '
                   'method="scrublet", which ships with scanpy.',
            suggested_tool="skin.doublet.call",
        ) from e

    scores = np.zeros(adata.n_obs, dtype=float)
    preds = np.zeros(adata.n_obs, dtype=bool)
    rows: list[dict[str, Any]] = []
    key = adata.obs[sample_key].astype(str).to_numpy()
    for s, rate in rates.items():
        m = key == s
        n = int(m.sum())
        if n < 50:
            rows.append({"sample": s, "n_cells": n, "skipped": "fewer than 50 cells"})
            continue
        X = adata[m].layers["counts"]
        clf = dd.BoostClassifier(n_iters=10, standard_scaling=True, random_state=seed)
        lab = clf.fit(X).predict()
        sco = clf.doublet_score()
        scores[m] = np.nan_to_num(sco)
        preds[m] = np.nan_to_num(lab).astype(bool)
        rows.append({"sample": s, "n_cells": n, "n_called": int(preds[m].sum()),
                     "rate": round(float(preds[m].mean()), 4), "expected_rate": round(rate, 4)})
    return rows, scores, preds


def _call_scdblfinder(adata: Any, sample_key: str, rates: dict[str, float], ctx: Ctx,
                      seed: int) -> tuple[list[dict[str, Any]], Any, Any]:
    import numpy as np

    from ..runtimes.bridge import run_r_script

    res = run_r_script("doublet_scdblfinder", adata=adata, project_id=ctx.project_id,
                       params={"sample_key": sample_key, "seed": seed,
                               "expected_rates": rates},
                       python_fallback="skin.doublet.call")
    scores = np.asarray(res["scores"], dtype=float)
    preds = np.asarray(res["predicted"], dtype=bool)
    if scores.size != adata.n_obs:
        raise BadParam(f"scDblFinder returned {scores.size} scores for {adata.n_obs} cells",
                       remedy="The h5ad round trip may have reordered cells. "
                              'Retry with method="scrublet".')
    return res.get("per_sample", []), scores, preds


@tool("skin.doublet.cluster_enrichment", category="doublet",
      summary="Per-cluster doublet fraction with a binomial test.")
def cluster_enrichment(dataset_id: str, cluster_key: str, score_key: str = "predicted_doublet",
                       project_id: str = "", dry_run: bool = False, seed: int = 0,
                       *, ctx: Ctx) -> None:
    """Which clusters are enriched for doublet calls? This drives the decision.

    A cluster whose doublet fraction is far above the object-wide rate is a
    candidate for removal. One that merely contains a few scattered calls is not.

    Args:
        dataset_id: Handle or label with both cluster and doublet columns.
        cluster_key: obs column of cluster ids.
        score_key: "predicted_doublet" (binary) or "doublet_score" (continuous).
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    import numpy as np
    from scipy.stats import binomtest
    from statsmodels.stats.multitest import multipletests

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, cluster_key)
    require_obs(adata, score_key)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    clusters = list(map(str, adata.obs[cluster_key].astype(str).unique()))
    key = adata.obs[cluster_key].astype(str).to_numpy()
    if score_key == "doublet_score":
        vals = adata.obs[score_key].to_numpy(dtype=float)
        thr = float(np.percentile(vals, 90))
        called = vals > thr
        ctx.warn(f"score_key='doublet_score' has no threshold; used the 90th percentile "
                 f"({thr:.3f}) for the test. Prefer 'predicted_doublet'.")
    else:
        called = adata.obs[score_key].to_numpy().astype(bool)
    base = float(called.mean())

    rows, pvals = [], []
    for c in clusters:
        m = key == c
        n, k = int(m.sum()), int(called[m].sum())
        p = binomtest(k, n, base, alternative="greater").pvalue if n else 1.0
        pvals.append(p)
        rows.append({"cluster": c, "n_cells": n, "n_doublet": k,
                     "fraction": round(k / max(n, 1), 4),
                     "enrichment": round((k / max(n, 1)) / max(base, 1e-9), 2), "p": p})
    if pvals:
        fdr = multipletests(pvals, method="fdr_bh")[1]
        for r, q in zip(rows, fdr):
            r["fdr"] = float(q)
            r.pop("p")
    rows.sort(key=lambda r: -r["enrichment"])

    flagged = [r["cluster"] for r in rows if r.get("fdr", 1) < 0.05 and r["enrichment"] > 2]
    if flagged:
        ctx.warn(f"Clusters {flagged} are significantly doublet-enriched (>2x, FDR<0.05). "
                 f"Before dropping them, check whether they carry two complete lineage "
                 f"programs (heterotypic doublets) or one coherent intermediate program "
                 f"(possibly real biology) — skin.annotate.contamination_audit distinguishes "
                 f"these.")

    ctx.summary = {"cluster_key": cluster_key, "baseline_rate": round(base, 4),
                   "per_cluster": rows[:30], "flagged_clusters": flagged}
    ctx.suggest("skin.annotate.contamination_audit", "skin.doublet.filter",
                "skin.sub.drop_clusters")


@tool("skin.doublet.filter", category="doublet", destructive=True,
      summary="Remove predicted doublets and mint a new handle.")
def filter(dataset_id: str, score_key: str = "predicted_doublet", threshold: float | None = None,
           confirm: bool = False, label: str = "", project_id: str = "", dry_run: bool = False,
           seed: int = 0, *, ctx: Ctx) -> None:
    """Remove doublet-called cells. Do this AFTER clustering, not before.

    Args:
        dataset_id: Handle or label with doublet calls.
        score_key: "predicted_doublet" (binary) or "doublet_score" (needs threshold).
        threshold: Score cutoff when using the continuous score.
        confirm: Required when more than 30% of cells would be removed.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report what would be removed.
        seed: RNG seed.
    """

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    require_obs(adata, score_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    # Warn when nothing on this lineage has ever been clustered.
    steps = store.get_steps(ctx.project_id, limit=500)
    clustered = any(s["tool"] == "skin.cluster.leiden" and s["ok"] for s in steps)
    if not clustered:
        ctx.warn("skin.cluster.leiden has never run in this project. Filtering doublets "
                 "before clustering removes your only chance to see whether the calls "
                 "concentrate in one cluster (heterotypic, remove) or scatter across many "
                 "(threshold too aggressive). The recommended order is "
                 "call -> preprocess -> cluster -> cluster_enrichment -> filter.")

    if score_key == "doublet_score":
        if threshold is None:
            raise BadParam("threshold is required with score_key='doublet_score'",
                           remedy="Pass a cutoff, or use score_key='predicted_doublet'.")
        drop = adata.obs[score_key].to_numpy(dtype=float) > float(threshold)
    else:
        drop = adata.obs[score_key].to_numpy().astype(bool)

    n_drop = int(drop.sum())
    pct = round(100 * n_drop / max(adata.n_obs, 1), 1)
    ctx.code = (f"keep = ~adata.obs[{score_key!r}].astype(bool)\n" if threshold is None
                else f"keep = adata.obs[{score_key!r}] <= {threshold!r}\n") + \
               "adata = adata[keep].copy()\n"
    if dry_run:
        ctx.summary = {"would_remove": n_drop, "pct": pct, "n_after": int(adata.n_obs - n_drop)}
        return
    if pct > 30:
        confirm_or_raise(confirm, dry_run, "skin.doublet.filter",
                         f"This removes {pct}% of cells ({n_drop}). That is far above any "
                         f"plausible multiplet rate — check skin.doublet.cluster_enrichment "
                         f"first.")

    adata = adata[~drop].copy()
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="doublet.filter",
                         params={"score_key": score_key, "threshold": threshold},
                         label=label or "doublet_filtered")
    ctx.dataset_id = dsid
    ctx.summary = {"dataset_id": dsid, "n_removed": n_drop, "pct_removed": pct,
                   "n_obs": int(adata.n_obs)}
    ctx.suggest("skin.integrate.preprocess", "skin.cluster.leiden")
