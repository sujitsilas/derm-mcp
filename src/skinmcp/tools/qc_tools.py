"""`skin.qc.*` — per-sample QC statistics, threshold discovery, filtering.

`sample_stats` is threshold **discovery**, not filtering. Nothing in this module
removes a cell until `apply_filters` is called with an explicit threshold dict,
and `apply_filters` refuses to delete more than 30% of cells without
confirmation.

The skin-specific parts that matter: keratin/collagen ambient probes (the
dominant failure mode in enzymatically dissociated skin), probe-based chemistry
handling (Flex has no usable mitochondrial signal), and the neutrophil trap
(a cohort-wide min_genes of 500 silently deletes wound neutrophils).
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .. import knowledge as K
from .. import registry
from ..errors import BadParam, NotFound
from ..memory import store
from ._base import Ctx, confirm_or_raise, require_obs, tool

logger = logging.getLogger(__name__)

CORE_METRICS = ("n_genes_by_counts", "total_counts", "pct_counts_mt", "complexity")


def _metrics_code(organism: str) -> str:
    """Emitted preamble that recreates the per-cell QC columns.

    Every QC tool that reads `n_genes_by_counts` has to emit this: the columns
    are computed on the in-memory object and the exported notebook reloads the
    handle from disk, where they are not persisted. Without it the notebook
    dies on the first filter cell with a KeyError.
    """
    pats = K.qc_patterns(organism)
    return (
        "import scanpy as sc, numpy as np\n"
        "if 'n_genes_by_counts' not in adata.obs:\n"
        f"    adata.var['mt']   = adata.var_names.str.match(r{pats['mito']!r})\n"
        f"    adata.var['ribo'] = adata.var_names.str.match(r{pats['ribo']!r})\n"
        f"    adata.var['hb']   = adata.var_names.str.match(r{pats['hb']!r})\n"
        "    sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'],\n"
        "                               percent_top=[20,50], log1p=False, inplace=True,\n"
        "                               layer='counts')\n"
        "    adata.obs['complexity'] = (np.log10(adata.obs['n_genes_by_counts'] + 1) /\n"
        "                               np.log10(adata.obs['total_counts'] + 1))\n"
    )


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _pattern_fraction(adata: Any, pattern: str) -> Any:
    """Per-cell fraction of counts in genes matching `pattern`."""
    import numpy as np
    import scipy.sparse as sp

    rx = re.compile(pattern)
    mask = np.array([bool(rx.match(str(g))) for g in adata.var_names])
    X = adata.layers.get("counts", adata.X)
    tot = np.asarray(X.sum(1)).ravel() if sp.issparse(X) else np.asarray(X).sum(1)
    tot = np.where(tot == 0, 1.0, tot)
    if not mask.any():
        return np.zeros(adata.n_obs), 0
    sub = X[:, mask]
    s = np.asarray(sub.sum(1)).ravel() if sp.issparse(sub) else np.asarray(sub).sum(1)
    return s / tot, int(mask.sum())


def compute_cell_metrics(adata: Any, organism: str) -> dict[str, int]:
    """Populate the per-cell QC columns in obs. Idempotent."""
    import numpy as np
    import scanpy as sc

    pats = K.qc_patterns(organism)
    counts_found: dict[str, int] = {}
    for key, pat in (("mt", pats["mito"]), ("ribo", pats["ribo"]), ("hb", pats["hb"])):
        rx = re.compile(pat)
        m = np.array([bool(rx.match(str(g))) for g in adata.var_names])
        adata.var[key] = m
        counts_found[key] = int(m.sum())

    use_counts = "counts" in adata.layers
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt", "ribo", "hb"], percent_top=[20, 50], log1p=False,
        inplace=True, layer="counts" if use_counts else None,
    )
    # scater-style complexity: how many genes per unit of sequencing depth.
    with np.errstate(divide="ignore", invalid="ignore"):
        comp = np.log10(adata.obs["n_genes_by_counts"].to_numpy() + 1) / \
               np.log10(adata.obs["total_counts"].to_numpy() + 1)
    adata.obs["complexity"] = np.nan_to_num(comp, nan=0.0, posinf=0.0, neginf=0.0)

    for name, pat in (("frac_keratin", pats["ambient_keratin"]),
                      ("frac_collagen", pats["ambient_collagen"]),
                      ("frac_cornified", pats["ambient_cornified"])):
        frac, n = _pattern_fraction(adata, pat)
        adata.obs[name] = frac
        counts_found[name] = n
    return counts_found


def _mad_bounds(x: Any, n_mads: float, log: bool = True) -> tuple[float, float, float, float]:
    """scater-style outlier bounds on the log1p scale, returned on the raw scale."""
    import numpy as np

    v = np.asarray(x, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (float("nan"),) * 4
    t = np.log1p(v) if log else v
    med = float(np.median(t))
    mad = float(np.median(np.abs(t - med)))
    # 1.4826 rescales MAD to a normal-consistent sigma, as scater does.
    sigma = mad * 1.4826
    lo, hi = med - n_mads * sigma, med + n_mads * sigma
    if log:
        return float(np.expm1(lo)), float(np.expm1(hi)), float(np.expm1(med)), sigma
    return lo, hi, med, sigma


def _sample_row(sub: Any, sample: str, organism: str) -> dict[str, Any]:
    import numpy as np

    o = sub.obs
    q = [1, 5, 25, 50, 75, 95, 99]

    def qs(col: str) -> dict[str, float]:
        v = o[col].to_numpy(dtype=float)
        return {f"q{p}": float(np.percentile(v, p)) for p in q}

    def mad_of(col: str) -> float:
        v = o[col].to_numpy(dtype=float)
        return float(np.median(np.abs(v - np.median(v))))

    n = int(sub.n_obs)
    return {
        "sample": sample,
        "n_cells": n,
        "median_genes": float(o["n_genes_by_counts"].median()),
        "mad_genes": mad_of("n_genes_by_counts"),
        "genes_quantiles": qs("n_genes_by_counts"),
        "median_counts": float(o["total_counts"].median()),
        "mad_counts": mad_of("total_counts"),
        "counts_quantiles": qs("total_counts"),
        "median_pct_mt": float(o["pct_counts_mt"].median()),
        "median_pct_ribo": float(o["pct_counts_ribo"].median()),
        "median_pct_hb": float(o["pct_counts_hb"].median()),
        "median_complexity": float(o["complexity"].median()),
        "median_pct_in_top20": float(o["pct_counts_in_top_20_genes"].median()),
        "median_pct_in_top50": float(o["pct_counts_in_top_50_genes"].median()),
        "median_frac_keratin": float(o["frac_keratin"].median()),
        "median_frac_collagen": float(o["frac_collagen"].median()),
        "median_frac_cornified": float(o["frac_cornified"].median()),
        "expected_doublet_rate": round(K.expected_doublet_rate(n), 4),
    }


def _flag(sample: str, name: str, severity: str, evidence: Any) -> dict[str, Any]:
    return {"sample": sample, "flag": name, "severity": severity, "evidence": evidence}


# --------------------------------------------------------------------------- #
# tools
# --------------------------------------------------------------------------- #

@tool("skin.qc.sample_stats", category="qc",
      summary="Per-sample QC statistics, ambient probes, and quality flags.")
def sample_stats(dataset_id: str, sample_key: str = "Sample", raw_path: str = "",
                 project_id: str = "", dry_run: bool = False, seed: int = 0,
                 *, ctx: Ctx) -> None:
    """Compute per-sample QC statistics and flag problem samples.

    This is threshold DISCOVERY. Nothing is filtered. Call
    skin.qc.recommend_thresholds next, then preview, then apply.

    Reports per sample: cell count, median/MAD/quantiles of genes and counts,
    mito/ribo/haemoglobin fractions, complexity, top-20/50 gene fraction,
    skin-specific keratin/collagen/cornified ambient probes, and the expected
    10x multiplet rate for the recovered cell count.

    Args:
        dataset_id: Handle or label.
        sample_key: obs column identifying samples.
        raw_path: Optional path to a `raw_feature_bc_matrix` for empty-droplet
            profiling. Leave empty to skip.
        project_id: Defaults to the active project.
        dry_run: Return the plan and code without computing.
        seed: RNG seed.
    """
    import numpy as np

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, sample_key)
    organism = registry.get_organism(adata)
    ctx.inputs = {"dataset_id": dataset_id}
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    pats = K.qc_patterns(organism)
    ctx.code = (
        "import scanpy as sc, numpy as np, re\n"
        f"adata.var['mt']   = adata.var_names.str.match(r{pats['mito']!r})\n"
        f"adata.var['ribo'] = adata.var_names.str.match(r{pats['ribo']!r})\n"
        f"adata.var['hb']   = adata.var_names.str.match(r{pats['hb']!r})\n"
        "sc.pp.calculate_qc_metrics(adata, qc_vars=['mt','ribo','hb'], "
        "percent_top=[20,50], log1p=False, inplace=True, layer='counts')\n"
        "adata.obs['complexity'] = (np.log10(adata.obs['n_genes_by_counts']+1) /\n"
        "                           np.log10(adata.obs['total_counts']+1))\n"
        f"# skin ambient probes: keratin {pats['ambient_keratin']!r}, "
        f"collagen {pats['ambient_collagen']!r}\n"
        f"stats = adata.obs.groupby({sample_key!r}).describe()\n"
    )
    if dry_run:
        ctx.summary = {"sample_key": sample_key, "organism": organism,
                       "n_samples": int(adata.obs[sample_key].nunique())}
        return

    found = compute_cell_metrics(adata, organism)
    chemistry = registry.skinmcp_uns(adata).get("chemistry", "10x_3prime_v3")
    rules = K.platform_rules(chemistry)
    thr = K.qc_flag_thresholds()

    samples = list(map(str, adata.obs[sample_key].astype(str).unique()))
    rows = [_sample_row(adata[adata.obs[sample_key].astype(str) == s], s, organism)
            for s in samples]

    # Cohort-level MAD outlier detection on the four core metrics.
    flags: list[dict[str, Any]] = []
    cohort: dict[str, dict[str, float]] = {}
    n_mads = float(thr["outlier_vs_cohort_mads"]["threshold"])
    for metric, field in (("n_genes_by_counts", "median_genes"),
                          ("total_counts", "median_counts"),
                          ("pct_counts_mt", "median_pct_mt"),
                          ("complexity", "median_complexity")):
        vals = np.array([r[field] for r in rows], dtype=float)
        med = float(np.median(vals))
        mad = float(np.median(np.abs(vals - med))) or 1e-9
        cohort[metric] = {"cohort_median": med, "cohort_mad": mad}
        for r, v in zip(rows, vals):
            if abs(v - med) > n_mads * mad * 1.4826:
                flags.append(_flag(r["sample"], "outlier_vs_cohort", "warn",
                                   {"metric": metric, "value": round(float(v), 3),
                                    "cohort_median": round(med, 3),
                                    "n_mads": round(abs(v - med) / (mad * 1.4826), 1)}))

    mito_key = "high_mito_human" if organism == "human" else "high_mito_mouse"
    mito_is_viability = rules.get("mito_is_viability_metric", True)
    for r in rows:
        s = r["sample"]
        if r["n_cells"] < thr["very_low_cell_count"]["threshold"]:
            flags.append(_flag(s, "low_cell_count", "exclude_candidate", {"n_cells": r["n_cells"]}))
        elif r["n_cells"] < thr["low_cell_count"]["threshold"]:
            flags.append(_flag(s, "low_cell_count", "warn", {"n_cells": r["n_cells"]}))
        if r["median_genes"] < thr["low_median_genes"]["threshold"]:
            flags.append(_flag(s, "low_median_genes", "warn",
                               {"median_genes": round(r["median_genes"], 1)}))
        if mito_is_viability and r["median_pct_mt"] > thr[mito_key]["threshold"]:
            flags.append(_flag(s, "high_mito", "warn",
                               {"median_pct_mt": round(r["median_pct_mt"], 2)}))
        if r["median_frac_keratin"] > thr["high_ambient_keratin"]["threshold"]:
            flags.append(_flag(s, "high_ambient_keratin", "warn",
                               {"median_frac_keratin": round(r["median_frac_keratin"], 3),
                                "interpretation": "keratinocyte soup from epidermal "
                                                  "dissociation; consider skin.qc.estimate_ambient"}))
        if r["median_frac_collagen"] > thr["high_ambient_collagen"]["threshold"]:
            flags.append(_flag(s, "high_ambient_collagen", "warn",
                               {"median_frac_collagen": round(r["median_frac_collagen"], 3),
                                "interpretation": "dermal/fibroblast soup"}))
        if r["median_pct_hb"] > thr["high_hemoglobin"]["threshold"] * 100:
            flags.append(_flag(s, "high_hemoglobin", "warn",
                               {"median_pct_hb": round(r["median_pct_hb"], 2)}))
        if r["median_complexity"] < thr["low_complexity"]["threshold"]:
            flags.append(_flag(s, "low_complexity", "warn",
                               {"median_complexity": round(r["median_complexity"], 3)}))
        if r["expected_doublet_rate"] > thr["saturating_doublet_rate"]["threshold"]:
            flags.append(_flag(s, "saturating_doublet_rate", "info",
                               {"expected_rate": r["expected_doublet_rate"],
                                "n_cells": r["n_cells"]}))

    if not mito_is_viability:
        note = rules.get("note", "mitochondrial fraction is not a viability metric for "
                                 "this chemistry")
        ctx.warn(f"chemistry={chemistry}: {note} The high_mito flag was not evaluated.")
    if found.get("mt", 0) == 0:
        ctx.warn(f"No mitochondrial genes matched {pats['mito']!r}. For probe-based Flex "
                 f"assays this is expected. Otherwise check the reference annotation.")

    # Optional empty-droplet profile correlation.
    ambient_corr = None
    if raw_path:
        ambient_corr = _empty_droplet_correlation(raw_path, adata, ctx)

    recommended = _recommend(adata, rows, sample_key, organism, chemistry, "mad", 3.0)

    import pandas as pd

    tbl = ctx.tabledir() / "qc_sample_stats.csv"
    pd.json_normalize(rows).to_csv(tbl, index=False)
    ctx.add_artifact("table", tbl, caption="per-sample QC statistics")

    # Inline rows are compact, and above ~8 samples only the FLAGGED ones are
    # inlined. The 4 KB return budget would otherwise truncate to an arbitrary
    # prefix, which tells a small model nothing; a cohort range plus the problem
    # samples is what it actually needs to pick a threshold. The complete table
    # (quantiles included) is the artifact above.
    def compact(r: dict[str, Any]) -> dict[str, Any]:
        return {"sample": r["sample"], "n_cells": r["n_cells"],
                "median_genes": round(r["median_genes"], 1),
                "median_counts": round(r["median_counts"], 1),
                "median_pct_mt": round(r["median_pct_mt"], 2),
                "complexity": round(r["median_complexity"], 3),
                "frac_keratin": round(r["median_frac_keratin"], 3),
                "frac_collagen": round(r["median_frac_collagen"], 3),
                "exp_doublet_rate": r["expected_doublet_rate"]}

    flagged_names = {f["sample"] for f in flags}
    inline_all = len(rows) <= 8
    shown = rows if inline_all else [r for r in rows if r["sample"] in flagged_names][:8]

    def rng_of(field: str, nd: int = 1) -> dict[str, float]:
        v = np.array([r[field] for r in rows], dtype=float)
        return {"min": round(float(v.min()), nd), "median": round(float(np.median(v)), nd),
                "max": round(float(v.max()), nd)}

    ctx.summary = {
        "n_samples": len(rows),
        "sample_key": sample_key,
        "organism": organism,
        "chemistry": chemistry,
        "genes_matched": found,
        "cohort_range": {
            "n_cells": rng_of("n_cells", 0), "median_genes": rng_of("median_genes"),
            "median_counts": rng_of("median_counts"), "median_pct_mt": rng_of("median_pct_mt", 2),
            "frac_keratin": rng_of("median_frac_keratin", 3),
            "frac_collagen": rng_of("median_frac_collagen", 3),
        },
        "per_sample": [compact(r) for r in shown],
        "per_sample_shown": ("all" if inline_all else "flagged only"),
        "flags": flags[:15],
        "n_flags": len(flags),
        "flagged_samples": sorted(flagged_names),
        "recommended_thresholds": recommended["cohort"],
        "empty_droplet_correlation": ambient_corr,
        "full_table": str(tbl),
    }
    if not inline_all:
        ctx.warn(f"{len(rows)} samples: only flagged samples are inline. Cohort min/median/max "
                 f"is in `cohort_range`; the complete per-sample table with quantiles is at "
                 f"{tbl}.")

    ctx.suggest("skin.qc.recommend_thresholds", "skin.qc.plot_sample_stats",
                "skin.qc.estimate_ambient")


def _empty_droplet_correlation(raw_path: str, adata: Any, ctx: Ctx) -> dict[str, Any] | None:
    """Correlate the ambient (soup) profile with the filtered mean profile."""
    from pathlib import Path

    import numpy as np
    import scanpy as sc
    import scipy.sparse as sp

    p = Path(raw_path).expanduser()
    if not p.exists():
        ctx.warn(f"raw_path {p} does not exist; skipped empty-droplet profiling.")
        return None
    try:
        raw = sc.read_10x_h5(str(p)) if p.suffix == ".h5" else \
            sc.read_10x_mtx(str(p), var_names="gene_symbols", cache=False)
        raw.var_names_make_unique()
    except Exception as e:  # noqa: BLE001 - optional diagnostic, never fatal
        ctx.warn(f"could not read raw matrix at {p}: {e}")
        return None

    tot = np.asarray(raw.X.sum(1)).ravel() if sp.issparse(raw.X) else np.asarray(raw.X).sum(1)
    empty = raw[(tot > 0) & (tot <= 100)]
    if empty.n_obs < 100:
        ctx.warn("fewer than 100 droplets with 1-100 UMIs; empty-droplet profile is unreliable.")
        return None
    shared = [g for g in adata.var_names if g in set(map(str, empty.var_names))]
    soup = np.asarray(empty[:, shared].X.sum(0)).ravel()
    soup = soup / max(soup.sum(), 1)
    Xc = adata[:, shared].layers.get("counts", adata[:, shared].X)
    cell = np.asarray(Xc.sum(0)).ravel()
    cell = cell / max(cell.sum(), 1)
    r = float(np.corrcoef(np.log1p(soup * 1e4), np.log1p(cell * 1e4))[0, 1])
    top_soup = [shared[i] for i in np.argsort(-soup)[:15]]
    return {"n_empty_droplets": int(empty.n_obs), "pearson_r_log": round(r, 3),
            "top_ambient_genes": top_soup,
            "interpretation": ("High correlation plus keratin/collagen at the top of the "
                               "ambient list means the soup is epidermal/dermal debris. "
                               "Fix it at the count level with skin.qc.estimate_ambient; "
                               "dropping those genes from the feature space is not a fix.")}


def _recommend(adata: Any, rows: list[dict[str, Any]], sample_key: str, organism: str,
               chemistry: str, method: str, n_mads: float) -> dict[str, Any]:
    """Build MAD-based and preset-based thresholds, per sample and cohort-wide."""
    import numpy as np

    preset = K.platform_preset(organism, chemistry)
    rules = K.platform_rules(chemistry)
    mito_usable = rules.get("mito_is_viability_metric", True)

    per_sample: dict[str, dict[str, Any]] = {}
    for r in rows:
        s = r["sample"]
        sub = adata.obs[adata.obs[sample_key].astype(str) == s]
        g_lo, g_hi, _, _ = _mad_bounds(sub["n_genes_by_counts"], n_mads)
        c_lo, c_hi, _, _ = _mad_bounds(sub["total_counts"], n_mads)
        m_lo, m_hi, _, _ = _mad_bounds(sub["pct_counts_mt"], n_mads, log=False)
        per_sample[s] = {
            "min_genes": int(max(g_lo, 0)), "max_genes": int(g_hi),
            "min_counts": int(max(c_lo, 0)), "max_counts": int(c_hi),
            "max_pct_mt": (round(float(m_hi), 2) if mito_usable else None),
        }

    def agg(field: str, fn: Any) -> Any:
        vals = [v[field] for v in per_sample.values() if v[field] is not None]
        return fn(vals) if vals else None

    mad_cohort = {
        "min_genes": int(agg("min_genes", np.median) or 0),
        "max_genes": int(agg("max_genes", np.median) or 0),
        "min_counts": int(agg("min_counts", np.median) or 0),
        "max_counts": int(agg("max_counts", np.median) or 0),
        "max_pct_mt": (round(float(agg("max_pct_mt", np.median)), 2) if mito_usable else None),
    }
    if method == "fixed":
        cohort = dict(preset)
    elif method == "both":
        cohort = {
            "min_genes": max(mad_cohort["min_genes"], preset["min_genes"]),
            "max_genes": min(mad_cohort["max_genes"], preset["max_genes"]),
            "min_counts": max(mad_cohort["min_counts"], preset["min_counts"]),
            "max_counts": min(mad_cohort["max_counts"], preset["max_counts"]),
            "max_pct_mt": (min(x for x in (mad_cohort["max_pct_mt"], preset["max_pct_mt"])
                               if x is not None) if mito_usable and preset["max_pct_mt"]
                           is not None and mad_cohort["max_pct_mt"] is not None else
                           (mad_cohort["max_pct_mt"] if mito_usable else None)),
        }
    else:
        cohort = mad_cohort
    if not mito_usable:
        cohort["max_pct_mt"] = None
    return {"per_sample": per_sample, "cohort": cohort, "preset": preset,
            "mad_cohort": mad_cohort}


@tool("skin.qc.recommend_thresholds", category="qc",
      summary="MAD-based (default) or platform-preset filtering thresholds, with rationale.")
def recommend_thresholds(dataset_id: str, sample_key: str = "Sample", method: str = "mad",
                         n_mads: float = 3.0, project_id: str = "", dry_run: bool = False,
                         seed: int = 0, *, ctx: Ctx) -> None:
    """Propose filtering thresholds. MAD is the default and the recommended answer.

    Returns both per-sample and cohort-wide thresholds so you can choose, plus a
    `rationale` string you can paste straight into skin.memory.set_param.

    Fires a `neutrophil_risk` warning whenever the proposed min_genes exceeds
    250 and neutrophil transcripts are detectable in the fraction that would be
    discarded — in wound and burn skin, neutrophils legitimately carry 200-600
    genes and a cohort-wide floor of 500 deletes them silently.

    Args:
        dataset_id: Handle or label.
        sample_key: obs column identifying samples.
        method: "mad" (per-sample, scater-style), "fixed" (platform preset), or
            "both" (the stricter of the two).
        n_mads: How many median absolute deviations count as an outlier.
        project_id: Defaults to the active project.
        dry_run: Return the plan only.
        seed: RNG seed.
    """

    if method not in ("mad", "fixed", "both"):
        raise BadParam(f"method must be mad|fixed|both, got {method!r}")
    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, sample_key)
    organism = registry.get_organism(adata)
    chemistry = registry.skinmcp_uns(adata).get("chemistry", "10x_3prime_v3")
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    ctx.code = (
        _metrics_code(organism)
        + "\n"
        f"def mad_bounds(v, n_mads={n_mads}, log=True):\n"
        "    \"\"\"scater-style outlier bounds; 1.4826 rescales MAD to a normal sigma.\"\"\"\n"
        "    v = np.asarray(v, float); v = v[np.isfinite(v)]\n"
        "    t = np.log1p(v) if log else v\n"
        "    med = np.median(t); sigma = np.median(np.abs(t - med)) * 1.4826\n"
        "    lo, hi = med - n_mads * sigma, med + n_mads * sigma\n"
        "    return (np.expm1(lo), np.expm1(hi)) if log else (lo, hi)\n"
        "\n"
        "thresholds_per_sample = {}\n"
        f"for s in adata.obs[{sample_key!r}].astype(str).unique():\n"
        f"    o = adata.obs[adata.obs[{sample_key!r}].astype(str) == s]\n"
        "    g_lo, g_hi = mad_bounds(o['n_genes_by_counts'])\n"
        "    c_lo, c_hi = mad_bounds(o['total_counts'])\n"
        "    m_lo, m_hi = mad_bounds(o['pct_counts_mt'], log=False)\n"
        "    thresholds_per_sample[s] = dict(min_genes=int(max(g_lo, 0)), "
        "max_genes=int(g_hi),\n"
        "                                    min_counts=int(max(c_lo, 0)), "
        "max_counts=int(c_hi),\n"
        "                                    max_pct_mt=round(float(m_hi), 2))\n"
    )
    if dry_run:
        ctx.summary = {"method": method, "n_mads": n_mads, "chemistry": chemistry}
        return

    if "n_genes_by_counts" not in adata.obs:
        compute_cell_metrics(adata, organism)
    samples = list(map(str, adata.obs[sample_key].astype(str).unique()))
    rows = [_sample_row(adata[adata.obs[sample_key].astype(str) == s], s, organism)
            for s in samples]
    rec = _recommend(adata, rows, sample_key, organism, chemistry, method, n_mads)
    cohort = rec["cohort"]

    rules = K.platform_rules(chemistry)
    if cohort.get("max_pct_mt") is None:
        ctx.warn(f"max_pct_mt is null for chemistry={chemistry}. "
                 f"{rules.get('note', 'The mitochondrial filter is skipped for this chemistry.')} "
                 f"This means SKIP the filter, not use 0.")
    if rules.get("doublet_calls_unreliable"):
        ctx.warn("FFPE: doublet callers are substantially less reliable here. Weight the "
                 "cluster-level doublet enrichment more than the per-cell score.")

    # --- the neutrophil trap ------------------------------------------------- #
    thr = K.qc_flag_thresholds()["neutrophil_risk_min_genes"]["threshold"]
    neut_risk = _neutrophil_risk(adata, cohort, organism, thr)
    if neut_risk["at_risk"]:
        ctx.warn(
            f"neutrophil_risk: the proposed min_genes={cohort['min_genes']} would discard "
            f"{neut_risk['n_discarded']} cells, of which {neut_risk['n_neutrophil_like']} "
            f"({neut_risk['pct_neutrophil_like']}%) express {neut_risk['probe_genes']}. "
            f"Neutrophils in wound/burn skin legitimately carry 200-600 genes. Consider "
            f"min_genes<=200, or filter per cell type after clustering."
        )

    lost = _preview_counts(adata, cohort, sample_key)
    rationale = (
        f"{method} thresholds (n_mads={n_mads}) on {chemistry} {organism} data: "
        f"min_genes={cohort['min_genes']}, max_genes={cohort['max_genes']}, "
        f"min_counts={cohort['min_counts']}, max_counts={cohort['max_counts']}, "
        f"max_pct_mt={cohort['max_pct_mt']}. "
        f"Would remove {lost['total_removed']} / {int(adata.n_obs)} cells "
        f"({lost['pct_removed']}%)."
        + (" NOTE: neutrophil_risk fired; see warnings." if neut_risk["at_risk"] else "")
    )

    # Per-sample thresholds are only inlined for small cohorts; the budget would
    # otherwise evict `cohort`, which is the field the caller actually acts on.
    import pandas as pd

    ptbl = ctx.tabledir() / "qc_recommended_thresholds.csv"
    pd.DataFrame(rec["per_sample"]).T.to_csv(ptbl)
    ctx.add_artifact("table", ptbl, caption=f"per-sample {method} thresholds")

    ctx.summary = {
        "method": method, "n_mads": n_mads, "chemistry": chemistry, "organism": organism,
        "cohort": cohort, "platform_preset": rec["preset"], "mad_cohort": rec["mad_cohort"],
        "preview": {k: v for k, v in lost.items() if k != "per_sample"},
        "neutrophil_risk": neut_risk, "rationale": rationale,
        "per_sample": (rec["per_sample"] if len(rows) <= 6 else None),
        "per_sample_table": str(ptbl),
    }
    ctx.suggest("skin.qc.preview_filters", "skin.memory.set_param", "skin.qc.apply_filters")


def _neutrophil_risk(adata: Any, thresholds: dict[str, Any], organism: str,
                     floor: float) -> dict[str, Any]:
    """Would this min_genes floor delete real neutrophils?"""
    import numpy as np
    import scipy.sparse as sp

    min_genes = thresholds.get("min_genes") or 0
    out = {"at_risk": False, "min_genes": min_genes, "floor": floor}
    if min_genes <= floor:
        return out
    probes = K.present(adata, K.lineages(organism)["Neutrophils"][:6])
    if not probes:
        return {**out, "note": "no neutrophil probe genes present in var_names"}
    discard = (adata.obs["n_genes_by_counts"].to_numpy() < min_genes)
    n_disc = int(discard.sum())
    if n_disc == 0:
        return out

    def _neutrophil_like(mask: Any) -> Any:
        """>=2 probes detected AND they carry >=1% of the cell's counts.

        A single detected S100a8 transcript is ambient, not a neutrophil. The
        two-marker requirement plus a fraction floor is what keeps this warning
        specific enough to act on — otherwise it fires on every cell and gets
        ignored, which is worse than not having it.
        """
        sub = adata[mask, probes]
        Xp = sub.layers.get("counts", sub.X)
        n_det = (np.asarray((Xp > 0).sum(1)).ravel() if sp.issparse(Xp)
                 else (np.asarray(Xp) > 0).sum(1))
        probe_sum = (np.asarray(Xp.sum(1)).ravel() if sp.issparse(Xp)
                     else np.asarray(Xp).sum(1))
        tot = adata.obs["total_counts"].to_numpy()[mask]
        frac = probe_sum / np.maximum(tot, 1)
        return (n_det >= 2) & (frac >= 0.01)

    hit_disc = _neutrophil_like(discard)
    n_neut = int(hit_disc.sum())
    pct = round(100 * n_neut / max(n_disc, 1), 1)
    # Compare against the retained fraction: the warning is about *enrichment* in
    # what you are throwing away, not about neutrophils existing.
    keep = ~discard
    pct_kept = (round(100 * float(_neutrophil_like(keep).mean()), 1)
                if keep.sum() else 0.0)
    return {
        "at_risk": bool(n_neut >= 10 and pct >= 2.0 and pct > 1.5 * max(pct_kept, 0.1)),
        "min_genes": min_genes, "floor": floor, "probe_genes": probes,
        "n_discarded": n_disc, "n_neutrophil_like": n_neut,
        "pct_neutrophil_like": pct, "pct_neutrophil_like_in_kept": pct_kept,
        "criterion": ">=2 neutrophil probes detected and >=1% of counts",
    }


def _preview_counts(adata: Any, thresholds: dict[str, Any], sample_key: str) -> dict[str, Any]:
    import numpy as np

    o = adata.obs
    keep = np.ones(adata.n_obs, dtype=bool)
    reasons: dict[str, int] = {}
    checks = [
        ("min_genes", "n_genes_by_counts", "ge"), ("max_genes", "n_genes_by_counts", "le"),
        ("min_counts", "total_counts", "ge"), ("max_counts", "total_counts", "le"),
        ("max_pct_mt", "pct_counts_mt", "le"),
    ]
    for key, col, op in checks:
        v = thresholds.get(key)
        if v is None or col not in o:
            continue
        m = (o[col].to_numpy() >= v) if op == "ge" else (o[col].to_numpy() <= v)
        reasons[key] = int((~m & keep).sum())
        keep &= m
    per_sample = {}
    if sample_key in o:
        s = o[sample_key].astype(str).to_numpy()
        for name in np.unique(s):
            m = s == name
            per_sample[str(name)] = {"before": int(m.sum()),
                                     "after": int((m & keep).sum()),
                                     "removed": int((m & ~keep).sum())}
    removed = int((~keep).sum())
    return {"n_before": int(adata.n_obs), "n_after": int(keep.sum()),
            "total_removed": removed, "pct_removed": round(100 * removed / max(adata.n_obs, 1), 1),
            "removed_by_criterion": reasons, "per_sample": per_sample, "_keep": keep}


@tool("skin.qc.preview_filters", category="qc",
      summary="Dry-run a threshold set: cells lost per sample and their marker profile.")
def preview_filters(dataset_id: str, thresholds: dict[str, Any], sample_key: str = "Sample",
                    project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
    """Show what a threshold set would remove, and what those cells look like.

    Always call this before apply_filters. The "cells lost by putative lineage"
    table is the part that catches the neutrophil trap and, in FFPE data,
    catches wholesale loss of a low-complexity but real population.

    Args:
        dataset_id: Handle or label.
        thresholds: {"min_genes":..., "max_genes":..., "min_counts":...,
            "max_counts":..., "max_pct_mt":...}. Null/absent keys are skipped.
        sample_key: obs column identifying samples.
        project_id: Defaults to the active project.
        dry_run: No effect; this tool never modifies anything.
        seed: RNG seed.
    """
    import numpy as np
    import scipy.sparse as sp

    adata = registry.load(ctx.project_id, dataset_id)
    organism = registry.get_organism(adata)
    if "n_genes_by_counts" not in adata.obs:
        compute_cell_metrics(adata, organism)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)

    ctx.code = (
        _metrics_code(organism)
        + f"\nth = {thresholds!r}\n"
        "keep = np.ones(adata.n_obs, bool)\n"
        "if th.get('min_genes'): keep &= adata.obs['n_genes_by_counts'] >= th['min_genes']\n"
        "if th.get('max_genes'): keep &= adata.obs['n_genes_by_counts'] <= th['max_genes']\n"
        "if th.get('min_counts'): keep &= adata.obs['total_counts'] >= th['min_counts']\n"
        "if th.get('max_counts'): keep &= adata.obs['total_counts'] <= th['max_counts']\n"
        "if th.get('max_pct_mt') is not None:\n"
        "    keep &= adata.obs['pct_counts_mt'] <= th['max_pct_mt']\n"
        "# preview only — nothing is removed here\n"
    )
    prev = _preview_counts(adata, thresholds, sample_key)
    keep = prev.pop("_keep")
    lost = ~keep

    # Lineage profile of the discarded fraction: score each lineage's marker set
    # on the lost cells and compare with the kept cells.
    lineage_profile: list[dict[str, Any]] = []
    if lost.sum() > 0:
        for name, genes in K.lineages(organism).items():
            g = K.present(adata, genes)
            if len(g) < 3:
                continue
            X = adata[:, g].layers.get("counts", adata[:, g].X)
            det = (np.asarray((X > 0).sum(1)).ravel() if sp.issparse(X)
                   else (np.asarray(X) > 0).sum(1))
            frac_lost = float((det[lost] >= 2).mean())
            frac_kept = float((det[keep] >= 2).mean()) if keep.sum() else 0.0
            if frac_lost > 0.02:
                lineage_profile.append({
                    "lineage": name,
                    "pct_of_lost_cells": round(100 * frac_lost, 1),
                    "pct_of_kept_cells": round(100 * frac_kept, 1),
                    "enrichment_in_lost": round(frac_lost / max(frac_kept, 1e-6), 2),
                })
        lineage_profile.sort(key=lambda d: -d["enrichment_in_lost"])

    neut = _neutrophil_risk(adata, thresholds, organism,
                            K.qc_flag_thresholds()["neutrophil_risk_min_genes"]["threshold"])
    if neut.get("at_risk"):
        ctx.warn(f"neutrophil_risk: {neut['n_neutrophil_like']} of {neut['n_discarded']} "
                 f"discarded cells express {neut['probe_genes']}. Lower min_genes or filter "
                 f"per cell type after clustering.")
    for d in lineage_profile[:3]:
        if d["enrichment_in_lost"] > 2.0:
            ctx.warn(f"'{d['lineage']}' is {d['enrichment_in_lost']}x enriched in the discarded "
                     f"fraction ({d['pct_of_lost_cells']}% of lost cells). Verify this is "
                     f"debris and not a real low-complexity population.")
    if prev["pct_removed"] > 30:
        ctx.warn(f"These thresholds remove {prev['pct_removed']}% of cells. apply_filters "
                 f"will require confirm=True above 30%.")

    ctx.summary = {**prev, "thresholds": thresholds,
                   "lost_by_lineage": lineage_profile[:12], "neutrophil_risk": neut}
    ctx.suggest("skin.qc.apply_filters", "skin.qc.recommend_thresholds")


@tool("skin.qc.apply_filters", category="qc", destructive=True,
      summary="Apply thresholds and mint a new filtered handle.")
def apply_filters(dataset_id: str, thresholds: dict[str, Any], sample_key: str = "Sample",
                  exclude_samples: list[str] | None = None, min_cells_per_gene: int = 3,
                  confirm: bool = False, label: str = "", project_id: str = "",
                  dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Filter cells and mint a new handle. The input handle is never modified.

    Requires confirm=True when more than 30% of cells would be removed.

    Args:
        dataset_id: Handle or label.
        thresholds: Same shape as preview_filters.
        sample_key: obs column identifying samples.
        exclude_samples: Sample names to drop entirely (the exclude_candidate flags).
        min_cells_per_gene: Drop genes detected in fewer cells than this. 0 disables.
        confirm: Required when cell loss exceeds 30%.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report what would happen; mint nothing.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    if "n_genes_by_counts" not in adata.obs:
        compute_cell_metrics(adata, organism)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.inputs = {"dataset_id": parent}

    excl = list(exclude_samples or [])
    if excl:
        require_obs(adata, sample_key)
        known = set(adata.obs[sample_key].astype(str))
        unknown = [s for s in excl if s not in known]
        if unknown:
            raise NotFound(f"exclude_samples not present: {unknown}",
                           remedy=f"Known samples: {sorted(known)}")

    prev = _preview_counts(adata, thresholds, sample_key)
    keep = prev.pop("_keep")
    if excl:
        import numpy as np

        keep &= ~np.isin(adata.obs[sample_key].astype(str).to_numpy(), excl)
    n_after = int(keep.sum())
    pct_removed = round(100 * (adata.n_obs - n_after) / max(adata.n_obs, 1), 1)

    ctx.code = (
        _metrics_code(organism)
        + "\n"
        f"th = {thresholds!r}\n"
        "keep = np.ones(adata.n_obs, bool)\n"
        "if th.get('min_genes'): keep &= adata.obs['n_genes_by_counts'] >= th['min_genes']\n"
        "if th.get('max_genes'): keep &= adata.obs['n_genes_by_counts'] <= th['max_genes']\n"
        "if th.get('min_counts'): keep &= adata.obs['total_counts'] >= th['min_counts']\n"
        "if th.get('max_counts'): keep &= adata.obs['total_counts'] <= th['max_counts']\n"
        "if th.get('max_pct_mt') is not None: keep &= adata.obs['pct_counts_mt'] <= th['max_pct_mt']\n"
        + (f"keep &= ~adata.obs[{sample_key!r}].astype(str).isin({excl!r})\n" if excl else "")
        + "adata = adata[keep].copy()\n"
        + (f"sc.pp.filter_genes(adata, min_cells={min_cells_per_gene})\n"
           if min_cells_per_gene else "")
    )
    if dry_run:
        ctx.summary = {"would_keep": n_after, "pct_removed": pct_removed,
                       "thresholds": thresholds, "exclude_samples": excl}
        return

    if pct_removed > 30:
        confirm_or_raise(confirm, dry_run, "skin.qc.apply_filters",
                         f"These thresholds remove {pct_removed}% of cells "
                         f"({adata.n_obs - n_after} of {adata.n_obs}). Run "
                         f"skin.qc.preview_filters first and check `lost_by_lineage`.")
    if n_after == 0:
        raise BadParam("thresholds would remove every cell",
                       remedy="Run skin.qc.recommend_thresholds to get a sane starting point.")

    adata = adata[keep].copy()
    n_genes_before = int(adata.n_vars)
    if min_cells_per_gene:
        sc.pp.filter_genes(adata, min_cells=min_cells_per_gene)

    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="qc.apply_filters",
                         params={"thresholds": thresholds, "exclude_samples": excl,
                                 "min_cells_per_gene": min_cells_per_gene},
                         label=label or "filtered")
    ctx.dataset_id = dsid
    ctx.summary = {
        "dataset_id": dsid, "parent_id": parent,
        "n_obs_before": int(prev["n_before"]), "n_obs_after": int(adata.n_obs),
        "pct_removed": pct_removed, "n_vars_before": n_genes_before,
        "n_vars_after": int(adata.n_vars), "removed_by_criterion": prev["removed_by_criterion"],
        "per_sample": prev["per_sample"], "excluded_samples": excl,
    }
    ctx.suggest("skin.doublet.call", "skin.integrate.preprocess", "skin.memory.set_param")


@tool("skin.qc.plot_sample_stats", category="qc",
      summary="Violin/scatter QC grid per sample with threshold lines drawn.")
def plot_sample_stats(dataset_id: str, sample_key: str = "Sample",
                      thresholds: dict[str, Any] | None = None, save_prefix: str = "qc_stats",
                      project_id: str = "", dry_run: bool = False, seed: int = 0,
                      *, ctx: Ctx) -> None:
    """Per-sample QC violins plus the counts-vs-genes scatter, with thresholds overlaid.

    Args:
        dataset_id: Handle or label.
        sample_key: obs column identifying samples.
        thresholds: Optional threshold dict; drawn as dashed lines.
        save_prefix: Output filename stem.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from ..style.rcparams import savefig, style

    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, sample_key)
    organism = registry.get_organism(adata)
    if "n_genes_by_counts" not in adata.obs:
        compute_cell_metrics(adata, organism)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    th = thresholds or {}

    ctx.code = (
        "import scanpy as sc\n"
        f"sc.pl.violin(adata, ['n_genes_by_counts','total_counts','pct_counts_mt'],\n"
        f"             groupby={sample_key!r}, rotation=90, multi_panel=True)\n"
        "sc.pl.scatter(adata, x='total_counts', y='n_genes_by_counts', color='pct_counts_mt')\n"
    )
    if dry_run:
        ctx.summary = {"would_plot": ["violins", "scatter"], "sample_key": sample_key}
        return

    metrics = [("n_genes_by_counts", "Genes / cell", ("min_genes", "max_genes")),
               ("total_counts", "UMIs / cell", ("min_counts", "max_counts")),
               ("pct_counts_mt", "% mitochondrial", (None, "max_pct_mt")),
               ("complexity", "Complexity", (None, None)),
               ("frac_keratin", "Keratin fraction", (None, None)),
               ("frac_collagen", "Collagen fraction", (None, None))]
    samples = list(map(str, adata.obs[sample_key].astype(str).unique()))
    order = sorted(samples)
    with style("standard"):
        fig, axes = plt.subplots(3, 3, figsize=(6 * 3, 4.2 * 3))
        af = axes.flatten()
        for i, (col, lab, (lo_k, hi_k)) in enumerate(metrics):
            ax = af[i]
            data = [adata.obs.loc[adata.obs[sample_key].astype(str) == s, col].to_numpy()
                    for s in order]
            parts = ax.violinplot(data, showextrema=False, widths=0.85)
            for pc in parts["bodies"]:
                pc.set_facecolor("#4E79A7")
                pc.set_alpha(0.65)
            for j, d in enumerate(data, start=1):
                ax.scatter([j], [np.median(d)], color="black", s=14, zorder=4)
            for k, style_ in ((lo_k, "--"), (hi_k, "--")):
                v = th.get(k) if k else None
                if v is not None:
                    ax.axhline(v, color="#C0392B", lw=1.2, ls=style_)
            ax.set_xticks(range(1, len(order) + 1))
            ax.set_xticklabels(order, rotation=90, fontsize=10)
            ax.set_ylabel(lab, fontsize=15, fontweight="bold")
            if col in ("total_counts", "n_genes_by_counts"):
                ax.set_yscale("log")

        # counts vs genes, coloured by %mt
        ax = af[6]
        sc_ = ax.scatter(adata.obs["total_counts"], adata.obs["n_genes_by_counts"],
                         c=adata.obs["pct_counts_mt"], s=2, alpha=0.4, cmap="viridis",
                         rasterized=True)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("UMIs / cell", fontsize=15, fontweight="bold")
        ax.set_ylabel("Genes / cell", fontsize=15, fontweight="bold")
        fig.colorbar(sc_, ax=ax, label="% mito")
        if th.get("min_genes"):
            ax.axhline(th["min_genes"], color="#C0392B", lw=1.2, ls="--")
        if th.get("min_counts"):
            ax.axvline(th["min_counts"], color="#C0392B", lw=1.2, ls="--")

        # cells per sample
        ax = af[7]
        n_per = [int((adata.obs[sample_key].astype(str) == s).sum()) for s in order]
        ax.bar(range(len(order)), n_per, color="#59A14F")
        ax.set_xticks(range(len(order)))
        ax.set_xticklabels(order, rotation=90, fontsize=10)
        ax.set_ylabel("Cells", fontsize=15, fontweight="bold")

        af[8].set_visible(False)
        fig.tight_layout()
        paths = savefig(fig, ctx.figdir("qc") / save_prefix)
        plt.close(fig)

    aid = ctx.add_artifact("figure", paths["pdf"], caption="per-sample QC panel",
                           params={"thresholds": th, "sample_key": sample_key})
    ctx.summary = {"artifact_id": aid, "paths": paths, "n_samples": len(order),
                   "cells_per_sample": dict(zip(order, n_per))}
    ctx.suggest("skin.qc.recommend_thresholds", "skin.qc.preview_filters")


@tool("skin.qc.cell_cycle_score", category="qc",
      summary="Organism-aware S/G2M phase scoring.")
def cell_cycle_score(dataset_id: str, label: str = "", project_id: str = "",
                     dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Score S and G2M phase and assign a phase label, using the organism's gene lists.

    Run this before deciding whether to regress out cell cycle. In wound and burn
    skin, proliferation is often the biology rather than a nuisance — check the
    phase distribution per cluster before regressing anything.

    Args:
        dataset_id: Handle or label. Should be log-normalized.
        label: Human alias for the new handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    import scanpy as sc

    adata = registry.load(ctx.project_id, dataset_id, copy=True)
    organism = registry.get_organism(adata)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    cc = K.cell_cycle(organism)
    s_genes = K.present(adata, cc["s_genes"])
    g2m_genes = K.present(adata, cc["g2m_genes"])

    ctx.code = (
        "import scanpy as sc\n"
        f"s_genes = {s_genes[:8]!r} + ...  # {len(s_genes)} present\n"
        f"g2m_genes = {g2m_genes[:8]!r} + ...  # {len(g2m_genes)} present\n"
        "sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)\n"
    )
    if dry_run:
        ctx.summary = {"n_s_genes": len(s_genes), "n_g2m_genes": len(g2m_genes)}
        return
    if len(s_genes) < 5 or len(g2m_genes) < 5:
        raise BadParam(
            f"only {len(s_genes)} S and {len(g2m_genes)} G2M genes present",
            remedy="The object is probably subset to HVGs or a marker panel. Score cell "
                   "cycle on the full gene space before feature selection.",
        )
    if registry.get_x_state(adata) == "counts":
        ctx.warn("X is raw counts; cell-cycle scores are meant for log-normalized data. "
                 "Run skin.integrate.preprocess first for a meaningful score.")

    sc.tl.score_genes_cell_cycle(adata, s_genes=s_genes, g2m_genes=g2m_genes)
    dsid = registry.mint(ctx.project_id, adata, parent_id=parent, op="qc.cell_cycle_score",
                         params={"n_s": len(s_genes), "n_g2m": len(g2m_genes)},
                         label=label)
    ctx.dataset_id = dsid
    vc = adata.obs["phase"].value_counts().to_dict()
    ctx.summary = {"dataset_id": dsid, "n_s_genes": len(s_genes),
                   "n_g2m_genes": len(g2m_genes), "phase_counts": vc}
    ctx.suggest("skin.integrate.preprocess", "skin.cluster.cluster_qc")


@tool("skin.qc.estimate_ambient", category="qc", needs_r=True,
      summary="Per-sample ambient RNA contamination fraction (SoupX/DecontX).")
def estimate_ambient(dataset_id: str, raw_path: str = "", method: str = "decontx",
                     sample_key: str = "Sample", apply_correction: bool = False,
                     label: str = "", project_id: str = "", dry_run: bool = False,
                     seed: int = 0, *, ctx: Ctx) -> None:
    """Estimate (and optionally remove) ambient RNA at the COUNT level.

    This is the real fix for keratin/collagen soup. Excluding those genes from
    the feature space (skin.annotate.regress_markers) stops them driving
    clusters and DE, but leaves the contaminating counts in the matrix where
    they still distort library sizes, size factors, and the neighbour graph.

    Args:
        dataset_id: Handle or label.
        raw_path: Path to `raw_feature_bc_matrix` for the empty-droplet profile.
            SoupX needs it; DecontX can run without it, less accurately.
        method: "decontx" or "soupx". Both run in the R container.
        sample_key: obs column identifying samples. Estimation is per sample.
        apply_correction: Write the corrected counts into a new handle. Default
            False — inspect the contamination fractions first.
        label: Human alias for the corrected handle.
        project_id: Defaults to the active project.
        dry_run: Report the plan only.
        seed: RNG seed.
    """
    from ..runtimes.bridge import run_r_script

    if method not in ("decontx", "soupx"):
        raise BadParam(f"method must be decontx|soupx, got {method!r}")
    adata = registry.load(ctx.project_id, dataset_id)
    require_obs(adata, sample_key)
    parent = store.resolve_dataset_ref(ctx.project_id, dataset_id)
    ctx.code = (f"# R bridge: {method} per sample on layers['counts']\n"
                f"# runtimes/r/scripts/ambient_{method}.R\n")
    if dry_run:
        ctx.summary = {"method": method, "sample_key": sample_key,
                       "n_samples": int(adata.obs[sample_key].nunique()),
                       "apply_correction": apply_correction}
        return

    result = run_r_script(
        f"ambient_{method}", adata=adata, project_id=ctx.project_id,
        params={"sample_key": sample_key, "raw_path": raw_path,
                "apply_correction": bool(apply_correction), "seed": seed},
        python_fallback="skin.annotate.regress_markers",
    )
    per_sample = result.get("per_sample", [])
    ctx.summary = {"method": method, "per_sample": per_sample,
                   "mean_contamination": result.get("mean_contamination"),
                   "r_log_tail": (result.get("log") or "")[-400:]}

    high = [r for r in per_sample if (r.get("contamination") or 0) > 0.2]
    if high:
        ctx.warn(f"{len(high)} samples have >20% estimated ambient: "
                 f"{[r['sample'] for r in high][:6]}. Correct at the count level before "
                 f"clustering; gene exclusion downstream will not fix library sizes.")

    if apply_correction and result.get("corrected_path"):
        import anndata as ad

        corrected = ad.read_h5ad(result["corrected_path"])
        adata2 = adata.copy()
        adata2.layers["counts_raw"] = adata2.layers["counts"].copy()
        adata2.layers["counts"] = corrected[adata2.obs_names, adata2.var_names].X.copy()
        adata2.X = adata2.layers["counts"].copy()
        registry.set_x_state(adata2, "counts")
        dsid = registry.mint(ctx.project_id, adata2, parent_id=parent,
                             op="qc.estimate_ambient",
                             params={"method": method, "apply_correction": True},
                             label=label or "decontaminated")
        ctx.dataset_id = dsid
        ctx.summary["dataset_id"] = dsid
        ctx.warn("Corrected counts are in layers['counts']; the originals are preserved "
                 "in layers['counts_raw']. Re-run skin.integrate.preprocess from here.")
    ctx.suggest("skin.integrate.preprocess", "skin.annotate.contamination_audit")
