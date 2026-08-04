"""Derive structure from the object, never from expected column names.

An AnnData may be raw or fully analysed, and may use any naming convention. Any
inference done by guessing at names like "Type" or "Timepoint" silently returns
nothing on an object that does not use them, with no way for the caller to tell.
Everything here works from the data instead.

Column *parameters* on tools (`sample_key="Sample"`, `covariates=["Timepoint"]`)
are a different thing and stay as they are: they appear in the tool schema, are
documented, and the caller can override them. It is hidden inference that is the
problem, not visible defaults.
"""

from __future__ import annotations

from typing import Any


def is_categorical(series: Any) -> bool:
    import pandas as pd

    return isinstance(series.dtype, pd.CategoricalDtype) or series.dtype == object


def balance(series: Any) -> float:
    """Normalised entropy of the level distribution. 1.0 = perfectly even."""
    import numpy as np

    vc = series.astype(str).value_counts()
    if vc.size < 2:
        return 0.0
    p = (vc / vc.sum()).to_numpy()
    return float(-(p * np.log(p)).sum() / np.log(vc.size))


def categoricals(adata: Any, max_levels: int = 30) -> dict[str, list[str]]:
    """Categorical obs columns with at most `max_levels` levels, and their levels."""
    out: dict[str, list[str]] = {}
    for c in adata.obs.columns:
        s = adata.obs[c]
        if not is_categorical(s):
            continue
        lv = sorted(map(str, s.astype(str).unique()))
        if len(lv) <= max_levels:
            out[str(c)] = lv
    return out


def groupable(adata: Any, max_levels: int = 30) -> list[str]:
    """Columns usable as a grouping, best first.

    Ordered by level count ascending so low-cardinality factors surface before
    cluster assignments, with evenness as the tie-break so near-constant columns
    sink. No column names are consulted.
    """
    cats = categoricals(adata, max_levels)
    ok = {c: lv for c, lv in cats.items() if len(lv) >= 2}
    return sorted(ok, key=lambda c: (len(ok[c]), -balance(adata.obs[c]), c))


def constant_columns(adata: Any) -> list[str]:
    return [c for c, lv in categoricals(adata, max_levels=1).items() if len(lv) == 1]


def sample_level_columns(adata: Any, sample_key: str, max_levels: int = 50) -> list[str]:
    """Columns that are constant within every level of `sample_key`.

    That is the definition of a sample-level attribute — condition, timepoint,
    batch, sex — and it holds whatever those columns happen to be called.
    """
    if sample_key not in adata.obs.columns:
        return []
    obs = adata.obs
    out = []
    for c in obs.columns:
        if str(c) == str(sample_key) or not is_categorical(obs[c]):
            continue
        n = obs.groupby(sample_key, observed=True)[c].nunique(dropna=False)
        if (n <= 1).all() and obs[c].astype(str).nunique() <= max_levels:
            out.append(str(c))
    return out


def confounding_candidates(adata: Any, batch_key: str, max_levels: int = 12) -> list[str]:
    """Columns worth checking a batch key against.

    Any low-cardinality categorical could be the biological variable; which one
    it is depends on the experiment, not on whether it is spelled "Type".
    """
    return [c for c in groupable(adata, max_levels) if str(c) != str(batch_key)]
