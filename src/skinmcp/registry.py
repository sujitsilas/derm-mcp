"""Handle registry: immutable `dataset_id`s, a lineage graph, and an LRU cache.

The model never receives a matrix. It receives a `ds_xxxxxxxx` handle; the
AnnData behind it lives on disk as gzip `.h5ad` and is memoised in a small LRU.

Handles are content-addressed on ``(parent_id, op, resolved_params)``, so a
retried call — which small models do constantly — returns the existing handle
instead of recomputing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .config import CONFIG
from .errors import InvalidHandle, missing_counts
from .memory import store

if TYPE_CHECKING:  # pragma: no cover - typing only
    from anndata import AnnData

logger = logging.getLogger(__name__)

_LOCK = threading.RLock()
_CACHE: OrderedDict[str, AnnData] = OrderedDict()

#: How X is currently scaled. Stored explicitly in ``uns["skinmcp"]["x_state"]``
#: and never re-detected, because the `X.max() > 50` heuristic silently
#: misclassifies small or heavily filtered objects.
X_STATES = ("counts", "lognorm", "scaled", "unknown")


def make_dataset_id(parent_id: str | None, op: str, params: dict[str, Any]) -> str:
    payload = json.dumps(
        {"parent": parent_id or "", "op": op, "params": params},
        sort_keys=True, default=str, ensure_ascii=False,
    )
    return "ds_" + hashlib.sha1(payload.encode()).hexdigest()[:8]


def object_path(project_id: str, dataset_id: str) -> Path:
    return CONFIG.ensure_project_dirs(project_id) / "objects" / f"{dataset_id}.h5ad"


# --------------------------------------------------------------------------- #
# uns bookkeeping
# --------------------------------------------------------------------------- #

def skinmcp_uns(adata: AnnData) -> dict[str, Any]:
    u = adata.uns.setdefault("skinmcp", {})
    if not isinstance(u, dict):
        u = {}
        adata.uns["skinmcp"] = u
    return u


def get_x_state(adata: AnnData) -> str:
    return str(skinmcp_uns(adata).get("x_state", "unknown"))


def set_x_state(adata: AnnData, state: str) -> None:
    if state not in X_STATES:
        raise ValueError(f"x_state must be one of {X_STATES}, got {state!r}")
    skinmcp_uns(adata)["x_state"] = state


def get_organism(adata: AnnData) -> str:
    return str(skinmcp_uns(adata).get("organism", "unknown"))


def set_organism(adata: AnnData, organism: str) -> None:
    skinmcp_uns(adata)["organism"] = organism


def detect_x_state(adata: AnnData) -> str:
    """Only for freshly ingested objects with no recorded state.

    `X.max()` is a heuristic and we say so: the result is written to
    ``uns`` once and every later read uses the stored value.
    """
    import numpy as np
    import scipy.sparse as sp

    X = adata.X
    if X is None:
        return "unknown"
    sub = X[: min(2000, X.shape[0])]
    arr = sub.data if sp.issparse(sub) else np.asarray(sub)
    if arr.size == 0:
        return "unknown"
    mx, mn = float(np.nanmax(arr)), float(np.nanmin(arr))
    if mn < -0.01:
        return "scaled"
    is_int = bool(np.allclose(arr, np.round(arr), atol=1e-6))
    if is_int and mx > 30:
        return "counts"
    if mx <= 30:
        return "lognorm"
    return "counts"


# --------------------------------------------------------------------------- #
# LRU cache
# --------------------------------------------------------------------------- #

def _nbytes(adata: AnnData) -> int:
    import scipy.sparse as sp

    total = 0
    for m in (adata.X, *adata.layers.values()):
        if m is None:
            continue
        if sp.issparse(m):
            total += int(m.data.nbytes + m.indices.nbytes + m.indptr.nbytes)
        else:
            total += int(getattr(m, "nbytes", 0))
    return total


def _evict_if_needed() -> None:
    while len(_CACHE) > CONFIG.cache_max_objects:
        k, _ = _CACHE.popitem(last=False)
        logger.info("evicting %s from AnnData cache (object count)", k)
    while _CACHE and sum(_nbytes(a) for a in _CACHE.values()) > CONFIG.cache_max_gb * 1e9:
        if len(_CACHE) == 1:
            break  # a single object over the limit still has to be usable
        k, _ = _CACHE.popitem(last=False)
        logger.info("evicting %s from AnnData cache (size)", k)


def cache_put(dataset_id: str, adata: AnnData) -> None:
    with _LOCK:
        _CACHE[dataset_id] = adata
        _CACHE.move_to_end(dataset_id)
        _evict_if_needed()


def cache_drop(dataset_id: str) -> None:
    with _LOCK:
        _CACHE.pop(dataset_id, None)


def cache_clear() -> None:
    with _LOCK:
        _CACHE.clear()


def cache_info() -> dict[str, Any]:
    with _LOCK:
        return {
            "n_cached": len(_CACHE),
            "ids": list(_CACHE.keys()),
            "approx_gb": round(sum(_nbytes(a) for a in _CACHE.values()) / 1e9, 3),
            "max_objects": CONFIG.cache_max_objects,
            "max_gb": CONFIG.cache_max_gb,
        }


# --------------------------------------------------------------------------- #
# load / mint
# --------------------------------------------------------------------------- #

def exists(project_id: str, dataset_id: str) -> bool:
    return store.get_dataset(project_id, dataset_id) is not None and \
        object_path(project_id, dataset_id).exists()


def load(project_id: str, dataset_id: str, *, copy: bool = False) -> AnnData:
    """Resolve a handle (or a human label) to an in-memory AnnData."""
    import anndata as ad

    resolved = store.resolve_dataset_ref(project_id, dataset_id)
    if resolved is None:
        known = [d["dataset_id"] for d in store.list_datasets(project_id)][-10:]
        raise InvalidHandle(
            f"unknown dataset handle {dataset_id!r}",
            remedy=f"Known handles in this project: {known or '(none yet)'}. "
                   "Call skin.memory.brief to see the project state.",
            suggested_tool="skin.memory.brief",
        )
    with _LOCK:
        if resolved in _CACHE:
            _CACHE.move_to_end(resolved)
            a = _CACHE[resolved]
            return a.copy() if copy else a
    path = object_path(project_id, resolved)
    if not path.exists():
        raise InvalidHandle(
            f"{resolved} is registered but its .h5ad is missing at {path}",
            remedy="The project directory may have moved. Re-run the step that produced it, "
                   "or re-load the source data with skin.io.load_h5ad.",
        )
    adata = ad.read_h5ad(path)
    cache_put(resolved, adata)
    return adata.copy() if copy else adata


def mint(
    project_id: str,
    adata: AnnData,
    *,
    parent_id: str | None,
    op: str,
    params: dict[str, Any],
    label: str = "",
    allow_no_counts: bool = False,
    x_state: str | None = None,
) -> str:
    """Persist `adata` under a new deterministic handle and register the lineage.

    Refuses to mint a handle without ``layers['counts']`` unless explicitly
    allowed: every pseudobulk DE and every subclustering re-normalization in this
    server reads from that layer, and losing it silently is unrecoverable.
    """
    if "counts" not in adata.layers and not allow_no_counts:
        raise missing_counts(f"the object produced by {op}")

    dsid = make_dataset_id(parent_id, op, params)
    path = object_path(project_id, dsid)

    if x_state is not None:
        set_x_state(adata, x_state)
    elif get_x_state(adata) == "unknown":
        set_x_state(adata, detect_x_state(adata))

    u = skinmcp_uns(adata)
    u["dataset_id"] = dsid
    u["parent_id"] = parent_id or ""
    u["op"] = op
    u["project_id"] = project_id

    if not path.exists():
        _sanitize_for_h5ad(adata)
        adata.write_h5ad(path, compression="gzip")

    store.upsert_dataset(
        project_id, dsid, parent_id=parent_id, op=op, params=params, path=str(path),
        n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), x_state=get_x_state(adata),
        label=label,
    )
    cache_put(dsid, adata)
    return dsid


def _sanitize_for_h5ad(adata: AnnData) -> None:
    """Make uns/obs writable by h5py.

    Object-dtype obs columns and nested/heterogeneous uns entries are the two
    things that make `write_h5ad` throw after a long computation. Coerce them
    rather than losing the step's work.
    """
    import numpy as np
    import pandas as pd

    for col in list(adata.obs.columns):
        s = adata.obs[col]
        if isinstance(s.dtype, pd.CategoricalDtype):
            continue
        if s.dtype == object:
            adata.obs[col] = s.astype(str)
        elif pd.api.types.is_bool_dtype(s):
            adata.obs[col] = s.astype(np.int8)
    for col in list(adata.var.columns):
        if adata.var[col].dtype == object:
            adata.var[col] = adata.var[col].astype(str)

    def _clean(v: Any, depth: int = 0) -> Any:
        if depth > 4:
            return str(v)
        if isinstance(v, dict):
            return {str(k): _clean(x, depth + 1) for k, x in v.items()}
        if isinstance(v, (list, tuple)):
            if not v:
                return []
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    return [str(x) for x in v]
                return arr
            except (ValueError, TypeError):
                return [str(x) for x in v]
        if isinstance(v, (str, bytes, int, float, bool, np.generic, np.ndarray, pd.DataFrame)):
            return v
        if v is None:
            # anndata round-trips None in uns. Coercing it to "" would corrupt
            # scanpy's own bookkeeping — uns["log1p"]["base"] is None by default,
            # and rank_genes_groups then does np.log("") and dies.
            return None
        return str(v)

    adata.uns = {str(k): _clean(v) for k, v in adata.uns.items()}


# --------------------------------------------------------------------------- #
# lineage
# --------------------------------------------------------------------------- #

def lineage_tree(project_id: str, dataset_id: str = "") -> str:
    """ASCII lineage tree; the whole forest when `dataset_id` is empty."""
    rows = store.list_datasets(project_id)
    if not rows:
        return "(no datasets yet)"
    by_id = {r["dataset_id"]: r for r in rows}
    children: dict[str, list[str]] = {}
    for r in rows:
        children.setdefault(r.get("parent_id") or "", []).append(r["dataset_id"])

    if dataset_id:
        resolved = store.resolve_dataset_ref(project_id, dataset_id)
        if resolved is None:
            raise InvalidHandle(f"unknown dataset handle {dataset_id!r}")
        chain, cur = [], resolved
        while cur:
            chain.append(cur)
            cur = by_id.get(cur, {}).get("parent_id") or ""
        roots = [chain[-1]]
        keep = set(chain)
        stack = [resolved]
        while stack:
            n = stack.pop()
            for c in children.get(n, []):
                keep.add(c)
                stack.append(c)
    else:
        roots = children.get("", [])
        keep = set(by_id)

    out: list[str] = []

    def walk(node: str, prefix: str, last: bool) -> None:
        if node not in keep:
            return
        r = by_id[node]
        lbl = f" [{r['label']}]" if r.get("label") else ""
        conn = "" if not prefix and last else ("└── " if last else "├── ")
        out.append(f"{prefix}{conn}{node}{lbl}  {r['n_obs']}×{r['n_vars']}  "
                   f"op={r.get('op')}  X={r.get('x_state')}")
        kids = [c for c in children.get(node, []) if c in keep]
        newpre = prefix + ("" if not prefix and last else ("    " if last else "│   "))
        for i, c in enumerate(kids):
            walk(c, newpre, i == len(kids) - 1)

    for i, r in enumerate(roots):
        walk(r, "", i == len(roots) - 1)
    return "\n".join(out)
