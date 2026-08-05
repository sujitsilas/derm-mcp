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

#: Sanitisation notes from the most recent `mint` on *this* thread, waiting to be
#: collected by the caller. Thread-local and consumed once: the MCP SDK runs every
#: sync tool in a worker thread, so a module-level attribute here would let one
#: tool call read another's notes, and a tool that never minted would re-report
#: whatever the previous one left behind.
_TLS = threading.local()


def take_mint_notes() -> list[str]:
    """Return and clear the notes left by this thread's last `mint`."""
    notes = getattr(_TLS, "mint_notes", None) or []
    _TLS.mint_notes = []
    return list(notes)

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
    """Bytes held by X and the real layers.

    `adata.layers` carries an entry keyed `None` that *is* X — anndata's way of
    spelling "the default layer". Iterating `.values()` therefore yields X a
    second time, and every cached object was being charged twice for its
    largest matrix. The cache was not unsafe as a result, only wasteful: it
    evicted objects it had room for and paid to re-read them.
    """
    import scipy.sparse as sp

    real_layers = [v for k, v in adata.layers.items() if k is not None]
    total = 0
    for m in (adata.X, *real_layers):
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
    budget = CONFIG.effective_cache_gb() * 1e9
    while _CACHE and sum(_nbytes(a) for a in _CACHE.values()) > budget:
        if len(_CACHE) == 1:
            break  # a single object over the limit still has to be usable
        k, _ = _CACHE.popitem(last=False)
        logger.info("evicting %s from AnnData cache (size, budget %.1f GB)", k, budget / 1e9)


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


#: Extensions that mean "this is a file the user wants loaded", mapped to the
#: tool that loads them.
_LOADERS = {".h5ad": "skin.io.load_h5ad", ".rds": "skin.io.load_seurat_rds",
            ".loom": "skin.io.load_h5ad", ".h5": "skin.io.load_10x",
            ".mtx": "skin.io.load_10x"}


def find_by_op(project_id: str, op: str, params_match: dict[str, Any] | None = None,
               ) -> list[str]:
    """Handles produced by `op`, newest last, optionally filtered on params.

    Lets a tool that needs derived state (a marker table, a DE result) point at
    the handle that actually carries it. Tools mint a NEW handle, so a caller
    that keeps using the pre-tool handle gets "not found" and is told to re-run
    work it already did — the lineage forked and nothing said so.
    """
    out = []
    for d in store.list_datasets(project_id):
        if d.get("op") != op:
            continue
        if params_match:
            try:
                params = json.loads(d.get("params_json") or "{}")
            except ValueError:
                continue
            if any(str(params.get(k)) != str(v) for k, v in params_match.items()):
                continue
        out.append(d["dataset_id"])
    return out


def bad_handle(project_id: str, ref: str) -> InvalidHandle:
    """The error to raise when a dataset ref does not resolve.

    Passing a FILE PATH where a handle belongs is the single most common model
    mistake, and a bare "unknown handle" tells it nothing — it retries the same
    call, or invents a handle. Every resolution site should raise this so the
    reply always names the tool that actually fixes it.
    """
    known = [d["dataset_id"] for d in store.list_datasets(project_id)][-10:]
    suffix = Path(ref).suffix.lower()
    if suffix in _LOADERS or "/" in ref or "\\" in ref:
        loader = _LOADERS.get(suffix, "skin.io.load_h5ad")
        # Say whether the file is actually there. Without this the message reads
        # as "wrong path" and the caller hunts for a better one: one real session
        # cycled three candidate paths through skin.io.set_label seventeen times
        # in a row, starting with the correct one, because nothing told it the
        # path was fine and the *tool* was wrong.
        exists = False
        try:
            exists = Path(ref).expanduser().is_file()
        except OSError:
            pass
        if exists:
            msg = (f"THE FILE EXISTS at {ref} — the path is correct. What is wrong "
                   f"is the tool: this argument takes a dataset handle, not a path.")
            remedy = (f"Call {loader}(path={ref!r}, organism=...) NOW. Do not try other "
                      f"paths; this one is right. That call returns a `dataset_id` like "
                      f"'ds_1a2b3c4d', which is what this argument wants.")
        else:
            msg = f"{ref!r} is a file path, not a dataset handle (and no file is there)"
            remedy = (f"Check the path exists, then call {loader}(path=..., organism=...). "
                      f"It returns a `dataset_id` like 'ds_1a2b3c4d', and every later tool "
                      f"takes THAT, never the path.")
        return InvalidHandle(
            msg, remedy=remedy, suggested_tool=loader,
            details={"given": ref, "file_exists": exists, "known_handles": known},
        )
    hint = (f"Known handles in this project: {known}." if known else
            "No data is loaded yet — start with skin.io.load_h5ad (one file) or "
            "skin.io.build_multisample (several).")
    return InvalidHandle(
        f"unknown dataset handle {ref!r}",
        remedy=hint,
        suggested_tool="skin.memory.brief" if known else "skin.io.load_h5ad",
        details={"given": ref, "known_handles": known},
    )


def estimate_resident_bytes(path: Path) -> int:
    """How much RAM reading this .h5ad will actually cost, read from its header.

    Deliberately measured rather than guessed from the file size: compression
    ratios vary by an order of magnitude between lzf, gzip and none, so a
    size-based rule either refuses files that fit or waves through ones that do
    not. Sums the matrices only (X plus every layer), which dominate — obs/var
    are megabytes beside gigabytes. Returns 0 when the header cannot be read,
    which the caller treats as "unknown, proceed".
    """
    import h5py

    def _elem(g: Any) -> int:
        if isinstance(g, h5py.Dataset):                      # dense
            n = 1
            for d in g.shape:
                n *= int(d)
            return n * g.dtype.itemsize
        if isinstance(g, h5py.Group) and "data" in g:        # csr/csc
            data = g["data"]
            # values + int32 indices + indptr (negligible)
            return int(data.shape[0]) * (data.dtype.itemsize + 4)
        return 0

    total = 0
    try:
        with h5py.File(path, "r") as f:
            if "X" in f:
                total += _elem(f["X"])
            if "layers" in f:
                for k in f["layers"]:
                    total += _elem(f["layers"][k])
    except (OSError, KeyError, AttributeError, TypeError) as e:
        logger.debug("could not size %s: %s", path, e)
        return 0
    return total


def _admit(resolved: str, path: Path) -> None:
    """Refuse a load that cannot fit, rather than being killed attempting it.

    Without this the failure mode is not a MemoryError — it is the OS or a
    native allocator taking the whole process down, which the host reports only
    as "MCP error -32000: Connection closed", losing every result in the
    session. A typed refusal keeps the server answering and tells the caller
    what to do about it.
    """
    from .config import available_ram_gb
    from .errors import wont_fit

    need = estimate_resident_bytes(path)
    free = available_ram_gb() * 1e9
    if need <= 0 or free <= 0:
        return                                   # cannot measure; do not block
    if need > free * 0.5:
        # Reclaim what we control before declaring defeat: cached objects are
        # on disk already, so dropping them costs a re-read, not the analysis.
        logger.info("%s needs %.1f GB with %.1f GB free — clearing the AnnData cache",
                    resolved, need / 1e9, free / 1e9)
        cache_clear()
        free = available_ram_gb() * 1e9
    # read_h5ad needs headroom above the final object for decompression
    # buffers, so refuse well before the object alone would fill memory.
    if need > free * 0.8:
        raise wont_fit(resolved, need / 1e9, free / 1e9)


def admit_external(path: Path) -> None:
    """`_admit` for a file the user pointed at, before it has a handle."""
    _admit(Path(path).name, Path(path))


def guard_dense(adata: AnnData, op: str, *, remedy: str = "", itemsize: int = 4) -> float:
    """Refuse an operation that would densify a matrix too big to hold.

    Returns the projected size in GB so the caller can report it.

    Sparse single-cell matrices are 5-10% dense, so materialising one costs an
    order of magnitude more than it occupies. `sc.pp.scale` is the usual
    culprit: zero-centering cannot be done in place on a sparse matrix, so
    scanpy densifies, and a 78k x 20k object becomes 6.4 GB on top of the
    sparse original, its lognorm copy and `adata.raw`.

    Unlike a segfault, this failure is an OS kill. faulthandler cannot catch
    SIGKILL, so it leaves no traceback and no crash log -- the server simply
    stops mid-call and the host reports a closed connection. There is no way to
    handle it after the fact, which is why it has to be refused beforehand.
    """
    from .config import available_ram_gb
    from .errors import InsufficientMemory

    need = float(adata.n_obs) * float(adata.n_vars) * itemsize
    free = available_ram_gb() * 1e9
    need_gb = need / 1e9
    if free <= 0:
        return need_gb                          # cannot measure; do not block
    if need > free * 0.5:
        cache_clear()
        free = available_ram_gb() * 1e9
    if need > free * 0.7:
        raise InsufficientMemory(
            f"{op} would densify a {adata.n_obs} x {adata.n_vars} matrix, which needs "
            f"about {need_gb:.1f} GB on top of what is already held, and only "
            f"{free / 1e9:.1f} GB is free",
            remedy=remedy or (
                "Do NOT retry — this fails the same way and risks the OS killing the "
                "server mid-analysis, losing the session. Free memory first (unloading "
                "the local LLM is usually the largest single win), or work on a smaller "
                "object: skin.sub.extract on one compartment produces a handle that "
                "fits comfortably."),
            suggested_tool="skin.sub.extract",
            details={"needed_gb": round(need_gb, 2), "free_gb": round(free / 1e9, 2),
                     "n_obs": int(adata.n_obs), "n_vars": int(adata.n_vars)},
        )
    return need_gb


def load(project_id: str, dataset_id: str, *, copy: bool = False) -> AnnData:
    """Resolve a handle (or a human label) to an in-memory AnnData."""
    import anndata as ad

    resolved = store.resolve_dataset_ref(project_id, dataset_id)
    if resolved is None:
        raise bad_handle(project_id, dataset_id)
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
    _admit(resolved, path)
    adata = ad.read_h5ad(path)
    cache_put(resolved, adata)
    return adata.copy() if copy else adata


def peek(project_id: str, dataset_id: str) -> AnnData:
    """Open a handle for *metadata only* — obs, var, shape, keys. Never the matrix.

    `describe` used to go through `load()`, which reads the whole object: 2.7 GB
    for a 78k x 20k file, just to list obs columns. Beside a resident local LLM
    that was enough to kill the server mid-call, and the crash log showed exactly
    that stack (read_h5ad <- registry.load <- io_tools.describe).

    Backed mode reads the same metadata for a fraction of the memory. The result
    is deliberately NOT cached: it holds an open file handle, and callers that
    need the matrix should use `load()`.
    """
    import anndata as ad

    resolved = store.resolve_dataset_ref(project_id, dataset_id)
    if resolved is None:
        raise bad_handle(project_id, dataset_id)
    with _LOCK:
        cached = _CACHE.get(resolved)
    if cached is not None:
        return cached                      # already in memory; nothing to save
    path = object_path(project_id, resolved)
    if not path.exists():
        raise InvalidHandle(
            f"{resolved} is registered but its .h5ad is missing at {path}",
            remedy="The project directory may have moved. Re-run the step that "
                   "produced it, or re-load with skin.io.load_h5ad.")
    try:
        return ad.read_h5ad(path, backed="r")
    except (OSError, ValueError, TypeError):
        # Some objects cannot be opened backed (odd dtypes, older writers).
        # Correctness beats the memory saving.
        return load(project_id, resolved)


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

    # Written every time, not only when the path is new. The handle is
    # content-addressed on (parent, op, params) — which does not include the
    # *code*, so a fixed tool recomputes a different object under the same id.
    # Skipping the write then left the freshly computed object in the cache
    # while the stale one stayed on disk: same handle, two different answers
    # depending on whether the server had restarted. That is exactly the
    # irreproducibility the registry exists to prevent, and it is silent.
    #
    # The write is not the expensive part — by the time we are here the work has
    # already been redone, so re-serialising costs a fraction of a second (lzf)
    # against a guarantee that a handle means one thing.
    notes = _sanitize_for_h5ad(adata)
    tmp = path.with_suffix(".h5ad.tmp")
    try:
        adata.write_h5ad(tmp, compression=CONFIG.h5ad_compression)
        tmp.replace(path)
    except Exception:
        # Never leave a half-written .h5ad behind: the handle would be
        # registered but unreadable, and every later load would fail with a
        # confusing error far from the cause. Writing to a temporary first also
        # means an interrupted rewrite cannot destroy a good existing object.
        tmp.unlink(missing_ok=True)
        raise
    _TLS.mint_notes = notes

    store.upsert_dataset(
        project_id, dsid, parent_id=parent_id, op=op, params=params, path=str(path),
        n_obs=int(adata.n_obs), n_vars=int(adata.n_vars), x_state=get_x_state(adata),
        label=label,
    )
    cache_put(dsid, adata)
    return dsid


#: h5py treats "/" as a group separator, so it can never appear in a key. Cell
#: type labels routinely contain one — "MΦ-Res/Rep", "MΦ-IFN/AS DCs" — and
#: scanpy stores rank_genes_groups results as structured arrays whose FIELD
#: NAMES are those labels. Writing such an object raises
#: "Forward slashes are not allowed in keys". We rename rather than drop, and
#: record the mapping in uns["skinmcp"]["h5ad_key_renames"] so it stays
#: traceable and reversible.
_SLASH_SUB = "∕"  # U+2215 DIVISION SLASH — renders like "/", legal as a key


def _safe_key(k: str, taken: set[str]) -> str:
    out = str(k).replace("/", _SLASH_SUB)
    if out in taken:
        i = 2
        while f"{out}_{i}" in taken:
            i += 1
        out = f"{out}_{i}"
    return out


def _deslash(obj: Any, renames: dict[str, str], path: str = "uns") -> Any:
    """Recursively rename dict keys, DataFrame columns and structured-array
    fields containing '/'. All three become h5py keys on write."""
    import numpy as np
    import pandas as pd

    if isinstance(obj, pd.DataFrame):
        bad = [c for c in obj.columns if "/" in str(c)]
        if bad:
            mapping: dict[Any, str] = {}
            taken = {str(c) for c in obj.columns if "/" not in str(c)}
            for c in bad:
                nc = _safe_key(str(c), taken)
                taken.add(nc)
                mapping[c] = nc
                renames[f"{path}.{c}"] = nc
            obj = obj.rename(columns=mapping)
        return obj
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for k, v in obj.items():
            nk = str(k)
            if "/" in nk:
                nk = _safe_key(nk, set(out))
                renames[f"{path}[{k!r}]"] = nk
            out[nk] = _deslash(v, renames, f"{path}[{nk!r}]")
        return out
    if isinstance(obj, np.ndarray) and obj.dtype.names:
        names = list(obj.dtype.names)
        if any("/" in n for n in names):
            new: list[str] = []
            for n in names:
                nn = _safe_key(n, set(new)) if "/" in n else n
                if nn != n:
                    renames[f"{path}.{n}"] = nn
                new.append(nn)
            # Build a new array rather than reassigning .dtype in place: the
            # `names` field of a rank_genes_groups record is object dtype, and
            # numpy refuses to reinterpret arrays holding references.
            src = obj
            obj = np.empty(src.shape,
                           dtype=np.dtype([(nn, src.dtype[o])
                                           for nn, o in zip(new, names)]))
            for nn, o in zip(new, names):
                obj[nn] = src[o]
        return obj
    if isinstance(obj, (list, tuple)):
        return type(obj)(_deslash(v, renames, path) for v in obj)
    return obj


def _sanitize_for_h5ad(adata: AnnData) -> list[str]:
    """Make uns/obs writable by h5py. Returns human-readable notes about fixes.

    Object-dtype obs columns and nested/heterogeneous uns entries are the two
    things that make `write_h5ad` throw after a long computation. Coerce them
    rather than losing the step's work.
    """
    import numpy as np
    import pandas as pd

    notes: list[str] = []

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

    # Rename any remaining "/"-bearing keys/fields. Done last so it also covers
    # anything _clean produced.
    renames: dict[str, str] = {}
    adata.uns = _deslash(adata.uns, renames)
    if renames:
        # Stored as a JSON STRING, not a dict: the map is keyed by the ORIGINAL
        # paths, which contain the very slashes we are working around — writing
        # it as a dict would fail for exactly the same reason.
        import json as _json

        skinmcp_uns(adata)["h5ad_key_renames"] = _json.dumps(renames, ensure_ascii=False)
        where = sorted({r.split("[")[1].split("]")[0].strip("'\"")
                        if "[" in r else r.split(".")[0] for r in renames})
        notes.append(
            f"{len(renames)} uns keys/fields contained '/' and were renamed with "
            f"U+2215 so the object could be written (h5py forbids '/' in keys). "
            f"Affected: {where[:4]}. Cell labels in obs are UNCHANGED; the mapping "
            f"is in uns['skinmcp']['h5ad_key_renames'].")
    return notes


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
