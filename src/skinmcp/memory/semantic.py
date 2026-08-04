"""Semantic memory, backed by mem0.

WHAT LIVES HERE vs IN store.py
------------------------------
mem0 owns everything free-text — annotation rationales, decisions, notes — and
all recall over them. Semantic search is a real upgrade on the FTS5 index it
replaces: "why did we drop that cluster?" now works without guessing the exact
words that were written.

`store.py` keeps the *registry*: dataset handles and lineage, the ordered step
log with its executable code, DE run ids mapped to result tables, artifact ids
mapped to paths. Those are exact-lookup keys, not recollections. If
`get_run("de_d0357f6c")` were answered by nearest-neighbour search, a volcano
could be drawn from a different run's table and the reproducibility test — which
re-executes the exported notebook and compares DE to 1e-4 — would be meaningless.
Vector recall is the wrong tool for a primary key; that is the only reason any
memory remains in SQLite.

RUNNING LOCALLY
---------------
Configured against LM Studio's OpenAI-compatible endpoint by default, so nothing
leaves the machine and no API key is needed. Telemetry is disabled. The vector
store is a per-project Qdrant directory, which ships with mem0 and needs no
server.

If mem0 or its backend is unavailable, recall degrades to a keyword scan over
the same rows and says so. A memory layer must never be the reason a tool call
fails.
"""

from __future__ import annotations

import atexit
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..config import CONFIG

logger = logging.getLogger(__name__)

#: Kinds of free-text memory. Used as mem0 metadata so recall can be filtered.
KINDS = ("annotation", "decision", "note", "parameter")


def _mem0_config(project_id: str) -> dict[str, Any]:
    """Local-only mem0 configuration: LM Studio + on-disk Qdrant."""
    store_dir = CONFIG.project_dir(project_id) / "memory_vectors"
    store_dir.mkdir(parents=True, exist_ok=True)
    return {
        "llm": {
            "provider": CONFIG.mem0_provider,
            "config": {"model": CONFIG.mem0_llm_model,
                       "lmstudio_base_url": CONFIG.mem0_base_url},
        },
        "embedder": {
            "provider": CONFIG.mem0_provider,
            "config": {"model": CONFIG.mem0_embed_model,
                       "lmstudio_base_url": CONFIG.mem0_base_url},
        },
        "vector_store": {
            "provider": "qdrant",
            "config": {"collection_name": f"skinmcp_{project_id}",
                       "path": str(store_dir),
                       "embedding_model_dims": CONFIG.mem0_embed_dims},
        },
    }


@lru_cache(maxsize=8)
def _client(project_id: str) -> Any | None:
    """One mem0 Memory per project, or None if it cannot be constructed."""
    if CONFIG.memory_backend == "sqlite":
        return None
    # mem0 phones home by default; a lab machine analysing patient-adjacent data
    # should not be sending usage events anywhere.
    os.environ.setdefault("MEM0_TELEMETRY", "False")
    os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
    try:
        Memory = _import_memory()
        return Memory.from_config(_mem0_config(project_id))
    except Exception as e:  # noqa: BLE001 - any failure means "fall back"
        logger.warning("mem0 unavailable (%s: %s); semantic recall falls back to "
                       "keyword search", type(e).__name__, e)
        return None


def _import_memory() -> Any:
    """Prefer an installed mem0; otherwise use the copy vendored beside this file."""
    try:
        from mem0 import Memory

        return Memory
    except ImportError:
        import sys

        sys.path.insert(0, str(Path(__file__).parent))
        from mem0 import Memory  # type: ignore[no-redef]

        return Memory


def _close_clients() -> None:
    """Close Qdrant handles before the interpreter tears down.

    qdrant-client's __del__ runs after sys.meta_path is cleared and raises an
    ImportError onto stderr at every exit; harmless, but an MCP server's stderr
    is where real problems are supposed to be visible.
    """
    try:
        _client.cache_clear()
    except Exception:  # noqa: BLE001 - shutdown must not raise
        pass


atexit.register(_close_clients)


def available(project_id: str) -> bool:
    return _client(project_id) is not None


def remember(project_id: str, kind: str, text: str, metadata: dict[str, Any] | None = None,
             author: str = "model") -> str | None:
    """Store one free-text memory. Returns the mem0 id, or None if unavailable.

    `infer=False`: the rationale a user wrote is the record, and an LLM
    paraphrase of it is not. mem0 still embeds it, so recall stays semantic.
    """
    if not text or not text.strip():
        return None
    m = _client(project_id)
    if m is None:
        return None
    meta = {"kind": kind, "project_id": project_id, "author": author,
            **{k: str(v) for k, v in (metadata or {}).items()}}
    try:
        res = m.add(text, user_id=project_id, metadata=meta, infer=False)
        results = res.get("results") if isinstance(res, dict) else None
        return (results[0].get("id") if results else None)
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0 add failed (%s); the row is still in SQLite", e)
        return None


def recall(project_id: str, query: str, limit: int = 15,
           kind: str = "") -> list[dict[str, Any]] | None:
    """Semantic search. Returns None when mem0 is unavailable."""
    m = _client(project_id)
    if m is None:
        return None
    try:
        # mem0 2.x takes the entity scope inside `filters`; passing user_id at
        # the top level raises rather than being ignored.
        filters: dict[str, Any] = {"user_id": project_id}
        if kind:
            filters["kind"] = kind
        res = m.search(query, filters=filters, limit=limit)
        rows = res.get("results", res) if isinstance(res, dict) else res
        return [{"text": r.get("memory", ""),
                 "kind": (r.get("metadata") or {}).get("kind", ""),
                 "score": round(float(r.get("score", 0.0)), 4),
                 "metadata": {k: v for k, v in (r.get("metadata") or {}).items()
                              if k not in ("project_id",)}}
                for r in (rows or [])]
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0 search failed (%s); falling back", e)
        return None


def forget_project(project_id: str) -> None:
    m = _client(project_id)
    if m is None:
        return
    try:
        m.delete_all(user_id=project_id)
    except Exception as e:  # noqa: BLE001
        logger.warning("mem0 delete_all failed: %s", e)


def status(project_id: str) -> dict[str, Any]:
    return {
        "backend": "mem0" if available(project_id) else "sqlite-keyword",
        "configured": {"provider": CONFIG.mem0_provider,
                       "base_url": CONFIG.mem0_base_url,
                       "llm_model": CONFIG.mem0_llm_model,
                       "embed_model": CONFIG.mem0_embed_model},
        "note": ("Free-text memory (rationales, decisions, notes) is semantic. "
                 "Dataset handles, the step log, DE runs and artifacts stay in "
                 "SQLite because they are looked up by exact key."),
    }
