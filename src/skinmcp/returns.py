"""The `ToolResult` envelope, summarizers, and the return-size budget.

Two rules that the rest of the server depends on:

1. The model never sees a matrix. Anything bigger than a handful of scalars goes
   to disk and comes back as a resource URI.
2. Every return is hard-capped at ``CONFIG.max_return_bytes``. When a summary is
   too big it is spilled to ``{project}/returns/{step_id}.json`` and replaced by
   a one-line pointer, so a 32k-context caller can't be flooded by one call.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

from .config import CONFIG

ArtifactKind = Literal["figure", "table", "h5ad", "text", "notebook", "model", "log", "bundle"]


class Artifact(BaseModel):
    id: str
    kind: ArtifactKind
    path: str
    uri: str = ""
    caption: str = ""


class ToolResult(BaseModel):
    ok: bool = True
    dataset_id: str | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    artifacts: list[Artifact] = Field(default_factory=list)
    code: str = ""
    memory_ref: str = ""
    next_suggested_tools: list[str] = Field(default_factory=list)
    error: dict[str, Any] | None = None
    #: Set when the payload was spilled to disk by the size budget.
    truncated: bool = False


# --------------------------------------------------------------------------- #
# JSON coercion
# --------------------------------------------------------------------------- #

def jsonable(obj: Any, *, float_digits: int = 4) -> Any:
    """Coerce numpy/pandas scalars and containers into plain JSON types.

    NaN/Inf become None rather than the bare literals `NaN`/`Infinity`, which are
    invalid JSON and which several MCP clients reject outright.
    """
    import numpy as np

    if obj is None or isinstance(obj, (str, bool)):
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        if math.isnan(f) or math.isinf(f):
            return None
        return round(f, float_digits)
    if isinstance(obj, np.ndarray):
        return [jsonable(x, float_digits=float_digits) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): jsonable(v, float_digits=float_digits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [jsonable(x, float_digits=float_digits) for x in obj]
    if isinstance(obj, Path):
        return str(obj)
    # pandas objects, categoricals, timestamps, anything else with a sane repr
    for attr in ("item", "isoformat"):
        fn = getattr(obj, attr, None)
        if callable(fn):
            try:
                return jsonable(fn(), float_digits=float_digits)
            except Exception:  # noqa: BLE001 - best-effort coercion
                pass
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            return df_records(obj)
        if isinstance(obj, pd.Series):
            return jsonable(obj.to_dict(), float_digits=float_digits)
    except ImportError:
        pass
    return str(obj)


def df_records(df: Any, limit: int = 20, float_digits: int = 4) -> list[dict[str, Any]]:
    """First `limit` rows of a DataFrame as JSON records. Never the whole table."""
    head = df.head(limit)
    return [jsonable(r, float_digits=float_digits) for r in head.to_dict(orient="records")]


def top_n(mapping: dict[str, Any], n: int = 20) -> dict[str, Any]:
    """Truncate a dict to `n` entries, recording how many were dropped."""
    items = list(mapping.items())
    if len(items) <= n:
        return jsonable(dict(items))
    out = jsonable(dict(items[:n]))
    out["_truncated"] = f"{len(items) - n} more entries omitted"
    return out


def cat_summary(series: Any, max_levels: int = 20) -> dict[str, Any]:
    """Value counts for a categorical obs column, capped at `max_levels`."""
    vc = series.astype(str).value_counts()
    d: dict[str, Any] = {"n_levels": int(vc.size), "counts": top_n(vc.to_dict(), max_levels)}
    return d


# --------------------------------------------------------------------------- #
# Size budget
# --------------------------------------------------------------------------- #

def _size(payload: dict[str, Any]) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode())


def _shrink_in_place(obj: Any, list_cap: int) -> Any:
    """Recursively cap list lengths and string lengths."""
    if isinstance(obj, list):
        capped = [_shrink_in_place(x, list_cap) for x in obj[:list_cap]]
        if len(obj) > list_cap:
            capped.append(f"...{len(obj) - list_cap} more")
        return capped
    if isinstance(obj, dict):
        return {k: _shrink_in_place(v, list_cap) for k, v in obj.items()}
    if isinstance(obj, str) and len(obj) > 600:
        return obj[:600] + "…[truncated]"
    return obj


def enforce_budget(
    payload: dict[str, Any],
    *,
    project_id: str | None,
    step_id: int | None,
    max_bytes: int | None = None,
) -> dict[str, Any]:
    """Shrink `payload` to fit the return budget, spilling the overflow to disk.

    Degrades in three stages so the model keeps as much signal as possible:
    progressively tighter list caps, then dropping `code` to a resource, then
    replacing `summary` wholesale with a pointer to the spilled JSON.
    """
    limit = max_bytes or CONFIG.max_return_bytes
    if _size(payload) <= limit:
        return payload

    full = json.dumps(payload, ensure_ascii=False, indent=2, default=str)
    spill_uri = ""
    if project_id:
        try:
            d = CONFIG.ensure_project_dirs(project_id) / "returns"
            name = f"step_{step_id:05d}.json" if step_id is not None else "last_return.json"
            (d / name).write_text(full, encoding="utf-8")
            spill_uri = f"skin://project/{project_id}/return/{name}"
        except Exception:  # noqa: BLE001 - spilling is best-effort, never fatal
            spill_uri = ""

    for cap in (20, 10, 5, 3):
        payload = _shrink_in_place(payload, cap)
        if _size(payload) <= limit:
            payload["truncated"] = True
            if spill_uri:
                payload.setdefault("warnings", []).append(f"Full return spilled to {spill_uri}")
            return payload

    code = payload.get("code", "")
    if code:
        payload["code"] = code[:400] + "\n# …full code in the exported notebook"
        if _size(payload) <= limit:
            payload["truncated"] = True
            if spill_uri:
                payload.setdefault("warnings", []).append(f"Full return spilled to {spill_uri}")
            return payload

    keep = {
        "ok": payload.get("ok", True),
        "dataset_id": payload.get("dataset_id"),
        "summary": {"note": "Return exceeded the size budget.", "full_result": spill_uri},
        "warnings": (payload.get("warnings") or [])[:3],
        "artifacts": (payload.get("artifacts") or [])[:5],
        "code": "",
        "memory_ref": payload.get("memory_ref", ""),
        "next_suggested_tools": (payload.get("next_suggested_tools") or [])[:4],
        "truncated": True,
    }
    return keep
