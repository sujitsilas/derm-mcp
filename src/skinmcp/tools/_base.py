"""Tool registration, auto-provenance, and the shared execution contract.

Every tool in this server is a plain Python function decorated with `@tool`.
The decorator is what makes the reproducibility guarantee hold: it writes a
`step` row on *every* call (success or failure), attaches the resolved
parameters and package versions, seeds every RNG, and clamps the return to the
size budget.
"""

from __future__ import annotations

import functools
import inspect
import logging
import random
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..config import CONFIG
from ..errors import BadParam, NoProject, SkinMCPError
from ..memory import store
from ..returns import Artifact, ToolResult, enforce_budget, jsonable
from ..runtimes import python_manifest

logger = logging.getLogger(__name__)


@dataclass
class ToolSpec:
    name: str
    category: str
    fn: Callable[..., Any]
    summary: str
    destructive: bool = False
    needs_network: bool = False
    needs_r: bool = False
    #: False for the two tools that bootstrap or enumerate projects and therefore
    #: cannot require one to already be open.
    needs_project: bool = True


REGISTRY: dict[str, ToolSpec] = {}

#: The project every tool falls back to when `project_id` is omitted. Set by
#: `skin.memory.open_project`. Keeping it server-side is what lets a small model
#: stop threading an id through 40 consecutive calls.
_ACTIVE_PROJECT: str | None = None


def set_active_project(project_id: str) -> None:
    global _ACTIVE_PROJECT
    _ACTIVE_PROJECT = project_id


def get_active_project() -> str | None:
    return _ACTIVE_PROJECT


def resolve_project(project_id: str = "") -> str:
    pid = project_id or _ACTIVE_PROJECT
    if not pid:
        raise NoProject(
            "no project is open",
            remedy="Call skin.memory.open_project(name=..., organism=...) first. It "
                   "creates or re-attaches a project and returns a resume briefing.",
            suggested_tool="skin.memory.open_project",
        )
    if store.get_project(pid) is None:
        raise NoProject(
            f"project {pid!r} does not exist",
            remedy="Call skin.memory.list_projects to see what is available, or "
                   "skin.memory.open_project to create one.",
            suggested_tool="skin.memory.list_projects",
        )
    return pid


# --------------------------------------------------------------------------- #
# The per-call context object tools mutate to build their result
# --------------------------------------------------------------------------- #

@dataclass
class Ctx:
    """Accumulates everything the decorator needs to build a `ToolResult`."""

    tool: str
    project_id: str
    seed: int = 0
    dry_run: bool = False
    dataset_id: str | None = None
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    code: str = ""
    #: False when `code` is an illustrative sketch rather than a runnable block.
    #: The notebook exporter comments those cells out instead of shipping a
    #: notebook that raises NameError on the third cell.
    code_executable: bool = True
    next_tools: list[str] = field(default_factory=list)
    inputs: dict[str, Any] = field(default_factory=dict)
    step_id: int | None = None

    def warn(self, msg: str) -> None:
        if msg not in self.warnings:
            self.warnings.append(msg)

    def suggest(self, *tools: str) -> None:
        for t in tools:
            if t not in self.next_tools:
                self.next_tools.append(t)

    def add_artifact(self, kind: str, path: str | Path, caption: str = "",
                     params: dict[str, Any] | None = None, artifact_id: str = "") -> str:
        aid = artifact_id or store.new_id("art", 8)
        p = str(path)
        uri = f"skin://figure/{aid}" if kind == "figure" else f"skin://artifact/{aid}"
        self.artifacts.append(Artifact(id=aid, kind=kind, path=p, uri=uri, caption=caption))
        store.record_artifact(self.project_id, aid, self.step_id, kind, p, caption, params or {})
        # Sidecar JSON so a figure found on a shared drive months later is traceable.
        if kind in ("figure", "table"):
            try:
                import json as _json
                side = Path(p).with_suffix(".json")
                side.write_text(_json.dumps(jsonable({
                    "artifact_id": aid, "tool": self.tool, "project_id": self.project_id,
                    "dataset_id": self.dataset_id, "caption": caption,
                    "params": params or {}, "created_at": store.now(),
                }), indent=2), encoding="utf-8")
            except OSError:
                pass
        return aid

    def figdir(self, group: str = "") -> Path:
        d = CONFIG.ensure_project_dirs(self.project_id) / "figures" / (group or "misc")
        d.mkdir(parents=True, exist_ok=True)
        return d

    def tabledir(self) -> Path:
        d = CONFIG.ensure_project_dirs(self.project_id) / "tables"
        d.mkdir(parents=True, exist_ok=True)
        return d


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except ImportError:
        pass


def _resolved_params(name: str, sig: inspect.Signature, args: tuple,
                     kwargs: dict) -> dict[str, Any]:
    """Bind the call to the tool's *public* signature and fill in defaults.

    This is what makes the exported notebook reproducible: the step row records
    what the tool actually ran with, not just what the caller happened to pass.
    `sig` excludes `ctx`, which the decorator injects.
    """
    try:
        bound = sig.bind(*args, **kwargs)
    except TypeError as e:
        raise BadParam(str(e), remedy=f"Expected signature: {name}{sig}") from e
    bound.apply_defaults()
    return jsonable(dict(bound.arguments))


def tool(
    name: str,
    *,
    category: str,
    summary: str = "",
    destructive: bool = False,
    needs_network: bool = False,
    needs_r: bool = False,
    needs_project: bool = True,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Register a function as an MCP tool with automatic provenance.

    The wrapped function receives a `ctx: Ctx` keyword argument it uses to record
    warnings, artifacts, and the reproducing code. Its return value is ignored;
    everything user-visible goes through `ctx`.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        sig = inspect.signature(fn)
        if "ctx" not in sig.parameters:
            raise RuntimeError(f"{name}: tool functions must accept a `ctx` parameter")
        public_params = [p for k, p in sig.parameters.items() if k != "ctx"]
        public_sig = sig.replace(parameters=public_params)

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            started = store.now()
            t0 = time.perf_counter()
            params = _resolved_params(name, public_sig, args, kwargs)
            seed = int(params.get("seed", 0) or 0)
            dry_run = bool(params.get("dry_run", False))
            _seed_everything(seed)

            project_id = ""
            ctx: Ctx | None = None
            try:
                if needs_project:
                    project_id = resolve_project(str(params.get("project_id", "") or ""))
                else:
                    # Bootstrap tools (open_project / list_projects) run without one.
                    project_id = str(params.get("project_id", "") or "") or \
                        (_ACTIVE_PROJECT or "")
                ctx = Ctx(tool=name, project_id=project_id, seed=seed, dry_run=dry_run)
                kwargs2 = dict(kwargs)
                kwargs2["ctx"] = ctx
                fn(*args, **kwargs2)
                ok, err_payload, err_text = True, None, ""
            except SkinMCPError as e:
                ok, err_payload, err_text = False, e.to_dict(), f"{e.code}: {e.message}"
                logger.warning("%s failed: %s", name, err_text)
            except Exception as e:  # noqa: BLE001 - boundary: nothing escapes as a traceback
                ok = False
                err_text = f"{type(e).__name__}: {e}"
                logger.error("%s raised: %s\n%s", name, err_text, traceback.format_exc())
                err_payload = {
                    "code": "INTERNAL",
                    "message": err_text,
                    "remedy": "This is a bug in skin-mcp. The traceback is on the server's "
                              "stderr. Try skin.io.describe on the input handle to check "
                              "its state, or re-run with dry_run=True.",
                }

            dur = time.perf_counter() - t0
            if ctx is None:
                # Failed before a project could be resolved: nothing to log against.
                return ToolResult(
                    ok=False, error=err_payload, warnings=[],
                    next_suggested_tools=["skin.memory.open_project", "skin.memory.list_projects"],
                ).model_dump()

            # A bootstrap tool may have set ctx.project_id while running.
            project_id = ctx.project_id or project_id
            if not project_id:
                return ToolResult(
                    ok=ok, summary=jsonable(ctx.summary), warnings=ctx.warnings,
                    error=err_payload, next_suggested_tools=ctx.next_tools,
                ).model_dump()

            step_id = store.record_step(
                project_id, tool=name, params=params, inputs=ctx.inputs,
                outputs={"dataset_id": ctx.dataset_id,
                         "artifacts": [a.id for a in ctx.artifacts],
                         "code_executable": ctx.code_executable},
                code=ctx.code, versions=python_manifest.capture(name), seed=seed,
                started_at=started, duration_s=dur, ok=ok, error=err_text,
            )
            ctx.step_id = step_id
            for a in ctx.artifacts:
                store.record_artifact(project_id, a.id, step_id, a.kind, a.path, a.caption, params)

            if dry_run and ok:
                ctx.warn("dry_run=True — nothing was executed or written. "
                         "Re-run with dry_run=False to apply.")

            result = ToolResult(
                ok=ok,
                dataset_id=ctx.dataset_id,
                summary=jsonable(ctx.summary),
                warnings=ctx.warnings,
                artifacts=ctx.artifacts,
                code=ctx.code,
                memory_ref=f"step_{step_id:05d}",
                next_suggested_tools=ctx.next_tools,
                error=err_payload,
            )
            return enforce_budget(result.model_dump(), project_id=project_id, step_id=step_id)

        # functools.wraps copies the tool function's annotations, including
        # `-> None`. The MCP SDK builds its output schema from that annotation
        # and would then reject the dict the wrapper actually returns, so both
        # the signature and the annotation have to describe the WRAPPER, not the
        # wrapped function. This only surfaces over a real client connection —
        # tests/test_mcp_client.py exercises that path.
        public_sig = public_sig.replace(return_annotation=dict[str, Any])
        wrapper.__signature__ = public_sig  # type: ignore[attr-defined]
        wrapper.__annotations__ = {
            k: v for k, v in getattr(fn, "__annotations__", {}).items() if k != "ctx"
        }
        wrapper.__annotations__["return"] = dict[str, Any]
        wrapper.skin_tool_name = name  # type: ignore[attr-defined]
        REGISTRY[name] = ToolSpec(
            name=name, category=category, fn=wrapper,
            summary=summary or (fn.__doc__ or "").strip().split("\n")[0],
            destructive=destructive, needs_network=needs_network, needs_r=needs_r,
            needs_project=needs_project,
        )
        return wrapper

    return deco


# --------------------------------------------------------------------------- #
# Shared helpers used across tool modules
# --------------------------------------------------------------------------- #

def require_obs(adata: Any, key: str) -> None:
    from ..errors import missing_obs

    if key not in adata.obs.columns:
        raise missing_obs(key, list(adata.obs.columns))


def require_obsm(adata: Any, key: str) -> None:
    from ..errors import no_embedding

    if key not in adata.obsm:
        raise no_embedding(key, list(adata.obsm.keys()))


def pick_rep(adata: Any, preferred: str = "") -> str:
    """Resolve the embedding to work on, preferring the integrated one."""
    from ..errors import no_embedding

    if preferred and preferred in adata.obsm:
        return preferred
    for cand in ("X_pca_harmony", "X_scVI", "X_pca"):
        if cand in adata.obsm:
            return cand
    raise no_embedding(preferred or "X_pca_harmony", list(adata.obsm.keys()))


def confirm_or_raise(confirm: bool, dry_run: bool, op: str, impact: str) -> None:
    """Two-step gate for destructive operations (§9.7)."""
    from ..errors import needs_confirm

    if dry_run or confirm:
        return
    raise needs_confirm(op, impact)
