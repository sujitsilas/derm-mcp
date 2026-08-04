"""Return-envelope contract: shape, size budget, and typed errors."""

from __future__ import annotations

import json

import pytest

from skinmcp.returns import ToolResult, enforce_budget
from skinmcp.tools import io_tools, memory_tools, qc_tools

BUDGET = 4096


def _size(payload) -> int:
    return len(json.dumps(payload, ensure_ascii=False, default=str).encode())


def test_envelope_validates(loaded):
    pid, ds = loaded
    r = io_tools.describe(dataset_id=ds, project_id=pid)
    ToolResult.model_validate(r)
    for k in ("ok", "summary", "warnings", "artifacts", "code", "next_suggested_tools"):
        assert k in r


@pytest.mark.parametrize("call", [
    lambda pid, ds: io_tools.describe(dataset_id=ds, project_id=pid),
    lambda pid, ds: io_tools.lineage(project_id=pid),
    lambda pid, ds: memory_tools.brief(project_id=pid),
    lambda pid, ds: memory_tools.timeline(project_id=pid),
    lambda pid, ds: qc_tools.sample_stats(dataset_id=ds, project_id=pid),
    lambda pid, ds: qc_tools.recommend_thresholds(dataset_id=ds, project_id=pid),
    lambda pid, ds: qc_tools.preview_filters(dataset_id=ds, project_id=pid,
                                             thresholds={"min_genes": 200}),
])
def test_returns_fit_the_budget(loaded, call):
    pid, ds = loaded
    r = call(pid, ds)
    assert _size(r) <= BUDGET, f"return is {_size(r)} bytes, budget is {BUDGET}"


def test_oversized_summary_spills_to_a_resource(project):
    payload = {"ok": True, "summary": {"big": [{"i": i, "pad": "x" * 200}
                                               for i in range(400)]},
               "warnings": [], "artifacts": [], "code": "", "memory_ref": "",
               "next_suggested_tools": []}
    out = enforce_budget(payload, project_id=project, step_id=1)
    assert _size(out) <= BUDGET
    assert out["truncated"] is True
    assert any("skin://" in w for w in out["warnings"])


def test_nan_and_inf_become_null():
    import numpy as np

    from skinmcp.returns import jsonable

    out = jsonable({"a": np.nan, "b": np.inf, "c": -np.inf, "d": np.float32(1.5),
                    "e": np.int64(3), "f": np.array([1, 2])})
    assert out["a"] is None and out["b"] is None and out["c"] is None
    assert out["d"] == 1.5 and out["e"] == 3 and out["f"] == [1, 2]
    json.dumps(out)  # must be valid JSON, not NaN/Infinity


def test_errors_are_typed_with_remedies(project):
    r = io_tools.describe(dataset_id="nope", project_id=project)
    assert r["ok"] is False
    e = r["error"]
    assert e["code"] == "INVALID_HANDLE"
    assert e["message"] and e["remedy"]
    assert "suggested_tool" in e


def test_missing_obs_key_lists_available(loaded):
    pid, ds = loaded
    r = qc_tools.sample_stats(dataset_id=ds, sample_key="NotAColumn", project_id=pid)
    assert r["ok"] is False
    assert r["error"]["code"] == "MISSING_OBS_KEY"
    assert "Sample" in r["error"]["remedy"]


def test_no_project_is_typed():
    from skinmcp.tools import _base

    old = _base.get_active_project()
    try:
        _base.set_active_project("")
        r = io_tools.describe(dataset_id="x")
        assert r["ok"] is False
        assert r["error"]["code"] == "NO_PROJECT"
        assert "skin.memory.open_project" in r["next_suggested_tools"]
    finally:
        _base.set_active_project(old or "")


def test_dry_run_executes_nothing(loaded):
    from skinmcp.memory import store

    pid, ds = loaded
    before = len(store.list_datasets(pid))
    r = qc_tools.apply_filters(dataset_id=ds, project_id=pid,
                               thresholds={"min_genes": 200}, dry_run=True)
    assert r["ok"] and r["dataset_id"] is None
    assert len(store.list_datasets(pid)) == before
    assert any("dry_run" in w for w in r["warnings"])


def test_every_step_is_logged_including_failures(loaded):
    from skinmcp.memory import store

    pid, ds = loaded
    io_tools.describe(dataset_id="bogus", project_id=pid)
    steps = store.get_steps(pid, limit=50)
    failed = [s for s in steps if not s["ok"]]
    assert failed, "failures must still append a provenance row"
    assert failed[-1]["tool"] == "skin.io.describe"
    assert "INVALID_HANDLE" in (failed[-1]["error"] or "")


def test_steps_capture_versions(loaded):
    import json as _json

    from skinmcp.memory import store

    pid, ds = loaded
    qc_tools.sample_stats(dataset_id=ds, project_id=pid)
    s = store.get_steps(pid, limit=5, include_code=True)[-1]
    v = _json.loads(s["versions_json"])
    assert "scanpy" in v and "python" in v


class TestNothingCanKillTheProcess:
    """A tool must never take the server down.

    An exception escaping the decorator surfaces to the client as
    `MCP error -32000: Connection closed`, and the whole session — including work
    that already succeeded — is lost. This actually happened: a pandas MultiIndex
    reached the error payload, `json.dumps` refused its tuple keys, and the
    TypeError escaped because the error payload was not coerced and the
    post-processing block sat outside the try/except.
    """

    @staticmethod
    def _temp_tool(name):
        """Register a throwaway tool and remove it again.

        Leaving it in the global REGISTRY would change the tool count and the
        schema-test parametrisation depending on import order.
        """
        import contextlib

        from skinmcp.tools._base import REGISTRY

        @contextlib.contextmanager
        def cm():
            try:
                yield
            finally:
                REGISTRY.pop(name, None)

        return cm()

    def test_unserialisable_summary_degrades_instead_of_raising(self, project):
        from skinmcp.tools._base import Ctx, tool

        @tool("skin.help.__unserialisable", category="help", summary="test only")
        def _bad(project_id: str = "", dry_run: bool = False, seed: int = 0,
                 *, ctx: Ctx) -> None:
            """Emit something json cannot represent.

            Args:
                project_id: project.
                dry_run: unused.
                seed: unused.
            """
            ctx.summary = {("a", "b"): "tuple key", "obj": object()}

        with self._temp_tool("skin.help.__unserialisable"):
            r = _bad(project_id=project)       # must return, not raise
            assert isinstance(r, dict)
            assert r["ok"] in (True, False)
            json.dumps(r)                      # and the result itself must serialise

    def test_unserialisable_error_details_degrade(self, project):
        import pandas as pd

        from skinmcp.errors import BadParam
        from skinmcp.tools._base import Ctx, tool

        @tool("skin.help.__baderror", category="help", summary="test only")
        def _raiser(project_id: str = "", dry_run: bool = False, seed: int = 0,
                    *, ctx: Ctx) -> None:
            """Raise with details carrying a MultiIndex.

            Args:
                project_id: project.
                dry_run: unused.
                seed: unused.
            """
            s = pd.DataFrame({"a": ["x", "y"], "b": ["p", "q"], "n": [1, 2]}) \
                .groupby(["a", "b"], observed=True)["n"].sum()
            raise BadParam("nope", details={"cells": s.to_dict()})

        with self._temp_tool("skin.help.__baderror"):
            r = _raiser(project_id=project)
            assert r["ok"] is False
            assert r["error"]["code"] == "BAD_PARAM"
            json.dumps(r)                      # tuple keys must have been coerced

    def test_provenance_failure_is_not_fatal(self, project, monkeypatch):
        import sqlite3

        from skinmcp.memory import store
        from skinmcp.tools import io_tools

        def boom(*a, **k):
            raise sqlite3.OperationalError("database is locked")

        monkeypatch.setattr(store, "record_step", boom)
        r = io_tools.lineage(project_id=project)
        assert isinstance(r, dict)
        assert any("provenance log" in w for w in r["warnings"]), \
            "a failed provenance write should warn, not crash"
