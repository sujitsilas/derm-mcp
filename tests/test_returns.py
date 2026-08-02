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
