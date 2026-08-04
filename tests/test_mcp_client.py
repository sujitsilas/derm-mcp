"""Exercise the real MCP call path, not just direct Python calls.

Several failure modes are invisible to direct invocation and only appear once a
client is on the other end of the protocol — output-schema validation against the
function's return annotation being the one that bit us. Anything that would break
a Claude Desktop or LM Studio connection has to be caught here.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from skinmcp.config import CONFIG
from skinmcp.server import build_server


def _text(result) -> str:
    return result.content[0].text


def call(mcp, name: str, args: dict) -> dict:
    res = asyncio.run(mcp.call_tool(name, args))
    assert not res.is_error, f"{name} errored over MCP: {_text(res)[:600]}"
    return json.loads(_text(res))


@pytest.fixture()
def mcp():
    return build_server()


def test_tools_list_over_protocol(mcp):
    """The default profile must carry a whole analysis end to end.

    Counting tools is the wrong assertion — the default is `core` precisely to
    keep the list short enough that a local model still picks correctly. What
    matters is that nothing on the load -> QC -> cluster -> annotate -> DE ->
    plot path is gated away.
    """
    tools = asyncio.run(mcp.list_tools())
    names = {t.name for t in tools}
    for expected in ("skin.memory.open_project", "skin.io.load_h5ad",
                     "skin.qc.sample_stats", "skin.cluster.leiden",
                     "skin.sub.extract", "skin.de.pseudobulk",
                     "skin.plot.volcano_grid", "skin.plot.umap",
                     "skin.help.workflow"):
        assert expected in names, f"{expected} is missing from the default profile"
    for t in tools:
        assert t.description, f"{t.name} has no description"
        assert t.input_schema.get("type") == "object"


def test_full_profile_is_a_superset():
    """`full` adds the specialist namespaces without changing the core ones."""
    from skinmcp.config import CONFIG

    core = {t.name for t in asyncio.run(build_server().list_tools())}
    old = CONFIG.profile
    try:
        CONFIG.profile = "full"
        full = {t.name for t in asyncio.run(build_server().list_tools())}
    finally:
        CONFIG.profile = old
    assert core < full, "full must expose strictly more than core"
    assert {n.rsplit(".", 1)[0] for n in full - core} <= {
        "skin.ccc", "skin.traj", "skin.atlas", "skin.runtime", "skin.abundance",
        "skin.export", "skin.report", "skin.bench",
    }, sorted(full - core)


def test_prompts_list_and_get(mcp):
    prompts = asyncio.run(mcp.list_prompts())
    assert len(prompts) >= 10
    got = asyncio.run(mcp.get_prompt("sop_qc_and_filter", {}))
    body = got.messages[0].content.text
    assert "skin.qc.sample_stats" in body
    assert "neutrophil" in body.lower()


def test_full_call_roundtrip(mcp, project, golden_path):
    """The failure this file exists for: a tool returning its dict cleanly."""
    out = call(mcp, "skin.memory.open_project",
               {"name": "mcp_roundtrip", "organism": "mouse"})
    assert out["ok"] is True
    pid = out["summary"]["project_id"]
    assert out["next_suggested_tools"]

    out = call(mcp, "skin.io.load_h5ad",
               {"path": str(golden_path), "organism": "mouse", "project_id": pid})
    assert out["ok"] and out["dataset_id"].startswith("ds_")
    ds = out["dataset_id"]

    out = call(mcp, "skin.io.describe", {"dataset_id": ds, "project_id": pid})
    assert out["summary"]["n_obs"] == 7200

    out = call(mcp, "skin.qc.sample_stats", {"dataset_id": ds, "project_id": pid})
    assert out["summary"]["n_samples"] == 12
    assert len(json.dumps(out).encode()) <= CONFIG.max_return_bytes


def test_errors_come_back_as_data_not_protocol_errors(mcp, project):
    """A typed tool error must be an ok=False payload, not an MCP error frame —
    otherwise a small model sees a transport failure instead of a remedy."""
    res = asyncio.run(mcp.call_tool("skin.io.describe",
                                    {"dataset_id": "ds_nope", "project_id": project}))
    payload = json.loads(_text(res))
    assert payload["ok"] is False
    assert payload["error"]["code"] == "INVALID_HANDLE"
    assert payload["error"]["remedy"]


def test_resources_are_readable(mcp, project, golden_path):
    out = call(mcp, "skin.memory.open_project",
               {"name": "res_test", "organism": "mouse"})
    pid = out["summary"]["project_id"]
    out = call(mcp, "skin.io.load_h5ad",
               {"path": str(golden_path), "organism": "mouse", "project_id": pid})
    ds = out["dataset_id"]

    contents = list(asyncio.run(mcp.read_resource(f"skin://project/{pid}/summary")))
    brief = json.loads(contents[0].content)
    assert brief["project"]["project_id"] == pid

    contents = list(asyncio.run(mcp.read_resource(f"skin://dataset/{ds}/obs_schema")))
    schema = json.loads(contents[0].content)
    assert schema["n_obs"] == 7200
    assert "Sample" in schema["obs"]

    contents = list(asyncio.run(mcp.read_resource("skin://knowledge/markers/mouse")))
    km = json.loads(contents[0].content)
    assert "Keratinocytes" in km["lineages"]


def test_core_profile_hides_advanced_namespaces():
    old = CONFIG.profile
    try:
        CONFIG.profile = "core"
        names = {t.name for t in asyncio.run(build_server(profile="core").list_tools())}
        assert "skin.qc.sample_stats" in names
        assert not any(n.startswith("skin.ccc.") for n in names)
        assert not any(n.startswith("skin.atlas.") for n in names)
    finally:
        CONFIG.profile = old
        build_server(profile=old)
