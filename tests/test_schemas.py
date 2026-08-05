"""Schema hygiene: the ergonomics contract for a 30B local tool-caller.

Every tool: small JSON schema, enums where a domain is fixed, few required
arguments, `dry_run` and `seed` present, and a docstring with an Args block.
"""

from __future__ import annotations

import inspect
import json

import pytest

from skinmcp.config import CONFIG
from skinmcp.server import build_server
from skinmcp.tools._base import REGISTRY

build_server()
TOOLS = sorted(REGISTRY.items())
NAMES = [n for n, _ in TOOLS]


@pytest.mark.parametrize("name", NAMES)
def test_has_dry_run_and_seed(name):
    sig = inspect.signature(REGISTRY[name].fn)
    assert "dry_run" in sig.parameters, f"{name} must accept dry_run"
    assert "seed" in sig.parameters, f"{name} must accept seed"
    assert sig.parameters["dry_run"].default is False
    assert sig.parameters["seed"].default == 0


@pytest.mark.parametrize("name", NAMES)
def test_max_six_required_args(name):
    sig = inspect.signature(REGISTRY[name].fn)
    required = [k for k, p in sig.parameters.items()
                if p.default is inspect.Parameter.empty
                and p.kind not in (p.VAR_POSITIONAL, p.VAR_KEYWORD)]
    assert len(required) <= 6, f"{name} has {len(required)} required args: {required}"


@pytest.mark.parametrize("name", NAMES)
def test_docstring_has_args_block(name):
    doc = REGISTRY[name].fn.__doc__ or ""
    assert doc.strip(), f"{name} has no docstring"
    sig = inspect.signature(REGISTRY[name].fn)
    if len(sig.parameters) > 2:
        assert "Args:" in doc, f"{name} docstring has no Args block"
        for p in sig.parameters:
            assert f"{p}:" in doc, f"{name} docstring does not document {p!r}"


@pytest.mark.parametrize("name", NAMES)
def test_all_params_have_json_types(name):
    """No bare `Any`-typed required params — a small model cannot guess a shape."""
    sig = inspect.signature(REGISTRY[name].fn)
    for k, p in sig.parameters.items():
        if p.default is inspect.Parameter.empty and k != "ctx":
            assert p.annotation is not inspect.Parameter.empty, \
                f"{name}.{k} is required but untyped"


def test_tool_names_are_namespaced():
    for n in NAMES:
        parts = n.split(".")
        assert parts[0] == "skin", f"{n} must start with 'skin.'"
        assert len(parts) == 3, f"{n} must be skin.<namespace>.<verb>"


def test_mcp_schemas_are_small():
    """Each tool's JSON schema must stay under 2 KB (§10)."""
    import asyncio

    mcp = build_server()
    big = []
    for t in asyncio.run(mcp.list_tools()):
        size = len(json.dumps(t.input_schema).encode())
        if size > 2048:
            big.append((t.name, size))
    assert not big, f"schemas over 2 KB: {big}"


def test_core_profile_is_small_enough():
    """`--profile core` must expose a set a 30B model can hold in context.

    The ceiling moved from 95 to 110 when composition, export and runtime were
    promoted into core: gating `skin.abundance` had left a real session unable to
    find `skin.abundance.proportions` when asked for exactly that plot, and a
    missing tool is a worse failure than a long list — the model improvises and
    burns the whole budget, whereas a long list only makes it slower to choose.

    If selection does start to degrade, the fix is trimming rarely-used tools out
    of `skin.io` / `skin.meta` (one session reached for `set_label` when it wanted
    `load_h5ad`), NOT re-gating whole workflow namespaces.
    """
    old = CONFIG.profile
    try:
        CONFIG.profile = "core"
        n = sum(1 for n in NAMES if CONFIG.namespace_enabled(n))
        assert n <= 110, f"core profile exposes {n} tools"
        assert n >= 40, f"core profile exposes only {n} tools; too little to work with"
    finally:
        CONFIG.profile = old


def test_destructive_tools_have_confirm():
    for name, spec in TOOLS:
        if not spec.destructive:
            continue
        sig = inspect.signature(spec.fn)
        assert "confirm" in sig.parameters, f"{name} is destructive but has no confirm"


def test_every_namespace_is_documented():
    from skinmcp.tools.help_tools import CATEGORY_BLURB

    cats = {s.category for _, s in TOOLS}
    missing = cats - set(CATEGORY_BLURB)
    assert not missing, f"undocumented categories: {missing}"


def test_prompts_exist():
    from pathlib import Path

    import skinmcp

    d = Path(skinmcp.__file__).parent / "prompts"
    names = {p.stem for p in d.glob("*.md")}
    expected = {"sop_new_project", "sop_qc_and_filter", "sop_first_pass_annotation",
                "sop_decontamination_loop", "sop_subcluster", "sop_pseudobulk_de",
                "sop_trajectory", "sop_abundance", "sop_communication",
                "sop_finalize_and_export"}
    assert expected <= names, f"missing SOP prompts: {expected - names}"


def test_prompts_registered_with_server():
    import asyncio

    mcp = build_server()
    got = {p.name for p in asyncio.run(mcp.list_prompts())}
    assert "sop_qc_and_filter" in got
    assert len(got) >= 10
