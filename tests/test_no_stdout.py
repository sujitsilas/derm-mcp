"""A single `print()` in a tool corrupts the JSON-RPC stream on stdio.

This is a hard CI gate, not a style preference.
"""

from __future__ import annotations

import ast
import io
import re
from contextlib import redirect_stdout
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "skinmcp"
PY_FILES = sorted(p for p in SRC.rglob("*.py"))


@pytest.mark.parametrize("path", PY_FILES, ids=lambda p: str(p.relative_to(SRC)))
def test_no_bare_print_or_stdout(path: Path):
    src = path.read_text(encoding="utf-8")
    tree = ast.parse(src, filename=str(path))
    offenders = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "print":
            offenders.append(("print()", node.lineno))
        if isinstance(node, ast.Attribute) and node.attr in ("stdout", "__stdout__"):
            val = node.value
            if isinstance(val, ast.Name) and val.id == "sys":
                offenders.append(("sys.stdout", node.lineno))
    assert not offenders, f"{path.relative_to(SRC)}: {offenders}"


def test_no_print_in_generated_code_strings():
    """The `code` strings tools emit end up in the notebook, not on stdout —
    but a stray print in one would still be executed there. Keep them clean."""
    bad = []
    for p in PY_FILES:
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            if re.search(r'["\']\s*print\(', line) and "sc.pl" not in line:
                bad.append(f"{p.relative_to(SRC)}:{i}")
    assert not bad, f"print() inside emitted code strings: {bad}"


def test_importing_the_server_writes_nothing_to_stdout():
    buf = io.StringIO()
    with redirect_stdout(buf):
        import importlib

        import skinmcp.server as s

        importlib.reload(s)
        s.build_server()
    assert buf.getvalue() == "", f"stdout polluted at import: {buf.getvalue()[:400]!r}"


def test_running_a_tool_writes_nothing_to_stdout(loaded):
    from skinmcp.tools import cluster_tools, integrate_tools, io_tools, qc_tools

    pid, ds = loaded
    buf = io.StringIO()
    with redirect_stdout(buf):
        io_tools.describe(dataset_id=ds, project_id=pid)
        qc_tools.sample_stats(dataset_id=ds, project_id=pid)
        p = integrate_tools.preprocess(dataset_id=ds, project_id=pid, n_hvg=200,
                                       n_comps=10)["dataset_id"]
        n = cluster_tools.neighbors(dataset_id=p, project_id=pid, use_rep="X_pca",
                                    n_pcs=10)["dataset_id"]
        cluster_tools.leiden(dataset_id=n, project_id=pid, resolution=0.5)
    assert buf.getvalue() == "", f"stdout polluted by a tool: {buf.getvalue()[:400]!r}"
