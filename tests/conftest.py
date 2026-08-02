from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

GOLDEN = Path(__file__).parent / "golden" / "golden_mouse_skin.h5ad"


@pytest.fixture(scope="session")
def golden_path() -> Path:
    if not GOLDEN.exists():
        import subprocess
        import sys

        subprocess.run([sys.executable, str(GOLDEN.parent / "make_golden.py")], check=True)
    return GOLDEN


@pytest.fixture()
def project(tmp_path_factory):
    """A throwaway project root plus an open project, torn down afterwards."""
    from skinmcp.config import CONFIG
    from skinmcp.memory import store
    from skinmcp.registry import cache_clear
    from skinmcp.tools import memory_tools

    root = Path(tempfile.mkdtemp(prefix="skinmcp_test_"))
    old = CONFIG.project_root
    CONFIG.project_root = root
    store.close_all()
    cache_clear()

    r = memory_tools.open_project(name="test_project", organism="mouse",
                                  description="pytest fixture")
    assert r["ok"], r
    pid = r["summary"]["project_id"]
    yield pid

    store.close_all()
    cache_clear()
    CONFIG.project_root = old
    shutil.rmtree(root, ignore_errors=True)


@pytest.fixture()
def loaded(project, golden_path):
    """The golden dataset loaded into a fresh project."""
    from skinmcp.tools import io_tools

    r = io_tools.load_h5ad(path=str(golden_path), organism="mouse", label="raw",
                           project_id=project)
    assert r["ok"], r
    return project, r["dataset_id"]
