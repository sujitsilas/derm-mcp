"""The execution escape hatch, and the shared runtime it runs in.

Sessions repeatedly abandoned skin-mcp for the host's Python sandbox — which has
no numpy, no pandas and no filesystem — because nothing here could run code the
typed tools did not already cover. These tests pin the properties that make the
escape hatch safe to reach for.
"""

from __future__ import annotations

import pytest

from skinmcp.config import CONFIG
from skinmcp.tools import runtime_tools


@pytest.fixture()
def raw_exec_on():
    old = CONFIG.allow_raw_exec
    CONFIG.allow_raw_exec = True
    yield
    CONFIG.allow_raw_exec = old


class TestSharedRuntime:
    def test_runtime_is_shared_not_per_project(self):
        """One env for the machine. Building it costs minutes and hundreds of MB,
        so paying that per project is pure waste on a single-user workstation."""
        venv = CONFIG.shared_venv()
        assert CONFIG.project_root in venv.parents
        # and it must not sit under any project directory
        assert "proj_" not in str(venv)

    def test_exec_scratch_stays_per_project(self):
        """The runtime is shared but the code that ran is part of one project's record."""
        a = CONFIG.exec_dir("proj_aaaaaaaa")
        b = CONFIG.exec_dir("proj_bbbbbbbb")
        assert a != b and a.is_dir() and b.is_dir()


class TestGuards:
    def test_disabled_without_allow_raw_exec(self, project):
        r = runtime_tools.exec_python(code="print(1)", confirm=True, project_id=project)
        assert not r["ok"]
        assert "disabled" in r["error"]["message"]

    def test_names_create_when_the_runtime_is_missing(self, project, raw_exec_on, tmp_path):
        old = CONFIG.project_root
        CONFIG.project_root = tmp_path          # a root with no runtime built
        try:
            r = runtime_tools.exec_python(code="print(1)", confirm=True, project_id=project)
        finally:
            CONFIG.project_root = old
        assert not r["ok"]
        assert r["error"]["suggested_tool"] == "skin.runtime.create"

    def test_dry_run_never_executes(self, project, raw_exec_on):
        r = runtime_tools.exec_python(code="raise SystemExit(3)", confirm=True,
                                      project_id=project, dry_run=True)
        assert r["ok"]
        assert "exit_code" not in r["summary"]


class TestRVersionPreflight:
    def test_mismatched_lockfile_fails_fast_with_the_real_reason(self, tmp_path):
        """Restoring a Bioc 3.20 lock under R 4.6 404s after installing nothing,
        reporting only "package 'BiocManager' is not available"."""
        import json

        from skinmcp.errors import RuntimeUnavailable

        lock = tmp_path / "renv.lock"
        lock.write_text(json.dumps({"R": {"Version": "4.4.2"},
                                    "Bioconductor": {"Version": "3.20"}}))
        with pytest.raises(RuntimeUnavailable) as e:
            runtime_tools._check_lock_matches_r(lock, "R version 4.6.1 (2026-06-24)")
        assert "4.4.2" in str(e.value) and "4.6" in str(e.value)
        assert "Bioconductor 3.20" in e.value.remedy

    def test_matching_version_passes(self, tmp_path):
        import json

        lock = tmp_path / "renv.lock"
        lock.write_text(json.dumps({"R": {"Version": "4.6.0"}}))
        runtime_tools._check_lock_matches_r(lock, "R version 4.6.1 (2026-06-24)")

    def test_a_malformed_lock_does_not_raise(self, tmp_path):
        lock = tmp_path / "renv.lock"
        lock.write_text("not json")
        runtime_tools._check_lock_matches_r(lock, "R version 4.6.1")


class TestArtifactCapture:
    def test_snapshot_only_tracks_capturable_files(self, tmp_path):
        (tmp_path / "keep.png").write_bytes(b"x")
        (tmp_path / "keep.csv").write_text("a,b")
        (tmp_path / "ignore.txt").write_text("noise")
        (tmp_path / "ignore.log").write_text("noise")
        seen = {p.name for p in runtime_tools._snapshot(tmp_path)}
        assert seen == {"keep.png", "keep.csv"}

    def test_snapshot_recurses(self, tmp_path):
        sub = tmp_path / "figures" / "umap"
        sub.mkdir(parents=True)
        (sub / "deep.pdf").write_bytes(b"x")
        assert any(p.name == "deep.pdf" for p in runtime_tools._snapshot(tmp_path))
