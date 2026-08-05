"""Choosing which R to drive, and noticing when it does not match the lockfile.

Bioconductor pins each release to one R minor version (3.20 -> R 4.4,
3.23 -> R 4.6) and publishes binaries only for it, so an renv.lock and the R
it was cut against are a matched pair. Taking whatever `Rscript` is on PATH
means a routine R upgrade silently invalidates the lockfile — which is exactly
what happened here: a lock pinned to R 4.4.2 against an installed R 4.6.1.
"""

from __future__ import annotations

from skinmcp.config import CONFIG
from skinmcp.runtimes import bridge


class TestRscriptSelection:
    def test_defaults_to_path(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "")
        monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/Rscript")
        assert bridge.rscript_path() == "/usr/local/bin/Rscript"

    def test_configured_interpreter_wins(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "/opt/R/4.4.2/bin/Rscript")
        monkeypatch.setattr(bridge.shutil, "which", lambda _: "/usr/local/bin/Rscript")
        assert bridge.rscript_path() == "/opt/R/4.4.2/bin/Rscript"

    def test_tilde_is_expanded(self, monkeypatch):
        # rig puts its quick links under ~/.local/bin, so ~ is the common case.
        monkeypatch.setattr(CONFIG, "rscript", "~/.local/bin/Rscript-4.4")
        assert not bridge.rscript_path().startswith("~")
        assert bridge.rscript_path().endswith("/.local/bin/Rscript-4.4")

    def test_falls_back_to_bare_name_when_nothing_resolves(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "")
        monkeypatch.setattr(bridge.shutil, "which", lambda _: None)
        assert bridge.rscript_path() == "Rscript"

    def test_missing_pinned_interpreter_is_reported_as_unavailable(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "/nope/does/not/exist/Rscript")
        ok, msg = bridge.rscript_available()
        assert ok is False
        assert "SKINMCP_RSCRIPT" in msg


class TestStaleToolchainVars:
    """A stale SDK pin in the user's shell breaks every source compile.

    A real ~/.zshrc here exported SDKROOT=.../MacOSX15.2.sdk after Xcode had
    moved to MacOSX26.5.sdk. SDKROOT overrides everything -- even
    `xcrun --show-sdk-path` returns it -- so packages failed to build with
    "'stdio.h' file not found", surfacing from renv as `R CMD config CC`
    failing. The error invites `xcode-select --install`, which does not help:
    the command line tools were installed and healthy the whole time.
    """

    def test_drops_a_variable_naming_a_missing_path(self):
        env, dropped = bridge.clean_build_env(
            {"SDKROOT": "/Applications/Xcode.app/does/not/exist/MacOSX15.2.sdk"})
        assert dropped == ["SDKROOT"]
        assert "SDKROOT" not in env

    def test_keeps_a_variable_naming_a_real_path(self, tmp_path):
        env, dropped = bridge.clean_build_env({"SDKROOT": str(tmp_path)})
        assert dropped == []
        assert env["SDKROOT"] == str(tmp_path)

    def test_keeps_a_path_list_with_at_least_one_live_entry(self, tmp_path):
        import os

        val = os.pathsep.join([str(tmp_path), "/nope/gone"])
        env, dropped = bridge.clean_build_env({"LIBRARY_PATH": val})
        assert dropped == []
        assert env["LIBRARY_PATH"] == val

    def test_drops_a_path_list_where_nothing_survives(self):
        import os

        env, dropped = bridge.clean_build_env(
            {"LIBRARY_PATH": os.pathsep.join(["/nope/a", "/nope/b"])})
        assert dropped == ["LIBRARY_PATH"]

    def test_unset_and_empty_are_left_alone(self):
        env, dropped = bridge.clean_build_env({"CPATH": ""})
        assert dropped == []
        assert bridge.clean_build_env({})[1] == []

    def test_unrelated_variables_are_never_touched(self):
        env, _ = bridge.clean_build_env(
            {"PATH": "/nope/gone", "HOME": "/nope/gone", "R_LIBS": "/nope/gone"})
        assert env["PATH"] == "/nope/gone"
        assert env["R_LIBS"] == "/nope/gone"


class TestHermeticInvocation:
    """R must not read the user's startup files.

    A real ~/.Rprofile here ended with

        .libPaths(c("~/Library/R/arm64/4.6/library", .libPaths()))

    which put an R 4.6 library in front for every session, including the pinned
    R 4.4. R then loaded an renv built by a different R and segfaulted inside
    dyn.load before any vetted script ran. Setting R_LIBS does not help — the
    profile runs afterwards and prepends itself.
    """

    def test_startup_files_are_disabled(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "/opt/R/4.4.2/bin/Rscript")
        cmd = bridge.rscript_cmd("-e", "1")
        assert cmd[0] == "/opt/R/4.4.2/bin/Rscript"
        assert "--no-init-file" in cmd and "--no-site-file" in cmd
        # Flags must precede the payload or Rscript treats them as script args.
        assert cmd.index("--no-init-file") < cmd.index("-e")

    def test_renviron_is_still_honoured(self, monkeypatch):
        # --vanilla would also drop ~/.Renviron, which carries machine settings
        # like R_MAX_VSIZE that we do want.
        monkeypatch.setattr(CONFIG, "rscript", "/opt/R/4.4.2/bin/Rscript")
        assert "--no-environ" not in bridge.rscript_cmd("-e", "1")
        assert "--vanilla" not in bridge.rscript_cmd("-e", "1")

    def test_status_reports_an_ignored_rprofile(self, monkeypatch, tmp_path):
        rp = tmp_path / ".Rprofile"
        rp.write_text('.libPaths(c("~/Library/R/arm64/4.6/library", .libPaths()))\n')
        monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path))
        assert bridge.user_rprofile_touches_libpaths() == str(rp)
        monkeypatch.setattr(bridge, "rscript_available", lambda: (True, "R version 4.4.3 (x)"))
        assert "ignored_rprofile" in bridge.runtime_status()

    def test_silent_when_the_rprofile_leaves_libpaths_alone(self, monkeypatch, tmp_path):
        (tmp_path / ".Rprofile").write_text('options(repos="https://cran.r-project.org")\n')
        monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path))
        assert bridge.user_rprofile_touches_libpaths() == ""


class TestStaleLibraryDetection:
    """A library built by another R is an obstacle, not a built runtime.

    R refuses to load a package built by an R with different internals, so
    treating "directory is non-empty" as "already built" made skin.runtime.create
    report success having installed nothing, and every R-backed tool then failed
    at library().
    """

    def _pkg(self, lib, name: str, built: str):
        d = lib / name
        d.mkdir(parents=True)
        (d / "DESCRIPTION").write_text(
            f"Package: {name}\nVersion: 1.0\n"
            f"Built: R {built}; aarch64-apple-darwin23; 2026-04-25 15:49:08 UTC; unix\n")

    def test_reads_the_r_version_a_library_was_built_by(self, tmp_path):
        from skinmcp.tools.runtime_tools import _library_r_minor

        for n in ("bit", "blob", "hms"):
            self._pkg(tmp_path, n, "4.6.0")
        assert _library_r_minor(tmp_path) == "4.6"

    def test_majority_wins_over_a_stray_package(self, tmp_path):
        from skinmcp.tools.runtime_tools import _library_r_minor

        for n in ("aaa", "bbb", "ccc"):
            self._pkg(tmp_path, n, "4.4.3")
        self._pkg(tmp_path, "ddd", "4.6.0")
        assert _library_r_minor(tmp_path) == "4.4"

    def test_empty_and_undescribed_libraries_are_unknown(self, tmp_path):
        from skinmcp.tools.runtime_tools import _library_r_minor

        assert _library_r_minor(tmp_path) == ""
        (tmp_path / "junk").mkdir()
        assert _library_r_minor(tmp_path) == ""


class TestLockSatisfaction:
    """"Non-empty" is not "built".

    A restore that dies partway leaves renv and BiocManager behind and nothing
    else. Treating that as a built runtime made skin.runtime.create report
    already_exists on a library holding 2 of ~150 packages, twice in a row,
    after which every vetted script failed at library().
    """

    def _lock(self, tmp_path, *names):
        import json

        p = tmp_path / "renv.lock"
        p.write_text(json.dumps({
            "R": {"Version": "4.4.2"}, "Bioconductor": {"Version": "3.20"},
            "Packages": {n: {"Package": n, "Version": "1.0"} for n in names}}))
        return p

    def test_partial_library_is_not_satisfied(self, tmp_path):
        from skinmcp.tools.runtime_tools import _lock_satisfied_by

        lock = self._lock(tmp_path, "renv", "BiocManager", "DESeq2", "miloR")
        lib = tmp_path / "library"
        for n in ("renv", "BiocManager"):      # exactly what an aborted restore leaves
            (lib / n).mkdir(parents=True)
        assert _lock_satisfied_by(lock, lib) is False

    def test_complete_library_is_satisfied(self, tmp_path):
        from skinmcp.tools.runtime_tools import _lock_satisfied_by

        lock = self._lock(tmp_path, "renv", "DESeq2")
        lib = tmp_path / "library"
        for n in ("renv", "DESeq2"):
            (lib / n).mkdir(parents=True)
        assert _lock_satisfied_by(lock, lib) is True

    def test_unreadable_or_empty_lock_is_never_satisfied(self, tmp_path):
        from skinmcp.tools.runtime_tools import _lock_satisfied_by

        lib = tmp_path / "library"
        lib.mkdir()
        assert _lock_satisfied_by(tmp_path / "missing.lock", lib) is False
        (tmp_path / "empty.lock").write_text('{"Packages": {}}')
        assert _lock_satisfied_by(tmp_path / "empty.lock", lib) is False


class TestBiocArchiveFallback:
    """Bioconductor release branches drift past the versions a lockfile pins.

    A branch serves exactly one version per package and its *binary* repo has
    no archive, so renv gives up without trying <repo>/bioc/src/contrib/Archive
    -- where the pinned tarball is still served. 4 of 13 Bioconductor pins here
    had drifted (celda, fgsea, scDblFinder, scater).
    """

    def test_parses_the_package_renv_gave_up_on(self):
        from skinmcp.tools.runtime_tools import _unretrievable

        assert _unretrievable(
            "Error: failed to retrieve package 'bioc::celda@1.22.0'") == ("celda", "1.22.0")
        assert _unretrievable(
            "failed to retrieve package 'scDblFinder@1.20.0'") == ("scDblFinder", "1.20.0")

    def test_ignores_unrelated_failures(self):
        from skinmcp.tools.runtime_tools import _unretrievable

        # Must not trigger the archive path for a compiler error or a 404 on
        # something that is not a version pin problem.
        assert _unretrievable("ERROR: compilation failed for package 'scran'") is None
        assert _unretrievable("") is None

    def test_reads_the_bioconductor_release_from_the_lock(self):
        from pathlib import Path as P

        from skinmcp.tools.runtime_tools import _bioc_release

        assert _bioc_release(P("src/skinmcp/runtimes/r/renv.lock")) == "3.20"

    def test_gives_up_when_the_lock_names_no_bioconductor_release(self, tmp_path):
        from skinmcp.tools.runtime_tools import _install_from_bioc_archive

        lock = tmp_path / "renv.lock"
        lock.write_text('{"Packages": {}}')
        assert _install_from_bioc_archive(
            "celda", "1.22.0", lock, tmp_path, tmp_path / "log", tmp_path) is False


class TestPartialLockfile:
    """renv aborts the whole restore over one unretrievable package.

    Dropping that package is the only way to get the other 150 installed, so
    the dropped ones can be fetched afterwards with their dependencies present.
    """

    def test_drops_only_the_named_packages(self, tmp_path):
        import json

        from skinmcp.tools.runtime_tools import _lock_without
        src = tmp_path / "renv.lock"
        src.write_text(json.dumps({
            "R": {"Version": "4.4.2", "Repositories": [{"Name": "CRAN", "URL": "u"}]},
            "Bioconductor": {"Version": "3.20"},
            "Packages": {n: {"Package": n} for n in ("celda", "DESeq2", "miloR")}}))
        out = json.loads(_lock_without(src, ["celda"], tmp_path / "w").read_text())
        assert set(out["Packages"]) == {"DESeq2", "miloR"}
        # Repositories and the Bioc release must survive or the partial restore
        # resolves against the wrong branch.
        assert out["R"]["Repositories"] == [{"Name": "CRAN", "URL": "u"}]
        assert out["Bioconductor"]["Version"] == "3.20"

    def test_leaves_the_original_untouched(self, tmp_path):
        import json

        from skinmcp.tools.runtime_tools import _lock_without
        src = tmp_path / "renv.lock"
        src.write_text(json.dumps({"Packages": {"celda": {}, "DESeq2": {}}}))
        _lock_without(src, ["celda"], tmp_path / "w")
        assert set(json.loads(src.read_text())["Packages"]) == {"celda", "DESeq2"}


class TestGithubToken:
    """A lockfile can pin GitHub-sourced packages, and api.github.com
    rate-limits unauthenticated callers to 60/hour — surfacing as a bare
    network error that never mentions rate limits."""

    def test_existing_credentials_are_left_alone(self, monkeypatch):
        from skinmcp.tools import runtime_tools

        monkeypatch.setenv("GITHUB_PAT", "already-set")
        monkeypatch.setattr(runtime_tools.shutil, "which",
                            lambda _: (_ for _ in ()).throw(AssertionError("must not shell out")))
        assert runtime_tools._github_env()["GITHUB_PAT"] == "already-set"

    def test_borrows_the_gh_cli_token(self, monkeypatch):
        import subprocess as sp

        from skinmcp.tools import runtime_tools

        monkeypatch.delenv("GITHUB_PAT", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setattr(runtime_tools.shutil, "which", lambda _: "/usr/bin/gh")
        monkeypatch.setattr(runtime_tools.subprocess, "run",
                            lambda *a, **k: sp.CompletedProcess(a, 0, "gho_tok\n", ""))
        assert runtime_tools._github_env()["GITHUB_PAT"] == "gho_tok"

    def test_survives_gh_being_absent_or_logged_out(self, monkeypatch):
        import subprocess as sp

        from skinmcp.tools import runtime_tools

        monkeypatch.delenv("GITHUB_PAT", raising=False)
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setattr(runtime_tools.shutil, "which", lambda _: None)
        assert "GITHUB_PAT" not in runtime_tools._github_env()

        monkeypatch.setattr(runtime_tools.shutil, "which", lambda _: "/usr/bin/gh")
        monkeypatch.setattr(runtime_tools.subprocess, "run",
                            lambda *a, **k: sp.CompletedProcess(a, 1, "", "not logged in"))
        assert "GITHUB_PAT" not in runtime_tools._github_env()


class TestCellChatPinIsReal:
    """The lockfile pinned a CellChat commit that does not exist (GitHub 422).

    A 40-hex string looks like a valid pin and fails only at restore time, so
    assert the shape we corrected it to and that the version still matches the
    tag it now names.
    """

    def test_pin_names_a_tag_and_a_plausible_sha(self):
        import json
        from pathlib import Path as P

        cc = json.loads(P("src/skinmcp/runtimes/r/renv.lock").read_text())["Packages"]["CellChat"]
        assert cc["RemoteSha"] == "a0d3b2d231d46c8787177fffeac908270c253747"
        assert cc["RemoteRef"] == f"v{cc['Version']}", "ref should name the version's tag"
        assert cc["Version"] == "2.1.2"


class TestLockfileVersionCheck:
    def test_reads_the_pinned_r_version(self):
        # The committed lock targets a specific R; if this ever returns "" the
        # mismatch warning silently stops working.
        assert bridge.lockfile_r_version().count(".") >= 1

    def _status(self, monkeypatch, version: str, lock: str = "4.4.2"):
        monkeypatch.setattr(bridge, "rscript_available", lambda: (True, version))
        monkeypatch.setattr(bridge, "lockfile_r_version", lambda: lock)
        return bridge.runtime_status()

    def test_flags_a_minor_version_mismatch(self, monkeypatch):
        s = self._status(monkeypatch, "Rscript (R) version 4.6.1 (2026-06-24)")
        assert s["r_version_matches_lockfile"] is False
        assert "rig add 4.4" in s["version_conflict"]
        # Regenerating the lock repins every package, so it must not read as
        # the obvious fix.
        assert "reproducibility decision" in s["version_conflict"]

    def test_accepts_a_matching_minor_version(self, monkeypatch):
        s = self._status(monkeypatch, "Rscript (R) version 4.4.3 (2025-02-28)")
        assert s["r_version_matches_lockfile"] is True
        assert "version_conflict" not in s

    def test_unknown_when_r_is_absent(self, monkeypatch):
        monkeypatch.setattr(bridge, "rscript_available", lambda: (False, "not on PATH"))
        s = bridge.runtime_status()
        assert s["available"] is False
        assert s["backend"] == "none"
        assert s["r_version_matches_lockfile"] is None
        assert "version_conflict" not in s

    def test_reports_which_interpreter_ran(self, monkeypatch):
        monkeypatch.setattr(CONFIG, "rscript", "/opt/R/4.4.2/bin/Rscript")
        s = self._status(monkeypatch, "Rscript (R) version 4.4.2 (2024-10-31)")
        assert s["local_r"]["executable"] == "/opt/R/4.4.2/bin/Rscript"
        assert s["local_r"]["pinned"] is True
