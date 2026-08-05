#!/usr/bin/env python
"""Finish the R 4.4 setup once `rig add 4.4` has run.

Everything here is unprivileged; the two steps that need admin rights
(installing rig, installing R itself) are deliberately not attempted:

    brew install --cask rig
    rig add 4.4

Run this afterwards:

    .venv/bin/python scripts/setup_r44.py

It finds the R 4.4 interpreter rig installed, restores renv.lock against it
into the shared managed library, and verifies every library the vetted scripts
load. It is idempotent -- re-running it re-verifies without rebuilding.
"""

from __future__ import annotations

import argparse
import glob
import re
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skinmcp.config import CONFIG  # noqa: E402
from skinmcp.runtimes import bridge  # noqa: E402

#: What the vetted scripts under runtimes/r/scripts actually load.
NEEDED = ["CellChat", "DESeq2", "Matrix", "Seurat", "SingleCellExperiment",
          "celda", "ggplot2", "jsonlite", "miloR", "scDblFinder"]


def _version_of(exe: str) -> str:
    """The R minor version `exe` reports, or "" if it is not a working Rscript."""
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return ""
    m = re.search(r"(\d+\.\d+)\.\d+", (p.stdout or p.stderr))
    return m.group(1) if m else ""


def find_rscript(want: str) -> str:
    """Locate an Rscript reporting `want` (e.g. "4.4").

    Searched rather than hardcoded because rig's layout varies: quick links are
    named after the version, and on arm64 macs the framework directory may or
    may not carry an `-arm64` suffix depending on whether both architectures
    are installed.
    """
    candidates: list[str] = []
    for name in (f"Rscript-{want}", f"Rscript{want}"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    candidates += sorted(glob.glob(
        f"/Library/Frameworks/R.framework/Versions/{want}*/Resources/bin/Rscript"))
    on_path = shutil.which("Rscript")
    if on_path:
        candidates.append(on_path)          # last: only if it happens to be right

    for c in candidates:
        if _version_of(c) == want:
            return c
    return ""


LMSTUDIO_CFG = Path.home() / ".lmstudio/apps/bionic/.internal/ng-mcp.json"


def wire_lmstudio(rscript: str, cfg: Path = LMSTUDIO_CFG) -> str:
    """Put SKINMCP_RSCRIPT into the skin-mcp entry of LM Studio's launch config.

    Backs the file up first and rewrites only that one env key, because this is
    someone else's application config and it also holds unrelated servers.
    """
    import json

    if not cfg.is_file():
        return f"no LM Studio config at {cfg}; set SKINMCP_RSCRIPT yourself"
    try:
        data = json.loads(cfg.read_text(encoding="utf-8"))
    except (OSError, ValueError) as e:
        return f"could not read {cfg}: {e}"

    servers = [s for s in data.get("servers", []) if s.get("name") == "skin-mcp"]
    if not servers:
        return f"no server named 'skin-mcp' in {cfg}; set SKINMCP_RSCRIPT yourself"

    changed = False
    for s in servers:
        env = s.setdefault("connection", {}).setdefault("env", {})
        if env.get("SKINMCP_RSCRIPT") != rscript:
            env["SKINMCP_RSCRIPT"] = rscript
            changed = True
    if not changed:
        return "already set; no change"

    backup = cfg.with_suffix(f".json.bak-{__import__('time').strftime('%Y%m%d-%H%M%S')}")
    try:
        shutil.copy2(cfg, backup)
        cfg.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    except OSError as e:
        return f"could not write {cfg}: {e}"
    note = f"set SKINMCP_RSCRIPT (backup: {backup.name})"
    if any(s.get("enabled") is False for s in servers):
        note += "\n  NOTE: this server is currently disabled in LM Studio -- re-enable it"
    return note


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify-only", action="store_true",
                    help="check the library without restoring")
    ap.add_argument("--no-wire", action="store_true",
                    help="do not touch the LM Studio launch config")
    args = ap.parse_args()

    want = bridge.lockfile_r_version().rsplit(".", 1)[0] or "4.4"
    print(f"renv.lock targets R {want}.x")

    rscript = find_rscript(want)
    if not rscript:
        print(f"\nNo R {want}.x found. The two steps that need admin rights have "
              f"not been run yet:\n\n    brew install --cask rig\n    rig add {want}\n\n"
              f"Then re-run this script.", file=sys.stderr)
        return 2
    print(f"found R {want}: {rscript}")

    # Everything below drives that interpreter specifically, never PATH's.
    CONFIG.rscript = rscript

    ok, ver = bridge.rscript_available()
    status = bridge.runtime_status()
    if not ok or status.get("r_version_matches_lockfile") is not True:
        print(f"\n{status.get('version_conflict', ver)}", file=sys.stderr)
        return 3
    print(f"version check passed: {ver}")

    lib = CONFIG.shared_renv() / "library"
    if not args.verify_only:
        print(f"\nrestoring renv.lock into {lib}\n(binaries come from Bioconductor's "
              f"big-sur-arm64 repo, so this is a download, not a compile)\n")
        from skinmcp.tools import _base, memory_tools, runtime_tools

        # Goes through skin.runtime.create rather than shelling out to renv
        # directly, so the restore is the same code path the server uses and
        # lands in the provenance log like any other step.
        if not _base.get_active_project():
            memory_tools.open_project(name="runtime_setup", organism="mouse")
        res = runtime_tools.create(kind="r", force=False)
        if not res.get("ok"):
            err = res.get("error") or {}
            print(f"\nrestore failed: {err.get('message')}\n{err.get('remedy', '')}",
                  file=sys.stderr)
            return 4
        print(f"restore finished: {res.get('summary')}")

    # The real test is not "did renv exit 0" but "do the scripts' libraries load".
    print("\nverifying the libraries the vetted scripts load:")
    # The package list is baked into the expression rather than passed as
    # trailing argv. `Rscript -e expr --args pkg...` puts the literal "--args"
    # into commandArgs(TRUE), and requireNamespace("--args") raises rather than
    # returning FALSE, which aborted the loop before it checked anything.
    # tryCatch for the same reason: one bad name must not hide the other ten.
    pkgs = ", ".join(repr(n) for n in NEEDED).replace("'", '"')
    expr = (f".libPaths(c('{lib}', .libPaths())); "
            f"for (p in c({pkgs})) {{ "
            "ok <- tryCatch(requireNamespace(p, quietly=TRUE), error=function(e) FALSE); "
            "cat(p, if (isTRUE(ok)) 'OK' else 'MISSING', '\\n') }")
    # Through rscript_cmd, not a bare [rscript, "-e", ...]: a user ~/.Rprofile
    # that calls .libPaths() puts another R's library on the path, and loading a
    # package built by a different R segfaults. Verifying under different startup
    # conditions than the server uses would also be testing the wrong thing.
    p = subprocess.run(bridge.rscript_cmd("-e", expr),
                       capture_output=True, text=True, timeout=1800)
    missing, checked = [], set()
    for line in p.stdout.splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] in NEEDED:
            print(f"  {parts[0]:<22} {parts[1]}")
            checked.add(parts[0])
            if parts[1] == "MISSING":
                missing.append(parts[0])
    # A package that produced no verdict at all is not "fine" -- say so.
    unchecked = [n for n in NEEDED if n not in checked]
    if unchecked:
        print(f"  (no verdict for: {', '.join(unchecked)})")
        missing += unchecked
    if p.returncode != 0:
        print(p.stderr[-1500:], file=sys.stderr)

    print("\n" + "=" * 68)
    if missing:
        print(f"STILL MISSING: {', '.join(missing)}")
        print("Those R-backed tools will fall back to Python; the rest work.")
    else:
        print("All libraries load.")

    if args.no_wire:
        print(f"\nPoint the server at this interpreter yourself:\n\n"
              f"    SKINMCP_RSCRIPT={rscript}\n")
    else:
        print(f"\nLM Studio config: {wire_lmstudio(rscript)}")
        print(f"  SKINMCP_RSCRIPT={rscript}")
        print("\nRestart the skin-mcp server in LM Studio to pick this up.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
