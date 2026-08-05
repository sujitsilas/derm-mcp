"""`skin.runtime.*` — runtime status, R container management, version manifest.

`manifest` is a deliverable, not a debug tool: it is what you paste into a
methods section, and what answers "which version of scDblFinder produced this?"
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..config import CONFIG
from ..errors import BadParam, RuntimeUnavailable
from ..memory import store
from ..runtimes import bridge, python_manifest
from ._base import Ctx, tool

logger = logging.getLogger(__name__)


@tool("skin.runtime.status", category="runtime",
      summary="Python and R runtime availability, image digest, vetted script list.")
def status(project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Report what runtimes are available right now, and what to do if R is not.

    Args:
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    r = bridge.runtime_status()
    shared_py = CONFIG.shared_python()
    shared_lib = CONFIG.shared_renv() / "library"
    ctx.summary = {
        "project_dir": str(CONFIG.project_dir(ctx.project_id)),
        "python": {"packages": python_manifest.capture(), "offline": CONFIG.offline,
                   "profile": CONFIG.profile, "running_under": sys.executable,
                   "managed_venv": str(shared_py) if shared_py.exists() else None},
        "r": {**r, "working_dir": str(CONFIG.project_dir(ctx.project_id)),
              "managed_library": str(shared_lib) if shared_lib.is_dir() else None},
        "python_fallbacks": {
            "skin.doublet.call(method='scdblfinder')": "method='scrublet'",
            "skin.abundance.milo_r": "skin.abundance.milo_py",
            "skin.de.deseq2_r": "skin.de.pseudobulk (PyDESeq2)",
            "skin.ccc.cellchat_r": "skin.ccc.liana",
            "skin.qc.estimate_ambient": "no pure-Python equivalent; "
                                        "skin.annotate.regress_markers is a partial, "
                                        "explicitly-labelled substitute",
        },
    }
    if not r["available"]:
        ctx.warn("No R backend. Every R-backed tool will return a typed RUNTIME_UNAVAILABLE "
                 "naming its Python fallback rather than a traceback. Build one with "
                 "skin.runtime.create(kind='r').")
    ctx.suggest("skin.runtime.create", "skin.runtime.manifest")


@tool("skin.runtime.create", category="runtime",
      summary="Build the managed Python (uv) or R (renv) environment. Shared; build once.")
def create(kind: str = "r", backend: str = "", force: bool = False,
           project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Build one of the two managed runtimes, shared by every project.

    Both live under `{project_root}/runtimes` rather than inside a project:
    building them costs minutes and hundreds of MB, and on a single-user
    workstation there is no reason to pay that again per project.

    Safe to call at any time — if the environment already exists this returns
    immediately without rebuilding. R-backed tools still execute with the
    *calling project's* directory as their working directory, so outputs stay
    with the analysis that produced them.

    Args:
        kind: "python" (uv venv with skin-mcp installed) or "r" (renv library
            restored from the committed renv.lock).
        backend: Unused; kept for call compatibility.
        force: Rebuild even if it already exists.
        project_id: Defaults to the active project.
        dry_run: Report the command without running it.
        seed: Unused.
    """
    if kind not in ("python", "r"):
        raise BadParam("kind must be python|r")
    if kind == "python":
        _create_python(ctx, force, dry_run)
        return
    _create_renv(ctx, force, dry_run)
    return


def _create_python(ctx: Ctx, force: bool, dry_run: bool) -> None:
    """The one managed uv venv, shared by every project."""
    uv = shutil.which("uv")
    venv = CONFIG.shared_venv()
    py = CONFIG.shared_python()
    repo = Path(__file__).resolve().parents[3]
    cmds = [[uv or "uv", "venv", str(venv)],
            [uv or "uv", "pip", "install", "--python", str(py), "-e", str(repo)]]
    ctx.code = "\n".join(" ".join(c) for c in cmds) + "\n"
    if dry_run:
        ctx.summary = {"kind": "python", "venv": str(venv), "exists": py.exists(),
                       "commands": cmds}
        return
    if not uv:
        raise RuntimeUnavailable(
            "uv is not on PATH",
            remedy="Install uv (https://docs.astral.sh/uv/). Until then the server "
                   "runs analysis in its own interpreter, which skin.runtime.manifest "
                   "records either way.")
    if py.exists() and not force:
        # Idempotent on purpose: an agent may call this mid-analysis, and the
        # shared env is meant to be built once and then left alone.
        ctx.summary = {"kind": "python", "already_exists": True, "python": str(py)}
        return
    log = CONFIG.ensure_project_dirs(ctx.project_id) / "logs" / "runtime_python.log"
    for cmd in cmds:
        code, tail = bridge._run(cmd, log, timeout=3600,
                                 cwd=CONFIG.project_dir(ctx.project_id))
        if code != 0:
            raise RuntimeUnavailable(f"{' '.join(cmd[:3])} exited {code}",
                                     remedy=f"Log tail:\n{tail[-800:]}")
    ctx.add_artifact("log", log, caption="project python runtime build log")
    ctx.summary = {"kind": "python", "venv": str(venv), "python": str(py)}
    ctx.suggest("skin.runtime.status", "skin.runtime.manifest")


def _check_lock_matches_r(lock: Path, installed: str) -> None:
    """Refuse a restore that cannot possibly succeed, before spending minutes on it.

    renv.lock pins an R version and a Bioconductor release, and Bioconductor
    only publishes binaries for the R version it was cut against. Restoring a
    Bioc 3.20 lock (R 4.4) under R 4.6 fails on a 404 for
    `.../3.20/bioc/bin/macosx/.../4.6/PACKAGES` after installing nothing — an
    error that says "package 'BiocManager' is not available" and gives no hint
    that the R version is the problem.
    """
    import json
    import re

    try:
        meta = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return
    want = str((meta.get("R") or {}).get("Version") or "")
    have = (re.search(r"(\d+\.\d+)\.\d+", installed or "") or [None, ""])[1]
    bioc = str((meta.get("Bioconductor") or {}).get("Version") or "")
    if not want or not have or want.rsplit(".", 1)[0] == have:
        return
    raise RuntimeUnavailable(
        f"renv.lock targets R {want} but R {have}.x is installed",
        remedy=(
            f"Bioconductor {bioc or 'in this lockfile'} publishes no binaries for "
            f"R {have}, so the restore fails on a 404 having installed nothing. "
            f"Either install R {want.rsplit('.', 1)[0]}.x and point PATH at it, or "
            f"regenerate renv.lock against R {have} — that changes the pinned "
            f"package versions, so it is a reproducibility decision, not a fix to "
            f"apply silently. Until then every R-backed tool falls back to Python "
            f"(skin.runtime.status lists each one)."),
        suggested_tool="skin.runtime.status",
    )


def _r_minor(version_text: str) -> str:
    """"4.4" out of anything reporting an R version, or "" if absent."""
    import re

    m = re.search(r"(\d+\.\d+)\.\d+", version_text or "")
    return m.group(1) if m else ""


def _library_r_minor(lib: Path) -> str:
    """The R minor version the packages in `lib` were built by.

    Every installed package records `Built: R x.y.z` in its DESCRIPTION. A
    handful is enough to tell whether the library belongs to the R we are about
    to run, and reading all of them costs hundreds of file opens for no more
    certainty.
    """
    seen: dict[str, int] = {}
    for pkg in sorted(p for p in lib.iterdir() if p.is_dir())[:8]:
        try:
            text = (pkg / "DESCRIPTION").read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
            if line.startswith("Built:"):
                v = _r_minor(line)
                if v:
                    seen[v] = seen.get(v, 0) + 1
                break
    return max(seen, key=lambda k: seen[k]) if seen else ""


def _lock_satisfied_by(lock: Path, lib: Path) -> bool:
    """Is every package the lockfile names actually installed in `lib`?

    "The directory is not empty" is not the same question. A restore that dies
    partway leaves renv and BiocManager behind, and treating that as a built
    runtime made this tool report success on a library holding 2 of ~150
    packages -- after which every vetted script failed at `library()`.
    """
    import json

    try:
        pkgs = json.loads(lock.read_text(encoding="utf-8")).get("Packages", {})
    except (OSError, ValueError):
        return False
    return bool(pkgs) and all((lib / name).is_dir() for name in pkgs)


def _unretrievable(log_tail: str) -> tuple[str, str] | None:
    """The (package, version) renv gave up retrieving, if that is why it failed."""
    import re

    m = re.search(r"failed to retrieve package '(?:bioc::)?([A-Za-z0-9._]+)@([0-9.\-]+)'",
                  log_tail or "")
    return (m.group(1), m.group(2)) if m else None


def _unbuildable(log_tail: str) -> str | None:
    """The package whose *build* failed, if that is why the restore stopped.

    Distinct from `_unretrievable`: the tarball arrived, the compile or the
    post-install load test failed. Here that was xgboost — a transitive
    dependency of scDblFinder — whose OpenMP link silently produced a shared
    object with no libomp dependency, failing at load with
    "symbol not found in flat namespace '___kmpc_dispatch_deinit'".

    One unbuildable transitive dependency must not cost the caller Seurat,
    DESeq2, miloR and CellChat. Deferring it completes the rest of the library
    and confines the loss to the tools that actually need it, each of which has
    a documented Python fallback.
    """
    import re

    m = re.search(r"install of package '([A-Za-z0-9._]+)' failed", log_tail or "")
    return m.group(1) if m else None


def _bioc_release(lock: Path) -> str:
    import json

    try:
        meta = json.loads(lock.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((meta.get("Bioconductor") or {}).get("Version") or "")


def _install_from_bioc_archive(pkg: str, version: str, lock: Path, lib: Path,
                               log: Path, cwd: Path) -> bool:
    """Install one exact Bioconductor version from the source archive.

    Only ever called *after* the rest of the lockfile is in place: an archived
    tarball is source-only and `repos=NULL` disables dependency resolution, so
    installing celda first reports its 30 imports "not available" and rolls
    back. With the library otherwise complete, those imports are already there.
    """
    release = _bioc_release(lock)
    if not release:
        return False
    url = (f"https://bioconductor.org/packages/{release}/bioc/src/contrib/"
           f"Archive/{pkg}/{pkg}_{version}.tar.gz")
    expr = (f".libPaths(c('{lib}', .libPaths())); "
            f"install.packages('{url}', repos=NULL, type='source', lib='{lib}')")
    logger.info("fetching %s %s from the Bioconductor %s source archive", pkg, version,
                release)
    code, _ = bridge._run(bridge.rscript_cmd("-e", expr), log, timeout=3600, cwd=cwd)
    return code == 0 and (lib / pkg).is_dir()


def _github_env() -> dict[str, str]:
    """The environment for a restore, with a GitHub token if one can be found.

    A lockfile may pin GitHub-sourced packages (CellChat here). renv fetches
    those through api.github.com, which rate-limits unauthenticated callers to
    60 requests an hour per IP — so the restore fails on a network error that
    says nothing about rate limits. If the user has an authenticated `gh`, its
    token is already the credential they would have typed, so borrow it rather
    than asking. Never logged or persisted; it lives in the child's env.
    """

    env, dropped = bridge.clean_build_env()
    if dropped:
        logger.warning("dropped stale toolchain vars %s before installing R packages",
                       dropped)
    if env.get("GITHUB_PAT") or env.get("GITHUB_TOKEN"):
        return env
    gh = shutil.which("gh")
    if not gh:
        return env
    try:
        p = subprocess.run([gh, "auth", "token"], capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.TimeoutExpired):
        return env
    token = (p.stdout or "").strip()
    if p.returncode == 0 and token:
        env["GITHUB_PAT"] = token
        logger.info("using the gh CLI's token for GitHub-sourced packages")
    return env


#: Where Homebrew puts libomp on arm64 and x86_64 macOS.
_LIBOMP_PREFIXES = ("/opt/homebrew/opt/libomp", "/usr/local/opt/libomp")


def _openmp_makevars(out_dir: Path) -> Path | None:
    """A Makevars supplying OpenMP flags, if a libomp is installed.

    CRAN's macOS R ships with SHLIB_OPENMP_*FLAGS empty, so any package using
    OpenMP either skips it or, worse, rolls its own detection. xgboost's
    configure found Homebrew's libomp and emitted `-lomp` — but R links shared
    objects with `-undefined dynamic_lookup`, so when that `-lomp` did not
    resolve the link still *succeeded*, producing a .so with no libomp
    dependency. It failed only at load, with "symbol not found in flat
    namespace '___kmpc_dispatch_deinit'", which reads like a broken libomp
    rather than one that was never linked.

    Supplying the flags here, with an explicit -rpath so the loader can find
    the library at run time, is CRAN's documented remedy. Written to the
    managed runtime directory and passed as R_MAKEVARS_USER, so the user's own
    ~/.R/Makevars is neither read nor overwritten.
    """
    prefix = next((p for p in _LIBOMP_PREFIXES if (Path(p) / "lib/libomp.dylib").exists()),
                  None)
    if not prefix:
        return None
    out = out_dir / "Makevars-openmp"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        f"SHLIB_OPENMP_CFLAGS = -Xpreprocessor -fopenmp -I{prefix}/include\n"
        f"SHLIB_OPENMP_CXXFLAGS = -Xpreprocessor -fopenmp -I{prefix}/include\n"
        f"PKG_LIBS += -L{prefix}/lib -lomp -Wl,-rpath,{prefix}/lib\n",
        encoding="utf-8")
    return out


def _lock_without(lock: Path, drop: list[str], out_dir: Path) -> Path:
    """A copy of the lockfile with `drop` removed, for a partial restore.

    renv aborts the whole restore over a single unretrievable package, so the
    only way to get the other 150 installed is to stop asking for that one.
    """
    import json

    meta = json.loads(lock.read_text(encoding="utf-8"))
    meta["Packages"] = {k: v for k, v in meta.get("Packages", {}).items() if k not in drop}
    out = out_dir / "renv-partial.lock"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return out


def _create_renv(ctx: Ctx, force: bool, dry_run: bool) -> None:
    """The one managed renv library, restored from the committed renv.lock."""
    lock = Path(bridge.R_DIR) / "renv.lock"
    lib = CONFIG.shared_renv() / "library"
    ok, ver = bridge.rscript_available()
    # renv must be installed INTO the managed library and the library must be
    # *prepended* to the search path. `.libPaths(x)` replaces it, so installing
    # renv to the default user library and then switching paths left
    # "there is no package called 'renv'" — the reason this never once built.
    # No pkgType override: R's per-platform default already prefers binaries
    # where they exist ("both" on macOS/Windows) and source where they do not
    # (Linux). Forcing "binary" would break every Linux install.
    def _restore_expr(from_lock: Path) -> str:
        return (f"dir.create('{lib}', recursive=TRUE, showWarnings=FALSE); "
                f".libPaths(c('{lib}', .libPaths())); "
                f"if (!requireNamespace('renv', quietly=TRUE)) "
                f"install.packages('renv', lib='{lib}', "
                f"repos='https://cloud.r-project.org'); "
                f"renv::restore(lockfile='{from_lock}', library='{lib}', prompt=FALSE)")

    expr = _restore_expr(lock)
    cmd = bridge.rscript_cmd("-e", expr)
    ctx.code = f"Rscript -e \"renv::restore(lockfile='{lock}', library='{lib}')\"\n"
    if dry_run:
        ctx.summary = {"kind": "r", "library": str(lib), "rscript_available": ok,
                       "lockfile": str(lock)}
        return
    if not ok:
        raise RuntimeUnavailable(
            "Rscript is not on PATH",
            remedy="Install R 4.4.x. Every R-backed tool has a pure-Python fallback "
                   "listed in skin.runtime.status.",
            suggested_tool="skin.runtime.status")
    if lib.is_dir() and any(lib.iterdir()) and not force:
        # Idempotent, same reason as the Python venv above -- but "non-empty" is
        # not the same as "usable". R refuses to load a package built by an R
        # with different internals, so a library left behind by a previous R
        # minor version is not a built runtime, it is an obstacle. Treating it
        # as "already_exists" made this tool report success while installing
        # nothing, and every R-backed tool then failed at `library()` instead.
        built = _library_r_minor(lib)
        running = _r_minor(ver)
        if built and running and built != running:
            stale = lib.with_name(f"library-r{built}-{store.now()[:10]}")
            try:
                lib.rename(stale)
            except OSError as e:
                raise RuntimeUnavailable(
                    f"the managed library holds packages built for R {built}, but this "
                    f"server runs R {running}, and it could not be moved aside: {e}",
                    remedy=f"Move or delete {lib} by hand, then re-run this tool.") from e
            ctx.warn(f"The managed library held packages built for R {built} while this "
                     f"server runs R {running}; R cannot load those. Moved to {stale.name} "
                     f"and rebuilding from the lockfile. Delete it once you are satisfied.")
        elif _lock_satisfied_by(lock, lib):
            ctx.summary = {"kind": "r", "already_exists": True, "library": str(lib),
                           "r_minor": built or "unknown"}
            return
        # Otherwise the library exists but is incomplete -- an abandoned restore
        # leaves renv and BiocManager behind and nothing else. Falling through
        # resumes it; renv skips whatever is already installed, so this costs
        # nothing when the library is nearly complete.
    _check_lock_matches_r(lock, ver)
    lib.mkdir(parents=True, exist_ok=True)
    log = CONFIG.ensure_project_dirs(ctx.project_id) / "logs" / "runtime_renv.log"
    proj = CONFIG.project_dir(ctx.project_id)
    renv_env = _github_env()
    mk = _openmp_makevars(CONFIG.shared_renv())
    if mk:
        renv_env["R_MAKEVARS_USER"] = str(mk)
    code, tail = bridge._run(cmd, log, timeout=7200, cwd=proj, env=renv_env)

    # A Bioconductor release branch keeps taking point updates while serving
    # exactly one version per package, and its *binary* repo has no archive at
    # all -- so a lockfile written early in a release cycle asks for versions the
    # branch no longer offers. renv then aborts the ENTIRE restore over one such
    # package (here: celda 1.22.0, where the branch had moved to 1.22.1).
    #
    # The pinned tarballs are still served from <repo>/bioc/src/contrib/Archive,
    # so the pins can be honoured exactly rather than silently drifting to
    # whatever the branch serves today. But they are source-only and install
    # with repos=NULL, which does no dependency resolution -- celda alone needs
    # 30 imports. So they have to come last: defer the stragglers, restore
    # everything else, then fetch them once their dependencies exist.
    #
    # A package that fails to *build* is dropped the same way but not retried
    # from the archive: the tarball was never the problem. Its dependents fail
    # next and get dropped in turn, so the loop converges on "everything that
    # can be built", and only the tools needing that subtree fall back.
    deferred: list[tuple[str, str]] = []
    unbuildable: list[str] = []
    while code != 0 and len(deferred) + len(unbuildable) < 12:
        stuck = _unretrievable(tail)
        broke = None if stuck else _unbuildable(tail)
        if stuck and stuck not in deferred:
            deferred.append(stuck)
        elif broke and broke not in unbuildable:
            unbuildable.append(broke)
        else:
            break
        drop = [p for p, _ in deferred] + unbuildable
        partial = _lock_without(lock, drop, proj / "runtimes")
        code, tail = bridge._run(bridge.rscript_cmd("-e", _restore_expr(partial)),
                                 log, timeout=7200, cwd=proj, env=renv_env)

    for pkg in unbuildable:
        ctx.warn(f"{pkg} could not be built on this machine and was left out, along with "
                 f"anything that depends on it. The rest of the library is installed. "
                 f"Tools needing {pkg} return RUNTIME_UNAVAILABLE naming their Python "
                 f"fallback; see {log}.")

    if code == 0 and deferred:
        arch_log = log.with_name("runtime_renv_archive.log")
        for pkg, version in deferred:
            if _install_from_bioc_archive(pkg, version, lock, lib, arch_log, proj):
                ctx.warn(f"{pkg} {version} is pinned in the lockfile but the "
                         f"Bioconductor {_bioc_release(lock)} branch has moved past it; "
                         f"installed from the source archive so the pin stays exact.")
            else:
                ctx.warn(f"{pkg} {version} could not be installed: the Bioconductor "
                         f"{_bioc_release(lock)} branch no longer serves it and the source "
                         f"archive did not yield a working build. Tools needing {pkg} will "
                         f"fall back to Python; see {arch_log}.")
        ctx.add_artifact("log", arch_log, caption="Bioconductor archive install log")
    ctx.add_artifact("log", log, caption="renv restore log")
    if code != 0:
        # R itself needs no compiler; only *source* installs do, and renv falls
        # back to source when no binary exists for a package. Say so, rather than
        # leaving the reader to decode a toolchain error from the log tail.
        hint = ""
        if "config CC" in tail or "developer tools" in tail or "gcc" in tail:
            hint = ("\n\nThis failed compiling a package from source, which needs a C "
                    "toolchain: on macOS `xcode-select --install`, on Debian/Ubuntu "
                    "`apt install build-essential gfortran`. Binaries avoid it where "
                    "the platform publishes them.")
        raise RuntimeUnavailable(f"renv::restore exited {code}",
                                 remedy=f"Log tail:\n{tail[-900:]}{hint}")
    ctx.summary = {"kind": "r", "library": str(lib), "r_version": ver,
                   "log_tail": tail[-400:]}
    ctx.suggest("skin.runtime.status", "skin.runtime.manifest")


@tool("skin.runtime.manifest", category="runtime",
      summary="Full version table for both runtimes. A deliverable, not a debug tool.")
def manifest(project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """The version table for a methods section: Python packages, R image, pins.

    Args:
        project_id: Defaults to the active project.
        dry_run: No effect.
        seed: Unused.
    """
    import json

    r = bridge.runtime_status()
    # Enumerate the managed renv library so the manifest reports the versions
    # that actually ran, not whatever is in the user library.
    r_pkgs: dict[str, str] = {}
    lib = CONFIG.shared_renv() / "library"
    if r["available"]:
        try:
            expr = (f".libPaths('{lib}'); "
                    "ip <- installed.packages()[, c('Package','Version')]; "
                    "cat(paste(ip[,1], ip[,2], sep='=', collapse='\\n'))")
            proc = subprocess.run(bridge.rscript_cmd("-e", expr),
                                  capture_output=True, text=True, timeout=120)
            for line in proc.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    r_pkgs[k.strip()] = v.strip()
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("could not enumerate R packages: %s", e)

    m = python_manifest.full_manifest()
    m["r"] = {"backend": r["backend"], "available": r["available"],
              "library": str(lib) if lib.is_dir() else None,
              "packages": r_pkgs, "vetted_scripts": r["scripts"]}
    m["skinmcp"] = {"profile": CONFIG.profile, "offline": CONFIG.offline,
                    "allow_raw_exec": CONFIG.allow_raw_exec,
                    "project_root": str(CONFIG.project_root),
                    "python_runtime": "uv", "r_runtime": "renv"}
    m.pop("pinned", None) if False else None
    m["monocle_note"] = (
        "py-monocle (github.com/bioturing-org/py-monocle) publishes no license, so it is not "
        "vendored. skin.traj.monocle uses the upstream package when installed and otherwise "
        "a shipped implementation of the same published algorithm (SimplePPT, Mao et al. "
        "SDM 2015; monocle3, Cao et al. Nature 2019). The return field `implementation` "
        "records which one ran.")

    if not dry_run:
        p = CONFIG.ensure_project_dirs(ctx.project_id) / "runtimes" / "manifest.json"
        p.write_text(json.dumps(m, indent=2, default=str), encoding="utf-8")
        ctx.add_artifact("text", p, caption="full runtime manifest")
        ctx.summary = {"path": str(p), **_manifest_summary(m)}
    else:
        ctx.summary = _manifest_summary(m)
    ctx.suggest("skin.export.methods_paragraph", "skin.export.bundle")


def _manifest_summary(m: dict[str, Any]) -> dict[str, Any]:
    return {
        "python_version": m["python"]["version"],
        "platform": m["python"]["platform"],
        "key_packages": {k: v for k, v in m["python"]["packages"].items()
                         if k in ("scanpy", "anndata", "numpy", "scipy", "pydeseq2",
                                  "harmonypy", "gseapy", "leidenalg", "mcp", "skin-mcp")},
        "r": {"backend": m["r"]["backend"], "available": m["r"]["available"],
              "n_r_packages": len(m["r"]["packages"])},
        "pinned": m["pinned"],
    }


@tool("skin.runtime.exec_r", category="runtime", needs_r=True,
      summary="Run a named, vetted R script against a handle. No arbitrary code.")
def exec_r(script_id: str, dataset_id: str = "", params: dict[str, Any] | None = None,
           timeout_s: int = 3600, project_id: str = "", dry_run: bool = False,
           seed: int = 0, *, ctx: Ctx) -> None:
    """Execute one of the vetted scripts under runtimes/r/scripts.

    The model supplies a script id and typed parameters, never R source.
    Arbitrary R is a separate, opt-in tool.

    Args:
        script_id: Script name without the .R extension. skin.runtime.status lists them.
        dataset_id: Handle to pass in as input.h5ad. Optional for scripts that
            read from a path parameter.
        params: Typed parameters written to params.json for the script.
        timeout_s: Wall-clock limit.
        project_id: Defaults to the active project.
        dry_run: Report the script and params without running.
        seed: RNG seed, forwarded to the script.
    """
    from .. import registry

    avail = bridge.available_scripts()
    if script_id not in avail:
        from ..errors import NotFound

        raise NotFound(f"no vetted script named {script_id!r}",
                       remedy=f"Available: {avail}")
    if dry_run:
        ctx.summary = {"script_id": script_id, "params": params or {},
                       "runtime": bridge.runtime_status()["backend"]}
        return
    adata = registry.load(ctx.project_id, dataset_id) if dataset_id else None
    res = bridge.run_r_script(script_id, adata=adata, project_id=ctx.project_id,
                              params={**(params or {}), "seed": seed}, timeout=timeout_s)
    ctx.dataset_id = store.resolve_dataset_ref(ctx.project_id, dataset_id) if dataset_id else None
    log_tail = res.pop("log", "")
    ctx.summary = {"script_id": script_id, "result": res, "log_tail": log_tail[-600:]}


@tool("skin.runtime.exec_r_raw", category="runtime", needs_r=True, destructive=True,
      summary="Run arbitrary R. Disabled unless the server was started with --allow-raw-exec.")
def exec_r_raw(code: str, dataset_id: str = "", timeout_s: int = 900, confirm: bool = False,
               project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Escape hatch for arbitrary R. Off by default, and deliberately awkward.

    Nothing this produces is reproducible from the exported notebook unless you
    paste the code into it yourself, so prefer adding a vetted script.

    Args:
        code: R source. The handle, if given, is written to /work/input.h5ad.
        dataset_id: Optional handle to make available to the script.
        timeout_s: Wall-clock limit.
        confirm: Required.
        project_id: Defaults to the active project.
        dry_run: Report whether raw exec is enabled.
        seed: Unused.
    """
    from .. import registry
    from ._base import confirm_or_raise

    if dry_run:
        ctx.summary = {"enabled": CONFIG.allow_raw_exec, "n_chars": len(code)}
        return
    if not CONFIG.allow_raw_exec:
        raise BadParam(
            "arbitrary R execution is disabled",
            remedy="Restart the server with --allow-raw-exec. Vetted scripts "
                   f"({bridge.available_scripts()}) run without it via skin.runtime.exec_r.",
            suggested_tool="skin.runtime.exec_r")
    confirm_or_raise(confirm, dry_run, "skin.runtime.exec_r_raw",
                     "This runs unreviewed R code inside the runtime with access to the "
                     "project working directory.")
    adata = registry.load(ctx.project_id, dataset_id) if dataset_id else None
    res = bridge.exec_r_raw(code, project_id=ctx.project_id, adata=adata, timeout=timeout_s)
    ctx.warn("Raw R output is not reproducible from the exported notebook. Move anything "
             "you intend to keep into a vetted script under runtimes/r/scripts/.")
    ctx.summary = {"exit_code": res["exit_code"], "log_tail": res["log"][-800:],
                   "work_dir": res["work_dir"]}


#: File kinds worth registering when user code leaves them in the output dir.
_CAPTURED = {".png": "figure", ".pdf": "figure", ".svg": "figure",
             ".csv": "table", ".tsv": "table", ".h5ad": "h5ad"}


def _snapshot(root: Path) -> dict[Path, float]:
    """Every capturable file under `root`, with its mtime."""
    out: dict[Path, float] = {}
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in _CAPTURED:
            try:
                out[p] = p.stat().st_mtime
            except OSError:
                pass
    return out


@tool("skin.runtime.exec_python", category="runtime", destructive=True,
      summary="Run arbitrary Python in the managed venv. Needs --allow-raw-exec.")
def exec_python(code: str, dataset_id: str = "", timeout_s: int = 600,
                confirm: bool = False, project_id: str = "", dry_run: bool = False,
                seed: int = 0, *, ctx: Ctx) -> None:
    """Escape hatch for analysis the typed tools do not cover.

    Runs in a **subprocess** against the managed venv, never in the server
    process: a crash, an OOM or a runaway loop kills the child and leaves the
    server answering. Prefer a typed tool when one exists — it is reproducible
    from the exported notebook and this is only as reproducible as the code you
    pasted, which is recorded verbatim.

    The script is handed two names:
      `adata`  — the handle, already read from disk (only if `dataset_id` given)
      `OUTDIR` — this project's output directory

    Anything written into `OUTDIR` (.png/.pdf/.svg/.csv/.tsv/.h5ad) is picked up
    afterwards and registered as an artifact, so figures and tables show up in
    the briefing like any other result.

    Args:
        code: Python source.
        dataset_id: Optional handle to bind as `adata`.
        timeout_s: Wall-clock limit; the child is killed past it.
        confirm: Required, because this runs unreviewed code.
        project_id: Defaults to the active project.
        dry_run: Report whether raw exec is enabled, without running anything.
        seed: Seeds `random` and numpy in the child.
    """
    from ._base import confirm_or_raise

    py = CONFIG.shared_python()
    if dry_run:
        ctx.summary = {"enabled": CONFIG.allow_raw_exec, "n_chars": len(code),
                       "python": str(py), "venv_exists": py.exists()}
        return
    if not CONFIG.allow_raw_exec:
        raise BadParam(
            "arbitrary Python execution is disabled",
            remedy="Restart the server with --allow-raw-exec. Most analysis does not "
                   "need it — skin.abundance.proportions, skin.plot.* and skin.de.* "
                   "cover the common cases and are reproducible from the notebook.",
            suggested_tool="skin.help.workflow")
    if not py.exists():
        raise RuntimeUnavailable(
            "the managed Python runtime has not been built",
            remedy="Run skin.runtime.create(kind='python') once. It is shared by every "
                   "project, so this is a one-time cost.",
            suggested_tool="skin.runtime.create")
    confirm_or_raise(confirm, dry_run, "skin.runtime.exec_python",
                     "This runs unreviewed Python with access to your data and "
                     "project directory.")

    outdir = ctx.outroot()
    resolved = store.resolve_dataset_ref(ctx.project_id, dataset_id) if dataset_id else ""
    h5ad = ""
    if dataset_id:
        if not resolved:
            from .. import registry

            raise registry.bad_handle(ctx.project_id, dataset_id)
        h5ad = (store.get_dataset(ctx.project_id, resolved) or {}).get("path", "")
        ctx.dataset_id = resolved

    preamble = (
        "import random, sys\n"
        "from pathlib import Path\n"
        f"random.seed({seed})\n"
        "try:\n"
        "    import numpy as _np; _np.random.seed(%d)\n" % seed
        + "except Exception: pass\n"
        f"OUTDIR = Path(r{str(outdir)!r})\n"
        + (f"import anndata as _ad\nadata = _ad.read_h5ad(r{str(h5ad)!r})\n" if h5ad else "")
    )
    script = CONFIG.exec_dir(ctx.project_id) / f"{store.new_id('exec', 8)}.py"
    script.write_text(preamble + "\n" + code, encoding="utf-8")

    before = _snapshot(outdir)
    log = CONFIG.ensure_project_dirs(ctx.project_id) / "logs" / f"{script.stem}.log"
    exit_code, tail = bridge._run([str(py), str(script)], log, timeout=timeout_s, cwd=outdir)

    # Register whatever the code left behind, so it reaches the briefing.
    after = _snapshot(outdir)
    new = [p for p, m in after.items() if before.get(p) != m]
    for p in sorted(new)[:20]:
        ctx.add_artifact(_CAPTURED[p.suffix.lower()], p, caption=f"exec_python: {p.name}")

    ctx.code = code
    ctx.summary = {"exit_code": exit_code, "stdout_tail": tail[-1500:],
                   "n_artifacts": len(new), "script": str(script)}
    if exit_code != 0:
        ctx.warn(f"exit code {exit_code} — the code failed; see stdout_tail. The server "
                 f"itself is unaffected because this ran in a separate process.")
    ctx.suggest("skin.io.describe", "skin.memory.note")
