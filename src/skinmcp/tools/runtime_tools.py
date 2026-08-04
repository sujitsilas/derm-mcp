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
    proj_py = CONFIG.project_python(ctx.project_id)
    proj_lib = CONFIG.project_renv(ctx.project_id) / "library"
    ctx.summary = {
        "project_dir": str(CONFIG.project_dir(ctx.project_id)),
        "python": {"packages": python_manifest.capture(), "offline": CONFIG.offline,
                   "profile": CONFIG.profile, "running_under": sys.executable,
                   "project_venv": str(proj_py) if proj_py.exists() else None},
        "r": {**r, "working_dir": str(CONFIG.project_dir(ctx.project_id)),
              "project_library": str(proj_lib) if proj_lib.is_dir() else None},
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
      summary="Create this project's runtime: uv venv (python) or pinned container/renv (r).")
def create(kind: str = "r", backend: str = "", force: bool = False,
           project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Create a runtime **inside this project's directory**.

    The environment that produced a result sits beside the result, and R-backed
    tools execute with the project directory as their working directory. Point
    `--project-root` at a working directory and the whole project — data, env,
    objects, figures, memory — is self-contained and movable.

    Args:
        kind: "python" (uv venv at {project}/runtimes/venv, resolved from the
            committed uv.lock) or "r" (renv library at {project}/runtimes/renv,
            restored from renv.lock).
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
    """uv venv inside the project, resolved from the committed lock."""
    uv = shutil.which("uv")
    venv = CONFIG.project_venv(ctx.project_id)
    py = CONFIG.project_python(ctx.project_id)
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
        ctx.summary = {"kind": "python", "already_exists": True, "python": str(py)}
        ctx.warn("Project venv already exists; pass force=True to rebuild.")
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


def _create_renv(ctx: Ctx, force: bool, dry_run: bool) -> None:
    """renv library inside the project, restored from the committed renv.lock."""
    lock = Path(bridge.R_DIR) / "renv.lock"
    lib = CONFIG.project_renv(ctx.project_id) / "library"
    ok, ver = bridge.rscript_available()
    expr = (f"install.packages('renv', repos='https://cloud.r-project.org'); "
            f".libPaths('{lib}'); "
            f"renv::restore(lockfile='{lock}', library='{lib}', prompt=FALSE)")
    cmd = [shutil.which("Rscript") or "Rscript", "-e", expr]
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
        ctx.summary = {"kind": "r", "already_exists": True, "library": str(lib)}
        ctx.warn("renv library already present; pass force=True to rebuild.")
        return
    lib.mkdir(parents=True, exist_ok=True)
    log = CONFIG.ensure_project_dirs(ctx.project_id) / "logs" / "runtime_renv.log"
    code, tail = bridge._run(cmd, log, timeout=7200,
                             cwd=CONFIG.project_dir(ctx.project_id))
    ctx.add_artifact("log", log, caption="renv restore log")
    if code != 0:
        raise RuntimeUnavailable(f"renv::restore exited {code}",
                                 remedy=f"Log tail:\n{tail[-900:]}")
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
    # Enumerate the project's renv library so the manifest reports the versions
    # that actually ran, not whatever is in the user library.
    r_pkgs: dict[str, str] = {}
    lib = CONFIG.project_renv(ctx.project_id) / "library"
    if r["available"]:
        try:
            expr = (f".libPaths('{lib}'); "
                    "ip <- installed.packages()[, c('Package','Version')]; "
                    "cat(paste(ip[,1], ip[,2], sep='=', collapse='\\n'))")
            proc = subprocess.run([shutil.which("Rscript") or "Rscript", "-e", expr],
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
        "py-monocle (github.com/bioturing/py-monocle) publishes no license, so it is not "
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
