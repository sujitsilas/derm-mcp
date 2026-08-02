"""`skin.runtime.*` — runtime status, R container management, version manifest.

`manifest` is a deliverable, not a debug tool: it is what you paste into a
methods section, and what answers "which version of scDblFinder produced this?"
"""

from __future__ import annotations

import logging
import shutil
import subprocess
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
    ctx.summary = {
        "python": {"packages": python_manifest.capture(), "offline": CONFIG.offline,
                   "profile": CONFIG.profile},
        "r": r,
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
      summary="Build or pull the pinned R container / bootstrap renv.")
def create(kind: str = "r", backend: str = "docker", force: bool = False,
           project_id: str = "", dry_run: bool = False, seed: int = 0, *, ctx: Ctx) -> None:
    """Build the pinned R runtime so R-backed tools work.

    Args:
        kind: "r" (the only kind in v1).
        backend: "docker" (pinned rocker/r-ver image) or "renv" (local R + renv.lock).
        force: Rebuild even if the image already exists.
        project_id: Defaults to the active project.
        dry_run: Report the command without running it.
        seed: Unused.
    """
    if kind != "r":
        raise BadParam("kind must be 'r' in v1")
    if backend not in ("docker", "renv"):
        raise BadParam("backend must be docker|renv")

    r_dir = Path(bridge.R_DIR)
    log_path = CONFIG.ensure_project_dirs(ctx.project_id) / "logs" / "runtime_create.log"

    if backend == "docker":
        ok, ver = bridge.docker_available()
        present, digest = bridge.image_present()
        cmd = [shutil.which(CONFIG.docker_bin) or CONFIG.docker_bin, "build",
               "-t", CONFIG.r_image, str(r_dir)]
        ctx.code = " ".join(cmd) + "\n"
        if dry_run:
            ctx.summary = {"command": cmd, "docker_available": ok,
                           "image_present": present, "image": CONFIG.r_image}
            return
        if not ok:
            raise RuntimeUnavailable(
                f"Docker is not usable: {ver}",
                remedy="Start Docker Desktop, or use backend='renv' with a local R "
                       "install. Every R-backed tool has a Python fallback listed in "
                       "skin.runtime.status.",
                suggested_tool="skin.runtime.status")
        if present and not force:
            ctx.summary = {"already_built": True, "image": CONFIG.r_image,
                           "image_id": digest}
            ctx.warn("Image already present; pass force=True to rebuild.")
            return
        code, tail = bridge._run(cmd, log_path, timeout=5400)
        present, digest = bridge.image_present()
        ctx.add_artifact("log", log_path, caption="R image build log")
        if code != 0:
            raise RuntimeUnavailable(
                f"docker build exited {code}",
                remedy=f"Build log tail:\n{tail[-900:]}",
                suggested_tool="skin.runtime.status")
        ctx.summary = {"backend": "docker", "image": CONFIG.r_image, "image_id": digest,
                       "log_tail": tail[-600:], "scripts": bridge.available_scripts()}
    else:
        ok, ver = bridge.rscript_available()
        cmd = [shutil.which("Rscript") or "Rscript", "-e",
               f"install.packages('renv', repos='https://cloud.r-project.org'); "
               f"renv::restore(lockfile='{r_dir / 'renv.lock'}', prompt=FALSE)"]
        ctx.code = " ".join(cmd[:2]) + " renv::restore(...)\n"
        if dry_run:
            ctx.summary = {"command": "renv::restore", "rscript_available": ok}
            return
        if not ok:
            raise RuntimeUnavailable(
                "Rscript is not on PATH",
                remedy="Install R 4.4.x, or use backend='docker'.",
                suggested_tool="skin.runtime.status")
        code, tail = bridge._run(cmd, log_path, timeout=5400)
        ctx.add_artifact("log", log_path, caption="renv restore log")
        if code != 0:
            raise RuntimeUnavailable(f"renv::restore exited {code}",
                                     remedy=f"Log tail:\n{tail[-900:]}")
        ctx.summary = {"backend": "renv", "r_version": ver, "log_tail": tail[-600:]}
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
    r_pkgs: dict[str, str] = {}
    if r["backend"] == "docker" and r["docker"]["image_present"]:
        try:
            exe = shutil.which(CONFIG.docker_bin) or CONFIG.docker_bin
            p = subprocess.run(
                [exe, "run", "--rm", CONFIG.r_image, "Rscript", "-e",
                 "ip <- installed.packages()[, c('Package','Version')]; "
                 "cat(paste(ip[,1], ip[,2], sep='=', collapse='\\n'))"],
                capture_output=True, text=True, timeout=120)
            for line in p.stdout.splitlines():
                if "=" in line:
                    k, v = line.split("=", 1)
                    r_pkgs[k.strip()] = v.strip()
        except (subprocess.TimeoutExpired, OSError) as e:
            logger.warning("could not enumerate R packages: %s", e)

    m = python_manifest.full_manifest()
    m["r"] = {"backend": r["backend"], "image": CONFIG.r_image,
              "image_id": r["docker"]["image_id"], "available": r["available"],
              "packages": r_pkgs, "vetted_scripts": r["scripts"]}
    m["skinmcp"] = {"profile": CONFIG.profile, "offline": CONFIG.offline,
                    "allow_raw_exec": CONFIG.allow_raw_exec,
                    "project_root": str(CONFIG.project_root)}
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
        "r": {"backend": m["r"]["backend"], "image": m["r"]["image"],
              "available": m["r"]["available"], "n_r_packages": len(m["r"]["packages"])},
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
