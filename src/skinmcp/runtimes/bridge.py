"""The R bridge: h5ad <-> SingleCellExperiment round-trip plus subprocess exec.

Only *named, vetted* scripts under `runtimes/r/scripts/*.R` can be run. The
model never supplies R code; it supplies a script id and typed parameters.
Arbitrary R execution lives behind `skin.runtime.exec_r_raw`, which is disabled
unless the server was started with `--allow-raw-exec`.

One I/O path: plain files over a shared temp dir.

Python -> R is the mtx layout (`matrix.mtx`, `genes.txt`, `barcodes.txt`,
`metadata.csv`, one CSV per reducedDim). R -> Python is the richer layout that
`r/scripts/seurat_export.R` writes and `runtimes/seurat_import.py` reads: every
assay and layer, feature metadata, reductions with loadings and stdev, graphs,
and a manifest.

There used to be a zellkonverter h5ad path in front of this. It was removed for
two reasons. It reaches Python through basilisk/reticulate, so a data handoff
between two working runtimes depended on provisioning a *third* — which failed
on a real machine while trying to install Python 3.14.0 through pyenv. And it
routes Seurat objects through `as.SingleCellExperiment`, which keeps one
assay's counts and logcounts: a post-SCTransform object lost the SCT assay,
scale.data, reduction loadings and the PCA stdev, silently.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from ..config import CONFIG
from ..errors import BadParam, NotFound, RuntimeUnavailable, r_unavailable

logger = logging.getLogger(__name__)

R_DIR = Path(__file__).parent / "r"
SCRIPT_DIR = R_DIR / "scripts"


def available_scripts() -> list[str]:
    if not SCRIPT_DIR.is_dir():
        return []
    # `_common.R` is a sourced helper, not a runnable entry point.
    return sorted(p.stem for p in SCRIPT_DIR.glob("*.R") if not p.stem.startswith("_"))


# --------------------------------------------------------------------------- #
# runtime detection
# --------------------------------------------------------------------------- #

def rscript_path() -> str:
    """The Rscript this server drives.

    Everything that shells out to R goes through here, so pinning an
    interpreter is one setting rather than five edits. See `Config.rscript`
    for why pinning matters: an renv.lock is only valid against the R minor
    version its Bioconductor release was cut for.
    """
    if CONFIG.rscript:
        return str(Path(CONFIG.rscript).expanduser())
    return shutil.which("Rscript") or "Rscript"


def rscript_cmd(*args: str) -> list[str]:
    """An Rscript command line that ignores the user's R startup files.

    `Rscript` reads ~/.Rprofile by default, and a startup file is free to call
    `.libPaths()`. A real one here ended with

        .libPaths(c("~/Library/R/arm64/4.6/library", .libPaths()))

    which forced an R *4.6* library to the front of the search path for every
    session — including the pinned R 4.4 this server drives. R then loaded
    packages built by a different R and segfaulted inside `dyn.load`, before
    any vetted script ran. Note that setting R_LIBS cannot defend against this:
    the profile runs afterwards and prepends itself.

    Reproducibility points the same way. The managed renv library is supposed
    to be the whole story of what a vetted script imported; a personal profile
    silently adding a library makes the recorded manifest a fiction. So the
    profile is skipped rather than merged. ~/.Renviron is still honoured — it
    carries settings like R_MAX_VSIZE that are about the machine, not the
    library path.
    """
    return [rscript_path(), "--no-init-file", "--no-site-file", *args]


#: Variables that pin a macOS SDK for the C toolchain. Each names a directory,
#: so each can be checked rather than guessed at.
_SDK_VARS = ("SDKROOT", "CPATH", "LIBRARY_PATH")

#: Compiler and linker flags that `conda activate` exports. R has its own
#: Makeconf and does not want these.
_CONDA_BUILD_VARS = ("CFLAGS", "CXXFLAGS", "CPPFLAGS", "LDFLAGS", "LDFLAGS_LD",
                     "CC", "CXX", "OBJC", "OBJCXX", "OBJC_FOR_BUILD", "FC", "F77")


def clean_build_env(env: dict[str, str] | None = None) -> tuple[dict[str, str], list[str]]:
    """Drop toolchain variables that point at paths which no longer exist.

    Returns the environment plus the names dropped, so the caller can say so.

    A real ~/.zshrc here exported

        SDKROOT=/Applications/Xcode.app/.../MacOSX15.2.sdk
        CPATH=${SDKROOT}/usr/include

    and Xcode has since moved to MacOSX26.5.sdk. SDKROOT overrides everything --
    even `xcrun --show-sdk-path` returns it -- so every source compile failed
    with "'stdio.h' file not found", and renv reported it as `R CMD config CC`
    failing. The advice that error invites, `xcode-select --install`, does not
    help: the command line tools were already installed and healthy. Only the
    stale variable was wrong.

    Unsetting beats correcting. An empty SDKROOT lets clang ask xcrun, which
    gets the right answer from whichever toolchain is actually installed;
    writing in a specific SDK path would just be a fresher version of the same
    mistake.
    """
    import os

    out = dict(os.environ if env is None else env)
    dropped = []
    for var in _SDK_VARS:
        val = out.get(var, "")
        # A ':'-joined LIBRARY_PATH is only stale if every entry is.
        parts = [p for p in val.split(os.pathsep) if p]
        if parts and not any(Path(p).exists() for p in parts):
            out.pop(var, None)
            dropped.append(var)

    # `conda activate` exports a full build toolchain — CFLAGS with
    # `-isystem $CONDA_PREFIX/include`, LDFLAGS with `-L$CONDA_PREFIX/lib` and an
    # rpath to match, CC/CXX pointing at conda's clang. An R package compiled
    # under those links against conda's C++ runtime and libraries while R itself
    # is linked against the system's, which is an ABI mix that fails at
    # dlopen-time with a missing symbol rather than at build time. R has its own
    # Makeconf and wants none of it. Only flags actually naming a conda prefix
    # are dropped, so a deliberate CFLAGS the user set for another reason stands.
    prefixes = [p for p in (out.get("CONDA_PREFIX"), out.get("CONDA_PREFIX_1")) if p]
    prefixes += [str(Path.home() / n) for n in ("anaconda3", "miniconda3", "miniforge3")]
    for var in _CONDA_BUILD_VARS:
        val = out.get(var, "")
        if val and any(pref in val for pref in prefixes):
            out.pop(var, None)
            dropped.append(var)
    return out, dropped


def user_rprofile_touches_libpaths() -> str:
    """The user's .Rprofile if it manipulates .libPaths, else "".

    Reported by `skin.runtime.status` so the interaction is visible: we ignore
    the file, and the user should know that, especially when the same file is
    what makes their interactive R sessions work.
    """
    p = Path.home() / ".Rprofile"
    try:
        if p.is_file() and ".libPaths(" in p.read_text(encoding="utf-8", errors="replace"):
            return str(p)
    except OSError:
        pass
    return ""


def rscript_available() -> tuple[bool, str]:
    if CONFIG.disable_r:
        return False, "the R backend is switched off (SKINMCP_DISABLE_R / --no-r)"
    exe = rscript_path()
    if not Path(exe).is_file() and not shutil.which(exe):
        return False, (f"{exe} is not executable (SKINMCP_RSCRIPT / --rscript)"
                       if CONFIG.rscript else "Rscript not on PATH")
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
        return True, (p.stdout or p.stderr).strip()[:120]
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)[:200]


def lockfile_r_version() -> str:
    """The R minor version renv.lock was cut against, or "" if unreadable."""
    import json

    try:
        meta = json.loads((R_DIR / "renv.lock").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ""
    return str((meta.get("R") or {}).get("Version") or "")


def runtime_status() -> dict[str, Any]:
    """R is a local install managed by renv. There is no container layer."""
    import re

    r_ok, r_ver = rscript_available()
    want = lockfile_r_version()
    have = (re.search(r"(\d+\.\d+)\.\d+", r_ver or "") or [None, ""])[1]
    # Surfaced here, not only from skin.runtime.create: a mismatch makes every
    # R-backed tool fall back to Python, and "why did this run in Python?" is
    # the question status() exists to answer.
    mismatch = bool(r_ok and want and have and want.rsplit(".", 1)[0] != have)
    out = {
        "backend": "renv" if r_ok else "none",
        "available": r_ok,
        "local_r": {"available": r_ok, "version": r_ver, "executable": rscript_path(),
                    "pinned": bool(CONFIG.rscript)},
        "lockfile_r": want,
        "r_version_matches_lockfile": (not mismatch) if (r_ok and want and have) else None,
        "scripts": available_scripts(),
        "raw_exec_enabled": CONFIG.allow_raw_exec,
    }
    if CONFIG.disable_r:
        out["disabled"] = (
            "The R backend is switched off. Every R-backed tool returns "
            "RUNTIME_UNAVAILABLE naming its pure-Python equivalent, which is a "
            "supported path, not a degraded one. Unset SKINMCP_DISABLE_R (or drop "
            "--no-r) to turn it back on."
        )
        return out
    rprofile = user_rprofile_touches_libpaths()
    if rprofile:
        out["ignored_rprofile"] = (
            f"{rprofile} calls .libPaths(), so R-backed tools run with "
            f"--no-init-file and do not see it. Your interactive R sessions are "
            f"unaffected. If that file hardcodes a library for one R version, it "
            f"will break other versions the same way it broke this one."
        )
    if mismatch:
        out["version_conflict"] = (
            f"renv.lock is pinned to R {want} but R {have}.x is what this server runs. "
            f"Bioconductor publishes binaries only for the R version each release "
            f"targets, so the restore cannot succeed. Install the matching R "
            f"side-by-side (e.g. `rig add {want.rsplit('.', 1)[0]}`) and point the "
            f"server at it with --rscript / SKINMCP_RSCRIPT, or regenerate renv.lock "
            f"against R {have} — that repins every package version, so it is a "
            f"reproducibility decision, not a fix to apply silently."
        )
    return out


# --------------------------------------------------------------------------- #
# mtx export / import (the fallback path, reference cell 3)
# --------------------------------------------------------------------------- #

def write_mtx_export(adata: Any, out_dir: Path, layer: str = "counts") -> Path:
    """Write the SCE-compatible export layout."""
    import pandas as pd
    import scipy.io as sio
    import scipy.sparse as sp

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    X = adata.layers.get(layer, adata.X)
    # SCE wants genes x cells.
    M = (X.T.tocsr() if sp.issparse(X) else sp.csr_matrix(X).T)
    sio.mmwrite(str(out_dir / "matrix.mtx"), M)
    # Trailing newline: R's readLines warns "incomplete final line found" on
    # every read without it. Harmless, but it puts a warning in front of the
    # user on a path that is already the fallback, which reads like a fault.
    (out_dir / "genes.txt").write_text(
        "\n".join(map(str, adata.var_names)) + "\n", encoding="utf-8")
    (out_dir / "barcodes.txt").write_text(
        "\n".join(map(str, adata.obs_names)) + "\n", encoding="utf-8")
    adata.obs.to_csv(out_dir / "metadata.csv")
    for k, v in adata.obsm.items():
        if hasattr(v, "shape") and len(v.shape) == 2 and v.shape[1] <= 100:
            pd.DataFrame(v, index=adata.obs_names).to_csv(out_dir / f"reducedDim_{k}.csv")
    return out_dir


def read_mtx_export(in_dir: Path) -> Any:
    """Read the SCE export layout back into AnnData."""
    import anndata as ad
    import numpy as np
    import pandas as pd
    import scipy.io as sio

    in_dir = Path(in_dir)
    M = sio.mmread(str(in_dir / "matrix.mtx")).tocsr()   # genes x cells
    genes = (in_dir / "genes.txt").read_text(encoding="utf-8").splitlines()
    barcodes = (in_dir / "barcodes.txt").read_text(encoding="utf-8").splitlines()
    X = M.T.tocsr()
    obs = pd.DataFrame(index=pd.Index(barcodes, name=None))
    meta_p = in_dir / "metadata.csv"
    if meta_p.exists():
        meta = pd.read_csv(meta_p, index_col=0)
        meta.index = meta.index.astype(str)
        obs = meta.reindex([str(b) for b in barcodes])
    var = pd.DataFrame(index=pd.Index(genes))
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obs_names = [str(b) for b in barcodes]
    adata.var_names = [str(g) for g in genes]
    adata.layers["counts"] = adata.X.copy()
    for f in sorted(in_dir.glob("reducedDim_*.csv")):
        key = f.stem.replace("reducedDim_", "")
        df = pd.read_csv(f, index_col=0)
        adata.obsm[key if key.startswith("X_") else f"X_{key}"] = np.asarray(df.values,
                                                                            dtype=float)
    return adata


def _write_h5ad_for_r(adata: Any, path: Path) -> None:
    from .. import registry

    a = adata.copy()
    registry._sanitize_for_h5ad(a)
    # zellkonverter chokes on raw/obsp; strip them for the round trip.
    a.raw = None
    a.obsp = {}
    # Uncompressed: this is a transient local handoff that R reads once and
    # deletes. Compressing it costs seconds of the caller's time to save disk
    # that is freed moments later, and rules out lzf, which rhdf5 cannot read.
    a.write_h5ad(path, compression=None)


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], log_path: Path, timeout: int = 3600,
         cwd: Path | None = None, env: dict | None = None) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout,
                               text=True, cwd=str(cwd) if cwd else None, env=env)
            code = p.returncode
        except subprocess.TimeoutExpired:
            lf.write(f"\n[skin-mcp] TIMEOUT after {timeout}s\n")
            code = 124
    tail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace")
                     .splitlines()[-40:])
    return code, tail


def run_r_script(
    script_id: str,
    *,
    adata: Any | None = None,
    project_id: str,
    params: dict[str, Any] | None = None,
    python_fallback: str = "",
    io: str = "auto",
    timeout: int = 3600,
) -> dict[str, Any]:
    """Run a vetted R script against an AnnData object.

    The object is written to a shared temp dir, the script runs in the pinned
    container (or locally under renv), and `result.json` plus any written
    objects are read back.
    """
    script = SCRIPT_DIR / f"{script_id}.R"
    if not script.exists():
        raise NotFound(
            f"no vetted R script named {script_id!r}",
            remedy=f"Available scripts: {available_scripts()}. Arbitrary R needs "
                   f"skin.runtime.exec_r_raw and --allow-raw-exec.",
        )

    status = runtime_status()
    if not status["available"]:
        raise r_unavailable(
            f"Rscript: {status['local_r']['version'] or 'not on PATH'}",
            python_fallback or "skin.runtime.status",
        )

    work_root = CONFIG.ensure_project_dirs(project_id) / "runtimes"
    work = Path(tempfile.mkdtemp(prefix=f"{script_id}_", dir=str(work_root)))
    params = dict(params or {})
    (work / "params.json").write_text(json.dumps(params, default=str), encoding="utf-8")

    if adata is not None:
        # Plain files, always. The h5ad path went through zellkonverter, which
        # reaches Python via basilisk — an embedded second interpreter that can
        # fail before any data is read, and did here. `io` is kept in the
        # signature so existing callers do not break; anything but "mtx" now
        # only earns a note in the log.
        if io not in ("auto", "mtx"):
            logger.info("io=%r ignored; the R bridge transports plain files only", io)
        write_mtx_export(adata, work / "input_mtx")
        (work / "io_mode.txt").write_text("mtx", encoding="utf-8")

    log_path = CONFIG.ensure_project_dirs(project_id) / "logs" / f"{script_id}.log"
    proj = CONFIG.project_dir(project_id)
    # Prepend the managed library rather than replacing R_LIBS_USER, which would
    # hide the user's own library — and with it any Seurat/harmony they already
    # have. Layering means the managed library only has to carry what is missing,
    # so a clone with a working R install needs almost nothing extra.
    env, dropped = clean_build_env()
    if dropped:
        logger.warning("dropped stale toolchain vars %s (they name paths that do not "
                       "exist); source compiles would fail with 'stdio.h' not found",
                       dropped)
    lib = CONFIG.shared_renv() / "library"
    if lib.is_dir():
        existing = env.get("R_LIBS", "")
        env["R_LIBS"] = f"{lib}{os.pathsep}{existing}".rstrip(os.pathsep)
    cmd = rscript_cmd(str(script), str(work))
    code, tail = _run(cmd, log_path, timeout=timeout, cwd=proj, env=env)
    result_p = work / "result.json"
    if code != 0 or not result_p.exists():
        raise RuntimeUnavailable(
            f"R script {script_id!r} exited with code {code}",
            remedy=(f"Log tail:\n{tail[-800:]}\n"
                    + (f"Pure-Python alternative: {python_fallback}"
                       if python_fallback else "See skin.runtime.status.")),
            suggested_tool=python_fallback or "skin.runtime.status",
        )
    out = json.loads(result_p.read_text(encoding="utf-8"))
    out["log"] = tail
    out["work_dir"] = str(work)
    return out


def seurat_to_h5ad(path: Path, *, assay: str = "", project_id: str) -> tuple[Any, str]:
    """Convert a Seurat `.rds` to AnnData via the file export.

    `assay` empty means "whatever the object's DefaultAssay is", which after
    SCTransform is SCT — the one you almost always want. Naming an assay pins
    which matrices become X and layers; every assay is exported either way, so
    the choice is reversible without re-running R.
    """
    from .seurat_import import read_seurat_export

    res = run_r_script("seurat_to_h5ad", adata=None, project_id=project_id,
                       params={"input_rds": str(path), "assay": assay},
                       python_fallback="skin.io.load_h5ad")
    files = Path(res.get("output_files", ""))
    if files.is_dir():
        adata = read_seurat_export(files, assay=assay)
        return adata, res.get("log", "")
    raise RuntimeUnavailable(
        "the Seurat conversion produced no output",
        remedy=f"R log tail:\n{res.get('log', '')[-600:]}",
    )


def exec_r_raw(code: str, *, project_id: str, adata: Any | None = None,
               timeout: int = 900) -> dict[str, Any]:
    """Run arbitrary R. Disabled unless the server was started with --allow-raw-exec."""
    if not CONFIG.allow_raw_exec:
        raise BadParam(
            "arbitrary R execution is disabled",
            remedy="Restart the server with --allow-raw-exec to enable it. Vetted scripts "
                   f"({available_scripts()}) run without it via skin.runtime.exec_r.",
        )
    work_root = CONFIG.ensure_project_dirs(project_id) / "runtimes"
    work = Path(tempfile.mkdtemp(prefix="raw_", dir=str(work_root)))
    (work / "raw.R").write_text(code, encoding="utf-8")
    if adata is not None:
        _write_h5ad_for_r(adata, work / "input.h5ad")
    if not runtime_status()["available"]:
        raise r_unavailable("Rscript not on PATH", "skin.runtime.status")
    cmd = rscript_cmd(str(work / "raw.R"))
    log_path = CONFIG.ensure_project_dirs(project_id) / "logs" / "exec_r_raw.log"
    code_, tail = _run(cmd, log_path, timeout=timeout)
    return {"exit_code": code_, "log": tail, "work_dir": str(work)}
