"""The R bridge: h5ad <-> SingleCellExperiment round-trip plus subprocess exec.

Only *named, vetted* scripts under `runtimes/r/scripts/*.R` can be run. The
model never supplies R code; it supplies a script id and typed parameters.
Arbitrary R execution lives behind `skin.runtime.exec_r_raw`, which is disabled
unless the server was started with `--allow-raw-exec`.

Two I/O paths, both implemented:
  1. primary  — `zellkonverter::readH5AD/writeH5AD` over a shared temp dir
  2. fallback — mtx export (`matrix.mtx`, `genes.txt`, `barcodes.txt`,
                `metadata.csv`, one CSV per reducedDim), which is the layout the
                reference notebook uses (cell 3)
"""

from __future__ import annotations

import json
import logging
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

def docker_available() -> tuple[bool, str]:
    exe = shutil.which(CONFIG.docker_bin)
    if not exe:
        return False, f"{CONFIG.docker_bin} not on PATH"
    try:
        p = subprocess.run([exe, "info", "--format", "{{.ServerVersion}}"],
                           capture_output=True, text=True, timeout=15)
        if p.returncode != 0:
            return False, (p.stderr or "docker daemon not responding").strip()[:200]
        return True, p.stdout.strip()
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)[:200]


def image_present() -> tuple[bool, str]:
    ok, _ = docker_available()
    if not ok:
        return False, ""
    exe = shutil.which(CONFIG.docker_bin) or CONFIG.docker_bin
    p = subprocess.run([exe, "images", "-q", CONFIG.r_image],
                       capture_output=True, text=True, timeout=20)
    digest = p.stdout.strip()
    return bool(digest), digest


def rscript_available() -> tuple[bool, str]:
    exe = shutil.which("Rscript")
    if not exe:
        return False, "Rscript not on PATH"
    try:
        p = subprocess.run([exe, "--version"], capture_output=True, text=True, timeout=15)
        return True, (p.stdout or p.stderr).strip()[:120]
    except (subprocess.TimeoutExpired, OSError) as e:
        return False, str(e)[:200]


def runtime_status() -> dict[str, Any]:
    docker_ok, docker_ver = docker_available()
    img_ok, img_id = image_present()
    r_ok, r_ver = rscript_available()
    backend = "docker" if (docker_ok and img_ok) else ("renv" if r_ok else "none")
    return {
        "backend": backend,
        "available": backend != "none",
        "docker": {"available": docker_ok, "version": docker_ver,
                   "image": CONFIG.r_image, "image_present": img_ok, "image_id": img_id},
        "local_r": {"available": r_ok, "version": r_ver},
        "scripts": available_scripts(),
        "raw_exec_enabled": CONFIG.allow_raw_exec,
    }


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
    (out_dir / "genes.txt").write_text("\n".join(map(str, adata.var_names)), encoding="utf-8")
    (out_dir / "barcodes.txt").write_text("\n".join(map(str, adata.obs_names)), encoding="utf-8")
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
    a.write_h5ad(path, compression="gzip")


# --------------------------------------------------------------------------- #
# execution
# --------------------------------------------------------------------------- #

def _run(cmd: list[str], log_path: Path, timeout: int = 3600) -> tuple[int, str]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as lf:
        try:
            p = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, timeout=timeout,
                               text=True)
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
            f"docker: {status['docker']['version'] or 'unavailable'}; "
            f"Rscript: {status['local_r']['version'] or 'unavailable'}",
            python_fallback or "skin.runtime.status",
        )

    work_root = CONFIG.ensure_project_dirs(project_id) / "runtimes"
    work = Path(tempfile.mkdtemp(prefix=f"{script_id}_", dir=str(work_root)))
    params = dict(params or {})
    (work / "params.json").write_text(json.dumps(params, default=str), encoding="utf-8")

    if adata is not None:
        use_mtx = io == "mtx"
        if io in ("auto", "h5ad"):
            try:
                _write_h5ad_for_r(adata, work / "input.h5ad")
            except Exception as e:  # noqa: BLE001 - documented fallback path
                if io == "h5ad":
                    raise
                logger.warning("h5ad round-trip failed (%s); falling back to mtx export", e)
                use_mtx = True
        if use_mtx:
            write_mtx_export(adata, work / "input_mtx")
        (work / "io_mode.txt").write_text("mtx" if use_mtx else "h5ad", encoding="utf-8")

    log_path = CONFIG.ensure_project_dirs(project_id) / "logs" / f"{script_id}.log"
    if status["backend"] == "docker":
        exe = shutil.which(CONFIG.docker_bin) or CONFIG.docker_bin
        cmd = [exe, "run", "--rm",
               "-v", f"{work}:/work",
               "-v", f"{SCRIPT_DIR}:/scripts:ro",
               "-w", "/work", CONFIG.r_image,
               "Rscript", f"/scripts/{script_id}.R", "/work"]
    else:
        cmd = [shutil.which("Rscript") or "Rscript", str(script), str(work)]

    code, tail = _run(cmd, log_path, timeout=timeout)
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


def seurat_to_h5ad(path: Path, *, assay: str = "RNA", project_id: str) -> tuple[Any, str]:
    """Convert a Seurat `.rds` to AnnData through the container."""
    import anndata as ad

    res = run_r_script("seurat_to_h5ad", adata=None, project_id=project_id,
                       params={"input_rds": str(path), "assay": assay},
                       python_fallback="skin.io.load_h5ad")
    out_p = Path(res.get("output_h5ad", ""))
    if out_p.exists():
        return ad.read_h5ad(out_p), res.get("log", "")
    mtx_dir = Path(res.get("output_mtx", ""))
    if mtx_dir.is_dir():
        return read_mtx_export(mtx_dir), res.get("log", "")
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
    status = runtime_status()
    if not status["available"]:
        raise r_unavailable("no R backend", "skin.runtime.status")
    if status["backend"] == "docker":
        exe = shutil.which(CONFIG.docker_bin) or CONFIG.docker_bin
        cmd = [exe, "run", "--rm", "-v", f"{work}:/work", "-w", "/work",
               CONFIG.r_image, "Rscript", "/work/raw.R"]
    else:
        cmd = [shutil.which("Rscript") or "Rscript", str(work / "raw.R")]
    log_path = CONFIG.ensure_project_dirs(project_id) / "logs" / "exec_r_raw.log"
    code_, tail = _run(cmd, log_path, timeout=timeout)
    return {"exit_code": code_, "log": tail, "work_dir": str(work)}
