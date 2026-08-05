"""Assemble the files written by ``runtimes/r/scripts/seurat_export.R`` into AnnData.

The pair exists because a single-shot converter is the wrong shape for this
problem. `zellkonverter::writeH5AD` goes Seurat -> SingleCellExperiment -> h5ad,
and the SCE coercion keeps one assay's counts and logcounts. An object that has
been through SCTransform carries several assays, each with counts/data/
scale.data plus its own feature metadata, and most of that does not survive the
round trip -- silently, so the resulting AnnData looks right until the matrix
you needed is missing three steps later.

Reading a directory of plain files fails visibly instead: a missing layer is a
missing file, and the manifest says what was written.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _read_mtx_genes_by_cells(path: Path):
    """Read a genes x cells .mtx and return it cells x genes, CSR."""
    import scipy.io as sio

    return sio.mmread(str(path)).tocsr().T.tocsr()


def _read_csv_indexed(path: Path):
    import pandas as pd

    df = pd.read_csv(path, index_col=0)
    df.index = df.index.astype(str)
    return df


def read_seurat_export(work: Path, *, assay: str = "", x_layer: str = "") -> Any:
    """Build an AnnData from a `seurat_export()` directory.

    Args:
        work: the directory `seurat_export()` wrote.
        assay: which Seurat assay to use as X/layers. Defaults to the object's
            DefaultAssay, which after SCTransform is "SCT".
        x_layer: "data" (normalised) or "counts". Defaults to data when present,
            because that is what scanpy's downstream steps expect X to be.

    Every matrix is reindexed against `barcodes.txt` rather than assumed to be
    aligned: R and Python disagree about ordering often enough, and a silent
    misalignment produces plausible nonsense.
    """
    import anndata as ad
    import numpy as np
    import pandas as pd

    work = Path(work)
    manifest = json.loads((work / "manifest.json").read_text(encoding="utf-8"))
    cells = (work / "barcodes.txt").read_text(encoding="utf-8").splitlines()

    assay = assay or manifest.get("default_assay") or ""
    assays = manifest.get("assays") or {}
    if assay not in assays:
        raise ValueError(
            f"assay {assay!r} not in the export; available: {sorted(assays)}. "
            f"Re-run seurat_export() or pass assay=.")
    info = assays[assay]
    layers_present = list(info.get("layers") or [])

    feats = (work / f"assay_{assay}_features.txt").read_text(encoding="utf-8").splitlines()

    mats: dict[str, Any] = {}
    for lyr in ("counts", "data"):
        p = work / f"assay_{assay}_{lyr}.mtx"
        if p.exists():
            mats[lyr] = _read_mtx_genes_by_cells(p)

    if not mats:
        raise ValueError(f"no counts or data matrix for assay {assay!r} in {work}")

    x_layer = x_layer or ("data" if "data" in mats else "counts")
    if x_layer not in mats:
        raise ValueError(f"x_layer {x_layer!r} not exported; have {sorted(mats)}")

    obs = pd.DataFrame(index=pd.Index([str(c) for c in cells]))
    meta_p = work / "metadata.csv"
    if meta_p.exists():
        obs = _read_csv_indexed(meta_p).reindex([str(c) for c in cells])

    var = pd.DataFrame(index=pd.Index([str(f) for f in feats]))
    fmeta_p = work / f"assay_{assay}_meta.csv"
    if fmeta_p.exists():
        fm = _read_csv_indexed(fmeta_p)
        var = fm.reindex([str(f) for f in feats])

    adata = ad.AnnData(X=mats[x_layer], obs=obs, var=var)
    adata.obs_names = [str(c) for c in cells]
    adata.var_names = [str(f) for f in feats]
    for lyr, m in mats.items():
        if lyr != x_layer:
            adata.layers[lyr] = m
    # scanpy's convention, and what every tool in this server reads.
    if "counts" in mats:
        adata.layers["counts"] = mats["counts"]

    # scale.data covers only the variable features, so it cannot be a layer --
    # layers must match var. Kept in obsm, which is shaped (n_obs, k).
    sd_p = work / f"assay_{assay}_scaledata.csv"
    if sd_p.exists():
        sd = pd.read_csv(sd_p, index_col=0)          # features x cells
        sd = sd.reindex(columns=[str(c) for c in cells])
        adata.obsm["X_scaledata"] = np.asarray(sd.values, dtype="float32").T
        adata.uns["scaledata_features"] = [str(i) for i in sd.index]

    for r, rinfo in (manifest.get("reductions") or {}).items():
        p = work / f"reduction_{r}.csv"
        if not p.exists():
            continue
        emb = _read_csv_indexed(p).reindex([str(c) for c in cells])
        key = r if r.startswith("X_") else f"X_{r}"
        adata.obsm[key] = np.asarray(emb.values, dtype="float64")
        sd_vals = rinfo.get("stdev") or []
        if r.lower() == "pca" and sd_vals:
            # ElbowPlot / variance-explained need these and they cannot be
            # recovered from the embedding alone.
            var_ratio = np.asarray(sd_vals, dtype=float) ** 2
            total = var_ratio.sum()
            adata.uns["pca"] = {
                "variance": var_ratio,
                "variance_ratio": var_ratio / total if total else var_ratio,
            }
        lp = work / f"reduction_{r}_loadings.csv"
        if lp.exists() and r.lower() == "pca":
            ld = _read_csv_indexed(lp).reindex([str(f) for f in feats])
            adata.varm["PCs"] = np.asarray(ld.fillna(0.0).values, dtype="float64")

    # Seurat's nn/snn graphs are cell x cell, which is obsp. Carrying them means
    # a clustering computed in R can be re-run or interrogated in Python without
    # rebuilding the neighbour graph, and scanpy reads obsp["connectivities"].
    for g in (manifest.get("graphs") or []):
        p = work / f"graph_{g}.mtx"
        if not p.exists():
            continue
        import scipy.io as sio

        adata.obsp[g] = sio.mmread(str(p)).tocsr()
    # scanpy looks for these two names specifically; point them at Seurat's
    # equivalents so sc.tl.leiden works on an imported object without ceremony.
    snn = next((g for g in (manifest.get("graphs") or []) if g.endswith("snn")), "")
    nn = next((g for g in (manifest.get("graphs") or []) if g.endswith("_nn")), "")
    if snn and "connectivities" not in adata.obsp:
        adata.obsp["connectivities"] = adata.obsp[snn]
    if nn and "distances" not in adata.obsp:
        adata.obsp["distances"] = adata.obsp[nn]
    if snn or nn:
        adata.uns["neighbors"] = {"connectivities_key": "connectivities",
                                  "distances_key": "distances",
                                  "params": {"method": "seurat"}}

    adata.uns["seurat_export"] = {
        "seurat_version": manifest.get("seurat_version"),
        "assay": assay,
        "x_layer": x_layer,
        "layers_exported": layers_present,
        "default_assay": manifest.get("default_assay"),
        "available_assays": sorted(assays),
    }
    if info.get("scaledata_skipped"):
        adata.uns["seurat_export"]["scaledata_skipped"] = info["scaledata_skipped"]
    return adata


def write_interchange(adata: Any, work: Path, *, assay: str = "RNA",
                      max_scaledata_cells: int = 50000) -> dict[str, Any]:
    """Write an AnnData in the same layout `seurat_export.R` produces.

    Deliberately one layout in both directions rather than two: the R and
    Python readers then agree by construction, and a file that one side writes
    is one the other already knows how to read. `r/scripts/seurat_from_files.R`
    turns this back into a Seurat object.

    What crosses: the counts and normalised matrices, cell metadata, feature
    metadata, every embedding in obsm with PCA loadings and stdev where they
    exist, and every cell-by-cell graph in obsp.
    """
    import json as _json

    import numpy as np
    import pandas as pd
    import scipy.io as sio
    import scipy.sparse as sp

    work = Path(work)
    work.mkdir(parents=True, exist_ok=True)

    cells = [str(c) for c in adata.obs_names]
    feats = [str(f) for f in adata.var_names]
    (work / "barcodes.txt").write_text("\n".join(cells) + "\n", encoding="utf-8")
    (work / f"assay_{assay}_features.txt").write_text("\n".join(feats) + "\n",
                                                      encoding="utf-8")

    adata.obs.to_csv(work / "metadata.csv")
    if len(adata.var.columns):
        adata.var.to_csv(work / f"assay_{assay}_meta.csv")

    def _as_genes_by_cells(m):
        # Every matrix on disk is genes x cells, which is what Seurat and the
        # mtx convention both expect; AnnData is the transposed one.
        return sp.csr_matrix(m).T.tocsr() if not sp.issparse(m) else m.T.tocsr()

    layers: list[str] = []
    x_state = str((adata.uns.get("skinmcp") or {}).get("x_state", "")) or "unknown"
    if "counts" in adata.layers:
        sio.mmwrite(str(work / f"assay_{assay}_counts.mtx"),
                    _as_genes_by_cells(adata.layers["counts"]))
        layers.append("counts")
    # X is the normalised matrix unless it is itself the counts.
    if adata.X is not None:
        name = "counts" if (x_state == "counts" and "counts" not in layers) else "data"
        if name not in layers:
            sio.mmwrite(str(work / f"assay_{assay}_{name}.mtx"),
                        _as_genes_by_cells(adata.X))
            layers.append(name)

    info: dict[str, Any] = {"n_features": len(feats), "layers": layers}
    if "X_scaledata" in adata.obsm:
        sd = np.asarray(adata.obsm["X_scaledata"])
        if sd.shape[0] <= max_scaledata_cells:
            names = [str(x) for x in (adata.uns.get("scaledata_features")
                                      or feats[: sd.shape[1]])]
            pd.DataFrame(sd.T, index=names, columns=cells).to_csv(
                work / f"assay_{assay}_scaledata.csv")
            info["layers"] = layers + ["scale.data"]
            info["scaledata_features"] = names

    man: dict[str, Any] = {
        "source": "anndata", "default_assay": assay, "n_cells": len(cells),
        "assays": {assay: info}, "reductions": {}, "graphs": [],
        "x_state": x_state,
    }

    for key, emb in adata.obsm.items():
        if key == "X_scaledata":
            continue
        arr = np.asarray(emb)
        if arr.ndim != 2:
            continue
        r = key[2:] if key.startswith("X_") else key
        pd.DataFrame(arr, index=cells,
                     columns=[f"{r}_{i + 1}" for i in range(arr.shape[1])]).to_csv(
            work / f"reduction_{r}.csv")
        rinfo: dict[str, Any] = {"n_dims": int(arr.shape[1]), "key": f"{r}_",
                                 "has_loadings": False}
        if r.lower() == "pca":
            variance = (adata.uns.get("pca") or {}).get("variance")
            if variance is not None:
                # Seurat wants stdev, AnnData stores variance.
                rinfo["stdev"] = list(np.sqrt(np.asarray(variance, dtype=float)))
            if "PCs" in adata.varm:
                pd.DataFrame(np.asarray(adata.varm["PCs"]), index=feats).to_csv(
                    work / f"reduction_{r}_loadings.csv")
                rinfo["has_loadings"] = True
        man["reductions"][r] = rinfo

    # obsp usually holds aliases: the importer points scanpy's "connectivities"
    # and "distances" at Seurat's snn/nn so sc.tl.leiden works, and writing both
    # names would double the largest non-matrix payload on disk for nothing.
    # Deduplicated by identity, with the alias recorded so it can be restored.
    seen: dict[int, str] = {}
    aliases: dict[str, str] = {}
    for key, g in adata.obsp.items():
        ident = id(g)
        if ident in seen:
            aliases[key] = seen[ident]
            continue
        sio.mmwrite(str(work / f"graph_{key}.mtx"), sp.csr_matrix(g))
        man["graphs"].append(key)
        seen[ident] = key
    if aliases:
        man["graph_aliases"] = aliases

    (work / "manifest.json").write_text(
        _json.dumps(man, indent=2, default=str), encoding="utf-8")
    return man
