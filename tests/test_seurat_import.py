"""The Seurat -> AnnData crossover, which is now the only R->Python transport.

zellkonverter was removed: it reaches Python through basilisk, so a handoff
between two working runtimes depended on provisioning a third (it failed on a
real machine trying to install Python 3.14.0 via pyenv), and it routes Seurat
objects through `as.SingleCellExperiment`, which keeps one assay's counts and
logcounts. A post-SCTransform object lost the SCT assay, scale.data, reduction
loadings and the PCA stdev -- silently.

The fixtures here mimic what `seurat_export.R` writes, so the reader is tested
against the layout rather than against a live R install.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
import scipy.io as sio
import scipy.sparse as sp

from skinmcp.runtimes.seurat_import import read_seurat_export


def _write_export(d, *, n=20, g=8, assays=("SCT",), reductions=("pca", "umap"),
                  stdev=None, scaledata=True, loadings=True):
    """Write a minimal but structurally faithful seurat_export() directory."""
    d.mkdir(parents=True, exist_ok=True)
    cells = [f"c{i}" for i in range(n)]
    feats = [f"g{i}" for i in range(g)]
    (d / "barcodes.txt").write_text("\n".join(cells) + "\n")

    rng = np.random.default_rng(0)
    man = {"seurat_version": "5.5.1", "default_assay": assays[0], "n_cells": n,
           "assays": {}, "reductions": {}, "graphs": []}

    import pandas as pd
    pd.DataFrame({"Sample": ["s1", "s2"] * (n // 2),
                  "seurat_clusters": [str(i % 3) for i in range(n)]},
                 index=cells).to_csv(d / "metadata.csv")

    for a in assays:
        (d / f"assay_{a}_features.txt").write_text("\n".join(feats) + "\n")
        layers = []
        for lyr in ("counts", "data"):
            m = sp.csr_matrix(rng.poisson(2, (g, n)).astype("float32"))  # genes x cells
            sio.mmwrite(str(d / f"assay_{a}_{lyr}.mtx"), m)
            layers.append(lyr)
        info = {"n_features": g, "layers": layers}
        if scaledata:
            k = max(1, g // 2)
            pd.DataFrame(rng.normal(size=(k, n)), index=feats[:k],
                         columns=cells).to_csv(d / f"assay_{a}_scaledata.csv")
            info["layers"] = layers + ["scale.data"]
        man["assays"][a] = info

    for r in reductions:
        k = 5 if r == "pca" else 2
        pd.DataFrame(rng.normal(size=(n, k)), index=cells,
                     columns=[f"{r}_{i}" for i in range(k)]).to_csv(d / f"reduction_{r}.csv")
        rinfo = {"n_dims": k, "key": f"{r}_", "has_loadings": False}
        if r == "pca":
            rinfo["stdev"] = list(stdev if stdev is not None else np.linspace(5, 1, k))
            if loadings:
                pd.DataFrame(rng.normal(size=(g, k)), index=feats,
                             columns=[f"PC_{i}" for i in range(k)]).to_csv(
                                 d / f"reduction_{r}_loadings.csv")
                rinfo["has_loadings"] = True
        man["reductions"][r] = rinfo

    (d / "manifest.json").write_text(json.dumps(man, indent=2))
    return cells, feats


class TestRoundTripShape:
    def test_reads_assay_layers_and_reductions(self, tmp_path):
        cells, feats = _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        assert a.shape == (len(cells), len(feats))
        assert list(a.obs_names) == cells
        assert list(a.var_names) == feats
        # X defaults to the normalised layer, which is what scanpy expects.
        assert a.uns["seurat_export"]["x_layer"] == "data"
        assert "counts" in a.layers
        assert {"X_pca", "X_umap"} <= set(a.obsm)

    def test_x_layer_is_selectable(self, tmp_path):
        _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e", x_layer="counts")
        assert a.uns["seurat_export"]["x_layer"] == "counts"

    def test_named_assay_wins_over_the_default(self, tmp_path):
        _write_export(tmp_path / "e", assays=("SCT", "originalexp"))
        a = read_seurat_export(tmp_path / "e", assay="originalexp")
        assert a.uns["seurat_export"]["assay"] == "originalexp"

    def test_unknown_assay_names_what_is_available(self, tmp_path):
        _write_export(tmp_path / "e", assays=("SCT",))
        with pytest.raises(ValueError, match="available"):
            read_seurat_export(tmp_path / "e", assay="RNA")


class TestPcaStdevSurvives:
    """`stdev` is the reason this is a file export and not a converter call.

    It cannot be recovered from the embeddings, and it is what ElbowPlot and
    any variance-explained calculation need. A user's own script computes
    `sum(varExplained[1:33])` from it, so losing it silently changes an answer
    rather than raising.
    """

    def test_variance_ratio_matches_stdev_squared(self, tmp_path):
        stdev = [4.0, 3.0, 2.0, 1.0, 1.0]
        _write_export(tmp_path / "e", stdev=stdev)
        a = read_seurat_export(tmp_path / "e")
        ev = np.asarray(stdev) ** 2
        assert np.allclose(a.uns["pca"]["variance"], ev)
        assert np.allclose(a.uns["pca"]["variance_ratio"], ev / ev.sum())

    def test_pcs_needed_for_80pct_is_reproducible(self, tmp_path):
        _write_export(tmp_path / "e", stdev=[5.0, 4.0, 1.0, 0.5, 0.5])
        a = read_seurat_export(tmp_path / "e")
        vr = a.uns["pca"]["variance_ratio"]
        assert int(np.argmax(np.cumsum(vr) > 0.80)) + 1 == 2

    def test_loadings_land_in_varm(self, tmp_path):
        _, feats = _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        assert a.varm["PCs"].shape == (len(feats), 5)


class TestOrderingIsNotAssumed:
    """R and Python disagree about ordering often enough that a silent
    misalignment is the realistic failure, and it produces plausible nonsense
    rather than an error."""

    def test_metadata_is_reindexed_onto_barcode_order(self, tmp_path):
        import pandas as pd

        d = tmp_path / "e"
        cells, _ = _write_export(d)
        md = pd.read_csv(d / "metadata.csv", index_col=0)
        md.iloc[::-1].to_csv(d / "metadata.csv")        # write it reversed
        a = read_seurat_export(d)
        assert list(a.obs_names) == cells
        assert list(a.obs["Sample"]) == list(md["Sample"])   # realigned, not reversed

    def test_embeddings_are_reindexed_too(self, tmp_path):
        import pandas as pd

        d = tmp_path / "e"
        cells, _ = _write_export(d)
        emb = pd.read_csv(d / "reduction_umap.csv", index_col=0)
        emb.iloc[::-1].to_csv(d / "reduction_umap.csv")
        a = read_seurat_export(d)
        assert np.allclose(a.obsm["X_umap"], emb.loc[cells].values)


class TestScaleDataPlacement:
    def test_scale_data_goes_to_obsm_not_layers(self, tmp_path):
        # It covers only the variable features, so its shape does not match var
        # and it cannot legally be a layer.
        _, feats = _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        k = len(a.uns["scaledata_features"])
        assert a.obsm["X_scaledata"].shape == (a.n_obs, k)
        assert k < len(feats)
        assert "scale.data" not in a.layers

    def test_absent_scale_data_is_not_an_error(self, tmp_path):
        _write_export(tmp_path / "e", scaledata=False)
        a = read_seurat_export(tmp_path / "e")
        assert "X_scaledata" not in a.obsm


class TestGraphsCross:
    """Seurat's nn/snn graphs are cell x cell, so they belong in obsp.

    Carrying them means a clustering computed in R can be re-run in Python
    without rebuilding the neighbour graph — verified on a real object, where
    re-clustering the imported graph reproduced the original 11 clusters.
    """

    def _with_graphs(self, d, n=20):
        cells, _ = _write_export(d)
        man = json.loads((d / "manifest.json").read_text())
        rng = np.random.default_rng(1)
        for g in ("SCT_nn", "SCT_snn"):
            m = sp.csr_matrix(rng.random((n, n)) < 0.2, dtype="float64")
            sio.mmwrite(str(d / f"graph_{g}.mtx"), m)
            man["graphs"].append(g)
        (d / "manifest.json").write_text(json.dumps(man))
        return cells

    def test_graphs_land_in_obsp(self, tmp_path):
        self._with_graphs(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        assert {"SCT_nn", "SCT_snn"} <= set(a.obsp)

    def test_scanpy_aliases_are_wired(self, tmp_path):
        # sc.tl.leiden looks for these names specifically.
        self._with_graphs(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        assert "connectivities" in a.obsp and "distances" in a.obsp
        assert a.uns["neighbors"]["connectivities_key"] == "connectivities"


class TestSymmetricWriter:
    """One layout in both directions, so each side reads what the other wrote."""

    def _round(self, tmp_path):
        _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        from skinmcp.runtimes.seurat_import import write_interchange

        man = write_interchange(a, tmp_path / "out", assay="SCT")
        return a, man

    def test_writes_the_layout_the_reader_expects(self, tmp_path):
        a, _ = self._round(tmp_path)
        back = read_seurat_export(tmp_path / "out")
        assert back.shape == a.shape
        assert list(back.obs_names) == list(a.obs_names)
        assert list(back.var_names) == list(a.var_names)
        assert "counts" in back.layers

    def test_reductions_and_loadings_survive(self, tmp_path):
        a, man = self._round(tmp_path)
        back = read_seurat_export(tmp_path / "out")
        assert {"X_pca", "X_umap"} <= set(back.obsm)
        assert back.varm["PCs"].shape == a.varm["PCs"].shape
        # stdev is stored as variance in AnnData and stdev in Seurat; the
        # conversion has to survive a full lap or ElbowPlot silently changes.
        assert np.allclose(back.uns["pca"]["variance"], a.uns["pca"]["variance"])

    def test_aliased_graphs_are_written_once(self, tmp_path):
        _write_export(tmp_path / "e")
        a = read_seurat_export(tmp_path / "e")
        m = sp.csr_matrix(np.eye(a.n_obs))
        a.obsp["SCT_snn"] = m
        a.obsp["connectivities"] = m          # same object, two names
        from skinmcp.runtimes.seurat_import import write_interchange

        man = write_interchange(a, tmp_path / "out", assay="SCT")
        assert man["graphs"] == ["SCT_snn"]
        assert man["graph_aliases"] == {"connectivities": "SCT_snn"}
        assert not (tmp_path / "out" / "graph_connectivities.mtx").exists()

    def test_matrices_are_written_genes_by_cells(self, tmp_path):
        # The on-disk convention is Seurat's/mtx's, the transpose of AnnData's.
        a, _ = self._round(tmp_path)
        m = sio.mmread(str(tmp_path / "out" / "assay_SCT_counts.mtx"))
        assert m.shape == (a.n_vars, a.n_obs)


class TestNoZellkonverterDependency:
    def test_nothing_in_the_r_scripts_imports_it(self):
        from pathlib import Path

        d = Path("src/skinmcp/runtimes/r/scripts")
        offenders = [p.name for p in d.glob("*.R")
                     if "library(zellkonverter)" in p.read_text()]
        assert offenders == [], f"{offenders} still load zellkonverter"

    def test_every_script_can_locate_its_helper(self):
        # `sys.frame(1)$ofile` is empty when Rscript runs a file directly, which
        # is how the bridge invokes these -- it aborted all 8 on line 2.
        from pathlib import Path

        d = Path("src/skinmcp/runtimes/r/scripts")
        bad = [p.name for p in d.glob("*.R")
               if "sys.frame(1)$ofile %||%" in p.read_text()]
        assert bad == [], f"{bad} use an idiom that fails under Rscript"
