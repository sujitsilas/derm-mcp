"""Recovering an object that was shared without its raw counts.

Processed objects are routinely handed over log-normalised with `layers['counts']`
stripped, which blocks pseudobulk DE, subclustering and re-normalisation. The old
error told the caller to re-load the file — which cannot conjure counts that are
not in it — so one session spent its whole budget inventing workarounds instead of
asking the user for the raw matrix.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scanpy as sc

from skinmcp import registry
from skinmcp.tools import io_tools, subcluster_tools


@pytest.fixture()
def pair(tmp_path):
    """A counts-free processed .h5ad plus the raw matrix it came from.

    Barcodes differ by the `-1` suffix and the raw matrix carries an extra gene,
    which is what these files look like in practice.
    """
    rng = np.random.default_rng(0)
    n, g = 120, 40
    bc = [f"AAACCT{i:04d}-1" for i in range(n)]
    genes = [f"Gene{i}" for i in range(g)]
    counts = rng.poisson(3.0, (n, g)).astype("float32")

    raw = ad.AnnData(
        X=np.hstack([counts, rng.poisson(1.0, (n, 1))]).astype("float32"),
        obs=pd.DataFrame(index=[b.split("-")[0] for b in bc]),
        var=pd.DataFrame(index=[*genes, "ExtraGene"]))
    raw.write_h5ad(tmp_path / "raw.h5ad")

    proc = ad.AnnData(
        X=counts.copy(),
        obs=pd.DataFrame({"Sample": ["s1"] * 60 + ["s2"] * 60,
                          "cell_types_full": ["Neutrophils"] * n}, index=bc),
        var=pd.DataFrame(index=genes))
    sc.pp.normalize_total(proc, target_sum=1e4)
    sc.pp.log1p(proc)
    proc.write_h5ad(tmp_path / "processed.h5ad")
    return tmp_path


@pytest.fixture()
def countsfree(project, pair):
    return project, io_tools.load_h5ad(
        path=str(pair / "processed.h5ad"), organism="mouse",
        allow_no_counts=True, project_id=project)["dataset_id"]


class TestMissingCountsIsActionable:
    def test_error_names_attach_counts_and_says_to_ask(self, countsfree):
        """The old remedy sent the caller back to load_h5ad, which cannot help."""
        pid, ds = countsfree
        r = subcluster_tools.extract(dataset_id=ds, label_key="cell_types_full",
                                     labels=["Neutrophils"], project_id=pid)
        assert not r["ok"]
        e = r["error"]
        assert e["code"] == "MISSING_COUNTS"
        assert e["suggested_tool"] == "skin.io.attach_counts"
        assert "ASK THE USER" in e["remedy"]


class TestAttachCounts:
    def test_attaches_across_barcode_suffix_drift(self, countsfree, pair):
        import scipy.sparse as sp

        pid, ds = countsfree
        r = io_tools.attach_counts(dataset_id=ds, path=str(pair / "raw.h5ad"),
                                   project_id=pid)
        assert r["ok"], r.get("error")
        assert "120/120" in r["summary"]["matched"]

        a = registry.load(pid, r["dataset_id"])
        c = a.layers["counts"]
        c = c.toarray() if sp.issparse(c) else c
        assert np.allclose(c, np.round(c)), "counts must stay integers"
        assert c.sum() > 0

    def test_unblocks_the_tool_that_failed(self, countsfree, pair):
        pid, ds = countsfree
        fixed = io_tools.attach_counts(dataset_id=ds, path=str(pair / "raw.h5ad"),
                                       project_id=pid)["dataset_id"]
        r = subcluster_tools.extract(dataset_id=fixed, label_key="cell_types_full",
                                     labels=["Neutrophils"], project_id=pid)
        assert r["ok"], r.get("error")
        assert r["summary"]["n_obs"] == 120

    def test_rejects_a_normalised_source(self, countsfree, pair):
        """Grafting normalised values on as "counts" makes DE wrong, not absent."""
        pid, ds = countsfree
        r = io_tools.attach_counts(dataset_id=ds, path=str(pair / "processed.h5ad"),
                                   project_id=pid)
        assert not r["ok"]
        assert "integer counts" in r["error"]["message"]

    def test_rejects_an_unrelated_matrix(self, countsfree, tmp_path):
        rng = np.random.default_rng(1)
        other = ad.AnnData(
            X=rng.poisson(2.0, (30, 40)).astype("float32"),
            obs=pd.DataFrame(index=[f"TTTT{i}" for i in range(30)]),
            var=pd.DataFrame(index=[f"Gene{i}" for i in range(40)]))
        p = tmp_path / "other.h5ad"
        other.write_h5ad(p)
        pid, ds = countsfree
        r = io_tools.attach_counts(dataset_id=ds, path=str(p), project_id=pid)
        assert not r["ok"]
        assert "do not match" in r["error"]["message"]

    def test_missing_file_asks_where_the_counts_are(self, countsfree, tmp_path):
        pid, ds = countsfree
        r = io_tools.attach_counts(dataset_id=ds, path=str(tmp_path / "nope.h5ad"),
                                   project_id=pid)
        assert not r["ok"]
        assert "raw counts" in r["error"]["remedy"]

    def test_dry_run_reports_overlap_without_minting(self, countsfree, pair):
        pid, ds = countsfree
        r = io_tools.attach_counts(dataset_id=ds, path=str(pair / "raw.h5ad"),
                                   project_id=pid, dry_run=True)
        assert r["ok"]
        assert "120/120" in r["summary"]["matched"]
        assert not r["summary"].get("dataset_id")
