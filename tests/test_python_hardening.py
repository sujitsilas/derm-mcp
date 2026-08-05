"""Guards for the two Python-side failures that survive into a real analysis.

Both were found by running the actual 78k x 20k object through the tool chain
rather than a fixture: one kills the process, the other returns a wrong answer.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest
import scipy.sparse as sp

from skinmcp import config, registry
from skinmcp.errors import InsufficientMemory


def _adata(n=200, g=100, labels=("A", "B")):
    x = sp.random(n, g, density=0.05, format="csr", dtype="float32")
    obs = pd.DataFrame(
        {"cell_type": pd.Categorical([labels[i % len(labels)] for i in range(n)])},
        index=[f"c{i}" for i in range(n)])
    a = ad.AnnData(X=x, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
    a.layers["counts"] = x.copy()
    return a


class TestDensificationGuard:
    """`sc.pp.scale` zero-centers, which cannot be done in place on a sparse
    matrix — scanpy densifies. On the real object that is 6.4 GB on top of the
    sparse original, its lognorm copy and adata.raw. The result is an OS kill,
    and faulthandler cannot catch SIGKILL: no traceback, no crash log, the
    server just stops mid-call. It has to be refused beforehand.
    """

    def test_refuses_when_the_dense_form_will_not_fit(self, monkeypatch):
        monkeypatch.setattr(config, "available_ram_gb", lambda: 1e-6)
        with pytest.raises(InsufficientMemory) as e:
            registry.guard_dense(_adata(), "sc.pp.scale")
        assert "do not retry" in e.value.remedy.lower()
        assert e.value.details["n_obs"] == 200

    def test_allows_and_reports_size_when_it_fits(self, monkeypatch):
        monkeypatch.setattr(config, "available_ram_gb", lambda: 64.0)
        gb = registry.guard_dense(_adata(n=1000, g=500), "sc.pp.scale")
        assert gb == pytest.approx(1000 * 500 * 4 / 1e9)

    def test_never_blocks_when_ram_is_unmeasurable(self, monkeypatch):
        monkeypatch.setattr(config, "available_ram_gb", lambda: 0.0)
        registry.guard_dense(_adata(), "sc.pp.scale")   # must not raise

    def test_sizes_from_the_dense_shape_not_the_sparse_nnz(self, monkeypatch):
        # The whole point: a 5%-dense matrix costs 20x its stored size once
        # materialised, so nnz is the wrong number to reason about.
        monkeypatch.setattr(config, "available_ram_gb", lambda: 64.0)
        a = _adata(n=1000, g=1000)
        assert a.X.nnz < 1000 * 1000 * 0.1
        assert registry.guard_dense(a, "op") == pytest.approx(1000 * 1000 * 4 / 1e9)


class TestParentClustersRenamed:
    """A subset inherits the parent's cluster labels, which no longer mean
    anything: they say which whole-tissue cluster each cell fell into, not how
    this compartment divides.

    On the real neutrophil subset, 5 of the parent's 24 `leiden_res0.8` labels
    survived and sat beside a freshly computed `leiden_res0.4` with 9. Nothing
    in the names distinguished them, so "run DE between clusters" could pick the
    stale one and return a confidently wrong result — worse than an error.
    """

    def test_inherited_leiden_is_renamed(self, project):
        a = _adata(n=200)
        a.obs["leiden_res0.8"] = pd.Categorical(["0", "1"] * 100)
        a.obs["louvain"] = pd.Categorical(["x", "y"] * 100)
        src = registry.mint(project, a, parent_id=None, op="test", params={})

        from skinmcp.tools import subcluster_tools

        out = subcluster_tools.extract(dataset_id=src, label_key="cell_type",
                                       labels=["A"], project_id=project)
        assert out["ok"], out.get("error")
        obs = registry.load(project, out["dataset_id"]).obs
        assert "parent_leiden_res0.8" in obs.columns
        assert "parent_louvain" in obs.columns
        assert "leiden_res0.8" not in obs.columns
        assert any("renamed" in w for w in out["warnings"]), \
            "a silent rename is its own trap"

    def test_the_column_defining_the_subset_is_untouched(self, project):
        # Extracting *by* a cluster column must leave that column addressable.
        a = _adata(n=200)
        a.obs["leiden_res0.5"] = pd.Categorical(["0", "1"] * 100)
        src = registry.mint(project, a, parent_id=None, op="test", params={})

        from skinmcp.tools import subcluster_tools

        out = subcluster_tools.extract(dataset_id=src, label_key="leiden_res0.5",
                                       labels=["0"], project_id=project)
        assert out["ok"], out.get("error")
        obs = registry.load(project, out["dataset_id"]).obs
        assert "leiden_res0.5" in obs.columns
        assert "parent_leiden_res0.5" not in obs.columns

    def test_ordinary_metadata_is_left_alone(self, project):
        a = _adata(n=200)
        a.obs["Sample"] = pd.Categorical(["s1", "s2"] * 100)
        a.obs["Timepoint"] = pd.Categorical(["d1", "d7"] * 100)
        src = registry.mint(project, a, parent_id=None, op="test", params={})

        from skinmcp.tools import subcluster_tools

        out = subcluster_tools.extract(dataset_id=src, label_key="cell_type",
                                       labels=["A"], project_id=project)
        obs = registry.load(project, out["dataset_id"]).obs
        assert {"Sample", "Timepoint", "cell_type"} <= set(obs.columns)
        assert not any(c.startswith("parent_") for c in obs.columns)


class TestNoArrowStringsAfterSubset:
    """The pyarrow segfault and the subset path meet here: extract calls
    astype(str) on a categorical, which was one of the two crash sites."""

    def test_subset_obs_is_not_arrow_backed(self, project):
        a = _adata(n=200)
        src = registry.mint(project, a, parent_id=None, op="test", params={})

        from skinmcp.tools import subcluster_tools

        out = subcluster_tools.extract(dataset_id=src, label_key="cell_type",
                                       labels=["A"], project_id=project)
        obs = registry.load(project, out["dataset_id"]).obs
        arr = obs["cell_type"].cat.categories.array
        assert "Arrow" not in type(arr).__name__
        assert np.all(obs["cell_type"].astype(str).to_numpy() == "A")
