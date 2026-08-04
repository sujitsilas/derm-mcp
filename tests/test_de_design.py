"""Design construction for pseudobulk DE.

Both bugs here made `skin.de.pseudobulk` fail on a perfectly ordinary design —
a condition column with four timepoints — while reporting a cause that was not
the real one. The golden fixture has two condition levels and no covariate
column, so neither was reachable from the existing tests.
"""

from __future__ import annotations

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from skinmcp.tools import de_tools, io_tools


@pytest.fixture()
def multilevel(project, tmp_path):
    """4 timepoints x 2 conditions x 3 replicates, one cell type.

    Enough replicates per arm that any INSUFFICIENT_REPLICATES result is a bug
    in the design code rather than a property of the data.
    """
    rng = np.random.default_rng(0)
    rows = []
    for tp in ("D7", "D10", "D14", "D19"):
        for cond in ("Burn", "Sham"):
            for rep in range(3):
                rows += [(f"S_{tp}_{cond}_{rep}", tp, cond)] * 40
    obs = pd.DataFrame(rows, columns=["Sample", "Timepoint", "Type"])
    obs["cell_type"] = "Mac"
    obs.index = [f"c{i}" for i in range(len(obs))]
    x = rng.poisson(5.0, (len(obs), 300)).astype("float32")
    a = ad.AnnData(X=x, obs=obs,
                   var=pd.DataFrame(index=[f"g{i}" for i in range(300)]))
    a.layers["counts"] = x.copy()
    p = tmp_path / "multi.h5ad"
    a.write_h5ad(p)
    return project, io_tools.load_h5ad(path=str(p), organism="mouse",
                                       project_id=project)["dataset_id"]


def _run(pid, ds, **kw):
    kw.setdefault("label_key", "cell_type")
    kw.setdefault("sample_key", "Sample")
    kw.setdefault("condition_key", "Timepoint")
    kw.setdefault("contrast", ["D14", "D7"])
    return de_tools.pseudobulk(dataset_id=ds, project_id=pid, **kw)


class TestMoreThanTwoConditionLevels:
    def test_runs_when_condition_has_four_levels(self, multilevel):
        """Units outside the contrasted arms must be dropped before PyDESeq2.

        Left in, the design's Categorical turns them into NaN, PyDESeq2 drops
        those rows internally, and the run dies on an index that no longer
        matches the counts it was handed.
        """
        pid, ds = multilevel
        r = _run(pid, ds)
        assert r["ok"], r.get("error")
        (lab,) = r["summary"]["per_label"]
        assert lab["n_samples_D14"] == 6 and lab["n_samples_D7"] == 6
        assert lab["n_genes_tested"] > 0

    def test_two_level_contrast_still_works(self, multilevel):
        """The narrow fix must not disturb the ordinary two-level case."""
        pid, ds = multilevel
        r = _run(pid, ds, condition_key="Type", contrast=["Burn", "Sham"])
        assert r["ok"], r.get("error")
        assert r["summary"]["per_label"][0]["n_samples_Burn"] == 12


class TestBlockingFactors:
    def test_condition_is_never_its_own_covariate(self, multilevel):
        """Every level of the condition holds exactly one arm.

        Used as a blocking factor it fails the "does this level contain both
        arms?" balance check for every level, dropping all units and reporting
        INSUFFICIENT_REPLICATES for a design that plainly had enough.
        """
        pid, ds = multilevel
        r = _run(pid, ds, covariates=["Timepoint"])
        assert r["ok"], r.get("error")
        assert r["summary"]["design"] == "~ Timepoint"
        assert any("contrast variable" in w for w in r["warnings"]), r["warnings"]

    def test_no_covariates_by_default(self, multilevel):
        """Defaulting to a *named* column silently changes the model.

        This defaulted to ["Timepoint"], which blocked on a column the caller
        never chose and broke outright when it was the contrast variable.
        """
        pid, ds = multilevel
        r = _run(pid, ds)
        assert r["ok"], r.get("error")
        # The design string is the whole story: condition only, nothing blocked.
        assert r["summary"]["design"] == "~ Timepoint", "condition only, no blocking"
        assert "+" not in r["summary"]["design"]

    def test_real_covariate_is_honoured(self, multilevel):
        """A genuine blocking factor still enters the design."""
        pid, ds = multilevel
        r = _run(pid, ds, covariates=["Type"])
        assert r["ok"], r.get("error")
        assert r["summary"]["design"] == "~ Type + Timepoint"
