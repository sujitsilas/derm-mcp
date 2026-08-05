"""Guards against the two ways this server dies instead of returning an error.

Both are regressions from real crashes on a 78k x 20k object, hosted by
LM Studio beside a resident local model. Neither produced a traceback: the
process took SIGSEGV and the client saw only "MCP error -32000: Connection
closed", losing the whole session.
"""

from __future__ import annotations

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from skinmcp import config, registry
from skinmcp.errors import InsufficientMemory


class TestNoArrowStrings:
    """pyarrow must stay out of the string/categorical path.

    pandas 3 resolves `mode.string_storage="auto"` to pyarrow whenever pyarrow
    is installed, and pyarrow's allocator answers an allocation failure with
    SIGSEGV rather than MemoryError. There are 170+ `.astype(str)` call sites
    in this package; every one of them was a way to kill the process. The fix
    is a single global setting in skinmcp/__init__.py, which is exactly the
    kind of thing a dependency bump undoes silently -- hence this test.
    """

    def test_string_storage_is_python(self):
        assert pd.get_option("mode.string_storage") == "python"

    def test_categorical_astype_str_is_not_arrow_backed(self):
        # The shape that crashed: a categorical carrying missing values (-1
        # codes), converted to string, which routed through pyarrow.compute.take.
        s = pd.Series(pd.Categorical.from_codes(
            np.array([0, 1, -1, 1], dtype=np.int8), categories=["Neutrophils", "Macrophages"]))
        arr = s.astype(str).array
        assert "Arrow" not in type(arr).__name__, (
            f"{type(arr).__name__} puts pyarrow back in the astype(str) path; "
            "an allocation failure there segfaults the server")

    def test_h5ad_categoricals_read_back_without_arrow(self, tmp_path: Path):
        n = 40
        a = ad.AnnData(
            X=np.ones((n, 3), dtype="float32"),
            obs=pd.DataFrame({"cell_type": pd.Categorical(["Neutrophils"] * n)},
                             index=[f"c{i}" for i in range(n)]),
            var=pd.DataFrame(index=list("abc")),
        )
        p = tmp_path / "t.h5ad"
        a.write_h5ad(p)
        back = ad.read_h5ad(p)
        assert "Arrow" not in type(back.obs["cell_type"].cat.categories.array).__name__


class TestLoadAdmission:
    """A load that cannot fit must be refused, not attempted.

    Attempting it is what gets the process killed; refusing keeps the server
    answering and tells the caller how to recover.
    """

    def _write(self, tmp_path: Path, n=200, g=50) -> Path:
        x = np.ones((n, g), dtype="float32")
        a = ad.AnnData(X=x, obs=pd.DataFrame(index=[f"c{i}" for i in range(n)]),
                       var=pd.DataFrame(index=[f"g{i}" for i in range(g)]))
        a.layers["counts"] = x.copy()
        p = tmp_path / "obj.h5ad"
        a.write_h5ad(p, compression=None)
        return p

    def test_estimate_matches_actual_matrix_bytes(self, tmp_path: Path):
        p = self._write(tmp_path, n=200, g=50)
        # X + counts, both dense float32
        expected = 2 * 200 * 50 * 4
        assert registry.estimate_resident_bytes(p) == expected

    def test_estimate_returns_zero_for_unreadable_file(self, tmp_path: Path):
        bad = tmp_path / "not.h5ad"
        bad.write_text("not hdf5")
        assert registry.estimate_resident_bytes(bad) == 0

    def test_refuses_when_it_cannot_fit(self, tmp_path: Path, monkeypatch):
        p = self._write(tmp_path)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 1e-6)
        with pytest.raises(InsufficientMemory) as e:
            registry.admit_external(p)
        # The remedy has to steer away from a retry: repeating the call is what
        # turns a refusal back into a crash.
        assert "do not retry" in e.value.remedy.lower()
        assert e.value.details["needed_gb"] >= 0

    def test_allows_when_it_fits(self, tmp_path: Path, monkeypatch):
        p = self._write(tmp_path)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 64.0)
        registry.admit_external(p)  # must not raise

    def test_never_blocks_when_ram_is_unmeasurable(self, tmp_path: Path, monkeypatch):
        # Platforms that will not report free RAM must not become unusable.
        p = self._write(tmp_path)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 0.0)
        registry.admit_external(p)  # must not raise


class TestCacheBudgetTracksLiveMemory:
    """The cache budget is fixed at start-up; the machine's free RAM is not.

    One real session started with 41 GB free and sized the cache at 16 GB. Six
    minutes later the local model was resident, 10 GB was free, and the cache
    was still working to the old budget.
    """

    def test_budget_shrinks_when_ram_is_taken(self, monkeypatch):
        monkeypatch.setattr(config.CONFIG, "cache_max_gb", 16.0)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 4.0)
        assert config.CONFIG.effective_cache_gb() == pytest.approx(2.0)

    def test_configured_cap_is_the_ceiling(self, monkeypatch):
        monkeypatch.setattr(config.CONFIG, "cache_max_gb", 2.0)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 200.0)
        assert config.CONFIG.effective_cache_gb() == pytest.approx(2.0)

    def test_falls_back_to_the_flag_when_ram_is_unmeasurable(self, monkeypatch):
        monkeypatch.setattr(config.CONFIG, "cache_max_gb", 7.0)
        monkeypatch.setattr(config, "available_ram_gb", lambda: 0.0)
        assert config.CONFIG.effective_cache_gb() == pytest.approx(7.0)
