"""Registry internals that only real-world files exercise.

Both cases here come from actual failures on a user's 675 MB macrophage object,
which the golden fixture is too tidy to reproduce.
"""

from __future__ import annotations

import threading

import anndata as ad
import numpy as np
import pandas as pd
import pytest

from skinmcp import registry


def _adata(n=20, g=5):
    x = np.random.default_rng(0).poisson(1.0, (n, g)).astype("float32")
    a = ad.AnnData(
        X=x,
        obs=pd.DataFrame({"sample": ["s1"] * (n // 2) + ["s2"] * (n - n // 2)},
                         index=[f"c{i}" for i in range(n)]),
        var=pd.DataFrame(index=[f"g{i}" for i in range(g)]),
    )
    a.layers["counts"] = x.copy()  # mint refuses an object that has lost counts
    return a


class TestSlashSanitisation:
    """h5py forbids '/' in keys, but cell-type labels are full of it.

    Real subtype names like 'MPhi-Res/Rep' end up as uns dict keys, DataFrame
    columns and structured-array field names, all three of which become h5py
    keys on write. Before this was handled, loading such a file failed with
    "Forward slashes are not allowed in keys".
    """

    def test_round_trip_with_slashes_everywhere(self, project):
        a = _adata()
        labels = ["MPhi-Res/Rep", "MPhi-IFN/AS DCs"]
        a.uns["rgg_sub"] = {
            # dict key with a slash
            "pts": pd.DataFrame(np.zeros((2, 2)), columns=labels, index=labels),
            # structured array whose FIELD NAMES are the subtype labels
            "names": np.zeros(3, dtype=[(lbl, "f4") for lbl in labels]),
        }
        a.uns["scores/by group"] = {"a/b": 1.0}

        dsid = registry.mint(project, a, parent_id=None, op="test", params={})
        back = registry.load(project, dsid)

        # Round-trips through disk without raising, and the labels survive
        # somewhere recoverable rather than being silently dropped.
        assert back.n_obs == a.n_obs
        renames = registry.skinmcp_uns(back).get("h5ad_key_renames")
        assert renames, "a rename map must be recorded so labels can be restored"
        assert any("/" in k for k in renames)

    def test_mint_reports_the_renames_it_made(self, project):
        a = _adata()
        a.uns["scores/by group"] = {"x": 1}
        registry.mint(project, a, parent_id=None, op="test", params={})
        notes = registry.take_mint_notes()
        assert any("/" in n or "rename" in n.lower() for n in notes), notes


class TestMintNotes:
    """Notes are per-thread and consumed once.

    The MCP SDK runs every sync tool in a worker thread, so a module-level
    attribute let one tool call read another's notes, and a tool that never
    minted re-reported whatever the previous one left behind.
    """

    def test_notes_are_cleared_after_being_taken(self, project):
        a = _adata()
        a.uns["needs/rename"] = 1
        registry.mint(project, a, parent_id=None, op="test", params={})
        assert registry.take_mint_notes(), "first take should see the notes"
        assert registry.take_mint_notes() == [], "second take must be empty"

    def test_clean_mint_reports_no_notes(self, project):
        registry.mint(project, _adata(), parent_id=None, op="test", params={})
        assert registry.take_mint_notes() == []

    def test_notes_do_not_leak_across_threads(self, project):
        """Thread B minting must not change what thread A collects."""
        dirty = _adata()
        dirty.uns["a/b"] = 1
        started, b_done = threading.Event(), threading.Event()

        def thread_b():
            started.wait(5)
            registry.mint(project, dirty, parent_id=None, op="test_b", params={})
            b_done.set()

        t = threading.Thread(target=thread_b)
        t.start()
        # A mints something clean, then lets B mint something dirty before
        # collecting. Under the old shared attribute A would pick up B's notes.
        registry.mint(project, _adata(n=22), parent_id=None, op="test_a", params={})
        started.set()
        b_done.wait(10)
        assert registry.take_mint_notes() == [], "A must not see B's notes"
        t.join(5)


class TestPathVsHandle:
    """A file path where a handle belongs is the single most common model error."""

    def test_path_names_the_right_loader(self, project):
        from skinmcp.errors import InvalidHandle

        with pytest.raises(InvalidHandle) as e:
            registry.load(project, "/data/macrophages.h5ad")
        msg = str(e.value) + str(getattr(e.value, "remedy", ""))
        assert "load_h5ad" in msg, msg

    def test_an_existing_file_says_so_loudly(self, project, tmp_path):
        """Otherwise the message reads as "wrong path" and the caller hunts.

        A real session cycled three candidate paths through skin.io.set_label
        seventeen times, starting with the correct one, because nothing said the
        path was fine and the tool was wrong.
        """
        p = tmp_path / "real.h5ad"
        p.write_bytes(b"not really an h5ad, but it is on disk")
        err = registry.bad_handle(project, str(p))
        assert err.details["file_exists"] is True
        blob = f"{err.message} {err.remedy}"
        assert "EXISTS" in blob and "load_h5ad" in blob
        assert "not try other paths" in blob or "path is correct" in blob

    def test_a_missing_file_is_reported_differently(self, project, tmp_path):
        err = registry.bad_handle(project, str(tmp_path / "nope.h5ad"))
        assert err.details["file_exists"] is False
        assert "EXISTS" not in err.message


class TestRepeatedFailureEscalation:
    """A caller looping on one broken call needs a different answer, not the same one."""

    def test_third_identical_failure_escalates(self, project):
        from skinmcp.tools import _base, io_tools

        _base._clear_repeat()
        bad = "/definitely/not/here/x.h5ad"
        seen = [io_tools.set_label(dataset_id=bad, label="x", project_id=project)
                for _ in range(3)]
        assert all(not r["ok"] for r in seen)
        assert "repeated_failures" not in seen[0]["error"]
        assert seen[2]["error"]["repeated_failures"] == 3
        assert seen[2]["error"]["remedy"].startswith("STOP.")

    def test_a_success_resets_the_streak(self, project):
        from skinmcp.tools import _base, io_tools, memory_tools

        _base._clear_repeat()
        bad = "/definitely/not/here/x.h5ad"
        for _ in range(3):
            io_tools.set_label(dataset_id=bad, label="x", project_id=project)
        assert memory_tools.list_projects()["ok"]
        again = io_tools.set_label(dataset_id=bad, label="x", project_id=project)
        assert "repeated_failures" not in again["error"], "success must clear the count"


class TestClusteringGuardrails:
    """Signals that were missing when a subclustering run went wrong.

    20k neutrophils at resolution 0.8 gave 14 clusters; four were hard to name,
    44% of cells ended up in "Other"/"Low Quality", and every tool reported
    success without a word about either.
    """

    def test_many_clusters_suggests_a_lower_resolution(self):
        from skinmcp.tools import cluster_tools

        assert cluster_tools.leiden.__wrapped__.__defaults__[0] == 0.5, \
            "leiden must start low and be raised, not the reverse"

    def test_subclustering_also_starts_low(self):
        import inspect

        from skinmcp.tools import subcluster_tools

        for fn in (subcluster_tools.pipeline, subcluster_tools.recluster):
            assert inspect.signature(fn).parameters["resolution"].default == 0.5, fn

    def test_bulk_discard_labels_are_flagged(self):
        from skinmcp.tools.annotate_tools import _warn_on_discards

        class Ctx:
            def __init__(self):
                self.msgs = []

            def warn(self, m):
                self.msgs.append(m)

        ctx = Ctx()
        _warn_on_discards(ctx, {"Mature": 4641, "Inflammatory": 4077,
                                "Stress/IFN": 2552, "Other": 5215,
                                "Low Quality": 3739}, 20224)
        assert ctx.msgs and "44%" in ctx.msgs[0]

    def test_a_small_discard_is_not_flagged(self):
        from skinmcp.tools.annotate_tools import _warn_on_discards

        class Ctx:
            def __init__(self):
                self.msgs = []

            def warn(self, m):
                self.msgs.append(m)

        ctx = Ctx()
        _warn_on_discards(ctx, {"Mature": 9000, "Inflam": 9000, "Doublet": 500}, 18500)
        assert ctx.msgs == [], "a few percent of debris is normal and must stay quiet"
