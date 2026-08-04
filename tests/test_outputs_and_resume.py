"""Where results are written, and what a resumed session can see.

Both behaviours come from one real failure: a session was compacted mid-analysis,
the next agent could not find the figures (they were under ~/.skinmcp, which the
host app could not read), concluded they were missing, and recomputed the split
UMAP and a D10 pseudobulk DE that both already existed.
"""

from __future__ import annotations

from pathlib import Path

from skinmcp.memory import recall, store
from skinmcp.tools import io_tools, memory_tools


class TestOutputLocation:
    def test_results_land_beside_the_data(self, project, golden_path, tmp_path):
        """Not inside a dotfile the user never chose and the host cannot list."""
        data = tmp_path / "study" / "counts.h5ad"
        data.parent.mkdir()
        data.write_bytes(golden_path.read_bytes())

        io_tools.load_h5ad(path=str(data), organism="mouse", project_id=project)
        assert store.get_output_dir(project) == str(data.parent.resolve())

    def test_explicit_output_dir_wins(self, project, golden_path, tmp_path):
        """A directory the user named is never overridden by a later load."""
        chosen = tmp_path / "chosen"
        chosen.mkdir()
        memory_tools.open_project(name="test_project", organism="mouse",
                                  output_dir=str(chosen), project_id=project)
        data = tmp_path / "elsewhere" / "counts.h5ad"
        data.parent.mkdir()
        data.write_bytes(golden_path.read_bytes())

        io_tools.load_h5ad(path=str(data), organism="mouse", project_id=project)
        assert store.get_output_dir(project) == str(chosen.resolve())

    def test_unwritable_output_dir_falls_back(self, project, tmp_path):
        """A bad location must not make every plotting tool fail."""
        from skinmcp.tools._base import Ctx

        store.set_output_dir(project, "/proc/nope/cannot/write/here")
        ctx = Ctx(tool="t", project_id=project, seed=0, dry_run=False)
        assert ctx.figdir("umap").is_dir()


class TestResumeBriefing:
    def test_briefing_lists_existing_artifacts_with_absolute_paths(
            self, project, golden_path, tmp_path):
        """A resumed session must be able to open what is already there."""
        data = tmp_path / "counts.h5ad"
        data.write_bytes(golden_path.read_bytes())
        ds = io_tools.load_h5ad(path=str(data), organism="mouse",
                                project_id=project)["dataset_id"]
        from skinmcp.tools import qc_tools

        r = qc_tools.sample_stats(dataset_id=ds, project_id=project)
        assert r["ok"] and r["artifacts"], r.get("error")

        b = recall.brief(project)
        assert b["output_dir"] == str(tmp_path.resolve())
        assert b["artifacts"], "a rendered figure must appear in the briefing"
        # Paths are relative to output_dir (absolute ones cost ~80 bytes each and
        # blew the 4 KB briefing budget). What matters is that joining them onto
        # output_dir reaches the real file.
        for line in b["artifacts"]:
            rel = line.split(" ", 1)[1]
            assert (Path(b["output_dir"]) / rel).exists(), rel
        assert "ALREADY EXIST" in b["resume_note"]
        assert b["n_artifacts"] >= len(b["artifacts"])

    def test_repeated_failures_collapse(self, project):
        """One stuck loop used to fill the briefing and hide real progress."""
        steps = [{"step_id": i, "tool": "skin.io.set_label", "ok": 0,
                  "error": "NOT_FOUND: unknown handle 'x.h5ad'"} for i in range(1, 9)]
        steps.append({"step_id": 9, "tool": "skin.de.pseudobulk", "ok": 0,
                      "error": "INSUFFICIENT_REPLICATES: not enough samples"})
        flags = recall.open_flags(project, steps)
        assert len(flags) == 2, flags
        assert "x8" in flags[0] and "last step 8" in flags[0]

    def test_successful_steps_are_not_flagged(self, project):
        ok = [{"step_id": 1, "tool": "skin.io.load_h5ad", "ok": 1, "error": ""}]
        assert recall.open_flags(project, ok) == []
